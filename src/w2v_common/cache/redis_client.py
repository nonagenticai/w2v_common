"""Async Redis client with JSON serialization and connection pooling.

All configuration is read from environment variables or constructor parameters.
Superset of all service-specific Redis clients.
"""

from __future__ import annotations

import builtins
import json
import logging
import os
from collections.abc import Awaitable
from functools import lru_cache
from typing import Any, cast

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import MaxConnectionsError, RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError

logger = logging.getLogger(__name__)


class RedisUnavailableError(RedisError):
    """Redis is UNREACHABLE: not connected, refused, reset, or timed out.

    Raised only by the ``*_strict`` accessors (:meth:`RedisClient.get_strict`),
    which exist so a caller can tell an OUTAGE from a MISS. Subclasses
    ``RedisError`` so every existing ``except RedisError`` already covers it.
    """


def is_redis_outage(exc: BaseException) -> bool:
    """True only when ``exc`` means the Redis SERVER could not be reached.

    Connection-class failures: :class:`RedisUnavailableError`, redis
    ``ConnectionError`` / ``TimeoutError``, and a raw ``OSError``.

    Deliberately NOT an outage, because the server is up and other commands
    keep working — backing off from them would disable a healthy cache:

    * ``MaxConnectionsError`` — the LOCAL pool is exhausted by a burst. It
      subclasses redis ``ConnectionError``, hence the explicit exclusion.
    * ``ResponseError`` and subclasses (``OOM``, ``READONLY``, ``WRONGTYPE``)
      — the server answered; typically only writes are refused.
    """
    if isinstance(exc, MaxConnectionsError):
        return False
    return isinstance(
        exc, (RedisUnavailableError, RedisConnectionError, RedisTimeoutError, OSError)
    )


class RedisClient:
    """Async Redis client for caching operations.

    Attributes:
        is_available: Whether the client is connected.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        max_connections: int | None = None,
        socket_timeout: int | None = None,
        socket_connect_timeout: int | None = None,
        decode_responses: bool | None = None,
        health_check_interval: int | None = None,
    ) -> None:
        # REDIS_URL is a required env var in production (see fleet-wide rule
        # `feedback-no-hardcoded-urls-ports-in-code`). Falling back to
        # `redis://localhost:6379/0` silently masks a missing secret as a
        # runtime connection-refused; raise loud at construction instead.
        redis_url_resolved = redis_url or os.getenv("REDIS_URL")
        if not redis_url_resolved:
            raise RuntimeError(
                "REDIS_URL is required: pass it explicitly to RedisClient(...) "
                "or set the REDIS_URL environment variable."
            )
        self._redis_url = redis_url_resolved
        self._max_connections = max_connections or int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
        self._socket_timeout = socket_timeout or int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
        self._socket_connect_timeout = socket_connect_timeout or int(
            os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5")
        )
        if decode_responses is not None:
            self._decode_responses = decode_responses
        else:
            self._decode_responses = os.getenv("REDIS_DECODE_RESPONSES", "true").lower() == "true"
        if health_check_interval is not None:
            self._health_check_interval = health_check_interval
        else:
            self._health_check_interval = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "0"))
        self._pool: ConnectionPool | None = None
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        """Establish connection pool and verify connectivity."""
        if self._pool is None:
            self._pool = ConnectionPool.from_url(
                self._redis_url,
                max_connections=self._max_connections,
                socket_timeout=self._socket_timeout,
                socket_connect_timeout=self._socket_connect_timeout,
                decode_responses=self._decode_responses,
                health_check_interval=self._health_check_interval,
            )
            self._client = redis.Redis(connection_pool=self._pool)
            try:
                ping_result = self._client.ping()
                if hasattr(ping_result, "__await__"):
                    await ping_result
                logger.info("Redis connection successful")
            except (RedisError, OSError, ConnectionError, TimeoutError) as e:
                logger.error("Redis connection failed: %s", e)
                raise

    async def disconnect(self) -> None:
        """Close the Redis client and connection pool."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._pool:
            await self._pool.aclose()
            self._pool = None

    async def __aenter__(self) -> RedisClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()

    def _ensure_connected(self) -> redis.Redis:
        if self._client is None:
            raise RuntimeError("RedisClient not connected. Call connect() first.")
        return self._client

    @property
    def is_available(self) -> bool:
        """Check if client is connected and available."""
        return self._client is not None

    def _serialize(self, value: Any) -> str:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return json.dumps(value)

    def _deserialize(self, value: str | None) -> Any:
        if value is None:
            return None
        try:
            return json.loads(value)
        except TypeError, ValueError:
            logger.warning("cache: corrupt JSON value at deserialize, treating as miss")
            return None

    # ------------------------------------------------------------------
    # Core key-value operations
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Any | None:
        """Get a deserialized value by key, or None on miss/error."""
        try:
            client = self._ensure_connected()
        except RuntimeError:
            return None
        try:
            value = await client.get(key)
            return self._deserialize(value) if value else None
        except RedisError as e:
            logger.error("Error getting key '%s': %s", key, e)
            return None

    async def get_strict(self, key: str) -> object:
        """:meth:`get`, but a failure RAISES instead of reading as a miss.

        :meth:`get` returns ``None`` both when the key is absent and when Redis
        is down, so a caller cannot react to an outage (back off, skip the
        follow-up calls that would each burn the socket timeout too). This is
        the additive variant for callers that must tell the two apart; the
        contract of :meth:`get` is unchanged.

        Returns:
            The deserialized value, or ``None`` when the key does not exist.

        Raises:
            RedisUnavailableError: the client is not connected, or the command
                failed for a connection-class reason per :func:`is_redis_outage`
                (the original is ``__cause__``).
            RedisError: any other failure (pool exhausted, a server error
                reply) propagates unchanged — not a miss, but not an outage.
        """
        try:
            client = self._ensure_connected()
        except RuntimeError as e:
            raise RedisUnavailableError(str(e)) from e
        try:
            value = await client.get(key)
        except (RedisError, OSError) as e:
            if is_redis_outage(e):
                raise RedisUnavailableError(f"GET failed: {e}") from e
            raise
        return self._deserialize(value) if value else None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a serialized value with optional TTL."""
        try:
            client = self._ensure_connected()
        except RuntimeError:
            return False
        try:
            serialized = self._serialize(value)
            if ttl is not None:
                result = await client.setex(key, ttl, serialized)
            else:
                result = await client.set(key, serialized)
            return bool(result)
        except RedisError as e:
            logger.error("Error setting key '%s': %s", key, e)
            return False

    async def set_nx(self, key: str, value: Any, *, ex: int) -> bool | None:
        """Atomic ``SET key value NX EX ex`` — set only if the key is absent.

        The public API for distributed locks/leases, where at most one caller
        may win a key. Tri-state return so callers can distinguish *contention*
        from a Redis *outage* (collapsing the two would let a transient outage
        read as "someone else holds it" and wedge every caller):

        * ``True``  — the key was set; this caller now owns it.
        * ``False`` — the key already existed (NX precondition failed;
          another caller holds it — contended).
        * ``None``  — Redis is unavailable / not connected; the caller decides
          whether to fall open. Deliberately distinct from ``False``.

        Unlike :meth:`set`, whose ``ttl`` path uses ``SETEX`` (no NX), this
        issues a single atomic ``SET`` with both ``nx`` and ``ex`` so the
        check-and-set cannot race. ``ex`` (seconds) is required — an
        unexpiring lock is a footgun.
        """
        try:
            client = self._ensure_connected()
        except RuntimeError:
            return None
        try:
            serialized = self._serialize(value)
            result = await client.set(key, serialized, nx=True, ex=ex)
            # redis-py returns ``True`` on success, ``None`` when the key
            # already exists (NX precondition failed).
            return bool(result)
        except RedisError as e:
            logger.error("Error on SET NX for key '%s': %s", key, e)
            return None

    async def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        try:
            client = self._ensure_connected()
        except RuntimeError:
            return False
        try:
            result = await client.delete(key)
            return bool(result)
        except RedisError as e:
            logger.error("Error deleting key '%s': %s", key, e)
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis."""
        try:
            client = self._ensure_connected()
        except RuntimeError:
            return False
        try:
            result = await client.exists(key)
            return bool(result)
        except RedisError as e:
            logger.error("Error checking key '%s': %s", key, e)
            return False

    async def ping(self) -> bool:
        """Ping Redis to check connectivity."""
        try:
            client = self._ensure_connected()
        except RuntimeError:
            return False
        try:
            ping_result = client.ping()
            if hasattr(ping_result, "__await__"):
                result = await ping_result
            else:
                result = ping_result
            return bool(result)
        except RedisError as e:
            logger.error("Redis ping failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Counter operations
    # ------------------------------------------------------------------

    async def incr(self, key: str) -> int:
        """Increment a key."""
        client = self._ensure_connected()
        try:
            return await client.incr(key)
        except (RedisError, OSError, ConnectionError, TimeoutError) as e:
            logger.error("Error incr '%s': %s", key, e)
            return 0

    async def expire(self, key: str, seconds: int) -> bool:
        """Set key expiry."""
        client = self._ensure_connected()
        try:
            return bool(await client.expire(key, seconds))
        except (RedisError, OSError, ConnectionError, TimeoutError) as e:
            logger.error("Error expire '%s': %s", key, e)
            return False

    # ------------------------------------------------------------------
    # Hash operations
    # ------------------------------------------------------------------

    async def hset(self, key: str, field: str, value: str) -> bool:
        """Set a hash field value."""
        client = self._ensure_connected()
        try:
            await cast(Awaitable[int], client.hset(key, field, value))
            return True
        except (RedisError, OSError, ConnectionError, TimeoutError) as e:
            logger.error("Error hset '%s': %s", key, e)
            return False

    async def hget(self, key: str, field: str) -> str | None:
        """Get a single hash field value."""
        client = self._ensure_connected()
        try:
            return await cast(Awaitable[str | None], client.hget(key, field))
        except (RedisError, OSError, ConnectionError, TimeoutError) as e:
            logger.error("Error hget '%s.%s': %s", key, field, e)
            return None

    async def hgetall(self, key: str) -> dict:
        """Get all hash fields."""
        client = self._ensure_connected()
        try:
            return await cast(Awaitable[dict], client.hgetall(key))
        except (RedisError, OSError, ConnectionError, TimeoutError) as e:
            logger.error("Error hgetall '%s': %s", key, e)
            return {}

    async def hdel(self, key: str, *fields: str) -> int:
        """Delete one or more hash fields. Returns number deleted."""
        client = self._ensure_connected()
        try:
            return await cast(Awaitable[int], client.hdel(key, *fields))
        except (RedisError, OSError, ConnectionError, TimeoutError) as e:
            logger.error("Error hdel '%s': %s", key, e)
            return 0

    # ------------------------------------------------------------------
    # List operations
    # ------------------------------------------------------------------

    async def lpush(self, key: str, *values: str) -> int:
        """Push values to the left of a list."""
        client = self._ensure_connected()
        try:
            return await cast(Awaitable[int], client.lpush(key, *values))
        except (RedisError, OSError, ConnectionError, TimeoutError) as e:
            logger.error("Error lpush '%s': %s", key, e)
            return 0

    async def rpop(self, key: str) -> str | None:
        """Pop a value from the right of a list."""
        client = self._ensure_connected()
        try:
            return await cast(Awaitable[str | None], client.rpop(key))
        except (RedisError, OSError, ConnectionError, TimeoutError) as e:
            logger.error("Error rpop '%s': %s", key, e)
            return None

    async def llen(self, key: str) -> int:
        """Get the length of a list."""
        client = self._ensure_connected()
        try:
            return await cast(Awaitable[int], client.llen(key))
        except (RedisError, OSError, ConnectionError, TimeoutError) as e:
            logger.error("Error llen '%s': %s", key, e)
            return 0

    # ------------------------------------------------------------------
    # Set operations
    # ------------------------------------------------------------------

    async def sadd(self, key: str, *members: str) -> int:
        """Add members to a set."""
        client = self._ensure_connected()
        try:
            return await cast(Awaitable[int], client.sadd(key, *members))
        except (RedisError, OSError, ConnectionError, TimeoutError) as e:
            logger.error("Error sadd '%s': %s", key, e)
            return 0

    async def srem(self, key: str, *members: str) -> int:
        """Remove members from a set."""
        client = self._ensure_connected()
        try:
            return await cast(Awaitable[int], client.srem(key, *members))
        except (RedisError, OSError, ConnectionError, TimeoutError) as e:
            logger.error("Error srem '%s': %s", key, e)
            return 0

    async def scard(self, key: str) -> int:
        """Get the number of members in a set."""
        client = self._ensure_connected()
        try:
            return await cast(Awaitable[int], client.scard(key))
        except (RedisError, OSError, ConnectionError, TimeoutError) as e:
            logger.error("Error scard '%s': %s", key, e)
            return 0

    async def smembers(self, key: str) -> builtins.set[str]:
        """Get all members of a set."""
        client = self._ensure_connected()
        try:
            return await cast(Awaitable[set[str]], client.smembers(key))
        except (RedisError, OSError, ConnectionError, TimeoutError) as e:
            logger.error("Error smembers '%s': %s", key, e)
            return set()

    # ------------------------------------------------------------------
    # Scripting
    # ------------------------------------------------------------------

    async def eval_script(
        self, script: str, keys: list[str] | None = None, args: list[Any] | None = None
    ) -> Any:
        """Evaluate a Lua script atomically on the Redis server.

        Args:
            script: Lua script source code.
            keys: List of Redis keys the script accesses.
            args: List of additional arguments passed to the script.

        Returns:
            The return value from the Lua script.

        Raises:
            RedisError: On connection or execution failure.
        """
        client = self._ensure_connected()
        return await cast(
            Awaitable[Any],
            client.eval(script, len(keys or []), *(keys or []), *(args or [])),
        )

    # ------------------------------------------------------------------
    # Pub/Sub
    # ------------------------------------------------------------------

    async def publish(self, channel: str, message: str) -> int:
        """Publish a message to a Redis pub/sub channel.

        Args:
            channel: The channel name to publish to.
            message: The message payload (already-serialized string).

        Returns:
            The number of subscribers that received the message.

        Raises:
            RedisError: On connection or execution failure.
        """
        client = self._ensure_connected()
        return await cast(Awaitable[int], client.publish(channel, message))

    def pubsub(self) -> Any:
        """Return a PubSub helper for subscribing to channels."""
        client = self._ensure_connected()
        return client.pubsub()

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    async def pipeline(self) -> Any:
        """Get a Redis pipeline."""
        client = self._ensure_connected()
        return client.pipeline()


# ------------------------------------------------------------------
# Global singleton management
# ------------------------------------------------------------------

_redis_client: RedisClient | None = None


@lru_cache
def get_redis_client() -> RedisClient:
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client


async def init_redis() -> RedisClient:
    """Initialize the global Redis client and connect."""
    client = get_redis_client()
    await client.connect()
    return client


async def close_redis() -> None:
    """Disconnect and destroy the global Redis client."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.disconnect()
        _redis_client = None
        get_redis_client.cache_clear()
