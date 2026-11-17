"""Tests for w2v_common.cache.redis_client.

Uses fakeredis for real command execution (serialization round-trips, TTL,
data structure operations) instead of AsyncMock call-checking.
Mock-based tests are kept only for error injection paths.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel
from redis.exceptions import RedisError

from w2v_common.cache.redis_client import RedisClient, close_redis, get_redis_client, init_redis

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redis_url_env(monkeypatch):
    """Most tests construct ``RedisClient()`` without a URL kwarg; provide a
    test-default REDIS_URL so the production no-fallback guard doesn't fire
    in unit tests that aren't testing the guard itself."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")


class TestRedisClientInit:
    """Tests for RedisClient initialization."""

    def test_default_init(self, monkeypatch):
        monkeypatch.delenv("REDIS_HEALTH_CHECK_INTERVAL", raising=False)
        client = RedisClient()
        assert client._redis_url == "redis://localhost:6379/0"
        assert client._max_connections == 10
        assert client._socket_timeout == 5
        assert client._socket_connect_timeout == 5
        assert client._decode_responses is True
        assert client._health_check_interval == 0
        assert client._pool is None
        assert client._client is None

    def test_raises_when_redis_url_unset(self, monkeypatch):
        """REDIS_URL is required: no env + no kwarg → fail loud at construction.

        Regression guard for the silent localhost fallback (rule
        `feedback-no-hardcoded-urls-ports-in-code`).
        """
        monkeypatch.delenv("REDIS_URL", raising=False)
        with pytest.raises(RuntimeError, match="REDIS_URL is required"):
            RedisClient()

    def test_redis_url_kwarg_when_env_unset(self, monkeypatch):
        """An explicit ``redis_url=`` kwarg satisfies the requirement even
        when the env var is missing."""
        monkeypatch.delenv("REDIS_URL", raising=False)
        client = RedisClient(redis_url="redis://explicit:6380/2")
        assert client._redis_url == "redis://explicit:6380/2"

    def test_custom_init(self):
        client = RedisClient(
            redis_url="redis://custom:6380/1",
            max_connections=20,
            socket_timeout=10,
            socket_connect_timeout=15,
            decode_responses=False,
            health_check_interval=45,
        )
        assert client._redis_url == "redis://custom:6380/1"
        assert client._max_connections == 20
        assert client._socket_timeout == 10
        assert client._socket_connect_timeout == 15
        assert client._decode_responses is False
        assert client._health_check_interval == 45

    def test_health_check_interval_from_env(self, monkeypatch):
        """REDIS_HEALTH_CHECK_INTERVAL env var sets the field when no kwarg given."""
        monkeypatch.setenv("REDIS_HEALTH_CHECK_INTERVAL", "60")
        client = RedisClient()
        assert client._health_check_interval == 60

    def test_health_check_interval_kwarg_overrides_env(self, monkeypatch):
        """Explicit kwarg wins over the env var."""
        monkeypatch.setenv("REDIS_HEALTH_CHECK_INTERVAL", "60")
        client = RedisClient(health_check_interval=15)
        assert client._health_check_interval == 15


# ---------------------------------------------------------------------------
# Connect/Disconnect lifecycle (needs mocks for real connection pool)
# ---------------------------------------------------------------------------


class TestRedisClientConnect:
    """Tests for connect/disconnect lifecycle."""

    @patch("w2v_common.cache.redis_client.ConnectionPool")
    async def test_connect_success(self, mock_pool_cls):
        # Use a fresh client (not fakeredis-injected) to test real connect logic
        client = RedisClient()
        mock_pool = MagicMock()
        mock_pool_cls.from_url.return_value = mock_pool
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        with patch("w2v_common.cache.redis_client.redis.Redis", return_value=mock_redis):
            await client.connect()
            assert client._pool is not None
            assert client._client is not None

    @patch("w2v_common.cache.redis_client.ConnectionPool")
    async def test_connect_passes_health_check_interval_to_pool(self, mock_pool_cls):
        """Regression: ConnectionPool.from_url must receive health_check_interval
        so a Redis restart doesn't leave the pool full of stale connections that
        fail every ping (seen in prod 2026-04-18)."""
        client = RedisClient(health_check_interval=42)
        mock_pool_cls.from_url.return_value = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        with patch("w2v_common.cache.redis_client.redis.Redis", return_value=mock_redis):
            await client.connect()

        kwargs = mock_pool_cls.from_url.call_args.kwargs
        assert kwargs.get("health_check_interval") == 42, (
            f"expected health_check_interval=42 in pool kwargs, got: {kwargs}"
        )

    @patch("w2v_common.cache.redis_client.ConnectionPool")
    async def test_connect_failure(self, mock_pool_cls):
        client = RedisClient()
        mock_pool = MagicMock()
        mock_pool_cls.from_url.return_value = mock_pool
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("cannot connect"))

        with (
            patch("w2v_common.cache.redis_client.redis.Redis", return_value=mock_redis),
            pytest.raises(ConnectionError),
        ):
            await client.connect()

    async def test_disconnect(self, fakeredis_client):
        await fakeredis_client.disconnect()
        assert fakeredis_client._client is None
        assert fakeredis_client._pool is None

    async def test_disconnect_when_not_connected(self):
        client = RedisClient()
        await client.disconnect()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestRedisClientProperties:
    """Tests for RedisClient properties."""

    def test_is_available_when_connected(self, fakeredis_client):
        assert fakeredis_client.is_available is True

    def test_is_available_when_not_connected(self):
        client = RedisClient()
        assert client.is_available is False


# ---------------------------------------------------------------------------
# Serialization (pure methods, no Redis needed)
# ---------------------------------------------------------------------------


class TestRedisClientSerialization:
    """Tests for serialization/deserialization."""

    def test_serialize_dict(self):
        client = RedisClient()
        result = client._serialize({"key": "value"})
        assert result == '{"key": "value"}'

    def test_serialize_pydantic_model(self):
        class MyModel(BaseModel):
            name: str
            value: int

        client = RedisClient()
        model = MyModel(name="test", value=42)
        result = client._serialize(model)
        assert '"name": "test"' in result
        assert '"value": 42' in result

    def test_deserialize_valid_json(self):
        client = RedisClient()
        result = client._deserialize('{"key": "value"}')
        assert result == {"key": "value"}

    def test_deserialize_none(self):
        client = RedisClient()
        result = client._deserialize(None)
        assert result is None

    def test_deserialize_invalid_json(self):
        client = RedisClient()
        result = client._deserialize("not-json{")
        assert result is None

    def test_deserialize_invalid_json_logs_warning(self, caplog):
        """A corrupt cached value is a miss AND a diagnostic signal — without
        the warning, silent cache-skipping is invisible in prod logs (mirrors
        the idempotency cache's 'was not valid JSON' warning)."""
        client = RedisClient()
        with caplog.at_level(logging.WARNING, logger="w2v_common.cache.redis_client"):
            result = client._deserialize("not-json{")
        assert result is None
        assert any(
            record.levelname == "WARNING" and "corrupt JSON value" in record.getMessage()
            for record in caplog.records
        ), f"expected a corrupt-JSON WARNING, got: {[r.getMessage() for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Core key-value operations (fakeredis — real round-trips)
# ---------------------------------------------------------------------------


class TestRedisClientCoreOperations:
    """Tests for core key-value operations with fakeredis."""

    async def test_set_and_get_dict_round_trip(self, fakeredis_client):
        """Set a dict, get it back — verify JSON serialization round-trip."""
        data = {"user": "alice", "score": 42, "tags": ["admin", "active"]}
        assert await fakeredis_client.set("test-key", data) is True

        result = await fakeredis_client.get("test-key")
        assert result == data
        assert result["user"] == "alice"
        assert result["tags"] == ["admin", "active"]

    async def test_set_and_get_string_round_trip(self, fakeredis_client):
        """String values are JSON-serialized (quoted)."""
        assert await fakeredis_client.set("str-key", "hello") is True
        result = await fakeredis_client.get("str-key")
        assert result == "hello"

    async def test_set_and_get_list_round_trip(self, fakeredis_client):
        assert await fakeredis_client.set("list-key", [1, 2, 3]) is True
        result = await fakeredis_client.get("list-key")
        assert result == [1, 2, 3]

    async def test_get_returns_none_on_cache_miss(self, fakeredis_client):
        result = await fakeredis_client.get("nonexistent-key")
        assert result is None

    async def test_set_with_ttl(self, fakeredis_client):
        """Set with TTL — verify key exists and has a TTL."""
        assert await fakeredis_client.set("ttl-key", {"temp": True}, ttl=300) is True

        # Key should exist
        assert await fakeredis_client.exists("ttl-key") is True

        # fakeredis supports checking TTL
        raw_ttl = await fakeredis_client._client.ttl("ttl-key")
        assert raw_ttl > 0
        assert raw_ttl <= 300

    async def test_set_without_ttl_has_no_expiry(self, fakeredis_client):
        await fakeredis_client.set("no-ttl", "persist")
        raw_ttl = await fakeredis_client._client.ttl("no-ttl")
        assert raw_ttl == -1  # -1 means no expiry

    async def test_delete_removes_key(self, fakeredis_client):
        await fakeredis_client.set("del-key", "value")
        assert await fakeredis_client.exists("del-key") is True

        assert await fakeredis_client.delete("del-key") is True
        assert await fakeredis_client.exists("del-key") is False

    async def test_delete_returns_false_for_missing_key(self, fakeredis_client):
        assert await fakeredis_client.delete("missing") is False

    async def test_exists_true_for_existing_key(self, fakeredis_client):
        await fakeredis_client.set("exist-key", "yes")
        assert await fakeredis_client.exists("exist-key") is True

    async def test_exists_false_for_missing_key(self, fakeredis_client):
        assert await fakeredis_client.exists("nope") is False

    async def test_ping(self, fakeredis_client):
        assert await fakeredis_client.ping() is True

    async def test_overwrite_key(self, fakeredis_client):
        """Setting the same key twice overwrites the value."""
        await fakeredis_client.set("ow-key", {"version": 1})
        await fakeredis_client.set("ow-key", {"version": 2})
        result = await fakeredis_client.get("ow-key")
        assert result == {"version": 2}

    # --- set_nx (atomic SET NX EX) ---

    async def test_set_nx_first_caller_wins(self, fakeredis_client):
        """First set_nx on an absent key sets it and returns True."""
        assert await fakeredis_client.set_nx("lock", "owner-a", ex=30) is True
        # Key exists with a TTL from the EX arg.
        raw_ttl = await fakeredis_client._client.ttl("lock")
        assert 0 < raw_ttl <= 30

    async def test_set_nx_second_caller_loses(self, fakeredis_client):
        """A contended key returns False and does not overwrite the owner."""
        assert await fakeredis_client.set_nx("lock", "owner-a", ex=30) is True
        assert await fakeredis_client.set_nx("lock", "owner-b", ex=30) is False
        # Owner unchanged — NX did not clobber the incumbent.
        assert await fakeredis_client.get("lock") == "owner-a"

    async def test_set_nx_returns_none_when_not_connected(self):
        """Not-connected is None (fall-open signal), distinct from False."""
        client = RedisClient()
        assert await client.set_nx("lock", "x", ex=30) is None

    async def test_set_nx_returns_none_on_redis_error(self, fakeredis_client):
        """A RedisError is None (outage), never False — so an outage can't
        read as contention and wedge the caller."""
        fakeredis_client._client.set = AsyncMock(side_effect=RedisError("timeout"))
        assert await fakeredis_client.set_nx("lock", "x", ex=30) is None

    # --- Error paths (need mock injection) ---

    async def test_get_returns_none_when_not_connected(self):
        client = RedisClient()
        assert await client.get("key") is None

    async def test_set_returns_false_when_not_connected(self):
        client = RedisClient()
        assert await client.set("key", "value") is False

    async def test_delete_returns_false_when_not_connected(self):
        client = RedisClient()
        assert await client.delete("key") is False

    async def test_exists_returns_false_when_not_connected(self):
        client = RedisClient()
        assert await client.exists("key") is False

    async def test_ping_returns_false_when_not_connected(self):
        client = RedisClient()
        assert await client.ping() is False

    async def test_get_swallows_redis_error(self, fakeredis_client):
        fakeredis_client._client.get = AsyncMock(side_effect=RedisError("timeout"))
        assert await fakeredis_client.get("key") is None

    async def test_set_swallows_redis_error(self, fakeredis_client):
        fakeredis_client._client.set = AsyncMock(side_effect=RedisError("disk full"))
        assert await fakeredis_client.set("key", "value") is False


# ---------------------------------------------------------------------------
# Counter operations (fakeredis)
# ---------------------------------------------------------------------------


class TestRedisClientCounterOperations:
    """Tests for counter operations with fakeredis."""

    async def test_incr_from_zero(self, fakeredis_client):
        """INCR on a non-existent key creates it with value 1."""
        result = await fakeredis_client.incr("counter")
        assert result == 1

    async def test_incr_increments(self, fakeredis_client):
        await fakeredis_client.incr("counter")
        await fakeredis_client.incr("counter")
        result = await fakeredis_client.incr("counter")
        assert result == 3

    async def test_expire_sets_ttl(self, fakeredis_client):
        await fakeredis_client.set("expire-key", "data")
        assert await fakeredis_client.expire("expire-key", 60) is True

        raw_ttl = await fakeredis_client._client.ttl("expire-key")
        assert 0 < raw_ttl <= 60

    async def test_expire_returns_false_for_missing_key(self, fakeredis_client):
        assert await fakeredis_client.expire("missing", 60) is False

    async def test_incr_returns_zero_on_error(self, fakeredis_client):
        fakeredis_client._client.incr = AsyncMock(side_effect=RedisError("error"))
        assert await fakeredis_client.incr("counter") == 0

    async def test_expire_returns_false_on_error(self, fakeredis_client):
        fakeredis_client._client.expire = AsyncMock(side_effect=RedisError("error"))
        assert await fakeredis_client.expire("key", 60) is False


# ---------------------------------------------------------------------------
# Hash operations (fakeredis)
# ---------------------------------------------------------------------------


class TestRedisClientHashOperations:
    """Tests for hash operations with fakeredis."""

    async def test_hset_and_hget_round_trip(self, fakeredis_client):
        assert await fakeredis_client.hset("myhash", "field1", "value1") is True
        result = await fakeredis_client.hget("myhash", "field1")
        assert result == "value1"

    async def test_hget_missing_field_returns_none(self, fakeredis_client):
        result = await fakeredis_client.hget("myhash", "missing")
        assert result is None

    async def test_hgetall_returns_all_fields(self, fakeredis_client):
        await fakeredis_client.hset("h", "f1", "v1")
        await fakeredis_client.hset("h", "f2", "v2")
        await fakeredis_client.hset("h", "f3", "v3")

        result = await fakeredis_client.hgetall("h")
        assert result == {"f1": "v1", "f2": "v2", "f3": "v3"}

    async def test_hgetall_empty_hash(self, fakeredis_client):
        result = await fakeredis_client.hgetall("empty-hash")
        assert result == {}

    async def test_hdel_removes_fields(self, fakeredis_client):
        await fakeredis_client.hset("h", "f1", "v1")
        await fakeredis_client.hset("h", "f2", "v2")

        deleted = await fakeredis_client.hdel("h", "f1", "f2")
        assert deleted == 2

        result = await fakeredis_client.hgetall("h")
        assert result == {}

    async def test_hdel_missing_fields_returns_zero(self, fakeredis_client):
        assert await fakeredis_client.hdel("h", "nonexistent") == 0

    # --- Error paths ---

    async def test_hset_returns_false_on_error(self, fakeredis_client):
        fakeredis_client._client.hset = AsyncMock(side_effect=RedisError("err"))
        assert await fakeredis_client.hset("h", "f", "v") is False

    async def test_hget_returns_none_on_error(self, fakeredis_client):
        fakeredis_client._client.hget = AsyncMock(side_effect=RedisError("err"))
        assert await fakeredis_client.hget("h", "f") is None

    async def test_hgetall_returns_empty_dict_on_error(self, fakeredis_client):
        fakeredis_client._client.hgetall = AsyncMock(side_effect=RedisError("err"))
        assert await fakeredis_client.hgetall("h") == {}

    async def test_hdel_returns_zero_on_error(self, fakeredis_client):
        fakeredis_client._client.hdel = AsyncMock(side_effect=RedisError("err"))
        assert await fakeredis_client.hdel("h", "f") == 0


# ---------------------------------------------------------------------------
# List operations (fakeredis)
# ---------------------------------------------------------------------------


class TestRedisClientListOperations:
    """Tests for list operations with fakeredis."""

    async def test_lpush_and_llen(self, fakeredis_client):
        result = await fakeredis_client.lpush("mylist", "a", "b", "c")
        assert result == 3
        assert await fakeredis_client.llen("mylist") == 3

    async def test_rpop_returns_oldest_element(self, fakeredis_client):
        """lpush adds to the left, rpop removes from the right (FIFO)."""
        await fakeredis_client.lpush("q", "first")
        await fakeredis_client.lpush("q", "second")

        result = await fakeredis_client.rpop("q")
        assert result == "first"

    async def test_rpop_empty_list_returns_none(self, fakeredis_client):
        assert await fakeredis_client.rpop("empty-list") is None

    async def test_llen_empty_list(self, fakeredis_client):
        assert await fakeredis_client.llen("empty-list") == 0

    # --- Error paths ---

    async def test_lpush_returns_zero_on_error(self, fakeredis_client):
        fakeredis_client._client.lpush = AsyncMock(side_effect=RedisError("err"))
        assert await fakeredis_client.lpush("list", "a") == 0

    async def test_rpop_returns_none_on_error(self, fakeredis_client):
        fakeredis_client._client.rpop = AsyncMock(side_effect=RedisError("err"))
        assert await fakeredis_client.rpop("list") is None

    async def test_llen_returns_zero_on_error(self, fakeredis_client):
        fakeredis_client._client.llen = AsyncMock(side_effect=RedisError("err"))
        assert await fakeredis_client.llen("list") == 0


# ---------------------------------------------------------------------------
# Set operations (fakeredis)
# ---------------------------------------------------------------------------


class TestRedisClientSetOperations:
    """Tests for set operations with fakeredis."""

    async def test_sadd_and_smembers(self, fakeredis_client):
        added = await fakeredis_client.sadd("myset", "a", "b", "c")
        assert added == 3

        members = await fakeredis_client.smembers("myset")
        assert members == {"a", "b", "c"}

    async def test_sadd_ignores_duplicates(self, fakeredis_client):
        await fakeredis_client.sadd("myset", "a", "b")
        added = await fakeredis_client.sadd("myset", "b", "c")
        assert added == 1  # Only "c" was new

    async def test_srem_removes_members(self, fakeredis_client):
        await fakeredis_client.sadd("myset", "a", "b", "c")
        removed = await fakeredis_client.srem("myset", "a", "c")
        assert removed == 2

        members = await fakeredis_client.smembers("myset")
        assert members == {"b"}

    async def test_scard_returns_count(self, fakeredis_client):
        await fakeredis_client.sadd("myset", "x", "y", "z")
        assert await fakeredis_client.scard("myset") == 3

    async def test_scard_empty_set(self, fakeredis_client):
        assert await fakeredis_client.scard("empty-set") == 0

    async def test_smembers_empty_set(self, fakeredis_client):
        assert await fakeredis_client.smembers("empty-set") == set()

    # --- Error paths ---

    async def test_sadd_returns_zero_on_error(self, fakeredis_client):
        fakeredis_client._client.sadd = AsyncMock(side_effect=RedisError("err"))
        assert await fakeredis_client.sadd("s", "a") == 0

    async def test_srem_returns_zero_on_error(self, fakeredis_client):
        fakeredis_client._client.srem = AsyncMock(side_effect=RedisError("err"))
        assert await fakeredis_client.srem("s", "a") == 0

    async def test_scard_returns_zero_on_error(self, fakeredis_client):
        fakeredis_client._client.scard = AsyncMock(side_effect=RedisError("err"))
        assert await fakeredis_client.scard("s") == 0

    async def test_smembers_returns_empty_set_on_error(self, fakeredis_client):
        fakeredis_client._client.smembers = AsyncMock(side_effect=RedisError("err"))
        assert await fakeredis_client.smembers("s") == set()


# ---------------------------------------------------------------------------
# Pub/Sub
# ---------------------------------------------------------------------------


class TestRedisClientPubSub:
    """Tests for publish + pubsub."""

    async def test_publish_to_channel_with_no_subscribers_returns_zero(self, fakeredis_client):
        n = await fakeredis_client.publish("recording:jobs:t1", '{"event":"status_changed"}')
        assert n == 0

    async def test_publish_delivers_to_subscriber(self, fakeredis_client):
        ps = fakeredis_client.pubsub()
        await ps.subscribe("recording:jobs:t1")
        # discard the subscribe-confirmation message
        await ps.get_message(timeout=1)
        await fakeredis_client.publish("recording:jobs:t1", '{"event":"status_changed"}')
        msg = await ps.get_message(timeout=1)
        assert msg is not None
        assert msg["type"] == "message"
        assert msg["data"] == '{"event":"status_changed"}'
        await ps.aclose()

    async def test_publish_raises_when_not_connected(self):
        client = RedisClient()
        with pytest.raises(RuntimeError, match="not connected"):
            await client.publish("ch", "msg")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestRedisClientPipeline:
    """Tests for pipeline."""

    async def test_pipeline_returns_pipeline_object(self, fakeredis_client):
        pipe = await fakeredis_client.pipeline()
        assert pipe is not None

    async def test_pipeline_raises_when_not_connected(self):
        client = RedisClient()
        with pytest.raises(RuntimeError, match="not connected"):
            await client.pipeline()


# ---------------------------------------------------------------------------
# _ensure_connected
# ---------------------------------------------------------------------------


class TestRedisClientEnsureConnected:
    def test_raises_when_not_connected(self):
        client = RedisClient()
        with pytest.raises(RuntimeError, match="not connected"):
            client._ensure_connected()

    def test_returns_client(self, fakeredis_client):
        result = fakeredis_client._ensure_connected()
        assert result is fakeredis_client._client


# ---------------------------------------------------------------------------
# Global singleton functions
# ---------------------------------------------------------------------------


class TestGlobalFunctions:
    """Tests for global singleton management functions."""

    def test_get_redis_client_creates_singleton(self):
        client = get_redis_client()
        assert isinstance(client, RedisClient)

    def test_get_redis_client_returns_same_instance(self):
        client1 = get_redis_client()
        client2 = get_redis_client()
        assert client1 is client2

    @patch("w2v_common.cache.redis_client.RedisClient.connect", new_callable=AsyncMock)
    async def test_init_redis(self, mock_connect):
        client = await init_redis()
        assert isinstance(client, RedisClient)
        mock_connect.assert_called_once()

    @patch("w2v_common.cache.redis_client.RedisClient.disconnect", new_callable=AsyncMock)
    async def test_close_redis(self, mock_disconnect):
        import w2v_common.cache.redis_client as mod

        mod._redis_client = RedisClient()
        await close_redis()
        mock_disconnect.assert_called_once()
        assert mod._redis_client is None

    async def test_close_redis_when_none(self):
        await close_redis()


# ---------------------------------------------------------------------------
# get_strict — additive outage-vs-miss discriminator
# ---------------------------------------------------------------------------


class TestGetStrict:
    """``get`` collapses "missing" and "Redis down" into ``None``;
    ``get_strict`` is the additive variant that keeps them apart."""

    async def test_hit_matches_get(self, fakeredis_client):
        await fakeredis_client.set("k", {"a": 1})
        assert await fakeredis_client.get_strict("k") == {"a": 1}
        assert await fakeredis_client.get_strict("k") == await fakeredis_client.get("k")

    async def test_miss_returns_none(self, fakeredis_client):
        assert await fakeredis_client.get_strict("absent") is None

    async def test_redis_error_raises_unavailable(self, fakeredis_client):
        from redis.exceptions import ConnectionError as RedisConnectionError

        from w2v_common.cache.redis_client import RedisUnavailableError

        raw = fakeredis_client._ensure_connected()
        boom = RedisConnectionError("down")
        with patch.object(raw, "get", AsyncMock(side_effect=boom)):
            with pytest.raises(RedisUnavailableError) as excinfo:
                await fakeredis_client.get_strict("k")
            # The default contract of ``get`` is untouched: still ``None``.
            assert await fakeredis_client.get("k") is None
        assert excinfo.value.__cause__ is boom
        # Subclasses RedisError so every existing ``except RedisError`` holds.
        assert isinstance(excinfo.value, RedisError)

    async def test_not_connected_raises_unavailable(self):
        from w2v_common.cache.redis_client import RedisUnavailableError

        client = RedisClient()
        with pytest.raises(RedisUnavailableError):
            await client.get_strict("k")
        assert await client.get("k") is None

    @pytest.mark.parametrize(
        "error_name", ["MaxConnectionsError", "ResponseError", "ReadOnlyError"]
    )
    async def test_non_outage_error_propagates_unwrapped(self, fakeredis_client, error_name):
        """Not a miss (so it must raise) but not an outage (so not Unavailable)."""
        import redis.exceptions as redis_exc

        from w2v_common.cache.redis_client import RedisUnavailableError

        raw = fakeredis_client._ensure_connected()
        boom = getattr(redis_exc, error_name)("server is up")
        with (
            patch.object(raw, "get", AsyncMock(side_effect=boom)),
            pytest.raises(RedisError) as excinfo,
        ):
            await fakeredis_client.get_strict("k")
        assert excinfo.value is boom
        assert not isinstance(excinfo.value, RedisUnavailableError)

    async def test_os_error_raises_unavailable(self, fakeredis_client):
        from w2v_common.cache.redis_client import RedisUnavailableError

        raw = fakeredis_client._ensure_connected()
        with (
            patch.object(raw, "get", AsyncMock(side_effect=ConnectionResetError("reset"))),
            pytest.raises(RedisUnavailableError),
        ):
            await fakeredis_client.get_strict("k")


def test_is_redis_outage_classification():
    import redis.exceptions as redis_exc

    from w2v_common.cache.redis_client import RedisUnavailableError, is_redis_outage

    for outage in (
        redis_exc.ConnectionError("refused"),
        redis_exc.TimeoutError("timed out"),
        redis_exc.BusyLoadingError("loading"),
        RedisUnavailableError("not connected"),
        OSError("unreachable"),
        TimeoutError("socket"),
    ):
        assert is_redis_outage(outage), outage
    # MaxConnectionsError IS a redis ConnectionError — the exclusion is explicit.
    assert issubclass(redis_exc.MaxConnectionsError, redis_exc.ConnectionError)
    for not_outage in (
        redis_exc.MaxConnectionsError("pool exhausted"),
        redis_exc.ResponseError("WRONGTYPE"),
        redis_exc.ReadOnlyError("READONLY"),
        redis_exc.OutOfMemoryError("OOM"),
        RedisError("generic"),
        ValueError("unrelated"),
    ):
        assert not is_redis_outage(not_outage), not_outage
