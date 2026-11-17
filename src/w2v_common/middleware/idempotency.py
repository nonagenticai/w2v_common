"""Idempotency-Key middleware for POST endpoints.

Implements the de-facto Stripe / IETF draft-ietf-httpapi-idempotency-key
shape: a client opts in by sending an ``Idempotency-Key`` header on a
mutating request. The first request executes normally and the
``(status, headers, body)`` tuple is cached under
``idem:{sha256(tenant|method|path|query|key|body)}`` with a 24h default TTL.
Subsequent requests carrying the same key replay the cached response verbatim,
even if the original handler created side effects (e.g. a new ``runs`` row), so
a client retrying after a transport blip can be sure the operation is applied
exactly once.

**Tenant is part of the cache key** (read from ``request.state.tenant_id``,
populated by ``AuthMiddleware`` which runs inner-adjacent to this middleware):
without it, tenant B replaying tenant A's key+path+body would receive A's
cached response — a cross-tenant leak. Only **successful** responses
(``status < 400``) are cached: a 4xx is a client error that a corrected retry
should be allowed to re-attempt rather than have pinned for 24h.

The middleware is intentionally narrow:

* Only ``POST`` is intercepted (PATCH / PUT replay risk is bounded by their
  natural idempotency; intercepting them too would also break partial
  update semantics that depend on the latest server state).
* The ``Idempotency-Key`` header is optional — endpoints don't change
  shape, callers opt in by sending it.
* If the cache is unavailable the middleware falls open (request runs
  through uncached). The trade-off is "duplicate run on retry once" vs.
  "every POST 503 when the cache blips" — the former is recoverable, the
  latter takes down the whole control plane.
* Falling open is FAST once an outage is ESTABLISHED. A read that fails
  because the store is unreachable (connection refused/reset/timeout — see
  :func:`w2v_common.cache.redis_client.is_redis_outage`) counts toward
  ``IDEMPOTENCY_OUTAGE_THRESHOLD`` (default 2) CONSECUTIVE such failures; any
  successful read resets the count. Below the threshold nothing changes: a
  single blip is handled exactly as before the circuit existed (that request
  treats the read as a miss, then locks and writes) and no other request is
  affected — so one dropped packet can never make a same-key retry re-execute
  a paid handler. The failure that REACHES the threshold ends the store work
  for its request (no lock, no write — each would burn the socket timeout
  again for a request already being served uncached), and opens a circuit:
  for ``IDEMPOTENCY_OUTAGE_COOLDOWN_SECONDS`` (default 10) no NEW request
  touches the store. The first request to arrive once the window has elapsed
  probes with a single read; success closes the circuit and that same request
  proceeds to lock + write. The window is purely time-based: it does not need
  traffic to expire, so on a quiet service the next keyed request — however
  much later — is the probe.
* The circuit gates ADMISSION only. A request that was admitted before the
  circuit opened is never affected by it: in particular a lock holder always
  attempts its own write and its own lock release, so a transient error seen
  by an unrelated request cannot cost a paid handler its cached response.
* Only unreachability opens the circuit. A pool-exhaustion error or a server
  error reply (``OOM``, ``READONLY``) falls open for that one call, exactly as
  before, and a failed lock or write never opens it — those say nothing
  reliable about whether cached responses can still be READ.
* The key in the URL hash includes the body sha256 to defend against a
  client reusing a key across distinct request bodies (a 422 in that
  case would be more pedantic, but the cache hit is harmless: the
  identical-body case wins, the distinct-body case never collides).

The cache is read from ``request.app.state.redis`` and only needs the
``IdempotencyStore`` surface below — any object exposing async
``get``/``set(..., ttl=...)``/``delete`` works. The canonical
``w2v_common.cache.redis_client.RedisClient`` satisfies it; the atomic
SET NX EX lock additionally uses the optional ``_ensure_connected`` accessor
and degrades to a non-atomic CAS when it is absent (test fakes, alternate
clients).

To enable in a service using :func:`w2v_common.asgi.create_app`, add an
``IdempotencySettings`` field to the service Settings and set
``IDEMPOTENCY_ENABLED=true``; the service is responsible for assigning a
connected cache to ``app.state.redis`` in an ``on_startup`` hook.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, Protocol

from redis.exceptions import RedisError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from ..cache.redis_client import is_redis_outage
from ..tenant import TenantError, parse_tenant_id, require_tenant_id

logger = logging.getLogger(__name__)

_INTERCEPTED_METHODS = frozenset({"POST"})
_HEADER = "Idempotency-Key"
_KEY_NAMESPACE = "idem"
# Cap the cached body to avoid making the cache a free response store; POST
# envelopes are well under this in practice. A request that returns more than
# 256 KiB is almost certainly a streaming file download misdirected to POST —
# skip caching the body rather than blow up the cache.
_MAX_CACHED_BODY_BYTES = 256 * 1024
# Lock TTL must outlast the slowest cached handler so a crash auto-heals
# (the cache reclaims the lock) without blocking retries indefinitely.
_LOCK_TTL_SECONDS = 30
# Polling budget for second-in concurrent requests: 30s total at 100ms each.
_LOCK_POLL_INTERVAL = 0.1
_LOCK_POLL_BUDGET = 30.0
# Framing headers that describe THIS transfer, not the payload. Replaying them
# verbatim from cache is wrong (a stale ``content-length`` or a ``chunked``
# ``transfer-encoding`` against a fixed body corrupts the response); strip them
# and let Starlette recompute ``content-length`` from the cached body.
_NON_REPLAYABLE_HEADERS = frozenset({"content-length", "transfer-encoding", "connection"})
# Marker written into the cache payload when the body is not valid UTF-8 and
# therefore had to be base64-encoded to survive the JSON round-trip.
_BODY_ENCODING_BASE64 = "base64"
# How long the store circuit stays open after an outage is observed. Same
# ``IDEMPOTENCY_`` prefix as ``IdempotencySettings`` so a settings field named
# ``outage_cooldown_seconds`` would read the identical variable.
_COOLDOWN_ENV = "IDEMPOTENCY_OUTAGE_COOLDOWN_SECONDS"
DEFAULT_OUTAGE_COOLDOWN_SECONDS = 10.0
# Consecutive outage-class READ failures needed to open the circuit. 2, not 1:
# the pool does not retry, so a single timeout blip on a healthy Redis is
# routine, and opening on it would make every same-key retry in the next
# window skip the store and re-execute its (paid) handler.
_THRESHOLD_ENV = "IDEMPOTENCY_OUTAGE_THRESHOLD"
DEFAULT_OUTAGE_THRESHOLD = 2

# The CAS fallback below is a *degraded*, non-atomic lock. In production it
# should never be reached (the canonical RedisClient exposes ``set_nx``), so the
# condition is worth a WARNING rather than the DEBUG it used to carry — the
# fleet runs LOG_LEVEL=WARNING, so a DEBUG line is invisible exactly where it
# matters. It is emitted once per (process, backend type): the condition is
# static for the lifetime of a backend, and the call site is on the hot path of
# every idempotent POST, so an unguarded WARNING would be a log flood.
_cas_fallback_warned: set[str] = set()


def _warn_cas_fallback_once(redis: object) -> None:
    """Emit the non-atomic-lock warning once per backend type per process."""
    backend = type(redis).__qualname__
    if backend in _cas_fallback_warned:
        return
    _cas_fallback_warned.add(backend)
    logger.warning(
        "idempotency: set_nx unavailable on %s; using NON-ATOMIC CAS lock "
        "(duplicate concurrent execution is possible)",
        backend,
    )


class _StoreUnavailableError(Exception):
    """Internal signal: a store call failed for an outage reason (not a miss)."""


class _Admission(Enum):
    CLOSED = "closed"  # store believed healthy: use it normally
    PROBE = "probe"  # cool-down elapsed: this ONE request may test the store
    SKIP = "skip"  # circuit open: do not touch the store


class _StoreCircuit:
    """Cool-down circuit breaker over the idempotency store.

    One per middleware instance, i.e. one per app per process. Every method is
    synchronous and none awaits, so under asyncio each transition is atomic
    with respect to other requests — that, not a lock, is what guarantees a
    single probe when many requests arrive as the cool-down expires.

    It opens after ``threshold`` CONSECUTIVE outage-class read failures (a
    successful read in between resets the count), so an isolated blip never
    opens it.

    It is consulted ONLY at admission (:meth:`admit`), before a request has
    touched the store. State is a timestamp compared against the clock on each
    admission — there is no timer and nothing that needs traffic during the
    window: whichever request arrives first at or after ``opened_at +
    cooldown`` is the probe, whether that is 10 seconds or 10 hours later.
    """

    def __init__(
        self, cooldown_seconds: float, clock: Callable[[], float], threshold: int = 1
    ) -> None:
        self._cooldown = cooldown_seconds
        self._threshold = threshold
        self._consecutive_failures = 0
        self._clock = clock
        self._opened_at: float | None = None
        self._first_opened_at = 0.0
        self._probing = False

    @property
    def is_open(self) -> bool:
        """True from the first observed outage until a probe succeeds.

        Observability only — request handling must go through :meth:`admit`.
        """
        return self._opened_at is not None

    def admit(self) -> _Admission:
        if self._opened_at is None:
            return _Admission.CLOSED
        if self._probing or self._clock() - self._opened_at < self._cooldown:
            return _Admission.SKIP
        self._probing = True
        return _Admission.PROBE

    def record_failure(self, reason: object, *, probe: bool) -> bool:
        """Count an outage-class read failure; return whether the circuit is open.

        ``False`` means the outage is not established yet: the caller must
        carry on exactly as it would have without a circuit.
        """
        if self._opened_at is None:
            self._consecutive_failures += 1
            if self._consecutive_failures < self._threshold:
                return False
            self._opened_at = self._first_opened_at = self._clock()
            logger.warning(
                "idempotency: store circuit OPEN after %d consecutive failed reads "
                "(%s); skipping the idempotency store for %.1fs windows — keyed "
                "requests run UN-DEDUPLICATED until a probe succeeds",
                self._consecutive_failures,
                reason,
                self._cooldown,
            )
        elif probe:
            # Failed probe: hold for another full window. A late failure from a
            # request that was already in flight when the circuit opened must
            # NOT extend the window, hence the ``probe`` guard.
            self._opened_at = self._clock()
            self._probing = False
            logger.info("idempotency: store probe failed (%s); circuit stays OPEN", reason)
        return True

    def record_success(self, *, probe: bool) -> None:
        if self._opened_at is None:
            # Closed: any good read proves reachability and clears the count.
            self._consecutive_failures = 0
            return
        # Only the designated probe may close: a stray success from a request
        # in flight during a partial outage would otherwise flap the circuit.
        if probe:
            logger.warning(
                "idempotency: store circuit CLOSED after %.1fs; deduplication restored",
                self._clock() - self._first_opened_at,
            )
            self._opened_at = None
            self._probing = False
            self._consecutive_failures = 0

    def abandon_probe(self) -> None:
        """Probe ended with no verdict (cancelled): let the next request probe."""
        self._probing = False


def _resolve_cooldown(explicit: float | None) -> float:
    raw: float | str = explicit if explicit is not None else os.getenv(_COOLDOWN_ENV, "")
    if raw == "":
        return DEFAULT_OUTAGE_COOLDOWN_SECONDS
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if value < 0:
        raise ValueError(f"{_COOLDOWN_ENV} must be a non-negative number of seconds, got {raw!r}")
    return value


def _resolve_threshold(explicit: int | None) -> int:
    raw: int | str = explicit if explicit is not None else os.getenv(_THRESHOLD_ENV, "")
    if raw == "":
        return DEFAULT_OUTAGE_THRESHOLD
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"{_THRESHOLD_ENV} must be an integer >= 1, got {raw!r}")
    return value


class IdempotencyStore(Protocol):
    """Minimal async cache surface the middleware needs from ``app.state.redis``.

    Satisfied by :class:`w2v_common.cache.redis_client.RedisClient`. The atomic
    SET NX EX fast-path uses the optional ``set_nx`` method (below); a backend
    exposing only ``get``/``set``/``delete`` still works via the middleware's
    non-atomic CAS fallback. A backend that also exposes ``get_strict`` (raise
    on outage instead of returning ``None``) is read through it, so an outage
    is recognised on the first call rather than mistaken for a miss.
    """

    async def get(self, key: str) -> Any: ...
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool: ...
    async def delete(self, key: str) -> bool: ...
    async def set_nx(self, key: str, value: Any, *, ex: int) -> bool | None: ...


class IdempotencyMiddleware(BaseHTTPMiddleware):
    """Cache POST responses keyed by client-supplied ``Idempotency-Key``."""

    def __init__(
        self,
        app: ASGIApp,
        ttl_seconds: int = 86400,
        *,
        outage_cooldown_seconds: float | None = None,
        outage_threshold: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(app)
        self._ttl_seconds = ttl_seconds
        self._outage_cooldown_seconds = _resolve_cooldown(outage_cooldown_seconds)
        self._outage_threshold = _resolve_threshold(outage_threshold)
        self._circuit = _StoreCircuit(self._outage_cooldown_seconds, clock, self._outage_threshold)

    @property
    def outage_threshold(self) -> int:
        """Consecutive unreachable reads that open the store circuit."""
        return self._outage_threshold

    @property
    def outage_cooldown_seconds(self) -> float:
        """Seconds the store is left alone after an outage is observed."""
        return self._outage_cooldown_seconds

    @property
    def store_circuit_open(self) -> bool:
        """True while the store is being skipped (outage seen, no good probe yet)."""
        return self._circuit.is_open

    async def dispatch(  # noqa: C901  # why: linear middleware pipeline; splitting the lock/cache/drain stages across helpers hurts readability more than the branch count
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Replay cached POST response when the client opts in via Idempotency-Key."""
        if request.method not in _INTERCEPTED_METHODS:
            return await call_next(request)
        idem_key = request.headers.get(_HEADER) or request.headers.get(_HEADER.lower())
        if not idem_key:
            return await call_next(request)

        redis: IdempotencyStore | None = getattr(request.app.state, "redis", None)
        if redis is None:
            # Cache unavailable at startup — fall open. A service that
            # enables idempotency runs in a degraded mode when its cache is
            # down; idempotency replay is not the place to add a second hard
            # dependency on it.
            return await call_next(request)

        body = await request.body()
        try:
            cache_key = self._compose_cache_key(request, idem_key, body)
        except TenantError as exc:
            # Fail CLOSED rather than fall open. Everywhere else in this
            # middleware a failure falls open (the request runs uncached) —
            # that is right for a cache blip and wrong here: falling open on a
            # missing tenant is exactly the shared-bucket collision the tenant
            # is in the key to prevent, and it would keep the defect invisible.
            # AuthMiddleware runs inner-adjacent, so a request that gets here
            # with no tenant is on an auth-disabled surface: name it.
            logger.warning("idempotency: refusing tenant-less request (%s)", exc)
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "TENANT_REQUIRED",
                        "message": (
                            "Idempotency-Key requires a tenant-scoped request; "
                            "forward the requestor's X-Tenant-Id"
                        ),
                        "status": 400,
                    }
                },
            )
        lock_key = f"{cache_key}:lock"

        admission = self._circuit.admit()
        if admission is _Admission.SKIP:
            # Store is known-down and inside its cool-down: serve uncached
            # without spending a socket timeout to rediscover that.
            request._body = body
            return await call_next(request)

        is_probe = admission is _Admission.PROBE
        # ``store_ok`` is the per-request short-circuit: once THIS request's
        # own read fails for an outage reason, it makes no further store
        # calls. Nothing another request does can clear it (see module doc).
        store_ok = True
        cached: dict[str, Any] | None = None
        try:
            cached = await self._read(redis, cache_key, probe=is_probe)
        except _StoreUnavailableError:
            store_ok = False
        finally:
            if is_probe:
                # No-op unless the read was cancelled before reaching a verdict.
                self._circuit.abandon_probe()
        if cached is not None:
            return _build_replay_response(cached)

        # Atomic lock: SET NX EX. Only the first concurrent request with this
        # key acquires it; the others poll the cache key until the leader
        # writes a result (or until the poll budget elapses).
        lock_state: bool | None = None
        if store_ok:
            lock_state = await self._acquire_lock(redis, lock_key)
        got_lock = lock_state is True
        if lock_state is False:
            # Another worker holds the lock — poll for their cached result.
            try:
                cached = await self._poll_for_cached(redis, cache_key)
            except _StoreUnavailableError:
                # The store died mid-poll: stop polling a dead store.
                store_ok = False
            if cached is not None:
                return _build_replay_response(cached)
            # Leader crashed before writing OR the cache lost the lock. Fall
            # through to execute; correctness degrades to "exactly-once if
            # nothing crashes" — the same guarantee the lock-less version
            # ever offered.
            if store_ok:
                logger.warning(
                    "idempotency: lock holder did not publish a result within %.1fs; "
                    "executing as best-effort",
                    _LOCK_POLL_BUDGET,
                )
        # ``lock_state is None`` means the cache is broken; skip the poll
        # (it would block the request for the whole budget without progress)
        # and execute uncached.

        try:
            # Replay-safe forwarding: starlette consumes ``request._body`` once;
            # re-attach so the downstream handler can read it again.
            request._body = body

            response = await call_next(request)
            # Only cache success. A 5xx is transient infra failure a retry
            # should re-attempt against fresh code paths; a 4xx is a client
            # error a corrected retry should be allowed to fix — neither should
            # be pinned for 24h.
            if response.status_code >= 400:
                return response

            # ``BaseHTTPMiddleware`` always hands us a streaming response
            # (Starlette's private ``_StreamingResponse``; not the public
            # ``StreamingResponse`` class, so we duck-type on the iterator
            # attribute instead of an isinstance check). Drain it so we can
            # cache + re-emit. If a future Starlette release flips to a
            # non-streaming Response, fall through uncached.
            if not hasattr(response, "body_iterator"):
                return response

            response_body = await _drain_response_body(response)

            cache_payload: dict[str, Any] = {
                "status": response.status_code,
                "headers": _replayable_headers(response.raw_headers),
                "media_type": response.media_type,
            }
            if len(response_body) <= _MAX_CACHED_BODY_BYTES:
                cache_payload.update(_encode_cached_body(response_body))
            else:
                # Body too large to cache verbatim, but the request executed
                # and the side effect is committed. Store an oversized marker
                # so a retry receives a deterministic status + headers and a
                # sentinel the client can branch on.
                cache_payload["body"] = ""
                cache_payload["_oversized"] = True
            # Skipped only when this request's OWN read hit an outage. The
            # circuit state is deliberately not consulted: the handler has
            # run, and this write is what makes a retry replay instead of
            # executing (and billing) a second time.
            if store_ok:
                await self._write(redis, cache_key, cache_payload)

            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=_replayable_headers(response.raw_headers),
                media_type=response.media_type,
            )
        finally:
            # A holder ALWAYS releases its own lock, whatever the circuit
            # says — an orphan lock stalls every retry of this key for the
            # whole poll budget.
            if got_lock:
                await self._release_lock(redis, lock_key)

    @staticmethod
    def _compose_cache_key(request: Request, idem_key: str, body: bytes) -> str:
        # Tenant scopes the key so one tenant's cached response can never be
        # replayed to another. AuthMiddleware sets ``request.state.tenant_id``
        # before this middleware runs (Idempotency is inner of Auth in the
        # canonical stack).
        #
        # ⭐ The ``or "anonymous"`` fallback that used to live here was a
        # cross-tenant cache-poisoning primitive, not a convenience: EVERY
        # tenant-less caller hashed into ONE shared bucket, so the first
        # response cached under a given Idempotency-Key was replayed to every
        # other tenant-less caller that reused it. Requiring the tenant means a
        # misconfigured surface fails loudly instead of serving one caller's
        # body to another.
        tenant = require_tenant_id(
            parse_tenant_id(
                getattr(request.state, "tenant_id", None),
                source="idempotency cache key",
            )
        )
        body_hash = hashlib.sha256(body).hexdigest()
        # The QUERY STRING is part of the request identity. Omitting it made
        # ``/search?q=alpha`` and ``/search?q=beta`` share one cache entry under
        # the same Idempotency-Key, so the second call replayed the first
        # call's response for a materially different request.
        query = request.url.query
        composite = hashlib.sha256(
            f"{tenant}|{request.method}|{request.url.path}|{query}|{idem_key}|{body_hash}".encode()
        ).hexdigest()
        return f"{_KEY_NAMESPACE}:{composite}"

    async def _read(
        self, redis: IdempotencyStore, key: str, *, probe: bool = False
    ) -> dict[str, Any] | None:
        """Read a cached response; ``None`` means "treat as a miss".

        That includes an unreachable-store failure that has not yet reached the
        outage threshold: it is logged and handled as a miss, as it always was.

        This is the circuit's only input: every request reads first, so an
        unreachable store is always discovered here.

        Raises:
            _StoreUnavailableError: the store is unreachable AND the circuit
                is open (this failure opened it, or it already was). The
                caller must stop using the store for this request.
        """
        # ``get_strict`` raises on outage where the canonical client's ``get``
        # returns ``None`` — without it an outage is indistinguishable from a
        # miss. Looked up on the TYPE, not the instance: a ``Mock``/``AsyncMock``
        # store answers ``getattr`` for any name, which would silently route
        # around the ``get.return_value`` a consumer's test stubbed.
        getter = redis.get
        if callable(getattr(type(redis), "get_strict", None)):
            getter = redis.get_strict  # ty: ignore[unresolved-attribute]
        try:
            raw = await getter(key)
        except (RedisError, OSError) as exc:
            if is_redis_outage(exc) and self._circuit.record_failure(
                f"GET failed: {exc}", probe=probe
            ):
                raise _StoreUnavailableError from exc
            # Either reachable but erroring (pool exhausted, server error
            # reply) or an unreachable blip below the outage threshold: fall
            # open for this read only, as before the circuit existed.
            logger.warning("idempotency: cache GET failed (%s); falling open", exc)
            return None
        self._circuit.record_success(probe=probe)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except TypeError, ValueError:
            logger.warning("idempotency: cached payload at %s was not valid JSON", key)
            return None

    async def _write(self, redis: IdempotencyStore, key: str, payload: dict[str, Any]) -> None:
        # Never feeds the circuit: a refused write (OOM, READONLY) says
        # nothing about whether cached responses can still be read.
        try:
            await redis.set(key, json.dumps(payload), ttl=self._ttl_seconds)
        except RedisError as exc:
            logger.warning("idempotency: cache SET failed (%s); response will not be cached", exc)

    @staticmethod
    async def _acquire_lock(redis: IdempotencyStore, lock_key: str) -> bool | None:
        """Atomic SET NX EX.

        Tri-state return:

        * ``True``  — this caller now owns the lock.
        * ``False`` — someone else owns the lock (contended; safe to poll).
        * ``None``  — the cache is broken or the lock primitive is
                     unavailable; skip polling and execute uncached (the
                     previous degraded mode).

        Uses the canonical w2v_common ``RedisClient.set_nx`` (public atomic
        ``SET NX EX``) whose tri-state return maps directly onto ours. Any
        backend lacking ``set_nx`` (a bare test fake exposing only
        get/set/delete) gets the non-atomic CAS fallback below.
        """
        set_nx = getattr(redis, "set_nx", None)
        if callable(set_nx):
            return await set_nx(lock_key, "1", ex=_LOCK_TTL_SECONDS)

        # Backend without ``set_nx`` (minimal test fake): non-atomic CAS that is
        # safe enough for unit tests; production always uses the real client.
        _warn_cas_fallback_once(redis)
        try:
            existing = await redis.get(lock_key)
            if existing is not None:
                return False
            await redis.set(lock_key, "1", ttl=_LOCK_TTL_SECONDS)
            return True
        except RedisError as redis_exc:
            logger.warning("idempotency: CAS lock failed (%s); falling open", redis_exc)
            return None

    @staticmethod
    async def _release_lock(redis: IdempotencyStore, lock_key: str) -> None:
        try:
            await redis.delete(lock_key)
        except RedisError as exc:
            logger.warning(
                "idempotency: cache DEL on lock %s failed (%s); will TTL-expire",
                lock_key,
                exc,
            )

    async def _poll_for_cached(
        self, redis: IdempotencyStore, cache_key: str
    ) -> dict[str, Any] | None:
        """Poll the cache key while another worker is executing the handler.

        Bounded by ``_LOCK_POLL_BUDGET`` so a stuck leader doesn't hang the
        follower forever.
        """
        elapsed = 0.0
        while elapsed < _LOCK_POLL_BUDGET:
            await asyncio.sleep(_LOCK_POLL_INTERVAL)
            elapsed += _LOCK_POLL_INTERVAL
            cached = await self._read(redis, cache_key)
            if cached is not None:
                return cached
        return None


def _replayable_headers(raw_headers: list[tuple[bytes, bytes]]) -> dict[str, str]:
    """Decode response headers, dropping per-transfer framing headers.

    ``content-length`` / ``transfer-encoding`` / ``connection`` describe the
    original transfer, not the payload — carrying them into a cached replay (or
    a rebuilt live response over the drained body) corrupts framing. Starlette
    recomputes ``content-length`` from the body we pass, so stripping them is
    both safe and necessary.
    """
    return {
        k.decode(): v.decode()
        for k, v in raw_headers
        if k.decode().lower() not in _NON_REPLAYABLE_HEADERS
    }


def _encode_cached_body(body: bytes) -> dict[str, str]:
    """Serialise a response body for the JSON cache payload without loss.

    The cache payload is JSON, which carries text but not arbitrary bytes. The
    previous implementation used ``decode("utf-8", errors="replace")``, which
    substitutes U+FFFD for every byte that is not valid UTF-8 — irreversible,
    so any binary response (audio, image, video) replayed from cache came back
    corrupt.

    Bodies that ARE valid UTF-8 keep the plain ``body`` string field: that
    round-trips exactly, stays readable in the cache, and is byte-identical to
    what entries written by earlier releases contain. Anything else is
    base64-encoded under ``body`` with a ``body_encoding`` marker.
    """
    try:
        return {"body": body.decode("utf-8")}
    except UnicodeDecodeError:
        return {
            "body": base64.b64encode(body).decode("ascii"),
            "body_encoding": _BODY_ENCODING_BASE64,
        }


def _decode_cached_body(cached: dict[str, Any]) -> bytes | str:
    """Inverse of :func:`_encode_cached_body`.

    Entries written before 0.57.0 have no ``body_encoding`` key and are read as
    plain text — the same way they were written — so an in-flight cache does
    not need to be purged on upgrade.
    """
    body = cached["body"]
    if cached.get("body_encoding") == _BODY_ENCODING_BASE64:
        return base64.b64decode(body)
    return body


def _build_replay_response(cached: dict[str, Any]) -> Response:
    replay_headers = {**cached["headers"], "Idempotent-Replay": "true"}
    if cached.get("_oversized"):
        replay_headers["Idempotent-Replay-Oversized"] = "true"
        return Response(
            content=b"",
            status_code=cached["status"],
            headers=replay_headers,
            media_type=cached.get("media_type"),
        )
    return Response(
        content=_decode_cached_body(cached),
        status_code=cached["status"],
        headers=replay_headers,
        media_type=cached.get("media_type"),
    )


async def _drain_response_body(response: Any) -> bytes:
    """Concatenate the StreamingResponse-like body iterator into a single bytes."""
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunks.append(chunk)
        elif isinstance(chunk, memoryview):
            chunks.append(bytes(chunk))
        else:
            chunks.append(chunk.encode())
    return b"".join(chunks)
