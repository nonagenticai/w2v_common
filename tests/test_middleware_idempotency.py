"""Tests for w2v_common.middleware.idempotency.IdempotencyMiddleware.

The first block ports the orchestrator's unit suite verbatim (a hand
``_FakeRedis`` that exercises the non-atomic CAS lock fallback). The second
block drives the canonical ``w2v_common.cache.redis_client.RedisClient``
backed by fakeredis so the SET NX EX fast-path (``_ensure_connected``) and
the serialize/deserialize round-trip are covered end-to-end. The third block
verifies the ``create_app`` opt-in gate via ``IdempotencySettings``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pydantic import Field

from w2v_common.asgi import create_app
from w2v_common.config import APISettings, IdempotencySettings
from w2v_common.middleware.idempotency import IdempotencyMiddleware

# CONTRACT CHANGE: the cache key REQUIRES a real tenant UUID. The middleware
# used to fall back to the literal "anonymous", which put every tenant-less
# caller in ONE shared bucket — a cross-tenant replay primitive. These fixtures
# therefore seed a tenant the way AuthMiddleware does in production.
_TENANT_A = "0195f48b-0000-7000-8000-00000000aa01"
_TENANT_B = "0195f48b-0000-7000-8000-00000000bb02"


def _seed_tenant(app: FastAPI, default: str = _TENANT_A) -> None:
    """Bind ``request.state.tenant_id`` OUTER of IdempotencyMiddleware.

    Starlette dispatches added middleware LIFO, so this must be added AFTER
    IdempotencyMiddleware to run before it — same ordering as AuthMiddleware in
    the canonical stack.
    """

    @app.middleware("http")
    async def _bind(request: Any, call_next: Any) -> Any:
        request.state.tenant_id = request.headers.get("X-Tenant", default)
        return await call_next(request)


class _FakeRedis:
    """Async stand-in for the RedisClient surface used by the middleware.

    Has no ``set_nx``, so it exercises the non-atomic CAS lock fallback path
    rather than the real ``set_nx`` (SET NX EX) fast-path.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self.get_calls = 0
        self.set_calls = 0
        self.delete_calls = 0
        self.fail_get = False
        self.fail_set = False

    async def get(self, key: str) -> str | None:
        self.get_calls += 1
        if self.fail_get:
            from redis.exceptions import RedisError

            raise RedisError("boom")
        return self._store.get(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        self.set_calls += 1
        if self.fail_set:
            from redis.exceptions import RedisError

            raise RedisError("boom")
        self._store[key] = value
        return True

    async def delete(self, key: str) -> bool:
        self.delete_calls += 1
        return self._store.pop(key, None) is not None


def _make_app(redis: Any) -> tuple[FastAPI, dict[str, int]]:
    """Build a FastAPI app with IdempotencyMiddleware + a counter-backed POST."""
    app = FastAPI()
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=3600)
    _seed_tenant(app)
    app.state.redis = redis

    counters: dict[str, int] = {"calls": 0}

    @app.post("/runs")
    async def create_run() -> dict[str, Any]:
        counters["calls"] += 1
        return {"run_id": "abc", "call": counters["calls"]}

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"ok": "true"}

    return app, counters


def test_post_without_header_runs_normally() -> None:
    redis = _FakeRedis()
    app, counters = _make_app(redis)
    with TestClient(app) as client:
        r1 = client.post("/runs", json={"x": 1})
        r2 = client.post("/runs", json={"x": 1})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert counters["calls"] == 2
    assert redis.get_calls == 0


def test_post_with_header_replays_cached_response() -> None:
    redis = _FakeRedis()
    app, counters = _make_app(redis)
    headers = {"Idempotency-Key": "k1"}
    with TestClient(app) as client:
        first = client.post("/runs", json={"x": 1}, headers=headers)
        second = client.post("/runs", json={"x": 1}, headers=headers)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert counters["calls"] == 1
    assert second.headers.get("Idempotent-Replay") == "true"


def test_different_body_under_same_key_does_not_collide() -> None:
    redis = _FakeRedis()
    app, counters = _make_app(redis)
    headers = {"Idempotency-Key": "k1"}
    with TestClient(app) as client:
        r1 = client.post("/runs", json={"x": 1}, headers=headers)
        r2 = client.post("/runs", json={"x": 2}, headers=headers)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert counters["calls"] == 2
    assert r2.headers.get("Idempotent-Replay") is None


def test_get_is_not_intercepted() -> None:
    redis = _FakeRedis()
    app, _ = _make_app(redis)
    with TestClient(app) as client:
        resp = client.get("/ping", headers={"Idempotency-Key": "k1"})
    assert resp.status_code == 200
    assert redis.get_calls == 0


def test_falls_open_when_redis_unavailable() -> None:
    app, counters = _make_app(redis=None)
    headers = {"Idempotency-Key": "k1"}
    with TestClient(app) as client:
        r1 = client.post("/runs", json={"x": 1}, headers=headers)
        r2 = client.post("/runs", json={"x": 1}, headers=headers)
    # No cache available → handler runs both times, but neither errors.
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert counters["calls"] == 2


def test_redis_get_error_falls_open_and_does_not_cache() -> None:
    redis = _FakeRedis()
    redis.fail_get = True
    app, counters = _make_app(redis)
    headers = {"Idempotency-Key": "k1"}
    with TestClient(app) as client:
        resp = client.post("/runs", json={"x": 1}, headers=headers)
    assert resp.status_code == 200
    assert counters["calls"] == 1


def test_5xx_responses_are_not_cached() -> None:
    redis = _FakeRedis()
    app = FastAPI()
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=3600)
    _seed_tenant(app)
    app.state.redis = redis

    state = {"hits": 0}

    from starlette.responses import JSONResponse

    @app.post("/explode")
    async def explode() -> JSONResponse:
        state["hits"] += 1
        return JSONResponse({"err": "server"}, status_code=500)

    headers = {"Idempotency-Key": "k1"}
    with TestClient(app) as client:
        first = client.post("/explode", headers=headers)
        second = client.post("/explode", headers=headers)
    assert first.status_code == 500
    assert second.status_code == 500
    assert state["hits"] == 2
    # Lock SETs happen but get released. The 5xx body itself must not be
    # cached — the cache_key must be absent from the store after both calls.
    body_keys = [k for k in redis._store if ":lock" not in k]
    assert body_keys == [], (
        f"5xx response body should never persist in idempotency cache: {body_keys}"
    )


def _make_tenant_app(redis: Any) -> tuple[FastAPI, dict[str, int]]:
    """App that seeds ``request.state.tenant_id`` from an ``X-Tenant`` header.

    The seeding middleware is added AFTER IdempotencyMiddleware so it runs
    OUTER (Starlette LIFO) — tenant is bound before idempotency dispatches,
    mirroring AuthMiddleware setting it in production.
    """
    app = FastAPI()
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=3600)
    _seed_tenant(app)

    app.state.redis = redis
    counters: dict[str, int] = {"calls": 0}

    @app.post("/runs")
    async def create_run() -> dict[str, Any]:
        counters["calls"] += 1
        return {"call": counters["calls"]}

    return app, counters


def test_tenantless_idempotent_post_is_refused_not_bucketed() -> None:
    """No tenant on the request => 400, NOT a shared "anonymous" cache bucket.

    The fallback this replaces made every tenant-less caller hash into ONE key,
    so the first response cached under a given Idempotency-Key was replayed to
    every other tenant-less caller reusing it. This is the only test that
    exercises the no-tenant path, so it is the one that keeps the fallback from
    coming back.
    """
    redis = _FakeRedis()
    app = FastAPI()
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=3600)
    app.state.redis = redis
    counters: dict[str, int] = {"calls": 0}

    @app.post("/runs")
    async def create_run() -> dict[str, Any]:
        counters["calls"] += 1
        return {"call": counters["calls"]}

    with TestClient(app) as client:
        resp = client.post("/runs", json={"x": 1}, headers={"Idempotency-Key": "k1"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "TENANT_REQUIRED"
    # The handler must not have run, and nothing may have been cached.
    assert counters["calls"] == 0
    assert redis.set_calls == 0


def test_non_uuid_tenant_is_refused_rather_than_hashed_verbatim() -> None:
    """A slug is not a tenant; hashing it would recreate the shared bucket."""
    redis = _FakeRedis()
    app, _counters = _make_tenant_app(redis)
    with TestClient(app) as client:
        resp = client.post(
            "/runs",
            json={"x": 1},
            headers={"Idempotency-Key": "k1", "X-Tenant": "n8n-dev"},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "TENANT_REQUIRED"


def test_cache_key_is_tenant_scoped() -> None:
    """Same key+path+body but different tenant must NOT cross-tenant replay."""
    redis = _FakeRedis()
    app, counters = _make_tenant_app(redis)
    headers = {"Idempotency-Key": "k1"}
    with TestClient(app) as client:
        a = client.post("/runs", json={"x": 1}, headers={**headers, "X-Tenant": _TENANT_A})
        b = client.post("/runs", json={"x": 1}, headers={**headers, "X-Tenant": _TENANT_B})
    # tenant-b must get its own execution, not tenant-a's cached response.
    assert a.headers.get("Idempotent-Replay") is None
    assert b.headers.get("Idempotent-Replay") is None
    assert counters["calls"] == 2
    # ...but the SAME tenant replays.
    with TestClient(app) as client:
        a2 = client.post("/runs", json={"x": 1}, headers={**headers, "X-Tenant": _TENANT_A})
    assert a2.headers.get("Idempotent-Replay") == "true"
    assert counters["calls"] == 2  # replayed from cache, handler not re-run


def test_4xx_responses_are_not_cached() -> None:
    """A 4xx is a client error a corrected retry should re-attempt, not have pinned."""
    redis = _FakeRedis()
    app = FastAPI()
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=3600)
    _seed_tenant(app)
    app.state.redis = redis

    state = {"hits": 0}

    from starlette.responses import JSONResponse

    @app.post("/bad")
    async def bad() -> JSONResponse:
        state["hits"] += 1
        return JSONResponse({"err": "client"}, status_code=422)

    headers = {"Idempotency-Key": "k1"}
    with TestClient(app) as client:
        first = client.post("/bad", headers=headers)
        second = client.post("/bad", headers=headers)
    assert first.status_code == 422
    assert second.status_code == 422
    assert state["hits"] == 2
    body_keys = [k for k in redis._store if ":lock" not in k]
    assert body_keys == [], f"4xx response body should never persist: {body_keys}"


def test_replayable_headers_strips_framing() -> None:
    from w2v_common.middleware.idempotency import _replayable_headers

    raw = [
        (b"content-type", b"application/json"),
        (b"Content-Length", b"12"),
        (b"transfer-encoding", b"chunked"),
        (b"connection", b"keep-alive"),
        (b"x-custom", b"v"),
    ]
    assert _replayable_headers(raw) == {"content-type": "application/json", "x-custom": "v"}


def test_lowercase_header_also_works() -> None:
    redis = _FakeRedis()
    app, counters = _make_app(redis)
    with TestClient(app) as client:
        r1 = client.post("/runs", json={"x": 1}, headers={"idempotency-key": "k1"})
        r2 = client.post("/runs", json={"x": 1}, headers={"idempotency-key": "k1"})
    assert r1.json() == r2.json()
    assert counters["calls"] == 1


def test_cache_key_includes_method_and_path() -> None:
    redis = _FakeRedis()
    app = FastAPI()
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=3600)
    _seed_tenant(app)
    app.state.redis = redis

    @app.post("/a")
    async def a_route() -> dict[str, str]:
        return {"r": "a"}

    @app.post("/b")
    async def b_route() -> dict[str, str]:
        return {"r": "b"}

    with TestClient(app) as client:
        ra = client.post("/a", headers={"Idempotency-Key": "shared"})
        rb = client.post("/b", headers={"Idempotency-Key": "shared"})
    assert ra.json() == {"r": "a"}
    assert rb.json() == {"r": "b"}


def test_cached_value_is_json_serialisable() -> None:
    redis = _FakeRedis()
    app, _ = _make_app(redis)
    headers = {"Idempotency-Key": "k1"}
    with TestClient(app) as client:
        client.post("/runs", json={"x": 1}, headers=headers)
    # Every value in the fake store must be valid JSON (so a different
    # process can replay it via the same code path).
    for value in redis._store.values():
        json.loads(value)


def test_oversized_response_persists_marker_and_replays_skeleton() -> None:
    """Responses > 256 KiB cache an ``_oversized`` skeleton so retry is deterministic.

    Without this, the original 256 KiB cap silently fell open: a client
    retrying after a 5xx hit a fresh handler invocation and the side
    effect (a new ``runs`` row) was duplicated.
    """
    redis = _FakeRedis()
    app = FastAPI()
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=3600)
    _seed_tenant(app)
    app.state.redis = redis

    state = {"hits": 0}

    from starlette.responses import Response

    big_body = b"x" * (300 * 1024)

    @app.post("/bulk")
    async def bulk() -> Response:
        state["hits"] += 1
        return Response(content=big_body, status_code=201, media_type="application/octet-stream")

    headers = {"Idempotency-Key": "k1"}
    with TestClient(app) as client:
        first = client.post("/bulk", headers=headers)
        second = client.post("/bulk", headers=headers)

    assert first.status_code == 201
    assert len(first.content) == 300 * 1024
    # Handler ran ONCE — the oversize marker prevents the retry from
    # reaching the route a second time.
    assert state["hits"] == 1
    # Replay surfaces the original status + the oversize sentinel so the
    # client can refetch the resource via its canonical GET.
    assert second.status_code == 201
    assert second.content == b""
    assert second.headers.get("Idempotent-Replay") == "true"
    assert second.headers.get("Idempotent-Replay-Oversized") == "true"


async def test_concurrent_same_key_executes_handler_exactly_once() -> None:
    """Two in-flight POSTs with the same Idempotency-Key must NOT both reach the handler.

    Pre-lock, both concurrent requests passed the cache-miss read, both
    executed the handler, both wrote conflicting cache entries — duplicate
    side effects despite the client opting into exactly-once semantics. The
    SET NX lock fixes this: one of them gains the lock, executes, and the
    other polls the cache key until the result is published.
    """
    redis = _FakeRedis()
    app = FastAPI()
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=3600)
    _seed_tenant(app)
    app.state.redis = redis

    counters = {"calls": 0}

    async def slow_handler() -> dict[str, int]:
        # Yield enough times to interleave with a concurrent request.
        counters["calls"] += 1
        await asyncio.sleep(0.05)
        return {"call": counters["calls"]}

    app.post("/runs")(slow_handler)

    headers = {"Idempotency-Key": "k-concurrent"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r1, r2 = await asyncio.gather(
            client.post("/runs", json={"x": 1}, headers=headers),
            client.post("/runs", json={"x": 1}, headers=headers),
        )

    assert r1.status_code == 200
    assert r2.status_code == 200
    # Exactly one of the two requests reached the handler; the other received
    # the cached replay.
    assert counters["calls"] == 1, (
        f"expected exactly-once handler execution under concurrent keys; got {counters['calls']}"
    )
    bodies = [r1.json(), r2.json()]
    # Both bodies identical (same {"call": 1}).
    assert bodies[0] == bodies[1] == {"call": 1}
    # Exactly one of the two responses carries the replay header.
    replays = sum(1 for r in (r1, r2) for v in [r.headers.get("Idempotent-Replay")] if v == "true")
    assert replays == 1, f"expected one replay marker across the pair; got {replays}"


@pytest.mark.parametrize("method", ["PATCH", "PUT", "DELETE"])
def test_only_post_is_intercepted(method: str) -> None:
    redis = _FakeRedis()
    app = FastAPI()
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=3600)
    _seed_tenant(app)
    app.state.redis = redis

    state = {"hits": 0}

    @app.api_route("/r", methods=["PATCH", "PUT", "DELETE"])
    async def handler() -> dict[str, int]:
        state["hits"] += 1
        return {"hits": state["hits"]}

    headers = {"Idempotency-Key": "k1"}
    with TestClient(app) as client:
        client.request(method, "/r", headers=headers)
        client.request(method, "/r", headers=headers)
    assert state["hits"] == 2
    assert redis.get_calls == 0


# ---------------------------------------------------------------------------
# Real RedisClient (fakeredis-backed) — exercises the SET NX EX fast-path via
# ``set_nx`` plus the serialize/deserialize round-trip the hand fake never
# covered.
# ---------------------------------------------------------------------------


async def test_real_redisclient_replays_via_nx_lock(fakeredis_client) -> None:
    app = FastAPI()
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=3600)
    _seed_tenant(app)
    app.state.redis = fakeredis_client

    counters = {"calls": 0}

    @app.post("/runs")
    async def create_run() -> dict[str, int]:
        counters["calls"] += 1
        return {"call": counters["calls"]}

    headers = {"Idempotency-Key": "k1"}
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        r1 = await client.post("/runs", json={"x": 1}, headers=headers)
        r2 = await client.post("/runs", json={"x": 1}, headers=headers)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json() == r2.json() == {"call": 1}
    assert counters["calls"] == 1
    assert r2.headers.get("Idempotent-Replay") == "true"

    # The body round-tripped through the canonical serializing client, and the
    # NX lock was released (only the idem cache key remains, no :lock key).
    raw = fakeredis_client._ensure_connected()
    keys = sorted(await raw.keys("*"))
    assert keys == [k for k in keys if k.startswith("idem:")]
    assert not any(k.endswith(":lock") for k in keys)
    assert len(keys) == 1


async def test_real_redisclient_concurrent_exactly_once(fakeredis_client) -> None:
    """Concurrent same-key POSTs through the real client hit the handler once.

    Exercises the atomic SET NX EX path (``set_nx``) — the hand
    ``_FakeRedis`` only ever drove the non-atomic CAS fallback.
    """
    app = FastAPI()
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=3600)
    _seed_tenant(app)
    app.state.redis = fakeredis_client

    counters = {"calls": 0}

    async def slow_handler() -> dict[str, int]:
        counters["calls"] += 1
        await asyncio.sleep(0.05)
        return {"call": counters["calls"]}

    app.post("/runs")(slow_handler)

    headers = {"Idempotency-Key": "k-concurrent"}
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        r1, r2 = await asyncio.gather(
            client.post("/runs", json={"x": 1}, headers=headers),
            client.post("/runs", json={"x": 1}, headers=headers),
        )

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert counters["calls"] == 1
    replays = sum(1 for r in (r1, r2) if r.headers.get("Idempotent-Replay") == "true")
    assert replays == 1


# ---------------------------------------------------------------------------
# create_app opt-in gate via IdempotencySettings.
# ---------------------------------------------------------------------------


class _IdemSettings(APISettings):
    """APISettings + an opt-in idempotency block, as a real service composes."""

    idempotency: IdempotencySettings = Field(default_factory=IdempotencySettings)


def _run_post_twice(app: FastAPI) -> None:
    headers = {"Idempotency-Key": "k1"}
    with TestClient(app) as client:
        client.post("/runs", json={"x": 1}, headers=headers)
        client.post("/runs", json={"x": 1}, headers=headers)


async def test_create_app_registers_idempotency_when_enabled(fakeredis_client) -> None:
    settings = _IdemSettings(
        port=8080, host="127.0.0.1", idempotency=IdempotencySettings(enabled=True)
    )
    counters = {"calls": 0}
    router = APIRouter()

    @router.post("/runs")
    async def create_run() -> dict[str, int]:
        counters["calls"] += 1
        return {"call": counters["calls"]}

    app = create_app(settings, routers=[router], auth_enabled=False)
    # auth_enabled=False, so no AuthMiddleware binds a tenant. The middleware
    # now REFUSES a tenant-less idempotent POST rather than hashing every such
    # caller into one shared "anonymous" bucket, so the auth-disabled surface
    # must bind one itself.
    _seed_tenant(app)
    app.state.redis = fakeredis_client

    headers = {"Idempotency-Key": "k1"}
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        r1 = await client.post("/runs", json={"x": 1}, headers=headers)
        r2 = await client.post("/runs", json={"x": 1}, headers=headers)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert counters["calls"] == 1
    assert r2.headers.get("Idempotent-Replay") == "true"


def test_create_app_skips_idempotency_when_disabled(fakeredis_client) -> None:
    settings = _IdemSettings(port=8081, host="127.0.0.1")  # enabled defaults False
    counters = {"calls": 0}
    router = APIRouter()

    @router.post("/runs")
    async def create_run() -> dict[str, int]:
        counters["calls"] += 1
        return {"call": counters["calls"]}

    app = create_app(settings, routers=[router], auth_enabled=False)
    app.state.redis = fakeredis_client

    _run_post_twice(app)
    # Middleware not registered → handler runs on every request.
    assert counters["calls"] == 2


def test_create_app_skips_idempotency_when_setting_absent(fakeredis_client) -> None:
    settings = APISettings(port=8082, host="127.0.0.1")  # no idempotency attr at all
    counters = {"calls": 0}
    router = APIRouter()

    @router.post("/runs")
    async def create_run() -> dict[str, int]:
        counters["calls"] += 1
        return {"call": counters["calls"]}

    app = create_app(settings, routers=[router], auth_enabled=False)
    app.state.redis = fakeredis_client

    _run_post_twice(app)
    assert counters["calls"] == 2


# --------------------------------------------------------------------------- #
# Defect 2 (0.57.0): lossy binary cache + query-string key collision
# --------------------------------------------------------------------------- #

_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe\xfd\x80\x81\x82"


def _make_binary_app(redis: Any) -> FastAPI:
    """App whose POST returns raw binary (the audio/image case)."""
    from fastapi.responses import Response as FastAPIResponse

    app = FastAPI()
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=3600)
    _seed_tenant(app)
    app.state.redis = redis

    @app.post("/render")
    async def render() -> FastAPIResponse:
        return FastAPIResponse(content=_PNG_BYTES, media_type="image/png")

    return app


def test_binary_body_round_trips_through_the_idempotency_cache() -> None:
    """A cached binary response must replay byte-for-byte.

    ``response_body.decode("utf-8", errors="replace")`` substitutes U+FFFD for
    every non-UTF-8 byte, which is irreversible: the replayed audio/image/video
    payload is corrupt.
    """
    redis = _FakeRedis()
    app = _make_binary_app(redis)
    with TestClient(app) as client:
        first = client.post("/render", headers={"Idempotency-Key": "bin-1"})
        second = client.post("/render", headers={"Idempotency-Key": "bin-1"})

    assert first.status_code == 200
    assert first.content == _PNG_BYTES
    assert second.headers.get("Idempotent-Replay") == "true"
    assert second.content == _PNG_BYTES, (
        "binary body was corrupted by the idempotency cache round-trip: "
        f"{second.content!r} != {_PNG_BYTES!r}"
    )


def test_cache_key_includes_the_query_string() -> None:
    """Same path + same key + same body but DIFFERENT query must not collide."""
    redis = _FakeRedis()
    app = FastAPI()
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=3600)
    _seed_tenant(app)
    app.state.redis = redis

    @app.post("/search")
    async def search(q: str) -> dict[str, str]:
        return {"q": q}

    with TestClient(app) as client:
        first = client.post("/search?q=alpha", headers={"Idempotency-Key": "same-key"})
        second = client.post("/search?q=beta", headers={"Idempotency-Key": "same-key"})

    assert first.json() == {"q": "alpha"}
    assert second.json() == {"q": "beta"}, (
        "the ?q=beta request replayed the cached ?q=alpha response — the cache "
        f"key ignores request.url.query (got {second.json()})"
    )


def test_cached_entry_is_not_replayed_across_differing_queries_via_store() -> None:
    """Key-level proof: the two composed cache keys must differ."""
    redis = _FakeRedis()
    app = FastAPI()
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=3600)
    _seed_tenant(app)
    app.state.redis = redis

    @app.post("/search")
    async def search(q: str) -> dict[str, str]:
        return {"q": q}

    with TestClient(app) as client:
        client.post("/search?q=alpha", headers={"Idempotency-Key": "k"})
        client.post("/search?q=beta", headers={"Idempotency-Key": "k"})

    payload_keys = [k for k in redis._store if not k.endswith(":lock")]
    assert len(payload_keys) == 2, (
        f"expected two distinct cache keys for two distinct query strings, got {payload_keys}"
    )


def test_pre_0_57_cache_entry_without_encoding_marker_still_replays() -> None:
    """Backward compatibility: entries written before 0.57.0 carry a plain
    ``body`` string and no ``body_encoding`` marker; they must keep replaying."""
    from w2v_common.middleware.idempotency import _build_replay_response

    legacy = {
        "status": 201,
        "headers": {"content-type": "application/json"},
        "media_type": "application/json",
        "body": '{"run_id":"abc"}',
    }
    response = _build_replay_response(legacy)
    assert response.status_code == 201
    assert response.body == b'{"run_id":"abc"}'
    assert response.headers["Idempotent-Replay"] == "true"


# --------------------------------------------------------------------------- #
# 0.62.0: the non-atomic CAS lock must be visible at LOG_LEVEL=WARNING
# --------------------------------------------------------------------------- #


class _NoSetNxRedis(_FakeRedis):
    """Distinct type so the warn-once dedupe key is unique to this test."""


async def test_cas_fallback_warns_at_warning_level_once_per_backend(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A backend without ``set_nx`` degrades to a NON-ATOMIC lock.

    In production that is a correctness hazard (concurrent duplicate execution),
    and the fleet runs ``LOG_LEVEL=WARNING`` — so it must be logged at WARNING,
    not DEBUG. It is also on the hot path of every idempotent POST, so it must
    be emitted only once per backend type to avoid a log flood.
    """
    import logging

    from w2v_common.middleware.idempotency import (
        IdempotencyMiddleware,
        _cas_fallback_warned,
    )

    _cas_fallback_warned.discard(_NoSetNxRedis.__qualname__)
    # Deliberately off-contract: ``IdempotencyStore`` declares ``set_nx``, so a
    # backend lacking it is exactly the degraded case this warning exists for.
    redis = cast(Any, _NoSetNxRedis())

    with caplog.at_level(logging.WARNING):
        first = await IdempotencyMiddleware._acquire_lock(redis, "idem:lock:one")
        second = await IdempotencyMiddleware._acquire_lock(redis, "idem:lock:two")

    assert first is True
    assert second is True

    records = [
        r
        for r in caplog.records
        if r.name == "w2v_common.middleware.idempotency" and "set_nx unavailable" in r.getMessage()
    ]
    assert len(records) == 1, f"expected exactly one warn-once record, got {records}"
    assert records[0].levelno == logging.WARNING, (
        f"CAS fallback must log at WARNING, got {records[0].levelname}"
    )
    assert "NON-ATOMIC" in records[0].getMessage()
    assert _NoSetNxRedis.__qualname__ in records[0].getMessage()


async def test_set_nx_backend_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """The production path (a backend exposing ``set_nx``) stays silent."""
    import logging

    from w2v_common.middleware.idempotency import IdempotencyMiddleware

    class _SetNxRedis(_FakeRedis):
        async def set_nx(self, key: str, value: str, ex: int | None = None) -> bool:
            return True

    with caplog.at_level(logging.WARNING):
        acquired = await IdempotencyMiddleware._acquire_lock(
            cast(Any, _SetNxRedis()), "idem:lock:x"
        )
        assert acquired is True

    assert not [r for r in caplog.records if "set_nx unavailable" in r.getMessage()]


# ---------------------------------------------------------------------------
# Store outage: fall open FAST, then hold a cool-down (circuit breaker).
#
# Before this, an unreachable Redis cost every keyed request three sequential
# store calls (read, lock, write), each burning the full socket timeout — ~15s
# of a caller's 30s budget — on a request the middleware was going to serve
# un-deduplicated anyway.
# ---------------------------------------------------------------------------


class _Clock:
    """Injectable monotonic clock; only moves when a test (or the store) says."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _OutageStore:
    """Store on the ``RedisClient`` surface whose calls cost ``call_seconds``.

    While ``down`` every call burns the timeout on the injected clock and then
    fails the way the real client does on that method: ``get_strict`` raises,
    ``set_nx`` returns ``None``, ``set``/``delete`` return ``False``.
    """

    def __init__(self, clock: _Clock, call_seconds: float = 5.0) -> None:
        self._clock = clock
        self._call_seconds = call_seconds
        self._data: dict[str, Any] = {}
        self.down = False
        self.calls: list[str] = []
        self.gate: asyncio.Event | None = None

    async def _enter(self, op: str) -> bool:
        """Record the call; return True when it must fail as an outage."""
        self.calls.append(op)
        if self.gate is not None:
            await self.gate.wait()
        if self.down:
            self._clock.now += self._call_seconds
            return True
        return False

    async def get(self, key: str) -> Any:
        raise AssertionError("middleware must read through get_strict when it exists")

    async def get_strict(self, key: str) -> Any:
        if await self._enter("get"):
            from redis.exceptions import TimeoutError as RedisTimeoutError

            raise RedisTimeoutError("timed out")
        return self._data.get(key)

    async def set_nx(self, key: str, value: Any, *, ex: int) -> bool | None:
        if await self._enter("set_nx"):
            return None
        if key in self._data:
            return False
        self._data[key] = value
        return True

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        if await self._enter("set"):
            return False
        self._data[key] = value
        return True

    async def delete(self, key: str) -> bool:
        if await self._enter("delete"):
            return False
        return self._data.pop(key, None) is not None


_COOLDOWN = 10.0


def _make_outage_app(
    store: Any, clock: _Clock, threshold: int = 1
) -> tuple[FastAPI, dict[str, int]]:
    """``threshold=1`` (open on the first failure) keeps the circuit-mechanics
    tests below single-step; the consecutive-failure rule has its own tests."""
    app = FastAPI()
    app.add_middleware(
        IdempotencyMiddleware,
        ttl_seconds=3600,
        outage_cooldown_seconds=_COOLDOWN,
        outage_threshold=threshold,
        clock=clock,
    )
    _seed_tenant(app)
    app.state.redis = store
    counters = {"calls": 0}

    @app.post("/runs")
    async def create_run() -> dict[str, int]:
        counters["calls"] += 1
        return {"call": counters["calls"]}

    return app, counters


def _asgi_client(app: FastAPI) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    )


async def test_plain_cache_miss_still_locks_and_writes() -> None:
    """(d) A MISS is not an outage: full read -> lock -> write -> unlock."""
    clock = _Clock()
    store = _OutageStore(clock)
    app, counters = _make_outage_app(store, clock)
    headers = {"Idempotency-Key": "k1"}
    async with _asgi_client(app) as client:
        r1 = await client.post("/runs", json={"x": 1}, headers=headers)
        assert store.calls == ["get", "set_nx", "set", "delete"]
        r2 = await client.post("/runs", json={"x": 1}, headers=headers)
    assert r1.json() == r2.json() == {"call": 1}
    assert counters["calls"] == 1
    assert r2.headers.get("Idempotent-Replay") == "true"
    assert store.calls == ["get", "set_nx", "set", "delete", "get"]


async def test_outage_costs_one_store_call_then_none_during_cooldown() -> None:
    """(a) one call for the request that discovers the outage, (b) zero after."""
    clock = _Clock()
    store = _OutageStore(clock)
    store.down = True
    app, counters = _make_outage_app(store, clock)
    headers = {"Idempotency-Key": "k1"}
    async with _asgi_client(app) as client:
        started = clock.now
        first = await client.post("/runs", json={"x": 1}, headers=headers)
        assert first.status_code == 200
        # (a) exactly ONE store call — not read + lock + write.
        assert store.calls == ["get"]
        assert clock.now - started == 5.0

        # (b) every request inside the cool-down performs ZERO store calls.
        for i in range(6):
            clock.now += 1.0
            resp = await client.post("/runs", json={"x": i}, headers={"Idempotency-Key": f"n{i}"})
            assert resp.status_code == 200
        assert store.calls == ["get"]
    # Fell open: every request was served, un-deduplicated.
    assert counters["calls"] == 7


async def test_cooldown_expiry_probes_once_and_dedup_recovers() -> None:
    """(c) one probe per window; a successful probe restores replay."""
    clock = _Clock()
    store = _OutageStore(clock)
    store.down = True
    app, counters = _make_outage_app(store, clock)
    headers = {"Idempotency-Key": "k1"}
    async with _asgi_client(app) as client:
        await client.post("/runs", json={"x": 1}, headers=headers)
        assert store.calls == ["get"]

        # Still down after the window: exactly one probe, then closed again.
        clock.now += _COOLDOWN
        await client.post("/runs", json={"x": 1}, headers=headers)
        assert store.calls == ["get", "get"]
        clock.now += 1.0
        await client.post("/runs", json={"x": 1}, headers=headers)
        assert store.calls == ["get", "get"]

        # Redis returns. Inside the window nothing is tried...
        store.down = False
        await client.post("/runs", json={"x": 1}, headers=headers)
        assert store.calls == ["get", "get"]
        assert counters["calls"] == 4

        # ...after it, the single probe succeeds and the SAME request goes on
        # to lock + write, so dedup is back with no warm-up request lost.
        clock.now += _COOLDOWN
        store.calls.clear()
        r1 = await client.post("/runs", json={"x": 1}, headers=headers)
        assert store.calls == ["get", "set_nx", "set", "delete"]
        r2 = await client.post("/runs", json={"x": 1}, headers=headers)
    assert r1.json() == r2.json() == {"call": 5}
    assert counters["calls"] == 5
    assert r2.headers.get("Idempotent-Replay") == "true"


async def test_concurrent_requests_at_cooldown_expiry_send_one_probe() -> None:
    """No thundering probe: N requests arriving together -> one store call."""
    clock = _Clock()
    store = _OutageStore(clock)
    store.down = True
    app, counters = _make_outage_app(store, clock)
    async with _asgi_client(app) as client:
        await client.post("/runs", json={"x": 0}, headers={"Idempotency-Key": "k0"})
        assert store.calls == ["get"]

        clock.now += _COOLDOWN
        store.calls.clear()
        store.gate = asyncio.Event()  # hold the probe in flight
        tasks = [
            asyncio.create_task(
                client.post("/runs", json={"x": i}, headers={"Idempotency-Key": f"c{i}"})
            )
            for i in range(8)
        ]
        # Seven of eight must finish WITHOUT waiting on the store at all.
        done, pending = await asyncio.wait(tasks, timeout=2.0)
        assert len(done) == 7
        assert len(pending) == 1
        assert store.calls == ["get"]
        store.gate.set()
        results = await asyncio.gather(*tasks)
    assert [r.status_code for r in results] == [200] * 8
    assert store.calls == ["get"]
    assert counters["calls"] == 9


async def test_cancelled_probe_does_not_wedge_the_circuit() -> None:
    """A probe that dies without an answer must not block the next probe."""
    clock = _Clock()
    store = _OutageStore(clock)
    store.down = True
    app, _ = _make_outage_app(store, clock)
    async with _asgi_client(app) as client:
        await client.post("/runs", json={"x": 0}, headers={"Idempotency-Key": "k0"})
        clock.now += _COOLDOWN
        store.calls.clear()
        store.gate = asyncio.Event()
        probe = asyncio.create_task(
            client.post("/runs", json={"x": 1}, headers={"Idempotency-Key": "k1"})
        )
        for _ in range(50):
            await asyncio.sleep(0)
            if store.calls:
                break
        assert store.calls == ["get"]
        probe.cancel()
        await asyncio.gather(probe, return_exceptions=True)

        store.gate = None
        store.down = False
        store.calls.clear()
        await client.post("/runs", json={"x": 2}, headers={"Idempotency-Key": "k2"})
    assert store.calls == ["get", "set_nx", "set", "delete"]


async def test_circuit_transitions_are_logged_once_each(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _Clock()
    store = _OutageStore(clock)
    store.down = True
    app, _ = _make_outage_app(store, clock)
    caplog.set_level("WARNING", logger="w2v_common.middleware.idempotency")
    async with _asgi_client(app) as client:
        for i in range(5):
            await client.post("/runs", json={"x": i}, headers={"Idempotency-Key": f"k{i}"})
        clock.now += _COOLDOWN
        await client.post("/runs", json={"x": 9}, headers={"Idempotency-Key": "k9"})  # failed probe
        clock.now += _COOLDOWN
        store.down = False
        for i in range(3):
            await client.post("/runs", json={"x": i}, headers={"Idempotency-Key": f"r{i}"})
    messages = [r.getMessage() for r in caplog.records]
    assert len([m for m in messages if "circuit OPEN" in m]) == 1
    assert len([m for m in messages if "circuit CLOSED" in m]) == 1


async def test_failed_lock_or_write_never_opens_the_circuit() -> None:
    """``set_nx -> None`` / ``set -> False`` carry no reason (the canonical
    client swallows it), so they may be OOM/READONLY on a readable Redis.
    They must not open the circuit, and the write is still attempted — the
    pre-circuit behaviour, which a later retry's replay depends on."""
    clock = _Clock()
    store = _OutageStore(clock)
    app, counters = _make_outage_app(store, clock)

    async def refused_set_nx(key: str, value: Any, *, ex: int) -> bool | None:
        store.calls.append("set_nx")
        return None

    async def refused_set(key: str, value: Any, ttl: int | None = None) -> bool:
        store.calls.append("set")
        return False

    store.set_nx = refused_set_nx  # ty: ignore[invalid-assignment]
    store.set = refused_set  # ty: ignore[invalid-assignment]
    headers = {"Idempotency-Key": "k1"}
    async with _asgi_client(app) as client:
        await client.post("/runs", json={"x": 1}, headers=headers)
        assert store.calls == ["get", "set_nx", "set"]
        # Reads are NOT disabled: the very next request consults the store.
        await client.post("/runs", json={"x": 1}, headers=headers)
    assert store.calls == ["get", "set_nx", "set"] * 2
    assert counters["calls"] == 2


@pytest.mark.parametrize("error_name", ["MaxConnectionsError", "ResponseError", "ReadOnlyError"])
async def test_non_outage_read_errors_do_not_open_the_circuit(error_name: str) -> None:
    """Pool exhaustion and server error replies are not unreachability."""
    import redis.exceptions as redis_exc

    clock = _Clock()
    store = _OutageStore(clock)
    app, counters = _make_outage_app(store, clock)
    real_get_strict = store.get_strict
    failures = {"left": 1}

    async def flaky_get_strict(key: str) -> Any:
        if failures["left"]:
            failures["left"] -= 1
            store.calls.append("get")
            raise getattr(redis_exc, error_name)("not an outage")
        return await real_get_strict(key)

    store.get_strict = flaky_get_strict  # ty: ignore[invalid-assignment]
    headers = {"Idempotency-Key": "k1"}
    async with _asgi_client(app) as client:
        # Falls open for the READ only: lock + write still happen, as on main.
        r1 = await client.post("/runs", json={"x": 1}, headers=headers)
        assert store.calls == ["get", "set_nx", "set", "delete"]
        r2 = await client.post("/runs", json={"x": 1}, headers=headers)
    assert store.calls == ["get", "set_nx", "set", "delete", "get"]
    assert r1.json() == r2.json() == {"call": 1}
    assert counters["calls"] == 1
    assert r2.headers.get("Idempotent-Replay") == "true"


class _ScriptedStore:
    """Healthy in-memory store whose READS follow a per-call script.

    Each read pops one step: ``"fail"`` raises a connection-class error, an
    ``asyncio.Event`` parks the read until set (then succeeds), and an empty
    script means succeed immediately.
    """

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.calls: list[str] = []
        self.read_script: list[Any] = []

    async def get(self, key: str) -> Any:
        raise AssertionError("middleware must read through get_strict when it exists")

    async def get_strict(self, key: str) -> Any:
        self.calls.append("get")
        step = self.read_script.pop(0) if self.read_script else None
        if step == "fail":
            from redis.exceptions import ConnectionError as RedisConnectionError

            raise RedisConnectionError("one transient reset")
        if isinstance(step, asyncio.Event):
            await step.wait()
        return self.data.get(key)

    async def set_nx(self, key: str, value: Any, *, ex: int) -> bool | None:
        self.calls.append("set_nx")
        if key in self.data:
            return False
        self.data[key] = value
        return True

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        self.calls.append("set")
        self.data[key] = value
        return True

    async def delete(self, key: str) -> bool:
        self.calls.append("delete")
        return self.data.pop(key, None) is not None


async def _until(predicate: Any) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never became true")


async def test_lock_holder_writes_and_releases_even_if_circuit_opens_mid_handler() -> None:
    """A paid handler must never lose its cached response to someone else's blip.

    A holds the lock with its handler running; an UNRELATED keyed request sees
    one transient read error (opening the circuit); Redis is healthy after. A
    must still SET its response and DEL its lock, so a retry of A's key replays
    instead of executing the handler a second time.
    """
    clock = _Clock()
    store = _ScriptedStore()
    app = FastAPI()
    app.add_middleware(
        IdempotencyMiddleware,
        ttl_seconds=3600,
        outage_cooldown_seconds=_COOLDOWN,
        outage_threshold=1,
        clock=clock,
    )
    _seed_tenant(app)
    app.state.redis = store
    renders = {"n": 0}
    release = asyncio.Event()

    @app.post("/render")
    async def render() -> dict[str, int]:
        renders["n"] += 1
        await release.wait()
        return {"n": renders["n"]}

    @app.post("/other")
    async def other() -> dict[str, int]:
        return {}

    key_a = {"Idempotency-Key": "A"}
    async with _asgi_client(app) as client:
        task_a = asyncio.create_task(client.post("/render", json={"x": 1}, headers=key_a))
        await _until(lambda: renders["n"] == 1)  # A holds the lock, handler running
        assert store.calls == ["get", "set_nx"]

        store.read_script = ["fail"]
        await client.post("/other", json={}, headers={"Idempotency-Key": "B"})
        assert store.calls == ["get", "set_nx", "get"]  # B opened the circuit

        release.set()
        first = await task_a
        # A wrote its response AND released its lock despite the open circuit.
        assert store.calls == ["get", "set_nx", "get", "set", "delete"]
        assert not any(k.endswith(":lock") for k in store.data)
        assert len(store.data) == 1

        # Retry of A's key once the window has passed: replayed, not re-run.
        clock.now += _COOLDOWN
        retry = await asyncio.wait_for(client.post("/render", json={"x": 1}, headers=key_a), 5)
    assert first.json() == retry.json() == {"n": 1}
    assert retry.headers.get("Idempotent-Replay") == "true"
    assert renders["n"] == 1


async def test_only_the_probe_may_close_the_circuit() -> None:
    """A read that was in flight when the circuit opened succeeds late.

    That stray success is not the probe: the circuit must stay open, so the
    next request inside the cool-down still makes ZERO store calls.
    """
    clock = _Clock()
    store = _ScriptedStore()
    app, counters = _make_outage_app(store, clock)
    parked = asyncio.Event()
    store.read_script = [parked, "fail"]
    async with _asgi_client(app) as client:
        slow = asyncio.create_task(
            client.post("/runs", json={"x": 1}, headers={"Idempotency-Key": "slow"})
        )
        await _until(lambda: store.calls == ["get"])  # admitted while CLOSED, read parked
        await client.post("/runs", json={"x": 2}, headers={"Idempotency-Key": "b"})
        assert store.calls == ["get", "get"]  # circuit is now open

        parked.set()
        await slow  # non-probe success; it was admitted, so it finishes normally
        assert store.calls == ["get", "get", "set_nx", "set", "delete"]

        store.calls.clear()
        await client.post("/runs", json={"x": 3}, headers={"Idempotency-Key": "c"})
        assert store.calls == []  # still open: skipped

        clock.now += _COOLDOWN
        await client.post("/runs", json={"x": 3}, headers={"Idempotency-Key": "c"})
    assert store.calls == ["get", "set_nx", "set", "delete"]
    assert counters["calls"] == 4


async def test_single_blip_never_opens_and_same_key_retry_is_replayed() -> None:
    """DEFAULT threshold. One timeout on an unrelated request, Redis healthy
    after: nobody else is affected and a same-key retry of the in-flight paid
    request is REPLAYED — one handler execution, as on origin/main."""
    clock = _Clock()
    store = _ScriptedStore()
    app = FastAPI()
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=3600, clock=clock)
    _seed_tenant(app)
    app.state.redis = store
    renders = {"n": 0}
    release = asyncio.Event()

    @app.post("/render")
    async def render() -> dict[str, int]:
        renders["n"] += 1
        await release.wait()
        return {"n": renders["n"]}

    @app.post("/other")
    async def other() -> dict[str, int]:
        return {}

    key_k = {"Idempotency-Key": "K"}
    async with _asgi_client(app) as client:
        task_a = asyncio.create_task(client.post("/render", json={"x": 1}, headers=key_k))
        await _until(lambda: renders["n"] == 1)  # A holds the lock, handler running
        store.calls.clear()

        store.read_script = ["fail"]
        await client.post("/other", json={}, headers={"Idempotency-Key": "B"})
        # The blip request behaves as on main: read-as-miss, then lock + write.
        assert store.calls == ["get", "set_nx", "set", "delete"]

        release.set()
        first = await task_a
        store.calls.clear()
        # NO clock advance: the retry lands immediately, inside what would
        # have been the cool-down had the single blip opened the circuit.
        retry = await asyncio.wait_for(client.post("/render", json={"x": 1}, headers=key_k), 5)
        assert store.calls == ["get"]
    assert first.json() == retry.json() == {"n": 1}
    assert retry.headers.get("Idempotent-Replay") == "true"
    assert renders["n"] == 1


async def test_two_consecutive_outage_reads_open_the_circuit() -> None:
    clock = _Clock()
    store = _OutageStore(clock)
    store.down = True
    app, counters = _make_outage_app(store, clock, threshold=2)
    async with _asgi_client(app) as client:
        # 1st: below threshold -> pays its own timeouts exactly as on main.
        await client.post("/runs", json={"x": 1}, headers={"Idempotency-Key": "k1"})
        assert store.calls == ["get", "set_nx", "set"]
        # 2nd: reaches the threshold -> one call, circuit opens.
        store.calls.clear()
        await client.post("/runs", json={"x": 2}, headers={"Idempotency-Key": "k2"})
        assert store.calls == ["get"]
        # ...and nothing after.
        for i in range(4):
            await client.post("/runs", json={"x": i}, headers={"Idempotency-Key": f"n{i}"})
        assert store.calls == ["get"]
    assert counters["calls"] == 6


async def test_a_success_between_two_outage_reads_resets_the_count() -> None:
    clock = _Clock()
    store = _ScriptedStore()
    app, counters = _make_outage_app(store, clock, threshold=2)
    store.read_script = ["fail", None, "fail"]
    async with _asgi_client(app) as client:
        for i in range(3):
            await client.post("/runs", json={"x": i}, headers={"Idempotency-Key": f"k{i}"})
        assert store.calls == ["get", "set_nx", "set", "delete"] * 3
        # Not open: the next request still consults the store.
        store.calls.clear()
        await client.post("/runs", json={"x": 9}, headers={"Idempotency-Key": "k9"})
        assert store.calls == ["get", "set_nx", "set", "delete"]

        # Control: the SAME two failures with no success between them DO open.
        store.read_script = ["fail", "fail"]
        for i in range(2):
            await client.post("/runs", json={"x": i}, headers={"Idempotency-Key": f"c{i}"})
        store.calls.clear()
        await client.post("/runs", json={"x": 9}, headers={"Idempotency-Key": "c9"})
        assert store.calls == []
    assert counters["calls"] == 7


async def test_simultaneous_failing_reads_open_the_circuit_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Six reads in flight when the store dies: one OPEN transition, and only
    the read below the threshold goes on to lock/write."""
    clock = _Clock()
    # Zero-cost calls: eight 5s timeouts would outlast the cool-down itself and
    # turn the final "still open" request into a legitimate probe.
    store = _OutageStore(clock, call_seconds=0.0)
    store.down = True
    store.gate = asyncio.Event()
    app, counters = _make_outage_app(store, clock, threshold=2)
    caplog.set_level("WARNING", logger="w2v_common.middleware.idempotency")
    async with _asgi_client(app) as client:
        tasks = [
            asyncio.create_task(
                client.post("/runs", json={"x": i}, headers={"Idempotency-Key": f"s{i}"})
            )
            for i in range(6)
        ]
        await _until(lambda: store.calls == ["get"] * 6)
        store.gate.set()
        results = await asyncio.gather(*tasks)
        assert [r.status_code for r in results] == [200] * 6
        assert sorted(store.calls) == ["get"] * 6 + ["set", "set_nx"]

        store.calls.clear()
        await client.post("/runs", json={"x": 9}, headers={"Idempotency-Key": "after"})
        assert store.calls == []
    assert counters["calls"] == 7
    assert len([r for r in caplog.records if "circuit OPEN" in r.getMessage()]) == 1


def test_threshold_default_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from w2v_common.middleware.idempotency import DEFAULT_OUTAGE_THRESHOLD

    app = FastAPI()
    monkeypatch.delenv("IDEMPOTENCY_OUTAGE_THRESHOLD", raising=False)
    assert DEFAULT_OUTAGE_THRESHOLD == 2
    assert IdempotencyMiddleware(app).outage_threshold == 2
    monkeypatch.setenv("IDEMPOTENCY_OUTAGE_THRESHOLD", "5")
    assert IdempotencyMiddleware(app).outage_threshold == 5
    assert IdempotencyMiddleware(app, outage_threshold=3).outage_threshold == 3
    for bad in (0, -1):
        with pytest.raises(ValueError, match="IDEMPOTENCY_OUTAGE_THRESHOLD"):
            IdempotencyMiddleware(app, outage_threshold=bad)
    monkeypatch.setenv("IDEMPOTENCY_OUTAGE_THRESHOLD", "0")
    with pytest.raises(ValueError, match="IDEMPOTENCY_OUTAGE_THRESHOLD"):
        IdempotencyMiddleware(app)


def test_mock_store_is_read_through_get_not_get_strict() -> None:
    """Consumer suites stub ``store.get.return_value`` on an ``AsyncMock``.

    A Mock answers ``getattr`` for ANY name, so detecting ``get_strict`` on the
    instance would silently bypass that stub and "miss" every time.
    """
    from unittest.mock import AsyncMock

    cached = {"status": 200, "headers": {}, "media_type": "application/json", "body": '{"c": 1}'}
    store = AsyncMock()
    store.get.return_value = json.dumps(cached)
    app, counters = _make_app(store)
    with TestClient(app) as client:
        resp = client.post("/runs", json={"x": 1}, headers={"Idempotency-Key": "k1"})
    assert resp.json() == {"c": 1}
    assert resp.headers.get("Idempotent-Replay") == "true"
    assert counters["calls"] == 0
    store.get.assert_awaited_once()
    store.get_strict.assert_not_called()


async def test_real_redisclient_outage_is_one_call_not_three(fakeredis_client) -> None:
    """End-to-end on the canonical client, whose ``get`` swallows outages."""
    from unittest.mock import AsyncMock

    from redis.exceptions import TimeoutError as RedisTimeoutError

    clock = _Clock()
    app, counters = _make_outage_app(fakeredis_client, clock)
    raw = fakeredis_client._ensure_connected()
    real_get, real_set = raw.get, raw.set
    down_get = AsyncMock(side_effect=RedisTimeoutError("timed out"))
    down_set = AsyncMock(side_effect=RedisTimeoutError("timed out"))
    raw.get, raw.set, raw.setex = down_get, down_set, down_set
    headers = {"Idempotency-Key": "k1"}
    try:
        async with _asgi_client(app) as client:
            for _ in range(4):
                assert (
                    await client.post("/runs", json={"x": 1}, headers=headers)
                ).status_code == 200
            assert down_get.await_count == 1
            assert down_set.await_count == 0

            raw.get, raw.set = real_get, real_set
            del raw.setex
            clock.now += _COOLDOWN
            r1 = await client.post("/runs", json={"x": 1}, headers=headers)
            r2 = await client.post("/runs", json={"x": 1}, headers=headers)
    finally:
        raw.get, raw.set = real_get, real_set
    assert counters["calls"] == 5
    assert r1.json() == r2.json() == {"call": 5}
    assert r2.headers.get("Idempotent-Replay") == "true"


def test_cooldown_default_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from w2v_common.middleware.idempotency import DEFAULT_OUTAGE_COOLDOWN_SECONDS

    app = FastAPI()
    monkeypatch.delenv("IDEMPOTENCY_OUTAGE_COOLDOWN_SECONDS", raising=False)
    assert DEFAULT_OUTAGE_COOLDOWN_SECONDS == 10.0
    assert IdempotencyMiddleware(app).outage_cooldown_seconds == 10.0
    monkeypatch.setenv("IDEMPOTENCY_OUTAGE_COOLDOWN_SECONDS", "2.5")
    assert IdempotencyMiddleware(app).outage_cooldown_seconds == 2.5
    # An explicit argument beats the environment.
    assert IdempotencyMiddleware(app, outage_cooldown_seconds=7).outage_cooldown_seconds == 7.0
    with pytest.raises(ValueError, match="IDEMPOTENCY_OUTAGE_COOLDOWN_SECONDS"):
        IdempotencyMiddleware(app, outage_cooldown_seconds=-1)
