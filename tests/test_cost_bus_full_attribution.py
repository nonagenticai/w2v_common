"""0.101.0: full attribution on cost events (cell plan G5).

Three additions, each pinned against billing-service's contract at origin/main:

* ``actor_id`` kwarg — WHO ACTED (verified human ``sub``), ``""`` = no person
  acted. billing ``cost_flusher`` reads top-level ``actor_id`` with
  ``_parse_uuid`` and NULLs anything malformed, so a non-UUID is refused here.
* ``event_key`` kwarg — a caller-derived deterministic key that survives a
  process restart; billing honours a producer key verbatim and dedupes on it
  globally (``processed_cost_event_keys``). uuid7 stays the fallback.
* ``METERING_REQUIRE_ATTRIBUTION=true`` now also demands ``funding`` and
  canonical-UUID ids. Default (flag unset) behaviour is unchanged.
"""

from __future__ import annotations

import uuid
from typing import Any, cast
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from prometheus_client import REGISTRY

from w2v_common.auth.identity_context import clear_identity_context, set_identity_context
from w2v_common.metering import (
    FUNDING_FIELD,
    InvalidActorIdError,
    InvalidEventKeyError,
    MissingAttributionError,
    cost_event_key,
    emit_cost_event,
    resolve_stream,
    schedule_cost_event,
)

_TENANT = uuid.UUID("0195f48b-d000-7000-8000-0000000000aa")
_HUMAN = "0195f48b-d000-7000-8000-0000000000bb"
_BOUND = "0195f48b-d000-7000-8000-0000000000cc"
_SERVICE_KEY = "w2v-g5-test"  # pragma: allowlist secret -- routing slug


@pytest.fixture
async def redis_conn():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("METERING_REQUIRE_ATTRIBUTION", raising=False)
    monkeypatch.delenv("SERVICE_ACTOR_ID", raising=False)
    clear_identity_context()
    yield
    clear_identity_context()


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        service_key=_SERVICE_KEY,
        cost_source="w2v-g5-test",
        tenant_id=_TENANT,
        provider="openai",
        model="gpt-x",
        input_tokens=1,
        output_tokens=2,
        latency_ms=3,
    )
    base.update(overrides)
    return base


def _full(**overrides: Any) -> dict[str, Any]:
    full: dict[str, Any] = dict(
        license_id=str(uuid.uuid7()),
        subject_id=str(uuid.uuid7()),
        app_id=str(uuid.uuid7()),
        funding="platform_external",
    )
    full.update(overrides)
    return _kwargs(**full)


async def _only_event(conn: fakeredis.aioredis.FakeRedis) -> dict[str, str]:
    entries = await conn.xrange(resolve_stream())
    assert len(entries) == 1
    return entries[0][1]


def set_actor_context(actor: str) -> None:
    set_identity_context(_TENANT, None, None, actor_id=actor)


def _metric(name: str, labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


# --- actor_id ---------------------------------------------------------------


async def test_explicit_actor_is_emitted_top_level_and_beats_the_bound_one(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    set_actor_context(_BOUND)
    await emit_cost_event(redis_conn, **_kwargs(actor_id=_HUMAN.upper()))
    assert (await _only_event(redis_conn))["actor_id"] == _HUMAN


async def test_empty_actor_means_no_person_acted_and_suppresses_the_ambient_one(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_actor_context(_BOUND)
    monkeypatch.setenv("SERVICE_ACTOR_ID", str(uuid.uuid7()))
    await emit_cost_event(redis_conn, **_kwargs(actor_id=""))
    assert "actor_id" not in await _only_event(redis_conn)


async def test_omitted_actor_still_comes_from_the_identity_context(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    set_actor_context(_BOUND)
    await emit_cost_event(redis_conn, **_kwargs())
    assert (await _only_event(redis_conn))["actor_id"] == _BOUND


@pytest.mark.parametrize(
    "bad",
    ["system", "agent:planner", "00000000-0000-0000-0000-000000000000", "{" + _HUMAN + "}"],
)
async def test_a_non_uuid_actor_is_refused_before_anything_is_sent(
    redis_conn: fakeredis.aioredis.FakeRedis, bad: str
) -> None:
    with pytest.raises(InvalidActorIdError):
        await emit_cost_event(redis_conn, **_kwargs(actor_id=bad))
    assert await redis_conn.xlen(resolve_stream()) == 0


def test_a_non_uuid_actor_raises_synchronously_from_schedule() -> None:
    with pytest.raises(InvalidActorIdError):
        schedule_cost_event(cast(Any, AsyncMock()), **_kwargs(actor_id="system"))


async def test_actor_given_twice_is_ambiguous_and_refused(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    with pytest.raises(InvalidActorIdError):
        await emit_cost_event(
            redis_conn, **_kwargs(actor_id=_HUMAN, metadata_extra={"actor_id": _BOUND})
        )


# --- event_key --------------------------------------------------------------


def test_cost_event_key_is_deterministic_and_part_sensitive() -> None:
    a = cost_event_key(_SERVICE_KEY, "job-1", "narrate", 1)
    assert a == cost_event_key(_SERVICE_KEY, "job-1", "narrate", 1)
    # a new attempt is a new physical call -> a new key
    assert a != cost_event_key(_SERVICE_KEY, "job-1", "narrate", 2)
    # the separator cannot be forged by moving it between parts
    assert cost_event_key("a|b", "c") != cost_event_key("a", "b|c")
    assert len(a) <= 64
    assert uuid.UUID(a).version == 5


@pytest.mark.parametrize("parts", [(), ("job", ""), ("job", None), ("  ",)])
def test_cost_event_key_refuses_empty_parts(parts: tuple[object, ...]) -> None:
    with pytest.raises(InvalidEventKeyError):
        cost_event_key(*parts)


async def test_caller_key_is_used_verbatim_and_counted(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    key = cost_event_key(_SERVICE_KEY, "job-7", "llm", 1)
    labels = {"service_key": _SERVICE_KEY, "source": "caller"}
    before = _metric("w2v_cost_event_key_source_total", labels)
    await emit_cost_event(redis_conn, **_kwargs(event_key=key))
    assert (await _only_event(redis_conn))["event_key"] == key
    assert _metric("w2v_cost_event_key_source_total", labels) - before == 1.0


async def test_same_caller_key_survives_a_restart_where_uuid7_does_not(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    """Two emits of ONE physical call (e.g. a job step redelivered after a crash)."""
    key = cost_event_key(_SERVICE_KEY, "job-9", "tts", 1)
    await emit_cost_event(redis_conn, **_kwargs(event_key=key))
    await emit_cost_event(redis_conn, **_kwargs(event_key=key))
    await emit_cost_event(redis_conn, **_kwargs())
    await emit_cost_event(redis_conn, **_kwargs())
    keys = [f["event_key"] for _id, f in await redis_conn.xrange(resolve_stream())]
    assert keys[0] == keys[1] == key
    assert keys[2] != keys[3]
    assert uuid.UUID(keys[2]).version == 7


async def test_omitted_key_is_counted_as_minted(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    labels = {"service_key": _SERVICE_KEY, "source": "minted"}
    before = _metric("w2v_cost_event_key_source_total", labels)
    await emit_cost_event(redis_conn, **_kwargs())
    assert _metric("w2v_cost_event_key_source_total", labels) - before == 1.0


@pytest.mark.parametrize("bad", ["", " key", "has space", "unknown", "NULL", "x" * 65, "kéy"])
async def test_an_unusable_key_is_refused_before_anything_is_sent(
    redis_conn: fakeredis.aioredis.FakeRedis, bad: str
) -> None:
    with pytest.raises(InvalidEventKeyError):
        await emit_cost_event(redis_conn, **_kwargs(event_key=bad))
    assert await redis_conn.xlen(resolve_stream()) == 0


def test_an_unusable_key_raises_synchronously_from_schedule() -> None:
    with pytest.raises(InvalidEventKeyError):
        schedule_cost_event(cast(Any, AsyncMock()), **_kwargs(event_key="unknown"))


# --- METERING_REQUIRE_ATTRIBUTION: funding + canonical ids ------------------


async def test_flag_on_requires_funding(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    kwargs = _full()
    del kwargs["funding"]
    with pytest.raises(MissingAttributionError, match="funding"):
        await emit_cost_event(redis_conn, **kwargs)
    assert await redis_conn.xlen(resolve_stream()) == 0


def test_flag_on_requires_funding_synchronously_from_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    kwargs = _full()
    del kwargs["funding"]
    with pytest.raises(MissingAttributionError, match="funding"):
        schedule_cost_event(cast(Any, AsyncMock()), **kwargs)


async def test_flag_on_requires_a_licence(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    with pytest.raises(MissingAttributionError, match="license_id"):
        await emit_cost_event(redis_conn, **_full(license_id=""))


async def test_flag_on_accepts_an_app_key_for_app_id(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """billing ``split_asserted_app`` resolves a non-uuid app_id as the app KEY."""
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    await emit_cost_event(redis_conn, **_full(app_id="w2v"))
    fields = await _only_event(redis_conn)
    assert fields["app_id"] == "w2v"


@pytest.mark.parametrize("app_id", ["W2V", "w2v app", "app-not-a-key!", "-w2v"])
async def test_flag_on_refuses_a_malformed_app_key(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch, app_id: str
) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    with pytest.raises(MissingAttributionError, match="not a canonical UUID: app_id"):
        await emit_cost_event(redis_conn, **_full(app_id=app_id))


@pytest.mark.parametrize("field", ["license_id", "subject_id", "app_id"])
async def test_flag_on_refuses_a_non_uuid_id(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """billing ``_parse_uuid`` stores these as NULL: present-but-garbage is unattributed."""
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    with pytest.raises(MissingAttributionError, match=f"not a canonical UUID: {field}"):
        await emit_cost_event(redis_conn, **_full(**{field: "Lic not a uuid"}))


async def test_flag_on_refuses_a_non_uuid_metadata_actor(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    with pytest.raises(MissingAttributionError, match="actor_id"):
        await emit_cost_event(redis_conn, **_full(metadata_extra={"actor_id": "system"}))


async def test_flag_on_full_attribution_is_emitted(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    key = cost_event_key(_SERVICE_KEY, "job-1", "s", 1)
    await emit_cost_event(redis_conn, **_full(actor_id=_HUMAN, event_key=key))
    fields = await _only_event(redis_conn)
    assert fields["funding"] == "platform_external"
    assert fields["actor_id"] == _HUMAN
    assert fields["event_key"] == key


async def test_flag_off_is_unchanged_funding_miss_is_counted_not_raised(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    """The release must be inert for consumers: no raise, the row still lands."""
    before = _metric("w2v_cost_event_unattributed_total", {"field": FUNDING_FIELD})
    kwargs = _full(license_id="lic-not-a-uuid")
    del kwargs["funding"]
    await emit_cost_event(redis_conn, **kwargs)
    assert await redis_conn.xlen(resolve_stream()) == 1
    after = _metric("w2v_cost_event_unattributed_total", {"field": FUNDING_FIELD})
    assert after - before == 1.0


def test_funding_series_exists_from_import() -> None:
    assert (
        REGISTRY.get_sample_value("w2v_cost_event_unattributed_total", {"field": FUNDING_FIELD})
        is not None
    )


# --- 0.104.0: strict mode makes a disarmed metering a BOOT failure -------------


async def test_strict_refuses_boot_without_metering_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from w2v_common.metering import MeteringDisarmedError, init_metering_redis

    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    with pytest.raises(MeteringDisarmedError, match="METERING_REDIS_URL is not set"):
        await init_metering_redis("")


async def test_strict_refuses_boot_when_metering_redis_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import redis.asyncio as aioredis

    from w2v_common.metering import MeteringDisarmedError, init_metering_redis

    class _Down:
        async def ping(self) -> bool:
            raise ConnectionError("down")

    monkeypatch.setattr(aioredis, "from_url", lambda url, **kw: _Down())
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    with pytest.raises(MeteringDisarmedError, match="unreachable") as excinfo:
        await init_metering_redis(
            "redis://:hunter2@metering:6379/0"
        )  # pragma: allowlist secret -- fake DSN
    assert "hunter2" not in str(excinfo.value)


async def test_flag_off_disarmed_metering_still_boots(monkeypatch: pytest.MonkeyPatch) -> None:
    from w2v_common.metering import init_metering_redis

    monkeypatch.delenv("METERING_REQUIRE_ATTRIBUTION", raising=False)
    assert await init_metering_redis("") is None
