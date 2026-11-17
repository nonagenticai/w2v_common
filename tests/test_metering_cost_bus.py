"""Tests for the shared cost-bus emitter (w2v_common.metering)."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from w2v_common.auth.identity_context import (
    clear_identity_context,
    set_identity_context,
    tenant_level_project_id,
)
from w2v_common.metering import (
    ATTRIBUTION_FIELDS,
    CANONICAL_COST_STREAM,
    MissingAttributionError,
    close_metering_redis,
    cost_bus,
    drain_pending,
    emit_cost_event,
    resolve_stream,
    schedule_cost_event,
)
from w2v_common.tenant import InvalidTenantIdError, MissingTenantError


@pytest.fixture
async def redis_conn():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


# A real registered-tenant-shaped UUID. Tenant ids are ``uuid.UUID`` as of
# 0.67.0 — a slug like "tenant-099" is rejected, not coerced.
_TENANT = uuid.UUID("0195f48b-d000-7000-8000-0000000000aa")


def _common_kwargs(**overrides: Any) -> dict[str, Any]:
    base = dict(
        service_key="w2v-test",  # pragma: allowlist secret -- routing slug, not a credential
        cost_source="w2v-test",
        tenant_id=_TENANT,
        provider="deepseek",
        model="deepseek-chat",
        input_tokens=10,
        output_tokens=20,
        latency_ms=42,
    )
    base.update(overrides)
    return base


# --- resolve_stream -------------------------------------------------------


def test_resolve_stream_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("METERING_STREAM", raising=False)
    assert resolve_stream() == CANONICAL_COST_STREAM == "billing:costs"


def test_resolve_stream_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METERING_STREAM", "custom:bus")
    assert resolve_stream() == "custom:bus"


# --- emit_cost_event ------------------------------------------------------


async def test_emit_writes_to_resolved_stream(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("METERING_STREAM", raising=False)
    await emit_cost_event(redis_conn, **_common_kwargs())
    assert await redis_conn.xlen(CANONICAL_COST_STREAM) == 1
    entries = await redis_conn.xrange(CANONICAL_COST_STREAM)
    fields = entries[0][1]
    assert fields["tenant_id"] == str(_TENANT)
    assert fields["service_key"] == "w2v-test"  # pragma: allowlist secret -- routing slug
    assert fields["cost_source"] == "w2v-test"
    assert fields["served_model"] == "deepseek-chat"  # defaults to model
    assert fields["cost_usd"] == "0.00000000"
    assert json.loads(fields["usage_extra"]) == {}


async def test_emit_explicit_stream_overrides_env(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("METERING_STREAM", "env:bus")
    await emit_cost_event(redis_conn, stream="explicit:bus", **_common_kwargs())
    assert await redis_conn.xlen("explicit:bus") == 1
    assert await redis_conn.xlen("env:bus") == 0


@pytest.mark.parametrize(
    ("bad_tenant", "expected"),
    [
        ("", MissingTenantError),
        ("   ", MissingTenantError),
        (None, MissingTenantError),
        # The nil UUID is no longer special-cased into existence by this module,
        # but a caller that hardcodes it still gets a real tenant-shaped value —
        # so the guard that matters is the SLUG one below plus the absence cases.
        ("tenant-099", InvalidTenantIdError),
        ("anonymous", InvalidTenantIdError),
        ("0195f48b-d000-7000-8000", InvalidTenantIdError),
    ],
)
async def test_emit_refuses_unattributable_tenant(
    redis_conn: fakeredis.aioredis.FakeRedis,
    bad_tenant: object,
    expected: type[Exception],
) -> None:
    """No tenant => raise, never substitute.

    This is the regression test for the nil-UUID sentinel: the emitter used to
    silently rewrite a falsy tenant to 00000000-... , a VALID uuid that passed
    every type check and dirty-row audit while mis-attributing cost. It must now
    fail loud AND write nothing.
    """
    with pytest.raises(expected):
        await emit_cost_event(redis_conn, **_common_kwargs(tenant_id=bad_tenant))
    assert await redis_conn.xlen(resolve_stream()) == 0


async def test_schedule_refuses_unattributable_tenant_synchronously(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    """schedule_ is fire-and-forget, so the tenant check must run in OUR frame.

    Raised inside the spawned task it would surface only as an "exception never
    retrieved" warning — i.e. still silent.
    """
    with pytest.raises(MissingTenantError):
        schedule_cost_event(redis_conn, **_common_kwargs(tenant_id=""))
    await asyncio.sleep(0)
    assert await redis_conn.xlen(resolve_stream()) == 0


async def test_emit_accepts_uuid_shaped_string(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    """Consumers may still pass a UUID-shaped str; it is parsed, not stringified."""
    await emit_cost_event(redis_conn, **_common_kwargs(tenant_id=str(_TENANT).upper()))
    entries = await redis_conn.xrange(resolve_stream())
    assert entries[0][1]["tenant_id"] == str(_TENANT)


async def test_emit_never_raises_on_redis_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cost_bus, "_DEFAULT_EMIT_RETRY_BACKOFFS", (0.0, 0.0))
    broken = AsyncMock()
    broken.xadd.side_effect = RuntimeError("redis down")
    # Must not propagate — metering is best-effort. The emitter only ever calls
    # ``xadd``, so a duck-typed double stands in for the Redis client; cast
    # because the static signature (deliberately) demands the real class.
    await emit_cost_event(cast(Any, broken), **_common_kwargs())
    # ``assert_awaited_once`` here used to encode the DEFECT: a bare
    # ``RuntimeError`` was not in the retryable set, so the emit gave up after
    # one attempt and dropped the event. It is now retried to exhaustion (2
    # backoffs + the final attempt) and only then swallowed.
    assert broken.xadd.await_count == 3


# --- schedule_cost_event (GC-safe) ---------------------------------------


async def test_schedule_retains_task_and_emits(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    assert not cost_bus._background_tasks
    schedule_cost_event(redis_conn, **_common_kwargs())
    # The task is retained (strong ref) so it cannot be GC'd before XADD.
    assert len(cost_bus._background_tasks) == 1
    # Let the background task run to completion.
    await asyncio.gather(*list(cost_bus._background_tasks))
    assert await redis_conn.xlen(resolve_stream()) == 1
    # Done callback discards the reference.
    assert not cost_bus._background_tasks


# --- FIX B1: bounded producer-side retry ---------------------------------


class _FlakyRedis:
    """Wraps fakeredis, failing the first ``fail_times`` xadd calls.

    Proves the retry actually delivers (real XADD lands) rather than just
    asserting the mock was re-called.
    """

    def __init__(self, inner: fakeredis.aioredis.FakeRedis, fail_times: int) -> None:
        self._inner = inner
        self._remaining_failures = fail_times
        self.xadd_calls = 0

    async def xadd(self, *args: Any, **kwargs: Any) -> Any:
        self.xadd_calls += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RedisConnectionError("billing-redis rolling")
        return await self._inner.xadd(*args, **kwargs)


async def test_emit_retries_then_delivers(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Zero backoff keeps the test fast; one retry slot is enough for one failure.
    monkeypatch.setattr(cost_bus, "_DEFAULT_EMIT_RETRY_BACKOFFS", (0.0, 0.0))
    flaky = _FlakyRedis(redis_conn, fail_times=1)
    # Duck-typed xadd-only double stands in for the Redis client (see above).
    await emit_cost_event(cast(Any, flaky), **_common_kwargs())
    assert flaky.xadd_calls == 2  # first failed, retry succeeded
    # The event was actually delivered to the stream.
    assert await redis_conn.xlen(resolve_stream()) == 1


async def test_emit_always_fails_holds_never_raises_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cost_bus, "_DEFAULT_EMIT_RETRY_BACKOFFS", (0.0, 0.0))
    always = AsyncMock()
    always.xadd.side_effect = RedisConnectionError("billing-redis down")
    # Must not raise after the bounded attempts (1 initial + 2 retries).
    await emit_cost_event(cast(Any, always), **_common_kwargs())
    assert always.xadd.await_count == 3


# --- FIX B2: shutdown-drain ----------------------------------------------


class _SlowRedis:
    """Wraps fakeredis with a small delay before each xadd, so emits are still
    in-flight when ``drain_pending`` is called."""

    def __init__(self, inner: fakeredis.aioredis.FakeRedis, delay: float) -> None:
        self._inner = inner
        self._delay = delay

    async def xadd(self, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(self._delay)
        return await self._inner.xadd(*args, **kwargs)


async def test_drain_pending_awaits_inflight_emits(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    assert not cost_bus._background_tasks
    slow = _SlowRedis(redis_conn, delay=0.05)
    n = 5
    for _ in range(n):
        schedule_cost_event(cast(Any, slow), **_common_kwargs())
    # All emits are scheduled but still in-flight (slow xadd).
    assert len(cost_bus._background_tasks) == n
    await drain_pending(timeout=5.0)
    # Every XADD landed — none dropped.
    assert await redis_conn.xlen(resolve_stream()) == n
    assert not cost_bus._background_tasks


async def test_drain_pending_no_pending_returns_immediately() -> None:
    assert not cost_bus._background_tasks
    # Must return at once and never raise with nothing pending.
    await drain_pending()
    await drain_pending(timeout=0.0)


async def test_close_metering_redis_drains_before_close(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    assert not cost_bus._background_tasks
    slow = _SlowRedis(redis_conn, delay=0.05)
    n = 3
    for _ in range(n):
        schedule_cost_event(cast(Any, slow), **_common_kwargs())
    assert len(cost_bus._background_tasks) == n
    closer = AsyncMock()
    await close_metering_redis(cast(Any, closer))
    # Drain ran first: all emits landed before the client was closed.
    assert await redis_conn.xlen(resolve_stream()) == n
    closer.aclose.assert_awaited_once()
    assert not cost_bus._background_tasks


async def test_close_metering_redis_none_still_safe() -> None:
    # No client, no pending tasks — must be a no-op that never raises.
    await close_metering_redis(None)


# --- w2v_common#182: the no-running-loop drop must be impossible to miss ----
#
# ⚠️ These replace `test_schedule_no_running_loop_closes_coro_no_raise`, which
# asserted ONLY "no raise, no leaked coroutine". That test passed on the defect:
# it pinned the SILENCE rather than catching it. The drop is still a drop — this
# module has no durable spool to defer an XADD to — but it is now counted on the
# same counter as `ok`/`failed`, reproduced in a WARNING, reported to the caller
# as `False`, and optionally escalated to a raise.


def _dropped_no_loop_count() -> float:
    """Reads the SAME counter/labels `ok` and `failed` are read from.

    Deliberately not a new metric: "is this service dropping cost events?" is a
    ratio against that service's own `ok`, and a numerator in a different series
    cannot be divided by a denominator in another.
    """
    return _cost_event_count(cost_bus.DROPPED_NO_LOOP)


def _no_loop_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and record.getMessage().startswith("cost_event_dropped_no_running_loop ")
    ]


def test_schedule_no_running_loop_counts_the_drop_and_returns_false(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ARM B (no running loop): the drop is LOUD.

    RED on origin/main: the `except RuntimeError` branch called `coro.close()`
    and returned `None` after a `logger.debug` — no counter, no WARNING, and a
    return value indistinguishable from the success path.
    """
    before = _dropped_no_loop_count()
    before_ok = _cost_event_count("ok")
    assert not cost_bus._background_tasks

    # A plain (sync) test => `asyncio.get_running_loop()` raises RuntimeError.
    with caplog.at_level(logging.WARNING, logger=cost_bus.logger.name):
        handed_off = schedule_cost_event(cast(Any, AsyncMock()), **_common_kwargs())

    # 1. The positive handoff marker says the emit path was never entered.
    assert handed_off is False
    # 2. The drop is counted, on the same counter/service_key as ok|failed.
    assert _dropped_no_loop_count() - before == 1.0
    # 3. A drop is not a success: `ok` must not have moved.
    assert _cost_event_count("ok") - before_ok == 0.0
    # 4. Exactly one WARNING, not a debug line nobody has enabled.
    assert len(_no_loop_warnings(caplog)) == 1
    # 5. Still never-raises by default, and still spawns nothing.
    assert not cost_bus._background_tasks


def test_schedule_no_running_loop_warning_reproduces_the_whole_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The WARNING IS the durable record — billing must be able to rebuild the row.

    There is no replay. If a field billing needs is missing from this line, the
    revenue evidence is gone, so every one of them is pinned here individually.
    """
    with caplog.at_level(logging.WARNING, logger=cost_bus.logger.name):
        schedule_cost_event(
            cast(Any, AsyncMock()),
            **_common_kwargs(
                license_id="lic-7",
                subject_id="sub-7",
                app_id="app-7",
                served_model="deepseek-chat-0324",
                cost_usd=0.001234,
                provider_request_id="req-abc-123",
            ),
        )

    records = _no_loop_warnings(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    for fragment in (
        "service_key=w2v-test",  # pragma: allowlist secret -- routing slug
        "cost_source=w2v-test",  # pragma: allowlist secret -- routing slug
        f"tenant_id={_TENANT}",
        "license_id=lic-7",
        "subject_id=sub-7",
        "app_id=app-7",
        "provider=deepseek",
        "model=deepseek-chat",
        "served_model=deepseek-chat-0324",
        "input_tokens=10",
        "output_tokens=20",
        "latency_ms=42",
        "cost_usd=0.00123400",
        "provider_request_id=req-abc-123",
    ):
        assert fragment in message, f"{fragment!r} missing from {message!r}"


def test_schedule_no_running_loop_never_builds_the_coroutine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loop is resolved BEFORE the coroutine exists.

    The old shape built the coroutine first and then had to `coro.close()` it,
    so "orphaned coroutine" was a class that had to be handled. Resolving the
    loop first makes it unreachable — which is why the replaced test's
    `cr_frame is None` assertion no longer has a subject.
    """
    calls: list[object] = []

    def _never(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - must not run
        calls.append(args)
        raise AssertionError("emit_cost_event must not be invoked with no running loop")

    monkeypatch.setattr(cost_bus, "emit_cost_event", _never)

    assert schedule_cost_event(cast(Any, AsyncMock()), **_common_kwargs()) is False
    assert calls == []


def test_require_loop_is_off_by_default_and_reads_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ⚠️ The LITERAL name, not `cost_bus._REQUIRE_LOOP_ENV`. Reading the name out
    # of the module under test makes a rename invisible — and the name is the
    # contract: a consumer that sets `METERING_REQUIRE_LOOP=true` in its CI env
    # would silently stop escalating.
    assert cost_bus._REQUIRE_LOOP_ENV == "METERING_REQUIRE_LOOP"
    monkeypatch.delenv("METERING_REQUIRE_LOOP", raising=False)
    assert cost_bus.require_loop() is False
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("METERING_REQUIRE_LOOP", truthy)
        assert cost_bus.require_loop() is True, truthy
    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("METERING_REQUIRE_LOOP", falsy)
        assert cost_bus.require_loop() is False, falsy


def test_schedule_no_running_loop_escalates_when_require_loop_is_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`METERING_REQUIRE_LOOP=true` turns the latent trap into a red build.

    ⚠️ Caught as `BaseException` and type-asserted on purpose: a mutant that
    downgraded the failure to a `Skipped` would satisfy a bare
    `pytest.raises(...)`-shaped expectation and read as exit 0.
    """
    monkeypatch.setenv("METERING_REQUIRE_LOOP", "true")
    before = _dropped_no_loop_count()

    raised: BaseException | None = None
    with caplog.at_level(logging.WARNING, logger=cost_bus.logger.name):
        try:
            schedule_cost_event(cast(Any, AsyncMock()), **_common_kwargs())
        except BaseException as exc:  # noqa: BLE001 — the type IS the assertion
            raised = exc

    assert type(raised) is cost_bus.NoRunningLoopError, raised
    # It must stay catchable by a caller already guarding the sync/async boundary.
    assert isinstance(raised, RuntimeError)
    # ⭐ COUNT BEFORE RAISE: turning escalation on must not blind the metric that
    # measures the very thing it escalates.
    assert _dropped_no_loop_count() - before == 1.0
    assert len(_no_loop_warnings(caplog)) == 1


async def test_schedule_with_a_running_loop_is_unchanged_and_reports_handoff(
    redis_conn: fakeredis.aioredis.FakeRedis, caplog: pytest.LogCaptureFixture
) -> None:
    """ARM A (running loop): behaviour is unchanged, and now SAYS it is.

    The event is still handed off and still lands on the stream; nothing is
    counted as dropped and nothing is warned about. The only difference is the
    positive marker the caller can now read.
    """
    before = _dropped_no_loop_count()
    with caplog.at_level(logging.WARNING, logger=cost_bus.logger.name):
        handed_off = schedule_cost_event(redis_conn, **_common_kwargs())
        assert handed_off is True
        # A real task was spawned and strongly retained.
        assert len(cost_bus._background_tasks) == 1
        await drain_pending()

    assert await redis_conn.xlen(resolve_stream()) == 1
    assert _dropped_no_loop_count() - before == 0.0
    assert _no_loop_warnings(caplog) == []


def test_disarmed_and_loop_free_is_counted_once_as_no_loop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Interaction with `dropped_disabled` (0.79.0): exactly ONE of them fires.

    A `None` redis_conn normally records `dropped_disabled` — but that happens
    inside `emit_cost_event`, and with no running loop the coroutine is never
    created, so the emit is never entered. Counting both would double-count one
    lost event and inflate every drop ratio; counting neither is the original
    defect. The correct answer is the one that names the reason nothing was sent.
    """
    before_no_loop = _dropped_no_loop_count()
    before_disabled = _cost_event_count("dropped_disabled")

    with caplog.at_level(logging.WARNING, logger=cost_bus.logger.name):
        assert schedule_cost_event(None, **_common_kwargs()) is False

    assert _dropped_no_loop_count() - before_no_loop == 1.0
    assert _cost_event_count("dropped_disabled") - before_disabled == 0.0
    assert len(_no_loop_warnings(caplog)) == 1


def test_the_no_loop_surface_is_re_exported_from_the_package() -> None:
    """Consumers import from `w2v_common.metering`, never from `.cost_bus`.

    ⚠️ A CALL-SITE-class assertion: every consumer in the fleet (tts_backend,
    modality_gateway, mlx_backend, mlx_ingestion_backend) imports the package,
    so a fix defined in `cost_bus` but not re-exported here is a fix nobody can
    reach. Graded by import, not by reading `__all__`.
    """
    import w2v_common.metering as metering_pkg

    assert metering_pkg.DROPPED_NO_LOOP is cost_bus.DROPPED_NO_LOOP
    assert metering_pkg.NoRunningLoopError is cost_bus.NoRunningLoopError
    assert metering_pkg.require_loop is cost_bus.require_loop
    assert metering_pkg.cost_events_total is cost_bus.cost_events_total
    for name in ("DROPPED_NO_LOOP", "NoRunningLoopError", "require_loop", "cost_events_total"):
        assert name in metering_pkg.__all__, name


def test_dropped_no_loop_label_is_the_wired_one() -> None:
    """The wiring point, not the helper: the branch must use THIS label value.

    A body mutant that renames the constant while the branch keeps passing a
    literal would otherwise survive.
    """
    assert cost_bus.DROPPED_NO_LOOP == "dropped_no_loop"
    assert cost_bus.DROPPED_NO_LOOP not in ("ok", "failed")


# --- non-finite float sanitisation (valid-JSON billing integrity) ---------


def _reject_nonfinite(token: str) -> float:
    """`json.loads` parse_constant hook: RAISE on NaN/Infinity/-Infinity.

    Python's ``json.loads`` accepts the bare tokens by default, so a lenient
    round-trip would hide the bug. This makes parsing STRICT (RFC 8259) — the
    same rejection a Go/Rust/JS consumer would perform.
    """
    raise ValueError(f"non-finite JSON constant not allowed: {token!r}")


def test_build_fields_sanitizes_nonfinite_usage_extra_and_cost() -> None:
    """A non-finite usage_extra value or cost_usd must never yield invalid JSON.

    RED on origin/main: usage_extra serialises to ``{"ratio": NaN, "x": Infinity}``
    (bare tokens, invalid JSON) and cost_usd renders as ``"nan"``. GREEN after
    the fix: non-finite floats become JSON null, finite values are unchanged,
    cost_usd is a finite figure.
    """
    fields = cost_bus._build_fields(
        service_key="w2v-test",  # pragma: allowlist secret -- routing slug
        cost_source="w2v-test",
        tenant_id=_TENANT,
        license_id="",
        subject_id="",
        app_id="",
        provider="deepseek",
        model="deepseek-chat",
        served_model=None,
        input_tokens=1,
        output_tokens=2,
        latency_ms=3,
        cost_usd=float("nan"),
        provider_request_id="",
        usage_extra={
            "ratio": float("nan"),
            "x": float("inf"),
            "neg": float("-inf"),
            "ok": 1.5,
            "nested": {"bad": float("nan"), "good": 2},
            "arr": [float("inf"), 3, "s"],
        },
    )

    raw = fields["usage_extra"]
    # 1. No bare non-finite token survives in the serialised string.
    for token in ("NaN", "Infinity", "-Infinity"):
        assert token not in raw
    # 2. STRICT parse (rejects NaN/Infinity) succeeds — i.e. it IS valid JSON.
    parsed = json.loads(raw, parse_constant=_reject_nonfinite)
    # 3. Non-finite floats became JSON null; finite values are byte-identical.
    assert parsed == {
        "ratio": None,
        "x": None,
        "neg": None,
        "ok": 1.5,
        "nested": {"bad": None, "good": 2},
        "arr": [None, 3, "s"],
    }
    # 4. cost_usd is a finite figure, never the literal "nan"/"inf".
    assert fields["cost_usd"] == "0.00000000"
    assert "nan" not in fields["cost_usd"]
    assert "inf" not in fields["cost_usd"]


def test_build_fields_finite_values_unchanged() -> None:
    """Sanitisation must be a no-op for already-finite payloads (no reformat)."""
    fields = cost_bus._build_fields(
        service_key="w2v-test",  # pragma: allowlist secret -- routing slug
        cost_source="w2v-test",
        tenant_id=_TENANT,
        license_id="",
        subject_id="",
        app_id="",
        provider="deepseek",
        model="deepseek-chat",
        served_model=None,
        input_tokens=1,
        output_tokens=2,
        latency_ms=3,
        cost_usd=0.001234567,
        provider_request_id="",
        usage_extra={"a": 1.5, "b": [1, 2], "c": "s"},
    )
    assert json.loads(fields["usage_extra"]) == {"a": 1.5, "b": [1, 2], "c": "s"}
    assert fields["cost_usd"] == "0.00123457"


# --- init_metering_redis must never log the DSN password (credential leak) ---
# The connect success path used to log the raw METERING_REDIS_URL at INFO, so
# every metering-enabled service put its Redis password into Loki once per
# process start. These assert on the FORMATTED record: a "%s" template with the
# DSN still in record.args leaks through virtually every handler, so asserting
# on the template alone would pass while the leak persisted.
_LEAKY_DSN = "redis://:s3cr3t-p4ss@metering-redis.w2v.svc:6379/0"  # pragma: allowlist secret -- fake DSN for a leak assertion


@pytest.fixture
def _fake_from_url(monkeypatch: pytest.MonkeyPatch) -> None:
    import redis.asyncio as aioredis

    def _from_url(url: str, **kwargs: Any) -> Any:
        return fakeredis.aioredis.FakeRedis(decode_responses=True)

    monkeypatch.setattr(aioredis, "from_url", _from_url)


def _connect_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if "Metering Redis connected" in record.getMessage()
    ]


async def test_init_metering_redis_never_logs_dsn_password(
    _fake_from_url: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO", logger=cost_bus.logger.name):
        client = await cost_bus.init_metering_redis(_LEAKY_DSN)
    assert client is not None
    await client.aclose()

    messages = _connect_messages(caplog)
    assert len(messages) == 1
    message = messages[0]
    assert "s3cr3t-p4ss" not in message
    assert "@" not in message
    # Still diagnostically useful: host (and port) survive.
    assert "metering-redis.w2v.svc" in message
    assert "6379" in message


async def test_init_metering_redis_logs_host_port_without_userinfo(
    _fake_from_url: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO", logger=cost_bus.logger.name):
        client = await cost_bus.init_metering_redis("redis://metering-redis.w2v.svc:6379/0")
    assert client is not None
    await client.aclose()

    assert _connect_messages(caplog) == ["Metering Redis connected: metering-redis.w2v.svc:6379"]


async def test_init_metering_redis_malformed_url_does_not_raise(
    _fake_from_url: None, caplog: pytest.LogCaptureFixture
) -> None:
    # Port is not an integer -> urlsplit().port raises ValueError.
    with caplog.at_level("INFO", logger=cost_bus.logger.name):
        client = await cost_bus.init_metering_redis("redis://:pw@host:not-a-port/0")
    assert client is not None
    await client.aclose()

    assert _connect_messages(caplog) == ["Metering Redis connected: <unparseable>"]


# --- FIX C1: the retry set must cover the failure production actually hits ---
# Measured on mini2 (2026-08-28) across the three live launchd services:
# 43 ``cost_event_emit_failed`` records, of which **29 terminate in a bare
# ``RuntimeError: unable to perform operation on <TCPTransport closed=True ...>``
# raised by asyncio, not by redis-py. That class was NOT in the retryable set,
# so those 29 events were dropped without a single retry. A test that injects a
# ``redis.ConnectionError`` passes against the OLD code and proves nothing —
# these inject the real shapes.

_CLOSED_TRANSPORT_MSG = (
    "unable to perform operation on <TCPTransport closed=True reading=False "
    "0x9958dd810>; the handler is closed"
)


class _FlakyRaising:
    """xadd-only double that raises ``exc`` for the first ``fail_times`` calls."""

    def __init__(
        self, inner: fakeredis.aioredis.FakeRedis, fail_times: int, exc: BaseException
    ) -> None:
        self._inner = inner
        self._remaining_failures = fail_times
        self._exc = exc
        self.xadd_calls = 0

    async def xadd(self, *args: Any, **kwargs: Any) -> Any:
        self.xadd_calls += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise self._exc
        return await self._inner.xadd(*args, **kwargs)


async def test_emit_retries_on_bare_asyncio_runtime_error(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale-transport ``RuntimeError`` must retry, not silently drop the event."""
    monkeypatch.setattr(cost_bus, "_DEFAULT_EMIT_RETRY_BACKOFFS", (0.0, 0.0))
    flaky = _FlakyRaising(redis_conn, fail_times=1, exc=RuntimeError(_CLOSED_TRANSPORT_MSG))
    await emit_cost_event(cast(Any, flaky), **_common_kwargs())
    assert flaky.xadd_calls == 2, "bare RuntimeError bypassed the retry and dropped the event"
    assert await redis_conn.xlen(resolve_stream()) == 1


async def test_emit_retries_on_connection_reset_error(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ConnectionResetError`` (an OSError, not a redis error) must also retry."""
    monkeypatch.setattr(cost_bus, "_DEFAULT_EMIT_RETRY_BACKOFFS", (0.0, 0.0))
    flaky = _FlakyRaising(
        redis_conn, fail_times=1, exc=ConnectionResetError(54, "Connection reset by peer")
    )
    await emit_cost_event(cast(Any, flaky), **_common_kwargs())
    assert flaky.xadd_calls == 2
    assert await redis_conn.xlen(resolve_stream()) == 1


# --- FIX C2: the failure log must never carry the DSN credential -------------

_LEAKY_EMIT_DSN = "redis://metering:sup3rs3cret@billing-redis.example:6379/0"  # pragma: allowlist secret -- fake DSN for a leak assertion


def _rendered(record: logging.LogRecord) -> str:
    """Render a record exactly as a handler would — message PLUS exception text."""
    return logging.Formatter("%(message)s").format(record)


def _emit_failure_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "cost_event_emit_failed" in r.getMessage()]


async def test_emit_failure_log_never_contains_the_dsn_password(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A transport error whose text embeds the DSN must not reach the log verbatim."""
    monkeypatch.setattr(cost_bus, "_DEFAULT_EMIT_RETRY_BACKOFFS", (0.0,))
    broken = AsyncMock()
    broken.xadd.side_effect = RuntimeError(f"connection to {_LEAKY_EMIT_DSN} is closed")

    with caplog.at_level("WARNING", logger=cost_bus.logger.name):
        await emit_cost_event(cast(Any, broken), **_common_kwargs())

    records = _emit_failure_records(caplog)
    assert len(records) == 1
    rendered = _rendered(records[0])
    assert "sup3rs3cret" not in rendered
    assert "metering:sup3rs3cret@" not in rendered
    # The cause must survive redaction — this is a redaction, not a gag.
    assert "RuntimeError" in rendered
    assert "billing-redis.example:6379" in rendered


async def test_init_metering_redis_failure_log_never_contains_the_dsn_password(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import redis.asyncio as aioredis

    def _boom(url: str, **kwargs: Any) -> Any:
        raise RuntimeError(f"cannot connect using {url}")

    monkeypatch.setattr(aioredis, "from_url", _boom)

    with caplog.at_level("WARNING", logger=cost_bus.logger.name):
        client = await cost_bus.init_metering_redis(_LEAKY_EMIT_DSN)
    assert client is None

    records = [r for r in caplog.records if "metering_redis_init_failed" in r.getMessage()]
    assert len(records) == 1
    rendered = _rendered(records[0])
    assert "sup3rs3cret" not in rendered
    assert "RuntimeError" in rendered
    assert "billing-redis.example:6379" in rendered


# --- FIX C3: the client must be built with staleness defences ---------------


async def test_init_metering_redis_sets_connection_resilience_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No health check / keepalive is why a closed transport survives an idle period."""
    import redis.asyncio as aioredis

    captured: dict[str, Any] = {}

    def _from_url(url: str, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return fakeredis.aioredis.FakeRedis(decode_responses=True)

    monkeypatch.setattr(aioredis, "from_url", _from_url)
    client = await cost_bus.init_metering_redis("redis://metering-redis.w2v.svc:6379/0")
    assert client is not None
    await client.aclose()

    assert captured.get("decode_responses") is True
    assert captured.get("health_check_interval") == 30
    assert captured.get("socket_keepalive") is True
    assert captured.get("retry_on_timeout") is True


# --- FIX C4: a success signal, so a success:failure ratio is computable ------


def _cost_event_count(outcome: str) -> float:
    from prometheus_client import REGISTRY

    value = REGISTRY.get_sample_value(
        "w2v_cost_events_total",
        {"service_key": "w2v-test", "outcome": outcome},  # pragma: allowlist secret -- routing slug
    )
    return 0.0 if value is None else value


async def test_emit_success_increments_the_ok_counter(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    before = _cost_event_count("ok")
    await emit_cost_event(redis_conn, **_common_kwargs())
    assert _cost_event_count("ok") - before == 1.0


async def test_emit_failure_increments_the_failed_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cost_bus, "_DEFAULT_EMIT_RETRY_BACKOFFS", (0.0,))
    broken = AsyncMock()
    broken.xadd.side_effect = RuntimeError(_CLOSED_TRANSPORT_MSG)
    before_failed = _cost_event_count("failed")
    before_ok = _cost_event_count("ok")
    await emit_cost_event(cast(Any, broken), **_common_kwargs())
    assert _cost_event_count("failed") - before_failed == 1.0
    assert _cost_event_count("ok") - before_ok == 0.0


class TestRedisEndpointNeverBreaksBoot:
    """`_redis_endpoint` renders a log string; it must be TOTAL.

    Its caller re-invokes it from inside its own `except` handler, so an escape here
    does not merely spoil a log line — it replaces the real connection error and
    aborts service startup. Shipped in 0.67.9 as `except ValueError`, which took out
    six lifespan tests in ai_content_generation_service the same day.
    """

    def test_a_non_string_url_is_unparseable_not_an_exception(self) -> None:
        """A mocked settings attribute is not a `str`.

        `urlsplit` reaches `if url.find(':') > 0`, and comparing a MagicMock to an
        int raises TypeError — a type `except ValueError` does not cover.
        """
        from unittest.mock import MagicMock

        assert cost_bus._redis_endpoint(MagicMock()) == "<unparseable>"

    def test_a_real_dsn_still_renders_host_and_port_without_userinfo(self) -> None:
        """The redaction property the function exists for is unchanged."""
        rendered = cost_bus._redis_endpoint("redis://:sekrit@metering.example:6380/2")
        assert rendered == "metering.example:6380"
        assert "sekrit" not in rendered

    def test_a_malformed_port_is_still_unparseable(self) -> None:
        """The original ValueError path (`.port` out of range) is preserved."""
        assert cost_bus._redis_endpoint("redis://host:99999/0") == "<unparseable>"


# --- event_key: the producer-side idempotency key --------------------------
#
# billing-service claims ``event_key`` in the ``ProcessedCostEventKey`` inbox
# (``cost_flusher.py:459-493``) and drops a repeat. An event that arrives with
# NO event_key skips the claim entirely and is billed unconditionally. Since
# billing is append-only — no refund, void or reversal path exists anywhere —
# a duplicate that reaches the ledger is money the tenant never gets back.


class _AcceptThenFailRedis:
    """xadd that DELIVERS the write and *then* raises, ``fail_times`` times.

    This is the dangerous shape ``_FlakyRedis`` does not cover: that double
    fails BEFORE the write lands, so its retry is genuinely free. A redis
    timeout raised AFTER the server accepted the XADD leaves the event in the
    stream and still looks like a transport failure to the caller, so the
    bounded retry re-adds the SAME charge under a fresh stream message id.
    """

    def __init__(self, inner: Any, fail_times: int) -> None:
        self._inner = inner
        self._fail_times = fail_times
        self.xadd_calls = 0

    async def xadd(self, *args: Any, **kwargs: Any) -> Any:
        self.xadd_calls += 1
        result = await self._inner.xadd(*args, **kwargs)  # the write LANDS
        if self.xadd_calls <= self._fail_times:
            raise RedisTimeoutError("timed out waiting for the reply")
        return result


async def _stream_events(redis_conn: Any) -> list[dict[str, str]]:
    entries = await redis_conn.xrange(resolve_stream())
    return [fields for _msg_id, fields in entries]


async def test_emit_stamps_an_event_key(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    await emit_cost_event(cast(Any, redis_conn), **_common_kwargs())
    events = await _stream_events(redis_conn)
    assert len(events) == 1
    key = events[0].get("event_key")
    assert key, "no event_key ⇒ billing-service bills the event unconditionally"
    # Must be parseable as the uuid7 billing's inbox column expects.
    assert uuid.UUID(key).version == 7


async def test_a_retry_after_a_landed_write_reuses_the_same_event_key(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cost_bus, "_DEFAULT_EMIT_RETRY_BACKOFFS", (0.0, 0.0))
    flaky = _AcceptThenFailRedis(redis_conn, fail_times=1)

    await emit_cost_event(cast(Any, flaky), **_common_kwargs())

    assert flaky.xadd_calls == 2
    events = await _stream_events(redis_conn)
    # The duplicate IS in the stream — that is inherent to an ambiguous timeout
    # and is not what this fix prevents.
    assert len(events) == 2
    keys = {e["event_key"] for e in events}
    # ...but both copies carry ONE key, so billing's inbox claim collapses them
    # to a single charge. Two distinct keys here = the tenant pays twice.
    assert len(keys) == 1, f"retry minted a second billable identity: {keys}"


async def test_two_distinct_calls_get_distinct_event_keys(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    # The key must not be constant/derived-from-payload: two genuinely separate
    # metered calls with IDENTICAL payloads are two charges and must both bill.
    await emit_cost_event(cast(Any, redis_conn), **_common_kwargs())
    await emit_cost_event(cast(Any, redis_conn), **_common_kwargs())
    events = await _stream_events(redis_conn)
    assert len({e["event_key"] for e in events}) == 2


# --- metadata_extra passthrough -------------------------------------------
#
# billing_service persists a fixed metadata allowlist (`src/services/_parsers.py`)
# into `billing_cost_events.metadata`. kag_common 0.30.0 already ships these via
# a `metadata_extra` kwarg; w2v_common had NO kwarg for any of them, so its
# consumers (mlx_backend, mlx_ingestion_backend, modality_gateway, tts_backend)
# physically could not stamp them. Measured over 24h: mlx-backend emitted 14,096
# rows — 82% of all metered traffic — with an EMPTY metadata column, and
# `caller_client_id`, the one field that names which service spent the money, was
# zero across the whole estate.


def _build_fields_kwargs(**overrides: Any) -> dict[str, Any]:
    """The full positional-free kwarg set `_build_fields` requires."""
    base: dict[str, Any] = dict(
        service_key="w2v-test",  # pragma: allowlist secret -- routing slug
        cost_source="w2v-test",
        tenant_id=_TENANT,
        license_id="",
        subject_id="",
        app_id="",
        provider="deepseek",
        model="deepseek-chat",
        served_model=None,
        input_tokens=1,
        output_tokens=2,
        latency_ms=3,
        cost_usd=0.001234567,
        provider_request_id="",
        usage_extra={"ok": 1.5},
    )
    base.update(overrides)
    return base


async def test_metadata_extra_puts_allowlisted_keys_on_the_wire(
    redis_conn: fakeredis.aioredis.FakeRedis,
) -> None:
    """RED on origin/main: `emit_cost_event` has no `metadata_extra` kwarg at all.

    On origin/main this raises `TypeError: emit_cost_event() got an unexpected
    keyword argument 'metadata_extra'` — which is precisely why every consumer
    ships an empty `metadata` column today.
    """
    await emit_cost_event(
        cast(Any, redis_conn),
        **_common_kwargs(),
        metadata_extra={"job_id": "j1", "stage": "s"},
    )
    events = await _stream_events(redis_conn)
    assert len(events) == 1
    assert events[0]["job_id"] == "j1"
    assert events[0]["stage"] == "s"


def test_metadata_extra_drops_a_key_outside_the_allowlist(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unknown key is dropped HERE, with a DEBUG line naming it.

    Shipping it instead would have it discarded silently by billing's own
    allowlist loop three hops away, where nothing can see it and the row still
    looks correct. The allowlisted sibling in the same mapping must survive, so
    this cannot pass by dropping everything.
    """
    with caplog.at_level(logging.DEBUG, logger=cost_bus.__name__):
        fields = cost_bus._build_fields(
            **_build_fields_kwargs(metadata_extra={"job_id": "j1", "not_on_the_allowlist": "x"})
        )
    assert fields["job_id"] == "j1"
    assert "not_on_the_allowlist" not in fields
    assert any(
        "cost_event_metadata_keys_dropped" in r.getMessage()
        and "not_on_the_allowlist" in r.getMessage()
        for r in caplog.records
    ), "a dropped key must be named at DEBUG, not vanish"


def test_metadata_extra_skips_a_none_value_rather_than_stringifying_it() -> None:
    """`str(None)` is the TRUTHY literal "None".

    Every value on this wire is coerced to `str`, and billing's receiver stores a
    key when `if c.get(k)` is truthy. A naive `fields[key] = str(value)` would
    therefore persist the string "None" as if it were real attribution — worse
    than no metadata, because it is confidently wrong. "Not provided" must be
    ABSENT, not "None".
    """
    fields = cost_bus._build_fields(
        **_build_fields_kwargs(metadata_extra={"job_id": None, "stage": "s", "worker_name": None})
    )
    assert "job_id" not in fields
    assert "worker_name" not in fields
    assert "None" not in fields.values()
    # ...and the None values did not poison the sibling that DID have a value.
    assert fields["stage"] == "s"


def test_omitting_metadata_extra_emits_a_byte_identical_field_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The safety property: omitting the kwarg emits EXACTLY today's dict.

    `metadata_extra` is keyword-only with a `None` default, so all 19 consumers
    pick up the new library without a source change and without a wire change.
    The two volatile fields are pinned (uuid7 `event_key`, `timestamp`) so the
    comparison is against a LITERAL transcribed from origin/main's output — an
    added, removed, renamed or reformatted field fails this test, which a
    key-set or subset assertion would not.
    """
    pinned_key = uuid.UUID("0195f48b-d000-7000-8000-00000000beef")
    monkeypatch.setattr(cost_bus.uuid, "uuid7", lambda: pinned_key)

    class _PinnedDatetime:
        @staticmethod
        def now(_tz: Any = None) -> Any:
            return datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

    monkeypatch.setattr(cost_bus, "datetime", _PinnedDatetime)

    expected = {
        "event_key": "0195f48b-d000-7000-8000-00000000beef",
        "tenant_id": str(_TENANT),
        "license_id": "",
        "subject_id": "",
        "app_id": "",
        "service_key": "w2v-test",  # pragma: allowlist secret -- routing slug
        "cost_source": "w2v-test",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "served_model": "deepseek-chat",
        "input_tokens": "1",
        "output_tokens": "2",
        "cost_usd": "0.00123457",
        "latency_ms": "3",
        "provider_request_id": "",
        "usage_extra": '{"ok": 1.5}',
        "timestamp": "2026-01-02T03:04:05+00:00",
        # 0.85.0: the calling app's own project is always attributed; with no
        # W2V project bound that is the tenant-level project.
        "project_id": tenant_level_project_id(_TENANT),
    }

    omitted = cost_bus._build_fields(**_build_fields_kwargs())
    assert omitted == expected

    # Explicitly-None and empty-mapping take the same early return.
    assert cost_bus._build_fields(**_build_fields_kwargs(metadata_extra=None)) == expected
    assert cost_bus._build_fields(**_build_fields_kwargs(metadata_extra={})) == expected


# ---------------------------------------------------------------------------
# actor_id — WHO ACTED, alongside subject_id (WHO PAYS). billing alembic 0021.
# ---------------------------------------------------------------------------


@pytest.fixture
def _no_identity(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Guarantee a clean identity context around each actor test.

    ContextVars survive across tests in the same task, so an actor left bound by
    one test would silently satisfy the next — and the ABSENT-key assertions are
    exactly the ones that would pass for the wrong reason. ``SERVICE_ACTOR_ID``
    is cleared for the same reason: an inherited one would fill the absent key.
    """
    monkeypatch.delenv("SERVICE_ACTOR_ID", raising=False)
    clear_identity_context()
    yield
    clear_identity_context()


def test_actor_id_is_sourced_from_the_context_var(_no_identity: Any) -> None:
    """An actor bound by the JWT lane reaches the wire under `actor_id`."""
    set_identity_context(
        tenant_id=_TENANT,
        license_id=None,
        subject_id="pays-for-it",
        actor_id="0195f48b-0000-7000-8000-0000000000ac",
    )
    fields = cost_bus._build_fields(**_build_fields_kwargs())
    assert fields["actor_id"] == "0195f48b-0000-7000-8000-0000000000ac"
    # WHO PAYS is a DIFFERENT field and must not have been overwritten or
    # mirrored: billing carries both, and collapsing them is a mis-bill.
    assert fields["subject_id"] == ""


def test_actor_id_is_absent_when_no_actor_is_bound(_no_identity: Any) -> None:
    """No actor => the KEY IS GONE. Not "", not "None".

    billing's column is nullable and never defaulted; `if c.get(k)` stores a
    truthy "None" as real attribution, and an "" is a written-down claim that
    attribution was attempted when it was not.
    """
    set_identity_context(tenant_id=_TENANT, license_id=None, subject_id="s")
    fields = cost_bus._build_fields(**_build_fields_kwargs())
    assert "actor_id" not in fields
    assert "None" not in fields.values()
    assert "" not in [v for k, v in fields.items() if k == "actor_id"]


def test_explicit_metadata_actor_id_beats_the_context_var(_no_identity: Any) -> None:
    """A caller that states the actor wins over the ambient one.

    The re-dispatch case: a queue worker running under its own context knows the
    actor from the persisted row, and that record — not whoever happens to be
    bound to this process right now — is the truth.
    """
    set_identity_context(
        tenant_id=_TENANT, license_id=None, subject_id="s", actor_id="ambient-actor"
    )
    fields = cost_bus._build_fields(
        **_build_fields_kwargs(metadata_extra={"actor_id": "from-the-row"})
    )
    assert fields["actor_id"] == "from-the-row"


def test_explicit_none_actor_suppresses_the_context_var(_no_identity: Any) -> None:
    """Naming the key means you are talking about it — including to clear it.

    Mirrors `set_identity_context(tenant_source=None)`: an explicit None is how a
    caller says "no actor for this event", so the ambient one must NOT leak back
    in. It lands as an ABSENT key, never the literal "None".
    """
    set_identity_context(
        tenant_id=_TENANT, license_id=None, subject_id="s", actor_id="ambient-actor"
    )
    fields = cost_bus._build_fields(**_build_fields_kwargs(metadata_extra={"actor_id": None}))
    assert "actor_id" not in fields
    assert "None" not in fields.values()


async def test_actor_id_reaches_the_stream_end_to_end(
    redis_conn: fakeredis.aioredis.FakeRedis, _no_identity: Any
) -> None:
    """The allowlist admits `actor_id`, so it survives all the way to the XADD."""
    actor = "0195f48b-0000-7000-8000-0000000000ad"
    set_identity_context(tenant_id=_TENANT, license_id=None, subject_id="s", actor_id=actor)
    await emit_cost_event(cast(Any, redis_conn), **_common_kwargs())
    events = await _stream_events(redis_conn)
    assert len(events) == 1
    assert events[0]["actor_id"] == actor


# --- attribution: license_id / subject_id / app_id ------------------------
#
# Billing marks all three REQUIRED, but ``_parse_uuid`` maps "" to NULL with no
# exception, metric or log — 94,615 unattributable ``billing_metering_events``
# rows in 14 days. These tests hold the three properties that end that:
# it is OBSERVED, it is ESCALATABLE, and it does NOT become tenant_id.


def _unattributed_count(field: str) -> float:
    from prometheus_client import REGISTRY

    value = REGISTRY.get_sample_value("w2v_cost_event_unattributed_total", {"field": field})
    return 0.0 if value is None else value


def test_every_attribution_field_has_a_zero_series_before_it_ever_fires() -> None:
    """A labelled Counter with no child exports NOTHING.

    On a ``/metrics`` scrape "no series" is indistinguishable from "the library
    was never imported" — which is the exact question this metric answers. The
    module instantiates every known label at import time, so the zero is real.
    """
    from prometheus_client import REGISTRY

    for field in ATTRIBUTION_FIELDS:
        assert (
            REGISTRY.get_sample_value("w2v_cost_event_unattributed_total", {"field": field})
            is not None
        ), field


async def test_omitting_attribution_emits_the_row_counts_it_and_warns(
    redis_conn: fakeredis.aioredis.FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Default behaviour: the revenue is recorded, the miss is made visible.

    ⛔ Deliberately NOT a raise. A cost event is money; an unattributed row can
    be repaired later, a dropped one cannot.
    """
    monkeypatch.delenv("METERING_REQUIRE_ATTRIBUTION", raising=False)
    before = {field: _unattributed_count(field) for field in ATTRIBUTION_FIELDS}
    with caplog.at_level(logging.WARNING, logger="w2v_common.metering.cost_bus"):
        await emit_cost_event(redis_conn, **_common_kwargs())

    assert await redis_conn.xlen(resolve_stream()) == 1
    for field in ATTRIBUTION_FIELDS:
        assert _unattributed_count(field) - before[field] == 1.0, field
    warnings = [
        r.getMessage() for r in caplog.records if "cost_event_unattributed" in r.getMessage()
    ]
    assert len(warnings) == 1
    for field in ATTRIBUTION_FIELDS:
        assert field in warnings[0]


async def test_an_empty_string_is_as_unattributed_as_an_omission(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both sink to the same NULL column at the receiver, so both are counted."""
    monkeypatch.delenv("METERING_REQUIRE_ATTRIBUTION", raising=False)
    before = _unattributed_count("app_id")
    await emit_cost_event(
        redis_conn,
        **_common_kwargs(license_id=str(uuid.uuid7()), subject_id=str(uuid.uuid7()), app_id="  "),
    )
    assert _unattributed_count("app_id") - before == 1.0


async def test_full_attribution_is_not_counted_and_reaches_the_stream(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("METERING_REQUIRE_ATTRIBUTION", raising=False)
    before = {field: _unattributed_count(field) for field in ATTRIBUTION_FIELDS}
    license_id, subject_id, app_id = (str(uuid.uuid7()) for _ in range(3))
    await emit_cost_event(
        redis_conn,
        **_common_kwargs(license_id=license_id, subject_id=subject_id, app_id=app_id),
    )
    for field in ATTRIBUTION_FIELDS:
        assert _unattributed_count(field) == before[field], field
    fields = (await redis_conn.xrange(resolve_stream()))[0][1]
    assert fields["license_id"] == license_id
    assert fields["subject_id"] == subject_id
    assert fields["app_id"] == app_id


async def test_escalation_flag_turns_a_missing_field_into_a_raise(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    with pytest.raises(MissingAttributionError) as excinfo:
        await emit_cost_event(redis_conn, **_common_kwargs())
    assert "app_id" in str(excinfo.value)
    assert await redis_conn.xlen(resolve_stream()) == 0


def test_escalation_raises_synchronously_from_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside the detached task this would be an 'exception never retrieved'."""
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    with pytest.raises(MissingAttributionError):
        schedule_cost_event(cast(Any, AsyncMock()), **_common_kwargs())


async def test_escalation_is_off_unless_the_flag_is_truthy(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "false")
    await emit_cost_event(redis_conn, **_common_kwargs())
    assert await redis_conn.xlen(resolve_stream()) == 1


async def test_tenant_id_still_hard_fails_and_attribution_does_not_soften_it(
    redis_conn: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tenant_id keeps its 2026-08-26 treatment: raise, never record.

    The softer attribution policy applies to the OTHER three fields only. A
    wrong/absent tenant bills the wrong customer; a missing app_id only costs
    the attribution — different blast radius, different default.
    """
    monkeypatch.delenv("METERING_REQUIRE_ATTRIBUTION", raising=False)
    with pytest.raises(MissingTenantError):
        await emit_cost_event(
            redis_conn,
            **_common_kwargs(
                tenant_id="",
                license_id=str(uuid.uuid7()),
                subject_id=str(uuid.uuid7()),
                app_id=str(uuid.uuid7()),
            ),
        )
    assert await redis_conn.xlen(resolve_stream()) == 0


# --- w2v_metering_enabled: "this process bills nothing" must be visible ------
#
# ``init_metering_redis`` returns None on an empty METERING_REDIS_URL and every
# consumer then drops cost events behind ``if app.state.metering_redis:`` — a
# guard cost_bus cannot see. Four production services billed nothing and looked
# healthy. A Gauge (not a counter) is the right shape: it alerts with NO
# traffic, which is exactly the state a silently-disarmed service is in.


def _metering_enabled() -> float | None:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value("w2v_metering_enabled")


def test_metering_enabled_series_exists_from_import() -> None:
    """An absent series is indistinguishable from 'library never loaded'."""
    assert _metering_enabled() is not None


async def test_empty_url_sets_the_gauge_to_zero_and_returns_none() -> None:
    cost_bus.metering_enabled.set(1)
    client = await cost_bus.init_metering_redis("")
    assert client is None
    assert _metering_enabled() == 0.0


async def test_successful_init_sets_the_gauge_to_one(_fake_from_url: None) -> None:
    cost_bus.metering_enabled.set(0)
    client = await cost_bus.init_metering_redis("redis://metering-redis.w2v.svc:6379/0")
    assert client is not None
    assert _metering_enabled() == 1.0


async def test_failed_init_sets_the_gauge_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    import redis.asyncio as aioredis

    def _boom(url: str, **kwargs: Any) -> Any:
        raise RedisConnectionError("metering redis unreachable")

    monkeypatch.setattr(aioredis, "from_url", _boom)
    cost_bus.metering_enabled.set(1)
    client = await cost_bus.init_metering_redis("redis://metering-redis.w2v.svc:6379/0")
    assert client is None
    assert _metering_enabled() == 0.0


async def test_emit_with_no_connection_counts_dropped_disabled_and_never_raises() -> None:
    """The drop is COUNTABLE, on the existing outcome label — not a new metric.

    ``failed`` already covers redis errors and bad payloads; a second metric
    name here would split the billing dashboard.
    """
    before = _cost_event_count("dropped_disabled")
    before_failed = _cost_event_count("failed")
    await emit_cost_event(None, **_common_kwargs())
    assert _cost_event_count("dropped_disabled") - before == 1.0
    assert _cost_event_count("failed") - before_failed == 0.0


async def test_schedule_with_no_connection_drops_exactly_once() -> None:
    """The fire-and-forget sibling must accept ``None`` too, and count it ONCE.

    ``emit_cost_event`` widening alone is not enough: the entire point of
    accepting ``None`` is that a consumer can delete its
    ``if app.state.metering_redis:`` guard, and the guardless consumers on the
    fire-and-forget path call THIS function. It forwards ``None`` verbatim into
    the spawned coroutine, so the SAME disarmed early-return runs and
    ``dropped_disabled`` must move by exactly 1 — not 2 (a second count here
    would mean the wrapper is observing the drop as well as the emit) and not 0
    (which would mean the task never ran or the guard was bypassed).
    """
    assert not cost_bus._background_tasks
    before = _cost_event_count("dropped_disabled")
    before_failed = _cost_event_count("failed")
    before_ok = _cost_event_count("ok")

    schedule_cost_event(None, **_common_kwargs())
    assert len(cost_bus._background_tasks) == 1
    await asyncio.gather(*list(cost_bus._background_tasks))

    assert _cost_event_count("dropped_disabled") - before == 1.0
    assert _cost_event_count("failed") - before_failed == 0.0
    assert _cost_event_count("ok") - before_ok == 0.0
    assert not cost_bus._background_tasks


async def test_schedule_with_no_connection_still_validates_the_tenant() -> None:
    """A disarmed process must not become a place where caller bugs hide.

    The synchronous tenant pre-check runs BEFORE the ``None`` is forwarded, so a
    missing tenant still raises in the caller's frame rather than being masked
    by the drop.
    """
    before = _cost_event_count("dropped_disabled")
    with pytest.raises(MissingTenantError):
        schedule_cost_event(None, **_common_kwargs(tenant_id=""))
    assert _cost_event_count("dropped_disabled") == before
