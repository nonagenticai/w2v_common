"""Tests for the shared GPU admission gate (``w2v_common.admission``, #217).

Every test uses a unique lane name, which isolates the default-REGISTRY
``gpu_admission_*`` series between tests.
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
import uuid

import httpx
import pytest
from fastapi import FastAPI, Request
from prometheus_client import REGISTRY

from w2v_common.admission import (
    AdmissionGate,
    ClientGone,
    GateClosedError,
    GpuBusy,
    LaneConfig,
    env_prefix,
    install_gpu_busy_handler,
)
from w2v_common.observability.metrics import render_latest


def _lane() -> str:
    return f"t{uuid.uuid7().hex[-12:]}"


def _gate(lane: str, active: int, waiting: int, wait_s: float, seed: float = 10.0) -> AdmissionGate:
    return AdmissionGate(
        {lane: LaneConfig(max_active=active, max_waiting=waiting, max_wait_s=wait_s, seed_s=seed)},
        environ={},
    )


def _sample(name: str, **labels: str) -> float:
    value = REGISTRY.get_sample_value(name, labels)
    return 0.0 if value is None else value


class _FakeRequest:
    def __init__(self, gone: bool) -> None:
        self.gone = gone
        self.calls = 0

    async def is_disconnected(self) -> bool:
        self.calls += 1
        return self.gone


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


# ── FIFO ──────────────────────────────────────────────────────────────────────


async def test_waiters_are_admitted_in_arrival_order() -> None:
    lane = _lane()
    gate = _gate(lane, active=1, waiting=10, wait_s=10)
    order: list[int] = []
    release_holder = asyncio.Event()

    async def holder() -> None:
        async with gate.slot(lane):
            await release_holder.wait()

    async def waiter(i: int) -> None:
        async with gate.slot(lane):
            order.append(i)
            # Yield while holding the slot so a non-FIFO wakeup could reorder.
            await asyncio.sleep(0.001 * (8 - i))

    h = asyncio.create_task(holder())
    await _settle()
    tasks = []
    for i in range(8):
        tasks.append(asyncio.create_task(waiter(i)))
        await _settle()
    assert gate.lane(lane).waiting == 8
    release_holder.set()
    await asyncio.gather(h, *tasks)
    assert order == list(range(8))


async def test_newcomer_cannot_overtake_a_queued_waiter() -> None:
    lane = _lane()
    gate = _gate(lane, active=1, waiting=5, wait_s=10)
    order: list[str] = []
    ev = asyncio.Event()

    async def holder() -> None:
        async with gate.slot(lane):
            await ev.wait()

    async def named(n: str) -> None:
        async with gate.slot(lane):
            order.append(n)

    h = asyncio.create_task(holder())
    await _settle()
    first = asyncio.create_task(named("queued-first"))
    await _settle()
    ev.set()
    # The holder's release wakes "queued-first"; a newcomer arriving in the same
    # tick must queue behind it, not take the freed permit.
    late = asyncio.create_task(named("late"))
    await asyncio.gather(h, first, late)
    assert order == ["queued-first", "late"]


# ── rejections ────────────────────────────────────────────────────────────────


async def test_full_queue_rejects_immediately_with_retry_after() -> None:
    lane = _lane()
    gate = _gate(lane, active=1, waiting=2, wait_s=30, seed=10)
    ev = asyncio.Event()

    async def hold() -> None:
        async with gate.slot(lane):
            await ev.wait()

    tasks = [asyncio.create_task(hold()) for _ in range(3)]  # 1 active + 2 waiting
    await _settle()
    assert (gate.lane(lane).active, gate.lane(lane).waiting) == (1, 2)

    t0 = time.monotonic()
    with pytest.raises(GpuBusy) as info:
        async with gate.slot(lane):
            pytest.fail("must not be admitted")
    assert time.monotonic() - t0 < 0.05
    assert info.value.reason == "full"
    assert info.value.lane == lane
    # position = 2 waiting ahead -> ceil(10 * 3 / 1) = 30
    assert info.value.retry_after == 30
    assert _sample("gpu_admission_rejected_total", lane=lane, reason="full") == 1
    assert _sample("gpu_admission_rejected_total", lane=lane, reason="timeout") == 0
    assert gate.lane(lane).waiting == 2
    ev.set()
    await asyncio.gather(*tasks)


async def test_max_waiting_zero_rejects_when_all_slots_busy() -> None:
    lane = _lane()
    gate = _gate(lane, active=1, waiting=0, wait_s=30)
    ev = asyncio.Event()

    async def hold() -> None:
        async with gate.slot(lane):
            await ev.wait()

    t = asyncio.create_task(hold())
    await _settle()
    with pytest.raises(GpuBusy, match="full"):
        async with gate.slot(lane):
            pass
    ev.set()
    await t


async def test_wait_timeout_rejects_and_unqueues() -> None:
    lane = _lane()
    gate = _gate(lane, active=1, waiting=5, wait_s=0.1, seed=4)
    ev = asyncio.Event()

    async def hold() -> None:
        async with gate.slot(lane):
            await ev.wait()

    t = asyncio.create_task(hold())
    await _settle()
    t0 = time.monotonic()
    with pytest.raises(GpuBusy) as info:
        async with gate.slot(lane):
            pytest.fail("must not be admitted")
    elapsed = time.monotonic() - t0
    assert 0.09 <= elapsed < 0.5
    assert info.value.reason == "timeout"
    assert info.value.retry_after == 4  # nobody left waiting: ceil(4 * 1 / 1)
    assert gate.lane(lane).waiting == 0
    assert _sample("gpu_admission_waiting", lane=lane) == 0
    assert _sample("gpu_admission_rejected_total", lane=lane, reason="timeout") == 1
    assert _sample("gpu_admission_rejected_total", lane=lane, reason="full") == 0
    ev.set()
    await t
    # The timed-out waiter did not leak a permit or a slot.
    async with gate.slot(lane):
        assert gate.lane(lane).active == 1
    assert gate.lane(lane).active == 0


async def test_timeout_is_measured_from_arrival_not_per_attempt() -> None:
    """B's queue position changes mid-wait (the slot turns over to A, ahead of it).

    B must still be rejected ``max_wait_s`` after it ARRIVED (~0.4 s), not
    ``max_wait_s`` after the turnover re-armed its wait (~0.25 + 0.4 = 0.65 s).
    """
    lane = _lane()
    gate = _gate(lane, active=1, waiting=5, wait_s=0.4)
    release_h = asyncio.Event()
    release_a = asyncio.Event()

    async def hold(ev: asyncio.Event) -> None:
        async with gate.slot(lane):
            await ev.wait()

    h = asyncio.create_task(hold(release_h))
    await _settle()
    a = asyncio.create_task(hold(release_a))  # queued at position 0
    await _settle()

    b_arrival = time.monotonic()

    async def b() -> float:
        with pytest.raises(GpuBusy) as info:
            async with gate.slot(lane):
                pytest.fail("B must not be admitted")
        assert info.value.reason == "timeout"
        return time.monotonic() - b_arrival

    bt = asyncio.create_task(b())  # queued at position 1
    await _settle()
    assert gate.lane(lane).waiting == 2

    await asyncio.sleep(0.25)
    release_h.set()  # H leaves, A is admitted; B moves up to position 0
    await asyncio.sleep(0.02)
    assert (gate.lane(lane).active, gate.lane(lane).waiting) == (1, 1)

    elapsed = await bt
    assert 0.38 <= elapsed < 0.58, elapsed
    release_a.set()
    await asyncio.gather(h, a)


# ── disconnect ────────────────────────────────────────────────────────────────


async def test_disconnected_client_is_skipped_not_rejected() -> None:
    lane = _lane()
    gate = _gate(lane, active=1, waiting=5, wait_s=5)
    calls: list[int] = []
    req = _FakeRequest(gone=True)

    with pytest.raises(ClientGone):
        await gate.run(lane, calls.append, 1, request=req)
    assert req.calls == 1
    assert calls == []  # the job never ran
    assert gate.lane(lane).active == 0
    assert _sample("gpu_admission_active", lane=lane) == 0
    assert _sample("gpu_admission_rejected_total", lane=lane, reason="full") == 0
    assert _sample("gpu_admission_rejected_total", lane=lane, reason="timeout") == 0
    # The slot was returned: a connected request is admitted at once.
    await gate.run(lane, calls.append, 2, request=_FakeRequest(gone=False))
    assert calls == [2]


async def test_disconnect_is_checked_at_admission_after_queueing() -> None:
    lane = _lane()
    gate = _gate(lane, active=1, waiting=5, wait_s=5)
    ev = asyncio.Event()
    req = _FakeRequest(gone=False)

    async def hold() -> None:
        async with gate.slot(lane):
            await ev.wait()

    h = asyncio.create_task(hold())
    await _settle()
    queued = asyncio.create_task(gate.slot(lane, request=req).__aenter__())
    await _settle()
    assert req.calls == 0  # not yet admitted
    req.gone = True  # client leaves while queued
    ev.set()
    with pytest.raises(ClientGone):
        await queued
    await h
    assert req.calls == 1
    assert gate.lane(lane).active == 0


# ── cancellation: the slot follows the THREAD ─────────────────────────────────


async def test_cancelled_awaiter_keeps_slot_until_thread_finishes() -> None:
    lane = _lane()
    gate = _gate(lane, active=1, waiting=5, wait_s=10)
    started = threading.Event()
    finish = threading.Event()

    def blocking() -> str:
        started.set()
        finish.wait(5)
        return "done"

    runner = asyncio.create_task(gate.run(lane, blocking))
    assert await asyncio.to_thread(started.wait, 2)
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner

    # Awaiter is gone, the thread is not: the slot must still be held.
    assert gate.lane(lane).active == 1
    assert _sample("gpu_admission_active", lane=lane) == 1
    second = asyncio.create_task(gate.run(lane, lambda: "second"))
    await asyncio.sleep(0.05)
    assert not second.done()
    assert gate.lane(lane).waiting == 1

    finish.set()
    assert await asyncio.wait_for(second, 2) == "second"
    assert gate.lane(lane).active == 0
    assert _sample("gpu_admission_active", lane=lane) == 0
    assert await gate.shutdown()


async def test_run_executes_off_loop_in_lane_executor_and_propagates_errors() -> None:
    lane = _lane()
    gate = _gate(lane, active=2, waiting=0, wait_s=1)
    loop_thread = threading.get_ident()
    names: list[str] = []

    def where(k: int, *, mult: int) -> int:
        assert threading.get_ident() != loop_thread
        names.append(threading.current_thread().name)
        return k * mult

    assert await gate.run(lane, functools.partial(where, mult=7), 3) == 21
    assert names[0].startswith(f"gpu-lane-{lane}")

    def boom() -> None:
        raise RuntimeError("model crashed")

    with pytest.raises(RuntimeError, match="model crashed"):
        await gate.run(lane, boom)
    assert gate.lane(lane).active == 0
    assert await gate.shutdown()


# ── Retry-After math + EWMA ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("seed", "active", "position", "expected"),
    [
        (10.0, 1, 0, 10),
        (10.0, 1, 4, 50),
        (10.0, 2, 4, 25),
        (10.0, 4, 0, 3),  # ceil(2.5)
        (3.0, 2, 2, 5),  # ceil(4.5)
        (0.01, 1, 0, 1),  # clamped up to 1
        (10.0, 1, 63, 120),  # 640 clamped down to 120
        (119.5, 1, 0, 120),  # ceil(119.5) = 120, at the clamp
    ],
)
def test_retry_after_math_and_clamp(seed: float, active: int, position: int, expected: int) -> None:
    lane = _lane()
    gate = _gate(lane, active=active, waiting=64, wait_s=10, seed=seed)
    assert gate.lane(lane).retry_after(position) == expected


async def test_ewma_moves_toward_observed_service_time() -> None:
    lane = _lane()
    gate = _gate(lane, active=1, waiting=0, wait_s=1, seed=10)
    async with gate.slot(lane):
        await asyncio.sleep(0.05)
    ewma = gate.lane(lane).ewma_s
    # 0.2 * ~0.05 + 0.8 * 10 = ~8.01
    assert 8.0 < ewma < 8.1
    assert gate.lane(lane).retry_after(0) == 9


# ── env parsing ───────────────────────────────────────────────────────────────


def test_env_overrides_code_defaults_per_field() -> None:
    default = LaneConfig(max_active=1, max_waiting=64, max_wait_s=120)
    env = {
        "GPU_LANE_TTS_MAIN_MAX_ACTIVE": "3",
        "GPU_LANE_TTS_MAIN_MAX_WAIT_S": "2.5",
        "GPU_LANE_TTS_MAIN_SEED_S": "4",
        "GPU_LANE_OTHER_MAX_WAITING": "1",
    }
    cfg = LaneConfig.from_env("tts-main", default, env)
    assert cfg == LaneConfig(max_active=3, max_waiting=64, max_wait_s=2.5, seed_s=4.0)
    assert LaneConfig.from_env("unset", default, env) == default
    assert env_prefix("tts-main") == "GPU_LANE_TTS_MAIN_"


def test_gate_reads_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    lane = _lane()
    monkeypatch.setenv(f"{env_prefix(lane)}MAX_WAITING", "7")
    gate = AdmissionGate({lane: LaneConfig(max_active=2, max_waiting=64, max_wait_s=120)})
    assert gate.lane(lane).config.max_waiting == 7
    assert gate.lane(lane).config.max_active == 2
    assert gate.lane(lane).ewma_s == 10.0  # default seed


def test_blank_env_value_keeps_default() -> None:
    default = LaneConfig(max_active=2, max_waiting=8, max_wait_s=30)
    assert LaneConfig.from_env("x", default, {"GPU_LANE_X_MAX_ACTIVE": "  "}) == default


@pytest.mark.parametrize(
    ("key", "raw", "match"),
    [
        ("MAX_ACTIVE", "two", "not a valid int"),
        ("MAX_ACTIVE", "0", "max_active must be in 1..256"),
        ("MAX_ACTIVE", "257", "max_active must be in 1..256"),
        ("MAX_WAITING", "-1", "max_waiting must be in 0..10000"),
        ("MAX_WAITING", "10001", "max_waiting must be in 0..10000"),
        ("MAX_WAIT_S", "0", "max_wait_s must be finite"),
        ("MAX_WAIT_S", "inf", "max_wait_s must be finite"),
        ("MAX_WAIT_S", "-inf", "max_wait_s must be finite"),
        ("MAX_WAIT_S", "nan", "max_wait_s must be finite"),
        ("MAX_WAIT_S", "1e308", "max_wait_s must be finite"),
        ("MAX_WAIT_S", "3600.5", "max_wait_s must be finite"),
        ("SEED_S", "nan", "seed_s must be finite"),
        ("SEED_S", "inf", "seed_s must be finite"),
        ("SEED_S", "1e308", "seed_s must be finite"),
        ("SEED_S", "-1", "seed_s must be finite"),
        ("SEED_S", "3601", "seed_s must be finite"),
    ],
)
def test_bad_env_value_fails_loud(key: str, raw: str, match: str) -> None:
    default = LaneConfig(max_active=1, max_waiting=1, max_wait_s=1)
    with pytest.raises(ValueError, match=match) as info:
        LaneConfig.from_env("x", default, {f"GPU_LANE_X_{key}": raw})
    assert f"GPU_LANE_X_{key}" in str(info.value)


@pytest.mark.parametrize("field", ["max_wait_s", "seed_s"])
@pytest.mark.parametrize("value", [float("inf"), float("nan"), 1e308])
def test_non_finite_or_huge_seconds_rejected_in_code_config(field: str, value: float) -> None:
    kwargs = {"max_active": 1, "max_waiting": 1, "max_wait_s": 1.0, "seed_s": 1.0, field: value}
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        LaneConfig(**kwargs)


def test_largest_accepted_seed_cannot_overflow_retry_after() -> None:
    lane = _lane()
    gate = _gate(lane, active=1, waiting=10_000, wait_s=3600, seed=3600)
    assert gate.lane(lane).retry_after(10_000) == 120
    assert gate.lane(lane).retry_after(0) == 120


def test_unknown_lane_and_empty_gate_raise() -> None:
    with pytest.raises(ValueError, match="at least one lane"):
        AdmissionGate({}, environ={})
    gate = _gate("known", 1, 0, 1)
    with pytest.raises(KeyError, match="known"):
        gate.lane("nope")


# ── metrics ───────────────────────────────────────────────────────────────────


async def test_metrics_track_active_waiting_wait_and_service() -> None:
    lane = _lane()
    gate = _gate(lane, active=1, waiting=3, wait_s=5)
    ev = asyncio.Event()

    async def hold(delay: float) -> None:
        async with gate.slot(lane):
            await asyncio.sleep(delay)
            await ev.wait()

    tasks = [asyncio.create_task(hold(0.02)) for _ in range(3)]
    await _settle()
    assert _sample("gpu_admission_active", lane=lane) == 1
    assert _sample("gpu_admission_waiting", lane=lane) == 2
    ev.set()
    await asyncio.gather(*tasks)
    assert _sample("gpu_admission_active", lane=lane) == 0
    assert _sample("gpu_admission_waiting", lane=lane) == 0
    assert _sample("gpu_admission_wait_seconds_count", lane=lane) == 3
    assert _sample("gpu_admission_service_seconds_count", lane=lane) == 3
    # Two waiters queued behind >=0.02 s of service each.
    assert _sample("gpu_admission_wait_seconds_sum", lane=lane) >= 0.04
    assert _sample("gpu_admission_service_seconds_sum", lane=lane) >= 0.06


def test_metric_names_are_exact_and_unprefixed() -> None:
    lane = _lane()
    _gate(lane, 1, 0, 1)
    text = render_latest()[0].decode()
    for line in (
        f'gpu_admission_active{{lane="{lane}"}} 0.0',
        f'gpu_admission_waiting{{lane="{lane}"}} 0.0',
        f'gpu_admission_rejected_total{{lane="{lane}",reason="full"}} 0.0',
        f'gpu_admission_rejected_total{{lane="{lane}",reason="timeout"}} 0.0',
    ):
        assert line in text
    assert "# TYPE gpu_admission_wait_seconds histogram" in text
    assert "# TYPE gpu_admission_service_seconds histogram" in text
    assert "w2v_gpu_admission" not in text


# ── HTTP: handler + concurrency ───────────────────────────────────────────────


def _app(lane: str, active: int, waiting: int, wait_s: float, finish: threading.Event) -> FastAPI:
    gate = _gate(lane, active=active, waiting=waiting, wait_s=wait_s, seed=10)
    app = FastAPI()
    install_gpu_busy_handler(app)
    app.state.gate = gate

    def infer() -> str:
        finish.wait(10)
        return "ok"

    @app.post("/infer")
    async def infer_route(request: Request) -> dict[str, str]:
        return {"result": await gate.run(lane, infer, request=request)}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


async def test_concurrency_1_4_5_twenty_requests() -> None:
    lane = _lane()
    finish = threading.Event()
    app = _app(lane, active=1, waiting=4, wait_s=30, finish=finish)
    gate: AdmissionGate = app.state.gate
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=20) as client:
        try:
            t0 = time.monotonic()
            pending = [asyncio.create_task(client.post("/infer")) for _ in range(20)]
            # The 15 surplus requests must come back without waiting on the GPU.
            finished: set[asyncio.Task[httpx.Response]] = set()
            deadline = t0 + 5
            while len(finished) < 15:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                newly, _ = await asyncio.wait(
                    set(pending) - finished, timeout=left, return_when=asyncio.FIRST_COMPLETED
                )
                finished |= newly
            reject_elapsed = time.monotonic() - t0
            # Give any wrongly-admitted extra request time to finish too.
            await asyncio.sleep(0.1)
            done = [t for t in pending if t.done()]
            assert len(done) == 15
            assert reject_elapsed < 2.0

            lane_state = gate.lane(lane)
            assert (lane_state.active, lane_state.waiting) == (1, 4)
            assert _sample("gpu_admission_active", lane=lane) == 1
            assert _sample("gpu_admission_waiting", lane=lane) == 4
            assert _sample("gpu_admission_rejected_total", lane=lane, reason="full") == 15

            for t in done:
                resp = t.result()
                assert resp.status_code == 503
                # 4 waiting -> ceil(10 * 5 / 1) = 50
                assert resp.headers["Retry-After"] == "50"
                assert resp.json() == {"error": "gpu_busy", "lane": lane, "retry_after": 50}

            # /health stays responsive while the lane is saturated.
            for _ in range(5):
                h0 = time.monotonic()
                health = await client.get("/health")
                assert health.status_code == 200
                assert time.monotonic() - h0 < 0.2
            assert (lane_state.active, lane_state.waiting) == (1, 4)
        finally:
            finish.set()
        rest = await asyncio.wait_for(asyncio.gather(*pending), 10)
    assert sorted(r.status_code for r in rest) == [200] * 5 + [503] * 15
    assert gate.lane(lane).active == 0
    assert await gate.shutdown()


async def test_timeout_maps_to_503_and_never_429() -> None:
    lane = _lane()
    finish = threading.Event()
    app = _app(lane, active=1, waiting=4, wait_s=0.1, finish=finish)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=10) as client:
        try:
            first = asyncio.create_task(client.post("/infer"))
            await asyncio.sleep(0.05)
            second = await client.post("/infer")
        finally:
            finish.set()
        assert (await first).status_code == 200
    assert second.status_code == 503
    assert second.headers["Retry-After"] == "10"
    assert second.json() == {"error": "gpu_busy", "lane": lane, "retry_after": 10}
    assert await app.state.gate.shutdown()


async def test_client_gone_maps_to_quiet_499() -> None:
    lane = _lane()
    gate = _gate(lane, 1, 1, 1)
    app = FastAPI()
    install_gpu_busy_handler(app)

    @app.post("/infer")
    async def infer_route() -> dict[str, str]:
        async with gate.slot(lane, request=_FakeRequest(gone=True)):
            return {"never": "reached"}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/infer")
    assert resp.status_code == 499
    assert resp.content == b""
    assert _sample("gpu_admission_rejected_total", lane=lane, reason="full") == 0


# ── shutdown ──────────────────────────────────────────────────────────────────


async def test_shutdown_waits_for_inflight_thread_by_default() -> None:
    lane = _lane()
    gate = _gate(lane, active=1, waiting=2, wait_s=10)
    started = threading.Event()
    finish = threading.Event()

    def blocking() -> str:
        started.set()
        finish.wait(5)
        return "done"

    job = asyncio.create_task(gate.run(lane, blocking))
    assert await asyncio.to_thread(started.wait, 2)
    closer = asyncio.create_task(gate.shutdown())
    await asyncio.sleep(0.05)
    assert not closer.done()  # still waiting on the GPU thread
    finish.set()
    assert await asyncio.wait_for(closer, 2) is True
    assert await job == "done"
    assert gate.lane(lane).active == 0


async def test_shutdown_timeout_returns_false_and_slot_still_follows_thread() -> None:
    lane = _lane()
    gate = _gate(lane, active=1, waiting=2, wait_s=10)
    started = threading.Event()
    finish = threading.Event()

    def blocking() -> None:
        started.set()
        finish.wait(5)

    job = asyncio.create_task(gate.run(lane, blocking))
    assert await asyncio.to_thread(started.wait, 2)
    t0 = time.monotonic()
    assert await gate.shutdown(timeout=0.05) is False
    assert time.monotonic() - t0 < 0.5
    assert gate.lane(lane).active == 1
    # Closed + busy: a newcomer is refused at once, not queued behind the thread.
    t1 = time.monotonic()
    with pytest.raises(GateClosedError):
        async with gate.slot(lane):
            pytest.fail("must not be admitted after shutdown")
    assert time.monotonic() - t1 < 0.05
    assert gate.lane(lane).waiting == 0
    finish.set()
    await asyncio.wait_for(job, 2)
    assert gate.lane(lane).active == 0


async def test_run_and_slot_after_shutdown_raise_gate_closed() -> None:
    lane = _lane()
    gate = _gate(lane, active=1, waiting=2, wait_s=10)
    assert await gate.run(lane, lambda: 1) == 1
    assert await gate.shutdown() is True
    calls: list[int] = []
    with pytest.raises(GateClosedError, match=lane):
        await gate.run(lane, calls.append, 1)
    with pytest.raises(GateClosedError):
        async with gate.slot(lane):
            pytest.fail("must not be admitted after shutdown")
    assert calls == []
    with pytest.raises(GateClosedError):
        _ = gate.lane(lane).executor  # never silently recreated
    assert gate.lane(lane).closed
    assert gate.lane(lane).active == 0
    assert await gate.shutdown() is True  # idempotent


async def test_waiter_queued_at_shutdown_is_refused_on_admission() -> None:
    lane = _lane()
    gate = _gate(lane, active=1, waiting=2, wait_s=10)
    started = threading.Event()
    finish = threading.Event()
    calls: list[int] = []

    def blocking() -> None:
        started.set()
        finish.wait(5)

    job = asyncio.create_task(gate.run(lane, blocking))
    assert await asyncio.to_thread(started.wait, 2)
    queued = asyncio.create_task(gate.run(lane, calls.append, 1))
    await _settle()

    async def queued_slot() -> None:
        async with gate.slot(lane):
            calls.append(2)

    queued2 = asyncio.create_task(queued_slot())
    await _settle()
    assert gate.lane(lane).waiting == 2
    closer = asyncio.create_task(gate.shutdown())
    finish.set()
    with pytest.raises(GateClosedError):
        await asyncio.wait_for(queued, 2)
    with pytest.raises(GateClosedError):
        await asyncio.wait_for(queued2, 2)
    assert await asyncio.wait_for(closer, 2) is True
    await job
    assert calls == []
    assert gate.lane(lane).active == 0
