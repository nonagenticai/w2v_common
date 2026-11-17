"""Shared GPU admission gate (#217).

One :class:`AdmissionGate` per GPU service holds one named :class:`Lane` per GPU
resource. A lane admits at most ``max_active`` requests at once, queues at most
``max_waiting`` more in strict arrival order, and refuses everything else with
:class:`GpuBusy` (mapped to ``503`` + ``Retry-After`` by
:func:`w2v_common.admission.install_gpu_busy_handler`). The gate never emits 429.

Contract (owner ruling 2026-10-15): global first-come-first-served, no per-tenant
fairness at this layer; callers own their own concurrency.

Two entry points:

* ``async with gate.slot(lane, request=req):`` — for AWAITED, NON-THREAD work only
  (e.g. an async client call to a GPU process). The slot is released when the
  ``async with`` body exits, INCLUDING on cancellation — so wrapping
  ``asyncio.to_thread`` / ``run_in_executor`` in ``slot()`` frees the slot while
  the thread is still on the GPU. Thread or in-process GPU work MUST use ``run()``.
* ``await gate.run(lane, fn, *args, request=req)`` — the work is a BLOCKING
  callable and runs in the lane's own ``ThreadPoolExecutor(max_workers=max_active)``,
  never on the event loop. The slot is released when the THREAD finishes, not
  when the awaiting coroutine does: cancelling a coroutine that awaits a thread
  does not stop the thread, and releasing on cancellation would admit the next
  request into a GPU model that is still busy (the transcription_worker_service
  ``_release_after_decode`` lesson).

Disconnect skip: pass the Starlette ``Request`` as ``request=``. Behind any
``BaseHTTPMiddleware`` its ``is_disconnected()`` never turns True, so a service
MUST call :func:`~w2v_common.admission.install_disconnect_probe` on its app (the
probe then wraps the whole built stack); the gate reads it (see ``disconnect.py``).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from collections.abc import AsyncIterator, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Literal

from w2v_common.admission.disconnect import SupportsDisconnect, client_disconnected
from w2v_common.observability.metrics import (
    gpu_admission_active,
    gpu_admission_rejected_total,
    gpu_admission_service_seconds,
    gpu_admission_wait_seconds,
    gpu_admission_waiting,
)

logger = logging.getLogger(__name__)

RejectReason = Literal["full", "timeout"]

RETRY_AFTER_MIN_S = 1
RETRY_AFTER_MAX_S = 120
DEFAULT_SEED_S = 10.0
# Sanity ceilings: a value past these is a typo or an attack on the config, not a
# tuning choice, and a float like ``inf``/``1e308`` would otherwise overflow
# ``math.ceil`` on every rejection (a 500 where the contract promises a 503).
MAX_ACTIVE_LIMIT = 256
MAX_WAITING_LIMIT = 10_000
MAX_SECONDS_LIMIT = 3600.0
# Weight of the newest sample in the service-time EWMA.
EWMA_ALPHA = 0.2


class GpuBusy(Exception):  # noqa: N818 — name fixed by the admission contract
    """The lane refused the request: queue full, or waited past ``max_wait_s``."""

    def __init__(self, lane: str, retry_after: int, reason: RejectReason) -> None:
        super().__init__(f"gpu lane {lane!r} busy ({reason}); retry after {retry_after}s")
        self.lane = lane
        self.retry_after = retry_after
        self.reason: RejectReason = reason


class GateClosedError(RuntimeError):
    """The gate (or lane) was shut down; no new work is admitted."""

    def __init__(self, lane: str) -> None:
        super().__init__(f"gpu lane {lane!r} is shut down; no new work is admitted")
        self.lane = lane


class ClientGone(Exception):  # noqa: N818 — a signal, not an error condition
    """The client disconnected before its job was admitted; the job was skipped.

    Not a rejection: no ``gpu_admission_rejected_total`` increment. The installed
    handler answers with a bodyless 499 that nobody is listening for.
    """

    def __init__(self, lane: str) -> None:
        super().__init__(f"client disconnected before admission to gpu lane {lane!r}")
        self.lane = lane


@dataclass(frozen=True, slots=True)
class LaneConfig:
    """Limits for one lane. Code passes defaults; env overrides per field."""

    max_active: int
    max_waiting: int
    max_wait_s: float
    seed_s: float = DEFAULT_SEED_S

    def __post_init__(self) -> None:
        if not 1 <= self.max_active <= MAX_ACTIVE_LIMIT:
            raise ValueError(f"max_active must be in 1..{MAX_ACTIVE_LIMIT}, got {self.max_active}")
        if not 0 <= self.max_waiting <= MAX_WAITING_LIMIT:
            raise ValueError(
                f"max_waiting must be in 0..{MAX_WAITING_LIMIT}, got {self.max_waiting}"
            )
        for field in ("max_wait_s", "seed_s"):
            value = getattr(self, field)
            # ``math.isfinite`` first: ``nan`` compares False against everything.
            if not (math.isfinite(value) and 0 < value <= MAX_SECONDS_LIMIT):
                raise ValueError(
                    f"{field} must be finite and in (0, {MAX_SECONDS_LIMIT:g}], got {value}"
                )

    @classmethod
    def from_env(
        cls,
        lane: str,
        default: LaneConfig,
        environ: Mapping[str, str] | None = None,
    ) -> LaneConfig:
        """``default`` with ``GPU_LANE_<NAME>_MAX_ACTIVE|_MAX_WAITING|_MAX_WAIT_S|_SEED_S``
        applied. A set-but-unparseable value raises (fail loud at startup)."""
        env = os.environ if environ is None else environ
        prefix = env_prefix(lane)
        cfg = default
        for suffix, field, conv in (
            ("MAX_ACTIVE", "max_active", int),
            ("MAX_WAITING", "max_waiting", int),
            ("MAX_WAIT_S", "max_wait_s", float),
            ("SEED_S", "seed_s", float),
        ):
            key = f"{prefix}{suffix}"
            raw = env.get(key)
            if raw is None or raw.strip() == "":
                continue
            try:
                value = conv(raw.strip())
            except ValueError as exc:
                raise ValueError(f"{key}={raw!r} is not a valid {conv.__name__}") from exc
            try:
                cfg = replace(cfg, **{field: value})
            except ValueError as exc:
                raise ValueError(f"{key}={raw!r}: {exc}") from exc
        return cfg


def env_prefix(lane: str) -> str:
    """``"tts-main"`` -> ``"GPU_LANE_TTS_MAIN_"``."""
    return f"GPU_LANE_{re.sub(r'[^A-Za-z0-9]', '_', lane).upper()}_"


class Lane:
    """One GPU resource: semaphore + bounded FIFO waiters + EWMA service time."""

    def __init__(self, name: str, config: LaneConfig) -> None:
        self.name = name
        self.config = config
        self._sem = asyncio.Semaphore(config.max_active)
        self._waiting = 0
        self._active = 0
        self._ewma = config.seed_s
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False
        self._inflight: set[asyncio.Future[None]] = set()
        gpu_admission_active.labels(lane=name).set(0)
        gpu_admission_waiting.labels(lane=name).set(0)
        for reason in ("full", "timeout"):
            gpu_admission_rejected_total.labels(lane=name, reason=reason)

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return self._waiting

    @property
    def ewma_s(self) -> float:
        return self._ewma

    def retry_after(self, position: int) -> int:
        """ceil(ewma * (position+1) / max_active), clamped to 1..120 seconds."""
        raw = math.ceil(self._ewma * (position + 1) / self.config.max_active)
        return max(RETRY_AFTER_MIN_S, min(RETRY_AFTER_MAX_S, raw))

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def executor(self) -> ThreadPoolExecutor:
        if self._closed:
            raise GateClosedError(self.name)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_active,
                thread_name_prefix=f"gpu-lane-{self.name}",
            )
        return self._executor

    def _reject(self, reason: RejectReason, position: int) -> GpuBusy:
        gpu_admission_rejected_total.labels(lane=self.name, reason=reason).inc()
        busy = GpuBusy(self.name, self.retry_after(position), reason)
        logger.warning(
            "gpu_admission_rejected lane=%s reason=%s active=%d waiting=%d retry_after=%d",
            self.name,
            reason,
            self._active,
            self._waiting,
            busy.retry_after,
        )
        return busy

    async def acquire(self, request: SupportsDisconnect | None = None) -> float:
        """Take a slot or raise. Returns the monotonic admission time.

        The caller owns exactly one :meth:`release` after a successful return.
        """
        if self._closed:
            raise GateClosedError(self.name)
        arrival = time.monotonic()
        # ``Semaphore.locked()`` is True while any un-cancelled waiter exists, so a
        # newcomer can never overtake the queue through this fast path.
        if self._sem.locked():
            if self._waiting >= self.config.max_waiting:
                raise self._reject("full", self._waiting)
            self._waiting += 1
            gpu_admission_waiting.labels(lane=self.name).set(self._waiting)
            try:
                remaining = self.config.max_wait_s - (time.monotonic() - arrival)
                # asyncio.Semaphore wakes waiters in FIFO order; wait_for's
                # cancellation of a woken-but-timed-out acquire hands the permit on.
                await asyncio.wait_for(self._sem.acquire(), timeout=max(remaining, 0.0))
            except TimeoutError:
                self._waiting -= 1
                gpu_admission_waiting.labels(lane=self.name).set(self._waiting)
                gpu_admission_wait_seconds.labels(lane=self.name).observe(
                    time.monotonic() - arrival
                )
                raise self._reject("timeout", self._waiting) from None
            except BaseException:
                self._waiting -= 1
                gpu_admission_waiting.labels(lane=self.name).set(self._waiting)
                raise
            self._waiting -= 1
            gpu_admission_waiting.labels(lane=self.name).set(self._waiting)
        else:
            await self._sem.acquire()  # does not block: not locked, no waiters

        admitted = time.monotonic()
        gpu_admission_wait_seconds.labels(lane=self.name).observe(admitted - arrival)
        self._active += 1
        gpu_admission_active.labels(lane=self.name).set(self._active)

        if self._closed:  # shut down while this request was queued
            self.release(admitted, record=False)
            raise GateClosedError(self.name)

        if request is not None:
            try:
                gone = await client_disconnected(request)
            except BaseException:
                self.release(admitted, record=False)
                raise
            if gone:
                self.release(admitted, record=False)
                logger.info("gpu_admission_client_gone lane=%s", self.name)
                raise ClientGone(self.name)
        return admitted

    def release(self, admitted: float, *, record: bool = True) -> None:
        """Return a slot taken by :meth:`acquire`. Must run on the loop thread."""
        if record:
            service = time.monotonic() - admitted
            gpu_admission_service_seconds.labels(lane=self.name).observe(service)
            self._ewma = EWMA_ALPHA * service + (1 - EWMA_ALPHA) * self._ewma
        self._active -= 1
        gpu_admission_active.labels(lane=self.name).set(self._active)
        self._sem.release()

    def track(self, fut: asyncio.Future[None]) -> None:
        """Register a job's completion signal (set after its slot is released)."""
        self._inflight.add(fut)
        fut.add_done_callback(self._inflight.discard)

    async def shutdown(self, *, timeout: float | None = None) -> bool:  # noqa: ASYNC109 — a deadline that RETURNS False (threads still running) is the contract; asyncio.timeout would cancel and lose that answer
        """Stop admitting, wait for in-flight threads, then stop the executor.

        Returns True when every in-flight job finished, False when ``timeout``
        expired first (those threads keep running; their slots are released when
        they end). Idempotent. Afterwards ``acquire``/``run`` raise :class:`GateClosedError`.
        """
        self._closed = True
        pending = set(self._inflight)
        finished = True
        if pending:
            _, still = await asyncio.wait(pending, timeout=timeout)
            finished = not still
        if self._executor is not None:
            # Never blocks: either every job is done, or we chose not to wait longer.
            self._executor.shutdown(wait=False, cancel_futures=True)
        return finished


class AdmissionGate:
    """Named GPU lanes for one service.

    ``AdmissionGate({"whisper": LaneConfig(max_active=1, max_waiting=64, max_wait_s=120)})``
    — the mapping holds CODE defaults; ``GPU_LANE_WHISPER_*`` env overrides them.
    """

    def __init__(
        self,
        lanes: Mapping[str, LaneConfig],
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if not lanes:
            raise ValueError("AdmissionGate needs at least one lane")
        self._lanes: dict[str, Lane] = {
            name: Lane(name, LaneConfig.from_env(name, default, environ))
            for name, default in lanes.items()
        }

    def lane(self, name: str) -> Lane:
        try:
            return self._lanes[name]
        except KeyError:
            raise KeyError(f"unknown gpu lane {name!r}; known: {sorted(self._lanes)}") from None

    @property
    def lanes(self) -> Mapping[str, Lane]:
        return dict(self._lanes)

    @asynccontextmanager
    async def slot(
        self, lane: str, request: SupportsDisconnect | None = None
    ) -> AsyncIterator[Lane]:
        """Hold one slot of ``lane`` for the body. Raises GpuBusy / ClientGone / GateClosedError.

        For awaited NON-THREAD work only. The slot is released when the body exits,
        including on cancellation, so a thread started inside the body (``to_thread``,
        ``run_in_executor``) can outlive its slot. Thread / GPU work: use :meth:`run`.
        """
        ln = self.lane(lane)
        admitted = await ln.acquire(request)
        try:
            yield ln
        finally:
            ln.release(admitted)

    async def run[*Ts, T](
        self,
        lane: str,
        fn: Callable[[*Ts], T],
        *args: *Ts,
        request: SupportsDisconnect | None = None,
    ) -> T:
        """Run blocking ``fn(*args)`` in the lane's executor under a slot.

        Keyword arguments go through ``functools.partial`` (``request`` is the
        gate's own keyword).

        The slot is released by the thread future's done-callback, so it stays
        held until the thread actually finishes even if this coroutine is
        cancelled (client disconnect, handler timeout). The await goes through
        ``asyncio.shield`` so cancellation never reaches the future.
        """
        ln = self.lane(lane)
        admitted = await ln.acquire(request)
        loop = asyncio.get_running_loop()
        try:
            fut: asyncio.Future[T] = loop.run_in_executor(ln.executor, fn, *args)
            released: asyncio.Future[None] = loop.create_future()
            ln.track(released)
        except BaseException:
            ln.release(admitted, record=False)
            raise

        def _done(f: asyncio.Future[T]) -> None:
            try:
                # Retrieve the exception so an abandoned (cancelled-awaiter) future
                # does not log "exception was never retrieved".
                if not f.cancelled():
                    f.exception()
            finally:
                ln.release(admitted)
                released.set_result(None)

        fut.add_done_callback(_done)
        return await asyncio.shield(fut)

    async def shutdown(self, *, timeout: float | None = None) -> bool:  # noqa: ASYNC109 — a deadline that RETURNS False (threads still running) is the contract; asyncio.timeout would cancel and lose that answer
        """Shut every lane down (see :meth:`Lane.shutdown`); True if all drained.

        ``timeout`` bounds the WHOLE call, not each lane.
        """
        results = await asyncio.gather(
            *(ln.shutdown(timeout=timeout) for ln in self._lanes.values())
        )
        return all(results)
