"""Stage-execution helpers: ``stage_call`` and ``stage_gather``."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Awaitable, Callable, Sequence

from w2v_common.stages.errors import (
    StageError,
    StageRetriesExhaustedError,
    StageTimeoutError,
)

logger = logging.getLogger(__name__)

_MIN_TIMEOUT_SECONDS = 1.0
_MAX_TIMEOUT_SECONDS = 3600.0
_MAX_RETRIES = 10
_MAX_CONCURRENCY = 64


def _validate_timeout(value: float, name: str) -> float:
    if value < _MIN_TIMEOUT_SECONDS:
        raise ValueError(f"{name} must be >= {_MIN_TIMEOUT_SECONDS}s, got {value}")
    if value > _MAX_TIMEOUT_SECONDS:
        raise ValueError(f"{name} must be <= {_MAX_TIMEOUT_SECONDS}s, got {value}")
    return value


def _validate_retries(value: int) -> int:
    if value < 0:
        raise ValueError(f"retries must be >= 0, got {value}")
    if value > _MAX_RETRIES:
        raise ValueError(f"retries must be <= {_MAX_RETRIES}, got {value}")
    return value


def _backoff_delay(
    attempt: int,
    initial: float,
    factor: float,
    jitter: float,
) -> float:
    """Compute exponential backoff with full-jitter scaling.

    ``attempt`` is 1-indexed. ``jitter`` is the +/- fraction applied to
    the computed delay; ``0.25`` means the actual sleep is uniformly
    sampled from ``[0.75x, 1.25x]`` of the nominal delay.

    Uses ``secrets.SystemRandom`` rather than ``random.uniform`` so the
    jitter is cryptographically-strength uniform; this matters when
    many parallel runs would otherwise synchronise their backoff and
    re-stampede the downstream after a recovery.
    """
    base = initial * (factor ** (attempt - 1))
    if jitter > 0.0:
        rng = secrets.SystemRandom()
        base *= 1.0 - jitter + rng.random() * 2.0 * jitter
    return max(0.0, base)


async def stage_call[R](
    coro_factory: Callable[[], Awaitable[R]],
    *,
    timeout_seconds: float,
    retries: int = 2,
    backoff_initial: float = 1.0,
    backoff_factor: float = 2.0,
    backoff_jitter: float = 0.25,
    stage_name: str = "stage_call",
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> R:
    """Execute one async call with a per-attempt timeout + retries.

    The caller passes a *factory* (zero-arg callable returning a fresh
    coroutine), not a coroutine instance, because retries need to
    construct a fresh coroutine each attempt — a coroutine can only be
    awaited once.

    Args:
        coro_factory: Zero-arg callable returning a fresh awaitable.
        timeout_seconds: Per-attempt wall-clock budget. All attempts
            timing out raises ``StageTimeoutError``.
        retries: Retries after the first attempt. ``retries=2`` → up to
            3 attempts.
        backoff_initial: Seconds before the first retry.
        backoff_factor: Multiplier applied per subsequent retry.
        backoff_jitter: +/- fraction of jitter applied to each backoff.
        stage_name: Identifier embedded in log lines.
        retry_on: Exception types that trigger a retry. Anything not in
            this tuple propagates immediately on its first occurrence
            (use this to keep programming errors loud).

    Returns:
        Whatever the awaited coroutine returns.

    Raises:
        StageTimeoutError: Every attempt timed out.
        StageRetriesExhaustedError: Every attempt raised a retry-eligible
            exception (the last one is attached as ``last_error``).
        ValueError: Invalid ``timeout_seconds`` or ``retries``.
    """
    timeout_seconds = _validate_timeout(timeout_seconds, "timeout_seconds")
    retries = _validate_retries(retries)
    attempts = retries + 1

    for attempt in range(1, attempts + 1):
        try:
            return await asyncio.wait_for(coro_factory(), timeout=timeout_seconds)
        except TimeoutError as exc:
            logger.warning(
                "stage_call %s: attempt %d/%d timed out after %.1fs",
                stage_name,
                attempt,
                attempts,
                timeout_seconds,
            )
            if attempt == attempts:
                raise StageTimeoutError(
                    f"stage {stage_name!r}: all {attempts} attempts timed out "
                    f"(per-attempt budget {timeout_seconds:.1f}s)"
                ) from exc
        except retry_on as exc:
            logger.warning(
                "stage_call %s: attempt %d/%d raised %s: %s",
                stage_name,
                attempt,
                attempts,
                type(exc).__name__,
                exc,
            )
            if attempt == attempts:
                raise StageRetriesExhaustedError(
                    stage_name=stage_name,
                    attempts=attempts,
                    last_error=exc,
                ) from exc

        delay = _backoff_delay(attempt, backoff_initial, backoff_factor, backoff_jitter)
        logger.debug(
            "stage_call %s: sleeping %.3fs before attempt %d",
            stage_name,
            delay,
            attempt + 1,
        )
        await asyncio.sleep(delay)

    raise AssertionError(  # pragma: no cover -- loop above always returns or raises
        f"stage_call {stage_name!r}: unreachable retry-exhausted fallthrough"
    )


async def stage_gather[T, R](
    items: Sequence[T],
    fn: Callable[[T], Awaitable[R]],
    *,
    concurrency: int,
    per_item_timeout_seconds: float,
    per_item_retries: int = 2,
    backoff_initial: float = 1.0,
    backoff_factor: float = 2.0,
    backoff_jitter: float = 0.25,
    stage_name: str = "stage_gather",
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> list[R]:
    """Run ``fn(item)`` for each item with bounded concurrency.

    Stage semantics: a stage either succeeds with full output or fails
    entirely. If any single item exhausts retries or times out, the
    entire gather raises and NO partial output is returned — passing
    partial output to a downstream stage corrupts the ``prior_output``
    contract.

    On failure the still-outstanding sibling tasks are **cancelled and
    awaited** before the error propagates, so no work continues
    unsupervised after the caller has moved on. Items that had already
    completed are attached to the raised
    :class:`~w2v_common.stages.errors.StageError` as ``partial_results``
    (index -> result): they represent work that was already performed and
    (for billed fan-outs) already charged, so the caller can record or
    refund them rather than discard them silently.

    Args:
        items: Input items. Order is preserved in the output list.
        fn: Async callable accepting one item and returning one result.
        concurrency: Maximum simultaneous in-flight calls.
        per_item_timeout_seconds: Per-attempt timeout for each item.
        per_item_retries: Retries per item.
        backoff_*: See ``stage_call``.
        stage_name: Identifier embedded in log lines (per-item index
            appended).
        retry_on: Exception types that trigger a retry per item.

    Returns:
        Results in the same order as ``items``.

    Raises:
        StageTimeoutError: One or more items timed out across all attempts.
        StageRetriesExhaustedError: One or more items exhausted retries.
        ValueError: Invalid ``concurrency`` or budget arguments.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if concurrency > _MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be <= {_MAX_CONCURRENCY}, got {concurrency}")
    per_item_timeout_seconds = _validate_timeout(
        per_item_timeout_seconds, "per_item_timeout_seconds"
    )
    per_item_retries = _validate_retries(per_item_retries)

    if not items:
        return []

    semaphore = asyncio.Semaphore(concurrency)
    results: dict[int, R] = {}

    async def _run_one(index: int, item: T) -> None:
        async with semaphore:
            results[index] = await stage_call(
                lambda: fn(item),
                timeout_seconds=per_item_timeout_seconds,
                retries=per_item_retries,
                backoff_initial=backoff_initial,
                backoff_factor=backoff_factor,
                backoff_jitter=backoff_jitter,
                stage_name=f"{stage_name}[{index}]",
                retry_on=retry_on,
            )

    tasks = [
        asyncio.create_task(_run_one(i, item), name=f"{stage_name}[{i}]")
        for i, item in enumerate(items)
    ]
    try:
        await asyncio.gather(*tasks)
    except BaseException as exc:
        # ``asyncio.gather`` propagates the FIRST exception immediately but
        # leaves the outstanding tasks running unsupervised — for a fan-out
        # over billed work (per-chapter TTS) that means paid calls continue,
        # complete, and have their results thrown away with no record and no
        # refund. Cancel the siblings and await their cancellation before
        # propagating so nothing outlives this call.
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if isinstance(exc, StageError):
            # Items that already finished were already charged; surface them
            # rather than dropping them.
            settled: dict[int, object] = dict(results)
            exc.partial_results = settled
        raise
    return [results[i] for i in range(len(items))]
