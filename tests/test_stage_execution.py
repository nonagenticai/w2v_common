"""Tests for `w2v_common.stages.execution`."""

from __future__ import annotations

import asyncio

import pytest

from w2v_common.stages import (
    StageRetriesExhaustedError,
    StageTimeoutError,
    stage_call,
    stage_gather,
)


class TestStageCall:
    async def test_returns_result_on_first_success(self) -> None:
        async def ok() -> str:
            return "ok"

        result = await stage_call(ok, timeout_seconds=1.0, retries=0)
        assert result == "ok"

    async def test_retries_on_retryable_exception_then_succeeds(self) -> None:
        attempts: list[int] = []

        async def flaky() -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("transient")
            return "ok"

        result = await stage_call(
            flaky,
            timeout_seconds=1.0,
            retries=2,
            backoff_initial=0.001,
            backoff_factor=1.0,
            backoff_jitter=0.0,
        )
        assert result == "ok"
        assert len(attempts) == 3

    async def test_exhausts_retries_and_raises_with_last_error(self) -> None:
        attempts: list[int] = []

        async def always_fails() -> str:
            attempts.append(1)
            raise ValueError(f"nope {len(attempts)}")

        with pytest.raises(StageRetriesExhaustedError) as exc_info:
            await stage_call(
                always_fails,
                timeout_seconds=1.0,
                retries=2,
                backoff_initial=0.001,
                backoff_factor=1.0,
                backoff_jitter=0.0,
                stage_name="bad_stage",
            )
        assert exc_info.value.attempts == 3
        assert exc_info.value.stage_name == "bad_stage"
        assert isinstance(exc_info.value.last_error, ValueError)
        assert "nope 3" in str(exc_info.value.last_error)
        assert len(attempts) == 3

    async def test_raises_stage_timeout_after_all_attempts_time_out(self) -> None:
        async def hangs() -> None:
            await asyncio.sleep(10.0)

        with pytest.raises(StageTimeoutError) as exc_info:
            await stage_call(
                hangs,
                timeout_seconds=1.0,
                retries=1,
                backoff_initial=0.001,
                backoff_factor=1.0,
                backoff_jitter=0.0,
                stage_name="hangy",
            )
        assert "hangy" in str(exc_info.value)
        assert "all 2 attempts timed out" in str(exc_info.value)

    async def test_non_retryable_exception_propagates_immediately(self) -> None:
        attempts: list[int] = []

        async def raises_keyerror() -> str:
            attempts.append(1)
            raise KeyError("programmer error")

        with pytest.raises(KeyError):
            await stage_call(
                raises_keyerror,
                timeout_seconds=1.0,
                retries=5,
                backoff_initial=0.001,
                backoff_factor=1.0,
                backoff_jitter=0.0,
                retry_on=(ValueError,),
            )
        assert len(attempts) == 1

    async def test_validates_timeout_lower_bound(self) -> None:
        async def fn() -> None:
            return None

        with pytest.raises(ValueError, match=r"timeout_seconds must be >= 1\.0s"):
            await stage_call(fn, timeout_seconds=0.5, retries=0)

    async def test_validates_timeout_upper_bound(self) -> None:
        async def fn() -> None:
            return None

        with pytest.raises(ValueError, match=r"timeout_seconds must be <= 3600\.0s"):
            await stage_call(fn, timeout_seconds=4000.0, retries=0)

    async def test_validates_retries_lower_bound(self) -> None:
        async def fn() -> None:
            return None

        with pytest.raises(ValueError, match="retries must be >= 0"):
            await stage_call(fn, timeout_seconds=1.0, retries=-1)

    async def test_validates_retries_upper_bound(self) -> None:
        async def fn() -> None:
            return None

        with pytest.raises(ValueError, match="retries must be <= 10"):
            await stage_call(fn, timeout_seconds=1.0, retries=99)


class TestStageGather:
    async def test_returns_results_in_order(self) -> None:
        async def square(x: int) -> int:
            return x * x

        result = await stage_gather(
            [1, 2, 3, 4, 5],
            square,
            concurrency=2,
            per_item_timeout_seconds=1.0,
            per_item_retries=0,
        )
        assert result == [1, 4, 9, 16, 25]

    async def test_empty_items_returns_empty_list(self) -> None:
        async def identity(x: int) -> int:
            return x

        result = await stage_gather(
            [],
            identity,
            concurrency=2,
            per_item_timeout_seconds=1.0,
            per_item_retries=0,
        )
        assert result == []

    async def test_respects_concurrency_limit(self) -> None:
        in_flight = 0
        max_in_flight = 0

        async def track(x: int) -> int:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return x

        await stage_gather(
            list(range(10)),
            track,
            concurrency=3,
            per_item_timeout_seconds=1.0,
            per_item_retries=0,
        )
        assert max_in_flight == 3

    async def test_one_failure_fails_entire_stage(self) -> None:
        async def fn(x: int) -> int:
            if x == 5:
                raise ValueError("bad item")
            await asyncio.sleep(0.001)
            return x

        with pytest.raises(StageRetriesExhaustedError):
            await stage_gather(
                list(range(10)),
                fn,
                concurrency=2,
                per_item_timeout_seconds=1.0,
                per_item_retries=0,
                backoff_initial=0.001,
                backoff_factor=1.0,
                backoff_jitter=0.0,
                stage_name="bad_gather",
            )

    async def test_one_timeout_fails_entire_stage(self) -> None:
        async def fn(x: int) -> int:
            if x == 3:
                await asyncio.sleep(10.0)
            return x

        with pytest.raises(StageTimeoutError):
            await stage_gather(
                list(range(5)),
                fn,
                concurrency=2,
                per_item_timeout_seconds=1.0,
                per_item_retries=0,
                backoff_initial=0.001,
                backoff_factor=1.0,
                backoff_jitter=0.0,
            )

    async def test_parallel_is_faster_than_serial(self) -> None:
        async def slow(x: int) -> int:
            await asyncio.sleep(0.05)
            return x

        loop = asyncio.get_event_loop()

        start = loop.time()
        await stage_gather(
            list(range(8)),
            slow,
            concurrency=4,
            per_item_timeout_seconds=1.0,
            per_item_retries=0,
        )
        parallel_elapsed = loop.time() - start

        # Strict serial would be ~0.4s; concurrency=4 should be ~0.1s.
        # Generous 0.25s ceiling absorbs CI scheduler jitter.
        assert parallel_elapsed < 0.25, f"parallel took {parallel_elapsed:.3f}s"

    async def test_per_item_retry_succeeds_on_third_attempt(self) -> None:
        per_item_attempts: dict[int, int] = {}

        async def flaky(x: int) -> int:
            per_item_attempts[x] = per_item_attempts.get(x, 0) + 1
            if per_item_attempts[x] < 3:
                raise ValueError(f"transient {x}")
            return x * 10

        result = await stage_gather(
            [1, 2, 3],
            flaky,
            concurrency=2,
            per_item_timeout_seconds=1.0,
            per_item_retries=2,
            backoff_initial=0.001,
            backoff_factor=1.0,
            backoff_jitter=0.0,
        )
        assert result == [10, 20, 30]
        assert all(per_item_attempts[i] == 3 for i in (1, 2, 3))

    async def test_validates_concurrency_lower_bound(self) -> None:
        async def fn(x: int) -> int:
            return x

        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            await stage_gather(
                [1],
                fn,
                concurrency=0,
                per_item_timeout_seconds=1.0,
            )

    async def test_validates_concurrency_upper_bound(self) -> None:
        async def fn(x: int) -> int:
            return x

        with pytest.raises(ValueError, match="concurrency must be <= 64"):
            await stage_gather(
                [1],
                fn,
                concurrency=100,
                per_item_timeout_seconds=1.0,
            )


class TestStageGatherCancelsSiblingsOnFailure:
    """Defect 1 (0.57.0): a failing item must not leave paid siblings running.

    ``stage_gather``'s one live consumer fans out BILLED TTS calls per
    audiobook chapter. Bare ``asyncio.gather`` propagates the first
    exception immediately but leaves the outstanding tasks running
    unsupervised, so their (already-charged) results are produced after the
    caller has moved on and are then discarded.
    """

    async def test_siblings_are_cancelled_when_one_item_fails(self) -> None:
        finished: list[int] = []
        cancelled: list[int] = []

        async def fn(x: int) -> int:
            if x == 0:
                raise ValueError("chapter 0 is poison")
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                cancelled.append(x)
                raise
            finished.append(x)
            return x

        with pytest.raises(StageRetriesExhaustedError):
            await stage_gather(
                [0, 1, 2, 3],
                fn,
                concurrency=4,
                per_item_timeout_seconds=5.0,
                per_item_retries=0,
            )

        # Give any unsupervised sibling the wall-clock it needs to complete.
        await asyncio.sleep(1.0)

        assert finished == [], (
            f"siblings {finished} ran to completion after the gather raised — "
            "billed work executed with its result discarded"
        )
        assert sorted(cancelled) == [1, 2, 3], (
            f"expected siblings 1,2,3 to observe CancelledError, saw {cancelled}"
        )

    async def test_completed_sibling_results_are_surfaced_on_the_error(self) -> None:
        """A sibling that ALREADY finished (and was already billed) must have its
        result reachable by the caller, not silently dropped."""

        async def fn(x: int) -> str:
            if x == 2:
                await asyncio.sleep(0.2)
                raise ValueError("chapter 2 failed after siblings settled")
            return f"audio-{x}"

        with pytest.raises(StageRetriesExhaustedError) as excinfo:
            await stage_gather(
                [0, 1, 2],
                fn,
                concurrency=3,
                per_item_timeout_seconds=5.0,
                per_item_retries=0,
            )

        assert excinfo.value.partial_results == {0: "audio-0", 1: "audio-1"}, (
            "completed (billed) item outputs must be surfaced on the raised error "
            "so the caller can record or refund them"
        )


def test_retries_exhausted_message_never_ends_on_an_empty_colon() -> None:
    import httpx

    from w2v_common.stages.errors import StageRetriesExhaustedError

    err = StageRetriesExhaustedError("scan", 2, httpx.ReadTimeout(""))
    assert str(err).endswith("ReadTimeout: raised with no error message")

    err = StageRetriesExhaustedError("scan", 2, ValueError("boom"))
    assert str(err).endswith("ValueError: boom")
