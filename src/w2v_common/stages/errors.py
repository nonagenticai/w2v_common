"""Typed errors raised by `w2v_common.stages` helpers."""

from __future__ import annotations


class StageError(RuntimeError):
    """Base class for stage-execution failures.

    Attributes:
        partial_results: Index -> result for the items that had already
            completed successfully when the stage failed. Populated by
            ``stage_gather``; empty for single-item ``stage_call``. A
            fan-out over BILLED work (e.g. per-chapter TTS) must be able
            to see which units were actually produced and charged, so the
            caller can record or refund them instead of silently dropping
            paid output on the floor.
    """

    partial_results: dict[int, object]

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.partial_results = {}


class StageTimeoutError(StageError):
    """A stage-level wall-clock budget was exceeded on every attempt.

    Distinct from the underlying client timeout: the client may have
    retried within its own budget; this error fires when the OUTER
    stage-level budget (across all attempts) is also exceeded.
    """


class StageRetriesExhaustedError(StageError):
    """Stage retries exhausted after retry-eligible failures.

    Attributes:
        stage_name: Identifier passed to `stage_call` / `stage_gather`
            for log correlation.
        attempts: Total attempts made (= retries + 1).
        last_error: The last exception caught before retries ran out.
    """

    def __init__(
        self,
        stage_name: str,
        attempts: int,
        last_error: BaseException,
    ) -> None:
        self.stage_name = stage_name
        self.attempts = attempts
        self.last_error = last_error
        # str() of several httpx errors (ReadTimeout, ConnectTimeout) is empty,
        # which left the user-facing message ending in a bare "ReadTimeout:".
        detail = str(last_error).strip() or "raised with no error message"
        super().__init__(
            f"stage {stage_name!r}: retries exhausted after {attempts} attempts: "
            f"{type(last_error).__name__}: {detail}"
        )
