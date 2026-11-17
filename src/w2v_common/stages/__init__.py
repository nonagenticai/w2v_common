"""Stage-execution primitives for w2v orchestrator handlers.

Stage handlers in `w2v_orchestrator/src/services/stages/` typically either
(a) make one async call to a downstream service client, or (b) loop over N
items calling a client per item. Both patterns already inherit retry +
per-call timeout from `BaseServiceClient` but lack:

- bounded concurrency for the loop case (e.g. `lesson_narrate` serialises
  TTS calls today; a 10-paragraph script becomes a 10x sequential wait
  even when TTS can serve in parallel),
- a stage-level wall-clock budget so a runaway downstream cannot stall a
  run beyond the stage's expected duration even if individual client
  calls each meet their own timeout,
- explicit typed failure modes (`StageTimeoutError`,
  `StageRetriesExhaustedError`) the run dispatcher can map to terminal
  run statuses without string-matching exception messages.

This module is intentionally tiny: two callables and two error types.
Anything more belongs in the stage handler or in `BaseServiceClient`.
"""

from w2v_common.stages.errors import (
    StageError,
    StageRetriesExhaustedError,
    StageTimeoutError,
)
from w2v_common.stages.execution import stage_call, stage_gather

__all__ = [
    "StageError",
    "StageRetriesExhaustedError",
    "StageTimeoutError",
    "stage_call",
    "stage_gather",
]
