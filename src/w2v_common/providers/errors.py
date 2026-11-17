"""Typed failures of run provider-slot resolution and tenant-provider calls.

Every message is safe to show a user and to log: no secret, no URL userinfo, no
request body. ``slot`` names the slot so a stage can attribute the failure.
"""

from __future__ import annotations

from typing import Final

#: Machine code the orchestrator's 403 carries (``src/services/providers/errors.py``).
BYO_PROVIDER_REQUIRED: Final = "byo_provider_required"
#: Machine code of the orchestrator's 409.
PROVIDER_UNAVAILABLE: Final = "provider_unavailable"
#: 403: the request did not arrive on the cell lane.
CELL_LANE_REQUIRED: Final = "cell_lane_required"
#: 422: not a resolvable registry slot.
UNKNOWN_SLOT: Final = "unknown_slot"
#: 404: the run does not exist for the signed tenant.
RUN_NOT_FOUND: Final = "run_not_found"


class ProviderSlotError(RuntimeError):
    """Base: the run's provider for ``slot`` could not be resolved or used."""

    #: True when the same call may succeed later (orchestrator rolling, provider 5xx).
    transient: bool = False

    def __init__(self, slot: str, message: str, *, status: int | None = None) -> None:
        self.slot = slot
        self.status = status
        super().__init__(message)


class ProviderSlotRequiredError(ProviderSlotError):
    """403 ``byo_provider_required``: the tenant must bring its own provider for ``slot``.

    Never answered with the platform default — the licence's ``byo_mode`` forbids it.
    """

    code: str = BYO_PROVIDER_REQUIRED

    def __init__(self, slot: str, message: str | None = None) -> None:
        super().__init__(
            slot, message or f"Configure your own {slot} provider to continue", status=403
        )

    @property
    def failure_detail(self) -> dict[str, str]:
        """The orchestrator's ``runs.failure_detail`` wire shape."""
        return {"code": self.code, "modality": self.slot, "message": str(self)}


class ProviderSlotUnavailableError(ProviderSlotError):
    """The slot exists but cannot be used now (409, 404 wrong-tenant run, 5xx, transport,
    malformed answer, no cell signer, no tenant), or the tenant provider failed.

    Never answered with the platform default: a tenant that configured its own
    provider must not silently get ours.
    """

    def __init__(
        self,
        slot: str,
        reason: str,
        *,
        status: int | None = None,
        transient: bool = False,
    ) -> None:
        self.reason = reason
        self.transient = transient
        super().__init__(slot, f"Provider '{slot}' is unavailable: {reason}", status=status)


class ProviderSlotConfigError(ProviderSlotError):
    """A deployment/programming fault no retry fixes: the orchestrator refused the
    caller's lane (403 ``cell_lane_required`` — CELL_ALLOWED_CALLERS / signing) or the
    slot name (422 ``unknown_slot``). Never negative-cached as "unavailable"."""

    def __init__(self, slot: str, code: str, *, status: int) -> None:
        self.code = code
        super().__init__(
            slot,
            f"Provider '{slot}' cannot be resolved from this service: orchestrator "
            f"answered {status} {code}",
            status=status,
        )


class ProviderHopRefusedError(ProviderSlotError):
    """The current inbound request arrived through a tenant-provider hop (loop guard)."""

    def __init__(self, slot: str) -> None:
        super().__init__(
            slot,
            f"Provider '{slot}' refused: this request was itself sent by a tenant provider "
            "adapter (X-W2V-Provider-Hop); resolving a slot here would loop",
        )


__all__ = [
    "BYO_PROVIDER_REQUIRED",
    "CELL_LANE_REQUIRED",
    "PROVIDER_UNAVAILABLE",
    "RUN_NOT_FOUND",
    "UNKNOWN_SLOT",
    "ProviderHopRefusedError",
    "ProviderSlotConfigError",
    "ProviderSlotError",
    "ProviderSlotRequiredError",
    "ProviderSlotUnavailableError",
]
