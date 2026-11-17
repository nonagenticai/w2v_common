"""THE GUARD: no destination may exist without a declared tenancy.

``tests/test_auth_outbound.py`` asserts properties of names that are ALREADY in
``METERED_DESTINATION_SERVICES``. That shape cannot detect an OMISSION — and an
omission is the only failure mode that has actually happened here, three times
(``video_production``, ``html_render``, and nine more measured 2026-09-13),
each found by accident.

So this file checks the complement: walk the destination registry
(``discovery._SERVICES``, which is what every orchestrator client resolves its
base URL through — ``LicensedServiceClient`` calls
``service_url(discovery_name or service_name)``) and fail on any name that is
neither classified ``DELEGATING`` nor EXPLICITLY declared otherwise. Absence is
an error. There is no "skip" list: the only way past this guard is to write
down what the destination does with a tenant.
"""

from __future__ import annotations

import pytest

from w2v_common.auth.destination_tenancy import (
    DESTINATION_TENANCY,
    DestinationTenancy,
    TenancyDeclaration,
    names_with_tenancy,
)
from w2v_common.auth.outbound import METERED_DESTINATION_SERVICES
from w2v_common.discovery import _SERVICES

_HOW_TO_FIX = (
    "Add it to w2v_common.auth.destination_tenancy.DESTINATION_TENANCY with an "
    "explicit DestinationTenancy and a written reason. Ask: does this destination "
    "introspect X-Tenant-Id as a LICENCE and file/bill per tenant (DELEGATING — "
    "it must also join METERED_DESTINATION_SERVICES, which is fail-closed and an "
    "owner decision), does it hold no per-tenant state at all (TENANT_AGNOSTIC), "
    "or is it measurably mis-filing today (MISFILING_PENDING_OWNER_DECISION)? "
    "There is no fourth answer and no way to abstain."
)


def test_every_registry_destination_declares_its_tenancy() -> None:
    """An undeclared peer is a FAILURE, not a silent plain backend.

    This is the whole point of the file. A new entry in ``discovery._SERVICES``
    — i.e. a new client — cannot be added without stating its tenancy.
    """
    undeclared = sorted(set(_SERVICES) - set(DESTINATION_TENANCY))
    assert not undeclared, (
        f"destination(s) with NO tenancy declaration: {undeclared}. "
        f"Undeclared means every customer's work there is filed under the "
        f"caller's service-account tenant, silently. {_HOW_TO_FIX}"
    )


def test_no_declaration_is_stale_or_a_typo() -> None:
    """A declared name must be a real registry name, or say it is off-registry.

    Without this, a typo ("slide_srevice") looks like a declaration and the
    REAL peer stays undeclared — the denylist failure mode wearing a new hat.
    """
    orphans = sorted(
        name
        for name, decl in DESTINATION_TENANCY.items()
        if name not in _SERVICES and not decl.off_registry
    )
    assert not orphans, (
        f"declared but not in discovery._SERVICES and not marked off_registry: "
        f"{orphans}. Either it is a typo, or the peer was removed and the "
        f"declaration should go with it, or it is reached by an explicit base "
        f"URL and must set off_registry=True."
    )


def test_delegating_declarations_match_the_metered_set_exactly() -> None:
    """The two directions cannot drift apart.

    Declaring DELEGATING without joining the metered set produces no
    ``X-Effective-Tenant-Id`` (the defect). Being in the metered set without
    declaring DELEGATING means the table has stopped describing reality.
    """
    declared = names_with_tenancy(DestinationTenancy.DELEGATING)
    assert declared == METERED_DESTINATION_SERVICES, (
        "DELEGATING declarations and METERED_DESTINATION_SERVICES disagree. "
        f"declared-but-not-metered={sorted(declared - METERED_DESTINATION_SERVICES)} "
        f"metered-but-not-declared={sorted(METERED_DESTINATION_SERVICES - declared)}"
    )


def test_the_empty_misfiling_parametrization_still_derives_its_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DISTINCT JOB: the SELECTOR is live — not that the register is drained.

    The drained register itself is asserted exactly once, in
    ``test_metered_destination_flip.py::test_tts_gateway_is_the_only_permitted_residue``.
    It is deliberately NOT re-asserted here: two copies of one barrier get
    mutated singly and then drift, which is the failure this whole file exists
    to prevent.

    What that test cannot see is THIS file's problem. As of 2026-09-15 the
    parametrized guard below collects ZERO cases, and a zero-case
    ``parametrize`` passes silently and forever — indistinguishable from a
    guard that is running. The open question is therefore not "is the register
    empty" (answered elsewhere) but "if something were parked there tomorrow,
    would ``names_with_tenancy`` actually hand it to the guard below, or has
    the selector gone blind?" Park a fake declaration and require the selector
    to produce it. If this ever fails, the guard below is decoration and no
    amount of re-draining would refill it.
    """
    monkeypatch.setitem(
        DESTINATION_TENANCY,
        "fake_parked_peer",
        TenancyDeclaration(
            DestinationTenancy.MISFILING_PENDING_OWNER_DECISION,
            "A fake declaration parked purely to prove the selector below still "
            "turns register membership into collected test cases.",
        ),
    )
    assert "fake_parked_peer" in names_with_tenancy(
        DestinationTenancy.MISFILING_PENDING_OWNER_DECISION
    ), (
        "names_with_tenancy(MISFILING_PENDING_OWNER_DECISION) did not return a "
        "destination that IS declared MISFILING_PENDING_OWNER_DECISION — the "
        "parametrized guard below can no longer collect cases, so its current "
        "emptiness proves nothing."
    )


@pytest.mark.parametrize(
    "name", sorted(names_with_tenancy(DestinationTenancy.MISFILING_PENDING_OWNER_DECISION))
)
def test_a_known_misfiling_destination_is_not_already_metered(name: str) -> None:
    """MISFILING means "not yet fixed". If it IS metered, the label is a lie.

    Once the owner flips a name into ``METERED_DESTINATION_SERVICES``, its
    declaration must move to ``DELEGATING`` — this keeps the defect register
    from quietly becoming a list of already-fixed things nobody rechecks.
    """
    assert name not in METERED_DESTINATION_SERVICES, (
        f"{name} is declared MISFILING_PENDING_OWNER_DECISION but is already in "
        "METERED_DESTINATION_SERVICES — move its declaration to DELEGATING."
    )


@pytest.mark.parametrize("name,decl", sorted(DESTINATION_TENANCY.items()))
def test_every_declaration_states_a_real_reason(name: str, decl: TenancyDeclaration) -> None:
    """A classification with no rationale is a denylist entry in disguise.

    The 40-character floor is deliberately crude: it is not a quality bar, it
    is a friction bar. It makes "" and "n/a" impossible, so the person adding a
    peer has to think about the answer for at least one sentence.
    """
    assert isinstance(decl.tenancy, DestinationTenancy), name
    assert len(decl.why.strip()) >= 40, (
        f"{name}: tenancy declarations require a written reason (got "
        f"{decl.why.strip()!r}). {_HOW_TO_FIX}"
    )


def test_the_guard_can_actually_fail() -> None:
    """Prove the enumeration bites, in-process.

    A guard nobody has watched fail is not a guard. Injecting a fake peer into
    a COPY of the registry set must make the undeclared-detection fire; if this
    ever passes trivially, the real assertion above has gone blind.
    """
    fake_registry = set(_SERVICES) | {"fake_undeclared_peer"}
    undeclared = fake_registry - set(DESTINATION_TENANCY)
    assert undeclared == {"fake_undeclared_peer"}
