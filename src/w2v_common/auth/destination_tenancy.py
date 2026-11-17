"""Every outbound destination DECLARES its tenancy. Absence is an error.

WHY THIS EXISTS
---------------
``auth.outbound.METERED_DESTINATION_SERVICES`` decides whether a hop carries
``X-Effective-Tenant-Id`` (the run's REAL tenant) or nothing at all. A
destination that is absent from that set gets no delegation header, so the
receiver introspects the caller's shared service-account licence and files
every customer's work under the SA tenant
``e148e8af-3751-49bf-a0fa-56ab46d31ddc``.

That absence is SILENT. It has now been discovered three separate times, each
time by accident:

* ``video_production`` — render cost on the SA tenant.
* ``html_render`` — 1044 of 1070 rows SA-filed; customers could not read back
  their own redesigns (fixed 2026-09-13, PR #174).
* nine more, measured live 2026-09-13 and recorded in
  :data:`DESTINATION_TENANCY` below.

``tests/test_auth_outbound.py`` could never have caught any of them: it asserts
things about names that are ALREADY in the set. A test written that way is
structurally incapable of noticing an omission.

THE SHAPE OF THE FIX — DECLARATION-REQUIRED, NOT A DENYLIST
-----------------------------------------------------------
A hand-maintained "these are fine to skip" list is the WRONG shape and is the
exact shape that has already failed repeatedly in this codebase: a new entry
that nobody remembers to add is indistinguishable from a correct one, so the
list goes quietly out of date and the guard reports green.

So the rule here is inverted. Every destination in ``discovery._SERVICES``
MUST appear in :data:`DESTINATION_TENANCY` with an explicit
:class:`DestinationTenancy` and a written rationale. A destination that appears
in NEITHER this mapping NOR the metered set is a FAILURE, not a pass. Adding a
client for a new peer is therefore impossible without answering, in writing:
*is this destination tenant-delegating, or genuinely tenant-agnostic?*

THIS MODULE CHANGES NO BEHAVIOUR. It is a declaration surface consumed only by
``tests/test_destination_tenancy_declared.py``. Nothing imports it at runtime,
and in particular it does NOT feed ``METERED_DESTINATION_SERVICES`` — moving a
name into that set is fail-closed (every caller of that peer must thereafter
present a vouched ``TenantSource`` or get ``UnvouchedTenantError``), which is
an outage-risk decision for the service owner, not a refactor.

THE OWNER RULING (2026-09-13)
----------------------------
All usage bills to the LICENSED TENANT that caused it. Eight of the nine
``MISFILING_PENDING_OWNER_DECISION`` destinations recorded here were flipped to
``DELEGATING`` and joined ``METERED_DESTINATION_SERVICES`` on the day of the
ruling. The ninth, ``tts_gateway``, was held back — not because the ruling
exempted it, but because it had no ``discovery._SERVICES`` entry, so the flip
alone would have been inert wiring. That blocker was removed on 2026-09-15 (the
gateway is now registered as ``("tts-gateway", "w2v", 6320)``) and the ninth
declaration flipped too, so the register is now EMPTY.

That closes the HOST-RESOLUTION half. The RECEIVER half is DEPLOYED and
ACCEPTING: ``tts-gateway-service-config`` sets ``AUTH_DELEGATING_SUBJECTS`` to
the orchestrator subject ``8382f0b3-c082-4f97-8d47-120ce5a18e05``
(gitops-production PR #3928, merge ``715a6b62``), so
``middleware._delegating_subjects()`` is not an empty frozenset there and the
reject branch does not fire on that account. Verified live 2026-09-15 off the
running pod ``tts-gateway-54989b667c-pwxvn`` (``printenv
AUTH_DELEGATING_SUBJECTS`` returned that subject).

NOTHING IS PROVEN END TO END. Deployed is not exercised, and one hop is not
even shipped: the chain is orchestrator -> tts_gateway (stamped, accepted) ->
gateway worker -> receiver, and the WORKER -> RECEIVER hop is NOT stamped. The
billing cost event is emitted downstream of that break, so a customer narration
today bills to the fleet SA licence tenant. The missing stamp exists on a
branch in ``tts_gateway``, but it ships dark behind
``SYNTH_EFFECTIVE_TENANT_ENABLED``, default false, because enabling it would
re-scope live in-cluster traffic through modality-gateway with
``license_id=None`` and risk a narration outage. OPEN QUESTION: whether that
flag can be enabled safely, and what would exercise the full chain once it is.
Its declaration below carries the mechanism and the evidence.

THE ENV OVERRIDE WAS A GLOBAL OFF SWITCH — IT NO LONGER IS
-----------------------------------------------------------
``W2V_METERED_DESTINATIONS`` used to REPLACE ``METERED_DESTINATION_SERVICES``
outright, so one env var set to a single name — or to an empty-ish value like
``","`` — silently demoted EVERY OTHER destination at once: no
``X-Effective-Tenant-Id`` anywhere, every tenant's work fleet-wide landing on
the caller's SA tenant, with no error and no log line. It is now ADDITIVE and
VALIDATED (``outbound._parse_metered_override``): a bare name ADDS, ``-name``
demotes one destination explicitly, and an empty, unknown or stale value raises
``MeteredDestinationConfigError`` instead of quietly demoting the fleet. The
declarations below still describe the CODE default; a deployment's env can now
only differ from it in ways it stated out loud.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "DESTINATION_TENANCY",
    "DestinationTenancy",
    "TenancyDeclaration",
    "declared_names",
    "names_with_tenancy",
]


class DestinationTenancy(Enum):
    """How one destination relates to the acting tenant."""

    #: The destination introspects ``X-Tenant-Id`` as a LICENCE TOKEN and must
    #: be told separately WHO the work is for. Such a name MUST also be in
    #: ``outbound.METERED_DESTINATION_SERVICES``; the guard checks both
    #: directions, so the two can never drift apart.
    DELEGATING = "delegating"

    #: The destination holds no per-tenant state and meters nothing: there is
    #: nothing for a tenant name to scope or to bill. A bare
    #: ``X-Tenant-Id: <uuid>`` (or the SA licence) is correct and complete.
    TENANT_AGNOSTIC = "tenant_agnostic"

    #: MEASURED to be mis-filing customer work under the caller's SA tenant,
    #: and NOT yet fixed. This is not an excuse slot — it is a defect register
    #: with an owner decision attached. Declaring a destination here asserts
    #: that the mis-filing is known, quantified in ``why``, and awaiting the
    #: fail-closed flip into ``METERED_DESTINATION_SERVICES``. A name declared
    #: here MUST NOT already be in that set (the guard enforces it): once it is
    #: flipped, the declaration moves to ``DELEGATING``.
    MISFILING_PENDING_OWNER_DECISION = "misfiling_pending_owner_decision"


@dataclass(frozen=True, slots=True)
class TenancyDeclaration:
    """One destination's declared tenancy, plus the reason it is that.

    ``why`` is REQUIRED and must be non-trivial. A classification with no
    stated reason is how a denylist entry is born; the guard rejects an empty
    or near-empty rationale so "declared" cannot decay into "listed".
    """

    tenancy: DestinationTenancy
    why: str
    #: True when this destination is NOT a ``discovery._SERVICES`` key — it is
    #: reached through an explicit base URL instead. Such a declaration cannot
    #: be cross-checked against the registry, so it must say so out loud rather
    #: than look like a typo'd discovery name.
    off_registry: bool = False


#: THE DECLARATION TABLE. Every ``discovery._SERVICES`` name must be a key
#: here. An undeclared name fails ``tests/test_destination_tenancy_declared.py``
#: with instructions; there is no way to add a peer and stay silent.
DESTINATION_TENANCY: dict[str, TenancyDeclaration] = {
    # --- DELEGATING: in METERED_DESTINATION_SERVICES today ----------------
    "worker_management": TenancyDeclaration(
        DestinationTenancy.DELEGATING,
        "The w2v LLM gateway. Reads X-Effective-Tenant-Id for BYO-credential "
        "resolution and cost attribution; without it every customer's model "
        "spend lands on the shared w2v SA tenant.",
    ),
    "inference_router": TenancyDeclaration(
        DestinationTenancy.DELEGATING,
        "Emits the cost events that worker_management's spend rolls up from, "
        "so the acting tenant is the billing key.",
    ),
    "modality_gateway": TenancyDeclaration(
        DestinationTenancy.DELEGATING,
        "The only downstream that demands a licence token in X-Tenant-Id and "
        "rejects a bare UUID outright (kag proxy.py).",
    ),
    "video_production": TenancyDeclaration(
        DestinationTenancy.DELEGATING,
        "Consumes RENDER cost. Absent from the metered set, every customer's "
        "render was filed under the orchestrator SA tenant — unattributable, "
        "not merely mislabelled.",
    ),
    "html_render": TenancyDeclaration(
        DestinationTenancy.DELEGATING,
        "Same defect as video_production one service over; 1044 of 1070 rows "
        "were SA-filed and its read path 404s on a tenant mismatch, so owners "
        "could not read back their own redesigns. Fixed 2026-09-13 (PR #174).",
    ),
    # --- TENANT_AGNOSTIC: nothing per-tenant to scope or bill -------------
    "orchestrator": TenancyDeclaration(
        DestinationTenancy.TENANT_AGNOSTIC,
        "This service itself / peer orchestrator hops. Not a downstream that "
        "files customer work; the tenant travels in the message body contract, "
        "and the orchestrator is the party that ESTABLISHES the tenant rather "
        "than one that needs to be told it out-of-band.",
    ),
    "tts": TenancyDeclaration(
        DestinationTenancy.TENANT_AGNOSTIC,
        "mlx-Kokoro launchd agent on mini2 (asr/tts host services). A stateless "
        "synthesis call: no per-tenant store, no usage table, no licence "
        "introspection. NOTE the live gitops override points W2V_TTS_URL at the "
        "modality gateway, in which case is_metered_destination resolves it as "
        "METERED by host — the declaration describes the registry default.",
    ),
    "transcription_worker": TenancyDeclaration(
        DestinationTenancy.TENANT_AGNOSTIC,
        "mlx-whisper launchd agent on mini2 (the ASR host service). Stateless "
        "transcription; no per-tenant rows, no metering.",
    ),
    "artifacts": TenancyDeclaration(
        DestinationTenancy.TENANT_AGNOSTIC,
        "Blob store addressed by artifact id. Tenancy is enforced by the "
        "caller holding the id, not by an out-of-band tenant name; it meters "
        "nothing.",
    ),
    "licenses": TenancyDeclaration(
        DestinationTenancy.TENANT_AGNOSTIC,
        "The licence authority itself — it RESOLVES tenancy from the presented "
        "token. Telling it which tenant to act as would be circular.",
    ),
    "course": TenancyDeclaration(
        DestinationTenancy.TENANT_AGNOSTIC,
        "Course content service: content is addressed by course id and the "
        "orchestrator owns the tenant scoping. No usage table, no licence "
        "introspection.",
    ),
    "webscraper": TenancyDeclaration(
        DestinationTenancy.TENANT_AGNOSTIC,
        "Shared fetcher (called directly; the retired KAG gateway once fronted it). "
        "Meters nothing and "
        "has no usage table; it resolves auth_config credentials from the licence "
        "it is PRESENTED, which TenantLicenceLane('webscraper') makes the acting "
        "tenant's own licence, so no out-of-band tenant name is needed.",
    ),
    "media_discovery": TenancyDeclaration(
        DestinationTenancy.TENANT_AGNOSTIC,
        "Federated stock-media search (called directly; the retired KAG gateway "
        "once fronted it). Emits "
        "no cost events (media-discovery src/auth/middleware.py introspects the "
        "presented licence only); tenancy is the presented licence itself.",
    ),
    "content": TenancyDeclaration(
        DestinationTenancy.TENANT_AGNOSTIC,
        "Content service: scopes by the bare X-Tenant-Id it is already sent on "
        "the plain-backend path. No metering, so no delegation header needed.",
    ),
    "video_compile": TenancyDeclaration(
        DestinationTenancy.DELEGATING,
        "video-compiler (its own namespace) emits its OWN cost event per render "
        "(service_key w2v-video-compiler, video_compiler_service "
        "src/api/routes/compile.py schedule_cost_event) under the tenant it "
        "introspects. The earlier TENANT_AGNOSTIC rationale — 'the render cost is "
        "accounted at video_production, so stamping here would double-count' — "
        "was wrong on both halves: video_production_service is the RECORDING "
        "service (/api/record/*), never calls the compiler and emits no cost "
        "event at all, and the orchestrator calls VideoCompileClient DIRECTLY "
        "(w2v_orchestrator src/services/stages/audiobook/stitch.py, course/stitch.py, "
        "primitives/compile_stage.py, demo_video/encode_master.py, "
        "demo_video/final_critique.py, shared/footage_critique.py). Measured "
        "2026-09-16: every w2v-video-compiler row was filed under the orchestrator "
        "SA tenant e148e8af-3751-49bf-a0fa-56ab46d31ddc, each one lockstep with a "
        "merqi audiobook run. The receiver already lists the orchestrator subject "
        "in AUTH_DELEGATING_SUBJECTS, so this flip is live, not inert. Flipped "
        "2026-09-16 under the 2026-09-13 owner ruling.",
    ),
    # --- THE OWNER FLIP, 2026-09-13 ---------------------------------------
    # All eight below were declared MISFILING_PENDING_OWNER_DECISION when this
    # table was written: each introspects the orchestrator's shared SA licence
    # and filed the work under SA tenant e148e8af-3751-49bf-a0fa-56ab46d31ddc.
    # The owner has now ruled: ALL USAGE BILLS TO THE LICENSED TENANT THAT
    # CAUSED IT. So each name moves to DELEGATING and joins
    # METERED_DESTINATION_SERVICES. That is FAIL-CLOSED by design — a caller
    # that cannot present a vouched TenantSource now gets UnvouchedTenantError
    # instead of a mis-filed row. A loud refusal is the intended outcome: an
    # unattributable row is worse than a failed call, because nobody finds it.
    "analysis": TenancyDeclaration(
        DestinationTenancy.DELEGATING,
        "15,663 rows were filed under the orchestrator SA tenant (measured live "
        "2026-09-13) — the largest single mis-filing in the fleet. It is also the "
        "discovery target of the critique, redesign and brand_extractor clients, "
        "so one classification fixes four call paths at once. Flipped by owner "
        "ruling 2026-09-13.",
    ),
    "video_analyzer": TenancyDeclaration(
        DestinationTenancy.DELEGATING,
        "2500 SA-tenant rows against 17 real-tenant rows (measured live "
        "2026-09-13): the 17 prove the receiver already honours delegation when "
        "it is sent, so the missing header was the whole defect. Flipped by "
        "owner ruling 2026-09-13.",
    ),
    "slide_service": TenancyDeclaration(
        DestinationTenancy.DELEGATING,
        "1719 SA-tenant rows against 7 real-tenant rows (measured live "
        "2026-09-13). Per-customer decks were owned by the service account, so "
        "the customer that paid for a deck could not be billed for it. Flipped "
        "by owner ruling 2026-09-13.",
    ),
    "app_inspector": TenancyDeclaration(
        DestinationTenancy.DELEGATING,
        "Mis-filed under the SA tenant AND carries an *_usage table, so BILLING "
        "attribution was wrong and not merely row ownership — the money moved to "
        "the wrong tenant, it was not just labelled oddly (measured 2026-09-13). "
        "Flipped by owner ruling 2026-09-13.",
    ),
    "bdd_playwright": TenancyDeclaration(
        DestinationTenancy.DELEGATING,
        "Mis-filed under the SA tenant AND carries an *_usage table, so every "
        "customer's Playwright run was billed to w2v's own tenant rather than to "
        "the licensee that requested it (measured 2026-09-13). Flipped by owner "
        "ruling 2026-09-13.",
    ),
    "ai_content_generation": TenancyDeclaration(
        DestinationTenancy.DELEGATING,
        "Generates per-customer copy — inference spend, the most expensive thing "
        "in the fleet — and filed every run under the orchestrator SA tenant "
        "(measured live 2026-09-13), so BYO-credential resolution also picked the "
        "wrong tenant's keys. Flipped by owner ruling 2026-09-13.",
    ),
    "frontend_test_runner": TenancyDeclaration(
        DestinationTenancy.DELEGATING,
        "Per-customer test runs were filed under the orchestrator SA tenant "
        "(measured live 2026-09-13), so run history was neither readable by nor "
        "billable to the customer that triggered it. Flipped by owner ruling "
        "2026-09-13.",
    ),
    "youtube_upload": TenancyDeclaration(
        DestinationTenancy.DELEGATING,
        "Per-customer uploads were filed under the orchestrator SA tenant "
        "(measured live 2026-09-13): the rows that record WHOSE video went to "
        "WHOSE channel all named the service account. Flipped by owner ruling "
        "2026-09-13.",
    ),
    # --- The ninth: predicate fixed here, receiver deployed, sender hop not shipped --
    "tts_gateway": TenancyDeclaration(
        DestinationTenancy.DELEGATING,
        "PREDICATE FIXED 2026-09-15 — NOT PROVEN END TO END. What is fixed "
        "HERE is HOST RESOLUTION: whether this library stamps "
        "X-Effective-Tenant-Id on the hop at all. Live billing additionally "
        "requires the RECEIVER to ACCEPT that header, and it does: "
        "tts-gateway-service-config sets AUTH_DELEGATING_SUBJECTS to the "
        "orchestrator subject 8382f0b3-c082-4f97-8d47-120ce5a18e05 "
        "(gitops-production PR #3928, merge 715a6b62), alongside the other "
        "eleven ConfigMaps in namespace w2v that name that subject — twelve "
        "including this one, counted live 2026-09-15. So "
        "middleware._delegating_subjects() is not an empty frozenset there, "
        "and the deployed pod tts-gateway-54989b667c-pwxvn does not take the "
        "reject branch on that account (printenv off that pod, 2026-09-15). "
        "STILL OPEN: only the orchestrator -> tts_gateway hop is stamped. The "
        "gateway worker -> receiver hop is NOT, and the billing cost event is "
        "emitted downstream of that break, so a customer narration today bills "
        "to the fleet SA licence tenant. The missing stamp is on a branch in "
        "tts_gateway, dark behind SYNTH_EFFECTIVE_TENANT_ENABLED (default "
        "false) because enabling it re-scopes live traffic through "
        "modality-gateway with license_id=None. Nothing has been exercised end "
        "to end. IDENTITY NOTE: 8382f0b3-c082-4f97-8d47-120ce5a18e05 is a "
        "subject.id in db licenses whose keycloak_sub is "
        "e148e8af-3751-49bf-a0fa-56ab46d31ddc — ONE identity in two shapes, "
        "not two principals; several sessions have read them as two. "
        "MEASURED 2026-09-15, stated as counts and nothing more: tts_jobs=536 "
        "rows (newest 2026-08-10) and tts_batch_jobs=546 rows (newest "
        "2026-09-11), all under tenant e148e8af-3751-49bf-a0fa-56ab46d31ddc; "
        "w2v_orchestrator.projects=1082 rows, 959 under that tenant and 123 "
        "across four other tenants. The broken link was never the "
        "owner decision and never the tenant (which is in scope at stamp time "
        "via the ContextVar): it was HOST RESOLUTION. is_metered_destination "
        "matches a destination by HOST, resolving each metered name through "
        "discovery.service_url(); 'tts_gateway' had no entry in "
        "discovery._SERVICES, so service_url() raised KeyError, the loop hit "
        "continue, and the match could never fire — while the bare-name "
        "comparison also missed, because the real target is the URL "
        "the tts-gateway URL. The fix therefore "
        "landed upstream of this table: the gateway is now registered as "
        "('tts-gateway', 'w2v', 6320) (confirmed live: kubectl -n w2v get svc "
        "tts-gateway, NodePort, port 6320), which makes the name resolvable "
        "and the PREDICATE real rather than inert. off_registry was dropped "
        "because it is no longer a true statement. "
        "outbound.unresolvable_metered_names() stays as the standing guard "
        "that no metered name is decoration.",
    ),
}


def declared_names() -> frozenset[str]:
    """Every destination name carrying an explicit tenancy declaration."""
    return frozenset(DESTINATION_TENANCY)


def names_with_tenancy(tenancy: DestinationTenancy) -> frozenset[str]:
    """The declared destinations classified as ``tenancy``."""
    return frozenset(name for name, decl in DESTINATION_TENANCY.items() if decl.tenancy is tenancy)
