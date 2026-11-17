"""Outbound tenant headers — the destination-dependent identity shape.

WHY THIS MODULE EXISTS SEPARATELY FROM ``identity_context``
-----------------------------------------------------------
``identity_context`` owns per-request ContextVar *plumbing* and the httpx
``event_hooks`` wiring. This module owns a *policy decision* that has nothing
to do with ContextVars: given a tenant, WHERE it is going, and HOW it was
obtained, what does the wire look like — and when must the hop be refused?

Keeping it here means:

* the policy is a **pure function**, so it is testable without an event loop,
  an httpx client, or a bound ContextVar;
* it has no import back into ``identity_context``, so ``identity_context`` can
  import *it* (the hook delegates the delegation gate here) with no cycle;
* a caller that builds headers by hand — a stage handler assembling a
  one-off request — gets the same rule as the shared hook,
  instead of re-deriving a weaker one.

THE CONTRACT (kag ``src/services/proxy.py:138`` is the reference)
------------------------------------------------------------------
1. Services read ``X-Tenant-Id`` only — never a body or query tenant.
2. Provenance is a ladder (:class:`w2v_common.tenant.TenantSource`). Only
   ``VOUCHED_TENANT_SOURCES`` may be DELEGATED.
3. A **plain backend** gets ``X-Tenant-Id: <uuid>`` verbatim — the tenant is a
   data scope there, nothing is spent, so provenance does not gate it and no
   caller loses access to a non-metered peer.
   A **metered / license-introspecting** peer gets this service's OWN
   ``X-Tenant-Id: Bearer <downstream license token>`` PLUS
   ``X-Effective-Tenant-Id: <tenant uuid>`` naming who the work is for.
4. Delegating with an UNVOUCHED provenance RAISES
   (:class:`w2v_common.tenant.UnvouchedTenantError`). It never silently
   delegates, and it never silently downgrades to "drop the header" — that
   would bill the service-account tenant for a customer's run.
5. Absent tenant = fail closed (:class:`w2v_common.tenant.MissingTenantError`).
   Never forward tenant-less, never substitute.
"""

from __future__ import annotations

import logging
import os
import uuid
from functools import cache
from urllib.parse import urlsplit

from ..discovery import list_services, service_url
from ..tenant import (
    VOUCHED_TENANT_SOURCES,
    MissingTenantError,
    TenantSource,
    UnvouchedTenantError,
)

# Trusted "effective tenant" side-channel. The canonical S2S auth channel is
# ``X-Tenant-Id: Bearer <service-account license token>`` — the receiver
# introspects it to the SERVICE-account tenant, NOT the customer/run tenant.
# This header carries the run's REAL tenant alongside that token. Receivers
# honour it ONLY when the introspected caller is a configured delegating
# service account (``AUTH_DELEGATING_SUBJECTS``), so a customer can never spoof
# another tenant — but the shared w2v SA subject IS on those lists, so sending
# it is live and load-bearing, not inert.
#
# Defined HERE rather than in ``identity_context`` (which re-exports it for
# every existing importer) so this module has no import back into
# ``identity_context`` and the hook there can call ``is_metered_destination``.
EFFECTIVE_TENANT_HEADER = "X-Effective-Tenant-Id"

_log = logging.getLogger(__name__)

__all__ = [
    "EFFECTIVE_TENANT_HEADER",
    "METERED_DESTINATION_SERVICES",
    "TENANT_HEADER",
    "MeteredDestinationConfigError",
    "build_downstream_tenant_headers",
    "is_metered_destination",
    "unresolvable_metered_names",
]

TENANT_HEADER = "X-Tenant-Id"

#: Discovery names of the peers that introspect ``X-Tenant-Id`` as a LICENSE
#: TOKEN and meter what they do — i.e. the ones for which naming the acting
#: tenant separately is both required and financially load-bearing.
#:
#: * ``modality_gateway`` — kag calls it "the ONLY downstream that demands a
#:   license token in ``X-Tenant-Id`` and rejects a bare UUID outright"
#:   (``kag/src/services/proxy.py::_is_license_introspecting_gateway``). Omitting
#:   it from this set in 0.67.0 was a REGRESSION: w2v callers sent it a bare
#:   uuid, which it refuses. It never shipped — the deploy gates held.
#: * ``worker_management`` — the w2v LLM gateway. It reads
#:   ``X-Effective-Tenant-Id`` for BYO-credential resolution and cost
#:   attribution; without it every customer's spend lands on the shared w2v SA
#:   tenant ``8382f0b3-c082-4f97-8d47-120ce5a18e05``.
#: * ``inference_router`` — emits the cost events worker_management's spend
#:   rolls up from.
#: * ``video_production`` — the other three consume INFERENCE cost; this one
#:   consumes RENDER cost, which is just as real and just as billable. The
#:   orchestrator calls it on its own service-account token, so with the name
#:   absent from this set no delegation header went out at all and every
#:   customer's render landed on the orchestrator SA tenant
#:   ``e148e8af-3751-49bf-a0fa-56ab46d31ddc`` — unattributable, not merely
#:   mislabelled. The receiver already accepts delegation in prod
#:   (``AUTH_DELEGATING_SUBJECTS`` is configured for video-production-service),
#:   so this classification is the whole of the fix.
#:
#: * ``html_render`` — the OTHER render peer, and the same defect as
#:   ``video_production`` one service over. It files each design/redesign job
#:   under whatever tenant the inbound credential introspects to, and the
#:   orchestrator's stage path (``web_design``, ``web_design_v2``,
#:   ``website_review`` redesign, ``promo`` html-segments → ``DesignClient``)
#:   presents only the shared service-account licence, so with the name absent
#:   from this set no delegation header went out and 1044 of 1070 rows in the
#:   service's database are filed under the orchestrator SA tenant
#:   ``e148e8af-3751-49bf-a0fa-56ab46d31ddc``. Its read path filters
#:   ``RedesignJob.tenant_id == <caller tenant>`` and 404s on a mismatch, so
#:   every customer's stage-created redesign is unreadable by its owner. The
#:   receiver already accepts delegation in prod (``AUTH_DELEGATING_SUBJECTS``
#:   is the shared w2v SA subject, ``TENANT_RLS_MODE=enforce``), so this
#:   classification is the whole of the fix.
#:
#: * ``analysis``, ``video_analyzer``, ``slide_service``, ``app_inspector``,
#:   ``bdd_playwright``, ``ai_content_generation``, ``frontend_test_runner``,
#:   ``youtube_upload`` — THE OWNER FLIP, 2026-09-13. All eight were measured
#:   live filing customer work under the orchestrator service-account tenant
#:   ``e148e8af-3751-49bf-a0fa-56ab46d31ddc`` (``analysis`` alone: 15,663 rows;
#:   ``video_analyzer`` 2500-vs-17; ``slide_service`` 1719-vs-7), and sat in
#:   ``destination_tenancy.MISFILING_PENDING_OWNER_DECISION`` waiting for the
#:   ruling that has now been given: **all usage bills to the LICENSED TENANT
#:   that caused it.** Each per-name rationale lives in
#:   ``destination_tenancy.DESTINATION_TENANCY`` — it is not duplicated here,
#:   because two hand-maintained copies of the same list is how the previous
#:   guard went blind. The flip is FAIL-CLOSED on purpose: a caller reaching
#:   one of these peers with an unvouched ``TenantSource`` now gets
#:   ``UnvouchedTenantError`` where it previously mis-filed in silence. A loud
#:   refusal is the intended outcome, not a regression — an unattributable row
#:   is worse than a failed call, because nobody ever finds it.
#:
#: ⚠️ MEMBERSHIP HERE MEANS "THIS DESTINATION READS ``X-Effective-Tenant-Id``",
#: not "this destination emits a billing event". The name is a MISNOMER kept
#: for compatibility: nothing in this library emits metering off this set. Its
#: only three consumers are ``identity_context._apply_outbound_tenant`` (stamp
#: the delegation header, and REFUSE an unvouched provenance),
#: ``build_downstream_tenant_headers`` below (the same policy, pure), and
#: ``clients.base._may_replay_transport_error`` (never replay an ambiguous
#: transport error against such a peer, since the first attempt may already
#: have been spent). Real metering is receiver-side config
#: (``METERING_STREAM``). So add a name here when the peer must be told WHO the
#: work is for — and know that doing so is FAIL-CLOSED: every caller of that
#: peer must thereafter present a vouched ``TenantSource``.
#:
#: Every OTHER peer in ``discovery._SERVICES`` is a plain backend by default.
#: That default is why ``auth.destination_tenancy`` exists: a peer's absence
#: from this set is SILENT and indistinguishable from a deliberate choice, so
#: every destination must now DECLARE its tenancy there and
#: ``tests/test_destination_tenancy_declared.py`` fails on any that does not.
#:
#: ⚠️  THE ENV OVERRIDE IS ADDITIVE — IT USED TO BE A FLEET-WIDE
#: OFF SWITCH.  ``W2V_METERED_DESTINATIONS`` (comma-separated discovery names
#: OR hostnames)
#: once REPLACED this set outright. That meant a deployment setting it to the
#: single name it wanted to ADD silently demoted every other destination at
#: once — no ``X-Effective-Tenant-Id`` anywhere, every customer's work fleet-
#: wide filed under the caller's service-account tenant, with no error and no
#: log line. ``","`` or ``" "`` parsed to the EMPTY set and demoted all of them,
#: and a typo'd name was swallowed by design. One environment variable could
#: therefore undo this entire module, and the only symptom was accounting that
#: looked plausible and was wrong. See ``_parse_metered_override`` for the
#: replacement semantics.
METERED_DESTINATION_SERVICES: frozenset[str] = frozenset(
    {
        "worker_management",
        "inference_router",
        "modality_gateway",
        "video_production",
        "html_render",
        # The owner flip, 2026-09-13 — see the rationale block above and the
        # per-name measurements in ``auth.destination_tenancy``.
        "analysis",
        "video_analyzer",
        "slide_service",
        "app_inspector",
        "bdd_playwright",
        "ai_content_generation",
        "frontend_test_runner",
        "youtube_upload",
        # The ninth destination of the 2026-09-13 ruling, flipped once the
        # mechanical blocker was removed: ``tts_gateway`` now has a
        # ``discovery._SERVICES`` entry, so this name RESOLVES to the host
        # the tts-gateway host and the predicate can actually
        # match a caller's base URL. Adding the name WITHOUT that entry would
        # have been inert wiring — see ``auth.destination_tenancy``.
        "tts_gateway",
        # 2026-09-16: video-compiler bills its own render (w2v-video-compiler)
        # and the orchestrator calls it directly for audiobooks/courses — the
        # "accounted at video_production" rationale that kept it out was wrong.
        # See ``auth.destination_tenancy``.
        "video_compile",
    }
)

#: Marks a deliberate DEMOTION in ``W2V_METERED_DESTINATIONS``. A deployment
#: must still be able to say "here, tts_gateway is NOT metered" — that was the one
#: legitimate use of the old replacement semantics — but it now has to say so
#: by name, per destination, rather than by omission.
_DEMOTE_PREFIX = "-"


class MeteredDestinationConfigError(RuntimeError):
    """``W2V_METERED_DESTINATIONS`` says something that cannot be honoured.

    Raised rather than shrugged off. The old parser swallowed every mistake —
    an empty value, a typo, a stale name — and each mistake degraded silently
    into "that peer is a plain backend", i.e. into mis-filed billing. A
    configuration error that costs money must be loud, and a process that
    cannot meter correctly must not start metering incorrectly.
    """


@cache
def _parse_metered_override(raw: str) -> frozenset[str]:
    """Resolve ``W2V_METERED_DESTINATIONS`` ADDITIVELY, or refuse to resolve it.

    The grammar is a comma-separated list of discovery names OR hostnames,
    each optionally prefixed with ``-`` to demote it:

    * ``"tts"`` → the code default PLUS ``tts``.
    * ``"wm.example"`` → the code default PLUS any destination whose URL has
      that hostname. Host forms are first-class here, not tolerated:
      :func:`is_metered_destination` matches on HOSTNAME, so a host entry is
      compared directly against the outbound URL's host and works.
    * ``"-tts_gateway"`` → the code default MINUS ``tts_gateway``.
    * ``"tts,-tts_gateway"`` → both.

    Why additive rather than replacing: under replacement, the low-effort
    mistake (name the one peer you care about) has the maximum blast radius
    (demote all the others), and nothing distinguishes it from intent. Under
    addition the low-effort mistake is a no-op on everything you did not name,
    and a demotion — the only dangerous edit — has to be typed out per
    destination. The dangerous thing became the verbose thing.

    VALIDATION IS ASYMMETRIC, AND THAT ASYMMETRY IS THE POINT (0.74.1)
    ------------------------------------------------------------------
    0.74.0 validated BOTH directions against ``discovery.list_services()`` and
    raised on any name that was not a discovery key. That was wrong twice over.

    * It broke live, valid config. Consumers legitimately name a **host**, not
      a discovery key — ``wm.example`` (w2v_orchestrator),
      ``modality-gateway.modality-gateway``
      (ai_content_generation_service) were both in production when 0.74.0
      shipped. Both are matched perfectly well by :func:`is_metered_destination`,
      which compares HOSTNAMES and not discovery keys, so both were correct
      config that the validator declared illegal. The next deploy would have
      raised on the first outbound call.
    * It was redundant. The footgun that justified validation was
      REPLACEMENT — one typo demoted the whole fleet. Addition already fixed
      that: under addition an unrecognised name adds a string that nothing
      matches. It is INERT, not dangerous. Converting a harmless unknown into
      a hard failure is a strictly worse trade: the blast radius of the typo
      went from zero to "the process cannot make outbound calls".

    So the ADD direction is permissive and merely LOUD — an unrecognised name
    is accepted and logged once at warning level, naming it, so a typo is
    visible without being fatal. Note that a bare host cannot be told apart
    from a typo statically; the warning says so rather than guessing.

    The DEMOTE direction stays strict, because there the failure modes invert:

    * an empty / whitespace / bare-comma value once parsed to the EMPTY set and
      demoted the entire fleet; it still raises, because there is no reading of
      "the operator set this variable" that means "meter nothing".
    * demoting a name that is not metered in the first place is a stale config
      line: it either predates a rename or never did anything. Silently
      accepting it leaves a destination METERED while the operator believes
      they demoted it — the one unknown that is genuinely dangerous, so it
      still raises.
    * and no combination of demotions may empty the set, for the same reason
      the empty value cannot.

    Raising from here surfaces on the first outbound call in the process rather
    than at literal import time — there is no startup hook in a library — but it
    is deterministic and immediate, and the message names the variable, the
    offending token and the legal names.
    """
    tokens = [token.strip() for token in raw.split(",")]
    tokens = [token for token in tokens if token]
    if not tokens:
        raise MeteredDestinationConfigError(
            "W2V_METERED_DESTINATIONS is set to "
            f"{raw!r}, which names no destination at all. Under the previous "
            "replacement semantics that demoted EVERY metered destination "
            "fleet-wide and billed every customer's work to the caller's "
            "service-account tenant, silently. Unset the variable to use the "
            "code default, or name destinations: 'name' adds one, '-name' "
            "demotes one."
        )

    additions: set[str] = set()
    removals: set[str] = set()
    for token in tokens:
        if token.startswith(_DEMOTE_PREFIX):
            name = token[len(_DEMOTE_PREFIX) :].strip()
            if not name:
                raise MeteredDestinationConfigError(
                    f"W2V_METERED_DESTINATIONS contains a bare {token!r} with no "
                    "destination after it."
                )
            removals.add(name.lower())
        else:
            # Normalise through ``_host_of`` so a URL-form value
            # (``https://wm.example:8887/``) reduces to the same host
            # ``is_metered_destination`` will compare against. A bare name or a
            # bare host passes through lowercased, unchanged.
            additions.add(_host_of(token))

    known = frozenset(list_services())

    # THE ADD DIRECTION IS INERT WHEN UNRECOGNISED, SO IT IS WARNED, NOT RAISED.
    # A host form is legitimate and common (``wm.example``); a typo is not
    # distinguishable from it here, and does nothing either way.
    unrecognised_additions = sorted(additions - known)
    if unrecognised_additions:
        _log.warning(
            "W2V_METERED_DESTINATIONS names %s, which is not a discovery service "
            "name. Accepted as a literal HOST match: it meters a destination only "
            "if an outbound URL's hostname equals it exactly (is_metered_destination "
            "compares hostnames). If that is a hostname, this is correct and "
            "working; if it is a typo, it is inert — it meters nothing and demotes "
            "nothing. Discovery names: %s",
            unrecognised_additions,
            sorted(known),
        )

    both = sorted(additions & removals)
    if both:
        raise MeteredDestinationConfigError(
            f"W2V_METERED_DESTINATIONS both adds and demotes {both}; the "
            "intent is unreadable, so neither is applied."
        )

    # THE DEMOTE DIRECTION STAYS FATAL. Unlike an unknown addition, an unknown
    # or stale demotion is not inert: the operator believes a destination was
    # demoted and it is still metered. Silently accepting that is the config
    # error worth an outage.
    stale = sorted(removals - METERED_DESTINATION_SERVICES)
    if stale:
        raise MeteredDestinationConfigError(
            f"W2V_METERED_DESTINATIONS demotes {stale}, which is not metered by "
            "default, so the line does nothing while reading as though it does. "
            "Remove it, or fix the name: a stale demotion means this deployment's "
            "config asserts something about metering that is not true, and the "
            f"destination stays METERED. Metered by default: "
            f"{sorted(METERED_DESTINATION_SERVICES)}."
        )

    resolved = frozenset((METERED_DESTINATION_SERVICES | additions) - removals)
    if not resolved:
        # Reachable only by naming every default destination as a demotion. It
        # is explicit rather than accidental, but the outcome is the fleet-wide
        # off switch this whole design exists to remove, so it is still refused.
        raise MeteredDestinationConfigError(
            f"W2V_METERED_DESTINATIONS is set to {raw!r}, which demotes EVERY "
            "metered destination and leaves nothing metered. That is the "
            "fleet-wide off switch this variable no longer provides: with it, "
            "every customer's work files under the caller's service-account "
            "tenant. Leave at least one destination metered."
        )
    return resolved


def _configured_metered_names() -> frozenset[str]:
    raw = os.getenv("W2V_METERED_DESTINATIONS")
    if raw is None:
        return METERED_DESTINATION_SERVICES
    return _parse_metered_override(raw)


def unresolvable_metered_names(names: frozenset[str] | None = None) -> frozenset[str]:
    """Metered names that ``is_metered_destination`` can never match from a URL.

    THE INERT-WIRING CHECK. ``is_metered_destination`` compares HOSTS: it
    resolves each metered name through ``service_url`` and compares that host
    against the destination's. A name that is not in ``discovery._SERVICES``
    has no host to resolve, so the only way it ever matches is a caller passing
    the bare literal name as the destination string — and real callers pass a
    base URL. Adding such a name to :data:`METERED_DESTINATION_SERVICES`
    therefore LOOKS like a fix, passes the declaration guard, and stamps no
    delegation header on any actual request.

    ``tts_gateway`` is exactly that case: it is reached through
    ``settings.downstream.tts_gateway_url``, never through discovery. This
    function exists so that fact is asserted by a test instead of remembered.
    """
    candidates = METERED_DESTINATION_SERVICES if names is None else names
    known = frozenset(list_services())
    return frozenset(name for name in candidates if name not in known)


def _host_of(value: str) -> str:
    """The hostname of ``value``, or ``value`` itself when it is a bare name.

    Matching on the HOST (not the whole string) is what lets one predicate
    serve both callers: ``build_downstream_tenant_headers`` is handed a base
    URL, while ``identity_event_hook`` is handed a full request URL with a path
    on it. A string-equality match would silently classify
    ``…:8887/workers/x/message`` as a plain backend — the exact
    fail-open this whole change exists to remove. The PORT is deliberately not
    compared either: ``discovery`` owns the port, and a caller reaching the
    same host on an override port is still the same metered peer.
    """
    text = value.strip().rstrip("/")
    if "://" in text:
        return (urlsplit(text).hostname or "").lower()
    return text.lower()


def is_metered_destination(destination: str) -> bool:
    """True when ``destination`` is a license-introspecting, metered peer.

    ``destination`` may be a full URL (a base URL or a request URL with a
    path), or a bare discovery name (``"worker_management"``). Matching on the
    DESTINATION — never on the call site — is what kag's
    ``_is_license_introspecting_gateway`` exists to enforce: its predecessor
    chose the outbound token by call site and, when gitops set the token but
    left one URL pointing at a plain backend, stamped the SA token onto every
    customer's request to that backend for months.

    Resolution is SYMMETRIC: the metered NAMES are resolved through
    ``service_url``, and so is a bare-name ``destination``. Resolving only one
    side made the predicate disagree with itself. Live gitops
    (``clusters/strato/services/w2v/configmaps.yaml``) sets
    ``W2V_TTS_URL`` to the modality-gateway, so ``service_url("tts")`` is a
    modality-gateway URL — a metered peer. Under the old one-sided matching,
    ``is_metered_destination("tts")`` was False while
    ``is_metered_destination(service_url("tts"))`` was True, and the two
    answers select DIFFERENT outbound tenant headers: the False branch sends a
    bare-UUID ``X-Tenant-Id`` to a gateway that introspects a license and
    rejects it. Latent today only because every live caller happens to pass a
    full URL (``identity_event_hook`` always does); it arms the moment a stage
    handler passes a discovery name.
    """
    target = _host_of(destination)
    if not target:
        return False
    # Candidate hosts for the DESTINATION side. A bare name also resolves
    # through its own ``W2V_<NAME>_URL`` override, mirroring what the loop
    # below does for the metered names.
    targets = {target}
    if "://" not in destination:
        try:
            resolved = _host_of(service_url(target))
        except KeyError:
            # ``destination`` is not a known discovery name (a hostname without
            # a scheme, say). Fall through to bare-string matching rather than
            # raising out of every outbound call — same rationale as below.
            pass
        else:
            if resolved:
                targets.add(resolved)
    for name in _configured_metered_names():
        if name.strip().lower() in targets:
            return True
        try:
            if _host_of(service_url(name)) in targets:
                return True
        except KeyError:
            # A metered name with no discovery entry. Two ways to get here, and
            # they are NOT the same:
            #
            # * an env-supplied HOST form (``wm.example``). The bare-string
            #   comparison one line above already matched it against the
            #   destination's hostname, so reaching this KeyError just means
            #   "this host is not that host" — it is working as designed.
            # * a hand-added entry in METERED_DESTINATION_SERVICES that is
            #   neither a discovery name nor a host any caller passes. That one
            #   is INERT rather than safe, and is what
            #   ``unresolvable_metered_names`` exists to assert against.
            continue
    return False


def build_downstream_tenant_headers(
    tenant_id: uuid.UUID | None,
    destination: str,
    tenant_source: TenantSource,
    *,
    downstream_license_token: str | None = None,
) -> dict[str, str]:
    """Build the outbound tenant headers for one hop to ``destination``.

    Args:
        tenant_id: the requestor's real tenant. ``None`` RAISES — there is no
            fallback tenant, by product-owner ruling.
        destination: the callee's base URL or discovery name. The identity
            shape follows the DESTINATION, not the call site.
        tenant_source: where ``tenant_id`` came from. **No default, on
            purpose** — this is kag's rule and it is the load-bearing one.
            Delegation is a privilege escalation, so a new call site must be
            forced to state its provenance; a default would let the next
            client silently re-open the hole. Omitting it is a ``TypeError``
            at call time rather than a quiet escalation in prod.
        downstream_license_token: this SERVICE's own license token. Defaults to
            the ``DOWNSTREAM_LICENSE_TOKEN`` env var (the same value
            :class:`w2v_common.config.downstream.DownstreamSettings` reads).
            Required only for a metered destination.

    Returns:
        The header dict to merge onto the outbound request.

    Raises:
        MissingTenantError: ``tenant_id`` is ``None``. Fail closed — a
            tenant-less call must not go out at all, not go out unattributed.
        UnvouchedTenantError: the hop would DELEGATE (metered destination)
            but ``tenant_source`` is not in ``VOUCHED_TENANT_SOURCES``.
        RuntimeError: a metered destination with no ``DOWNSTREAM_LICENSE_TOKEN``
            configured. Sending ``Bearer `` would 401 with a misleading
            "invalid token"; naming the missing config is the useful failure.
    """
    if tenant_id is None:
        raise MissingTenantError(
            f"outbound hop to {destination}: no tenant bound to this request; "
            "every message must carry the requestor's tenant_id and there is no default"
        )

    if not is_metered_destination(destination):
        # Plain backend: the caller's tenant, untouched. No delegation happens
        # here — the destination has no delegation header at all and the tenant
        # is a plain data scope — so provenance does not gate it.
        return {TENANT_HEADER: str(tenant_id)}

    # THE DELEGATED HOP. This service authenticates as ITSELF and names a
    # DIFFERENT tenant in X-Effective-Tenant-Id. The receiver acts on that name:
    # it selects that tenant's BYO provider credentials and bills that tenant.
    # That is only legitimate when this service can stand behind the name.
    if tenant_source not in VOUCHED_TENANT_SOURCES:
        raise UnvouchedTenantError(
            f"refusing to delegate tenant {tenant_id} to metered destination {destination}: "
            f"provenance is {tenant_source.value!r}, which is not vouched. Present "
            "'X-Tenant-Id: Bearer <license-token>' or arrive via a verified JWT claim."
        )

    token = (
        downstream_license_token
        if downstream_license_token is not None
        else os.getenv("DOWNSTREAM_LICENSE_TOKEN", "")
    )
    if not token:
        raise RuntimeError(
            f"metered destination {destination} requires this service's own license token, "
            "but DOWNSTREAM_LICENSE_TOKEN is unset"
        )

    return {
        TENANT_HEADER: f"Bearer {token}",
        # Preserve WHO the request is for. Without this the receiver resolves
        # BYO config and cost attribution against this service's SA tenant for
        # every customer.
        EFFECTIVE_TENANT_HEADER: str(tenant_id),
    }
