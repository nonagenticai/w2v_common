"""License token authentication middleware.

All configuration is read from environment variables (no coupling to service config modules).
"""

import asyncio
import hashlib
import hmac
import ipaddress
import logging
import math
import os
import time as _time
import uuid as _uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

import httpx
from cachetools import TTLCache
from fastapi import Request
from prometheus_client import REGISTRY, Counter
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from ..discovery import service_url
from ..observability.metrics_server import start_metrics_server_from_env
from ..tenant import TenantError, TenantSource, parse_tenant_id
from .cell_jwt import (
    CELL_AUTH_LANE_CELL,
    CELL_AUTH_LANE_DUAL,
    CELL_AUTH_LANE_LEGACY,
    CELL_TOKEN_HEADER,
    SCOPE_NONE,
    CellAuthMode,
    CellClaims,
    CellScope,
    CellTokenError,
    cell_auth_mode,
    precreate_cell_auth_series,
    record_cell_auth,
    record_cell_rejection,
    record_dual_both,
    record_legacy_delegation,
    record_unsigned_attribution,
    verify_cell_token,
    warn_if_no_cell_callers,
)
from .cell_service import SERVICE_LANE_STATE_ATTR, route_cell_service_callers
from .identity_context import (
    EFFECTIVE_TENANT_HEADER,
    PROVIDER_HOP_HEADER,
    bind_attribution_from_headers,
    bind_unsigned_attribution_subject,
    clear_identity_context,
    mark_inbound_request,
    mark_provider_hop,
    set_correlation_context,
    set_identity_context,
    set_w2v_job_context,
    set_w2v_project_context,
)
from .keycloak import _looks_like_jwt, get_jwt_validator, realm_of_issuer
from .license_app import (
    WRONG_APP_CODE,
    check_license_app,
    introspected_app_key,
    warn_if_license_app_unchecked,
)
from .models import User

logger = logging.getLogger(__name__)

# The licenses-service introspection payload is free-form JSON (active, tenant_id,
# subject_id, reason, ...); it has no fixed schema this library owns, so it stays
# a JSON object. Named once here so the shape reads intentionally rather than as
# scattered ``dict[str, Any]`` literals.
type _Introspection = dict[str, Any]

_license_cache: TTLCache[str, _Introspection] | None = None
_license_neg_cache: TTLCache[str, _Introspection] | None = None
_license_stale_cache: TTLCache[str, _Introspection] | None = None
_license_token_not_found_since: TTLCache[str, float] | None = None
_license_tnf_probe: TTLCache[str, _Introspection | None] | None = None

# Per-token single-flight locks. A cold-cache burst of concurrent requests for
# the SAME license token (e.g. a narration render firing many synth calls at
# once) would otherwise each miss the cache and independently POST the licenses
# endpoint. On a CPU-saturated box that stampede starves the event loop and
# trips the introspection timeout → spurious 401 (the 2026-06-27 TTS bug class).
# The lock collapses the burst into one upstream call; the rest read the cache.
# ⚠️ BOUNDED ON PURPOSE. The key is a hash of the CALLER-SUPPLIED license token and
# nothing authenticates before it is minted, so an unauthenticated flood of distinct
# bogus tokens used to add one Lock per token, forever, on every pod in the fleet —
# a remote memory-exhaustion primitive ending in OOMKill. The old comment claimed the
# map was "bounded in practice by the small number of distinct tenant tokens"; that is
# only true of well-behaved callers, which is precisely the assumption an attacker
# breaks. Every other cache in this module is a bounded TTLCache; this one was not.
_INTROSPECT_LOCK_MAXSIZE = max(1, int(os.getenv("AUTH_INTROSPECT_LOCK_MAXSIZE", "1024")))
_introspect_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()


def _reset_introspect_locks() -> None:
    """Test seam — production never clears this map wholesale."""
    _introspect_locks.clear()


def _acquire_introspect_lock(cache_key: str) -> asyncio.Lock:
    """Return the single-flight lock for ``cache_key``, evicting only IDLE entries.

    A naive bounded map (plain LRU / TTLCache) would be WRONG here: evicting a lock
    that a slow introspection still holds hands the next caller for that same key a
    FRESH lock, so two coroutines enter the critical section together — reintroducing
    exactly the upstream stampede the lock exists to prevent. So eviction skips any
    lock that is currently held, oldest-first.

    If every lock is held the map may briefly exceed the ceiling. That residue is
    bounded by in-flight concurrency (which the event loop and the introspection
    timeout already bound), NOT by attacker input, so it cannot grow without limit.
    """
    lock = _introspect_locks.get(cache_key)
    if lock is not None:
        _introspect_locks.move_to_end(cache_key)
        return lock
    if len(_introspect_locks) >= _INTROSPECT_LOCK_MAXSIZE:
        for key in list(_introspect_locks):
            if len(_introspect_locks) < _INTROSPECT_LOCK_MAXSIZE:
                break
            if not _introspect_locks[key].locked():
                del _introspect_locks[key]
    lock = asyncio.Lock()
    _introspect_locks[cache_key] = lock
    return lock


def _get_license_cache() -> TTLCache[str, _Introspection]:
    """Lazy-init license cache from env vars."""
    global _license_cache
    if _license_cache is None:
        maxsize = int(os.getenv("CACHE_LICENSE_MAXSIZE", "1000"))
        ttl = int(os.getenv("CACHE_LICENSE_TTL", "300"))
        _license_cache = TTLCache(maxsize=maxsize, ttl=ttl)
    return _license_cache


def _get_license_neg_cache() -> TTLCache[str, _Introspection]:
    """Negative cache for ``active: false`` introspection results.

    A revoked token will pin every request against the licenses endpoint
    without this. Cache the rejection briefly so revocation storms don't
    amplify into a Self-DoS on the licenses service. TTL deliberately
    short (60s default) so legitimate re-issuance is picked up quickly.
    """
    global _license_neg_cache
    if _license_neg_cache is None:
        maxsize = int(os.getenv("CACHE_LICENSE_MAXSIZE", "1000"))
        neg_ttl = int(os.getenv("CACHE_LICENSE_NEG_TTL", "60"))
        _license_neg_cache = TTLCache(maxsize=maxsize, ttl=neg_ttl)
    return _license_neg_cache


def _get_license_stale_cache() -> TTLCache[str, _Introspection]:
    """Last-known-good store for stale-while-error introspection.

    Every successful ``active`` introspection is also recorded here with a
    longer TTL than the positive cache. When a later introspection fails
    TRANSIENTLY (timeout / connection error / non-200 from the licenses
    service) we serve the stale entry instead of denying — a token that was
    valid seconds ago is not suddenly invalid just because the licenses
    endpoint is briefly slow or unreachable. This is what stops a spurious
    ``401 INVALID_TOKEN`` on long-running callers when an introspect refresh
    (positive cache expires every CACHE_LICENSE_TTL) happens to hit a blip.

    Bounded staleness (default 900s) preserves the security posture: a genuine
    revocation is still picked up within the stale TTL, a definitive
    ``active: false`` is NEVER served stale (it denies immediately + neg-caches),
    and a token never seen before has no stale entry so a cold-cache failure
    still fails closed.
    """
    global _license_stale_cache
    if _license_stale_cache is None:
        maxsize = int(os.getenv("CACHE_LICENSE_MAXSIZE", "1000"))
        stale_ttl = int(os.getenv("CACHE_LICENSE_STALE_TTL", "900"))
        _license_stale_cache = TTLCache(maxsize=maxsize, ttl=stale_ttl)
    return _license_stale_cache


def _token_not_found_grace() -> float:
    """Grace window (seconds) for a ``token_not_found`` introspection result.

    A 200 ``{active: false, reason: "token_not_found"}`` during an ADDITIVE
    token rotation is a propagation race — the new token's row has not
    committed/replicated yet, or a pod still carries a stale Secret value — NOT
    a rejection. The shared w2v SA-token rotation is additive
    (``provision_service_token.py --rotate`` mints a new token and never revokes
    the old one), and a genuinely revoked/expired token returns reason
    ``token_invalid`` / ``license_inactive`` instead. So ONLY ``token_not_found``
    is treated as transient. Two cases: a pod that introspected this token
    successfully within the stale TTL serves the WARM last-known-good (this is
    the case that actually kills the sustained fleet-401 — the shared SA token
    is hot on every pod); a cold pod that has never seen the token has nothing
    to serve and still returns ``None`` (a 401), because inventing an identity
    would be worse than a brief deny. Bounded from the FIRST sighting so a
    permanently-missing token settles to an authoritative 401 after the window.
    ``0`` disables (restores the old instant-deny). Mirrors ``worker_management``
    PR #148.

    Accepted tradeoff (same as WM #148): a token that is HARD-deleted (not
    revoked) also introspects ``token_not_found``, so a warm pod keeps honoring
    it for up to the grace window. That is bounded, applies only to tokens seen
    active within the stale TTL, and is the price of not 401-storming on every
    additive rotation; set the grace to 0 to opt out.

    A malformed value falls back to the default rather than raising: this runs
    on the request path with no enclosing handler, so a typo'd env var would
    otherwise turn every authenticated request into a 500.
    """
    raw = os.getenv("LICENSE_TOKEN_NOT_FOUND_GRACE_SECONDS", "120")
    try:
        grace = float(raw)
    except ValueError:
        logger.warning("invalid LICENSE_TOKEN_NOT_FOUND_GRACE_SECONDS=%r; using default 120", raw)
        return 120.0
    # NaN compares False against everything (silently disabling the grace) and
    # inf never elapses (a permanently-missing token would never settle).
    if not math.isfinite(grace) or grace < 0:
        logger.warning(
            "out-of-range LICENSE_TOKEN_NOT_FOUND_GRACE_SECONDS=%r; using default 120",
            raw,
        )
        return 120.0
    return grace


def _get_license_tnf_probe() -> TTLCache[str, _Introspection | None]:
    """Short-TTL throttle for upstream re-probes during a grace window.

    The grace path deliberately does NOT neg-cache, so recovery is immediate
    once a rotating token propagates. Without a throttle that means one upstream
    introspection PER REQUEST for as long as the token is missing — where the
    pre-grace behaviour cost one per ``CACHE_LICENSE_NEG_TTL`` (60s). Measured:
    20 concurrent requests for one missing token produced 20 upstream POSTs,
    serialized behind the single-flight lock, so a slow licenses service turns
    that into a latency queue.

    Caching the transient OUTCOME (warm-stale dict, or ``None``) for a few
    seconds bounds it to one upstream call per ``ttl`` per token, while keeping
    recovery latency an order of magnitude below the old 60s neg-cache.
    """
    global _license_tnf_probe
    if _license_tnf_probe is None:
        maxsize = int(os.getenv("CACHE_LICENSE_MAXSIZE", "1000"))
        ttl = int(os.getenv("LICENSE_TOKEN_NOT_FOUND_PROBE_TTL", "5"))
        _license_tnf_probe = TTLCache(maxsize=maxsize, ttl=max(ttl, 1))
    return _license_tnf_probe


def _get_license_token_not_found_since() -> TTLCache[str, float]:
    """First-sighting timestamps for in-flight ``token_not_found`` grace windows.

    Keyed by token hash → ``time.monotonic()`` of the first ``token_not_found``
    seen for that token. TTL matches the stale cache so a token that stops being
    introspected is forgotten and a genuinely re-issued token gets a fresh
    window. See :func:`_token_not_found_grace`.
    """
    global _license_token_not_found_since
    if _license_token_not_found_since is None:
        maxsize = int(os.getenv("CACHE_LICENSE_MAXSIZE", "1000"))
        ttl = int(os.getenv("CACHE_LICENSE_STALE_TTL", "900"))
        _license_token_not_found_since = TTLCache(maxsize=maxsize, ttl=ttl)
    return _license_token_not_found_since


# LICENSE_INTROSPECT_URL: env override wins, otherwise built from
# discovery.service_url("licenses"). Discovery is the single source of truth
# for in-cluster service URLs (see w2v_common.discovery +
# gitops-production/docs/config-externalization-policy.md). Resolved at
# module import; tests that need a different value should patch the symbol or
# set the LICENSE_INTROSPECT_URL env var before import.
LICENSE_INTROSPECT_URL = os.getenv(
    "LICENSE_INTROSPECT_URL",
    f"{service_url('licenses')}/license/introspect",
)

# Wall-clock budget for a single introspection POST. Default 10s (was a
# hardcoded 5s): on a CPU-saturated host (e.g. mini2 mid-kokoro-synth) the event
# loop can starve long enough that a tighter budget fires before the licenses
# endpoint replies, surfacing as a spurious 401. Tunable via env for hosts with
# a faster/slower licenses path.
LICENSE_INTROSPECT_TIMEOUT = float(os.getenv("LICENSE_INTROSPECT_TIMEOUT", "10.0"))

# REMOVED 2026-08-26 — ``DEFAULT_TENANT_ID`` (env AUTH_DEFAULT_TENANT_ID /
# DEFAULT_TENANT_ID), which defaulted to the nil-UUID sentinel and was applied
# on the trusted-network synthetic-identity path whenever ``X-Tenant-Id`` was
# absent or whitespace-only. Product-owner ruling: a tenant_id may never hold a
# sentinel, and a configurable default tenant is a sentinel with a config knob.
# That path now rejects the request (401 TENANT_REQUIRED) — see the
# trusted-network lane in ``dispatch``.

# TRUSTED_CIDRS defaults to EMPTY — no source IP is trusted unless the operator
# names the networks explicitly (w2v cell plan B22, "Trusted-network lane").
# The previous default (127.0.0.1/32,::1/128,172.16.0.0/12,10.0.0.0/8) made
# every container on a docker-compose bridge (172.16/12) or a customer LAN
# (10/8) "trusted" the moment AUTH_TRUST_INCLUSTER_IDENTITY or
# AUTH_TRUST_MESH_LICENSE_IDENTITY was switched on — the network a service
# happens to land on silently decided who could assert a tenant. Both lanes
# now require BOTH the flag AND an explicit CIDR list; either alone is inert.
#
# 192.168.0.0/16 deserves a specific warning if an operator adds it: it was
# the SSR-proxy bypass vector in the 2026-04-18 incident (an external caller
# routed through the in-cluster SSR proxy presented a 192.168.x source IP and
# got minted a synthetic in-cluster identity). k3s pod/service traffic is
# 10.42.0.0/16 / 10.43.0.0/16 — name exactly what the deployment needs.
_TRUSTED_CIDRS_RAW = os.getenv("TRUSTED_CIDRS", "")
TRUSTED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network(cidr.strip(), strict=False)
    for cidr in _TRUSTED_CIDRS_RAW.split(",")
    if cidr.strip()
]


# Whether to mint a synthetic in-cluster identity for unauthenticated requests
# whose source IP falls inside TRUSTED_NETWORKS.
#
# Default flipped to fail-closed in 0.6.9 (2026-04-27 migration). Phases 1 and
# 2 of the trust-incluster removal are shipped:
#   * Phase 1 (0.6.7): BaseServiceClient propagates license tokens via
#     X-Tenant-Id, so orchestrator -> downstream calls carry identity.
#   * Phase 2 (0.6.8): bare httpx clients in bdd_playwright_service and
#     content_service updated to send license tokens explicitly.
# All 14 service lockfiles bumped to 0.6.8, so the trusted-network fallback
# is no longer load-bearing in any caller. Default is now "false" to fail
# closed against unauthenticated in-cluster traffic. Operators that genuinely
# need the legacy fail-open behavior (e.g. mid-migration rollback) can opt
# back in by setting AUTH_TRUST_INCLUSTER_IDENTITY=true.
#
# 0.6.10: AuthSettings.trust_incluster_identity is the single source of
# truth; the AUTH_TRUST_INCLUSTER_IDENTITY env var still feeds the field
# via Pydantic settings (env_prefix="AUTH_"). The lookup happens at call
# time so test fixtures and hot-reload paths see fresh values.
def _trust_incluster_identity() -> bool:
    """Read trust_incluster_identity from AuthSettings at call time.

    AuthSettings reads AUTH_TRUST_INCLUSTER_IDENTITY from the environment via
    Pydantic settings, so env-var-driven config keeps working unchanged.
    Constructed fresh per call so tests using ``monkeypatch.setenv`` (and
    operators that mutate env at runtime) see updated values without having
    to reload this module.
    """
    from w2v_common.config import AuthSettings

    return AuthSettings().trust_incluster_identity


def _delegating_subjects() -> frozenset[str]:
    """Subjects allowed to assert an effective tenant via X-Effective-Tenant-Id.

    Read from AuthSettings at call time (env AUTH_DELEGATING_SUBJECTS, CSV) so
    test fixtures and runtime config changes are picked up without a reload —
    mirrors ``_trust_incluster_identity``. Empty (default) disables delegation.
    """
    from w2v_common.config import AuthSettings

    raw = AuthSettings().delegating_subjects or ""
    return frozenset(s.strip() for s in raw.split(",") if s.strip())


def _actor_asserting_subjects() -> frozenset[str]:
    """Subjects allowed to assert WHO ACTED via X-Actor-Id (licence lane only).

    Env AUTH_ACTOR_ASSERTING_SUBJECTS (CSV), read at call time like
    :func:`_delegating_subjects`. ⛔ Deliberately a SEPARATE list: being trusted
    to name the actor must not imply being trusted to re-scope the tenant (and
    vice versa), so AUTH_DELEGATING_SUBJECTS confers NO actor trust. Empty
    (default) = nobody may assert an actor.
    """
    from w2v_common.config import AuthSettings

    raw = AuthSettings().actor_asserting_subjects or ""
    return frozenset(s.strip() for s in raw.split(",") if s.strip())


def _is_actor_asserting_subject(subject: object) -> bool:
    """True when an introspected licence subject is on AUTH_ACTOR_ASSERTING_SUBJECTS."""
    return bool(subject) and str(subject) in _actor_asserting_subjects()


def _effective_tenant_unbilled_bases() -> tuple[str, ...]:
    """Path prefixes on which a delegator's missing effective-tenant is benign.

    Read from AuthSettings at call time (env AUTH_EFFECTIVE_TENANT_UNBILLED_PREFIXES,
    CSV) — mirrors :func:`_delegating_subjects`. Returned as BASE paths (trailing
    ``/`` stripped) for :func:`matches_public_prefix`, so matching is at a segment
    boundary. Empty (default) ⇒ no path is exempt from the missing-header signal.
    """
    from w2v_common.config import AuthSettings

    raw = AuthSettings().effective_tenant_unbilled_prefixes or ""
    return tuple((p.strip().rstrip("/") or "/") for p in raw.split(",") if p.strip())


# ---------------------------------------------------------------------------
# Platform licensing conventions (license-authz ext-authz gate).
#
# The platform gate reads the licence token from ``Authorization: Bearer
# <opaque>`` ONLY (license-authz ``main.py`` 1606/1616); it never looks at
# ``x-tenant-id`` inbound. On ALLOW it INJECTS a downstream header set —
# ``x-tenant-id`` (BARE tenant UUID), ``x-license-id``, ``x-subject-id``,
# ``x-app-id``, ``x-plan-id``, ``x-plan-code``, ``x-app-key``, ``x-actor-id``,
# ``x-caller-client-id``, ``x-caller-roles``.
#
# w2v's own convention is the opposite one: the licence token travels in
# ``X-Tenant-Id: Bearer <opaque>``. So ``X-Tenant-Id`` means TWO different
# things depending on the hop — a CREDENTIAL before the gate, a bare
# IDENTITY ASSERTION after it. Both shapes are handled, and they are told
# apart by the only thing that distinguishes them on the wire: the
# ``Bearer `` prefix. A value with the prefix is a credential and is
# introspected; a bare UUID is never treated as a credential.
# ---------------------------------------------------------------------------

#: Bare tenant UUID injected downstream by license-authz on ALLOW.
MESH_TENANT_HEADER = "x-tenant-id"
#: Licence row id injected downstream by license-authz on ALLOW.
MESH_LICENSE_HEADER = "x-license-id"
#: Licences Subject UUID injected downstream by license-authz on ALLOW.
MESH_SUBJECT_HEADER = "x-subject-id"
#: Key of the app the gated licence was issued for, injected by license-authz.
MESH_APP_KEY_HEADER = "x-app-key"
#: Acting principal injected downstream by license-authz on ALLOW.
MESH_ACTOR_HEADER = "x-actor-id"
#: HMAC marker proving the injected set really came from the gate.
#: license-authz emits it from its ``src/vouch.py``; see ``_dispatch_mesh_vouched``.
MESH_VOUCH_HEADER = "x-license-vouch"  # a header NAME, not a secret  # nosec B105

# ---------------------------------------------------------------------------
# X-License-Vouch verification — MUST stay byte-identical to the producer.
#
# Producer of record: license-authz ``src/vouch.py`` (``compute_vouch`` /
# ``canonical_message`` / ``verify_vouch``). This block is a deliberate,
# documented MIRROR of that module, not an independent design. Any change on
# either side is a wire break; the version tag below is how a verifier refuses
# a shape it does not understand instead of silently mis-verifying.
#
# Wire format:  ``v1:<unix_ts>:<nonce_hex>:<hmac_sha256_hex>``
#
# Canonical message (the exact bytes the MAC is taken over) — newline-joined,
# FIXED order, never sorted at runtime::
#
#     v1
#     <unix_ts>
#     <nonce_hex>
#     x-tenant-id=<value>
#     x-license-id=<value>
#     ...one line per MESH_VOUCHED_HEADERS entry, in that exact order...
#
# SUBTLETIES MIRRORED EXACTLY (each one silently breaks verification if
# "improved"):
#   * ``<unix_ts>`` and ``<nonce_hex>`` are INPUTS to the MAC, so neither can be
#     edited on the wire without invalidating it.
#   * An ABSENT header binds as the EMPTY STRING (``headers.get(n) or ""``), so
#     "header omitted" and "header present but empty" produce the SAME MAC and
#     neither can be swapped for the other. This matters in practice: the gate
#     omits ``x-actor-id`` entirely when there is no acting human, and Envoy
#     drops empty-valued headers on the wire — both arrive here as absent, and
#     both must still verify.
#   * Values are taken RAW — NOT ``.strip()``ed, NOT case-folded. The producer
#     MACs the value it put in its own dict. Stripping here would be a no-op
#     today only because license-authz's ``_SAFE_HEADER_RE`` /
#     ``_SAFE_LIST_HEADER_RE`` already reject whitespace; relying on that
#     coincidence instead of mirroring the producer is how drift starts.
#   * The framing is unambiguous for the same reason: no vouched value can
#     contain a newline or an ``=``, because the producer computes the vouch
#     AFTER that validation loop.
#   * Header lookup is case-insensitive here (Starlette ``Headers``) and
#     lowercase at the producer, so the NAMES in the message always match.
#   * The MAC is the lowercase hex digest of HMAC-SHA256 over the UTF-8 message,
#     keyed by the UTF-8 secret.
#
# Appending to MESH_VOUCHED_HEADERS changes every computed vouch — that is a
# breaking change and requires a version bump on BOTH sides.
# ---------------------------------------------------------------------------

#: Vouch format version. Mirrors ``vouch.VOUCH_VERSION``.
MESH_VOUCH_VERSION = "v1"

#: Every identity header license-authz binds into the MAC, in its FIXED order.
#: Mirrors ``vouch.VOUCHED_HEADERS`` — order is part of the wire contract.
#: The FULL set is signed (not just tenant/licence/subject) because the full set
#: is what an in-mesh peer can forge: flipping ``x-caller-roles`` or
#: ``x-app-key`` alone would otherwise be a free privilege escalation under an
#: otherwise-valid vouch.
MESH_VOUCHED_HEADERS: tuple[str, ...] = (
    "x-tenant-id",
    "x-license-id",
    "x-subject-id",
    "x-app-id",
    "x-plan-id",
    "x-plan-code",
    "x-app-key",
    "x-actor-id",
    "x-caller-client-id",
    "x-caller-roles",
)

#: Default freshness window, in seconds, applied to the MAC-bound timestamp.
#:
#: license-authz DELIBERATELY does not pick this — the acceptable skew is the
#: consumer's policy — so w2v picks it here.
#:
#: WHY REPLAY MATTERS: the vouch binds the identity tuple, so a captured header
#: cannot be re-aimed at a DIFFERENT tenant. What it can do without a window is
#: be replayed FOREVER for the SAME tenant. Any in-mesh peer that observes one
#: gated request (a log line, a proxy, a sidecar dump, a crash artefact) could
#: then replay that exact header set indefinitely and act as that tenant long
#: after the licence was revoked, downgraded, or the subject removed — the gate
#: is the only party that re-checks the licence, and a replay never reaches it.
#: Bounding the timestamp turns a permanent credential into a short-lived one.
#:
#: 60s is the default: generous enough for a real gate→pod hop (sub-millisecond)
#: plus ordinary NTP skew between k3s nodes, tight enough that a captured vouch
#: is useless within a minute. Applied SYMMETRICALLY — a timestamp too far in
#: the FUTURE is refused too, so a clock-skewed or forward-dated vouch cannot
#: buy itself a longer life than the window allows.
MESH_VOUCH_MAX_AGE_DEFAULT = 60

#: Which credential lane authenticated a request. ``lane`` in
#: {authorization-bearer, legacy-x-tenant-id, mesh-vouched}. The
#: ``legacy-x-tenant-id`` series is the deprecation burn-down signal: callers
#: are flipped to ``Authorization`` in a LATER release, and this counter is how
#: an operator knows when that flip is actually complete per service.
_LANE_COUNTER_NAME = "w2v_auth_license_lane_total"


def _lane_counter() -> Counter:
    """Register (or re-use) the lane counter.

    ``importlib.reload`` on this module is a real pattern in the test suite (the
    TRUSTED_CIDRS override test reloads it twice), and a second ``Counter(...)``
    with the same name raises ``Duplicated timeseries`` against the default
    ``REGISTRY``. Re-using the already-registered collector keeps a reload from
    turning an unrelated test red — the same trap the observability module
    avoids by never being reloaded.
    """
    try:
        return Counter(
            _LANE_COUNTER_NAME,
            "Requests authenticated per licence-credential lane.",
            ["lane"],
        )
    except ValueError:
        existing = REGISTRY._names_to_collectors.get(_LANE_COUNTER_NAME)
        if not isinstance(existing, Counter):  # pragma: no cover — defensive
            raise
        return existing


auth_license_lane_total = _lane_counter()

#: A caller stamped ``X-Effective-Tenant-Id`` but its subject is NOT in
#: ``AUTH_DELEGATING_SUBJECTS``, so the receiver keeps the CALLER's own tenant.
#: That behaviour is correct and stays observe-only (a 403 would break live
#: narration for any currently-unlisted sender, and an unlisted sender IS the
#: misconfiguration this path anticipates) — but until now the only signal was
#: a warning naming the subject alone, so an operator could not tell WHICH
#: customer was billed and which one was under-billed.
#:
#: ⛔ UNLABELLED, deliberately. This fires for ANY non-delegating caller that
#: stamps the header, and the population of such callers is every end user on
#: the fleet — a proxy misconfigured to stamp it would mint one series per
#: subject and blow up ``/metrics``. The ``subject`` (and both tenant ids) are
#: named in the accompanying ``logger.warning``, which is where unbounded
#: identifiers belong; the metric only has to answer "is this happening, and how
#: often". Contrast ``w2v_auth_effective_tenant_missing_total`` below, which
#: KEEPS its ``subject`` label because it can only fire for the small,
#: explicitly-allowlisted ``AUTH_DELEGATING_SUBJECTS`` set.
_EFFECTIVE_TENANT_REJECTED_NAME = "w2v_auth_effective_tenant_rejected_total"


def _effective_tenant_rejected_counter() -> Counter:
    """Register (or re-use) the rejected-delegation counter.

    Same duplicate-registration guard as :func:`_lane_counter` — this module is
    reloaded by the test suite and a second ``Counter(...)`` with the same name
    raises ``Duplicated timeseries`` against the default ``REGISTRY``.
    """
    try:
        return Counter(
            _EFFECTIVE_TENANT_REJECTED_NAME,
            "Effective-tenant assertions rejected because the subject is not delegating.",
        )
    except ValueError:
        existing = REGISTRY._names_to_collectors.get(_EFFECTIVE_TENANT_REJECTED_NAME)
        if not isinstance(existing, Counter):  # pragma: no cover — defensive
            raise
        return existing


auth_effective_tenant_rejected_total = _effective_tenant_rejected_counter()


def _record_effective_tenant_rejected() -> None:
    """Best-effort observation — must never break a request.

    Takes no subject: the series is unlabelled on purpose (see the note above).
    """
    try:
        auth_effective_tenant_rejected_total.inc()
    except Exception:  # noqa: BLE001
        logger.debug("effective_tenant_rejected_metric_failed", exc_info=True)


#: An ALLOWLISTED delegating caller sent NO ``X-Effective-Tenant-Id`` (absent or
#: empty). The receiver keeps the SERVICE ACCOUNT's own tenant, so the real
#: customer's run is billed to the service account and the customer is never
#: charged. This is NOT the same event as
#: ``w2v_auth_effective_tenant_rejected_total`` — that one counts a header that
#: WAS sent by a caller not entitled to send it; this one counts a header that a
#: caller entitled (and expected) to send simply did not. A shared counter with
#: a ``reason`` label would mix a spoof/misconfig signal with a stamping bug, so
#: it is a sibling series with its own help text. Labelled by ``subject`` — and
#: unlike the rejected counter above this label IS bounded: only a subject in
#: ``AUTH_DELEGATING_SUBJECTS`` can reach this branch, i.e. a handful of service
#: accounts. The kept tenant is an unbounded customer id and stays in the log.
#:
#: ⚠️ Observe-only on purpose: a delegating caller that forgets the header must
#: keep working (raising here would 401 every unstamped background run).
_EFFECTIVE_TENANT_MISSING_NAME = "w2v_auth_effective_tenant_missing_total"


def _effective_tenant_missing_counter() -> Counter:
    """Register (or re-use) the missing-header counter.

    Same duplicate-registration guard as :func:`_lane_counter` — this module is
    reloaded by the test suite and a second ``Counter(...)`` with the same name
    raises ``Duplicated timeseries`` against the default ``REGISTRY``.
    """
    try:
        return Counter(
            _EFFECTIVE_TENANT_MISSING_NAME,
            "Delegating callers that sent no effective-tenant header, so the "
            "service-account tenant was billed instead of the real customer.",
            ["subject"],
        )
    except ValueError:
        existing = REGISTRY._names_to_collectors.get(_EFFECTIVE_TENANT_MISSING_NAME)
        if not isinstance(existing, Counter):  # pragma: no cover — defensive
            raise
        return existing


auth_effective_tenant_missing_total = _effective_tenant_missing_counter()


def _record_effective_tenant_missing(subject: str) -> None:
    """Best-effort observation — must never break a request."""
    try:
        auth_effective_tenant_missing_total.labels(subject=subject).inc()
    except Exception:  # noqa: BLE001
        logger.debug("effective_tenant_missing_metric_failed", exc_info=True)


#: One-shot flags so a per-request deprecation/security warning does not become
#: a per-request log line. The counter above carries the volume; the log carries
#: the explanation exactly once per process.
_warned_legacy_license_header = False
_warned_mesh_vouch_unsecured = False


def _record_legacy_license_header() -> None:
    """Count + (once) warn that a caller still sends ``X-Tenant-Id: Bearer``.

    DEPRECATED, NOT REMOVED. Every existing w2v caller uses this shape; removing
    it now would 401 all of them. It is kept working verbatim and burned down by
    the ``lane="legacy-x-tenant-id"`` counter.
    """
    global _warned_legacy_license_header
    auth_license_lane_total.labels(lane="legacy-x-tenant-id").inc()
    if not _warned_legacy_license_header:
        _warned_legacy_license_header = True
        logger.warning(
            "DEPRECATED licence-credential shape: 'X-Tenant-Id: Bearer <token>'. "
            "The platform's license-authz gate reads the token from "
            "'Authorization: Bearer <token>' only; move callers to that header. "
            "The legacy shape still authenticates and will keep doing so until a "
            "later release (burn-down metric: w2v_auth_license_lane_total"
            '{lane="legacy-x-tenant-id"}).'
        )


def _env_flag(name: str) -> bool:
    """Read a boolean env var at call time (so monkeypatch/hot-reload is seen)."""
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _jwt_tenant_required() -> bool:
    """Whether a verified JWT with no ``tenant_id`` claim is rejected (P2 tenancy).

    ``AUTH_JWT_TENANT_REQUIRED``. **Defaults TRUE** — today's behaviour for
    every service: no claim → 401 ``JWT_INVALID``. Only an explicit false value
    (``0``/``false``/``no``/``off``) turns it off; anything unrecognised stays
    fail-closed. When off, a verified JWT whose ``tenant_id`` claim is ABSENT
    (missing / empty) yields an authenticated identity with NO tenant, and the
    app's own membership middleware must resolve one. A claim that is PRESENT
    but not a valid tenant UUID is still rejected — opt-out relaxes "the token
    must name a tenant", never "a named tenant must be real". Only the
    orchestrator, which installs that membership middleware, may turn it off.
    """
    raw = (os.getenv("AUTH_JWT_TENANT_REQUIRED") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _claim_is_absent(value: object) -> bool:
    """A ``tenant_id`` claim that is missing, ``None`` or blank text."""
    return value is None or (isinstance(value, str) and not value.strip())


#: Prefix of a w2v tenant API key (``w2v_sk_<prefix>_<secret>``). A bearer
#: starting with this is NEVER submitted to licence introspection: it goes to
#: the verifier the app registered with :func:`register_api_key_verifier`.
TENANT_API_KEY_PREFIX = "w2v_sk_"  # nosec B105  # pragma: allowlist secret — a prefix

#: ``async verifier(token) -> (tenant_id, key_id) | None``. ``None`` means the
#: key is unknown, revoked or expired; raising means "could not check".
ApiKeyVerifier = Callable[[str], Awaitable[tuple[_uuid.UUID | str, str] | None]]

_api_key_verifier: ApiKeyVerifier | None = None


def register_api_key_verifier(verifier: ApiKeyVerifier | None) -> None:
    """Register (or with ``None`` clear) the app's tenant API-key verifier.

    Process-wide: one verifier per service, registered at startup by the app
    that owns the ``tenant_api_keys`` table (the orchestrator). With no
    verifier registered every ``w2v_sk_`` bearer is a 401 — a service that has
    no key store cannot authenticate a key, and must not fall back to licence
    introspection (that would ship the secret to the licenses service).
    """
    global _api_key_verifier
    _api_key_verifier = verifier


def get_api_key_verifier() -> ApiKeyVerifier | None:
    """The currently registered tenant API-key verifier, if any."""
    return _api_key_verifier


def _trust_mesh_license_identity() -> bool:
    """Whether to honour license-authz's INJECTED identity headers.

    ``AUTH_TRUST_MESH_LICENSE_IDENTITY``. **Defaults OFF** — read
    :func:`AuthMiddleware._dispatch_mesh_vouched` for why an unconditional
    default-on would be a privilege-escalation primitive. Read via ``os.getenv``
    at call time, matching this module's existing ``TRUSTED_CIDRS`` /
    ``AUTH_INTROSPECT_LOCK_MAXSIZE`` precedent, so no config-model change is
    needed to turn it on.
    """
    return _env_flag("AUTH_TRUST_MESH_LICENSE_IDENTITY")


def _mesh_vouch_secret() -> str:
    """HMAC key for the ``X-License-Vouch`` marker the gate emits.

    ``AUTH_MESH_VOUCH_SECRET``.

    ⚠️ **ITS VALUE MUST EQUAL license-authz's ``LICENSE_VOUCH_SECRET``.** The
    env var names differ (each side names it in its own vocabulary) but they are
    ONE key: it keys an HMAC, and HMAC verification needs the same key that
    signed. Set them from the SAME secret material, or every gated request is
    refused with ``MESH_VOUCH_REQUIRED`` while both sides look configured.

    This is NOT a bare secret to echo back: the header carries an HMAC over the
    injected identity tuple, and the secret itself never travels. Comparing the
    presented header against this value directly (the pre-#176 behaviour) can
    never match a producer-format vouch.

    Empty (the default) means the lane rests on the peer-IP check alone, which
    this module logs about once per process.
    """
    return (os.getenv("AUTH_MESH_VOUCH_SECRET") or "").strip()


def _mesh_vouch_max_age() -> int:
    """Freshness window in seconds for the MAC-bound timestamp.

    ``AUTH_MESH_VOUCH_MAX_AGE_SECONDS``; see
    :data:`MESH_VOUCH_MAX_AGE_DEFAULT` for the window and why replay matters.
    A non-numeric or non-positive value falls back to the default rather than
    disabling the check — "0 means off" would make a typo into a silent
    downgrade to an unbounded-replay credential.
    """
    raw = (os.getenv("AUTH_MESH_VOUCH_MAX_AGE_SECONDS") or "").strip()
    if not raw:
        return MESH_VOUCH_MAX_AGE_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "AUTH_MESH_VOUCH_MAX_AGE_SECONDS=%r is not an integer; using %ds",
            raw,
            MESH_VOUCH_MAX_AGE_DEFAULT,
        )
        return MESH_VOUCH_MAX_AGE_DEFAULT
    if value <= 0:
        logger.warning(
            "AUTH_MESH_VOUCH_MAX_AGE_SECONDS=%d is not positive; using %ds",
            value,
            MESH_VOUCH_MAX_AGE_DEFAULT,
        )
        return MESH_VOUCH_MAX_AGE_DEFAULT
    return value


def _mesh_vouch_canonical_message(headers: Mapping[str, str], *, timestamp: int, nonce: str) -> str:
    """Rebuild the EXACT byte string license-authz took the MAC over.

    Mirrors ``vouch.canonical_message`` line for line. See the block comment
    above :data:`MESH_VOUCH_VERSION` for every subtlety this must preserve —
    fixed field order, ``absent == empty``, raw (unstripped) values.
    """
    lines = [MESH_VOUCH_VERSION, str(timestamp), nonce]
    lines.extend(f"{name}={headers.get(name) or ''}" for name in MESH_VOUCHED_HEADERS)
    return "\n".join(lines)


def _verify_mesh_vouch(
    headers: Mapping[str, str], presented: str, secret: str, *, now: int
) -> tuple[bool, str]:
    """Verify an ``X-License-Vouch`` against the request's OWN injected headers.

    Returns ``(ok, reason)``; ``reason`` is a short non-sensitive token for the
    refusal log. Neither the secret nor either MAC is ever returned or logged.

    Recomputes the canonical message from the headers actually present on THIS
    request, so a vouch minted over a different identity tuple cannot be
    transplanted onto one the peer forged.
    """
    if not secret or not presented:
        return False, "absent"
    parts = presented.split(":")
    if len(parts) != 4:
        return False, "malformed"
    version, ts_raw, nonce, mac = parts
    if version != MESH_VOUCH_VERSION:
        # Refuse an unknown shape rather than guess at it.
        return False, "unsupported_version"
    try:
        timestamp = int(ts_raw)
    except ValueError:
        return False, "malformed_timestamp"
    # Freshness BEFORE the MAC check is safe (the timestamp is MAC-bound, so a
    # tampered one fails below anyway) and keeps an obviously-stale replay from
    # costing a hash. Symmetric: future-dated is refused too.
    if abs(now - timestamp) > _mesh_vouch_max_age():
        return False, "stale"
    expected = hmac.new(
        secret.encode("utf-8"),
        _mesh_vouch_canonical_message(headers, timestamp=timestamp, nonce=nonce).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    # hmac.compare_digest, never ``==``: a short-circuiting comparison leaks the
    # length of the matching prefix and makes the MAC forgeable byte-by-byte.
    if not hmac.compare_digest(expected, mac):
        return False, "mac_mismatch"
    return True, "ok"


def _warn_mesh_vouch_unsecured_once() -> None:
    global _warned_mesh_vouch_unsecured
    if not _warned_mesh_vouch_unsecured:
        _warned_mesh_vouch_unsecured = True
        logger.warning(
            "AUTH_TRUST_MESH_LICENSE_IDENTITY is ON with no AUTH_MESH_VOUCH_SECRET: "
            "the license-authz identity lane is trusting headers that any peer "
            "inside TRUSTED_CIDRS could also set (the platform's header-strip "
            "filter is north-south only). Restrict the port to the waypoint, or "
            "set AUTH_MESH_VOUCH_SECRET on both the gate and this service."
        )


def _tenant_or_unauthorized(
    request: Request,
    raw: object,
    *,
    source: str,
    code: str,
    message: str,
    log_context: str = "",
) -> _uuid.UUID | Response:
    """Parse a tenant to a UUID, or build the 401 that rejects the request.

    Returning the ``Response`` (rather than raising) keeps ``dispatch`` a flat
    sequence of lanes. Every auth lane funnels through here, so there is
    exactly ONE place that decides what "no usable tenant" does — and it is
    always reject, never substitute.
    """
    try:
        return parse_tenant_id(raw, source=source)
    except TenantError as exc:
        logger.warning("%s rejected: %s%s", source, exc, log_context)
        return _unauthorized(request, code, message)


def _resolve_jwt_tenant(
    request: Request, claims: Mapping[str, Any], sub: str
) -> _uuid.UUID | Response | None:
    """The JWT lane's tenant: a UUID, ``None`` (opt-out, claim absent), or the 401.

    P2 tenancy opt-out (``AUTH_JWT_TENANT_REQUIRED=false``, orchestrator only):
    an ABSENT claim yields a tenant-LESS identity and the app's membership
    middleware resolves the tenant from w2v's own table. A PRESENT-but-invalid
    claim is rejected either way. Never a fallback to ``sub``.
    """
    raw = claims.get("tenant_id")
    if _claim_is_absent(raw) and not _jwt_tenant_required():
        return None
    return _tenant_or_unauthorized(
        request,
        raw,
        source="JWT tenant_id claim",
        code="JWT_INVALID",
        message="JWT has no valid tenant",
        log_context=f" (sub={sub}; a self-named tenant would bypass isolation)",
    )


def _resolve_license_identity(
    request: Request, result: _Introspection
) -> tuple[_uuid.UUID, str, str | None] | Response:
    """Resolve ``(tenant, subject, license)`` from an ACTIVE introspection, or the 401.

    Three-piece auth invariant: an active license MUST carry a real registered
    tenant. An empty, whitespace, or non-UUID value is always a bug (a
    misconfigured license, a buggy introspector, a legacy slug), never a valid
    state — accepting it would authenticate an "empty tenant" and silently
    bypass tenant isolation downstream. (We do not also require ``subject_id``:
    tenant-scoped service-account tokens legitimately carry no subject.)
    """
    tenant_id = _tenant_or_unauthorized(
        request,
        result.get("tenant_id"),
        source="license introspection",
        code="INVALID_TOKEN",
        message="license token has no valid tenant",
        log_context=" (active=true but an empty or slug tenant would bypass isolation)",
    )
    if isinstance(tenant_id, Response):
        return tenant_id
    subject_id = result.get("subject_id") or str(tenant_id)
    license_id = result.get("license_id")
    # Effective-tenant delegation (see _apply_effective_tenant): a trusted
    # service account may act on behalf of a run's real tenant via the
    # X-Effective-Tenant-Id side-channel, so cost attribution + BYO follow the
    # run tenant, not the service-account tenant. Inert unless
    # AUTH_DELEGATING_SUBJECTS lists the introspected subject.
    try:
        return _apply_effective_tenant(
            request, result.get("subject_id"), tenant_id, subject_id, license_id
        )
    except TenantError as exc:
        # A DELEGATING service account asserted a non-UUID effective tenant.
        # Never fall back to the service-account tenant here: that is the
        # mis-attribution bug (the SA gets billed for a customer run). Reject
        # and make the delegator fix its header.
        logger.warning("%s rejected: %s", EFFECTIVE_TENANT_HEADER, exc)
        return _unauthorized(
            request, "INVALID_TENANT", "X-Effective-Tenant-Id is not a valid tenant"
        )


def _apply_effective_tenant(
    request: Request,
    auth_subject: str | None,
    tenant_id: _uuid.UUID,
    subject_id: str,
    license_id: str | None,
) -> tuple[_uuid.UUID, str, str | None]:
    """Re-scope identity to X-Effective-Tenant-Id for a delegating service account.

    The Bearer license token authenticates the CALLER (a service account whose
    token introspects to the SA tenant). A trusted service account — its
    introspected ``auth_subject`` is in AUTH_DELEGATING_SUBJECTS — may act on
    behalf of another tenant by sending ``X-Effective-Tenant-Id``. This threads a
    background run's REAL tenant (not the SA tenant) downstream for cost
    attribution + BYO resolution. A customer token's subject is never a delegator,
    so a customer can never spoof another tenant; empty config ⇒ no-op.

    ⛔ **A licence may never survive a tenant re-scope.** The ``license_id``
    argument is the licence the INBOUND Bearer token introspected to — the
    delegating service account's OWN licence, owned by the SA's tenant. Once
    ``tenant_id`` has been re-scoped to the effective tenant, that licence
    belongs to somebody else, so when the delegator forwards no licence the
    answer is ``None`` — an honest absence — and never the SA's licence. This is
    the same provenance rule ``set_identity_context()`` already applies to
    ``tenant_source``: a value bound to one tenant must never follow the context
    onto a different one. ``None`` is the well-trodden path, not a new one: the
    Keycloak-JWT lane and the trusted-network lane below already bind
    ``license_id=None``, and billing's tenant-keyed ``resolve_license()`` runs
    precisely on the rows where ``license_id IS NULL``.

    ⚠️ ``subject_id`` deliberately does NOT get this rule. Subject is *who
    acted*, not *who pays* — a delegating SA genuinely is the actor when it
    forwards no run subject — and billing carries it as ``actor_id``, separate
    from the licence. Only the licence is a tenant-bound entitlement.

    ⛔ **Provenance marker (S95).** Downstream (w2v_orchestrator's
    ``require_known_tenant``) needs to tell "this tenant is real because a
    trusted delegator just vouched for it" apart from "this tenant is real
    because IT introspected/registered on its own" — a delegated run's real
    tenant legitimately has no ``tenants`` registry row or ``projects`` row
    yet (see ``AdminRepository.tenant_switchable``), and treating that as
    unknown is the S95 bug (403 on every marketing→w2v call). So the ONE
    branch below that actually accepts a re-scope stamps
    ``request.state.tenant_delegated = True``; every other branch — including
    the non-delegating-subject path and the "no header / same tenant" no-op —
    stamps it ``False`` first. This is a plain ``request.state`` attribute,
    not a return value, because the 3-tuple return shape here is pinned by
    ``test_no_return_pairs_a_re_scoped_tenant_with_the_callers_license``
    (an AST guard for issue #170) and must not grow a 4th element.

    Returns the (possibly re-scoped) ``(tenant_id, subject_id, license_id)``.
    """
    # Default-false FIRST, unconditionally, before any branch below. A
    # rejected assertion, a missing/equal header, or an exception raised by
    # parse_tenant_id() below must all leave this False — the accepted-delegation
    # branch is the only place it flips to True.
    request.state.tenant_delegated = False
    if not auth_subject or str(auth_subject) not in _delegating_subjects():
        # A non-delegating caller asserting a DIFFERENT tenant is either a spoof
        # attempt or a receiver missing AUTH_DELEGATING_SUBJECTS config — surface
        # it instead of silently dropping the header. The standard S2S hook stamps
        # this header unconditionally, so an equal-tenant value is a benign no-op
        # on the normal non-delegating path: don't warn on it (would be fleet-wide
        # log spam that drowns the signal a real re-scope attempt should raise).
        eff = (request.headers.get(EFFECTIVE_TENANT_HEADER) or "").strip()
        if eff and eff != str(tenant_id):
            subject = str(auth_subject) if auth_subject else "<missing>"
            _record_effective_tenant_rejected()
            # Name BOTH tenants: the asserted one is the customer whose usage
            # is about to be billed to somebody else, the kept one is who eats
            # it. A subject alone cannot answer either question.
            logger.warning(
                "effective-tenant header rejected (subject not delegating): "
                "subject=%s asserted_tenant=%s kept_tenant=%s",
                subject,
                eff,
                tenant_id,
            )
        return tenant_id, subject_id, license_id
    eff_raw = (request.headers.get(EFFECTIVE_TENANT_HEADER) or "").strip()
    # ⛔ These two used to be ONE condition, and that collapsed two opposite
    # situations into one silent branch. Both keep the caller's own tenant, but
    # only one of them is expected:
    #   * header ABSENT/EMPTY on an ALLOWLISTED delegator — SUSPICIOUS. This
    #     caller is in AUTH_DELEGATING_SUBJECTS precisely because it is supposed
    #     to stamp the header; not stamping it means the run's real customer is
    #     never billed and the service account silently absorbs the cost. It was
    #     the last invisible path on the delegation route: no exception, no log,
    #     no metric.
    #   * header EQUALS the caller's own tenant — legitimate and expected (the
    #     standard S2S hook stamps it unconditionally). Stays COMPLETELY silent:
    #     counting it would bury the signal above under fleet-wide volume.
    if not eff_raw:
        subject = str(auth_subject)
        unbilled = _effective_tenant_unbilled_bases()
        if unbilled and matches_public_prefix(request.url.path, unbilled):
            # A route that never bills (e.g. a worker's claim poll) cannot
            # mis-attribute cost, so the missing header is not the stamping bug
            # the counter/warning exist to surface. Stay out of both — at
            # production poll rates they would bury the real signal.
            logger.debug(
                "effective-tenant header absent on unbilled path: subject=%s path=%s",
                subject,
                request.url.path,
            )
            return tenant_id, subject_id, license_id
        _record_effective_tenant_missing(subject)
        # Warn, never raise: an unstamped delegating call must keep working.
        # Name the subject (which caller failed to stamp) and the kept tenant
        # (who wrongly absorbed the cost) — neither is answerable from the other.
        logger.warning(
            "effective-tenant header missing from a delegating caller: "
            "subject=%s kept_tenant=%s (the service-account tenant is being "
            "billed for a run that belongs to somebody else)",
            subject,
            tenant_id,
        )
        return tenant_id, subject_id, license_id
    if eff_raw == str(tenant_id):
        # A delegating SA making an ORDINARY call for its own tenant is not a
        # delegation — nothing was re-scoped, so the marker stays False.
        return tenant_id, subject_id, license_id
    # A delegating service account asserting a NON-UUID effective tenant is a
    # bug in the delegator, not something to silently ignore back onto the SA
    # tenant (that would mis-bill the service account for a customer run).
    # NOTE: parse_tenant_id() may raise here — the marker set above stays
    # False, which is correct: this assertion was NOT accepted.
    eff_tenant = parse_tenant_id(eff_raw, source=EFFECTIVE_TENANT_HEADER)
    # The assertion is now ACCEPTED: an allowlisted delegator, a present
    # header, naming a different, well-formed tenant. This is the one branch
    # that gets to flip the marker.
    request.state.tenant_delegated = True
    logger.info(
        "effective-tenant delegation: subject=%s acting as tenant=%s (service-account tenant=%s)",
        auth_subject,
        eff_tenant,
        tenant_id,
    )
    # Carry the run's own subject/license when the delegator forwarded them, so
    # per-user metering attributes to the customer rather than the service account.
    fwd_subject = (request.headers.get("X-Subject-Id") or "").strip()
    fwd_license = (request.headers.get("X-License-Id") or "").strip()
    # ⛔ NO `or license_id` HERE — see the docstring. `license_id` is the
    # DELEGATOR's licence and the tenant on the line above is somebody else's.
    # An empty forwarded X-License-Id is the NORMAL state whenever the
    # delegating run's own identity carries no licence (both the Keycloak-JWT
    # and trusted-network lanes bind license_id=None, and get_identity_headers()
    # then stamps `X-License-Id: ""` outbound), so the fallback fired routinely:
    # it billed 724 live cost events to one tenant under another tenant's active
    # licence before it was removed (issue #170).
    #
    # ⚠️ Restoring the fallback to make the "unlicensed" counter look better is
    # the exact regression this guards. Those events are unlicensed; NULL is the
    # instrument telling the truth. `test_delegated_absent_license_is_not_the_sa_license`
    # fails if `or license_id` comes back.
    if not fwd_license:
        logger.info(
            "effective-tenant delegation: no forwarded licence for tenant=%s; "
            "binding licence=None rather than the delegator's own (subject=%s)",
            eff_tenant,
            auth_subject,
        )
    return eff_tenant, fwd_subject or subject_id, fwd_license or None


PUBLIC_ENDPOINTS: set[str] = {
    "/",
    "/health",
    "/health/ready",
    "/health/live",
    "/healthz",
    "/ready",
    "/readyz",
    # Liveness twins of the readiness entries above. `/livez` is the standard
    # k8s liveness path (sibling of `/readyz`); it was originally omitted, so
    # services probing `/livez` got a 401 on liveness -> pod crashloop.
    "/live",
    "/livez",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/health",
    "/api/health/ready",
    "/api/health/live",
    "/favicon.ico",
    "/metrics",
}

PUBLIC_PREFIXES: tuple[str, ...] = (
    "/docs",
    "/redoc",
    "/health/",
    "/api/health/",
)

#: Path segments that make a candidate path UNTRUSTWORTHY as a prefix match.
#: An empty segment is ``//`` (which normalises away and does not route); ``.``
#: and ``..`` are traversal. None of them can appear in a real probe path, and
#: admitting them would let a prefix "match" a path that resolves elsewhere, so
#: the matcher denies rather than guesses.
_UNSAFE_SEGMENTS: frozenset[str] = frozenset({"", ".", ".."})

#: :data:`PUBLIC_PREFIXES` with trailing slashes stripped, so each entry is the
#: BASE segment path a candidate must match at a segment boundary.
PUBLIC_PREFIX_BASES: tuple[str, ...] = tuple((p.rstrip("/") or "/") for p in PUBLIC_PREFIXES)


def matches_public_prefix(path: str, bases: tuple[str, ...] = PUBLIC_PREFIX_BASES) -> bool:
    """Does ``path`` fall under one of ``bases`` at a **segment boundary**?

    A raw ``str.startswith`` is the wrong test for a path prefix and was the
    bug this replaces: ``"/docsomething".startswith("/docs")`` is ``True``, so
    ``/docs-internal/render``, ``/docsomething/anything`` and ``/redocument``
    were all treated as public — unauthenticated by
    :class:`AuthMiddleware` and unbilled by the entitlement gate. No route in
    the fleet occupied those shapes when this was found, so it was a LANDMINE
    rather than a live hole: the first route named ``/docs-export`` would have
    inherited the exemption silently.

    The rule: ``base`` matches ``path`` only when ``path`` **is** ``base`` or
    continues it after a ``/``. ``/health`` therefore covers ``/health/live``
    and ``/health/`` but never ``/healthy``; and a remainder containing an
    empty, ``.`` or ``..`` segment is refused outright, so ``/health/../x``
    does not inherit ``/health``'s exemption.

    Deliberately does NOT change WHICH paths are exempt — only how the prefix
    is matched. ``/health/diagnostics`` stays exempt because ``/health/`` is an
    exempt prefix by design; that is the exemption set, not a matching bug.
    """
    candidate = path.rstrip("/") or "/"
    for base in bases:
        if candidate == base:
            return True
        if not candidate.startswith(base + "/"):
            continue
        remainder = candidate[len(base) + 1 :]
        if all(seg not in _UNSAFE_SEGMENTS for seg in remainder.split("/")):
            return True
    return False


_lanes_warned_no_networks: set[str] = set()


def _warn_lane_without_networks(lane: str) -> None:
    """Warn once per lane per process when a trust lane is ON but inert.

    ``TRUSTED_CIDRS`` defaults to empty, so switching a trust flag on without
    also naming the networks makes the lane refuse every caller. That is the
    safe direction, but it must not be silent: an operator who set the flag
    expects it to do something.
    """
    if TRUSTED_NETWORKS or lane in _lanes_warned_no_networks:
        return
    _lanes_warned_no_networks.add(lane)
    logger.warning(
        "%s is enabled but TRUSTED_CIDRS is empty, so no peer is trusted and the "
        "lane refuses every caller. Set TRUSTED_CIDRS to the exact networks "
        "allowed to use it (there is no default).",
        lane,
    )


def _is_trusted_network(host: str | None) -> bool:
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host)
        return any(addr in net for net in TRUSTED_NETWORKS)
    except ValueError:
        return False


async def _introspect_license(token: str) -> _Introspection | None:
    """Validate a license token via the introspection endpoint.

    Args:
        token: The license token to validate.

    Returns:
        Introspection result dict if successful, None on failure.
    """
    cache = _get_license_cache()
    neg_cache = _get_license_neg_cache()
    stale_cache = _get_license_stale_cache()
    cache_key = f"lic:{hashlib.sha256(token.encode()).hexdigest()[:16]}"

    def _cached() -> _Introspection | None:
        if cache_key in cache:
            logger.debug("License introspection cache hit")
            return cache[cache_key]
        if cache_key in neg_cache:
            logger.debug("License introspection neg-cache hit (active=false)")
            return neg_cache[cache_key]
        return None

    tnf_probe = _get_license_tnf_probe()

    def _cached_or_throttled() -> tuple[bool, _Introspection | None]:
        # (hit?, value). The token_not_found grace path does NOT neg-cache (so
        # recovery is immediate once a rotating token propagates), but that
        # would mean one upstream introspection PER REQUEST for a missing token.
        # tnf_probe caches the transient OUTCOME (warm-stale dict, or None) for
        # a few seconds, so membership — not truthiness — is the hit test: a
        # cached None is still a hit and must short-circuit the upstream call.
        cached = _cached()
        if cached is not None:
            return True, cached
        if cache_key in tnf_probe:
            logger.debug("License introspection token_not_found probe throttle hit")
            return True, tnf_probe[cache_key]
        return False, None

    is_hit, value = _cached_or_throttled()
    if is_hit:
        return value

    # Cache miss: single-flight so a concurrent burst for the same cold token
    # collapses into ONE upstream introspection instead of stampeding the
    # licenses endpoint (see _introspect_locks). Re-check the caches inside the
    # lock — a prior waiter may have populated them while we were blocked.
    lock = _acquire_introspect_lock(cache_key)
    async with lock:
        is_hit, value = _cached_or_throttled()
        if is_hit:
            return value
        return await _introspect_license_uncached(token, cache, neg_cache, stale_cache, cache_key)


async def _introspect_license_uncached(
    token: str,
    cache: TTLCache[str, _Introspection],
    neg_cache: TTLCache[str, _Introspection],
    stale_cache: TTLCache[str, _Introspection],
    cache_key: str,
) -> _Introspection | None:
    """Perform the actual introspection POST and populate the caches.

    Always called under the per-token single-flight lock held by
    :func:`_introspect_license`.

    On a TRANSIENT failure (timeout / connection error / non-200), fall back to
    the last-known-good entry in ``stale_cache`` rather than returning ``None``
    (which the caller maps to a 401). A definitive ``active: false`` from a 200
    response is authoritative and denies immediately. See
    :func:`_get_license_stale_cache`.
    """
    logger.debug("Starting license introspection call")
    start = _time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=LICENSE_INTROSPECT_TIMEOUT) as client:
            resp = await client.post(LICENSE_INTROSPECT_URL, json={"token": token})
            duration_ms = (_time.monotonic() - start) * 1000
            if resp.status_code == 200:
                data: _Introspection = resp.json()
                if data.get("active"):
                    cache[cache_key] = data
                    # Record the good result for stale-while-error fallback.
                    stale_cache[cache_key] = data
                    # Recovery: a later token_not_found flip gets a fresh window.
                    _get_license_token_not_found_since().pop(cache_key, None)
                    logger.debug(
                        "License introspection succeeded: duration=%.1fms active=True",
                        duration_ms,
                    )
                    return data
                # 200 active:false is authoritative for a genuinely bad token
                # (reason token_invalid / license_inactive) — never served stale.
                # EXCEPT reason "token_not_found" during an ADDITIVE rotation is a
                # propagation race, not a rejection — ride it out as transient,
                # bounded from the first sighting. See _token_not_found_grace.
                grace = _token_not_found_grace()
                if data.get("reason") == "token_not_found" and grace > 0:
                    since = _get_license_token_not_found_since()
                    now = _time.monotonic()
                    first = since.get(cache_key)
                    if first is None:
                        since[cache_key] = first = now
                    if now - first < grace:
                        logger.warning(
                            "AUDIT introspect token_not_found treated as transient "
                            "(rotation race) elapsed=%.0fs window=%.0fs",
                            now - first,
                            grace,
                        )
                        # Throttle upstream re-probes: cache the transient
                        # outcome for a few seconds so a busy caller does not POST
                        # licenses once per request while the token propagates.
                        # NOT the neg_cache (which would delay recovery by its
                        # 60s TTL) — this TTL is ~5s.
                        stale = stale_cache.get(cache_key)
                        outcome = stale if stale is not None else None
                        _get_license_tnf_probe()[cache_key] = outcome
                        # Warm replica serves last-known-good; cold replica → None
                        # (caller denies/retries), never neg-cached.
                        return outcome
                    # Window elapsed: re-pin the ORIGINAL first-sighting so its
                    # TTL refreshes under continuous traffic — otherwise the
                    # since-entry evicts and the next token_not_found re-opens a
                    # fresh window, flapping a permanently-missing token forever
                    # instead of settling to an authoritative deny.
                    since[cache_key] = first
                    logger.warning(
                        "introspect token_not_found grace expired window=%.0fs; "
                        "returning authoritative inactive",
                        grace,
                    )
                # Definitive rejection — authoritative, never served stale.
                neg_cache[cache_key] = data
                logger.debug(
                    "License introspection succeeded: duration=%.1fms active=False",
                    duration_ms,
                )
                return data
            logger.warning(
                "License introspection returned status %d: duration=%.1fms",
                resp.status_code,
                duration_ms,
            )
    except (httpx.HTTPError, TimeoutError) as e:
        duration_ms = (_time.monotonic() - start) * 1000
        logger.warning("License introspection failed: %s duration=%.1fms", e, duration_ms)

    # Transient failure (non-200 or exception): serve the last-known-good result
    # if we have one, so a briefly slow/unreachable licenses service does not
    # spuriously 401 a token that was valid moments ago. Cold cache -> None ->
    # caller denies (fail-closed).
    stale = stale_cache.get(cache_key)
    if stale is not None:
        logger.warning(
            "License introspection unavailable; serving stale last-known-good (active=%s)",
            stale.get("active"),
        )
        return stale
    return None


def _unauthorized(request: Request, code: str, message: str) -> JSONResponse:
    """Emit a canonical 401 response in the fleet error-envelope shape.

    Matches the shared shape from ``w2v_common.asgi.error_envelope``
    (``{error: {code, message, status, request_id, timestamp}}``) so every
    error a caller sees — including auth rejections — has the same fields.
    Auth runs *before* ``RequestIdMiddleware``, so the id is read straight off
    the inbound ``X-Request-Id`` header (or generated) rather than from
    ``request.state``; the shape is duplicated inline rather than importing
    ``asgi.error_envelope`` to avoid an ``auth → asgi → auth`` import cycle
    (the asgi factory imports ``AuthMiddleware``). Includes
    ``WWW-Authenticate: Bearer`` per RFC 7235 §3.1.
    """
    return _error_envelope(request, 401, code, message, headers={"WWW-Authenticate": "Bearer"})


def _wrong_app(request: Request) -> JSONResponse:
    """403: the licence is valid but issued for an app this service does not serve (#222)."""
    return _error_envelope(
        request, 403, WRONG_APP_CODE, "license token is not issued for this service's app"
    )


def _service_unavailable(request: Request, code: str, message: str) -> JSONResponse:
    """503 in the :func:`_unauthorized` envelope: "could not check" is not "invalid"."""
    return _error_envelope(request, 503, code, message)


def _error_envelope(
    request: Request,
    status: int,
    code: str,
    message: str,
    *,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    request_id = request.headers.get("x-request-id") or str(_uuid.uuid7())
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "status": status,
                "request_id": request_id,
                "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
            }
        },
        headers=dict(headers) if headers else None,
    )


def _stash_identity(request: Request, user: User) -> None:
    """Stash the resolved identity on ``request.state`` for route handlers.

    ``request.state.user`` is the canonical identity object, but route code
    (and the ``/auth/me`` contract) reads the individual claims as flat
    attributes — the same way ``tenant_id`` already lands on
    ``request.state.tenant_id``. Without these, ``request.state.roles`` is
    never set and every consumer reading it (e.g. orchestrator ``/auth/me``)
    silently sees ``[]``. Mirrors the ``identity_context`` propagation pattern.

    Additive and empty-safe: the license / trusted-network paths build a
    ``User`` with no roles and no email, so this stashes ``roles=[]`` and
    ``email=None`` there without any special-casing.
    """
    request.state.user = user
    request.state.roles = list(user.roles or [])
    request.state.email = user.email
    request.state.email_verified = user.email_verified
    request.state.api_key_id = user.api_key_id
    request.state.name = user.preferred_username
    request.state.auth_realm = user.realm


def _has_legacy_credential(request: Request) -> bool:
    """Does the request carry a credential a non-cell lane would consume?

    Side-effect free (``_extract_license_token`` counts lanes): a non-empty
    ``X-Tenant-Id: Bearer …`` or ``Authorization: Bearer …`` — i.e. a licence
    token, a Keycloak JWT or a tenant API key. Mesh-vouched / trusted-network
    identities are not credentials and do not count.
    """
    for name in ("X-Tenant-Id", "Authorization"):
        value = request.headers.get(name) or ""
        if value[:7].lower() == "bearer " and value[7:].strip():
            return True
    return False


def _verify_cell_header(request: Request, mode: CellAuthMode) -> CellClaims | CellTokenError | None:
    """Verify ``X-W2V-Cell-Token``: claims, the (counted + logged) error, or ``None`` if absent.

    A ``scope=service`` token is admitted only when the route this request
    will hit opted in (``require_cell_service_caller``) for its issuer. The
    token itself is never logged.
    """
    token = (request.headers.get(CELL_TOKEN_HEADER) or "").strip()
    if not token:
        return None
    try:
        return verify_cell_token(token, service_callers=route_cell_service_callers(request.scope))
    except CellTokenError as exc:
        record_cell_rejection(exc.reason, exc.scope)
        logger.warning(
            "cell token rejected (%s, scope=%s, mode=%s) for %s %s",
            exc.reason,
            exc.scope,
            mode.value,
            request.method,
            request.url.path,
        )
        return exc


def _cell_attribution(
    request: Request, claims: CellClaims
) -> tuple[str | None, str | None, str | None]:
    """``(subject, licence, unsigned_subject)`` for a verified TENANT cell token.

    Signed claims present (0.93.0+ minter): they are the ONLY source —
    ``X-Subject-Id`` / ``X-License-Id`` are ignored even when they disagree,
    and a signed ``null`` stays ``None``.

    No claims (0.92.x minter): the headers, as before 0.93.0, but the subject
    is returned as ``unsigned_subject`` — attribution-only, kept out of the
    subject authz reads. The licence stays a billing attribute (no fleet authz
    reads it) so billing continuity is preserved. Counted per caller in
    ``w2v_cell_unsigned_attribution_total``.
    TODO(C-del): delete this fallback (and ``bind_unsigned_attribution_subject``)
    once that counter has been 0 fleet-wide for 24 h.
    """
    if claims.attribution_signed:
        return claims.subject, claims.license_id, None
    record_unsigned_attribution(claims.issuer)
    logger.debug(
        "cell token from %s carries no signed subject/license (pre-0.93.0 minter); "
        "using X-Subject-Id/X-License-Id as attribution only for %s %s",
        claims.issuer,
        request.method,
        request.url.path,
    )
    fwd_subject = (request.headers.get("X-Subject-Id") or "").strip()
    fwd_license = (request.headers.get("X-License-Id") or "").strip()
    return None, fwd_license or None, fwd_subject or None


def _count_cell_auth(
    call_next: Callable[[Request], Awaitable[Response]],
) -> Callable[[Request], Awaitable[Response]]:
    """Wrap ``call_next`` so reaching the handler counts ``w2v_cell_auth_total`` once.

    The lane is read when the handler is reached, i.e. after a lane has
    authenticated: ``cell`` / ``dual`` are stamped on ``request.state`` by the
    cell lane / the dual recorder; anything else authenticated on a legacy lane.
    A refused request (401/503) never reaches ``call_next`` and is not counted.
    """

    async def _counted(req: Request) -> Response:
        lane = getattr(req.state, "cell_auth_lane", CELL_AUTH_LANE_LEGACY)
        scope = (
            SCOPE_NONE
            if lane == CELL_AUTH_LANE_LEGACY
            else str(getattr(req.state, "cell_scope", None) or SCOPE_NONE)
        )
        record_cell_auth(lane, scope)
        return await call_next(req)

    return _counted


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to validate license tokens."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        extra_public_endpoints: Iterable[str] = (),
    ) -> None:
        """Build the auth middleware.

        Args:
            app: The wrapped ASGI application.
            extra_public_endpoints: Service-specific paths that must bypass
                license-token enforcement IN ADDITION to the fleet-wide
                :data:`PUBLIC_ENDPOINTS` (e.g. an OAuth ``/api/auth/callback``
                redirect target that carries no license token). Kept per
                instance rather than mutating the module-level set — mutating
                the shared set would leak the path to every other service that
                shares the process (e.g. a combined test runner). Paths are
                normalized to the same trailing-slash-stripped form
                :meth:`dispatch` compares against.
        """
        super().__init__(app)
        self._extra_public_endpoints: frozenset[str] = frozenset(
            (p.rstrip("/") or "/") for p in extra_public_endpoints
        )
        # P7: a dual/enforce receiver with no CELL_ALLOWED_CALLERS accepts no
        # cell caller (fail closed) — say so once, at startup.
        warn_if_no_cell_callers()
        # Zero-valued cell-auth series from the first scrape, so an absent
        # series means "not instrumented", never "no legacy traffic".
        precreate_cell_auth_series(_delegating_subjects())
        # #222: say once, at startup, when AUTH_LICENSE_APP_KEYS is unset and
        # licences for ANY app are therefore accepted.
        warn_if_license_app_unchecked()
        # Opt-in plaintext /metrics listener (W2V_METRICS_PORT). Every w2v service
        # installs this middleware, so this is the one generic startup hook.
        start_metrics_server_from_env()

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Validate license token or trusted network before forwarding request.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or handler.

        Returns:
            The response, or a 401 JSON error if unauthorized.
        """
        path = request.url.path.rstrip("/") or "/"

        if (
            path in PUBLIC_ENDPOINTS
            or path in self._extra_public_endpoints
            or matches_public_prefix(path)
        ):
            return await call_next(request)

        # Cost-correlation (fleet-wide): stash inbound X-Correlation-Id + X-Stage
        # so a w2v service that receives them propagates them onward (and onto WM
        # cost events) without per-service code. Orthogonal to auth; set for every
        # branch below. clear_identity_context() (in each finally) resets these
        # too. The orchestrator OVERWRITES these per DAG stage via
        # set_correlation_context(run_id, stage) — it originates the correlation.
        set_correlation_context(
            request.headers.get("x-correlation-id", ""),
            request.headers.get("x-stage", ""),
        )
        # Every non-public request is INBOUND work: the SERVICE_ACTOR_ID
        # fallback must not fill in for a caller that sent no actor.
        mark_inbound_request()
        # Loop guard for tenant provider slots (w2v_common.providers): a request a
        # tenant-provider adapter sent (X-W2V-Provider-Hop) never resolves a slot.
        mark_provider_hop(PROVIDER_HOP_HEADER.lower() in request.headers)
        # Denominator: every lane below calls ``call_next`` only once it has
        # AUTHENTICATED the request, so counting there counts each one once.
        call_next = _count_cell_auth(call_next)

        # P7 intra-cell lane, FIRST when enabled. ``off`` (the default, and every
        # out-of-cell consumer of this library) leaves the lane order exactly as
        # it was: the header is not even read.
        cell_mode = cell_auth_mode()
        cell_response, call_next = await self._cell_preamble(request, call_next, cell_mode)
        if cell_response is not None:
            return cell_response

        # Authorization carries the upstream caller's own JWT (e.g. FR forwards
        # a Keycloak user JWT). It is validated via Keycloak when JWT-shaped;
        # it is NEVER submitted to license introspection (that fallback caused
        # the 2026-04-26 prod 401s where FR's Keycloak bearer was misread as
        # a license token).
        jwt_bearer = self._extract_jwt_bearer(request)
        if jwt_bearer and _looks_like_jwt(jwt_bearer):
            validator = get_jwt_validator()
            if validator is not None:
                claims, jwt_error = await validator.validate(jwt_bearer)
                # ``is True``, not truthiness: consumer suites patch
                # ``get_jwt_validator`` with a bare MagicMock, whose auto-attribute
                # is truthy and would refuse every mocked user token.
                if claims and validator.is_service_only_issuer(claims.get("iss")) is True:
                    # Realm-per-app P0: a KEYCLOAK_SERVICE_ONLY_REALMS token
                    # (default secure-apps) is never a USER session here. That
                    # realm holds human accounts too, so being in
                    # KEYCLOAK_TRUSTED_ISSUERS (for S2S doors that verify it
                    # themselves) must not make it a w2v login.
                    logger.warning(
                        "JWT from service-only realm refused on %s %s: %s",
                        request.method,
                        path,
                        claims.get("iss"),
                    )
                    return _unauthorized(
                        request,
                        "JWT_SERVICE_REALM",
                        "Tokens from a service-only realm are not accepted as a user session",
                    )
                if claims:
                    sub = claims.get("sub", "")
                    # Three-piece auth invariant, same as the license lane
                    # below: an authenticated principal MUST carry a real
                    # tenant. This used to fall back to `or sub`, which
                    # authenticated a token with no `tenant_id` claim as a
                    # SELF-NAMED tenant equal to its own subject. The
                    # orchestrator's `/api/auth/me` WRITES on this path
                    # (`members.record_seen`), so the fallback would MINT a
                    # `tenant_members` row for a tenant that never existed.
                    # Reject fail-closed, mirroring the license lane.
                    #
                    # NOTE: nothing here gates WHICH client minted the token.
                    # `KEYCLOAK_AUDIENCES` is not set in the `w2v` namespace,
                    # and keycloak.py sets `verify_aud: bool(self._audiences)`
                    # — so with no audiences configured the audience check is
                    # skipped entirely and any portal-realm token is accepted
                    # regardless of `azp`. That control is INERT until
                    # `KEYCLOAK_AUDIENCES` is set, and setting it requires an
                    # `oidc-audience-mapper` to exist first (the declared realm
                    # has none for the w2v clients; setting the env var before
                    # the mapper lands would 401 every SPA user).
                    #
                    # Failing closed is safe because every portal client emits
                    # the claim, not because audiences gate anything: of the 35
                    # portal clients, 27 have the `tenant` client scope as a
                    # default and the other 8 (account, admin-cli, broker,
                    # realm-management, security-admin-console, account-console,
                    # oauth2-proxy-n8n, twenty-merqi) have `roles` as a default,
                    # which also carries a `tenant_id` mapper. All 29 portal
                    # users have a non-empty `tenant_id` attribute.
                    # Must be a real registered tenant UUID. A non-UUID claim
                    # (a legacy slug) is rejected too: it would be written into
                    # every downstream ``uuid`` tenant_id column as a coerced
                    # or invented value.
                    # (AUTH_JWT_TENANT_REQUIRED=false relaxes ONLY an absent
                    # claim — see _resolve_jwt_tenant.)
                    tenant_id = _resolve_jwt_tenant(request, claims, sub)
                    if isinstance(tenant_id, Response):
                        return tenant_id
                    realm_access = claims.get("realm_access") or {}
                    roles = (
                        list(realm_access.get("roles", []))
                        if isinstance(realm_access, dict)
                        else []
                    )
                    user = User(
                        sub=sub,
                        preferred_username=claims.get("preferred_username", "anonymous"),
                        tenant_id=tenant_id,
                        email=claims.get("email"),
                        # Literal boolean true only: a string "true" or a
                        # missing claim is NOT a verified address.
                        email_verified=claims.get("email_verified") is True,
                        roles=roles,
                        # The realm of the VERIFIED (pinned) issuer — which
                        # app's user directory this person came from.
                        realm=realm_of_issuer(claims.get("iss")),
                    )
                    _stash_identity(request, user)
                    request.state.tenant_id = str(tenant_id) if tenant_id is not None else None
                    request.state.license = None
                    # Keycloak JWT path — no opaque license token to forward.
                    request.state.license_token = None
                    set_identity_context(
                        tenant_id=tenant_id,
                        license_id=None,
                        subject_id=sub,
                        # Keycloak JWT path — no opaque license token to forward.
                        # Empty here means caller_token_event_hook is a no-op,
                        # so cached clients keep their service-account Bearer
                        # for downstream calls.
                        caller_license_token=None,
                        # VOUCHED: the tenant rides inside a signature this
                        # service verified against the realm JWKS, so it is not
                        # attacker-settable the way a bare header is. kag has no
                        # browser-JWT lane; this rung is w2v's addition (see
                        # TenantSource.JWT_CLAIM).
                        # Tenant-less (opt-out) identity: no tenant, so no
                        # provenance — the membership middleware sets both.
                        tenant_source=TenantSource.JWT_CLAIM if tenant_id is not None else None,
                        # WHO ACTED. This is the ONLY lane that knows a human
                        # acting principal: ``sub`` came out of a signature this
                        # service verified against the realm JWKS. It does NOT
                        # change who is billed — ``subject_id`` above is
                        # untouched — it only records, alongside it, which
                        # principal made the call (billing alembic 0021).
                        # The license-token lane names a subject, not a human,
                        # and the trusted-network lane's headers are settable by
                        # any pod in the CIDR; both therefore leave this empty
                        # rather than assert an actor they cannot back up.
                        actor_id=sub,
                    )
                    # Project from the header; X-Actor-Id is NOT honoured here
                    # (trusted=False): the verified ``sub`` IS the actor, and a
                    # human's token does not get to name somebody else.
                    bind_attribution_from_headers(request.headers, trusted=False)
                    try:
                        return await call_next(request)
                    finally:
                        clear_identity_context()
                # JWT-shaped but failed validation. Hard-fail with the reason
                # instead of silently falling through — surfaces issuer/kid/
                # signature/expiry problems in prod.
                logger.warning(
                    "JWT validation failed for %s %s: %s",
                    request.method,
                    path,
                    jwt_error,
                )
                return _unauthorized(request, "JWT_INVALID", f"JWT validation failed: {jwt_error}")
            # Validator not configured (KEYCLOAK_URL unset). Do NOT fall
            # through to license introspection — submitting a JWT to the
            # license endpoint just yields token_not_found and hides the
            # real misconfiguration. Log and reject.
            logger.warning(
                "Received JWT-shaped Authorization token but KEYCLOAK_URL is not "
                "configured — rejecting (configure KEYCLOAK_URL to enable JWT auth)."
            )
            return _unauthorized(
                request,
                "JWT_VALIDATOR_UNAVAILABLE",
                "JWT auth requested but Keycloak validator is not configured",
            )

        # Opaque licence tokens, in either accepted shape: the legacy
        # `X-Tenant-Id: Bearer <token>` (w2v's own convention, still what every
        # current caller sends) or the platform's `Authorization: Bearer
        # <opaque>` — reached only for a NON-JWT-shaped bearer, since every
        # JWT-shaped one was consumed and hard-failed above. See
        # `_extract_license_token` for the precedence rule and why it matters.
        license_token = self._extract_license_token(request)
        if license_token and license_token.lower().startswith(TENANT_API_KEY_PREFIX):
            # A w2v tenant API key is NEVER a licence token: introspecting it
            # would ship the tenant's secret to the licenses service. The prefix
            # match is CASE-INSENSITIVE: a ``W2V_SK_...`` that fell through would
            # be shipped to licences just the same — the verifier, not this
            # check, decides whether a mis-cased key is valid.
            return await self._dispatch_tenant_api_key(request, call_next, license_token)
        if license_token:
            return await self._dispatch_license_token(request, call_next, license_token, cell_mode)

        # No credential was presented to THIS service. Two lanes may still
        # establish an identity from what the network/gate asserted; both are
        # opt-in, and both refuse rather than default when they cannot.
        return await self._dispatch_asserted_identity(request, call_next)

    async def _dispatch_license_token(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
        license_token: str,
        cell_mode: CellAuthMode,
    ) -> Response:
        """The opaque-licence lane (introspection + optional delegation).

        Extracted from ``dispatch`` (complexity gate) when the P7 cell checks
        were added; behaviour is unchanged when ``cell_mode`` is ``off``.
        """
        if (
            cell_mode is CellAuthMode.ENFORCE
            and (request.headers.get(EFFECTIVE_TENANT_HEADER) or "").strip()
        ):
            # Delegation is the in-cell legacy shape; in ``enforce`` an
            # in-cell caller must present a cell token instead. Refused
            # BEFORE introspection: a header a customer has no reason to
            # send is not worth a licences round-trip.
            record_legacy_delegation("unverified")
            return _unauthorized(
                request,
                "CELL_TOKEN_REQUIRED",
                "in-cell delegation requires a cell token",
            )
        result = await _introspect_license(license_token)
        if not result or not result.get("active"):
            reason = result.get("reason", "invalid_token") if result else "invalid_token"
            return _unauthorized(
                request, "INVALID_TOKEN", f"Invalid or expired license token: {reason}"
            )
        # #222: a VALID licence for another app (a kag token on tts/asr) is
        # refused once AUTH_LICENSE_APP_KEYS declares this service's apps.
        if not check_license_app(introspected_app_key(result)):
            return _wrong_app(request)
        # Three-piece auth invariant: an active license MUST carry a
        # tenant_id. An empty/whitespace value is always a bug (a
        # misconfigured license or a buggy introspector), never a valid
        # state — accepting it would authenticate an "empty tenant" and
        # silently bypass tenant isolation downstream. Reject fail-closed,
        # mirroring the trusted-network path's existing empty-tenant guard
        # below. (We do not also require subject_id here: tenant-scoped
        # service-account tokens legitimately carry no subject, and the
        # trusted-network path enforces tenant only.)
        legacy_refusal = self._observe_legacy_cell_caller(request, result, cell_mode)
        if legacy_refusal is not None:
            return legacy_refusal
        resolved = _resolve_license_identity(request, result)
        if isinstance(resolved, Response):
            return resolved
        tenant_id, subject_id, license_id = resolved
        user = User(
            sub=str(tenant_id),
            preferred_username=subject_id,
            tenant_id=tenant_id,
        )
        _stash_identity(request, user)
        request.state.tenant_id = str(tenant_id)
        request.state.license = result
        # Preserve the raw license token so downstream code that needs to
        # forward the *caller's* identity (e.g. orchestrator → webscraper →
        # secrets-service for a tenant-scoped credential lookup) can do so
        # without falling back to the service-account DOWNSTREAM_LICENSE_TOKEN
        # — which resolves to a different tenant and would 404 the BOLA check.
        request.state.license_token = license_token
        set_identity_context(
            tenant_id=tenant_id,
            license_id=license_id,
            subject_id=(subject_id if subject_id != str(tenant_id) else result.get("subject_id")),
            # Pass the caller's RAW license token through to the
            # ContextVar so caller_token_event_hook can stamp it as
            # Bearer Authorization + X-Tenant-Id on outbound httpx
            # calls FROM CLIENTS THAT OPT IN to that hook (currently
            # ArtifactClient only). Other clients (WMClient, the
            # analysis-service client, etc.) ignore this var entirely
            # and keep their service-account DOWNSTREAM_LICENSE_TOKEN.
            caller_license_token=license_token,
            # VOUCHED: self-authenticating, introspected above. Still
            # license-token when _apply_effective_tenant re-scoped us to a
            # delegator's X-Effective-Tenant-Id — an allowlisted delegating
            # subject is exactly the party entitled to vouch for that name.
            tenant_source=TenantSource.LICENSE_TOKEN,
        )
        # X-Actor-Id is an assertion about WHO ACTED, honoured only from an
        # introspected subject on AUTH_ACTOR_ASSERTING_SUBJECTS — a list
        # SEPARATE from AUTH_DELEGATING_SUBJECTS (tenant re-scoping), so
        # granting one never grants the other. A customer token's subject is
        # never on it. Project is honoured from any authenticated caller.
        bind_attribution_from_headers(
            request.headers,
            trusted=_is_actor_asserting_subject(result.get("subject_id")),
        )
        try:
            return await call_next(request)
        finally:
            clear_identity_context()

    @staticmethod
    def _observe_legacy_cell_caller(
        request: Request, result: _Introspection, mode: CellAuthMode
    ) -> Response | None:
        """Count (and in ``enforce`` refuse) an in-cell caller on the licence lane.

        In-cell = the introspected subject is a delegating service account, or
        the request asserts ``X-Effective-Tenant-Id``. ``w2v_cell_legacy_delegation_total``
        is the per-receiver flip signal: 0 for 24 h → safe to ``enforce``. The
        label is the subject only when it is on the (small, operator-owned)
        delegating list; otherwise the fixed value ``unlisted``.
        """
        if mode is CellAuthMode.OFF:
            return None
        if getattr(request.state, "cell_claims", None) is not None:
            # ``dual`` with a VALID cell token alongside the licence: the caller
            # already signs, so this is not legacy-only traffic (counted in
            # w2v_cell_dual_both_total instead). Keeps the flip signal honest.
            return None
        subject = str(result.get("subject_id") or "")
        delegating = bool(subject) and subject in _delegating_subjects()
        asserted = bool((request.headers.get(EFFECTIVE_TENANT_HEADER) or "").strip())
        if not delegating and not asserted:
            return None
        record_legacy_delegation(subject if delegating else "unlisted")
        if mode is CellAuthMode.ENFORCE and delegating:
            logger.warning(
                "cell enforce: delegating subject %s refused on the licence lane for %s %s",
                subject,
                request.method,
                request.url.path,
            )
            return _unauthorized(
                request, "CELL_TOKEN_REQUIRED", "in-cell callers must present a cell token"
            )
        return None

    async def _dispatch_cell_token(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
        mode: CellAuthMode,
    ) -> Response | None:
        """The P7 intra-cell lane. ``None`` = not handled here, try the next lane.

        * No ``X-W2V-Cell-Token`` → ``None`` (customer/JWT/API-key lanes as before).
        * Invalid token → counted by reason; ``dual`` logs and falls through,
          ``enforce`` answers 401. The token itself is never logged.
        * Valid → tenant/actor/run/project from the SIGNED claims, provenance
          ``CELL_JWT`` (vouched), ``request.state.cell_caller = iss``.
        * Subject / licence (0.93.0) come from the SIGNED ``subject`` /
          ``license`` claims; ``X-Subject-Id`` / ``X-License-Id`` are ignored.
          A token without those claims (a 0.92.x minter) falls back to the
          headers — counted, and the header subject is ATTRIBUTION-ONLY (never
          what ``get_subject_id_from_context`` / authz sees).
        """
        claims = _verify_cell_header(request, mode)
        if claims is None:
            return None
        if isinstance(claims, CellTokenError):
            if mode is CellAuthMode.ENFORCE:
                return _unauthorized(
                    request, "CELL_TOKEN_INVALID", f"cell token rejected: {claims.reason}"
                )
            return None
        if claims.scope is CellScope.SERVICE:
            return await self._dispatch_cell_service(request, call_next, claims)
        auth_license_lane_total.labels(lane="cell-jwt").inc()
        subject, license_id, unsigned_subject = _cell_attribution(request, claims)
        user = User(
            sub=claims.actor or f"cell:{claims.issuer}",
            preferred_username=f"cell:{claims.issuer}",
            tenant_id=claims.tenant_id,
        )
        _stash_identity(request, user)
        request.state.tenant_id = str(claims.tenant_id)
        request.state.license = None
        request.state.license_token = None
        request.state.tenant_delegated = False
        request.state.cell_caller = claims.issuer
        request.state.cell_claims = claims
        request.state.cell_scope = claims.scope.value
        request.state.cell_auth_lane = CELL_AUTH_LANE_CELL
        set_identity_context(
            tenant_id=claims.tenant_id,
            license_id=license_id,
            subject_id=subject,
            caller_license_token=None,
            tenant_source=TenantSource.CELL_JWT,
            actor_id=claims.actor,
        )
        # Transitional (pre-0.93.0 caller): billing/forwarding only, never authz.
        bind_unsigned_attribution_subject(unsigned_subject)
        # fr project (X-Project-Id) from the header as on every lane; the
        # actor is the SIGNED claim, never a header (trusted=False).
        bind_attribution_from_headers(request.headers, trusted=False)
        # The w2v project and run are SIGNED claims and override any header.
        set_w2v_project_context(claims.project)
        set_w2v_job_context(claims.run)
        try:
            return await call_next(request)
        finally:
            clear_identity_context()

    @staticmethod
    async def _dispatch_cell_service(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
        claims: CellClaims,
    ) -> Response:
        """A verified ``scope=service`` token on a route that opted in for its issuer.

        The identity is the calling SERVICE and NO tenant: ``tenant_id`` None
        (never a sentinel), provenance None (not vouched — the licence gate is
        not skipped and refuses), no subject / licence / actor / project / run.
        Forwarded ``X-Subject-Id`` / ``X-Effective-Tenant-Id`` / attribution
        headers are ignored: there is no tenant for them to scope.
        """
        auth_license_lane_total.labels(lane="cell-jwt-service").inc()
        user = User(
            sub=f"cell:{claims.issuer}",
            preferred_username=f"cell:{claims.issuer}",
            tenant_id=None,
        )
        _stash_identity(request, user)
        request.state.tenant_id = None
        request.state.license = None
        request.state.license_token = None
        request.state.tenant_delegated = False
        request.state.cell_caller = claims.issuer
        request.state.cell_claims = claims
        request.state.cell_scope = claims.scope.value
        request.state.cell_auth_lane = CELL_AUTH_LANE_CELL
        # The ONLY place this is set: require_cell_service_caller admits on it,
        # never on a dual-mode RECORDED token (legacy lane authenticated).
        setattr(request.state, SERVICE_LANE_STATE_ATTR, True)
        set_identity_context(
            tenant_id=None,
            license_id=None,
            subject_id=None,
            caller_license_token=None,
            tenant_source=None,
            actor_id=None,
        )
        set_w2v_project_context(None)
        set_w2v_job_context(None)
        try:
            return await call_next(request)
        finally:
            clear_identity_context()

    async def _cell_preamble(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
        mode: CellAuthMode,
    ) -> tuple[Response | None, Callable[[Request], Awaitable[Response]]]:
        """Run the P7 cell step: a final response, or the ``call_next`` to continue with.

        ``off``: untouched. ``dual`` + a legacy credential: BEHAVIOUR-PRESERVING —
        the legacy lanes run exactly as in ``off`` (every request.state field,
        tenant resolution and error) and a valid cell token is only RECORDED
        alongside; consumers on main still read request.state.license /
        license_token / caller_license_token. Otherwise the cell lane proper.
        """
        if mode is CellAuthMode.OFF:
            return None, call_next
        if mode is CellAuthMode.DUAL and _has_legacy_credential(request):
            return None, self._observe_dual_cell_token(request, call_next)
        return await self._dispatch_cell_token(request, call_next, mode), call_next

    @staticmethod
    def _observe_dual_cell_token(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Callable[[Request], Awaitable[Response]]:
        """``dual`` + legacy credential: verify and RECORD the cell token only.

        Returns the ``call_next`` the legacy lanes should use. With no valid
        token it is the original one (invalid tokens are counted/logged exactly
        as before). With a valid token: ``request.state.cell_caller`` /
        ``cell_claims`` are set, ``w2v_cell_dual_both_total{caller}`` counts it,
        and the returned wrapper compares the tenant the legacy lane resolved
        against the signed one just before the handler runs — a mismatch keeps
        the legacy tenant, warns and counts
        ``w2v_cell_jwt_rejected_total{reason="tenant_mismatch"}``.
        """
        claims = _verify_cell_header(request, CellAuthMode.DUAL)
        if not isinstance(claims, CellClaims):
            return call_next
        request.state.cell_caller = claims.issuer
        request.state.cell_claims = claims
        request.state.cell_scope = claims.scope.value
        request.state.cell_auth_lane = CELL_AUTH_LANE_DUAL
        record_dual_both(claims.issuer, claims.scope.value)
        if claims.scope is CellScope.SERVICE:
            # A service token signs no tenant: there is nothing to compare the
            # legacy tenant against, so no mismatch check (and no false alarm).
            return call_next

        async def _checked(req: Request) -> Response:
            legacy_tenant = getattr(req.state, "tenant_id", None)
            if legacy_tenant is not None and str(legacy_tenant) != str(claims.tenant_id):
                record_cell_rejection("tenant_mismatch", claims.scope.value)
                logger.warning(
                    "cell token tenant mismatch (dual; legacy lane kept): caller=%s "
                    "cell_tenant=%s legacy_tenant=%s for %s %s",
                    claims.issuer,
                    claims.tenant_id,
                    legacy_tenant,
                    req.method,
                    req.url.path,
                )
            return await call_next(req)

        return _checked

    async def _dispatch_tenant_api_key(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
        token: str,
    ) -> Response:
        """Authenticate a ``w2v_sk_`` tenant API key via the app's registered verifier.

        No verifier registered → 401; verifier returns ``None`` → 401; verifier
        raises → 503 (a key store that could not be read is not an invalid
        key). The token is never logged, never cached here and never forwarded
        downstream (``caller_license_token`` stays empty, so outbound clients
        keep their own service-account credential).
        """
        verifier = _api_key_verifier
        if verifier is None:
            logger.warning(
                "tenant API key presented to %s %s but no verifier is registered",
                request.method,
                request.url.path,
            )
            return _unauthorized(
                request, "API_KEY_UNSUPPORTED", "tenant API keys are not accepted by this service"
            )
        try:
            verified = await verifier(token)
        except Exception as exc:  # noqa: BLE001 — arbitrary app code; any failure = could not check
            # Type only: an exception message from a key store may echo its input.
            logger.warning("tenant API key verifier failed: %s", type(exc).__name__)
            return _service_unavailable(
                request, "API_KEY_VERIFIER_UNAVAILABLE", "tenant API key could not be verified"
            )
        if verified is None:
            return _unauthorized(request, "INVALID_API_KEY", "invalid or revoked API key")
        raw_tenant, key_id = verified
        tenant_id = _tenant_or_unauthorized(
            request,
            raw_tenant,
            source="tenant API key",
            code="INVALID_API_KEY",
            message="API key has no valid tenant",
            log_context=f" (key_id={key_id})",
        )
        if isinstance(tenant_id, Response):
            return tenant_id
        user = User(
            sub=f"api-key:{key_id}",
            preferred_username=f"api-key:{key_id}",
            tenant_id=tenant_id,
            api_key_id=key_id,
        )
        _stash_identity(request, user)
        request.state.tenant_id = str(tenant_id)
        request.state.license = None
        request.state.license_token = None
        set_identity_context(
            tenant_id=tenant_id,
            license_id=None,
            subject_id=user.sub,
            # NEVER the key: caller_token_event_hook would stamp it on
            # outbound calls and leak the tenant's secret to every peer.
            caller_license_token=None,
            # VOUCHED: self-authenticating, verified against the tenant's own
            # key row by the app.
            tenant_source=TenantSource.TENANT_API_KEY,
        )
        # A key names a tenant, not a human: X-Actor-Id is not honoured.
        bind_attribution_from_headers(request.headers, trusted=False)
        try:
            return await call_next(request)
        finally:
            clear_identity_context()

    async def _dispatch_asserted_identity(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """The no-credential lanes, in descending order of strength.

        Split out of ``dispatch`` purely so each lane stays independently
        readable and ``dispatch`` stays under the complexity gate — the same
        reason ``_dispatch_trusted_network`` was extracted.
        """
        client_host = request.client.host if request.client else None

        # Platform gate lane: license-authz already introspected the licence at
        # the waypoint and injected its verdict. OFF unless the operator opted
        # in; returns ``None`` when this is not an injected request at all, so
        # the trusted-network lane below is unaffected.
        mesh_response = await self._dispatch_mesh_vouched(request, call_next, client_host)
        if mesh_response is not None:
            return mesh_response

        if _trust_incluster_identity():
            _warn_lane_without_networks("AUTH_TRUST_INCLUSTER_IDENTITY")
            if _is_trusted_network(client_host):
                return await self._dispatch_trusted_network(request, call_next, client_host)

        return _unauthorized(request, "AUTHENTICATION_REQUIRED", "Missing authentication token")

    async def _dispatch_mesh_vouched(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
        client_host: str | None,
    ) -> Response | None:
        """license-authz INJECTED-identity lane (OFF unless AUTH_TRUST_MESH_LICENSE_IDENTITY).

        When w2v sits behind the platform's ``license-authz`` Istio ext-authz
        gate, the gate has ALREADY introspected the licence. On ALLOW it injects
        ``x-tenant-id`` (a BARE tenant UUID), ``x-license-id``, ``x-subject-id``
        and friends. Nothing re-presentable as a credential survives that hop in
        the general case, so without this lane a gated w2v would 401 every
        caller — which is the whole reason this exists.

        ⚠️ **THE SPOOFING QUESTION, ANSWERED HONESTLY.** These are plain headers.
        Istio's header-strip filter on this platform is NORTH-SOUTH ONLY — the
        east-west header-spoofing gap is a known, open platform issue — so ANY
        in-mesh peer that can open a connection to this pod can set
        ``x-tenant-id`` + ``x-license-id`` + ``x-subject-id`` itself and be
        indistinguishable, on the wire, from the gate. There is no cryptographic
        marker in the injected set to check. **We therefore cannot distinguish
        genuine injection from a forgery by inspecting the request alone**, and a
        spoofable identity lane is worse than no lane: MESH_VOUCHED is a VOUCHED
        provenance, so a forged set would delegate an arbitrary victim tenant to
        a metered peer and spend their credentials.

        So the lane is not trusted silently. Three independent conditions gate it:

        1. ``AUTH_TRUST_MESH_LICENSE_IDENTITY`` must be explicitly ON. It
           **defaults OFF**, so this code is inert on every service until an
           operator states that only the gate can reach this port.
        2. The peer IP must be inside ``TRUSTED_CIDRS`` (same weak check the
           trusted-network lane uses — it vouches for the CALLER, not the
           tenant, and is a floor, never the proof).
        3. If ``AUTH_MESH_VOUCH_SECRET`` is configured, the request must also
           carry a ``X-License-Vouch`` that VERIFIES — an HMAC-SHA256, in
           license-authz's ``v1:<ts>:<nonce>:<mac>`` wire format, recomputed
           here over this request's OWN injected headers and compared with
           ``hmac.compare_digest``, plus a freshness window on the MAC-bound
           timestamp. This is the only condition of the three that is actually
           unforgeable by an in-mesh peer.

           ⚠️ It is an HMAC, NOT a bare shared secret echoed back. An earlier
           revision compared the presented header against the secret itself;
           that can never match a real vouch, so arming the lane would have
           refused every genuine gated request while both repos looked
           "implemented" — the lane dead, permanently, in the silent direction.

           ``AUTH_MESH_VOUCH_SECRET`` must hold the SAME key material as
           license-authz's ``LICENSE_VOUCH_SECRET``.

           **The posture when it is set vs unset, decided and stated:**

           * **Secret configured →** the vouch is MANDATORY. Absent, malformed,
             stale, wrong-version, or wrong-MAC are all a hard
             ``MESH_VOUCH_REQUIRED`` refusal, never a downgrade to the weaker
             (1)+(2) check. An operator who configured a key has stated that
             only the gate may establish identity here; silently accepting an
             unvouched set would make the key decorative and would let an
             in-mesh peer opt OUT of the one unforgeable condition simply by
             omitting a header.
           * **No secret configured →** unchanged: the lane rests on (1)+(2)
             and warns exactly once per process. license-authz also emits
             nothing when its own key is unset, so this is the honest
             representation of "neither side can vouch yet", and it keeps the
             already-deployed unkeyed rollout working.

           The vouch is an ADDITIONAL requirement layered on (1)+(2), never a
           replacement: a valid vouch from an off-cluster peer, or with an
           incomplete injected set, is still refused above.

        The full injected set is required (tenant AND licence AND subject).
        A bare ``x-tenant-id`` on its own is NOT this lane — it falls through to
        the trusted-network lane, where it stays ``UNVOUCHED_HEADER``, and is
        refused outright when that lane is off. A ``Bearer ``-prefixed
        ``X-Tenant-Id`` is a credential, never an injected identity, and was
        already consumed above.

        Returns ``None`` to mean "not my request, try the next lane"; a
        ``Response`` to mean the lane handled (or refused) it.
        """
        if not _trust_mesh_license_identity():
            return None

        raw_tenant = (request.headers.get(MESH_TENANT_HEADER) or "").strip()
        license_id = (request.headers.get(MESH_LICENSE_HEADER) or "").strip()
        subject_id = (request.headers.get(MESH_SUBJECT_HEADER) or "").strip()
        if not raw_tenant or not license_id or not subject_id:
            return None
        if raw_tenant.lower().startswith("bearer "):
            # A credential, not an injected identity. Already handled above; if
            # it got here it failed introspection and must not be laundered into
            # a vouched identity by the presence of sibling headers.
            return None
        _warn_lane_without_networks("AUTH_TRUST_MESH_LICENSE_IDENTITY")
        if not _is_trusted_network(client_host):
            logger.warning(
                "Ignoring license-authz identity headers from off-cluster peer %s "
                "(not in TRUSTED_CIDRS)",
                client_host or "<unknown>",
            )
            return None

        secret = _mesh_vouch_secret()
        if secret:
            presented = (request.headers.get(MESH_VOUCH_HEADER) or "").strip()
            ok, reason = _verify_mesh_vouch(
                request.headers, presented, secret, now=int(_time.time())
            )
            if not ok:
                logger.warning(
                    "license-authz identity headers presented without a valid %s "
                    "from %s (%s) — refusing",
                    MESH_VOUCH_HEADER,
                    client_host or "<unknown>",
                    reason,
                )
                return _unauthorized(
                    request,
                    "MESH_VOUCH_REQUIRED",
                    "license-authz identity headers are not vouched",
                )
        else:
            _warn_mesh_vouch_unsecured_once()

        tenant_id = _tenant_or_unauthorized(
            request,
            raw_tenant,
            source="license-authz x-tenant-id header",
            code="TENANT_REQUIRED",
            message="license-authz injected an unusable tenant",
            log_context=" (injected identity lane; there is no default tenant)",
        )
        if isinstance(tenant_id, Response):
            return tenant_id

        # #222: the gate injected the licence's app as x-app-key (MAC-bound
        # with the rest of the set when AUTH_MESH_VOUCH_SECRET is set).
        if not check_license_app((request.headers.get(MESH_APP_KEY_HEADER) or "").strip() or None):
            return _wrong_app(request)
        auth_license_lane_total.labels(lane="mesh-vouched").inc()
        user = User(
            sub=subject_id,
            preferred_username=subject_id,
            tenant_id=tenant_id,
        )
        _stash_identity(request, user)
        request.state.tenant_id = str(tenant_id)
        # The gate consumed the licence; this service never saw a token, so there
        # is nothing to forward. Downstream clients keep their service-account
        # bearer and re-resolve identity from the delegated tenant headers.
        request.state.license = None
        request.state.license_token = None
        set_identity_context(
            tenant_id=tenant_id,
            license_id=license_id,
            subject_id=subject_id,
            caller_license_token=None,
            tenant_source=TenantSource.MESH_VOUCHED,
            # WHO ACTED, when the gate named a human. Absent is left absent —
            # never defaulted to the subject (billing alembic 0021 keeps the two
            # apart: subject pays, actor acted).
            actor_id=(request.headers.get(MESH_ACTOR_HEADER) or "").strip() or None,
        )
        # Project only: the actor was already taken from the gate-injected
        # header above, so the helper must not re-bind it (trusted=False).
        bind_attribution_from_headers(request.headers, trusted=False)
        try:
            return await call_next(request)
        finally:
            clear_identity_context()

    async def _dispatch_trusted_network(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
        client_host: str | None,
    ) -> Response:
        """Trusted-network synthetic-identity lane (OFF unless AUTH_TRUST_INCLUSTER_IDENTITY).

        Extracted from ``dispatch`` so each auth lane stays independently
        readable — and so ``dispatch`` stays under the complexity gate now that
        every lane has to REJECT an unusable tenant rather than default one.
        """
        # The trusted network vouches for the CALLER, never for the
        # TENANT. This used to fall back to DEFAULT_TENANT_ID (the nil-UUID
        # sentinel) whenever the header was absent or whitespace-only, so
        # every in-cluster call that forgot to forward the requestor's
        # tenant was silently attributed to a fake one. There is no default
        # any more: the requestor's real tenant must be on the wire.
        tenant_id = _tenant_or_unauthorized(
            request,
            request.headers.get("X-Tenant-Id"),
            source="X-Tenant-Id header",
            code="TENANT_REQUIRED",
            message="trusted-network requests must forward the requestor's X-Tenant-Id",
            log_context=(
                f" (trusted-network caller {client_host or '<unknown>'};"
                " there is no default tenant)"
            ),
        )
        if isinstance(tenant_id, Response):
            return tenant_id
        user = User(
            sub=str(tenant_id),
            preferred_username="internal",
            tenant_id=tenant_id,
        )
        _stash_identity(request, user)
        request.state.tenant_id = str(tenant_id)
        request.state.license = None
        # No raw license token on trusted-network passthrough; downstream
        # callers must rely on tenant_id + service-account fallback.
        request.state.license_token = None
        set_identity_context(
            tenant_id=tenant_id,
            license_id=None,
            subject_id=str(tenant_id),
            # Trusted-network path has no caller license token — leave
            # the ContextVar empty so caller_token_event_hook is a
            # no-op and cached clients keep their service-account
            # Bearer. (Downstream cluster receivers re-resolve via
            # their own trust path.)
            caller_license_token=None,
            # NOT VOUCHED. The trusted network vouches for the CALLER, never
            # for the TENANT: any pod inside the trusted CIDRs can assert any
            # X-Tenant-Id. Accepting it as a data scope is fine; delegating it
            # to a metered peer would let an unauthenticated in-cluster caller
            # spend an arbitrary victim tenant's LLM credentials, so the
            # outbound hook refuses that hop.
            tenant_source=TenantSource.UNVOUCHED_HEADER,
        )
        # Project only. NEVER the actor: any pod inside the trusted CIDRs can
        # set any header, so an X-Actor-Id here is attribution we cannot back.
        bind_attribution_from_headers(request.headers, trusted=False)
        try:
            return await call_next(request)
        finally:
            clear_identity_context()

    def _extract_license_token(self, request: Request) -> str | None:
        """Extract an opaque licence token from either accepted credential shape.

        TWO shapes are accepted, in this order:

        1. ``X-Tenant-Id: Bearer <token>`` — w2v's own legacy convention
           (worker_management / requirements_management / procedure_management,
           procedure_management commit 16e5133). DEPRECATED but load-bearing:
           every current w2v caller sends this, and ``BaseServiceClient`` puts
           the CALLER's token here while its constructor-baked SERVICE-ACCOUNT
           token sits on ``Authorization``. That is precisely why this shape is
           checked FIRST — preferring ``Authorization`` would silently re-attribute
           every existing delegated call from the customer tenant to the service
           account's tenant.
        2. ``Authorization: Bearer <opaque>`` — the PLATFORM convention, which is
           the only shape the ``license-authz`` ext-authz gate can read. Reached
           only when the bearer is NOT JWT-shaped; ``dispatch`` has already
           consumed and hard-failed every JWT-shaped bearer above, so JWT
           validation is untouched and a JWT can still never be submitted to
           licence introspection (the 2026-04-26 prod 401s on
           /api/assets/procedures and /api/assets/secrets).

        A bare (non-``Bearer ``) ``X-Tenant-Id`` is NEVER a credential here: after
        the gate it is an injected identity assertion, handled by
        :meth:`_dispatch_mesh_vouched`.
        """
        x_tenant_id = request.headers.get("X-Tenant-Id")
        if x_tenant_id and x_tenant_id.lower().startswith("bearer "):
            token = x_tenant_id[7:].strip()
            if token:
                _record_legacy_license_header()
                return token
        bearer = self._extract_jwt_bearer(request)
        if bearer and not _looks_like_jwt(bearer):
            auth_license_lane_total.labels(lane="authorization-bearer").inc()
            return bearer
        return None

    def _extract_jwt_bearer(self, request: Request) -> str | None:
        """Extract a JWT bearer from `Authorization: Bearer <token>`.

        Returned value is candidate input for Keycloak JWT validation only.
        It is NEVER submitted to license introspection.
        """
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            return auth_header[7:].strip()
        return None
