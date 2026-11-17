"""Entitlement STATE enforcement — promoted out of ``w2v_orchestrator``.

Why this module exists (the owner's rule): *any* usage of w2v and any
subsequent service is only allowed against a live licence (tenant), measured
and billed; all non-sanctioned requests are blocked. As measured, only the
orchestrator enforced entitlement STATE. The other twelve services run
``w2v_common.auth.middleware.AuthMiddleware``, which introspects the licence
token for tenant IDENTITY — it answers "who are you", not "may you". A tenant
whose licence is ``expired`` or ``revoked`` still carries a syntactically valid
token, so those twelve served them. This module is the orchestrator's gate,
lifted verbatim in behaviour, so every service can refuse them.

The contract, unchanged from the source (see :func:`enforce_license_state`):

* entitled (``active`` / ``expiring_soon``) → pass.
* an authoritative refusal (``none`` / ``expired`` / ``revoked``) → **402**
  ``license_<state>``.
* anything else — ``unknown`` from an upstream failure, or a state this build
  does not recognise — → **503** ``license_unavailable`` with ``Retry-After``.

That last branch is an ALLOWLIST on purpose. The previous blocklist admitted
every state it had not been taught to refuse, which is how an introspection
outage became a fleet-wide ALLOW.

⚠️ **Availability is bought by the layers IN FRONT of the deny, never by
admitting unverified traffic**: the 60s positive cache and the 900s
stale-while-error window below. Drop the stale window and a licences blip
becomes a fleet-wide outage; loosen the deny and an outage becomes a
fleet-wide free tier. Both layers are carried across deliberately.

Ships DORMANT: ``W2V_LICENSE_GATE_ENABLED`` defaults to **False**, so adding
this dependency changes no behaviour until a service arms it in gitops.

``/health``, ``/healthz``, ``/ready`` (and their siblings) are exempt — that is
the platform rule, and :func:`is_license_exempt_path` makes it explicit and
testable rather than an accident of routing.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from cachetools import TTLCache
from fastapi import Depends as _Depends
from fastapi import HTTPException, Request, WebSocket, WebSocketException, params, status
from starlette.datastructures import State

from ..tenant import TenantSource
from .identity_context import (
    get_subject_id_from_context,
    get_tenant_id_from_context,
    get_tenant_source_from_context,
)
from .license_app import introspected_app_key
from .licenses_read_client import (
    LicenseNotActiveError,
    LicensesClientError,
    LicensesReader,
    get_licenses_read_client,
)
from .middleware import PUBLIC_ENDPOINTS, PUBLIC_PREFIXES, matches_public_prefix

_log = logging.getLogger(__name__)


LicenseStatusState = Literal["active", "expiring_soon", "expired", "revoked", "none", "unknown"]

#: License ``apps.key`` gated for the w2v product — the DEFAULT accepted key.
W2V_APP_KEY = "w2v"

#: Comma-separated licence app keys this service's gate accepts (default
#: ``w2v``). A request is entitled when ANY listed key is entitled. Shared
#: PLATFORM services (tts/asr, app key ``speech``) consumed by Studio AND the
#: agent platform set e.g. ``speech,w2v``. Read per call, like the gate flag.
LICENSE_GATE_APP_KEYS_ENV = "W2V_LICENSE_GATE_APP_KEYS"

#: Comma-separated subset of the accepted keys for which the licence-token FAST
#: PATH is armed (:func:`_introspected_license_verdict`). Default EMPTY = off,
#: so every key keeps the ~60s ``/licenses/active`` revocation semantics. A key
#: listed here but not in ``W2V_LICENSE_GATE_APP_KEYS`` has no effect.
LICENSE_GATE_TOKEN_APP_KEYS_ENV = "W2V_LICENSE_GATE_TOKEN_APP_KEYS"  # noqa: S105  # nosec B105 — env var name

#: Days before expiry at which an active license is flagged expiring-soon.
EXPIRING_SOON_DAYS = 30

#: The ONLY states that entitle. The enforcement chokepoint gates on this as an
#: ALLOWLIST, so ``has_license`` and the 402/503 verdict can never drift apart.
ENTITLED_STATES: frozenset[str] = frozenset({"active", "expiring_soon"})

#: License states that are an AUTHORITATIVE refusal: we reached the licenses
#: service, it answered, and the answer was "not entitled". These earn a 402.
LICENSE_BLOCKING_STATES: frozenset[str] = frozenset({"none", "expired", "revoked"})
_LICENSE_BLOCKING_STATES = LICENSE_BLOCKING_STATES  # source-parity alias

#: Error code for an INDETERMINATE verdict (``unknown``, or any state a future
#: licenses release introduces that this build does not recognise). Distinct
#: from every ``license_<state>`` 402 code on purpose.
LICENSE_UNAVAILABLE_ERROR = "license_unavailable"

#: Seconds a 503'd caller is told to wait.
LICENSE_RETRY_AFTER_SECONDS = 30

#: WebSocket close code for an AUTHORITATIVE refusal (the 402 analogue).
#: RFC 6455 1008 "Policy Violation" — the tenant's licence forbids this
#: connection. Terminal: reconnecting changes nothing until a licence is bought
#: or renewed. There is no HTTP status to return once a route is a WebSocket
#: route, so the verdict has to ride the CLOSE frame instead.
LICENSE_WS_CLOSE_POLICY = 1008

#: WebSocket close code for an INDETERMINATE verdict (the 503 analogue).
#: RFC 6455 1013 "Try Again Later" — we could NOT check the licence; this is
#: not a statement that the tenant is unlicensed, and a reconnect is the right
#: response. Kept distinct from 1008 for exactly the reason the 503 is kept
#: distinct from the 402: a licences outage must not look like a billing refusal.
LICENSE_WS_CLOSE_UNAVAILABLE = 1013

#: RFC 6455 caps a close ``reason`` at 123 BYTES. A longer reason makes the
#: close frame invalid and the peer sees a protocol error instead of the
#: verdict — so every reason emitted here is truncated to fit.
_WS_CLOSE_REASON_MAX_BYTES = 123

#: Upstream statuses meaning "this service account may not call that route" — a
#: PERMANENT misconfiguration, not a transient blip. These divert to the
#: inventory fallback (and log loudly) instead of degrading to ``unknown``.
_AUTHZ_STATUS_CODES: frozenset[int] = frozenset({401, 403})

#: A licenses ``status`` value that still denotes a live entitlement. Anything
#: else (``expired``/``revoked``/``suspended``/``pending``/…) is NOT active — an
#: unrecognised status must never entitle, so this is an allowlist.
_LIVE_RECORD_STATUSES: frozenset[str] = frozenset({"active"})

#: Positive TTL for a derived status, and the longer stale-while-error window.
#: A hit serves immediately, a miss is single-flighted, and an upstream failure
#: with a warm ``stale`` entry serves last-known-good rather than ``unknown``.
_STATUS_TTL_SECONDS = 60.0
_STALE_TTL_SECONDS = 900.0
_CACHE_MAXSIZE = 2048

#: ``(tenant_id, app_key, subject_id)`` — subject participates because the
#: inventory fallback is subject-scoped, so two subjects in one tenant can
#: legitimately derive different states.
type _StatusKey = tuple[str, str, str | None]
type _Record = dict[str, Any]

_status_cache: TTLCache[_StatusKey, LicenseStatusResult] = TTLCache(
    maxsize=_CACHE_MAXSIZE, ttl=_STATUS_TTL_SECONDS
)
_status_stale_cache: TTLCache[_StatusKey, LicenseStatusResult] = TTLCache(
    maxsize=_CACHE_MAXSIZE, ttl=_STALE_TTL_SECONDS
)
_status_locks: dict[_StatusKey, asyncio.Lock] = {}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_TRUE = {"1", "true", "yes", "on"}


def license_gate_enabled() -> bool:
    """Is the entitlement gate armed for this process?

    ``W2V_LICENSE_GATE_ENABLED``, **default False**. Read per call (not cached
    at import) so a service can flip it in a test or a restart-free rollout,
    and so the flag cannot be baked in by import order. Env-var reading matches
    ``auth.middleware``, which is this package's existing pattern.
    """
    return os.environ.get("W2V_LICENSE_GATE_ENABLED", "").strip().lower() in _TRUE


def license_gate_app_keys() -> tuple[str, ...]:
    """The licence app keys the gate accepts, in declared order, deduplicated.

    ``W2V_LICENSE_GATE_APP_KEYS`` unset or blank → ``("w2v",)``: Studio
    behaviour is unchanged unless a service opts in.
    """
    raw = os.environ.get(LICENSE_GATE_APP_KEYS_ENV) or ""
    keys = tuple(dict.fromkeys(k.strip() for k in raw.split(",") if k.strip()))
    return keys or (W2V_APP_KEY,)


def license_gate_token_app_keys(accepted: Sequence[str]) -> frozenset[str]:
    """Accepted keys whose licence-token fast path is armed (default: none).

    ``W2V_LICENSE_GATE_TOKEN_APP_KEYS`` intersected with ``accepted`` — a key
    the gate does not accept can never be entitled through the fast path.
    """
    raw = os.environ.get(LICENSE_GATE_TOKEN_APP_KEYS_ENV) or ""
    return frozenset(k.strip() for k in raw.split(",") if k.strip()) & frozenset(accepted)


def licenses_service_url() -> str | None:
    """Explicit licences-service URL override, or ``None`` for peer discovery.

    ``W2V_LICENSES_URL`` is the SAME knob :func:`w2v_common.discovery.service_url`
    already honours for every peer — deliberately not a new env name.
    """
    return os.environ.get("W2V_LICENSES_URL") or None


# ---------------------------------------------------------------------------
# Health exemption
# ---------------------------------------------------------------------------

#: Probe paths the gate must NEVER refuse. A 402/503 on a liveness probe
#: crashloops the pod, and an unlicensed tenant cannot be billed by killing our
#: own deployment. Built from ``AuthMiddleware``'s public set so the two
#: exemption lists cannot drift, plus the platform-mandated trio spelled out.
LICENSE_EXEMPT_PATHS: frozenset[str] = frozenset(PUBLIC_ENDPOINTS) | frozenset(
    {"/health", "/healthz", "/ready"}
)
LICENSE_EXEMPT_PREFIXES: tuple[str, ...] = PUBLIC_PREFIXES

#: The exempt prefixes as segment BASES, matched by
#: :func:`~w2v_common.auth.middleware.matches_public_prefix`.
LICENSE_EXEMPT_PREFIX_BASES: tuple[str, ...] = tuple(
    (p.rstrip("/") or "/") for p in LICENSE_EXEMPT_PREFIXES
)


def is_license_exempt_path(path: str) -> bool:
    """Is ``path`` exempt from the entitlement gate? Explicit and testable.

    Prefixes match at a **segment boundary** only. The previous raw
    ``startswith`` made ``/docsomething``, ``/docs-internal/render`` and
    ``/redocument`` exempt — unauthenticated AND unbilled — for any route that
    ever took one of those shapes. Which paths are exempt is unchanged; only
    how the prefix is matched.
    """
    normalised = path.rstrip("/") or "/"
    if path in LICENSE_EXEMPT_PATHS or normalised in LICENSE_EXEMPT_PATHS:
        return True
    return matches_public_prefix(path, LICENSE_EXEMPT_PREFIX_BASES)


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def clear_license_status_cache() -> None:
    """Drop every cached status. Test seam + a safety valve for a live flag flip."""
    _status_cache.clear()
    _status_stale_cache.clear()
    _status_locks.clear()


def invalidate_license_status(
    *, tenant_id: str | None, app_key: str, subject_id: str | None
) -> None:
    """Evict the cached status for one identity after a KNOWN state change.

    A mint (or any deliberate mutation) must drop both the fresh and the stale
    entry, or the immediate re-resolve reads back the pre-mint ``none``.
    """
    key: _StatusKey = (str(tenant_id), app_key, str(subject_id) if subject_id else None)
    _status_cache.pop(key, None)
    _status_stale_cache.pop(key, None)


def _status_code_of(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


@dataclass(frozen=True)
class LicenseStatusResult:
    state: LicenseStatusState
    has_license: bool
    expires_at: str | None
    days_remaining: int | None
    plan_code: str | None


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_iso(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError, TypeError:
        return None
    # A naive timestamp is treated as UTC so the arithmetic below stays tz-aware.
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _days_until(iso: str, now: datetime) -> int | None:
    """Whole days until ``iso`` (negative = days since expiry). None if unparseable."""
    dt = _parse_iso(iso)
    if dt is None:
        return None
    return math.ceil((dt - now).total_seconds() / 86400)


def _result(
    state: LicenseStatusState,
    *,
    expires_at: str | None = None,
    days_remaining: int | None = None,
    plan_code: str | None = None,
) -> LicenseStatusResult:
    return LicenseStatusResult(
        state=state,
        has_license=state in ENTITLED_STATES,
        expires_at=expires_at,
        days_remaining=days_remaining,
        plan_code=plan_code,
    )


def _plan_code(record: _Record) -> str | None:
    plan = record.get("plan")
    return plan.get("code") if isinstance(plan, dict) else None


def derive_active_state(
    detail: _Record,
    now: datetime,
    expiring_soon_days: int = EXPIRING_SOON_DAYS,
) -> LicenseStatusResult:
    """Pure: derive status from an ACTIVE license detail (200 from /licenses/active).

    A 200 whose expiry is already past (stale 200 / clock skew) is treated as
    ``expired`` — not a dismissible ``expiring_soon`` banner.
    """
    expires_at = detail.get("expires_at")
    days_remaining = _days_until(expires_at, now) if expires_at else None
    if days_remaining is None:
        state: LicenseStatusState = "active"
    elif days_remaining < 0:
        state = "expired"
    elif days_remaining <= expiring_soon_days:
        state = "expiring_soon"
    else:
        state = "active"
    return _result(
        state,
        expires_at=expires_at,
        days_remaining=days_remaining,
        plan_code=_plan_code(detail),
    )


def derive_inactive_state(
    records: list[_Record],
    app_key: str,
    tenant_id: str,
    now: datetime,
) -> LicenseStatusResult:
    """Pure: derive status with NO active license (404), from the subject's inventory.

    Precedence: revoked > expired > none (pending / future-dated count as none —
    there is no active entitlement yet).

    ``tenant_id`` is MANDATORY and scoping is done by :func:`_in_gate_scope`, the
    same predicate the 401/403 path uses. ``GET /licenses/user/{sub}`` returns the
    subject's licenses across EVERY tenant, so an app-key-only filter let a row
    belonging to tenant A decide the verdict for tenant B — a subject who had a
    revoked w2v licence in some other tenant was reported ``license_revoked`` in a
    tenant where they simply never had one (``license_none``). Same 402 either
    way, so this never wrongly ALLOWED, but the reason code (and the support
    path, and the "renew your licence" UI it drives) was wrong.
    """
    app_records = [r for r in records if _in_gate_scope(r, app_key, tenant_id)]
    if not app_records:
        return _result("none")

    revoked = next(
        (r for r in app_records if r.get("status") == "revoked" or r.get("revoked_at")),
        None,
    )
    if revoked is not None:
        exp = revoked.get("expires_at")
        return _result(
            "revoked",
            expires_at=exp,
            days_remaining=_days_until(exp, now) if exp else None,
            plan_code=_plan_code(revoked),
        )

    def _is_expired(r: _Record) -> bool:
        if r.get("status") == "expired":
            return True
        exp = r.get("expires_at")
        if not exp:
            return False
        dt = _parse_iso(exp)
        return dt is not None and dt <= now

    expired = sorted(
        (r for r in app_records if _is_expired(r)),
        key=lambda r: _parse_iso(r.get("expires_at") or "") or datetime.min.replace(tzinfo=UTC),
        reverse=True,  # most-recently-expired first
    )
    if expired:
        exp = expired[0].get("expires_at")
        return _result(
            "expired",
            expires_at=exp,
            days_remaining=_days_until(exp, now) if exp else None,
            plan_code=_plan_code(expired[0]),
        )

    # A record exists but is neither active (we got a 404) nor expired/revoked —
    # e.g. pending / future-dated. No active entitlement → none.
    return _result("none")


def _in_gate_scope(record: _Record, app_key: str, tenant_id: str) -> bool:
    """Does ``record`` entitle THIS (tenant, app)?

    ``GET /licenses/user/{sub}`` returns the subject's licenses across **every
    tenant**, so an app-key-only filter would let a license the subject holds in
    tenant A entitle their request in tenant B. Both dimensions are mandatory. A
    record with no ``tenant_id`` cannot be proven in-scope and never entitles.
    """
    if (record.get("app") or {}).get("key") != app_key:
        return False
    record_tenant = record.get("tenant_id")
    return bool(record_tenant) and str(record_tenant) == str(tenant_id)


def _is_live_record(record: _Record, now: datetime) -> bool:
    """Is this inventory record a currently-live entitlement?

    Conservative on every axis: revoked loses, an unrecognised ``status`` loses,
    a not-yet-started license loses, and an expiry at/before ``now`` loses. A
    record with no ``expires_at`` is perpetual and passes.
    """
    if record.get("revoked_at") or record.get("status") == "revoked":
        return False
    record_status = record.get("status")
    if record_status is not None and record_status not in _LIVE_RECORD_STATUSES:
        return False
    starts_at = record.get("starts_at")
    if starts_at:
        starts = _parse_iso(starts_at)
        if starts is not None and starts > now:
            return False  # future-dated: no entitlement YET
    expires_at = record.get("expires_at")
    if expires_at:
        expires = _parse_iso(expires_at)
        # Unparseable expiry → cannot prove it is in the future → not live.
        if expires is None or expires <= now:
            return False
    return True


def _expiry_rank(record: _Record) -> tuple[int, datetime]:
    """Sort key picking the longest-lived record; perpetual (no expiry) wins."""
    expires_at = record.get("expires_at")
    if not expires_at:
        return (1, datetime.max.replace(tzinfo=UTC))
    return (0, _parse_iso(expires_at) or datetime.min.replace(tzinfo=UTC))


def derive_inventory_state(
    records: list[_Record],
    app_key: str,
    tenant_id: str,
    now: datetime,
    expiring_soon_days: int = EXPIRING_SOON_DAYS,
) -> LicenseStatusResult:
    """Pure: derive status for ``(tenant_id, app_key)`` from the subject inventory.

    Unlike :func:`derive_inactive_state` (which presumes a 404 from
    ``/licenses/active`` already PROVED there is no active license, and so has
    no active exit), this is used when the primary route was never consulted —
    it answered 401/403. It must therefore be able to return ``active`` /
    ``expiring_soon``, otherwise a fully-paid tenant derives ``none`` and is
    402'd. It routes the winning record through :func:`derive_active_state` so
    the expiring-soon window and the negative-days hardening are applied by the
    SAME code that governs the 200 path.
    """
    scoped = [r for r in records if _in_gate_scope(r, app_key, tenant_id)]
    live = [r for r in scoped if _is_live_record(r, now)]
    if live:
        return derive_active_state(max(live, key=_expiry_rank), now, expiring_soon_days)
    return derive_inactive_state(scoped, app_key, tenant_id, now)


async def _fetch_inventory(
    client: LicensesReader, subject_id: str | None, tenant_id: str
) -> list[_Record]:
    """Subject inventory as plain wire dicts. Raises on upstream failure.

    A missing subject is NOT an empty inventory — it means we cannot look the
    entitlement up at all. Raising here routes it to the ``unknown`` fail-soft
    instead of an empty list, which would derive ``none`` and 402 the caller.
    """
    if not subject_id:
        raise LicensesClientError("cannot derive entitlement without a subject_id")
    records = await client.list_user_licenses(subject_id, tenant_id=tenant_id)
    return [r.model_dump() for r in records]


async def _resolve_uncached(
    *,
    client: LicensesReader,
    app_key: str,
    tenant_id: str,
    subject_id: str | None,
    when: datetime,
) -> LicenseStatusResult:
    """One full upstream resolution. Raises on failure (caller fails soft)."""
    try:
        detail = await client.get_active_license(tenant_id=tenant_id, app_key=app_key)
    except LicenseNotActiveError:
        # licenses' OWN "no active licence" (404 + detail.error=no_active_license)
        # — the verdict is settled: NOT entitled. The inventory read below only
        # sharpens the REASON (revoked / expired / none), so it must never be able
        # to turn a settled refusal into 'unknown' (a 503 that says "platform
        # broken, retry"). A delegated caller's SA is routinely refused
        # /licenses/user/{other subject} with 403 — that is expected here, not an
        # outage — so any failure of the refinement degrades to 'none'.
        if not subject_id:
            return _result("none")
        try:
            records = await client.list_user_licenses(subject_id, tenant_id=tenant_id)
        except LicensesClientError as refine_exc:
            _log.info(
                "license-status: no active license for tenant=%s app_key=%s; reason "
                "refinement from the subject inventory unavailable (%s), reporting "
                "state='none'",
                tenant_id,
                app_key,
                _status_code_of(refine_exc),
            )
            return _result("none")
        return derive_inactive_state([r.model_dump() for r in records], app_key, tenant_id, when)
    except LicensesClientError as exc:
        status_code = _status_code_of(exc)
        if status_code not in _AUTHZ_STATUS_CODES:
            raise
        # PERMANENT misconfiguration, not a blip: the SA is not permitted on
        # /licenses/active. ERROR (not WARNING) with a distinct, greppable
        # marker — as a WARNING this was indistinguishable from a transient
        # timeout and the gate sat inert for days without anyone noticing.
        _log.error(
            "license_gate_misconfigured: /licenses/active rejected this service "
            "account with %s for tenant=%s app_key=%s. The license gate CANNOT use "
            "its primary route and is deriving entitlement from the subject "
            "inventory instead. Fix: grant the SA an internal-service role on the "
            "licenses service (service-internal / license-admin).",
            status_code,
            tenant_id,
            app_key,
        )
        records = await _fetch_inventory(client, subject_id, tenant_id)
        return derive_inventory_state(records, app_key, tenant_id, when)

    if detail is not None and detail.conflict:
        # Multiple active licenses (data conflict) — entitled, expiry unknown.
        return _result("active")
    if detail is not None:
        return derive_active_state(detail.model_dump(), when)
    # A 404 WITHOUT licenses' discriminator (older licenses build, or a fake that
    # returns None). Behaviour deliberately unchanged: it may equally be a route
    # miss, so the refinement stays strict and a failure still fails soft to
    # 'unknown' rather than asserting 'none'. Refine to
    # expired / revoked / none from the subject's inventory when we know who
    # they are. derive_inactive_state (no active exit) is correct HERE and only
    # here, precisely because the 404 already ruled active out.
    records = await client.list_user_licenses(subject_id, tenant_id=tenant_id) if subject_id else []
    return derive_inactive_state([r.model_dump() for r in records], app_key, tenant_id, when)


async def _resolve_and_cache(
    *,
    client: LicensesReader,
    app_key: str,
    tenant_id: str,
    subject_id: str | None,
    when: datetime,
    cache_key: _StatusKey | None,
) -> LicenseStatusResult:
    """Resolve once, fail soft into last-known-good, and populate the caches.

    ``cache_key=None`` means "do not read or write the cache" — used by the
    pinned-``now`` seam so a time-travelling caller can neither read an entry
    derived at a different instant nor poison one for real traffic.
    """
    try:
        result = await _resolve_uncached(
            client=client,
            app_key=app_key,
            tenant_id=tenant_id,
            subject_id=subject_id,
            when=when,
        )
    except Exception:  # noqa: BLE001 — fail-soft to 'unknown', which DENIES with 503
        stale = _status_stale_cache.get(cache_key) if cache_key is not None else None
        if stale is not None:
            # Stale-while-error: this tenant resolved cleanly moments ago, so a
            # transient failure must not downgrade them to 'unknown'. THIS is
            # what buys availability — not admitting unverified traffic.
            _log.warning(
                "license-status lookup failed for tenant=%s app=%s; serving stale state=%s",
                tenant_id,
                app_key,
                stale.state,
                exc_info=True,
            )
            return stale
        _log.warning(
            "license-status lookup failed for tenant=%s app=%s; reporting state='unknown'",
            tenant_id,
            app_key,
            exc_info=True,
        )
        return _result("unknown")

    # Only DEFINITIVE results are cached. 'unknown' is an error state: caching
    # it would keep serving a degraded answer after the upstream recovers.
    if cache_key is not None and result.state != "unknown":
        _status_cache[cache_key] = result
        _status_stale_cache[cache_key] = result
    return result


async def get_license_status(
    *,
    client: LicensesReader,
    app_key: str,
    tenant_id: str | None,
    subject_id: str | None,
    gate_enabled: bool,
    now: datetime | None = None,
) -> LicenseStatusResult:
    """Resolve the license status for a request's tenant + subject.

    - gate disabled ⇒ everyone entitled (``state='active'``) — zero functional
      change until the gate is flipped on.
    - no tenant ⇒ ``none`` (the request's own auth gate handles unauthenticated
      callers; this is only the entitlement verdict).
    - 401/403 from ``/licenses/active`` ⇒ derive from the subject inventory
      (which CAN yield active) and log ``license_gate_misconfigured``.
    - any other upstream error ⇒ last-known-good if warm, else ``unknown``
      (which the chokepoint turns into a 503, never an ALLOW).

    Results are cached for 60s per ``(tenant, app_key, subject)`` and misses are
    single-flighted — this runs synchronously on EVERY gated request, so an
    uncached implementation puts a licenses round-trip in front of the whole API.

    Passing an explicit ``now`` bypasses the cache entirely: it is the
    pure-derivation test seam, and a caller pinning time must not read (or
    poison) an entry derived at a different instant.
    """
    if not gate_enabled:
        return _result("active")
    if not tenant_id:
        return _result("none")

    cache_key: _StatusKey = (str(tenant_id), app_key, str(subject_id) if subject_id else None)

    if now is not None:
        return await _resolve_and_cache(
            client=client,
            app_key=app_key,
            tenant_id=tenant_id,
            subject_id=subject_id,
            when=now,
            cache_key=None,
        )

    hit = _status_cache.get(cache_key)
    if hit is not None:
        return hit

    # Single-flight: a concurrent burst for one cold tenant collapses into ONE
    # upstream resolution instead of stampeding the licenses service. Re-check
    # inside the lock — a prior waiter may have populated the cache.
    if len(_status_locks) > _CACHE_MAXSIZE:
        # Unbounded growth guard. Dropping a held lock only risks a duplicate
        # upstream call, never a wrong answer.
        _status_locks.clear()
    lock = _status_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        hit = _status_cache.get(cache_key)
        if hit is not None:
            return hit
        return await _resolve_and_cache(
            client=client,
            app_key=app_key,
            tenant_id=tenant_id,
            subject_id=subject_id,
            when=_now(),
            cache_key=cache_key,
        )


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------


def enforce_license_state(result: LicenseStatusResult) -> None:
    """Deny unless the derived state is POSITIVELY entitled. Single chokepoint
    shared by every caller, so two paths can never disagree about what counts
    as entitled.

    Fail-CLOSED, and with the two refusals kept distinguishable:

    * entitled (``active`` / ``expiring_soon``) → pass.
    * an authoritative refusal (``none`` / ``expired`` / ``revoked``) → **402**
      ``license_<state>``. 402 Payment Required is a statement about the
      TENANT's licence: buy or renew one and the request succeeds. It is
      terminal for the caller — retrying changes nothing.
    * anything else — ``unknown`` from an upstream failure, or a state this
      build does not recognise — → **503** ``license_unavailable`` with
      ``Retry-After``. We are not saying the tenant is unlicensed (we do not
      know); we are saying WE could not check. 503 is the honest code: it is
      server-side, transient, and retryable, and it keeps a monitored
      licences-outage signal from hiding inside the 402 counter. Emitting 402
      here would tell an entitled customer they have no licence.

    This is an ALLOWLIST on purpose. The previous blocklist admitted every state
    it had not been taught to refuse, which is how an introspection outage
    became a fleet-wide ALLOW. Availability is bought by the layers in FRONT of
    this deny (the 60s positive cache and the 900s stale-while-error window in
    :func:`get_license_status`), never by admitting traffic whose entitlement
    was never established.
    """
    if result.state in ENTITLED_STATES:
        return
    if result.state in LICENSE_BLOCKING_STATES:
        raise HTTPException(
            status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": f"license_{result.state}",
                "message": "This tenant has no active Studio license.",
            },
        )
    raise HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": LICENSE_UNAVAILABLE_ERROR,
            "message": (
                "Could not verify this tenant's Studio license. This is not a "
                "statement about your entitlement — retry shortly."
            ),
            "state": result.state,
        },
        headers={"Retry-After": str(LICENSE_RETRY_AFTER_SECONDS)},
    )


#: Source-parity alias for the orchestrator's private name.
_enforce_license_state = enforce_license_state


def _authenticated_on_cell_lane(state: State) -> bool:
    """Did the P7 cell LANE authenticate this request (so the tenant is signed)?

    ``cell_caller`` alone is not enough: in ``dual`` a valid cell token riding
    alongside a legacy credential is only RECORDED (``cell_caller`` set) while
    the licence / JWT / API-key lane authenticates — and that lane's tenant,
    possibly different from the signed one, must be gated exactly as in
    ``off``. Only the cell lane sets the tenant provenance to ``CELL_JWT``.
    """
    return bool(getattr(state, "cell_caller", None)) and (
        get_tenant_source_from_context() is TenantSource.CELL_JWT
    )


def _introspected_license_verdict(
    state: State, tenant_id: str, token_keys: frozenset[str]
) -> LicenseStatusResult | None:
    """Fast path: entitle from the licence token ``AuthMiddleware`` introspected.

    OPT-IN per app key (``W2V_LICENSE_GATE_TOKEN_APP_KEYS``, default empty =
    off). ⚠️ The REVOCATION WINDOW is the price: the lookup path sees a revoked
    licence within the 60s status cache, but this path trusts the middleware's
    introspection, so a revoked licence stays entitled for up to
    ``CACHE_LICENSE_TTL`` (300s) and, during a licences outage, up to
    ``CACHE_LICENSE_STALE_TTL`` (900s) of last-known-good — i.e.
    max(introspection cache, LKG). Arm it only for a key whose consumers cannot
    use the lookup (``speech``: agent-api's tenant is invisible to the SA).

    A caller that authenticated with its OWN licence token has already proved
    its entitlement to the licences service: ``/license/introspect`` answered
    ``active`` for a token issued for exactly one app. Re-asking
    ``/licenses/active`` for that tenant needs cross-tenant visibility the
    service's Keycloak client may not have (agent-api's platform-owned licence
    is invisible to the w2v client → 403 → ``unknown`` → 503).

    Reads ONLY ``state.license`` — the introspection payload the licence lane
    of ``AuthMiddleware`` stores (every other lane sets it to ``None``); no
    client header is consulted. Returns a verdict only when it can ENTITLE:

    * ``active`` is true and the record is live (status ``active`` or absent,
      started, not expired — :func:`_is_live_record`);
    * its app key (``app.key`` / ``app_key``) is one of ``token_keys`` (armed
      AND accepted);
    * its tenant IS the served tenant and the tenant was not re-scoped by
      delegation (``tenant_delegated``) — a service account's licence never
      entitles the tenant it acts for.

    Anything else returns ``None`` and the normal ``/licenses/active`` lookup
    decides, so this path can only ever ADD an allow for a token that is itself
    the proof; it never refuses.
    """
    lic = getattr(state, "license", None)
    if not isinstance(lic, dict) or lic.get("active") is not True:
        return None
    if getattr(state, "tenant_delegated", False):
        return None
    lic_tenant = lic.get("tenant_id")
    if not lic_tenant or str(lic_tenant) != tenant_id:
        return None
    if introspected_app_key(lic) not in token_keys:
        return None
    now = _now()
    if not _is_live_record(lic, now):
        return None
    result = derive_active_state(lic, now)
    return result if result.state in ENTITLED_STATES else None


def _combine_verdicts(results: Sequence[LicenseStatusResult]) -> LicenseStatusResult:
    """Fold per-app-key verdicts into one, fail-closed.

    Any entitled key entitles. Otherwise an INDETERMINATE key (``unknown`` or a
    state this build does not recognise) wins over an authoritative refusal: we
    could not check that key, so claiming the tenant is unlicensed (402) would
    be a lie — the honest answer is 503. Only when every key refused
    authoritatively is the first refusal returned.
    """
    for r in results:
        if r.state in ENTITLED_STATES:
            return r
    for r in results:
        if r.state not in LICENSE_BLOCKING_STATES:
            return r
    return results[0]


async def _gate_verdict(state: State) -> LicenseStatusResult | None:
    """The verdict shared by the HTTP and WebSocket gates; ``None`` = skip.

    One function so the two transports can never decide differently.
    """
    if _authenticated_on_cell_lane(state):
        # P7 cell lane: an in-cell peer that verified (and gated) the tenant
        # upstream signed it for this hop; re-gating it here is a second
        # licences round-trip per in-cell call with nothing new to decide.
        return None
    tenant_id = getattr(state, "tenant_id", None) or get_tenant_id_from_context()
    # Subject is optional — it only refines a 404 into expired/revoked/none from
    # the inventory; resolve it tolerantly (never refuse from the gate itself).
    subject = getattr(state, "subject_id", None) or get_subject_id_from_context()
    accepted = license_gate_app_keys()
    token_keys = license_gate_token_app_keys(accepted)
    if tenant_id and token_keys:
        fast = _introspected_license_verdict(state, str(tenant_id), token_keys)
        if fast is not None:
            return fast
    results = [
        await get_license_status(
            client=get_licenses_read_client(),
            app_key=app_key,
            tenant_id=str(tenant_id) if tenant_id else None,
            subject_id=str(subject) if subject else None,
            gate_enabled=True,
        )
        for app_key in accepted
    ]
    return _combine_verdicts(results)


async def require_license(request: Request) -> None:
    """FastAPI dependency: refuse a request whose tenant is not entitled.

    A no-op unless ``W2V_LICENSE_GATE_ENABLED`` is true, and always a no-op on
    an exempt probe path. Entitled when ANY key in ``W2V_LICENSE_GATE_APP_KEYS``
    (default ``w2v``) is entitled, or — only for keys armed in
    ``W2V_LICENSE_GATE_TOKEN_APP_KEYS`` — when the request's own introspected
    licence token is an active licence for one of them
    (:func:`_introspected_license_verdict`).
    """
    if not license_gate_enabled():
        return
    if is_license_exempt_path(request.url.path):
        return
    result = await _gate_verdict(request.state)
    if result is not None:
        enforce_license_state(result)


#: Mount router-wide (``APIRouter(dependencies=LICENSE_GATE)``) or per-route.
#: Defined ONCE so a router-level and a per-route mount cannot drift into
#: enforcing different things.
LICENSE_GATE: Sequence[params.Depends] = (_Depends(require_license),)


# ---------------------------------------------------------------------------
# WebSocket enforcement
#
# WHY A SECOND ENTRY POINT EXISTS. :func:`require_license` is typed for a
# Starlette ``Request`` and reads ``request.url.path`` and ``request.state``. On
# a WebSocket route FastAPI passes a ``WebSocket``, which shares neither the
# ``Request`` contract nor its response model — and nothing in this module,
# ``middleware.py`` or ``asgi/factory.py`` ever branched on ``scope["type"]``.
# The consequence, measured: every WebSocket route in the fleet (the live one is
# ``tts_service`` ``/api/tts/ws``) was COMPLETELY UNGATED — an expired or
# revoked tenant was refused on the HTTP API and served on the socket.
#
# The verdict cannot ride an HTTP status here: by the time a dependency runs on
# a WS route the only thing left to say to the peer is a CLOSE frame. So the
# same derived state maps onto close codes instead (1008 / 1013), and the
# reason carries the SAME ``license_<state>`` / ``license_unavailable`` code the
# HTTP path puts in its JSON body, so one client can key off one vocabulary.
#
# ⚠️ ``AuthMiddleware`` is a ``BaseHTTPMiddleware``, which Starlette runs on
# ``http`` scope ONLY. It therefore does NOT stamp identity on a WebSocket
# handshake: a service that gates a WS route must set ``websocket.state.tenant_id``
# (and, ideally, ``subject_id``) itself — from its own handshake auth — or set
# the identity contextvars before the dependency runs. With neither, the tenant
# is unknown, which derives ``none`` and CLOSES. That is deliberate: resolving a
# tenant from the raw ``x-tenant-id`` mesh header here would invent a
# header-spoofable identity lane that the HTTP gate does not have (see
# ``middleware._dispatch_mesh_vouched`` on the east-west spoofing gap), and a
# gate that trusts a forgeable tenant is worse than one that closes.
# ---------------------------------------------------------------------------


def _ws_close_reason(code: str, message: str) -> str:
    """Build a close reason that fits RFC 6455's 123-byte cap, code first."""
    reason = f"{code}: {message}"
    encoded = reason.encode("utf-8")
    if len(encoded) <= _WS_CLOSE_REASON_MAX_BYTES:
        return reason
    # Truncate on a byte boundary that is still valid UTF-8; the CODE is the
    # part a client keys off, so it is never the part that gets cut.
    return encoded[:_WS_CLOSE_REASON_MAX_BYTES].decode("utf-8", errors="ignore")


def enforce_license_state_ws(result: LicenseStatusResult) -> None:
    """WebSocket twin of :func:`enforce_license_state`.

    Same ALLOWLIST, same three branches, same error vocabulary — only the
    transport differs. Raising ``WebSocketException`` is what closes the socket:
    Starlette's exception middleware turns it into ``websocket.close(code,
    reason)``, so the peer gets the verdict instead of a bare disconnect.

    It shares :data:`ENTITLED_STATES` / :data:`LICENSE_BLOCKING_STATES` with the
    HTTP chokepoint on purpose — two hand-written state lists is exactly how a
    socket would start admitting a state the API refuses.
    """
    if result.state in ENTITLED_STATES:
        return
    if result.state in LICENSE_BLOCKING_STATES:
        raise WebSocketException(
            code=LICENSE_WS_CLOSE_POLICY,
            reason=_ws_close_reason(
                f"license_{result.state}", "This tenant has no active Studio license."
            ),
        )
    raise WebSocketException(
        code=LICENSE_WS_CLOSE_UNAVAILABLE,
        reason=_ws_close_reason(
            LICENSE_UNAVAILABLE_ERROR,
            f"Could not verify this tenant's license (state={result.state}).",
        ),
    )


async def require_license_ws(websocket: WebSocket) -> None:
    """FastAPI dependency: close a WebSocket whose tenant is not entitled.

    The WS analogue of :func:`require_license`, and deliberately identical to it
    in every decision it makes:

    - a no-op unless ``W2V_LICENSE_GATE_ENABLED`` is true (ships DORMANT);
    - a no-op on an exempt probe path;
    - tenant/subject resolved from ``websocket.state`` first, then the identity
      contextvars — the exact precedence :func:`require_license` uses on
      ``request.state`` (``WebSocket`` and ``Request`` are both
      ``HTTPConnection``, so ``.state``, ``.headers`` and ``.url`` all exist);
    - the verdict comes from the SAME :func:`get_license_status`, so the socket
      shares the HTTP path's cache, single-flight and stale-while-error window.

    The only divergence is the refusal: :func:`enforce_license_state_ws` closes
    with 1008/1013 instead of returning 402/503.

    Mount it as ``LICENSE_GATE_WS`` (or ``Depends(require_license_ws)``) on the
    WebSocket route. It runs BEFORE ``websocket.accept()``, so an unlicensed
    peer never reaches the handler.
    """
    if not license_gate_enabled():
        return
    if is_license_exempt_path(websocket.url.path):
        return
    result = await _gate_verdict(websocket.state)
    if result is not None:
        enforce_license_state_ws(result)


#: WebSocket twin of :data:`LICENSE_GATE`. Mount router-wide
#: (``APIRouter(dependencies=LICENSE_GATE_WS)``) or per-route on a
#: ``@router.websocket(...)`` handler. Defined ONCE for the same reason.
LICENSE_GATE_WS: Sequence[params.Depends] = (_Depends(require_license_ws),)

__all__ = [
    "ENTITLED_STATES",
    "EXPIRING_SOON_DAYS",
    "LICENSE_BLOCKING_STATES",
    "LICENSE_EXEMPT_PATHS",
    "LICENSE_EXEMPT_PREFIXES",
    "LICENSE_GATE",
    "LICENSE_GATE_APP_KEYS_ENV",
    "LICENSE_GATE_TOKEN_APP_KEYS_ENV",
    "LICENSE_GATE_WS",
    "LICENSE_RETRY_AFTER_SECONDS",
    "LICENSE_UNAVAILABLE_ERROR",
    "LICENSE_WS_CLOSE_POLICY",
    "LICENSE_WS_CLOSE_UNAVAILABLE",
    "W2V_APP_KEY",
    "LicenseStatusResult",
    "LicenseStatusState",
    "LicensesReader",
    "clear_license_status_cache",
    "derive_active_state",
    "derive_inactive_state",
    "derive_inventory_state",
    "enforce_license_state",
    "enforce_license_state_ws",
    "get_license_status",
    "invalidate_license_status",
    "is_license_exempt_path",
    "license_gate_app_keys",
    "license_gate_enabled",
    "license_gate_token_app_keys",
    "licenses_service_url",
    "require_license",
    "require_license_ws",
]
