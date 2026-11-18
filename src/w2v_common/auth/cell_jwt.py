"""Intra-cell service tokens (P7): per-service Ed25519 keys, one JWKS file, short JWTs.

Every in-cell hop (a w2v service calling another w2v service) carries a fresh
``X-W2V-Cell-Token`` minted for THAT request. The token is an EdDSA (Ed25519)
JWT signed with the CALLING service's own private key; the callee verifies it
against a public keyring file mounted from one ConfigMap.

Claims::

    iss      issuing service (its Kubernetes Service name) — BOUND to the kid
    aud      the callee's Kubernetes Service name; must equal CELL_SERVICE_NAME
    cell     deployment id (CELL_DEPLOYMENT_ID); blocks cross-deployment replay
    scope    ``tenant`` (default; absent on pre-0.92.0 tokens) or ``service``
    tenant   scope=tenant: the run's tenant (canonical non-nil uuid, REQUIRED)
             scope=service: absent or null — any value (incl. the nil uuid) is refused
    run      the w2v job (orchestrator run id) or null; null when scope=service
    actor    the HUMAN who acted (canonical uuid) or null — never a service;
             null when scope=service
    project  the w2v project (canonical uuid) or null; null when scope=service
    subject  (0.93.0) the licences subject BILLED for the work (opaque id) or
             null; null when scope=service. Always PRESENT on a 0.93.0+
             tenant token — absent means a pre-0.93.0 minter (see below)
    license  (0.93.0) the licence id billed, same rules as ``subject``
    iat/nbf/exp   exp - iat <= 60 s; verified with <= 5 s clock skew
    jti      unique per token; replay-checked in-process for the token window

Configuration (environment, read at call time so a hot reload / monkeypatch is
seen; no secret is ever logged):

==========================  =================================================
``CELL_AUTH_MODE``          receiver lane: ``off`` (default) | ``dual`` | ``enforce``
``CELL_SERVICE_NAME``       this service's cell identity (iss outbound, aud inbound)
``CELL_DEPLOYMENT_ID``      the cell/deployment id (``strato-w2v``)
``CELL_JWT_SIGNING_KEY``    this service's Ed25519 private key, PEM (Secret)
``CELL_JWT_KID``            the kid of that key in the JWKS (Secret)
``CELL_JWKS_PATH``          public keyring file (default ``/etc/w2v-cell/jwks.json``)
``CELL_JWKS_RELOAD_SECONDS`` keyring re-read interval (default 60)
``CELL_ALLOWED_CALLERS``    CSV of issuers this receiver accepts; ``*`` = any
                            keyring issuer; unset/empty = NONE (fail closed)
==========================  =================================================

Keyring file format (the ``w2v-cell-jwks`` ConfigMap)::

    {"keys": [{"kid": "...", "iss": "w2v-orchestrator", "kty": "OKP",
               "crv": "Ed25519", "x": "<base64url public key>"}]}

A missing file empties the keyring (every cell token fails ``keyring_unavailable``
— a 401 in ``enforce``, a counted fall-through in ``dual``). A file that exists
but does not parse keeps the LAST GOOD keyring: kubelet swaps ConfigMap files
atomically, so a corrupt read is a transient or an operator typo, and dropping
every key on it would 401 the whole cell.

Service scope (0.92.0): a tenant-less caller (a tts-gateway worker polling
``/api/tts/claim``) mints ``scope=service`` EXPLICITLY (``mint_service_cell_token``
/ ``service_cell_token_hook``) — a tenant-scope mint that lost its tenant is an
error, never a silent downgrade. Receivers refuse a service token
(``service_scope_not_allowed``) unless the matched route opted in with
:func:`w2v_common.auth.cell_service.require_cell_service_caller` naming the
issuer (``service_caller_not_allowed`` otherwise); the accepted identity has NO
tenant, so tenant-scoped code fails loud.

Signed attribution (0.93.0): ``subject`` / ``license`` are signed claims, so
a cell caller can no longer bill an arbitrary licence or assert a subject via
``X-Subject-Id`` / ``X-License-Id``. A 0.93.0+ minter writes BOTH keys on every
tenant token (``null`` when unbound); the receiver then ignores those headers.
A token with NEITHER key came from a 0.92.x peer: the receiver falls back to
the headers (attribution only — the header subject never reaches authz) and
counts ``w2v_cell_unsigned_attribution_total{caller}``. Exactly one of the two
keys is ``invalid_claim``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, cast
from urllib.parse import urlsplit

import jwt
from cachetools import TTLCache
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from jwt.types import Options
from prometheus_client import REGISTRY, Counter

from ..discovery import cell_services, service_url
from ..tenant import TenantError, parse_tenant_id

logger = logging.getLogger(__name__)

#: A decoded JWT payload. Values are JSON and every one is type-checked before
#: use, so ``object`` (not ``Any``) is the honest element type.
type _Claims = dict[str, object]

#: Wire header. Deliberately NOT ``Authorization``: the Keycloak lane hard-fails
#: any JWT-shaped ``Authorization``, and the licence lane must keep working
#: beside this one during dual-accept.
CELL_TOKEN_HEADER: Final[str] = "X-W2V-Cell-Token"  # noqa: S105  # nosec B105 — a header name
#: JOSE ``typ`` every cell token carries; a token of any other type (a Keycloak
#: access token, an id token) is refused before its signature is looked at.
CELL_TOKEN_TYPE: Final[str] = "w2v-cell+jwt"  # noqa: S105  # nosec B105 — a JOSE typ
CELL_TOKEN_ALGORITHM: Final[str] = "EdDSA"  # noqa: S105  # nosec B105 — an alg name
CELL_TOKEN_TTL_SECONDS: Final[int] = 60
CELL_CLOCK_SKEW_SECONDS: Final[int] = 5
DEFAULT_JWKS_PATH: Final[str] = "/etc/w2v-cell/jwks.json"
DEFAULT_JWKS_RELOAD_SECONDS: Final[float] = 60.0

#: A cell token is ~400 bytes. Anything far larger is not one; refusing it
#: before base64/JSON parsing bounds the work an unauthenticated caller can buy.
_MAX_TOKEN_LENGTH: Final[int] = 4096
#: ``tenant`` is required only for ``scope=tenant`` — checked by hand in
#: :func:`_identity_claims` (a service token may omit it).
_REQUIRED_CLAIMS: Final[tuple[str, ...]] = (
    "iss",
    "aud",
    "cell",
    "iat",
    "nbf",
    "exp",
    "jti",
)
#: JOSE header parameters that would let a token name its OWN verification key.
#: PyJWT ignores them, but a token carrying one is an attack, not a cell token.
_FORBIDDEN_HEADER_PARAMS: Final[frozenset[str]] = frozenset({"jwk", "jku", "x5u", "x5c", "crit"})
_JTI_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
#: Same alphabet as ``identity_context.parse_w2v_job_id`` (a run id rides the
#: token and lands in billing metadata).
_RUN_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
#: ``subject`` / ``license`` are opaque ids (licences UUIDs in practice, but the
#: introspection contract does not promise a shape): printable ASCII, no
#: whitespace, bounded — they land in headers and billing rows.
_ATTRIBUTION_RE: Final[re.Pattern[str]] = re.compile(r"^[\x21-\x7e]{1,256}$")
#: The signed attribution claims, in (subject, license) order.
_ATTRIBUTION_CLAIMS: Final[tuple[str, str]] = ("subject", "license")
#: The replay cache must outlive every token it has seen: a token is acceptable
#: from ``iat - skew`` to ``exp + skew``, at most TTL + 2*skew seconds.
_REPLAY_WINDOW_SECONDS: Final[int] = CELL_TOKEN_TTL_SECONDS + 2 * CELL_CLOCK_SKEW_SECONDS
_REPLAY_CACHE_MAXSIZE: Final[int] = 200_000


class CellAuthMode(StrEnum):
    """Receiver-side lane mode (``CELL_AUTH_MODE``)."""

    OFF = "off"
    DUAL = "dual"
    ENFORCE = "enforce"


class CellScope(StrEnum):
    """What a cell token speaks for (claim ``scope``; absent = ``tenant``)."""

    #: A tenant's work: ``tenant`` claim required (every pre-0.92.0 token).
    TENANT = "tenant"
    #: The calling SERVICE itself, for no tenant (a worker's claim poll).
    #: Accepted only on routes that opt in; never carries tenant/run/actor/project.
    SERVICE = "service"


#: ``scope`` metric label when the token was refused before its scope was known.
SCOPE_UNKNOWN: Final[str] = "unknown"


class CellTokenError(Exception):
    """A cell token was refused. ``reason`` is a bounded metric label.

    ``scope`` is the bounded ``scope`` metric label: ``tenant`` / ``service``
    once the signed claim was read, ``unknown`` before.
    """

    def __init__(self, reason: str, detail: str = "", *, scope: str = SCOPE_UNKNOWN) -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.scope = scope


class CellConfigError(RuntimeError):
    """The signing side is half-configured (or its key is unusable).

    Raised on the outbound path, loudly: a service holding a broken key would
    otherwise send unauthenticated in-cell calls that only fail once a receiver
    flips to ``enforce``.
    """


@dataclass(frozen=True, slots=True)
class CellClaims:
    """The verified content of a cell token."""

    issuer: str
    audience: str
    deployment: str
    #: ``None`` exactly when ``scope`` is :attr:`CellScope.SERVICE`.
    tenant_id: uuid.UUID | None
    run: str | None
    actor: str | None
    project: str | None
    jti: str
    issued_at: int
    expires_at: int
    scope: CellScope = CellScope.TENANT
    #: The licences subject billed (signed ``subject`` claim) or ``None``.
    subject: str | None = None
    #: The licence id billed (signed ``license`` claim) or ``None``.
    license_id: str | None = None
    #: True when the token CARRIES the ``subject``/``license`` claims (a
    #: 0.93.0+ minter; values may still be ``None``). False = a pre-0.93.0
    #: token: the receiver may fall back to the legacy headers (transitional).
    attribution_signed: bool = False


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _counter(name: str, documentation: str, labels: list[str]) -> Counter:
    """Register or re-use a counter (the test suite reloads auth modules)."""
    try:
        return Counter(name, documentation, labels)
    except ValueError:
        existing = REGISTRY._names_to_collectors.get(name)
        if not isinstance(existing, Counter):  # pragma: no cover — defensive
            raise
        return existing


#: Cell tokens refused, by bounded ``reason`` and ``scope`` (tenant | service | unknown).
cell_jwt_rejected_total = _counter(
    "w2v_cell_jwt_rejected_total",
    "Inbound cell tokens refused, by reason and token scope.",
    ["reason", "scope"],
)
#: Licence-lane requests from in-cell callers (a delegating subject, or any
#: request carrying X-Effective-Tenant-Id) on a receiver whose cell lane is on.
#: THE FLIP SIGNAL: a receiver may go to ``enforce`` once this has stayed at 0
#: for 24 h. ``caller`` is bounded: an AUTH_DELEGATING_SUBJECTS entry, or one
#: of the fixed values ``unlisted`` / ``unverified``.
cell_legacy_delegation_total = _counter(
    "w2v_cell_legacy_delegation_total",
    "In-cell requests still authenticated on the legacy licence/delegation lane.",
    ["caller"],
)
#: ``dual`` requests that carried BOTH a valid cell token AND a legacy credential
#: (licence token / JWT / API key). Since 0.91.2 those stay on the legacy lane
#: (behaviour-preserving) and only record the cell caller, so they are NOT part
#: of ``w2v_cell_legacy_delegation_total``. ``caller`` = the verified ``iss``,
#: bounded by the keyring. Non-zero here and zero legacy-only traffic = the
#: callers already sign; the receiver's consumers must stop reading
#: ``request.state.license*`` before its own ``enforce`` flip.
cell_dual_both_total = _counter(
    "w2v_cell_dual_both_total",
    "dual-mode requests carrying both a valid cell token and a legacy credential.",
    ["caller", "scope"],
)
#: Tenant cell tokens accepted WITHOUT signed ``subject``/``license`` claims
#: (minted by a pre-0.93.0 peer), whose attribution therefore came from the
#: ``X-Subject-Id`` / ``X-License-Id`` headers. ``caller`` = the verified ``iss``
#: (bounded by the keyring). 0 fleet-wide for 24 h → every peer signs its
#: attribution and the header fallback can be deleted (C-del).
cell_unsigned_attribution_total = _counter(
    "w2v_cell_unsigned_attribution_total",
    "Tenant cell tokens without signed subject/license claims (header fallback used).",
    ["caller"],
)
#: Keyring file reads, by ``result`` in {ok, missing, invalid}.
cell_jwks_reload_total = _counter(
    "w2v_cell_jwks_reload_total",
    "Cell JWKS keyring file reloads, by result.",
    ["result"],
)


#: In-cell sends that went out WITHOUT a cell token although this service can
#: sign, by bounded ``reason`` (``unvouched_tenant``; ``service_hook_tenant_bound``
#: when a service-scope hook found a tenant bound and left the request alone;
#: ``service_hook_host_mismatch`` when a service-scope hook was asked to send to
#: a host that is not the in-cell endpoint of its audience).
#: Non-zero on a caller is why a receiver in ``enforce`` 401s it.
cell_token_unminted_total = _counter(
    "w2v_cell_token_unminted_total",
    "In-cell outbound requests sent without a cell token, by reason.",
    ["reason"],
)
#: THE DENOMINATOR for the counters above: every inbound request the auth
#: middleware AUTHENTICATED (counted once, just before the handler runs), by
#: ``lane`` and ``scope``. Without it, an absent/zero
#: ``w2v_cell_legacy_delegation_total`` cannot be told apart from "this receiver
#: saw no traffic at all". ``lane``: ``cell`` (the cell lane proper), ``dual``
#: (``dual`` mode, a valid cell token RECORDED alongside a legacy credential that
#: authenticated), ``legacy`` (any non-cell lane — licence, JWT, API key, mesh,
#: trusted network — in every mode, ``off`` included). ``scope``: the verified
#: token's scope (``tenant`` / ``service``) on ``cell``/``dual``; ``none`` on
#: ``legacy`` (no cell token was verified). Bounded labels only.
cell_auth_total = _counter(
    "w2v_cell_auth_total",
    "Inbound requests authenticated by the auth middleware, by cell lane and token scope.",
    ["lane", "scope"],
)

#: ``lane`` label values of ``w2v_cell_auth_total``.
CELL_AUTH_LANE_CELL: Final[str] = "cell"
CELL_AUTH_LANE_DUAL: Final[str] = "dual"
CELL_AUTH_LANE_LEGACY: Final[str] = "legacy"
#: ``scope`` label of ``w2v_cell_auth_total`` when no cell token was verified.
SCOPE_NONE: Final[str] = "none"

#: Every ``w2v_cell_jwt_rejected_total`` ``reason`` paired with the ``scope``
#: values it can actually carry. Refusals before the signature is verified are
#: always ``unknown``; after it, the token's own scope (or ``unknown`` for an
#: unrecognised ``scope`` claim). A test pins this against every reason literal
#: in the source, so a new reason cannot be added without a zero series.
_TOKEN_SCOPES: Final[tuple[str, ...]] = (CellScope.TENANT.value, CellScope.SERVICE.value)
CELL_REJECT_SERIES: Final[tuple[tuple[str, str], ...]] = (
    *(
        (reason, SCOPE_UNKNOWN)
        for reason in (
            "not_configured",
            "malformed",
            "bad_alg",
            "bad_type",
            "bad_header",
            "unknown_kid",
            "missing_claim",
            "bad_signature",
            "keyring_unavailable",
            "invalid_claim",
        )
    ),
    *(
        (reason, scope)
        for reason in (
            "issuer_mismatch",
            "wrong_audience",
            "wrong_deployment",
            "bad_lifetime",
            "expired",
            "not_yet_valid",
            "invalid_claim",
            "missing_claim",
            "invalid_tenant",
            "caller_not_allowed",
            "replay",
        )
        for scope in _TOKEN_SCOPES
    ),
    ("service_scope_not_allowed", CellScope.SERVICE.value),
    ("service_caller_not_allowed", CellScope.SERVICE.value),
    # middleware: dual-mode signed tenant != the legacy lane's tenant
    ("tenant_mismatch", CellScope.TENANT.value),
)
#: Fixed (non-subject) ``caller`` values of ``w2v_cell_legacy_delegation_total``.
LEGACY_DELEGATION_FIXED_CALLERS: Final[tuple[str, ...]] = ("unlisted", "unverified")


def record_cell_unminted(reason: str) -> None:
    """Best-effort: telemetry must never fail a request."""
    try:
        cell_token_unminted_total.labels(reason=reason).inc()
    except Exception:
        logger.debug("cell_token_unminted_metric_failed", exc_info=True)


def record_cell_rejection(reason: str, scope: str = SCOPE_UNKNOWN) -> None:
    """Best-effort: telemetry must never fail a request."""
    try:
        cell_jwt_rejected_total.labels(reason=reason, scope=scope).inc()
    except Exception:
        logger.debug("cell_jwt_rejected_metric_failed", exc_info=True)


def record_dual_both(caller: str, scope: str = CellScope.TENANT.value) -> None:
    """Best-effort: telemetry must never fail a request."""
    try:
        cell_dual_both_total.labels(caller=caller, scope=scope).inc()
    except Exception:
        logger.debug("cell_dual_both_metric_failed", exc_info=True)


def record_unsigned_attribution(caller: str) -> None:
    """Best-effort: telemetry must never fail a request."""
    try:
        cell_unsigned_attribution_total.labels(caller=caller).inc()
    except Exception:
        logger.debug("cell_unsigned_attribution_metric_failed", exc_info=True)


def record_legacy_delegation(caller: str) -> None:
    """Best-effort: telemetry must never fail a request."""
    try:
        cell_legacy_delegation_total.labels(caller=caller).inc()
    except Exception:
        logger.debug("cell_legacy_delegation_metric_failed", exc_info=True)


def record_cell_auth(lane: str, scope: str = SCOPE_NONE) -> None:
    """Best-effort: telemetry must never fail a request."""
    try:
        cell_auth_total.labels(lane=lane, scope=scope).inc()
    except Exception:
        logger.debug("cell_auth_metric_failed", exc_info=True)


def precreate_cell_auth_series(delegating_subjects: Iterable[str] = ()) -> None:
    """Expose every statically-known cell-auth series at 0 before any traffic.

    A labelled prometheus Counter emits nothing until its first ``.labels()``,
    so "no legacy traffic" and "never instrumented / no traffic" both read as an
    ABSENT series. Touching ``.labels(...)`` (no ``inc``) publishes a 0 sample.
    Pre-created: every ``w2v_cell_auth_total`` lane/scope, every
    ``w2v_cell_jwt_rejected_total`` reason/scope in :data:`CELL_REJECT_SERIES`,
    ``w2v_cell_legacy_delegation_total`` for the fixed callers plus this
    receiver's configured ``AUTH_DELEGATING_SUBJECTS``. NOT pre-created:
    ``w2v_cell_dual_both_total`` (``caller`` = a verified issuer, only known
    from live tokens). Idempotent; best-effort.
    """
    try:
        for lane, scope in (
            (CELL_AUTH_LANE_CELL, CellScope.TENANT.value),
            (CELL_AUTH_LANE_CELL, CellScope.SERVICE.value),
            (CELL_AUTH_LANE_DUAL, CellScope.TENANT.value),
            (CELL_AUTH_LANE_DUAL, CellScope.SERVICE.value),
            (CELL_AUTH_LANE_LEGACY, SCOPE_NONE),
        ):
            cell_auth_total.labels(lane=lane, scope=scope)
        for reason, scope in CELL_REJECT_SERIES:
            cell_jwt_rejected_total.labels(reason=reason, scope=scope)
        for caller in (*LEGACY_DELEGATION_FIXED_CALLERS, *delegating_subjects):
            cell_legacy_delegation_total.labels(caller=caller)
    except Exception:
        logger.debug("cell_auth_precreate_failed", exc_info=True)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

_warned_bad_mode: set[str] = set()


def cell_auth_mode() -> CellAuthMode:
    """``CELL_AUTH_MODE``, read at call time. Unset/empty = ``off``.

    An unrecognised value FAILS CLOSED to ``enforce`` (logged once): a typo in
    the mode must never silently disable a receiver's cell check.
    """
    raw = (os.getenv("CELL_AUTH_MODE") or "").strip().lower()
    if not raw:
        return CellAuthMode.OFF
    try:
        return CellAuthMode(raw)
    except ValueError:
        if raw not in _warned_bad_mode:
            _warned_bad_mode.add(raw)
            logger.error(
                "CELL_AUTH_MODE=%r is not one of off|dual|enforce; failing closed to enforce",
                raw,
            )
        return CellAuthMode.ENFORCE


def cell_service_name() -> str:
    """``CELL_SERVICE_NAME``: this service's cell identity, or ``""``."""
    return (os.getenv("CELL_SERVICE_NAME") or "").strip()


def cell_deployment_id() -> str:
    """``CELL_DEPLOYMENT_ID``: the deployment this cell runs as, or ``""``."""
    return (os.getenv("CELL_DEPLOYMENT_ID") or "").strip()


#: ``CELL_ALLOWED_CALLERS=*``: accept any issuer that has a key in the keyring.
ALLOW_ANY_CELL_CALLER = "*"

_warned_no_allowed_callers = False


def cell_allowed_callers() -> frozenset[str]:
    """``CELL_ALLOWED_CALLERS`` as a set of issuers.

    Unset/empty = NO caller is accepted (fail closed, since 0.91.1 — 0.91.0 read
    it as "any keyring issuer"). An explicit ``*`` member = any keyring issuer.
    Any other value (including the gitops deny-all sentinel
    ``_no-in-cell-callers_``) is an exact issuer list.
    """
    raw = os.getenv("CELL_ALLOWED_CALLERS") or ""
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def cell_caller_allowed(issuer: str) -> bool:
    """Whether ``CELL_ALLOWED_CALLERS`` admits ``issuer``. Unset/empty admits nobody."""
    allowed = cell_allowed_callers()
    if not allowed:
        warn_if_no_cell_callers()
        return False
    return ALLOW_ANY_CELL_CALLER in allowed or issuer in allowed


def warn_if_no_cell_callers() -> None:
    """Log ONE warning per process when a dual/enforce receiver accepts no caller.

    Called from :class:`~w2v_common.auth.middleware.AuthMiddleware` construction
    (startup) and from the deny path, so a misconfigured receiver is visible
    without a per-request log flood. A no-op in mode ``off``.
    """
    global _warned_no_allowed_callers
    if _warned_no_allowed_callers or cell_allowed_callers():
        return
    if cell_auth_mode() is CellAuthMode.OFF:
        return
    _warned_no_allowed_callers = True
    logger.warning(
        "CELL_ALLOWED_CALLERS is unset/empty on cell service %r (CELL_AUTH_MODE=%s): "
        "every in-cell caller is rejected; set an explicit issuer list, or '*' "
        "for any keyring issuer",
        cell_service_name() or "<CELL_SERVICE_NAME unset>",
        cell_auth_mode().value,
    )


def _jwks_path() -> str:
    return (os.getenv("CELL_JWKS_PATH") or "").strip() or DEFAULT_JWKS_PATH


def _jwks_reload_seconds() -> float:
    raw = (os.getenv("CELL_JWKS_RELOAD_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_JWKS_RELOAD_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.error("CELL_JWKS_RELOAD_SECONDS=%r is not a number; using 60", raw)
        return DEFAULT_JWKS_RELOAD_SECONDS
    return max(0.0, value)


# --------------------------------------------------------------------------- #
# Keyring (public keys, file-backed, hot-reloaded)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _KeyEntry:
    issuer: str
    key: Ed25519PublicKey


def _b64url_decode(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _parse_jwk(index: int, entry: object) -> tuple[str, _KeyEntry]:
    """One keyring entry → ``(kid, key)``; ``ValueError`` on anything off-spec."""
    if not isinstance(entry, dict):
        raise ValueError(f"key #{index} is not an object")
    entry = cast("dict[str, object]", entry)
    kid = entry.get("kid")
    issuer = entry.get("iss")
    if not isinstance(kid, str) or not kid.strip():
        raise ValueError(f"key #{index} has no kid")
    if not isinstance(issuer, str) or not issuer.strip():
        raise ValueError(f"key {kid!r} has no iss")
    if entry.get("kty") != "OKP" or entry.get("crv") != "Ed25519":
        raise ValueError(f"key {kid!r} is not an Ed25519 OKP key")
    if entry.get("alg", CELL_TOKEN_ALGORITHM) != CELL_TOKEN_ALGORITHM:
        raise ValueError(f"key {kid!r} declares a different alg")
    if "d" in entry:
        # A private key in the public keyring is a leak, not a key.
        raise ValueError(f"key {kid!r} carries private material")
    x = entry.get("x")
    if not isinstance(x, str):
        raise ValueError(f"key {kid!r} has no x")
    try:
        raw = _b64url_decode(x)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"key {kid!r} x is not base64url") from exc
    if len(raw) != 32:
        raise ValueError(f"key {kid!r} x is not a 32-byte Ed25519 key")
    return kid, _KeyEntry(issuer=issuer.strip(), key=Ed25519PublicKey.from_public_bytes(raw))


def parse_jwks(document: object) -> dict[str, _KeyEntry]:
    """Parse a cell JWKS document into ``{kid: (iss, public key)}``. Strict.

    Any malformed entry, a duplicate kid, or a non-Ed25519 key rejects the
    WHOLE document (``ValueError``): a keyring that is partially applied is
    harder to reason about than one that is refused and logged.
    """
    if not isinstance(document, dict):
        raise ValueError("JWKS document is not an object")
    keys = cast("dict[str, object]", document).get("keys")
    if not isinstance(keys, list):
        raise ValueError("JWKS document has no 'keys' list")
    out: dict[str, _KeyEntry] = {}
    for index, entry in enumerate(keys):
        kid, key = _parse_jwk(index, entry)
        if kid in out:
            raise ValueError(f"duplicate kid {kid!r}")
        out[kid] = key
    return out


class CellKeyring:
    """Public keys from the mounted JWKS file, re-read every ``reload_seconds``.

    Reloads lazily on access (no thread, no network). ``clock`` is injectable
    so tests can step time instead of sleeping.
    """

    def __init__(
        self,
        path: str,
        *,
        reload_seconds: float = DEFAULT_JWKS_RELOAD_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._path = path
        self._reload_seconds = reload_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._keys: dict[str, _KeyEntry] = {}
        self._available = False
        self._loaded_at: float | None = None

    @property
    def path(self) -> str:
        return self._path

    def reload(self) -> None:
        """Re-read the file now. Missing → empty; unparsable → keep last good."""
        with self._lock:
            first_load = self._loaded_at is None
            self._loaded_at = self._clock()
            try:
                text = Path(self._path).read_text(encoding="utf-8")
            except FileNotFoundError:
                if self._available or first_load:
                    logger.warning("cell JWKS file %s is missing; keyring empty", self._path)
                self._keys = {}
                self._available = False
                cell_jwks_reload_total.labels(result="missing").inc()
                return
            except OSError as exc:
                logger.error(
                    "cell JWKS file %s unreadable (%s); keeping last good keyring",
                    self._path,
                    type(exc).__name__,
                )
                cell_jwks_reload_total.labels(result="invalid").inc()
                return
            try:
                keys = parse_jwks(json.loads(text))
            except ValueError as exc:  # JSONDecodeError is a ValueError
                logger.error(
                    "cell JWKS file %s is invalid (%s); keeping last good keyring (%d keys)",
                    self._path,
                    exc,
                    len(self._keys),
                )
                cell_jwks_reload_total.labels(result="invalid").inc()
                return
            self._keys = keys
            self._available = True
            cell_jwks_reload_total.labels(result="ok").inc()

    def _maybe_reload(self) -> None:
        loaded_at = self._loaded_at
        if loaded_at is None or self._clock() - loaded_at >= self._reload_seconds:
            self.reload()

    def available(self) -> bool:
        """True when a keyring is loaded (a missing file makes this False)."""
        self._maybe_reload()
        return self._available

    def get(self, kid: str) -> _KeyEntry | None:
        """The key for ``kid``; raises ``keyring_unavailable`` when none is loaded."""
        self._maybe_reload()
        if not self._available:
            raise CellTokenError("keyring_unavailable", self._path)
        return self._keys.get(kid)

    def kids(self) -> frozenset[str]:
        self._maybe_reload()
        return frozenset(self._keys)


_keyring: CellKeyring | None = None
_keyring_lock = threading.Lock()


def get_cell_keyring() -> CellKeyring:
    """The process keyring for ``CELL_JWKS_PATH`` (rebuilt if the path changes)."""
    global _keyring
    path = _jwks_path()
    with _keyring_lock:
        if _keyring is None or _keyring.path != path:
            _keyring = CellKeyring(path, reload_seconds=_jwks_reload_seconds())
        return _keyring


# --------------------------------------------------------------------------- #
# Replay cache
# --------------------------------------------------------------------------- #

_replay_lock = threading.Lock()
_replay_seen: TTLCache[str, bool] = TTLCache(
    maxsize=_REPLAY_CACHE_MAXSIZE, ttl=_REPLAY_WINDOW_SECONDS
)


def _claim_jti(jti: str) -> bool:
    """Record ``jti``; False when it was already seen inside the window."""
    with _replay_lock:
        if jti in _replay_seen:
            return False
        _replay_seen[jti] = True
        return True


def reset_cell_state() -> None:
    """Drop cached keyring, signer and replay state (tests, key rotation drills)."""
    global _keyring, _signer_cache, _warned_no_allowed_callers
    _warned_no_allowed_callers = False
    with _keyring_lock:
        _keyring = None
    with _replay_lock:
        _replay_seen.clear()
    with _signer_lock:
        _signer_cache = None
    _warned_bad_mode.clear()


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def _optional_uuid(claims: _Claims, name: str) -> str | None:
    value = claims.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CellTokenError("invalid_claim", name)
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise CellTokenError("invalid_claim", name) from exc
    canonical = str(parsed)
    if canonical != value.lower() or parsed.int == 0:
        raise CellTokenError("invalid_claim", name)
    return canonical


def _int_claim(claims: _Claims, name: str) -> int:
    value = claims.get(name)
    # bool is an int subclass; a ``true`` timestamp is not a timestamp.
    if not isinstance(value, int) or isinstance(value, bool):
        raise CellTokenError("invalid_claim", name)
    return value


_DECODE_OPTIONS: Final[Options] = {
    "verify_signature": True,
    # Time, audience and issuer are checked below against an injectable clock
    # and the keyring binding — PyJWT's versions read the wall clock and know
    # nothing of the kid -> iss pairing.
    "verify_exp": False,
    "verify_nbf": False,
    "verify_iat": False,
    "verify_aud": False,
    "verify_iss": False,
    "require": list(_REQUIRED_CLAIMS),
}


def _verify_header(token: str, keyring: CellKeyring) -> _KeyEntry:
    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as exc:
        raise CellTokenError("malformed") from exc
    if header.get("alg") != CELL_TOKEN_ALGORITHM:
        # alg=none, HS256 (public-key-as-HMAC-secret confusion), RS256 …
        raise CellTokenError("bad_alg")
    if header.get("typ") != CELL_TOKEN_TYPE:
        raise CellTokenError("bad_type")
    if _FORBIDDEN_HEADER_PARAMS & set(header):
        raise CellTokenError("bad_header")
    kid = header.get("kid")
    if not isinstance(kid, str) or not kid:
        raise CellTokenError("unknown_kid")
    entry = keyring.get(kid)
    if entry is None:
        raise CellTokenError("unknown_kid")
    return entry


def _decode_signed(token: str, entry: _KeyEntry) -> _Claims:
    try:
        return jwt.decode(
            token,
            entry.key,
            algorithms=[CELL_TOKEN_ALGORITHM],
            options=_DECODE_OPTIONS,
        )
    except jwt.MissingRequiredClaimError as exc:
        raise CellTokenError("missing_claim", exc.claim) from exc
    except jwt.InvalidSignatureError as exc:
        raise CellTokenError("bad_signature") from exc
    except jwt.InvalidAlgorithmError as exc:
        raise CellTokenError("bad_alg") from exc
    except jwt.InvalidTokenError as exc:
        raise CellTokenError("malformed") from exc


def _verify_time(claims: _Claims, now: float) -> tuple[int, int]:
    iat = _int_claim(claims, "iat")
    nbf = _int_claim(claims, "nbf")
    exp = _int_claim(claims, "exp")
    skew = CELL_CLOCK_SKEW_SECONDS
    if exp <= iat or exp - iat > CELL_TOKEN_TTL_SECONDS:
        raise CellTokenError("bad_lifetime")
    if now > exp + skew:
        raise CellTokenError("expired")
    if now < nbf - skew or now < iat - skew:
        raise CellTokenError("not_yet_valid")
    return iat, exp


def _scope_claim(claims: _Claims) -> CellScope:
    """The ``scope`` claim; absent = ``tenant`` (every pre-0.92.0 token)."""
    if "scope" not in claims:
        return CellScope.TENANT
    raw = claims["scope"]
    if not isinstance(raw, str):
        raise CellTokenError("invalid_claim", "scope")
    try:
        return CellScope(raw)
    except ValueError as exc:
        raise CellTokenError("invalid_claim", "scope") from exc


def is_valid_attribution_value(value: object) -> bool:
    """Whether ``value`` may ride a cell token as ``subject`` / ``license``."""
    return isinstance(value, str) and _ATTRIBUTION_RE.match(value) is not None


def _attribution_claims(claims: _Claims, scope: CellScope) -> tuple[str | None, str | None, bool]:
    """``(subject, license, signed)`` with every shape enforced.

    ``signed`` is True when both keys are present (a 0.93.0+ minter, values
    may be null), False when both are absent (a pre-0.93.0 minter). One
    without the other is ``invalid_claim``: the minter always writes the pair,
    so a half pair is not a version skew. A service token must carry neither
    value (absent or null).
    """
    present = [name in claims for name in _ATTRIBUTION_CLAIMS]
    if any(present) and not all(present):
        raise CellTokenError("invalid_claim", "subject/license must be signed together")
    values: list[str | None] = []
    for name in _ATTRIBUTION_CLAIMS:
        value = claims.get(name)
        if value is None:
            values.append(None)
            continue
        if scope is CellScope.SERVICE:
            raise CellTokenError("invalid_claim", f"{name} on a service-scope token")
        if not is_valid_attribution_value(value):
            raise CellTokenError("invalid_claim", name)
        values.append(cast("str", value))
    return values[0], values[1], all(present)


def _identity_claims(
    claims: _Claims, scope: CellScope
) -> tuple[uuid.UUID | None, str | None, str | None, str | None, str]:
    """``(tenant, run, actor, project, jti)`` with every shape enforced.

    ``scope=service``: the tenant claim must be absent or JSON null (an empty
    string, the nil uuid or any real tenant is ``invalid_tenant``) and run /
    actor / project must be null — a service token speaks for no tenant's work.
    """
    tenant_id: uuid.UUID | None
    if scope is CellScope.SERVICE:
        if claims.get("tenant") is not None:
            raise CellTokenError("invalid_tenant", "service scope carries a tenant")
        for name in ("run", "actor", "project"):
            if claims.get(name) is not None:
                raise CellTokenError("invalid_claim", f"{name} on a service-scope token")
        tenant_id = None
    else:
        if "tenant" not in claims:
            raise CellTokenError("missing_claim", "tenant")
        try:
            tenant_id = parse_tenant_id(claims.get("tenant"), source="cell token tenant claim")
        except TenantError as exc:
            raise CellTokenError("invalid_tenant") from exc
    actor = _optional_uuid(claims, "actor")
    project = _optional_uuid(claims, "project")
    run = claims.get("run")
    if run is not None and (not isinstance(run, str) or not _RUN_RE.match(run)):
        raise CellTokenError("invalid_claim", "run")
    jti = claims.get("jti")
    if not isinstance(jti, str) or not _JTI_RE.match(jti):
        raise CellTokenError("invalid_claim", "jti")
    return tenant_id, run, actor, project, jti


def verify_cell_token(
    token: str,
    *,
    keyring: CellKeyring | None = None,
    now: float | None = None,
    service_callers: frozenset[str] | None = None,
) -> CellClaims:
    """Verify an inbound cell token for THIS service. Raises :class:`CellTokenError`.

    Order: shape → header (alg pinned to EdDSA, typ, kid) → signature against
    the kid's key → kid→iss binding → aud == ``CELL_SERVICE_NAME`` → cell ==
    ``CELL_DEPLOYMENT_ID`` → lifetime/exp/nbf (≤5 s skew) → scope + claim
    shapes → allowed callers → service-scope admission → jti replay. The jti
    is recorded only after everything else passed, so a forged (or refused)
    token cannot poison the replay cache.

    ``service_callers``: the issuers the TARGET ROUTE admits as service
    callers (from :func:`~w2v_common.auth.cell_service.require_cell_service_caller`).
    ``None`` (the default — every route that did not opt in) refuses every
    ``scope=service`` token as ``service_scope_not_allowed``; an issuer not in
    the set is ``service_caller_not_allowed``. Tenant tokens ignore it.
    """
    service = cell_service_name()
    deployment = cell_deployment_id()
    if not service or not deployment:
        raise CellTokenError("not_configured", "CELL_SERVICE_NAME / CELL_DEPLOYMENT_ID unset")
    if not token or len(token) > _MAX_TOKEN_LENGTH or token.count(".") != 2:
        raise CellTokenError("malformed")
    ring = keyring if keyring is not None else get_cell_keyring()
    entry = _verify_header(token, ring)
    claims = _decode_signed(token, entry)
    # Signed from here on: label every later refusal with the token's scope.
    raw_scope = claims.get("scope", CellScope.TENANT.value)
    scope_label = next((s.value for s in CellScope if s.value == raw_scope), SCOPE_UNKNOWN)
    try:
        return _verify_claims(claims, entry, service, deployment, now, service_callers)
    except CellTokenError as exc:
        exc.scope = scope_label
        raise


def _verify_claims(
    claims: _Claims,
    entry: _KeyEntry,
    service: str,
    deployment: str,
    now: float | None,
    service_callers: frozenset[str] | None,
) -> CellClaims:
    issuer = claims.get("iss")
    if not isinstance(issuer, str) or issuer != entry.issuer:
        # A valid key signing as somebody else: one compromised service must
        # not be able to impersonate another.
        raise CellTokenError("issuer_mismatch")
    audience = claims.get("aud")
    if not isinstance(audience, str) or audience != service:
        raise CellTokenError("wrong_audience")
    if claims.get("cell") != deployment:
        raise CellTokenError("wrong_deployment")
    iat, exp = _verify_time(claims, time.time() if now is None else now)
    scope = _scope_claim(claims)
    tenant_id, run, actor, project, jti = _identity_claims(claims, scope)
    subject, license_id, attribution_signed = _attribution_claims(claims, scope)

    if not cell_caller_allowed(issuer):
        raise CellTokenError("caller_not_allowed")
    if scope is CellScope.SERVICE:
        if service_callers is None:
            raise CellTokenError("service_scope_not_allowed")
        if issuer not in service_callers:
            raise CellTokenError("service_caller_not_allowed")
    if not _claim_jti(jti):
        raise CellTokenError("replay")
    return CellClaims(
        issuer=issuer,
        audience=audience,
        deployment=deployment,
        tenant_id=tenant_id,
        run=run,
        actor=actor,
        project=project,
        jti=jti,
        issued_at=iat,
        expires_at=exp,
        scope=scope,
        subject=subject,
        license_id=license_id,
        attribution_signed=attribution_signed and scope is CellScope.TENANT,
    )


# --------------------------------------------------------------------------- #
# Signing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Signer:
    fingerprint: str
    kid: str
    issuer: str
    deployment: str
    key: Ed25519PrivateKey


_signer_cache: _Signer | None = None
_signer_lock = threading.Lock()


def _load_signer() -> _Signer | None:
    """The configured signer, ``None`` when signing is not configured at all.

    All four of ``CELL_JWT_SIGNING_KEY``, ``CELL_JWT_KID``, ``CELL_SERVICE_NAME``
    and ``CELL_DEPLOYMENT_ID`` are needed; a partial set raises
    :class:`CellConfigError`. The parsed key is cached by a digest of the
    config, never by the key text itself.
    """
    global _signer_cache
    pem = (os.getenv("CELL_JWT_SIGNING_KEY") or "").strip()
    if not pem:
        return None
    kid = (os.getenv("CELL_JWT_KID") or "").strip()
    issuer = cell_service_name()
    deployment = cell_deployment_id()
    missing = [
        name
        for name, value in (
            ("CELL_JWT_KID", kid),
            ("CELL_SERVICE_NAME", issuer),
            ("CELL_DEPLOYMENT_ID", deployment),
        )
        if not value
    ]
    if missing:
        raise CellConfigError(f"CELL_JWT_SIGNING_KEY is set but {', '.join(missing)} is not")
    fingerprint = hashlib.sha256(
        "\x00".join((pem, kid, issuer, deployment)).encode("utf-8")
    ).hexdigest()
    with _signer_lock:
        cached = _signer_cache
        if cached is not None and cached.fingerprint == fingerprint:
            return cached
        try:
            key = serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
        except (ValueError, TypeError) as exc:
            # The message from cryptography never echoes key bytes, but keep it
            # out anyway: type only.
            raise CellConfigError(
                f"CELL_JWT_SIGNING_KEY is not a PEM private key ({type(exc).__name__})"
            ) from None
        if not isinstance(key, Ed25519PrivateKey):
            raise CellConfigError("CELL_JWT_SIGNING_KEY is not an Ed25519 key")
        _signer_cache = _Signer(
            fingerprint=fingerprint, kid=kid, issuer=issuer, deployment=deployment, key=key
        )
        return _signer_cache


def cell_signer_configured() -> bool:
    """True when this service can mint cell tokens (raises if half-configured)."""
    return _load_signer() is not None


def _mint_scope(
    scope: CellScope | str,
    tenant: uuid.UUID | str | None,
    run: str | None,
    actor: str | None,
    project: str | None,
    subject_id: str | None,
    license_id: str | None,
) -> tuple[CellScope, str | None]:
    """``(scope, tenant claim)`` for a mint, or ``ValueError`` / ``TenantError``."""
    try:
        token_scope = CellScope(scope)
    except ValueError:
        raise ValueError(f"unknown cell token scope {scope!r}") from None
    if token_scope is CellScope.TENANT:
        # A lost tenant is an ERROR here, never a silent service-scope downgrade.
        return token_scope, str(parse_tenant_id(tenant, source="mint_cell_token"))
    if tenant is not None:
        raise ValueError("a service-scope cell token must not carry a tenant")
    if run is not None or actor is not None or project is not None:
        raise ValueError("a service-scope cell token must not carry run/actor/project")
    if subject_id is not None or license_id is not None:
        raise ValueError("a service-scope cell token must not carry subject/license")
    return token_scope, None


def mint_cell_token(
    aud: str,
    *,
    tenant: uuid.UUID | str | None = None,
    scope: CellScope | str = CellScope.TENANT,
    run: str | None = None,
    actor: str | None = None,
    project: str | None = None,
    subject_id: str | None = None,
    license_id: str | None = None,
    now: float | None = None,
) -> str:
    """Mint one cell token for a single request to ``aud``.

    Never cache or reuse the result: every request gets its own (60 s TTL,
    unique jti). Raises :class:`CellConfigError` when signing is not
    configured and ``ValueError`` for a bad argument: ``scope="tenant"`` (the
    default) needs a real tenant (``TenantError`` for a missing/nil one — a
    lost tenant is NEVER downgraded to a service token); ``scope="service"``
    must carry no tenant, run, actor or project; an unknown scope, or a
    malformed run/actor/project/subject/license.

    ``subject_id`` / ``license_id`` (0.93.0) are the billed licences subject
    and licence; a tenant token ALWAYS carries both claims (``null`` when not
    given), which tells the receiver to ignore ``X-Subject-Id`` /
    ``X-License-Id``. A service token carries neither.
    """
    token_scope, tenant_claim = _mint_scope(
        scope, tenant, run, actor, project, subject_id, license_id
    )
    signer = _load_signer()
    if signer is None:
        raise CellConfigError("cell signing is not configured (CELL_JWT_SIGNING_KEY unset)")
    if not aud:
        raise ValueError("cell token audience is empty")
    for name, value in (("actor", actor), ("project", project)):
        if value is not None:
            try:
                _optional_uuid({name: value}, name)
            except CellTokenError as exc:
                raise ValueError(f"cell token {name} is not a canonical uuid") from exc
    if run is not None and not _RUN_RE.match(run):
        raise ValueError("cell token run is not a valid w2v job id")
    for name, value in (("subject", subject_id), ("license", license_id)):
        if value is not None and not is_valid_attribution_value(value):
            raise ValueError(f"cell token {name} is not a valid opaque id")
    issued = int(time.time() if now is None else now)
    payload: dict[str, object] = {
        "iss": signer.issuer,
        "aud": aud,
        "cell": signer.deployment,
        "scope": token_scope.value,
        "tenant": tenant_claim,
        "run": run,
        "actor": actor.lower() if actor else None,
        "project": project.lower() if project else None,
        "iat": issued,
        "nbf": issued,
        "exp": issued + CELL_TOKEN_TTL_SECONDS,
        "jti": uuid.uuid7().hex,
    }
    if token_scope is CellScope.TENANT:
        # Both keys, always: their PRESENCE is what tells a 0.93.0+ receiver
        # to ignore the X-Subject-Id / X-License-Id headers.
        payload["subject"] = subject_id
        payload["license"] = license_id
    return jwt.encode(
        payload,
        signer.key,
        algorithm=CELL_TOKEN_ALGORITHM,
        headers={"kid": signer.kid, "typ": CELL_TOKEN_TYPE},
    )


def mint_service_cell_token(aud: str, *, now: float | None = None) -> str:
    """Mint a ``scope=service`` (tenant-less) cell token for one request to ``aud``.

    For callers that act for NO tenant (a worker's claim poll). Accepted only
    on receiver routes that opted in for this issuer.
    """
    return mint_cell_token(aud, scope=CellScope.SERVICE, now=now)


def _host_port(url: str) -> tuple[str, int] | None:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if not host:
        return None
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    return host, port


def cell_audience_for_url(url: str) -> str | None:
    """The in-cell callee's identity (``aud``) for a request URL, else ``None``.

    Matches host AND port against every in-cell peer's resolved
    ``service_url`` (so ``W2V_<NAME>_URL`` overrides and ``W2V_LOCAL`` are
    honoured — with ``W2V_LOCAL`` every peer is ``localhost`` and only the port
    tells them apart). Anything else — WM, kag, artifacts, an internal-only hostname
    services, the open internet — is out of cell and never gets a cell token.
    """
    target = _host_port(url)
    if target is None:
        return None
    for name, k8s_service in cell_services().items():
        try:
            peer = _host_port(service_url(name))
        except KeyError:  # pragma: no cover — cell_services() only yields known names
            continue
        if peer == target:
            return k8s_service
    return None
