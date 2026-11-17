"""Per-tenant tenant-x-app licence tokens minted from the tenant's licence bundle.

Owner ruling 2026-10-15: there are no app-to-app licences, only tenant-x-app
licences. A w2v service calling worker_management therefore presents the END
tenant's own ``workers`` licence, not the shared ``svc-w2v-shared`` licence plus
``X-Effective-Tenant-Id`` delegation. This module mints that licence at runtime::

    POST {licenses}/internal/tenants/{tenant}/bundle-token?app=workers&ttl_minutes=60[&bundle=<key>]
    Authorization: Bearer <client_credentials JWT of BUNDLE_MINTER_CLIENT_ID>
    201 {token, expires_at, tenant_id, bundle, app, license_id, token_id}

The licences route admits realm role ``bundle-token-minter`` ONLY, and the
caller's api_client must own the tenant or hold an active grant on it. The SA is
a DEDICATED client (``w2v-bundle-minter`` in realm ``secure-apps``), never the
``w2v-backend`` client the rest of w2v_common uses — so its token URL is its own
setting (``BUNDLE_MINTER_TOKEN_URL``) with NO fallback to ``KEYCLOAK_URL``.

Template: worker_management ``src/services/ir_bundle_tokens.py``
``BundleTokenCache``. Differences, each deliberate:

* **Fail loud, not soft.** WM's IR path may go tokenless; a w2v -> WM call
  without the tenant's licence has nobody to bill, so every failure raises a
  typed :class:`BundleTokenError` carrying licences' ``detail.error`` code.
* Keyed per ``(tenant, app)``; optional ``bundle`` (``BUNDLE_MINTER_BUNDLE``)
  disambiguates a tenant holding several bundles that contain the app (else 409).

Rules (pinned by ``tests/test_bundle_tokens.py``):

* In memory only, per ``(tenant, app)``, until ``expires_at - 5 min``.
* Single-flight per key: N concurrent cold calls cause ONE mint; other keys
  do not wait. Locks are keyed by event loop as well (an asyncio.Lock binds to one).
* Refusals (404 ``tenant_not_found`` / ``app_not_in_bundle`` / ``no_app_licence``,
  403, 409) are negative-cached for :data:`NEGATIVE_TTL_S` and re-raised from the
  cache, so a tenant with no licence cannot turn every call into a licences
  call. Transient failures (5xx, transport, Keycloak) are NOT cached.
* Tokens, the SA JWT and the client secret are never logged.
* ``w2v_bundle_token_mint_total{app,outcome}`` counts every resolution.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final
from uuid import UUID

import httpx
from prometheus_client import REGISTRY, Counter

from ..discovery import service_url

logger = logging.getLogger(__name__)

#: worker_management's key in the licences app registry.
WORKERS_APP_KEY: Final = "workers"
TTL_MINUTES: Final = 60
#: A cached token is replaced this long before its own ``expires_at``.
REFRESH_LEAD_S: Final = 300.0
#: How long a licences REFUSAL is remembered before the next mint attempt.
NEGATIVE_TTL_S: Final = 60.0
_SA_REFRESH_LEAD_S: Final = 30.0
_SA_DEFAULT_EXPIRY_S: Final = 300
_HTTP_TIMEOUT_S: Final = 5.0
#: Past this many cache entries, expired ones are pruned on the next insert.
_MAX_ENTRIES: Final = 4096

#: Bounded outcome label set of ``w2v_bundle_token_mint_total``.
MINT_OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        "minted",  # fresh token from licences
        "cache_hit",  # served from the positive cache
        "negative_cache_hit",  # a remembered refusal, re-raised without a licences call
        "not_found",  # 404 (tenant_not_found / app_not_in_bundle / no_app_licence / ...)
        "forbidden",  # 403 (unknown_api_client / role missing)
        "conflict",  # 409 (ambiguous_bundle / ...)
        "unauthenticated",  # 401: our SA token was rejected
        "rejected",  # any other 4xx
        "server_error",  # 5xx
        "transport_error",
        "bad_response",  # 2xx with an unusable body
        "sa_error",  # the Keycloak client_credentials mint failed
        "not_configured",  # BUNDLE_MINTER_* settings missing
    }
)

_METRIC_NAME: Final = "w2v_bundle_token_mint_total"


def _mint_counter() -> Counter:
    """Register (or re-use, on module reload) the mint counter."""
    try:
        return Counter(
            _METRIC_NAME,
            "Tenant-x-app bundle-token resolutions by app key and outcome "
            "(minted|cache_hit|negative_cache_hit|not_found|forbidden|conflict|"
            "unauthenticated|rejected|server_error|transport_error|bad_response|"
            "sa_error|not_configured).",
            ["app", "outcome"],
        )
    except ValueError:
        existing = REGISTRY._names_to_collectors.get(_METRIC_NAME)
        if not isinstance(existing, Counter):  # pragma: no cover - defensive
            raise
        return existing


BUNDLE_TOKEN_MINT_TOTAL = _mint_counter()


def _record(app: str, outcome: str) -> None:
    if outcome not in MINT_OUTCOMES:  # pragma: no cover - programming error
        raise KeyError(f"unknown bundle-token outcome {outcome!r}")
    BUNDLE_TOKEN_MINT_TOTAL.labels(app=app, outcome=outcome).inc()


class WMAuthMode(StrEnum):
    """How w2v authenticates to worker_management (env ``WM_AUTH_MODE``).

    ``delegation`` (default): today's shared SA licence + ``X-Effective-Tenant-Id``.
    Kept for rollout only and deleted once WM enforces tenant licences.
    ``tenant_licence``: the END tenant's own minted ``workers`` licence; no
    delegation header; no tenant = :class:`NoTenantForLicenceError`.
    """

    DELEGATION = "delegation"
    TENANT_LICENCE = "tenant_licence"


def resolve_wm_auth_mode(value: str | WMAuthMode | None = None) -> WMAuthMode:
    """``value`` if given, else env ``WM_AUTH_MODE``, else ``delegation``.

    An unknown value RAISES ``ValueError``: silently reading a typo as
    ``delegation`` would keep billing the shared SA while the operator believes
    the flip happened.
    """
    raw = value if value is not None else os.environ.get("WM_AUTH_MODE", "")
    raw = str(raw).strip().lower() or WMAuthMode.DELEGATION.value
    try:
        return WMAuthMode(raw)
    except ValueError:
        allowed = ", ".join(m.value for m in WMAuthMode)
        raise ValueError(f"WM_AUTH_MODE={raw!r} is not one of: {allowed}") from None


class BundleTokenError(RuntimeError):
    """No tenant-x-app licence token could be obtained. Fail closed.

    ``error_code`` is licences' ``detail.error`` when it sent one (e.g.
    ``no_app_licence``, ``unknown_api_client``, ``ambiguous_bundle``), else a
    local code (``transport_error``, ``sa_error``, ``malformed_response``, ...).
    The message never contains a token or secret.
    """

    def __init__(
        self,
        tenant_id: str,
        app: str,
        error_code: str,
        *,
        status_code: int | None = None,
        outcome: str,
    ) -> None:
        self.tenant_id = tenant_id
        self.app = app
        self.error_code = error_code
        self.status_code = status_code
        self.outcome = outcome
        status = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"no {app} licence token for tenant {tenant_id}: {error_code}{status}")


class BundleTokenRefusedError(BundleTokenError):
    """licences REFUSED the mint (401/403/404/409/other 4xx). Not transient."""


class BundleTokenUnavailableError(BundleTokenError):
    """A transient failure: 5xx, transport error, Keycloak failure, bad body."""


class BundleMinterNotConfiguredError(BundleTokenError):
    """``WM_AUTH_MODE=tenant_licence`` but the ``BUNDLE_MINTER_*`` settings are missing."""


@dataclass(frozen=True)
class MintedLicence:
    """A minted tenant-x-app licence. ``token`` is hidden from ``repr``.

    ``license_id`` is the licence the token is ON (licences' response), which is
    the attribution an ``X-License-Id`` header must carry next to it.
    """

    token: str = field(repr=False)
    license_id: str | None = None
    token_id: str | None = None


@dataclass
class _Entry:
    licence: MintedLicence | None
    valid_until: float  # wall-clock seconds; used while now < this
    error: BundleTokenRefusedError | None = None


@dataclass
class _SAToken:
    token: str = field(repr=False)
    valid_until: float = 0.0


def _parse_expires_at(raw: object) -> float:
    if not isinstance(raw, str):
        raise ValueError("expires_at is not a string")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _error_code(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return ""
    if not isinstance(body, dict):
        return ""
    detail = body.get("detail")
    if isinstance(detail, dict) and detail.get("error"):
        return str(detail["error"])
    return ""


def _canonical_tenant(tenant_id: str) -> str:
    try:
        return str(UUID(str(tenant_id).strip()))
    except ValueError:
        raise BundleTokenRefusedError(
            str(tenant_id), "?", "invalid_tenant_id", outcome="rejected"
        ) from None


class BundleTokenMinter:
    """Per-(tenant, app) bundle-token cache with per-key single-flight minting."""

    def __init__(
        self,
        *,
        licenses_url: str,
        token_url: str,
        client_id: str,
        client_secret: str,
        bundle: str | None = None,
        ttl_minutes: int = TTL_MINUTES,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._licenses_url = licenses_url.rstrip("/")
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._bundle = bundle or None
        self._ttl_minutes = ttl_minutes
        self._transport = transport
        self._clock = clock
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._sa: _SAToken | None = None
        # Locks are keyed by the event-loop OBJECT (weakly: a closed loop's locks
        # go with it; an ``id()`` can be reused by a later loop), and held weakly
        # per key: a lock lives exactly as long as some coroutine is using it, so
        # the map cannot grow with every tenant ever seen.
        self._locks: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, weakref.WeakValueDictionary[tuple[str, str], asyncio.Lock]
        ] = weakref.WeakKeyDictionary()
        self._sa_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
            weakref.WeakKeyDictionary()
        )

    # -- cache -----------------------------------------------------------------

    def _fresh(self, key: tuple[str, str]) -> _Entry | None:
        entry = self._entries.get(key)
        if entry is not None and self._clock() < entry.valid_until:
            return entry
        return None

    def _serve(self, entry: _Entry, app: str) -> MintedLicence:
        if entry.error is not None:
            _record(app, "negative_cache_hit")
            raise entry.error
        if entry.licence is None:  # pragma: no cover - positive entries always carry one
            raise RuntimeError("bundle-token cache entry has neither a token nor an error")
        _record(app, "cache_hit")
        return entry.licence

    def _store(self, key: tuple[str, str], entry: _Entry) -> None:
        if len(self._entries) >= _MAX_ENTRIES:
            now = self._clock()
            for stale in [k for k, e in self._entries.items() if e.valid_until <= now]:
                del self._entries[stale]
        self._entries[key] = entry

    def _key_lock(self, key: tuple[str, str]) -> asyncio.Lock:
        per_loop = self._locks.setdefault(asyncio.get_running_loop(), weakref.WeakValueDictionary())
        lock = per_loop.get(key)
        if lock is None:
            lock = asyncio.Lock()
            per_loop[key] = lock
        return lock

    def invalidate(self, tenant_id: str, app: str = WORKERS_APP_KEY) -> None:
        """Drop the cached entry for ``(tenant, app)`` (e.g. WM answered 401/403)."""
        self._entries.pop((_canonical_tenant(tenant_id), app), None)

    async def token(self, tenant_id: str, app: str = WORKERS_APP_KEY) -> str:
        """A valid licence token for ``(tenant, app)``. Raises :class:`BundleTokenError`."""
        return (await self.licence(tenant_id, app)).token

    async def licence(self, tenant_id: str, app: str = WORKERS_APP_KEY) -> MintedLicence:
        """The minted licence (token + licence id) for ``(tenant, app)``."""
        key = (_canonical_tenant(tenant_id), app)
        entry = self._fresh(key)
        if entry is not None:
            return self._serve(entry, app)
        lock = self._key_lock(key)
        async with lock:
            entry = self._fresh(key)  # another caller may have minted while we waited
            if entry is not None:
                return self._serve(entry, app)
            return await self._mint(*key)

    # -- remote ----------------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self._transport, timeout=_HTTP_TIMEOUT_S)

    def _fresh_sa(self) -> str | None:
        if self._sa is not None and self._clock() < self._sa.valid_until:
            return self._sa.token
        return None

    async def _sa_token(self, client: httpx.AsyncClient, tenant: str, app: str) -> str:
        """The minter SA's access token: cached, refreshed 30 s early, single-flight."""
        cached = self._fresh_sa()
        if cached is not None:
            return cached
        loop = asyncio.get_running_loop()
        lock = self._sa_locks.get(loop)
        if lock is None:
            lock = self._sa_locks[loop] = asyncio.Lock()
        async with lock:
            cached = self._fresh_sa()
            if cached is not None:
                return cached
            return await self._fetch_sa(client, tenant, app)

    async def _fetch_sa(self, client: httpx.AsyncClient, tenant: str, app: str) -> str:
        now = self._clock()

        def fail(code: str, status: int | None = None) -> BundleTokenUnavailableError:
            _record(app, "sa_error")
            logger.error(
                "w2v_bundle_token outcome=sa_error client_id=%s status=%s error=%s",
                self._client_id,
                status,
                code,
            )
            return BundleTokenUnavailableError(
                tenant, app, code, status_code=status, outcome="sa_error"
            )

        try:
            resp = await client.post(
                self._token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
            )
        except httpx.HTTPError as exc:
            raise fail(f"minter_auth_unreachable:{type(exc).__name__}") from exc
        if resp.status_code != httpx.codes.OK:
            raise fail("minter_auth_rejected", resp.status_code)
        try:
            payload = resp.json()
            access = payload.get("access_token")
            expires_in = int(payload.get("expires_in") or _SA_DEFAULT_EXPIRY_S)
        except ValueError, TypeError, AttributeError:
            access, expires_in = None, 0
        if not isinstance(access, str) or not access:
            raise fail("minter_auth_malformed", resp.status_code)
        self._sa = _SAToken(access, now + max(expires_in - _SA_REFRESH_LEAD_S, 1))
        return access

    def _refuse(self, tenant: str, app: str, resp: httpx.Response) -> BundleTokenError:
        status = resp.status_code
        code = _error_code(resp)
        if status == httpx.codes.NOT_FOUND:
            outcome = "not_found"
        elif status == httpx.codes.FORBIDDEN:
            outcome = "forbidden"
        elif status == httpx.codes.CONFLICT:
            outcome = "conflict"
        elif status == httpx.codes.UNAUTHORIZED:
            outcome = "unauthenticated"
        elif status >= httpx.codes.INTERNAL_SERVER_ERROR:
            outcome = "server_error"
        else:
            outcome = "rejected"
        _record(app, outcome)
        logger.warning(
            "w2v_bundle_token outcome=%s tenant=%s app=%s status=%s error=%s",
            outcome,
            tenant,
            app,
            status,
            code or "-",
        )
        if status in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            # Our SA, not the tenant: drop it so the next attempt re-mints it.
            self._sa = None
        if outcome == "server_error":
            return BundleTokenUnavailableError(
                tenant, app, code or "server_error", status_code=status, outcome=outcome
            )
        err = BundleTokenRefusedError(
            tenant, app, code or outcome, status_code=status, outcome=outcome
        )
        if outcome in {"not_found", "forbidden", "conflict"}:
            self._store(
                (tenant, app),
                _Entry(licence=None, valid_until=self._clock() + NEGATIVE_TTL_S, error=err),
            )
        return err

    async def _mint(self, tenant: str, app: str) -> MintedLicence:
        params = {"app": app, "ttl_minutes": str(self._ttl_minutes)}
        if self._bundle:
            params["bundle"] = self._bundle
        async with self._client() as client:
            sa = await self._sa_token(client, tenant, app)
            try:
                resp = await client.post(
                    f"{self._licenses_url}/internal/tenants/{tenant}/bundle-token",
                    params=params,
                    headers={"Authorization": f"Bearer {sa}"},
                )
            except httpx.HTTPError as exc:
                _record(app, "transport_error")
                logger.warning(
                    "w2v_bundle_token outcome=transport_error tenant=%s app=%s error=%s",
                    tenant,
                    app,
                    type(exc).__name__,
                )
                raise BundleTokenUnavailableError(
                    tenant, app, "transport_error", outcome="transport_error"
                ) from exc
        if resp.status_code not in (httpx.codes.CREATED, httpx.codes.OK):
            raise self._refuse(tenant, app, resp)
        body: dict = {}
        try:
            body = resp.json()
            token = body["token"]
            expires_at = _parse_expires_at(body["expires_at"])
            body_tenant = str(UUID(str(body.get("tenant_id") or tenant)))
            body_app = str(body.get("app") or app)
        except ValueError, KeyError, TypeError, AttributeError:
            token, expires_at, body_tenant, body_app = None, 0.0, tenant, app
        if not isinstance(token, str) or not token or body_tenant != tenant or body_app != app:
            _record(app, "bad_response")
            logger.error(
                "w2v_bundle_token outcome=bad_response tenant=%s app=%s status=%s",
                tenant,
                app,
                resp.status_code,
            )
            raise BundleTokenUnavailableError(
                tenant,
                app,
                "malformed_response",
                status_code=resp.status_code,
                outcome="bad_response",
            )
        _record(app, "minted")
        license_id = body.get("license_id")
        token_id = body.get("token_id")
        minted = MintedLicence(
            token=token,
            license_id=str(license_id) if license_id else None,
            token_id=str(token_id) if token_id else None,
        )
        valid_until = expires_at - REFRESH_LEAD_S
        if valid_until > self._clock():
            self._store((tenant, app), _Entry(licence=minted, valid_until=valid_until))
        else:
            # licences caps expiry at the licence's own: still valid, not cacheable.
            self._entries.pop((tenant, app), None)
        logger.info(
            "w2v_bundle_token outcome=minted tenant=%s app=%s license_id=%s token_id=%s",
            tenant,
            app,
            minted.license_id,
            minted.token_id,
        )
        return minted


# -- process-wide minter -------------------------------------------------------

_state: dict[str, BundleTokenMinter | None] = {"minter": None}


def _build_minter_from_env() -> BundleTokenMinter | None:
    client_id = os.environ.get("BUNDLE_MINTER_CLIENT_ID", "").strip()
    client_secret = os.environ.get("BUNDLE_MINTER_CLIENT_SECRET", "").strip()
    token_url = os.environ.get("BUNDLE_MINTER_TOKEN_URL", "").strip()
    if not (client_id and client_secret and token_url):
        return None
    return BundleTokenMinter(
        # Same licences base URL LicensesReadClient uses (W2V_LICENSES_URL overrides).
        licenses_url=service_url("licenses"),
        token_url=token_url,
        client_id=client_id,
        client_secret=client_secret,
        bundle=os.environ.get("BUNDLE_MINTER_BUNDLE", "").strip() or None,
    )


def get_bundle_token_minter(app: str = WORKERS_APP_KEY) -> BundleTokenMinter:
    """The process-wide minter. Raises :class:`BundleMinterNotConfiguredError`.

    Needs ``BUNDLE_MINTER_CLIENT_ID`` + ``BUNDLE_MINTER_CLIENT_SECRET`` +
    ``BUNDLE_MINTER_TOKEN_URL``. There is no fallback to ``KEYCLOAK_URL`` /
    ``KEYCLOAK_CLIENT_ID``: that is the ``w2v-backend`` client in another realm,
    which cannot (and must not) hold ``bundle-token-minter``.
    """
    minter = _state["minter"]
    if minter is None:
        minter = _build_minter_from_env()
        if minter is None:
            _record(app, "not_configured")
            logger.error(
                "w2v_bundle_token outcome=not_configured -- WM_AUTH_MODE=tenant_licence needs "
                "BUNDLE_MINTER_CLIENT_ID / BUNDLE_MINTER_CLIENT_SECRET / BUNDLE_MINTER_TOKEN_URL"
            )
            raise BundleMinterNotConfiguredError(
                "-", app, "not_configured", outcome="not_configured"
            )
        _state["minter"] = minter
    return minter


def set_bundle_token_minter(minter: BundleTokenMinter | None) -> None:
    """Install a minter (tests / a service wiring its own); ``None`` resets to env."""
    _state["minter"] = minter


def reset_bundle_token_minter() -> None:
    """Forget the process-wide minter so the next call rebuilds it from env."""
    _state["minter"] = None


__all__ = [
    "BUNDLE_TOKEN_MINT_TOTAL",
    "MINT_OUTCOMES",
    "NEGATIVE_TTL_S",
    "REFRESH_LEAD_S",
    "TTL_MINUTES",
    "WORKERS_APP_KEY",
    "BundleMinterNotConfiguredError",
    "BundleTokenError",
    "BundleTokenMinter",
    "BundleTokenRefusedError",
    "BundleTokenUnavailableError",
    "MintedLicence",
    "WMAuthMode",
    "bound_tenant_licence",
    "get_bundle_token_minter",
    "is_licence_rejection",
    "licence_headers",
    "reset_bundle_token_minter",
    "resolve_wm_auth_mode",
    "set_bundle_token_minter",
]


async def bound_tenant_licence(app: str = WORKERS_APP_KEY) -> tuple[str, MintedLicence]:
    """``(end tenant, minted licence)`` for the tenant bound to this context.

    The tenant comes from :func:`~w2v_common.auth.identity_context.resolve_licence_tenant`
    (vouched provenance required; no tenant raises ``NoTenantForLicenceError``),
    so no licence is ever minted for an asserted tenant or for nobody.
    """
    from .identity_context import resolve_licence_tenant

    tenant = resolve_licence_tenant()
    return tenant, await get_bundle_token_minter(app).licence(tenant, app)


def licence_headers(licence: MintedLicence) -> dict[str, str]:
    """Wire headers for a hop authenticated by ``licence``.

    ``X-License-Id`` is the MINTED licence's id, so attribution names the licence
    that actually authenticated the call (never the inbound caller's licence);
    absent when licences did not return one.
    """
    headers = {"X-Tenant-Id": f"Bearer {licence.token}"}
    if licence.license_id:
        headers["X-License-Id"] = licence.license_id
    return headers


def is_licence_rejection(status_code: int) -> bool:
    """WM refused the presented licence: invalidate the cache and retry ONCE."""
    return status_code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN)
