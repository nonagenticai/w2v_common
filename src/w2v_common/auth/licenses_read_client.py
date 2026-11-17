"""READ-ONLY licenses-service client used by the entitlement gate.

Promoted from ``w2v_orchestrator``'s ``src/clients/licenses_client.py``, but
deliberately **only the read path**. The orchestrator's client also mints
licenses, regenerates raw tokens and lists plans; those are ADMIN operations
that belong to exactly one service. Shipping them in the shared library would
hand every one of the 13 w2v services the ability to mint the very entitlement
the gate is supposed to check — so ``create_license`` / ``regenerate_token`` /
``list_apps`` / ``list_plans`` are intentionally NOT here. If a service needs
them, it is the orchestrator and it already has them.

Two routes, both read:

* ``GET /licenses/active?tenant_id=&app_key=`` — 200 with the introspect detail
  (its id field is ``license_id``, **not** ``id``), 404 when there is no active
  license, 409 when several are active (a data conflict the caller reads as
  entitled-with-unknown-expiry).
* ``GET /licenses/user/{subject_id}`` — the subject's license inventory
  (``LicenseResponse`` shape: has ``id`` + ``revoked_at``). ⚠️ It returns the
  subject's licenses across EVERY tenant, so the caller must filter on
  ``tenant_id`` as well as app key — see ``entitlement._in_gate_scope``.

The base URL comes from :func:`w2v_common.discovery.service_url` (``licenses``),
so ``W2V_LICENSES_URL`` overrides it exactly like every other peer — no new
config pattern is invented here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Protocol
from uuid import UUID

import httpx
from pydantic import BaseModel

from ..discovery import service_url
from .identity_context import get_caller_license_token

_TIMEOUT_SECONDS = 15.0
#: Refresh a client-credentials token this many seconds before its stated expiry
#: so an in-flight call never races a same-instant expiry (matches the
#: orchestrator's ``ServiceAuthenticator`` and fr's ``ServiceAuthenticator``).
_SERVICE_TOKEN_REFRESH_BUFFER_SECONDS = 60.0
_SERVICE_TOKEN_TIMEOUT_SECONDS = 10.0

_log = logging.getLogger(__name__)


class LicensesClientError(RuntimeError):
    """A licenses-service call returned an unexpected status.

    Carries the upstream ``status_code`` as a first-class attribute: the gate
    must be able to tell an AUTHZ rejection (401/403 — this service account is
    not permitted on that route, a permanent misconfiguration) apart from a
    transient 5xx. String-matching the message for a code is how that
    distinction silently rots.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LicenseNotActiveError(LicensesClientError):
    """The licenses service's OWN authoritative "no active licence" answer.

    Raised by :meth:`LicensesReadClient.get_active_license` ONLY when the 404
    carries the route's published discriminator,
    ``{"detail": {"error": "no_active_license", ...}}`` (licenses ``main.py``
    ``_resolve_active_entitlement``; its OpenAPI 404 documents it and its
    ``tests/unit/test_endpoints.py`` pins it).

    A bare 404 is NOT this. A wrong base URL, a missing route or a proxy with no
    matching path also answers 404, and treating those as "no licence" would turn
    a misconfiguration into a confident ``license_none`` 402 for every tenant.
    Those keep returning ``None``.

    Subclasses :class:`LicensesClientError` (status 404) so any caller that does
    not know this type fails SOFT on it, never treats it as an entitlement.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=404)


#: The ``detail.error`` value licenses emits on its authoritative no-licence 404.
NO_ACTIVE_LICENSE_ERROR = "no_active_license"


def _is_no_active_license_body(resp: httpx.Response) -> bool:
    """Does this 404 carry licenses' own ``no_active_license`` discriminator?"""
    try:
        body = resp.json()
    except ValueError:
        return False
    if not isinstance(body, dict):
        return False
    detail = body.get("detail")
    return isinstance(detail, dict) and detail.get("error") == NO_ACTIVE_LICENSE_ERROR


class AppInfoView(BaseModel):
    """Nested app on a license record."""

    id: str | None = None
    key: str | None = None
    name: str | None = None


class PlanInfoView(BaseModel):
    """Nested plan on a license record."""

    id: str | None = None
    code: str | None = None
    name: str | None = None


class ActiveLicenseView(BaseModel):
    """``GET /licenses/active`` 200 body (the token-introspect shape).

    ``conflict=True`` is a client-synthesised sentinel for the 409 multi-active
    case; the wire never sends it. Extra fields are tolerated on purpose — the
    licenses service is free to add them and the gate reads by key.
    """

    model_config = {"extra": "allow"}

    license_id: str | None = None
    tenant_id: str | None = None
    subject_id: str | None = None
    expires_at: str | None = None
    app: AppInfoView | None = None
    plan: PlanInfoView | None = None
    conflict: bool = False


class LicenseView(BaseModel):
    """``GET /licenses/user/{id}`` item (the ``LicenseResponse`` shape).

    Revocation is a TIMESTAMP (``revoked_at``), not a bare ``revoked`` bool —
    conflating the two was a live bug in an earlier consumer.
    """

    model_config = {"extra": "allow"}

    id: str | None = None
    tenant_id: str | None = None
    subject_id: str | None = None
    status: str | None = None
    starts_at: str | None = None
    expires_at: str | None = None
    revoked_at: str | None = None
    app: AppInfoView | None = None
    plan: PlanInfoView | None = None


#: Async callable returning the bearer token to present to the licenses service.
type TokenProvider = Callable[[], Awaitable[str | None]]


class ServiceTokenError(RuntimeError):
    """A client-credentials token could not be minted."""


class _ServiceTokenMinter:
    """Caches the ``w2v-backend`` client-credentials access token.

    Identical contract to ``w2v_orchestrator``'s ``ServiceAuthenticator``: mint
    on demand, refresh ``_SERVICE_TOKEN_REFRESH_BUFFER_SECONDS`` before expiry,
    and de-duplicate concurrent refreshes behind an ``asyncio.Lock``. It lives
    here rather than being imported because ``w2v_common`` may not depend on a
    service package — this is the shared-library half of the same pattern.

    Configuration is the env block every w2v pod already speaks: ``KEYCLOAK_URL``
    / ``KEYCLOAK_REALM`` / ``KEYCLOAK_CLIENT_ID`` / ``KEYCLOAK_CLIENT_SECRET``, plus
    ``KEYCLOAK_SERVICE_REALM`` (realm-per-app P7): the realm the token is minted from
    when it differs from ``KEYCLOAK_REALM``. ``KEYCLOAK_REALM`` is ALSO the primary
    issuer realm of the inbound JWT validator (:mod:`w2v_common.auth.keycloak`), so it
    cannot simply be pointed at ``secure-apps`` — the service realm is its own setting.
    Unset = ``KEYCLOAK_REALM``, i.e. unchanged.
    """

    def __init__(
        self,
        *,
        keycloak_url: str,
        realm: str,
        client_id: str,
        client_secret: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._token_url = f"{keycloak_url.rstrip('/')}/realms/{realm}/protocol/openid-connect/token"
        self._client_id = client_id
        self._client_secret = client_secret
        self._client = client
        self._lock = asyncio.Lock()
        self._access_token: str | None = None
        self._expires_at: float = 0.0

    def _fresh(self) -> bool:
        return self._access_token is not None and time.monotonic() < self._expires_at

    async def token(self) -> str:
        if self._fresh() and self._access_token is not None:
            return self._access_token
        async with self._lock:
            if not (self._fresh() and self._access_token is not None):
                await self._refresh()
            if self._access_token is None:
                raise ServiceTokenError("no access token after refresh")
            return self._access_token

    async def _refresh(self) -> None:
        data = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        try:
            if self._client is not None:
                resp = await self._client.post(self._token_url, data=data)
            else:
                async with httpx.AsyncClient(timeout=_SERVICE_TOKEN_TIMEOUT_SECONDS) as client:
                    resp = await client.post(self._token_url, data=data)
        except httpx.HTTPError as exc:
            raise ServiceTokenError(f"token request failed: {exc}") from exc
        if resp.status_code != 200:
            raise ServiceTokenError(f"token endpoint returned {resp.status_code}")
        payload = resp.json()
        token = payload.get("access_token")
        if not token:
            raise ServiceTokenError("token response missing access_token")
        expires_in = float(payload.get("expires_in", 0) or 0)
        self._access_token = token
        self._expires_at = time.monotonic() + max(
            0.0, expires_in - _SERVICE_TOKEN_REFRESH_BUFFER_SECONDS
        )


_service_token_minter: _ServiceTokenMinter | None = None


def service_token_realm() -> str:
    """Realm the ``w2v-backend`` S2S token is minted from.

    ``KEYCLOAK_SERVICE_REALM`` (realm-per-app P7: ``secure-apps``), else
    ``KEYCLOAK_REALM``, else ``portal`` — the latter two exactly as before.
    """
    explicit = (os.environ.get("KEYCLOAK_SERVICE_REALM") or "").strip()
    return explicit or os.environ.get("KEYCLOAK_REALM", "portal")


def _get_service_token_minter() -> _ServiceTokenMinter | None:
    """Process-wide minter, or ``None`` when the SA env block is not configured.

    ``None`` is the pre-existing behaviour (caller token / static token), so a
    pod without the credentials is not made worse by this code path.
    """
    global _service_token_minter
    if _service_token_minter is not None:
        return _service_token_minter
    keycloak_url = os.environ.get("KEYCLOAK_URL")
    client_id = os.environ.get("KEYCLOAK_CLIENT_ID")
    client_secret = os.environ.get("KEYCLOAK_CLIENT_SECRET")
    if not (keycloak_url and client_id and client_secret):
        return None
    _service_token_minter = _ServiceTokenMinter(
        keycloak_url=keycloak_url,
        realm=service_token_realm(),
        client_id=client_id,
        client_secret=client_secret,
    )
    return _service_token_minter


def reset_service_token_minter() -> None:
    """Drop the cached minter — tests only (env changes between cases)."""
    global _service_token_minter
    _service_token_minter = None


async def _default_token() -> str | None:
    """Default credential, in the order that actually authorizes the gate.

    1. A client-credentials JWT for ``KEYCLOAK_CLIENT_ID`` (``w2v-backend`` in
       the fleet, a cell's own SA such as ``w2v-cell-m4``), minted in
       ``service_token_realm()``, when ``KEYCLOAK_URL`` / ``KEYCLOAK_CLIENT_ID``
       / ``KEYCLOAK_CLIENT_SECRET`` are set.
    2. The caller's own opaque licence token from the identity context.
    3. The static ``W2V_LICENSES_SERVICE_TOKEN``.

    WHY THE SA JWT IS FIRST, and why the previous order left the gate inert.
    The gate resolves entitlement for the tenant it is SERVING, which is not
    necessarily the tenant of whoever called. Credentials 2 and 3 are opaque
    licence tokens, and the licenses service pins every introspected licence
    token to exactly ``["license-reader"]`` on its OWN tenant
    (``auth/middleware.py`` ``LICENSE_TOKEN_ROLES``) with no ``azp`` at all. So
    for any tenant other than the token's own, ``GET /licenses/active`` answers
    403 ``"Not authorized to read another tenant's active license"`` and the
    fallback ``GET /licenses/user/{sub}`` answers 403 ``"Not authorized to read
    another subject's licenses"``. The gate then resolves ``unknown`` and returns
    503 ``license_unavailable`` — measured identically for a LICENSED tenant and
    an unlicensed one, which means the 402 ``license_none`` path never ran in
    production. No value of ``W2V_LICENSES_SERVICE_TOKEN`` can fix that: an
    opaque licence token structurally cannot carry a cross-tenant identity.

    A client-credentials JWT can, because it carries a verified ``iss`` realm
    and ``azp``. The token is minted in ``service_token_realm()``
    (``KEYCLOAK_SERVICE_REALM`` — ``secure-apps`` since realm-per-app P7 — else
    ``KEYCLOAK_REALM``). The licenses service admits it on the read routes by
    its roles (``require_internal_or_reader``: ``license-reader`` or an
    internal-service role), and honours the ``X-Tenant-Id`` this client sends
    only for a tenant the client's OWNER app — resolved from licenses'
    ``api_client`` table on the verified (realm, ``azp``), migration 050 —
    owns or holds an active grant on (a ``cross_owner`` client: any, audited).
    A token from an app-owned realm is additionally limited to that app's
    tenants (``realm_owner``, migration 072). There is no env allowlist of
    trusted proxy ``azp``s any more; ownership and grants decide it. Asking
    about a tenant outside that scope still 403s.

    A mint failure is NOT fatal: we fall through to the caller's token, which is
    still correct whenever the served tenant is the caller's own. Returning
    ``None`` sends no Authorization header, which the licenses service answers
    401 to — and a 401 is routed to the inventory fallback / ``unknown``, never
    to an ALLOW.
    """
    minter = _get_service_token_minter()
    if minter is not None:
        try:
            return await minter.token()
        except ServiceTokenError as exc:
            _log.warning(
                "licenses_service_token_mint_failed: falling back to the caller token; "
                "cross-tenant entitlement lookups will 403 until this is fixed (%s)",
                exc,
            )
    return get_caller_license_token() or os.environ.get("W2V_LICENSES_SERVICE_TOKEN") or None


class LicensesReadClient:
    """The read slice of the licenses service, and nothing else."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token_provider: TokenProvider | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = (base_url or service_url("licenses")).rstrip("/")
        self._token_provider = token_provider or _default_token
        self._client = client

    async def _get(self, path: str, *, tenant_id: UUID | str | None) -> httpx.Response:
        headers: dict[str, str] = {}
        token = await self._token_provider()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if tenant_id is not None:
            headers["X-Tenant-Id"] = str(tenant_id)
        url = f"{self._base}{path}"
        if self._client is not None:
            return await self._client.get(url, headers=headers)
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            return await client.get(url, headers=headers)

    async def get_active_license(
        self, *, tenant_id: UUID | str, app_key: str
    ) -> ActiveLicenseView | None:
        """200 → detail, 409 → ``ActiveLicenseView(conflict=True)``.

        404 WITH licenses' ``no_active_license`` discriminator → raises
        :class:`LicenseNotActiveError` (authoritatively not active). Any OTHER 404
        (route miss, wrong base URL, proxy) → ``None``, unchanged. Anything else
        raises :class:`LicensesClientError`."""
        resp = await self._get(
            f"/licenses/active?tenant_id={tenant_id}&app_key={app_key}", tenant_id=tenant_id
        )
        if resp.status_code == 200:
            detail = resp.json()
            return ActiveLicenseView.model_validate(detail) if isinstance(detail, dict) else None
        if resp.status_code == 404:
            if _is_no_active_license_body(resp):
                raise LicenseNotActiveError(
                    f"no active license for tenant={tenant_id} app_key={app_key}"
                )
            return None
        if resp.status_code == 409:
            return ActiveLicenseView(conflict=True)
        raise LicensesClientError(
            f"get_active_license returned {resp.status_code}: {resp.text[:200]}",
            status_code=resp.status_code,
        )

    async def list_user_licenses(
        self, subject_id: UUID | str, *, tenant_id: UUID | str
    ) -> list[LicenseView]:
        """The subject's license inventory. Raises on any non-200."""
        resp = await self._get(f"/licenses/user/{subject_id}", tenant_id=tenant_id)
        if resp.status_code != 200:
            raise LicensesClientError(
                f"list_user_licenses returned {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
            )
        rows = resp.json()
        return [LicenseView.model_validate(r) for r in rows] if isinstance(rows, list) else []


class LicensesReader(Protocol):
    """The read-only slice of the licenses client the entitlement gate consults.

    A Protocol, so a service can inject its own credential-bearing client and a
    test can inject a lightweight fake — while still rejecting anything missing
    either method. The gate depends on THIS, never on the concrete class.
    """

    async def get_active_license(
        self, *, tenant_id: str, app_key: str
    ) -> ActiveLicenseView | None: ...

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str) -> list[LicenseView]: ...


_default_client: LicensesReader | None = None


def get_licenses_read_client() -> LicensesReader:
    """Process-wide read client (lazily built so import never does I/O)."""
    global _default_client
    if _default_client is None:
        _default_client = LicensesReadClient()
    return _default_client


def set_licenses_read_client(client: LicensesReader | None) -> None:
    """Override (or reset, with ``None``) the process-wide client. Test seam and
    the hook a service uses to inject its own credential provider."""
    global _default_client
    _default_client = client
