"""The per-tenant licence lane for ANY out-of-cell metered hop.

Owner ruling 2026-10-15: there are no app-to-app licences, only tenant-x-app
licences. :mod:`w2v_common.auth.bundle_tokens` mints them; until this module only
worker_management used them (``WM_AUTH_MODE``). This generalises the switch so
every out-of-cell hop can present the CALLING tenant's own licence instead of the
shared ``DOWNSTREAM_LICENSE_TOKEN``::

    lane = TenantLicenceLane("artifacts")    # at client construction (boot)
    headers = await lane.headers()            # per request
    # shared mode:  {"X-Tenant-Id": "Bearer <DOWNSTREAM_LICENSE_TOKEN>"}
    # tenant mode:  {"X-Tenant-Id": "Bearer <tenant's artifacts licence>", "X-License-Id": <its id>}

Switch: env ``<HOP>_AUTH_MODE`` (``ARTIFACTS_AUTH_MODE``,
``MODALITY_GATEWAY_AUTH_MODE``, ``BILLING_AUTH_MODE``, ``WEBSCRAPER_AUTH_MODE``,
``MEDIA_DISCOVERY_AUTH_MODE``), values ``shared``
(default — byte-identical to today) or ``tenant_licence``. One variable per hop
so each hop flips independently, exactly like ``WM_AUTH_MODE`` (which keeps its
own name and its ``delegation`` default; WM is not routed through this module).

Rules:

* Unset / empty = ``shared`` = today's behaviour. An unknown value RAISES
  ``ValueError`` at construction: a typo must not silently keep billing the
  shared SA while the operator believes the flip happened.
* ``tenant_licence`` with the ``BUNDLE_MINTER_*`` settings missing raises
  :class:`BundleMinterNotConfiguredError` at CONSTRUCTION (boot), not on the
  first metered call, and never falls back to the shared token.
* In ``tenant_licence`` mode every failure (no tenant bound, unvouched tenant,
  mint refused e.g. 404 ``no_bundle``, licences down) RAISES. There is no
  shared-token fallback anywhere in this lane.
* ``w2v_bundle_token_mint_total{app,outcome}`` counts every resolution, labelled
  with this hop's app key.
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Any, Final

import httpx

from .bundle_tokens import (
    MintedLicence,
    bound_tenant_licence,
    get_bundle_token_minter,
    is_licence_rejection,
    licence_headers,
)
from .identity_context import tenant_licence_event_hook

#: ``httpx.Request.extensions`` key carrying the tenant a request's licence was
#: minted for, so the response hook can invalidate exactly that cache entry.
_LICENCE_TENANT_EXTENSION: Final = "w2v_licence_tenant"

#: Hop -> the licences app key its tenant licence is minted for.
#: Evidence (2026-10-16, origin/main of each repo):
#:
#: * ``artifacts``: route-registry ``/artifacts/`` app_key ``artifacts``
#:   (``route-registry.yaml:74``; licences ``scripts/seed_apps.py`` catalog).
#:   artifact_service's own middleware checks no app key in-cluster.
#: * ``modality_gateway``: ``w2v`` (owner decision: no separate modality app).
#: * ``billing``: ``w2v``. billing's ``/api/costs/*`` is NOT in license-authz's
#:   ``APP_ROUTE_REGISTRY`` (unregistered path => any active licence passes) and
#:   the licences catalog has no ``billing`` app, so the caller's own product
#:   licence is the only one a tenant can hold for it.
#: * ``webscraper``: ``w2v`` (2026-10-17, KAG removal). webscraper has no app in
#:   route-registry or the licences catalog and accepts any active licence, so
#:   the caller's own product licence is the one to present (same reasoning as
#:   billing). Env ``WEBSCRAPER_AUTH_MODE``.
#: * ``media_discovery``: ``w2v`` (2026-10-17, KAG removal). media-discovery
#:   introspects ``X-Tenant-Id: Bearer`` with no app-key check and has no app in
#:   route-registry or the catalog. Env ``MEDIA_DISCOVERY_AUTH_MODE``.
HOP_APP_KEYS: Final[dict[str, str]] = {
    "artifacts": "artifacts",
    "modality_gateway": "w2v",
    "billing": "w2v",
    "webscraper": "w2v",
    "media_discovery": "w2v",
}


class HopAuthMode(StrEnum):
    """How a w2v service authenticates to one out-of-cell hop (env ``<HOP>_AUTH_MODE``)."""

    SHARED = "shared"
    TENANT_LICENCE = "tenant_licence"


def hop_auth_mode_env(hop: str) -> str:
    """``artifacts`` -> ``ARTIFACTS_AUTH_MODE``; ``modality_gateway`` -> ``MODALITY_GATEWAY_AUTH_MODE``."""
    return f"{hop.strip().upper().replace('-', '_')}_AUTH_MODE"


def resolve_hop_auth_mode(hop: str, value: str | HopAuthMode | None = None) -> HopAuthMode:
    """``value`` if given, else env ``<HOP>_AUTH_MODE``, else ``shared``. Unknown RAISES."""
    env = hop_auth_mode_env(hop)
    raw = value if value is not None else os.environ.get(env, "")
    raw = str(raw).strip().lower() or HopAuthMode.SHARED.value
    try:
        return HopAuthMode(raw)
    except ValueError:
        allowed = ", ".join(m.value for m in HopAuthMode)
        raise ValueError(f"{env}={raw!r} is not one of: {allowed}") from None


class TenantLicenceLane:
    """Auth for one out-of-cell hop: the shared token (default) or the tenant's licence.

    Construct it where the client is constructed (service boot): that is where a
    bad mode or a missing minter config must fail.
    """

    def __init__(
        self,
        hop: str,
        *,
        app: str | None = None,
        mode: str | HopAuthMode | None = None,
        shared_token: str | None = None,
    ) -> None:
        self.hop = hop
        resolved_app = app or HOP_APP_KEYS.get(hop)
        if not resolved_app:
            raise ValueError(
                f"tenant licence lane: hop {hop!r} has no licences app key; pass app=<key>"
            )
        self.app = resolved_app
        self.mode = resolve_hop_auth_mode(hop, mode)
        self.shared_token = (
            shared_token
            if shared_token is not None
            else os.environ.get("DOWNSTREAM_LICENSE_TOKEN", "")
        )
        if self.mode is HopAuthMode.TENANT_LICENCE:
            # Boot-time proof the minter can be built. Raises
            # BundleMinterNotConfiguredError (counted not_configured{app}).
            get_bundle_token_minter(self.app)

    @property
    def tenant_licence(self) -> bool:
        return self.mode is HopAuthMode.TENANT_LICENCE

    async def licence(self) -> tuple[str, MintedLicence]:
        """``(end tenant, minted licence)`` for the bound tenant. tenant_licence mode only.

        Raises ``NoTenantForLicenceError`` / ``UnvouchedTenantError`` before any
        mint, and ``BundleTokenError`` (e.g. 404 ``no_bundle``) on a failed one.
        """
        if not self.tenant_licence:
            raise RuntimeError(f"{hop_auth_mode_env(self.hop)} is shared: no tenant licence")
        return await bound_tenant_licence(self.app)

    async def headers(self, *, authorization: bool = False) -> dict[str, str]:
        """Credential headers for one request on this hop.

        ``shared``: ``X-Tenant-Id: Bearer <DOWNSTREAM_LICENSE_TOKEN>`` (empty dict
        when no token is configured — exactly what today's clients send).
        ``tenant_licence``: the bound tenant's minted licence + its ``X-License-Id``.
        ``authorization=True`` also sets ``Authorization`` to the same Bearer, for
        receivers whose gateway reads the licence there (artifacts, billing).
        """
        if self.tenant_licence:
            _, minted = await self.licence()
            headers = licence_headers(minted)
        elif self.shared_token:
            headers = {"X-Tenant-Id": f"Bearer {self.shared_token}"}
        else:
            return {}
        if authorization:
            headers["Authorization"] = headers["X-Tenant-Id"]
        return headers

    def invalidate(self, tenant_id: str) -> None:
        """Drop the cached licence for ``tenant_id`` (the hop answered 401/403)."""
        if self.tenant_licence:
            get_bundle_token_minter(self.app).invalidate(tenant_id, self.app)

    # -- httpx wiring for a pooled client (tenant_licence mode only) ------------
    #
    # A client in tenant_licence mode bakes NOTHING credential-shaped into its
    # pooled ``httpx.AsyncClient``; it installs ``event_hooks()`` instead:
    #
    # * request: mint (or reuse the cached) licence for the bound tenant, put it
    #   on ``X-Tenant-Id: Bearer`` + ``X-License-Id``, then run
    #   ``tenant_licence_event_hook`` (identity stamping, never
    #   ``X-Effective-Tenant-Id``). No tenant / refused mint RAISES out of the
    #   send: nothing leaves on another credential.
    # * response: a 401/403 drops that tenant's cached licence, so the NEXT call
    #   re-mints. The failed call itself is not replayed here (the client's own
    #   retry policy decides that); WMClient/ArtifactClient keep their explicit
    #   re-mint-once.

    async def request_hook(self, request: httpx.Request) -> None:
        """httpx request hook: the bound tenant's minted licence on this request."""
        tenant, minted = await self.licence()
        for name, value in licence_headers(minted).items():
            request.headers[name] = value
        request.extensions[_LICENCE_TENANT_EXTENSION] = tenant
        await tenant_licence_event_hook(request)

    async def response_hook(self, response: httpx.Response) -> None:
        """httpx response hook: a refused licence is invalidated for the next call."""
        if not is_licence_rejection(response.status_code):
            return
        tenant = response.request.extensions.get(_LICENCE_TENANT_EXTENSION)
        if isinstance(tenant, str) and tenant:
            self.invalidate(tenant)

    def event_hooks(self) -> dict[str, list[Any]]:
        """``event_hooks=`` for a pooled client in tenant_licence mode."""
        if not self.tenant_licence:
            raise RuntimeError(f"{hop_auth_mode_env(self.hop)} is shared: no licence hooks")
        return {"request": [self.request_hook], "response": [self.response_hook]}


__all__ = [
    "HOP_APP_KEYS",
    "HopAuthMode",
    "TenantLicenceLane",
    "hop_auth_mode_env",
    "resolve_hop_auth_mode",
]
