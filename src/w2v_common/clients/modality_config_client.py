"""Client for worker_management's per-tenant BYO modality-config resolve seam.

The non-LLM modality backends (tts_backend, asr_backend, image_gen_backend,
mlx_backend) call ``GET /tenant-config/modality/{modality}/resolve`` on WM to
learn which provider + key to use for the current request's tenant. WM is the
sole holder of ``WM_SECRET_KEY`` and decrypts the key server-side, so the
backend never holds the master key.

Identity flows exactly like :class:`~w2v_common.clients.wm_client.WMClient`:
``identity_event_hook`` stamps the request-scoped ``X-Tenant-Id`` /
``X-License-Id`` / ``X-Subject-Id`` (from the inbound ContextVars) onto every
outbound call, so WM resolves the CUSTOMER tenant — a single shared client is
safe across concurrent requests from different tenants.

Fail-open policy: any transport/5xx error resolves to the platform default
(``ModalityRoute(configured=False, byo=False)``) so a WM blip never breaks
synthesis. The ONE hard stop is HTTP 409 (``byo_required`` tenant with no
configured provider) → :class:`ModalityBYORequiredError`, which the backend
should surface to the caller rather than fall back to owner hardware.

⚠️ The fail-open default is NOT unconditionally harmless, and the returned
route now says so. ``platform_default()`` means "use the platform provider",
which is the correct, harmless answer for a tenant whose posture genuinely IS
"unconfigured" — but it is the WRONG answer for a ``byo_required`` tenant,
whose posture is "refuse to serve until the tenant supplies its own key". The
409 branch is that policy gate, and a transport error or 5xx bypasses it: for
the duration of a WM roll (istio 502) or an 8s timeout, every resolve returns
the platform default and the caller spends PLATFORM credentials for a tenant
contractually required to bring its own. The route shape could not previously
express this — ``byo=False`` read as an affirmative measurement rather than a
measurement GAP.

:attr:`ModalityRoute.resolution_failed` closes that gap. It is True only on
the fail-open paths (transport error, 5xx, unparseable body) and False on
every successfully-resolved route, so a caller that cares can distinguish
"resolved: not configured" from "COULD NOT RESOLVE". It is declared
``compare=False`` so equality is unchanged (see the field's own note), and the
RETURNED route is otherwise byte-identical to before — no consumer changes
behaviour on upgrade. Acting on the flag (e.g. failing closed for BYO tenants)
is a deliberate, separately-decided policy change, not a consequence of this
field existing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from w2v_common.auth.bundle_tokens import (
    WORKERS_APP_KEY,
    BundleTokenError,
    MintedLicence,
    WMAuthMode,
    bound_tenant_licence,
    get_bundle_token_minter,
    is_licence_rejection,
    licence_headers,
    resolve_wm_auth_mode,
)
from w2v_common.auth.identity_context import identity_event_hook, tenant_licence_event_hook
from w2v_common.discovery import service_url
from w2v_common.observability.metrics import record_modality_resolve_failopen

logger = logging.getLogger(__name__)

VALID_MODALITIES: frozenset[str] = frozenset({"tts", "asr", "image", "embedding", "vlm"})


class ModalityBYORequiredError(Exception):
    """A byo_required tenant has no configured provider for the modality."""


@dataclass
class ModalityRoute:
    """Resolved provider for (tenant, modality). ``api_key`` is decrypted or None.

    ``configured``/``byo`` False means "use the platform self-hosted default":
    the backend keeps its current local model + emits its usual (non-byo) cost
    event. ``byo`` True means build the external adapter for ``provider`` with
    ``api_key``/``base_url`` and emit a byo cost event under ``cost_provider``.

    ``owner_provided`` True (managed BYO) means the external key was supplied by
    the platform owner, not the tenant: the backend still builds the external
    adapter, but the cost event must be emitted as a normal marked-up line
    (``usage_extra.owner_provided="true"``, WITHOUT the ``byo`` flag) so the
    tenant is billed markup on the real provider cost rather than a byo fee.
    Only meaningful when ``byo`` is True.

    ``resolution_failed`` True means "this route is a MEASUREMENT GAP, not an
    answer": ``resolve()`` could not reach WM (transport error), got a 5xx, or
    got a body it could not parse, and fell open to the platform default. Every
    other field on such a route is the fixed fail-open default, NOT the tenant's
    real posture. Callers that must not spend platform credentials for a
    ``byo_required`` tenant can branch on it; callers that keep today's
    fail-open behaviour simply ignore it.

    It is declared ``compare=False`` deliberately. Two routes that differ only
    in ``resolution_failed`` still compare equal, so an existing consumer (or
    test) doing ``route == ModalityRoute.platform_default()`` keeps its current
    answer — this field ADDS a way to observe the failure without CHANGING any
    equality any caller already relies on. ``repr`` is left on so the flag is
    visible in logs and pytest diffs. It is appended LAST with a default, so
    positional construction of the existing fields is unaffected.
    """

    configured: bool
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    cost_provider: str | None = None
    cost_source: str = "worker-management"
    byo: bool = False
    owner_provided: bool = False
    role: str | None = None
    resolution_failed: bool = field(default=False, compare=False)

    @classmethod
    def platform_default(cls, *, resolution_failed: bool = False) -> ModalityRoute:
        """The fail-open route: "use the platform self-hosted default".

        ``resolution_failed`` defaults to False so the existing zero-argument
        call sites are unchanged. ``resolve()`` passes True on its three
        fail-open paths to mark the route as a measurement gap.
        """
        return cls(configured=False, byo=False, resolution_failed=resolution_failed)


class ModalityConfigClient:
    """Async client for WM's ``/tenant-config/modality/{modality}/resolve``.

    Construct once per service and reuse — the ``httpx.AsyncClient`` is pooled
    and the identity hook reads ContextVars per request.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 8.0,
        license_token: str | None = None,
        *,
        auth_mode: str | WMAuthMode | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # ``WM_AUTH_MODE`` — same switch and semantics as ``WMClient``.
        self.auth_mode = resolve_wm_auth_mode(auth_mode)
        self._transport = transport
        resolved = (
            base_url or os.environ.get("WORKER_MANAGEMENT_URL") or service_url("worker_management")
        )
        self.base_url = resolved.rstrip("/")
        self.timeout = timeout
        self.license_token = license_token or os.environ.get("DOWNSTREAM_LICENSE_TOKEN", "")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {}
            hook = identity_event_hook
            if self.auth_mode is WMAuthMode.TENANT_LICENCE:
                # The END tenant's licence is set per request in resolve().
                hook = tenant_licence_event_hook
            elif self.license_token:
                headers["X-Tenant-Id"] = f"Bearer {self.license_token}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
                event_hooks={"request": [hook]},
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def _licence(self) -> tuple[str, MintedLicence] | None:
        """``(end tenant, workers licence)`` in tenant_licence mode, else ``None``."""
        if self.auth_mode is not WMAuthMode.TENANT_LICENCE:
            return None
        return await bound_tenant_licence(WORKERS_APP_KEY)

    async def _fetch(
        self, path: str, params: dict[str, str] | None, licence: tuple[str, MintedLicence] | None
    ) -> httpx.Response:
        """GET ``path``; with a tenant licence, re-mint and retry ONCE on 401/403."""
        client = await self._get_client()
        if licence is None:
            return await client.get(path, params=params)
        tenant, minted = licence
        resp = await client.get(path, params=params, headers=licence_headers(minted))
        if not is_licence_rejection(resp.status_code):
            return resp
        logger.warning(
            "WM rejected the workers licence for tenant %s (HTTP %d); re-minting once",
            tenant,
            resp.status_code,
        )
        minter = get_bundle_token_minter(WORKERS_APP_KEY)
        minter.invalidate(tenant, WORKERS_APP_KEY)
        minted = await minter.licence(tenant, WORKERS_APP_KEY)
        return await client.get(path, params=params, headers=licence_headers(minted))

    async def resolve(
        self, modality: str, *, tier: str | None = None, product: str | None = None
    ) -> ModalityRoute:
        """Resolve the effective provider for the current tenant + ``modality``.

        ``product`` (optional): when given, BYO posture is resolved PER-PRODUCT
        (a licenses ``tenant_app_byo_mode`` override wins over the tenant-wide
        posture). Omit for tenant-wide posture (backward compatible).

        Returns :meth:`ModalityRoute.platform_default` on any transport/5xx
        failure (fail-open), with ``resolution_failed=True`` so the caller can
        tell that route apart from a genuinely-unconfigured one, and increments
        ``w2v_modality_resolve_failopen_total{modality,reason}``. Raises
        :class:`ModalityBYORequiredError` on 409.

        ⚠️ The fail-open route is logged at ERROR, not WARNING. It is not a
        routine degradation: for a ``byo_required`` tenant it is a window in
        which the caller spends PLATFORM credentials against the tenant's
        contract, and it must be greppable and alertable rather than buried at
        WARNING with the retries.

        ⚠️ Both guards below are ``except Exception``, deliberately, not a
        narrower ``httpx.HTTPError``/specific-exception tuple. Same class as
        ``cost_bus._redis_endpoint`` (w2v_common 0.67.9 -> 0.67.10 #141) and
        ``metrics.record_downstream_call`` (0.67.11 -> 0.67.12 #145): a curated
        exception tuple encodes only the failures its author thought of, and
        this function's whole contract is "fail open to the platform default
        on ANY failure to get a usable route" — not just the ones that are an
        ``httpx.HTTPError``. Concretely, ``resp.json()`` raises
        ``json.JSONDecodeError`` (a ``ValueError`` subclass, not an
        ``httpx.HTTPError``) on a truncated body or an HTML error page from a
        proxy in front of WM, and that escaped uncaught before this fix — the
        caller got an exception instead of the documented default. The width
        is justified by what this function returns on the fail-open path:
        ``ModalityRoute.platform_default()``, a fixed, harmless value — there
        is nothing this can safely do with a request/response it cannot use
        except fall back to it. A ``BaseException`` (``KeyboardInterrupt``/
        ``SystemExit``) still escapes both guards, correctly: interpreter
        intent must propagate, and swallowing it here would be the real
        anti-pattern.
        """
        if modality not in VALID_MODALITIES:
            raise ValueError(f"unknown modality {modality!r} (expected {sorted(VALID_MODALITIES)})")

        params: dict[str, str] = {}
        if tier:
            params["tier"] = tier
        if product:
            params["product"] = product
        path = f"/tenant-config/modality/{modality}/resolve"
        # tenant_licence: tenant resolution and the mint happen OUTSIDE the
        # fail-open guard. A missing tenant or licence is not a WM blip; falling
        # open there would serve the platform default to a tenant we could not
        # even identify. NoTenantForLicenceError / BundleTokenError propagate.
        licence = await self._licence()
        try:
            resp = await self._fetch(path, params or None, licence)
        except BundleTokenError:
            raise  # a licence failure on the re-mint fails loud, never open
        except Exception as exc:  # noqa: BLE001 — see the contract above; fail open on ANY failure
            logger.error(
                "modality_config resolve transport error for %s; using platform default "
                "(resolution_failed=True — a byo_required tenant may be billed to the "
                "platform for the duration of this window): %s",
                modality,
                exc,
            )
            record_modality_resolve_failopen(modality, "transport_error")
            return ModalityRoute.platform_default(resolution_failed=True)

        if resp.status_code == 409:
            raise ModalityBYORequiredError(
                f"tenant must configure its own {modality} provider before use"
            )
        if resp.status_code >= 500:
            logger.error(
                "modality_config resolve %s returned %s; using platform default "
                "(resolution_failed=True — a byo_required tenant may be billed to the "
                "platform for the duration of this window)",
                modality,
                resp.status_code,
            )
            record_modality_resolve_failopen(modality, "server_error")
            return ModalityRoute.platform_default(resolution_failed=True)
        if resp.status_code != 200:
            # 4xx other than 409 (e.g. 401/404) is a caller/config error — do not
            # silently fall open (that would mask a real misconfiguration).
            resp.raise_for_status()

        try:
            data: dict[str, Any] = resp.json()
        except Exception as exc:  # noqa: BLE001 — see the contract above; fail open on ANY failure
            logger.error(
                "modality_config resolve %s returned an unparseable body; using platform "
                "default (resolution_failed=True — a byo_required tenant may be billed to "
                "the platform for the duration of this window): %s",
                modality,
                exc,
            )
            record_modality_resolve_failopen(modality, "unparseable_body")
            return ModalityRoute.platform_default(resolution_failed=True)
        return ModalityRoute(
            configured=bool(data.get("configured")),
            provider=data.get("provider"),
            model=data.get("model"),
            base_url=data.get("base_url"),
            api_key=data.get("api_key"),
            cost_provider=data.get("cost_provider"),
            cost_source=data.get("cost_source") or "worker-management",
            byo=bool(data.get("byo")),
            owner_provided=bool(data.get("owner_provided")),
            role=data.get("role"),
        )
