"""HTTP client for artifact_service — reusable across w2v services.

artifact_service (port 8556 in `artifacts` ns) is the multi-tenant artifact
registry. It proxies file bytes through storage_service (port 8391 in
`storage` ns), which holds the only MinIO credentials. Auth is by license
token (Authorization: Bearer …) plus optional X-Tenant-Id Bearer header.

This client is a thin async httpx wrapper. The license token is read from
``DOWNSTREAM_LICENSE_TOKEN`` env at construction time and sent as
``Authorization: Bearer <token>`` on every request. Identity headers (the
W2V S2S triple ``X-Tenant-Id`` / ``X-License-Id`` / ``X-Subject-Id``) are
layered in per-call by the request hook reading ContextVars — same pattern
as :class:`w2v_common.clients.wm_client.WMClient`.

Usage::

    artifact = ArtifactClient()  # reads DOWNSTREAM_LICENSE_TOKEN from env
    result = await artifact.upload_bytes(
        project_id=pipeline_job_id,
        filename="part-001.demo.spec.ts",
        content=script_bytes,
        content_type="text/plain",
        kind="other",
        step="part-001",
        tags=["w2v-script"],
    )
    print(result.artifact_id)

    out = await artifact.download_to_path(
        project_id=pipeline_job_id,
        artifact_id=result.artifact_id,
        dest=Path("/app/shared-scripts/<job>/parts/part-001.demo.spec.ts"),
    )

Auth mode (``ARTIFACTS_AUTH_MODE``, see :mod:`w2v_common.auth.tenant_licence_lane`):
``shared`` (default) is the behaviour above, unchanged. ``tenant_licence``
never sends ``DOWNSTREAM_LICENSE_TOKEN``: a request inside an inbound call
forwards the caller's own licence (as today), and one without it (background
task, CronJob) carries the bound tenant's minted ``artifacts`` licence. No
tenant / unvouched tenant / refused mint RAISES -- there is no shared fallback.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx

from w2v_common.auth.bundle_tokens import is_licence_rejection, licence_headers
from w2v_common.auth.identity_context import (
    caller_token_event_hook,
    get_caller_license_token,
    get_license_id_from_context,
    identity_event_hook,
    tenant_licence_event_hook,
)
from w2v_common.auth.tenant_licence_lane import HopAuthMode, TenantLicenceLane
from w2v_common.discovery import service_url

logger = logging.getLogger(__name__)


@dataclass
class ArtifactUploadResult:
    """Parsed UploadResponse from artifact_service."""

    artifact_id: UUID
    version: int
    storage_key: str
    size: int
    sha256: str


class ArtifactClient:
    """Async client for artifact_service.

    Constructor is cheap; the underlying httpx.AsyncClient is created lazily
    on first request and reused (connection-pooled). Call ``aclose()`` from
    your service's lifespan shutdown.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 120.0,
        license_token: str | None = None,
        *,
        auth_mode: str | HopAuthMode | None = None,
        licence_app: str | None = None,
    ) -> None:
        # Resolution order: explicit kwarg > ARTIFACT_SERVICE_URL env >
        # canonical in-cluster DNS via discovery.service_url(). discovery is
        # the single source of truth for in-cluster URL construction (see
        # w2v_common.discovery + gitops-production/docs/config-externalization-policy.md).
        resolved = base_url or os.environ.get("ARTIFACT_SERVICE_URL") or service_url("artifacts")
        self.base_url = resolved.rstrip("/")
        self.timeout = timeout
        self.license_token = license_token or os.environ.get("DOWNSTREAM_LICENSE_TOKEN", "")
        # Resolved at construction: an unknown ARTIFACTS_AUTH_MODE, or
        # tenant_licence without the BUNDLE_MINTER_* config, fails HERE (boot).
        self._lane = TenantLicenceLane(
            "artifacts", app=licence_app, mode=auth_mode, shared_token=self.license_token
        )
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    @property
    def auth_mode(self) -> HopAuthMode:
        return self._lane.mode

    async def _auth_headers(self) -> tuple[dict[str, str], str | None]:
        """Per-request credential headers and the tenant a licence was MINTED for.

        ``shared``: nothing per request (the pooled client carries the shared
        token and ``caller_token_event_hook`` overrides it, as before).
        ``tenant_licence``: the inbound caller's own licence when one is in
        scope (same subject symmetry as the shared lane), else the bound
        tenant's minted ``artifacts`` licence. Never the shared token.
        """
        if not self._lane.tenant_licence:
            return {}, None
        caller = get_caller_license_token()
        if caller:
            bearer = f"Bearer {caller}"
            headers = {"Authorization": bearer, "X-Tenant-Id": bearer}
            license_id = get_license_id_from_context()
            if license_id:
                headers["X-License-Id"] = license_id
            return headers, None
        tenant, minted = await self._lane.licence()
        headers = licence_headers(minted)
        headers["Authorization"] = headers["X-Tenant-Id"]
        return headers, tenant

    async def _with_auth[T](self, send: Callable[[dict[str, str]], Awaitable[T]]) -> T:
        """Run ``send(headers)``; a MINTED licence refused 401/403 is re-minted ONCE."""
        headers, minted_for = await self._auth_headers()
        try:
            return await send(headers)
        except httpx.HTTPStatusError as exc:
            if minted_for is None or not is_licence_rejection(exc.response.status_code):
                raise
            logger.warning(
                "artifact_service rejected the artifacts licence for tenant %s (HTTP %d); "
                "re-minting once",
                minted_for,
                exc.response.status_code,
            )
            self._lane.invalidate(minted_for)
        headers, _ = await self._auth_headers()
        return await send(headers)

    async def _get_client(self) -> httpx.AsyncClient:
        # Fast path: client already initialised and healthy — no lock needed.
        # httpx.AsyncClient is safe to share across coroutines once created.
        if self._client is not None and not self._client.is_closed:
            return self._client
        async with self._client_lock:
            # Re-check after acquiring lock — another coroutine may have built it.
            if self._client is None or self._client.is_closed:
                headers: dict[str, str] = {}
                hooks: list[Callable[[httpx.Request], Awaitable[None]]] = [
                    identity_event_hook,
                    caller_token_event_hook,
                ]
                if self._lane.tenant_licence:
                    # Credentials are set per REQUEST (_auth_headers); nothing
                    # credential-shaped is baked into the pooled client, and
                    # the hook refuses any request without a Bearer licence.
                    hooks = [tenant_licence_event_hook]
                elif self.license_token:
                    # artifact_service's ext_authz pipeline accepts the license token on
                    # Authorization: Bearer for cluster-internal S2S, in addition to the
                    # w2v-canonical X-Tenant-Id Bearer fallback. WMClient sends only
                    # X-Tenant-Id; this client sends both. Don't "normalize" by removing
                    # Authorization without verifying artifact_service's gateway config.
                    headers["Authorization"] = f"Bearer {self.license_token}"
                    headers["X-Tenant-Id"] = f"Bearer {self.license_token}"
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    headers=headers,
                    timeout=self.timeout,
                    # identity_event_hook stamps X-License-Id / X-Subject-Id
                    # (UUID-form, additive). caller_token_event_hook overrides
                    # Authorization + X-Tenant-Id with the inbound caller's
                    # Bearer when one is in scope, so artifact_service stores
                    # / serves under the introspected caller's subject_id.
                    # When no caller token is in scope (CronJob, startup task,
                    # Keycloak JWT path, trusted-network path),
                    # caller_token_event_hook is a no-op and the constructor's
                    # service-account Bearer above survives.
                    event_hooks={"request": hooks},
                )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def upload_bytes(
        self,
        *,
        project_id: UUID,
        filename: str,
        content: bytes,
        content_type: str = "text/plain",
        kind: str = "other",
        card_id: UUID | None = None,
        step: str | None = None,
        container_name: str | None = None,
        sensitivity: str = "internal",
        tags: list[str] | None = None,
    ) -> ArtifactUploadResult:
        """Upload raw bytes as an artifact and return its registry record.

        ``kind`` must be one of artifact_service's ArtifactKind enum:
        ``doc, presentation, image, dataset, json, graph, other``.
        ``content_type`` must be in artifact_service's ALLOWED_CONTENT_TYPES
        whitelist (default whitelist includes ``text/plain``).
        """
        client = await self._get_client()
        form: dict[str, Any] = {
            "filename": filename,
            "content_type": content_type,
            "kind": kind,
            "sensitivity": sensitivity,
        }
        if card_id is not None:
            form["card_id"] = str(card_id)
        if step is not None:
            form["step"] = step
        if container_name is not None:
            form["container_name"] = container_name
        if tags:
            form["tags"] = ",".join(tags)

        async def send(auth: dict[str, str]) -> httpx.Response:
            # Fresh file object per attempt: a retry must not send a drained stream.
            files = {"file": (filename, io.BytesIO(content), content_type)}
            resp = await client.post(
                f"/projects/{project_id}/artifacts/upload",
                files=files,
                data=form,
                headers=auth,
            )
            resp.raise_for_status()
            return resp

        resp = await self._with_auth(send)
        body = resp.json()
        return ArtifactUploadResult(
            artifact_id=UUID(body["artifact_id"]),
            version=int(body["version"]),
            storage_key=str(body["storage_key"]),
            size=int(body["size"]),
            sha256=str(body["sha256"]),
        )

    async def download_to_path(
        self,
        *,
        project_id: UUID,
        artifact_id: UUID,
        dest: Path,
    ) -> Path:
        """Download an artifact's bytes streamingly to *dest*.

        Creates parent directories. Overwrites *dest* if it exists.
        Returns *dest* on success.
        """
        client = await self._get_client()
        dest.parent.mkdir(parents=True, exist_ok=True)

        async def send(auth: dict[str, str]) -> Path:
            async with client.stream(
                "GET", f"/projects/{project_id}/artifacts/{artifact_id}/download", headers=auth
            ) as resp:
                resp.raise_for_status()
                with dest.open("wb") as fh:
                    async for chunk in resp.aiter_bytes(64 * 1024):
                        fh.write(chunk)
            return dest

        return await self._with_auth(send)
