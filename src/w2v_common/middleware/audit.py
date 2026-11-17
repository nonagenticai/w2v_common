"""Audit logging middleware for state-changing requests and auth failures.

Logs every mutation (POST/PUT/DELETE/PATCH) and every auth failure (401/403)
as a structured record with tenant, resource, status, client IP, duration, and
a non-reversible fingerprint of the bearer token. Promoted from the
orchestrator's copy (the superset) — analysis_service's variant lacked the
token fingerprint and the token-leak-safe tenant resolution.

The middleware must run **outer** than ``AuthMiddleware`` so that an auth
rejection (which short-circuits before inner middleware) is still logged; on
the success path it reads the tenant id that auth stamped onto
``request.state``, falling back to the ``x-tenant-id`` header (sanitised of any
``Bearer`` token material) for the auth-failure path where state is unset.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from ..tenant import TenantError, parse_tenant_id

_MUTATION_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})
_AUTH_FAILURE_CODES = frozenset({401, 403})

DEFAULT_AUDIT_LOGGER = "w2v.audit"


def fingerprint_api_key(api_key: str | None) -> str | None:
    """Return a short, non-reversible fingerprint of an API key."""
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def _resource_from_path(path: str) -> str | None:
    """Extract a logical resource name from a URL path."""
    parts = path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "api":
        resource = parts[1]
        if resource.endswith("s"):
            resource = resource[:-1]
        return resource
    return None


class AuditMiddleware(BaseHTTPMiddleware):
    """Log state-changing requests and auth failures for security auditing."""

    def __init__(self, app: ASGIApp, logger_name: str = DEFAULT_AUDIT_LOGGER) -> None:
        super().__init__(app)
        self._logger = logging.getLogger(logger_name)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Log state-changing requests (POST/PUT/DELETE/PATCH) and auth failures (401/403)."""
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000

        should_log = (
            request.method in _MUTATION_METHODS or response.status_code in _AUTH_FAILURE_CODES
        )

        if not should_log:
            return response

        # Prefer the tenant id resolved by the auth middleware (set on
        # request.state). The raw `x-tenant-id` header carries
        # "Bearer <token>" at this point — logging it would (a) leak
        # token material and (b) report wrong tenants for auth-failure
        # rows. Fall back to the header for the auth-failure path, where
        # state was never populated.
        #
        # ⭐ The literal ``"anonymous"`` this used to default to was a SENTINEL
        # in a tenant field — the same class of defect as the nil-UUID: it is a
        # value that reads as a tenant, aggregates as a tenant, and belongs to
        # nobody. Audit rows for 401s legitimately have NO tenant (that is why
        # they are 401s), so the honest representation is ``None``. A consumer
        # can then count "unattributed auth failures" without them silently
        # merging into one fake customer.
        #
        # The header fallback applies ONLY when auth never ran (state attribute
        # absent — the 401/403 short-circuit). Once auth has stamped
        # ``request.state.tenant_id`` — even as ``None``, which is the
        # tenant-less JWT lane (``AUTH_JWT_TENANT_REQUIRED=false``) — the
        # resolved value is final: a bare ``X-Tenant-Id`` is client-chosen and
        # must never be recorded as the authoritative tenant of an
        # authenticated request.
        if hasattr(request.state, "tenant_id"):
            raw_tenant = request.state.tenant_id
        else:
            raw_tenant = request.headers.get("x-tenant-id")
        tenant_id: str | None = None
        if isinstance(raw_tenant, uuid.UUID):
            tenant_id = str(raw_tenant)
        elif isinstance(raw_tenant, str):
            candidate = raw_tenant.strip()
            # A Bearer value is a credential, never a tenant. Dropping it (vs
            # mapping it to a sentinel) keeps the token out of the log AND
            # keeps the row honestly unattributed.
            if candidate and not candidate.lower().startswith("bearer "):
                try:
                    tenant_id = str(parse_tenant_id(candidate, source="audit x-tenant-id"))
                except TenantError:
                    # A non-UUID header value is somebody's hand-crafted slug.
                    # It is not a tenant, so it is not logged as one.
                    tenant_id = None
        # The caller's bearer credential for the audit fingerprint: a license
        # token rides in `X-Tenant-Id: Bearer <token>` (Authorization is the
        # Keycloak-JWT lane). Prefer the license token so license-authed
        # mutations get a non-null fingerprint, falling back to the JWT.
        xt_raw = request.headers.get("x-tenant-id", "")
        if xt_raw.lower().startswith("bearer "):
            api_key = xt_raw[7:].strip() or None
        else:
            api_key = (
                request.headers.get("authorization", "").removeprefix("Bearer ").strip() or None
            )
        resource = _resource_from_path(request.url.path)

        log_data = {
            "tenant_id": tenant_id,
            "method": request.method,
            "path": request.url.path,
            "resource": resource,
            "status_code": response.status_code,
            "ip": request.client.host if request.client else "unknown",
            "duration_ms": round(duration_ms, 1),
            "api_key_fingerprint": fingerprint_api_key(api_key),
        }

        if response.status_code in _AUTH_FAILURE_CODES:
            self._logger.warning("AUTH_FAILURE: %s", log_data)
        else:
            self._logger.info("AUDIT: %s", log_data)

        return response
