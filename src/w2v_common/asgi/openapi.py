"""Canonical OpenAPI customization for w2v services.

Every w2v service advertises the same license-token transport in its
OpenAPI schema: a ``BearerAuth`` security scheme plus a reusable
``TenantIdHeader`` parameter, applied to every non-public operation. This
was hand-cloned as a ~60-line ``custom_openapi()`` in 12 of 13 service
``asgi.py`` files (each carrying the same "Mirrors the kag / worker_management
/ …" comment).

``install_license_token_openapi`` folds that clone into one place. It reuses
``w2v_common.auth.middleware.PUBLIC_ENDPOINTS`` / ``PUBLIC_PREFIXES`` — the
same source of truth the auth middleware enforces against — so the schema's
"which operations are public" decision can never drift from the runtime
policy. ``create_app`` installs it by default.

The actual enforcement lives in ``AuthMiddleware``; this only shapes the
advertised schema so Swagger UI exposes the ``X-Tenant-Id: Bearer <token>``
transport correctly (the license token travels in ``X-Tenant-Id``, never in
``Authorization`` — that header is reserved for the caller's Keycloak JWT).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from w2v_common.auth.middleware import PUBLIC_ENDPOINTS, PUBLIC_PREFIXES

_BEARER_SCHEME: dict[str, str] = {
    "type": "http",
    "scheme": "bearer",
    "bearerFormat": "License",
    "description": (
        "License token. Transport: `X-Tenant-Id: Bearer <token>` "
        "(not the Authorization header — that is reserved for caller-side auth)."
    ),
}

_TENANT_ID_PARAMETER: dict[str, Any] = {
    "name": "X-Tenant-Id",
    "in": "header",
    "required": False,
    "description": "License token transport: `Bearer <token>`.",
    "schema": {"type": "string"},
}


def install_license_token_openapi(
    app: FastAPI,
    *,
    public_endpoints: set[str] | frozenset[str] = frozenset(PUBLIC_ENDPOINTS),
    public_prefixes: tuple[str, ...] = PUBLIC_PREFIXES,
) -> None:
    """Override ``app.openapi`` to advertise the license-token transport.

    Adds a ``BearerAuth`` security scheme + a reusable ``TenantIdHeader``
    parameter, then marks every non-public operation with
    ``security: [{BearerAuth: []}]`` and the header parameter. Public
    operations (health/docs/etc., per the shared auth policy) have their
    ``security`` cleared so Swagger UI does not prompt for a token.

    Idempotent per app: the generated schema is cached on
    ``app.openapi_schema`` exactly as FastAPI's default does.
    """
    public = frozenset(public_endpoints)

    def _is_public(path: str) -> bool:
        clean = path.rstrip("/") or "/"
        return clean in public or clean.startswith(public_prefixes)

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        from fastapi.openapi.utils import get_openapi

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=app.openapi_tags,
        )

        components = schema.setdefault("components", {})
        components.setdefault("securitySchemes", {})["BearerAuth"] = dict(_BEARER_SCHEME)
        components.setdefault("parameters", {})["TenantIdHeader"] = dict(_TENANT_ID_PARAMETER)

        for path, item in schema.get("paths", {}).items():
            is_public = _is_public(path)
            for operation in item.values():
                if not isinstance(operation, dict):
                    continue
                if is_public:
                    operation.pop("security", None)
                else:
                    operation.setdefault("security", [{"BearerAuth": []}])
                params = operation.setdefault("parameters", [])
                if not any(
                    p.get("name", "").lower() == "x-tenant-id" and p.get("in") == "header"
                    for p in params
                ):
                    params.append({"$ref": "#/components/parameters/TenantIdHeader"})

        app.openapi_schema = schema
        return app.openapi_schema

    # FastAPI stores `openapi` as a bound method; overriding the attribute on
    # the instance requires bypassing the (frozen) __setattr__ the same way the
    # per-service copies did.
    object.__setattr__(app, "openapi", custom_openapi)
