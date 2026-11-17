"""Canonical FastAPI app factory for w2v services.

`create_app()` builds a FastAPI instance pre-wired with the fleet-canonical
middleware stack (request-id, security headers, optional auth) plus the
canonical error envelope, then mounts the caller's routers.

Service consumers shrink from ~80 lines of hand-rolled `FastAPI(...)` +
middleware registration + exception handlers to one call::

    from w2v_common.asgi import create_app

    settings = MyServiceSettings()  # composes APISettings as `.api`
    app = create_app(settings, routers=[my_api_router])

``settings`` is any ``pydantic_settings.BaseSettings`` — the factory reads
its optional ``.api`` / ``.auth`` / ``.metrics`` / ``.idempotency`` /
``.audit`` sub-blocks by name, so a composite service Settings (the fleet
norm) and a bare ``APISettings`` both work.

Everything below is a knob — the goal is for 90% of services to call
`create_app(settings, routers)` and nothing else.

Beyond the middleware stack the factory also wires the two remaining pieces
every service ``asgi.py`` used to hand-roll identically: **CORS** (from
``settings.api.cors_*``) and the canonical **BearerAuth/TenantIdHeader
OpenAPI** customization (see ``openapi.install_license_token_openapi``). Both
are on by default and can be disabled per call.

For the surface area `create_app` does NOT cover, the factory accepts escape
hatches so services can extend without forking it:

- ``on_startup`` / ``on_shutdown``: async lifespan hooks called by a
  FastAPI lifespan context manager wired in by the factory.
- ``error_handlers``: a mapping of exception class (or status code int)
  to async handler; merged AFTER the canonical envelope handlers so a
  service can shadow them.
- ``extra_middleware``: ordered sequence of ``(middleware_cls, options)``
  tuples appended AFTER the canonical stack; first in the tuple is the
  outermost on the stack (per Starlette's add_middleware ordering).
- ``openapi_tags`` / ``redirect_slashes`` / ``swagger_ui_parameters`` /
  ``generate_unique_id_function``: forwarded to ``FastAPI(...)`` so a
  service never has to construct the app itself.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute, APIRouter
from pydantic_settings import BaseSettings

from w2v_common.asgi.error_envelope import register_error_envelope_handlers
from w2v_common.asgi.middleware import ASGIMiddleware, register_middleware
from w2v_common.asgi.openapi import install_license_token_openapi
from w2v_common.auth.middleware import PUBLIC_ENDPOINTS, AuthMiddleware
from w2v_common.middleware import (
    AuditMiddleware,
    IdempotencyMiddleware,
    MetricsMiddleware,
    RequestIdMiddleware,
    SecurityHeadersMiddleware,
)
from w2v_common.observability.metrics import metrics_router

logger = logging.getLogger(__name__)

LifespanHook = Callable[[FastAPI], Awaitable[None]]
ExceptionHandler = Callable[[Any, Any], Awaitable[Any]]


def _default_operation_id(route: APIRoute) -> str:
    """Fleet-canonical ``operation_id`` fallback: ``<tag>_<name>``.

    Every route is expected to set ``operation_id=`` explicitly, but a route
    that forgets the kwarg degrades to a stable ``<tag>_<func>`` instead of
    FastAPI's default ``<func>_<path>_<method>`` (which produces garbage
    method names in generated clients).
    """
    tag = route.tags[0] if route.tags else "default"
    return f"{tag}_{route.name}"


def _make_lifespan(
    on_startup: Sequence[LifespanHook],
    on_shutdown: Sequence[LifespanHook],
) -> Callable[[FastAPI], Any]:
    """Build the FastAPI lifespan context manager from startup/shutdown hooks.

    Shutdown hooks are best-effort: a raising hook is logged and teardown
    continues, so one failing resource cannot leak the others.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202  # why: FastAPI lifespan-context typing
        for hook in on_startup:
            await hook(app)
        try:
            yield
        finally:
            for hook in on_shutdown:
                try:
                    await hook(app)
                except Exception:
                    logger.exception(
                        "create_app on_shutdown hook %r raised; continuing teardown",
                        hook,
                    )

    return lifespan


def create_app(
    settings: BaseSettings,
    routers: Sequence[APIRouter] = (),
    *,
    title: str | None = None,
    description: str | None = None,
    version: str = "0.0.0",
    on_startup: Sequence[LifespanHook] = (),
    on_shutdown: Sequence[LifespanHook] = (),
    error_handlers: Mapping[type[Exception] | int, ExceptionHandler] | None = None,
    extra_middleware: Sequence[tuple[type[ASGIMiddleware], Mapping[str, Any]]] = (),
    auth_enabled: bool | None = None,
    auth_public_endpoints: Collection[str] = (),
    cors_enabled: bool | None = None,
    license_openapi: bool = True,
    openapi_tags: list[dict[str, Any]] | None = None,
    redirect_slashes: bool = True,
    swagger_ui_parameters: dict[str, Any] | None = None,
    generate_unique_id_function: Callable[[APIRoute], str] | None = None,
) -> FastAPI:
    """Build a canonical FastAPI app from ``settings`` + ``routers``.

    Args:
        settings: Any ``BaseSettings`` (typically a composite service
            Settings that composes ``APISettings`` as ``.api``) — the
            source of CORS/auth/metrics configuration, read by name.
        routers: Ordered sequence of routers to include on the app.
        title: Optional OpenAPI title override. Defaults to the service's
            package name from settings if available.
        description: Optional OpenAPI description.
        version: Service version string for OpenAPI metadata.
        on_startup: Async callables run after the app starts and before
            it serves traffic; receive the FastAPI app as their argument.
        on_shutdown: Async callables run during graceful shutdown after
            FastAPI has stopped accepting new requests.
        error_handlers: Optional map of exception class (or HTTP status
            code) to an async handler. Merged on top of the canonical
            error-envelope handlers so a service can shadow them.
        extra_middleware: Service-specific middleware appended AFTER the
            canonical stack. First in the sequence is the outermost on
            the stack (Starlette ``add_middleware`` semantics).
        auth_enabled: Override whether ``AuthMiddleware`` is registered.
            When ``None``, falls back to ``settings.auth.enabled`` if
            present, else ``True``. Pass ``False`` to disable auth (only
            sensible for purely internal launchd workers).
        auth_public_endpoints: Service-specific paths that must bypass
            license-token enforcement in addition to the fleet-wide
            :data:`~w2v_common.auth.middleware.PUBLIC_ENDPOINTS` — e.g. an
            OAuth ``/api/auth/callback`` redirect target that carries no
            license token. Threaded to both ``AuthMiddleware`` (enforcement)
            and ``install_license_token_openapi`` (the advertised schema) so
            "advertised == enforced" still holds. Replaces the per-service
            ``AuthMiddleware`` subclass that hand-maintained an extra public
            path.
        cors_enabled: Override whether ``CORSMiddleware`` is registered from
            ``settings.api.cors_*``. When ``None``, registers CORS whenever
            the settings expose an ``api`` block (i.e. always, for real
            services). Pass ``False`` to skip CORS.
        license_openapi: When ``True`` (default), install the canonical
            BearerAuth/TenantIdHeader OpenAPI customization
            (``install_license_token_openapi``). Pass ``False`` for a
            service that advertises no license-token transport.
        openapi_tags: Forwarded to ``FastAPI(openapi_tags=...)``.
        redirect_slashes: Forwarded to ``FastAPI(redirect_slashes=...)``.
            Services that pin ``operation_id`` per route set this ``False``.
        swagger_ui_parameters: Forwarded to
            ``FastAPI(swagger_ui_parameters=...)``.
        generate_unique_id_function: Forwarded to
            ``FastAPI(generate_unique_id_function=...)``. Defaults to the
            fleet-canonical ``<tag>_<name>`` fallback when ``None``.

    Returns:
        A configured ``FastAPI`` instance ready to be served by uvicorn.
    """

    app = FastAPI(
        title=title or _default_title(settings),
        description=description or _default_description(settings),
        version=version,
        lifespan=_make_lifespan(on_startup, on_shutdown),
        redirect_slashes=redirect_slashes,
        openapi_tags=openapi_tags,
        swagger_ui_parameters=swagger_ui_parameters,
        generate_unique_id_function=generate_unique_id_function or _default_operation_id,
    )

    _register_canonical_middleware(
        app,
        settings=settings,
        auth_enabled=auth_enabled,
        auth_public_endpoints=auth_public_endpoints,
        cors_enabled=cors_enabled,
    )

    for middleware_cls, options in extra_middleware:
        register_middleware(app, middleware_cls, **dict(options))

    # debug toggles whether 422/500 envelopes carry a `details` string. It may
    # live on the composite Settings (`.debug`) or its api block (`.api.debug`).
    debug = bool(
        getattr(getattr(settings, "api", None), "debug", False) or getattr(settings, "debug", False)
    )
    register_error_envelope_handlers(app, debug=debug)
    if error_handlers:
        for exc_or_code, handler in error_handlers.items():
            app.add_exception_handler(exc_or_code, handler)  # type: ignore[arg-type]  # why: FastAPI accepts both `type[Exception]` and `int` (status code); the stub widens to a single signature that loses that union narrowing

    for router in routers:
        app.include_router(router)

    metrics_settings = getattr(settings, "metrics", None)
    if metrics_settings is not None and getattr(metrics_settings, "enabled", False):
        app.include_router(metrics_router)

    if license_openapi:
        install_license_token_openapi(
            app,
            public_endpoints=frozenset(PUBLIC_ENDPOINTS) | frozenset(auth_public_endpoints),
        )

    return app


def _register_canonical_middleware(
    app: FastAPI,
    *,
    settings: BaseSettings,
    auth_enabled: bool | None,
    auth_public_endpoints: Collection[str],
    cors_enabled: bool | None,
) -> None:
    """Register the fleet-canonical middleware in the canonical order.

    Starlette executes middleware in reverse-registration order (last
    registered = first to see the request). To get an *outward* order of

        CORS → SecurityHeaders → RequestId → Audit → Auth → Idempotency → Metrics

    we register them in reverse: Metrics first (innermost, so it times the
    handler closest to the route), then Idempotency (runs after auth so the
    caller identity is known when keying cache lookups), Auth, Audit (outer
    than auth so 401/403 rejections are still logged), RequestId (outer than
    Audit so ``request.state.request_id`` + the ``X-Request-Id`` response
    header are set on EVERY request, including an Auth 401/403 that never
    reaches the inner app), SecurityHeaders, and CORS last (outermost) so a
    preflight ``OPTIONS`` is answered by CORS before ``AuthMiddleware`` can
    401 it.

    Metrics, Idempotency, and Audit are opt-in: registered only when the
    service composes the matching Settings block with ``enabled=True``.
    Idempotency additionally needs a connected cache on ``app.state.redis``
    (it falls open otherwise). CORS is registered from ``settings.api.cors_*``
    unless ``cors_enabled=False``.
    """
    metrics_settings = getattr(settings, "metrics", None)
    if metrics_settings is not None and getattr(metrics_settings, "enabled", False):
        register_middleware(app, MetricsMiddleware)

    idem_settings = getattr(settings, "idempotency", None)
    if idem_settings is not None and getattr(idem_settings, "enabled", False):
        register_middleware(
            app,
            IdempotencyMiddleware,
            ttl_seconds=getattr(idem_settings, "ttl_seconds", 86400),
        )

    if auth_enabled is None:
        auth_settings = getattr(settings, "auth", None)
        auth_enabled = bool(getattr(auth_settings, "enabled", True))
    if auth_enabled:
        register_middleware(
            app,
            AuthMiddleware,
            extra_public_endpoints=frozenset(auth_public_endpoints),
        )

    audit_settings = getattr(settings, "audit", None)
    if audit_settings is not None and getattr(audit_settings, "enabled", False):
        register_middleware(
            app,
            AuditMiddleware,
            logger_name=getattr(audit_settings, "logger_name", "w2v.audit"),
        )

    # RequestId is registered AFTER Auth + Audit so that at request time
    # (Starlette runs last-registered first) it sits OUTER of both. This is the
    # correctness fix that makes request.state.request_id + the X-Request-Id
    # response header present on EVERY request — including a 401/403 that
    # AuthMiddleware short-circuits without ever calling the inner app. Audit
    # stays outer of Auth (so it still sees those rejections) but inner of
    # RequestId (so the id is already bound when it records).
    register_middleware(app, RequestIdMiddleware)

    register_middleware(app, SecurityHeadersMiddleware)

    # CORS config lives at ``settings.api.cors_*`` on a composite service
    # Settings, or directly on ``settings`` when a bare ``APISettings`` is
    # passed (as in unit tests). Support both.
    api_settings = getattr(settings, "api", None)
    if api_settings is None and hasattr(settings, "cors_origins"):
        api_settings = settings
    if cors_enabled is None:
        cors_enabled = api_settings is not None and hasattr(api_settings, "cors_origins")
    if cors_enabled and api_settings is not None:
        register_middleware(
            app,
            CORSMiddleware,
            allow_origins=list(getattr(api_settings, "cors_origins", []) or []),
            allow_credentials=bool(getattr(api_settings, "cors_allow_credentials", True)),
            allow_methods=list(getattr(api_settings, "cors_allow_methods", ["*"]) or ["*"]),
            allow_headers=list(getattr(api_settings, "cors_allow_headers", ["*"]) or ["*"]),
        )


def _default_title(settings: BaseSettings) -> str:
    # Settings classes typically live at ``<service>/config.py`` and the
    # service name maps cleanly to the OpenAPI title.
    cls_name = settings.__class__.__name__
    if cls_name.endswith("Settings"):
        cls_name = cls_name[: -len("Settings")]
    return cls_name or "w2v service"


def _default_description(settings: BaseSettings) -> str:
    return f"{_default_title(settings)} — w2v service"
