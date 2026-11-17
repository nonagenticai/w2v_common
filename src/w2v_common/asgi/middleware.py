"""Typed wrapper around Starlette's ``add_middleware``.

Starlette's ``Starlette.add_middleware`` is annotated with PEP-612 ``ParamSpec``,
so its first argument is ``type[_MiddlewareFactory[P]]``. Custom middleware that
takes ``**options`` (the common case for CORS / Auth / RateLimit) trip the
type-checker at every call site, requiring ``# type: ignore[arg-type]``.

This module hides that single, narrowly-scoped suppression behind a
Protocol-typed boundary so service code stays clean.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from starlette.applications import Starlette
from starlette.types import ASGIApp, Receive, Scope, Send


@runtime_checkable
class ASGIMiddleware(Protocol):
    """Anything constructible as ``cls(app, **options)`` returning an ASGI callable."""

    def __init__(self, app: ASGIApp, /, *args: Any, **kwargs: Any) -> None: ...

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...


def register_middleware(
    app: Starlette,
    middleware_cls: type[ASGIMiddleware],
    /,
    **options: Any,
) -> None:
    """Register ``middleware_cls`` on ``app`` with ``**options``.

    Single point of contact with Starlette's stricter ``add_middleware`` typing,
    centralizing what would otherwise be a ``# type: ignore[arg-type]`` at
    every service-level call site.
    """
    # Starlette's add_middleware uses PEP-612 ParamSpec which our Protocol cannot
    # satisfy structurally; the runtime contract (cls(app, **options)) holds.
    app.add_middleware(middleware_cls, **options)  # type: ignore[arg-type]
