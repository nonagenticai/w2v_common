"""Tests for w2v_common.asgi.register_middleware."""

from __future__ import annotations

from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from w2v_common.asgi import ASGIMiddleware, register_middleware


class HeaderTaggingMiddleware:
    """Trivial pass-through middleware that stamps a header on every response."""

    def __init__(self, app: ASGIApp, /, *, header_name: str, header_value: str) -> None:
        self.app = app
        self.header_name = header_name
        self.header_value = header_value

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_header(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append(
                    (self.header_name.encode("latin-1"), self.header_value.encode("latin-1"))
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_header)


async def _hello(_request: Request) -> Response:
    return PlainTextResponse("hello")


def _build_app(**options: Any) -> Starlette:
    app = Starlette(routes=[Route("/", _hello)])
    register_middleware(app, HeaderTaggingMiddleware, **options)
    return app


def test_register_middleware_runs_on_request() -> None:
    app = _build_app(header_name="X-W2V-Test", header_value="ran")
    with TestClient(app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers.get("X-W2V-Test") == "ran"
    assert resp.text == "hello"


def test_register_middleware_passes_options_through() -> None:
    """Each registration receives its own options independently."""
    app = Starlette(routes=[Route("/", _hello)])
    register_middleware(app, HeaderTaggingMiddleware, header_name="X-One", header_value="1")
    register_middleware(app, HeaderTaggingMiddleware, header_name="X-Two", header_value="2")
    with TestClient(app) as client:
        resp = client.get("/")
    assert resp.headers.get("X-One") == "1"
    assert resp.headers.get("X-Two") == "2"


def test_asgimiddleware_protocol_runtime_checkable() -> None:
    """The Protocol is runtime_checkable so isinstance() works for duck-typed classes."""
    app = Starlette()
    instance = HeaderTaggingMiddleware(app, header_name="X", header_value="y")
    assert isinstance(instance, ASGIMiddleware)
