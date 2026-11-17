"""Differential check: the route lookup must name the route Starlette DISPATCHES.

``AuthMiddleware`` decides a service-scope cell token BEFORE routing, from
``route_cell_service_callers(scope)``. If that lookup resolved route A while
the router then dispatched route B, an opt-in on A would admit a service token
at B — a bypass. Ground truth here is independent of FastAPI internals: every
opt-in dependency records its issuers when it RUNS, so the union of what ran
on the real dispatch is compared with what the lookup predicted (both ``None``
when nothing opted in or nothing was served).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import APIRouter, Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Host, Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from w2v_common.auth.cell_service import _SERVICE_CALLERS_ATTR, route_cell_service_callers

_PREDICTED = "w2v_test_predicted"
_RAN = "w2v_test_ran"


def _opt(*callers: str) -> Any:
    """An opt-in marker (same attribute the real one carries) that records it ran."""

    def _dep(request: Request) -> None:
        request.scope[_RAN].update(callers)
        request.scope[_RAN + "_any"] = True

    setattr(_dep, _SERVICE_CALLERS_ATTR, frozenset(callers))
    return Depends(_dep)


class _Probe:
    """Outermost ASGI layer: predict BEFORE routing, exactly as AuthMiddleware does."""

    def __init__(self, app: ASGIApp, fastapi_app: FastAPI) -> None:
        self.app = app
        self.fastapi_app = fastapi_app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            # FastAPI.__call__ sets scope["app"] before its middleware stack runs
            scope[_PREDICTED] = route_cell_service_callers({**scope, "app": self.fastapi_app})
            scope[_RAN] = set()
            scope[_RAN + "_any"] = False
        await self.app(scope, receive, send)


async def _h() -> dict[str, Any]:
    return {}


def _starlette_ok(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def _tricky_app() -> FastAPI:
    app = FastAPI()

    # a. path-param sibling registered BEFORE a literal opted route
    a = APIRouter(prefix="/a")
    a.add_api_route("/{item}", _h, methods=["POST"])
    a.add_api_route("/claim", _h, methods=["POST"], dependencies=[_opt("A")])
    app.include_router(a)

    # b. included router PARTIAL (GET only) ahead of an app route FULL for POST
    b = APIRouter()
    b.add_api_route("/b/claim", _h, methods=["GET"], dependencies=[_opt("B-get")])
    app.include_router(b)
    app.add_api_route("/b/claim", _h, methods=["POST"], dependencies=[_opt("B-post")])

    # c. app route PARTIAL ahead of an included FULL
    app.add_api_route("/c/claim", _h, methods=["GET"], dependencies=[_opt("C-get")])
    c = APIRouter()
    c.add_api_route("/c/claim", _h, methods=["POST"], dependencies=[_opt("C-post")])
    app.include_router(c)

    # d. the SAME router included twice, opt-in only on the second inclusion
    d = APIRouter()
    d.add_api_route("/claim", _h, methods=["POST"])
    app.include_router(d, prefix="/d1")
    app.include_router(d, prefix="/d2", dependencies=[_opt("D2")])

    # e. the SAME router included twice under the SAME prefix: first wins
    e = APIRouter()
    e.add_api_route("/claim", _h, methods=["POST"])
    app.include_router(e, prefix="/e")
    app.include_router(e, prefix="/e", dependencies=[_opt("E2")])

    # f. a sibling router's opt-in must not leak onto a neighbour
    f1 = APIRouter(prefix="/f", dependencies=[_opt("F1")])
    f1.add_api_route("/one", _h, methods=["POST"])
    f2 = APIRouter(prefix="/f")
    f2.add_api_route("/two", _h, methods=["POST"])
    app.include_router(f1)
    app.include_router(f2)

    # g. route added to the router AFTER it was included
    g = APIRouter(prefix="/g")
    app.include_router(g, dependencies=[_opt("G")])
    g.add_api_route("/late", _h, methods=["POST"])

    # h. a Mount at the top level shadows a later opted route
    app.mount("/h", FastAPI())
    app.add_api_route("/h/claim", _h, methods=["POST"], dependencies=[_opt("H")])

    # i. Host routing
    host_app = FastAPI()
    host_app.add_api_route("/claim", _h, methods=["POST"], dependencies=[_opt("I")])
    app.router.routes.append(Host("i.example", app=host_app))

    # j. a mounted Starlette router whose opted path is a plain Starlette Route
    app.mount("/j", app=Mount("", routes=[Route("/claim", _starlette_ok, methods=["POST"])]))

    # k. nested include where the INNER router is also included elsewhere bare
    k_leaf = APIRouter()
    k_leaf.add_api_route("/claim", _h, methods=["POST"])
    k_mid = APIRouter(prefix="/k1", dependencies=[_opt("K")])
    k_mid.include_router(k_leaf)
    app.include_router(k_mid)
    app.include_router(k_leaf, prefix="/k2")

    # l. trailing slash: route registered with one
    app.add_api_route("/l/claim/", _h, methods=["POST"], dependencies=[_opt("L")])
    return app


CASES = [
    ("POST", "/a/claim", "http://t"),  # served by /{item}: no opt-in
    ("POST", "/a/other", "http://t"),
    ("POST", "/b/claim", "http://t"),
    ("GET", "/b/claim", "http://t"),
    ("POST", "/c/claim", "http://t"),
    ("GET", "/c/claim", "http://t"),
    ("POST", "/d1/claim", "http://t"),
    ("POST", "/d2/claim", "http://t"),
    ("POST", "/e/claim", "http://t"),
    ("POST", "/f/one", "http://t"),
    ("POST", "/f/two", "http://t"),
    ("POST", "/g/late", "http://t"),
    ("POST", "/h/claim", "http://t"),
    ("POST", "/claim", "http://i.example"),
    ("POST", "/claim", "http://other.example"),
    ("POST", "/j/claim", "http://t"),
    ("POST", "/k1/claim", "http://t"),
    ("POST", "/k2/claim", "http://t"),
    ("POST", "/l/claim/", "http://t"),
    ("POST", "/l/claim", "http://t"),  # redirect_slashes: nothing runs
    ("HEAD", "/b/claim", "http://t"),
]


@pytest.mark.parametrize(("method", "path", "base"), CASES)
async def test_lookup_predicts_the_dispatched_route(method: str, path: str, base: str) -> None:
    app = _tricky_app()
    seen: dict[str, Any] = {}

    async def _capture(scope: Scope, receive: Receive, send: Send) -> None:
        await app(scope, receive, send)
        seen.update(
            predicted=scope[_PREDICTED],
            ran=frozenset(scope[_RAN]) if scope[_RAN + "_any"] else None,
        )

    async with AsyncClient(transport=ASGITransport(app=_Probe(_capture, app)), base_url=base) as c:
        resp = await c.request(method, path)
    assert seen, "probe did not run"
    # A LOOSER prediction is a bypass; a stricter one refuses a legitimate
    # worker. On every locked FastAPI line the two are exactly equal.
    assert seen["predicted"] == seen["ran"], (resp.status_code, seen)
