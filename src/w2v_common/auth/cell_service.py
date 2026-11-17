"""Receiver opt-in for service-scope (tenant-less) cell tokens (P7, 0.92.0).

A ``scope=service`` cell token speaks for the calling SERVICE and for no
tenant (a tts-gateway worker polling ``POST /api/tts/claim``). Every receiver
refuses one by default (``w2v_cell_jwt_rejected_total{reason="service_scope_not_allowed"}``).
A route accepts it only by declaring, in code, which issuers may call it::

    router = APIRouter(
        prefix="/api/tts",
        dependencies=[Depends(require_cell_service_caller({"tts-gateway-worker"}))],
    )

The opt-in lives on the route because a service caller is only legitimate on
the handful of endpoints that are inherently global (a cross-tenant claim
queue). ``AuthMiddleware`` runs BEFORE routing, so it finds the matched route
itself (:func:`route_cell_service_callers`) and hands the declared issuers to
``verify_cell_token`` — the token is accepted or refused (and its jti burned)
in one place, and the dependency re-checks at handler time.

The accepted identity has NO tenant: ``request.state.tenant_id`` is ``None``,
the tenant ContextVar is unbound, provenance is ``None`` (not vouched), and no
project/run/actor/subject is bound — so any tenant-scoped code path (RLS, a
licence gate, a metered delegation) fails loud instead of acting for a
sentinel. ``request.state.cell_scope == "service"`` marks it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Final

from fastapi import HTTPException, Request
from starlette.routing import BaseRoute, Host, Match, Mount
from starlette.types import Scope

from .cell_jwt import ALLOW_ANY_CELL_CALLER, CellClaims, CellScope

logger = logging.getLogger(__name__)

#: Attribute that marks a dependency callable as a service-caller opt-in.
_SERVICE_CALLERS_ATTR: Final[str] = "__w2v_cell_service_callers__"
#: ``request.state`` flag set ONLY by the middleware's service LANE (the token
#: authenticated the request). In ``dual`` + a legacy credential the token is
#: merely RECORDED (``cell_claims``) while the legacy lane authenticates — that
#: must admit nothing, exactly as in ``off`` (the 0.91.2 licence-gate rule).
SERVICE_LANE_STATE_ATTR: Final[str] = "cell_service_lane"
#: Mounted sub-apps are walked at most this deep (a route tree is shallow;
#: the bound only stops a pathological self-mount from recursing forever).
_MAX_MOUNT_DEPTH: Final[int] = 8


def require_cell_service_caller(
    allowed: Iterable[str], *, allow_other_lanes: bool = False
) -> Callable[[Request], CellClaims | None]:
    """FastAPI dependency: admit service-scope cell tokens from ``allowed`` issuers.

    Attaching it to a route (or router) is the ONLY way a service token is
    accepted there. ``allowed`` is an explicit set of issuers (cell identities,
    e.g. ``"tts-gateway-worker"``); empty, blank or ``*`` raise ``ValueError``
    at import time — a wildcard service caller would be a global identity.
    The receiver's ``CELL_ALLOWED_CALLERS`` still applies on top.

    Returns the verified :class:`~w2v_common.auth.cell_jwt.CellClaims` of the
    service caller — only when the service LANE authenticated the request (a
    token merely recorded beside a legacy credential in ``dual`` admits nothing). A request that did not arrive as an admitted service
    caller is refused with 403 unless ``allow_other_lanes`` (the migration
    shape: the same route also serves the legacy licence-lane worker, whose
    own checks must then run as before), in which case it returns ``None``.
    """
    callers = frozenset(c.strip() for c in allowed)
    if not callers or "" in callers or ALLOW_ANY_CELL_CALLER in callers:
        raise ValueError(
            "require_cell_service_caller needs an explicit, non-empty issuer list (no '*')"
        )

    def _dependency(request: Request) -> CellClaims | None:
        claims = getattr(request.state, "cell_claims", None)
        if (
            getattr(request.state, SERVICE_LANE_STATE_ATTR, False) is True
            and isinstance(claims, CellClaims)
            and claims.scope is CellScope.SERVICE
            and claims.issuer in callers
        ):
            return claims
        if allow_other_lanes:
            return None
        raise HTTPException(status_code=403, detail="caller is not an admitted cell service")

    setattr(_dependency, _SERVICE_CALLERS_ATTR, callers)
    return _dependency


def _dependant_callers(dependant: object) -> frozenset[str] | None:
    """Union of every service-caller opt-in in a FastAPI dependant tree, or ``None``."""
    found: set[str] = set()
    opted = False
    stack: list[object] = [dependant]
    while stack:
        node = stack.pop()
        callers = getattr(getattr(node, "call", None), _SERVICE_CALLERS_ATTR, None)
        if isinstance(callers, frozenset):
            opted = True
            found.update(callers)
        stack.extend(getattr(node, "dependencies", ()) or ())
    return frozenset(found) if opted else None


def _included_router_type() -> type | None:
    """FastAPI >= 0.141 wraps ``include_router`` in ``_IncludedRouter``; older copies routes."""
    import fastapi.routing

    found = getattr(fastapi.routing, "_IncludedRouter", None)
    return found if isinstance(found, type) else None


#: Resolved once at import: the FastAPI in this venv either has the wrapper or not.
_INCLUDED_ROUTER: Final[type | None] = _included_router_type()


def _resolve_included(route: BaseRoute, scope: Scope, depth: int) -> frozenset[str] | None:
    """Resolve the route an ``_IncludedRouter`` (FastAPI >= 0.141) will dispatch to.

    Its public ``matches()`` says only THAT the inclusion matched (child scope
    ``{}``), not WHICH route, and the matched ``APIRoute`` is the router's
    ORIGINAL route — whose ``dependant`` lacks every dependency added at
    inclusion time (``include_router(..., dependencies=...)`` or an enclosing
    router's). FastAPI's own selection (``_match``) is the only thing that
    returns the effective route context with the composed dependant, so we ask
    it rather than re-deriving prefixes and dependency chains ourselves. A
    shape we do not recognise fails closed (``None``) — the version-matrix CI
    job is what turns that into a red build instead of rejected workers.
    """
    select = getattr(route, "_match", None)
    if not callable(select):
        return None
    result = select(scope)
    if not (isinstance(result, tuple) and len(result) == 4):
        return None
    match, child, inner, context = result
    if match is not Match.FULL or inner is None:
        return None
    scope = {**scope, **child}
    if context is not None:
        dependant = getattr(context, "dependant", None)
        if dependant is not None:
            return _dependant_callers(dependant)
        # Non-API route (Mount / websocket / Host): FastAPI built a prefixed
        # Starlette route for it; that copy is what actually serves.
        inner = getattr(context, "starlette_route", None) or inner
    return _resolve_matched(inner, scope, depth + 1)


def _resolve_matched(route: BaseRoute, scope: Scope, depth: int) -> frozenset[str] | None:
    """Opt-in of the route Starlette/FastAPI selected for ``scope`` (already FULL-matched)."""
    if depth >= _MAX_MOUNT_DEPTH:
        return None
    if _INCLUDED_ROUTER is not None and isinstance(route, _INCLUDED_ROUTER):
        return _resolve_included(route, scope, depth)
    dependant = getattr(route, "dependant", None)
    if dependant is not None:
        return _dependant_callers(dependant)
    if isinstance(route, (Mount, Host)):
        return _match_routes(route.routes, scope, depth + 1)
    return None


def _match_routes(routes: Sequence[BaseRoute], scope: Scope, depth: int) -> frozenset[str] | None:
    """Mirror Starlette's router: the first FULL match decides; anything else = no opt-in."""
    for route in routes:
        match, child = route.matches(_probe_scope(scope))
        if match is not Match.FULL:
            # PARTIAL is a method mismatch (Starlette answers 405): no opt-in.
            continue
        return _resolve_matched(route, {**scope, **child}, depth)
    return None


def _probe_scope(scope: Scope) -> Scope:
    """A copy FastAPI may scribble its routing bookkeeping into.

    ``_IncludedRouter.matches`` records itself under ``scope["fastapi"]``
    (``setdefault``); probing on the live request scope would leave that dict
    behind for the real dispatch.
    """
    fastapi_state = scope.get("fastapi")
    return {**scope, "fastapi": dict(fastapi_state) if isinstance(fastapi_state, dict) else {}}


def route_cell_service_callers(scope: Scope) -> frozenset[str] | None:
    """Service issuers the route this request will hit opted in, else ``None``.

    Resolved from the ASGI app's own routing table (``scope["app"]``), so the
    route's dependency is the single source of truth. Fails closed: no app,
    no matching route, a method mismatch or an unwalkable mount → ``None``.
    """
    app = scope.get("app")
    router = getattr(app, "router", None)
    routes = getattr(router, "routes", None)
    if not routes:
        return None
    try:
        return _match_routes(routes, _probe_scope(scope), 0)
    except Exception:  # could not check = no opt-in, never a 500
        # The lookup runs for EVERY cell token (tenant scope included) and leans
        # on FastAPI's private ``_IncludedRouter._match``: a FastAPI that changes
        # it must refuse service tokens, not crash the tenant traffic beside them.
        logger.exception("cell service route lookup failed; service token refused")
        return None
