"""Client-disconnect detection that survives ``BaseHTTPMiddleware``.

``Request.is_disconnected()`` is a non-blocking poll of the ``receive`` the
endpoint was handed. Behind any Starlette ``BaseHTTPMiddleware`` (w2v_common's
``AuthMiddleware``, ``RequestIdMiddleware``, ... and most services' own) that
``receive`` is the middleware's wrapper, which runs the poll inside its own task
group; the cancelled poll never surfaces the server's ``http.disconnect``.
Measured on starlette 1.7.0: the gate's disconnect skip fires on a bare app and
NEVER behind one such layer.

THE way to wire it: ``install_disconnect_probe(app)``. It wraps the app's BUILT
middleware stack (``build_middleware_stack``), so the probe is the outermost
layer by construction — whatever ``add_middleware`` calls come before or after
it. The probe wraps the server's ``receive`` without stealing from the app:

* every message passes through to the app unchanged, and a ``http.disconnect``
  seen on the way is recorded;
* :func:`client_disconnected` polls the server's ``receive`` directly — stepping
  it only up to its first suspension, never cancelling anything — never while
  the app is itself awaiting it, and any non-disconnect message the poll
  obtains is buffered and handed to the app's next ``receive()`` — so a body
  chunk is never lost, whenever the poll runs;
* once a disconnect is known, the app's ``receive()`` returns ``http.disconnect``.

Stepping is only safe on the SERVER's ``receive`` (or FastAPI telemetry's thin
pass-through wrapper, which sits outside the built stack). A ``BaseHTTPMiddleware``
outside the probe hands it ``receive_or_disconnect``, which opens an anyio
TaskGroup; stepping and closing that threw ``GeneratorExit`` into the open
TaskGroup ("coroutine ignored GeneratorExit" + a corrupted cancel scope → 500).
So a probe created by a plain ``app.add_middleware(DisconnectProbeMiddleware)``
polls only on POSITIVE evidence that nothing outside it rewrites ``receive``:
``scope["app"].user_middleware`` (Starlette sets ``scope["app"]`` before the
stack runs; index 0 is the outermost layer) must list no ``BaseHTTPMiddleware``
subclass before the probe. Pure-ASGI layers outside it — e.g. the
prometheus-fastapi-instrumentator middleware that ``Instrumentator().instrument(app)``
registers last — pass ``receive`` through and are fine. Anything else (a
``BaseHTTPMiddleware`` outside, no ``scope["app"]``, the probe not in the list)
means its ``poll()`` returns False and logs one WARNING. A pure-ASGI layer
outside the probe that itself wraps ``receive`` in a task group would defeat
this check; none exists in the fleet, and ``install_disconnect_probe`` (which
needs no check) is the way to wire new services.

The admission gate calls :func:`client_disconnected` for the ``request=`` it is
given, so routes keep passing the Starlette ``Request``. With no probe at all
the gate falls back to ``request.is_disconnected()``.
"""

from __future__ import annotations

import inspect
import logging
import weakref
from collections import deque
from collections.abc import Awaitable, Coroutine, Mapping
from typing import Protocol, cast

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

PROBE_SCOPE_KEY = "w2v_common.disconnect_probe"

_installed: weakref.WeakSet[Starlette] = weakref.WeakSet()
_warned_not_outermost = False

_DISCONNECT: Message = {"type": "http.disconnect"}


class SupportsDisconnect(Protocol):
    """The slice of ``starlette.requests.Request`` the gate needs."""

    async def is_disconnected(self) -> bool: ...


class DisconnectProbe:
    """Per-request view of the server's ``receive`` channel.

    ``direct=False`` marks a probe that cannot know it sits on the server's
    ``receive``: it passes messages through and records a disconnect it sees,
    but :meth:`poll` never steps the (possibly foreign) receive coroutine.
    """

    def __init__(self, receive: Receive, *, direct: bool = True) -> None:
        self._raw = receive
        self._direct = direct
        self._buffer: deque[Message] = deque()
        self._pending: Awaitable[Message] | None = None
        self._app_receiving = False
        self.disconnected = False

    async def receive(self) -> Message:
        """The ``receive`` handed to the app: buffered first, then the server's."""
        if self._buffer:
            return self._buffer.popleft()
        if self.disconnected:
            return dict(_DISCONNECT)
        pending, self._pending = self._pending, None
        self._app_receiving = True
        try:
            message = await (pending if pending is not None else self._raw())
        finally:
            self._app_receiving = False
        if message.get("type") == "http.disconnect":
            self.disconnected = True
        return message

    async def poll(self) -> bool:
        """True once the client is known to be gone. Never blocks, never steals."""
        if self.disconnected:
            return True
        if not self._direct:
            _warn_not_outermost()
            return False
        if self._app_receiving or self._pending is not None:
            # The app is awaiting the channel (or a receive is parked for it); a
            # second concurrent receive would race it. Whatever arrives reaches
            # the app and is recorded there.
            return False
        message = self._receive_if_ready()
        if message is None:
            return False
        if message.get("type") == "http.disconnect":
            self.disconnected = True
            return True
        self._buffer.append(message)  # not ours: the app gets it on its next receive()
        return False

    def _receive_if_ready(self) -> Message | None:
        """One server ``receive()``, only if it completes without suspending.

        The coroutine is stepped by hand up to its first suspension and closed
        there. No task cancellation is involved (Starlette's own
        ``is_disconnected()`` uses an already-cancelled anyio ``CancelScope``,
        which can swallow a real cancel of the calling task that lands at the
        same await), so a request task cancelled during a poll stays cancelled.
        """
        awaitable = self._raw()
        if not inspect.iscoroutine(awaitable):
            # Not steppable (e.g. a Future): park it for the app's next
            # receive() so its message can never be dropped.
            self._pending = awaitable
            return None
        coro = cast("Coroutine[object, None, Message]", awaitable)
        try:
            coro.send(None)
        except StopIteration as done:
            return cast("Message", done.value)
        coro.close()  # nothing was taken: the server still holds the message
        return None


def _warn_not_outermost() -> None:
    global _warned_not_outermost  # a once-per-process latch
    if _warned_not_outermost:
        return
    _warned_not_outermost = True
    logger.warning(
        "DisconnectProbeMiddleware added with app.add_middleware() has a "
        "BaseHTTPMiddleware outside it (or its position cannot be verified), so "
        "the GPU gate's client-disconnect skip is OFF for this app. Use "
        "w2v_common.admission.install_disconnect_probe(app)."
    )


def _only_pure_asgi_outside(app: object) -> bool:
    """True when ``app.user_middleware`` lists the probe with no
    ``BaseHTTPMiddleware`` subclass before it (index 0 = outermost)."""
    entries = getattr(app, "user_middleware", None)
    if not isinstance(entries, list):
        return False
    for entry in entries:
        cls = getattr(entry, "cls", None)
        if not isinstance(cls, type):
            return False  # a factory function: cannot vouch for it
        if issubclass(cls, DisconnectProbeMiddleware):
            return True
        if issubclass(cls, BaseHTTPMiddleware):
            return False
    return False  # the probe is not in this app's list: no evidence


class DisconnectProbeMiddleware:
    """Pure-ASGI layer exposing a :class:`DisconnectProbe` in the scope.

    Prefer :func:`install_disconnect_probe`. Added by ``app.add_middleware`` the
    probe polls only when ``scope["app"].user_middleware`` proves no
    ``BaseHTTPMiddleware`` sits outside it (see the module docstring); otherwise
    it never polls and warns once. If a probe already exists in the scope (an
    outer or installed one), this instance passes through. Non-HTTP scopes pass
    through untouched.
    """

    def __init__(self, app: ASGIApp, *, _outermost: bool = False) -> None:
        self.app = app
        self._outermost = _outermost
        self._verdict: bool | None = None

    def _direct(self, scope: Scope) -> bool:
        if self._outermost:
            return True
        if self._verdict is None:  # the stack is fixed once built: decide once
            self._verdict = _only_pure_asgi_outside(scope.get("app"))
        return self._verdict

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or isinstance(scope.get(PROBE_SCOPE_KEY), DisconnectProbe):
            await self.app(scope, receive, send)
            return
        probe = DisconnectProbe(receive, direct=self._direct(scope))
        scope[PROBE_SCOPE_KEY] = probe
        await self.app(scope, probe.receive, send)


def install_disconnect_probe(app: Starlette) -> None:
    """Make the disconnect probe the OUTERMOST layer of ``app``, by construction.

    Wraps ``app.build_middleware_stack`` so the probe encloses the whole built
    stack (``ServerErrorMiddleware``, every ``add_middleware`` layer, routing),
    including middleware added AFTER this call. Idempotent. Must run before the
    app serves its first request (Starlette builds the stack once, lazily).
    """
    if app in _installed:
        return
    if app.middleware_stack is not None:
        raise RuntimeError(
            "install_disconnect_probe() must run before the app starts serving: "
            "its middleware stack is already built"
        )
    build = app.build_middleware_stack

    def build_with_probe() -> ASGIApp:
        return DisconnectProbeMiddleware(build(), _outermost=True)

    # Shadow the bound method on THIS instance only; Starlette's __call__ looks
    # it up via ``self.build_middleware_stack()`` when it first builds the stack.
    vars(app)["build_middleware_stack"] = build_with_probe
    _installed.add(app)


async def client_disconnected(request: SupportsDisconnect) -> bool:
    """Whether the client behind ``request`` has gone.

    Uses the :class:`DisconnectProbe` that :func:`install_disconnect_probe` put
    in the request's scope; without one, ``request.is_disconnected()``.
    """
    scope: object = getattr(request, "scope", None)
    probe = scope.get(PROBE_SCOPE_KEY) if isinstance(scope, Mapping) else None
    if isinstance(probe, DisconnectProbe):
        return await probe.poll()
    return await request.is_disconnected()
