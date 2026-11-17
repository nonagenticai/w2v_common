"""The disconnect probe must be outermost by construction, and harmless when not.

Found adopting 0.99.0 (mlx_backend): with ``app.add_middleware(DisconnectProbeMiddleware)``
followed by another ``add_middleware`` of a ``BaseHTTPMiddleware``, the probe sat
INSIDE that layer. Its "server" receive was then the layer's
``receive_or_disconnect`` — an anyio TaskGroup — and stepping + closing it threw
``GeneratorExit`` into the open TaskGroup: "coroutine ignored GeneratorExit",
"Attempted to exit a cancel scope that isn't the current task's current cancel
scope", and a 500 for a CONNECTED client.

``install_disconnect_probe(app)`` wraps the built stack, so later
``add_middleware`` calls land inside it. A probe added the old way never steps a
receive it cannot vouch for: it polls False and warns once.

``install_disconnect_probe`` is looked up at call time (``admission.<name>``),
so on a wheel without it these tests FAIL rather than the module not importing.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import pytest
from fastapi import APIRouter, FastAPI

import w2v_common.admission.disconnect as disconnect_mod
from tests.test_admission_disconnect_through_middleware import (
    _abandon_while_queued,
    _bare,
    _canonical,
    _drive,
    _new_gate,
    _PassThrough,
    _router,
    _status,
)
from w2v_common import admission
from w2v_common.admission import DisconnectProbeMiddleware

_WARNING_FRAGMENT = "install_disconnect_probe(app)"


@pytest.fixture(autouse=True)
def _reset_warn_latch(monkeypatch: pytest.MonkeyPatch) -> None:
    if hasattr(disconnect_mod, "_warned_not_outermost"):
        monkeypatch.setattr(disconnect_mod, "_warned_not_outermost", False)


def _misordered(app: FastAPI) -> FastAPI:
    """The footgun: probe added by hand, then a BaseHTTPMiddleware added AFTER it
    (so it runs OUTSIDE the probe)."""
    app.add_middleware(DisconnectProbeMiddleware)
    app.add_middleware(_PassThrough)
    return app


def _probe_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and _WARNING_FRAGMENT in r.getMessage()
    ]


@pytest.mark.parametrize("base", [_bare, _canonical], ids=["bare", "create_app"])
async def test_a_probe_inside_a_base_http_middleware_does_not_break_requests(
    base: Callable[[APIRouter], FastAPI], caplog: pytest.LogCaptureFixture
) -> None:
    gate, lane = _new_gate()
    ran: list[str] = []
    app = _misordered(base(_router(gate, lane, ran)))

    caplog.set_level(logging.WARNING, logger=disconnect_mod.__name__)
    for text in ("first", "second"):
        task, messages = await _drive(app, f'{{"text": "{text}"}}'.encode(), asyncio.Event())
        async with asyncio.timeout(5):
            await task  # 0.99.0: ExceptionGroup(GeneratorExit / cancel-scope RuntimeError)
        assert _status(messages) == 200, messages

    assert ran == ["first", "second"]
    # The skip is off for this app, and it says so — once, not per request.
    assert len(_probe_warnings(caplog)) == 1
    assert await gate.shutdown()


async def test_a_misordered_probe_does_not_crash_on_a_disconnect_either(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gate, lane = _new_gate()
    ran: list[str] = []
    app = _misordered(_canonical(_router(gate, lane, ran)))
    caplog.set_level(logging.WARNING, logger=disconnect_mod.__name__)

    messages = await _abandon_while_queued(app, gate, lane)

    # Not skipped (it cannot see the disconnect safely), but no crash, and loud.
    assert _status(messages) == 200
    assert len(_probe_warnings(caplog)) == 1
    assert await gate.shutdown()


@pytest.mark.parametrize("base", [_bare, _canonical], ids=["bare", "create_app"])
async def test_installed_probe_stays_outermost_after_later_add_middleware(
    base: Callable[[APIRouter], FastAPI], caplog: pytest.LogCaptureFixture
) -> None:
    gate, lane = _new_gate()
    ran: list[str] = []
    app = base(_router(gate, lane, ran))
    admission.install_disconnect_probe(app)
    app.add_middleware(_PassThrough)  # added AFTER install: still inside the probe
    app.add_middleware(_PassThrough)
    caplog.set_level(logging.WARNING, logger=disconnect_mod.__name__)

    messages = await _abandon_while_queued(app, gate, lane)

    assert ran == [], "the abandoned job ran on the GPU"
    assert _status(messages) == 499
    assert _probe_warnings(caplog) == []
    assert await gate.shutdown()


async def test_installed_probe_plus_a_hand_added_one_still_skips(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A service mid-migration has both: the inner hand-added layer must defer to
    the installed outermost probe (pass through, no second probe, no warning)."""
    gate, lane = _new_gate()
    ran: list[str] = []
    app = _canonical(_router(gate, lane, ran))
    app.add_middleware(DisconnectProbeMiddleware)
    admission.install_disconnect_probe(app)
    app.add_middleware(_PassThrough)
    caplog.set_level(logging.WARNING, logger=disconnect_mod.__name__)

    messages = await _abandon_while_queued(app, gate, lane)

    assert ran == []
    assert _status(messages) == 499
    assert _probe_warnings(caplog) == []
    assert await gate.shutdown()


async def test_install_is_idempotent_and_wraps_the_whole_built_stack() -> None:
    gate, lane = _new_gate()
    app = _canonical(_router(gate, lane, []))
    admission.install_disconnect_probe(app)
    admission.install_disconnect_probe(app)
    app.add_middleware(_PassThrough)

    stack = app.build_middleware_stack()
    assert isinstance(stack, DisconnectProbeMiddleware)  # outermost layer
    layers = 0
    node: object = stack
    while node is not None:
        layers += isinstance(node, DisconnectProbeMiddleware)
        node = getattr(node, "app", None)
    assert layers == 1
    assert await gate.shutdown()


async def test_install_after_the_app_started_is_refused() -> None:
    gate, lane = _new_gate()
    app = _bare(_router(gate, lane, []))
    task, _ = await _drive(app, b'{"text": "boot"}', asyncio.Event())
    async with asyncio.timeout(5):
        await task  # builds the middleware stack
    with pytest.raises(RuntimeError, match="before the app starts serving"):
        admission.install_disconnect_probe(app)
    assert await gate.shutdown()
