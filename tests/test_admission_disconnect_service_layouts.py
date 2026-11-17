"""Hand-added probes in the fleet's REAL layouts keep their disconnect skip.

tts_backend, asr_backend, image_gen_backend and transcription_worker_service add
``DisconnectProbeMiddleware`` by hand (0.99.0 advice) as their LAST
``add_middleware`` call. tts/asr then call ``Instrumentator().instrument(app)``,
which registers the pure-ASGI ``PrometheusInstrumentatorMiddleware`` OUTSIDE the
probe. A first cut of 0.100.0 disabled every hand-added probe (``direct=False``),
so relocking these four would have silently turned their skip off. The probe now
polls when ``user_middleware`` shows no ``BaseHTTPMiddleware`` outside it.

Layouts below are ``add_middleware`` calls in source order, copied from each
repo's ``origin/main`` (``src/main.py`` / ``src/asgi.py``) on 2026-10-15. The
services' RequestId / SecurityHeaders / RequestLogging are ``BaseHTTPMiddleware``
subclasses (stand-ins here); Auth is w2v_common's real ``AuthMiddleware``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

import w2v_common.admission.disconnect as disconnect_mod
from tests.test_admission_disconnect_through_middleware import (
    ROUTE,
    _abandon_while_queued,
    _bare,
    _new_gate,
    _PassThrough,
    _router,
    _status,
)
from w2v_common.admission import DisconnectProbe, DisconnectProbeMiddleware
from w2v_common.admission.disconnect import PROBE_SCOPE_KEY
from w2v_common.auth.middleware import AuthMiddleware


class PrometheusInstrumentatorMiddleware:
    """Shape of prometheus_fastapi_instrumentator's middleware: pure ASGI, passes
    ``receive`` through untouched, wraps ``send`` to observe the status."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_wrapper(message: Message) -> None:
            await send(message)

        await self.app(scope, receive, send_wrapper)


class RequestIdMiddleware(_PassThrough):
    pass


class SecurityHeadersMiddleware(_PassThrough):
    pass


class RequestLoggingMiddleware(_PassThrough):
    pass


def _cors(app: FastAPI) -> None:
    app.add_middleware(CORSMiddleware, allow_origins=["*"])


def _auth(app: FastAPI) -> None:
    app.add_middleware(AuthMiddleware, extra_public_endpoints={ROUTE})


def _tts_or_asr(app: FastAPI) -> None:
    _cors(app)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    _auth(app)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(DisconnectProbeMiddleware)
    app.add_middleware(PrometheusInstrumentatorMiddleware)  # Instrumentator().instrument(app)


def _image_gen(app: FastAPI) -> None:
    _cors(app)
    _auth(app)
    app.add_middleware(PrometheusInstrumentatorMiddleware)  # instrument() BEFORE the probe
    app.add_middleware(DisconnectProbeMiddleware)


def _transcription_worker(app: FastAPI) -> None:
    _cors(app)
    _auth(app)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(DisconnectProbeMiddleware)


LAYOUTS: dict[str, Callable[[FastAPI], None]] = {
    "tts_backend": _tts_or_asr,
    "asr_backend": _tts_or_asr,
    "image_gen_backend": _image_gen,
    "transcription_worker_service": _transcription_worker,
}


@pytest.fixture(autouse=True)
def _reset_warn_latch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(disconnect_mod, "_warned_not_outermost", False)


def _probe_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "install_disconnect_probe" in r.getMessage()
    ]


@pytest.mark.parametrize("service", sorted(LAYOUTS))
async def test_real_service_layout_keeps_the_disconnect_skip(
    service: str, caplog: pytest.LogCaptureFixture
) -> None:
    gate, lane = _new_gate()
    ran: list[str] = []
    app = _bare(_router(gate, lane, ran))
    LAYOUTS[service](app)
    caplog.set_level(logging.WARNING, logger=disconnect_mod.__name__)

    messages = await _abandon_while_queued(app, gate, lane)

    assert ran == [], f"{service}: the abandoned job ran on the GPU"
    assert _status(messages) == 499, f"{service}: {messages}"
    assert _probe_warnings(caplog) == []
    assert await gate.shutdown()


async def test_base_http_middleware_outside_a_hand_added_probe_warns_and_never_polls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gate, lane = _new_gate()
    ran: list[str] = []
    app = _bare(_router(gate, lane, ran))
    _transcription_worker(app)
    app.add_middleware(RequestIdMiddleware)  # a later layer, OUTSIDE the probe
    caplog.set_level(logging.WARNING, logger=disconnect_mod.__name__)

    messages = await _abandon_while_queued(app, gate, lane)

    assert ran == ["abandoned"]  # skip off: no poll ...
    assert _status(messages) == 200  # ... and no crash
    assert len(_probe_warnings(caplog)) == 1


# ── the verdict itself, without a request ─────────────────────────────────────


class _App:
    def __init__(self, *classes: type) -> None:
        self.user_middleware = [Middleware(c) for c in classes]


class _SneakyBaseHTTP(RequestIdMiddleware):
    """An unknown class is judged by subclassing alone."""


async def _verdict(app: object | None) -> bool:
    seen: list[DisconnectProbe] = []

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope[PROBE_SCOPE_KEY])

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_m: Message) -> None:
        return None

    scope: dict[str, Any] = {"type": "http"}
    if app is not None:
        scope["app"] = app
    await DisconnectProbeMiddleware(inner)(scope, receive, send)
    return seen[0]._direct


@pytest.mark.parametrize(
    ("app", "direct"),
    [
        (_App(DisconnectProbeMiddleware, RequestIdMiddleware), True),
        (_App(PrometheusInstrumentatorMiddleware, DisconnectProbeMiddleware), True),
        (_App(CORSMiddleware, DisconnectProbeMiddleware), True),  # unknown non-BaseHTTP = pure
        (_App(RequestIdMiddleware, DisconnectProbeMiddleware), False),
        (
            _App(PrometheusInstrumentatorMiddleware, _SneakyBaseHTTP, DisconnectProbeMiddleware),
            False,
        ),
        (_App(PrometheusInstrumentatorMiddleware), False),  # probe not listed: no evidence
        (object(), False),  # no user_middleware
        (None, False),  # no scope["app"]
    ],
    ids=[
        "probe-outermost",
        "pure-asgi-outside",
        "unknown-pure-outside",
        "base-http-outside",
        "base-http-subclass-outside",
        "probe-not-listed",
        "app-without-user-middleware",
        "no-app-in-scope",
    ],
)
async def test_hand_added_probe_polls_only_on_positive_evidence(
    app: object | None, direct: bool
) -> None:
    assert await _verdict(app) is direct
