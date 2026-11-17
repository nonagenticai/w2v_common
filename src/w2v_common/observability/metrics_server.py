"""Opt-in dedicated plaintext Prometheus listener (``W2V_METRICS_PORT``).

Every w2v API port is STRICT mTLS, so the cluster's non-ambient (plaintext)
Prometheus cannot scrape ``GET /metrics`` there. The orchestrator solved that
for itself with a second bare listener on 6301 (``API_METRICS_PORT``, its own
``src/lifecycle/metrics.py``); every other service exposed ``w2v_cell_*`` and
``w2v_auth_*`` only on the mTLS port, i.e. nowhere Prometheus reads.

This is the generic version: set ``W2V_METRICS_PORT`` and the process serves the
default ``prometheus_client`` REGISTRY (the one every w2v counter registers
against) on that port. Unset (the default) = nothing is bound. It is started
from :class:`w2v_common.auth.middleware.AuthMiddleware` construction, which
every w2v service installs, so no per-service code is needed. Process-wide and
idempotent: one listener per process however many apps/middlewares exist.

gitops still has to expose it per service — containerPort, Service port +
``prometheus.io/port`` annotation, a PERMISSIVE workload-scoped PeerAuth for
the port and a NetworkPolicy admitting ns ``prometheus`` — exactly the four
pieces the orchestrator has for 6301. A bind failure (port taken, e.g. the
orchestrator's own 6301 listener) logs a warning and never raises.
"""

from __future__ import annotations

import logging
import os
import threading
from wsgiref.simple_server import WSGIServer

from prometheus_client import start_http_server

logger = logging.getLogger(__name__)

METRICS_PORT_ENV = "W2V_METRICS_PORT"
#: ONLY the user's local mother service lives on 8000 — never bind it.
_FORBIDDEN_PORTS = frozenset({8000})

_lock = threading.Lock()
_server: WSGIServer | None = None
_thread: threading.Thread | None = None
_port: int | None = None


def _configured_port() -> int | None:
    raw = (os.getenv(METRICS_PORT_ENV) or "").strip()
    if not raw:
        return None
    try:
        port = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; metrics listener not started", METRICS_PORT_ENV, raw
        )
        return None
    if not 1 <= port <= 65535 or port in _FORBIDDEN_PORTS:
        logger.warning(
            "%s=%d is not an allowed port; metrics listener not started", METRICS_PORT_ENV, port
        )
        return None
    return port


def start_metrics_server_from_env() -> int | None:
    """Start the listener if ``W2V_METRICS_PORT`` is set. Returns the bound port.

    Idempotent: a second call returns the already-bound port. Never raises.
    """
    global _server, _thread, _port
    with _lock:
        if _server is not None:
            return _port
        port = _configured_port()
        if port is None:
            return None
        try:
            server, thread = start_http_server(port)
        except OSError as exc:
            logger.warning("metrics listener failed to bind :%d: %s", port, exc)
            return None
        _server, _thread, _port = server, thread, port
        logger.info("Prometheus metrics listener on :%d (%s)", port, METRICS_PORT_ENV)
        return port


def active_port() -> int | None:
    """The port the listener is bound to, or ``None``."""
    return _port


def stop_metrics_server() -> None:
    """Shut the listener down (tests / orderly shutdown). Safe when none runs."""
    global _server, _thread, _port
    with _lock:
        if _server is not None:
            _server.shutdown()
            _server.server_close()
        if _thread is not None:
            _thread.join(timeout=5)
        _server, _thread, _port = None, None, None
