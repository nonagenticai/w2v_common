"""Opt-in dedicated plaintext Prometheus listener (``W2V_METRICS_PORT``, 0.91.2).

Only the orchestrator ran a plaintext metrics port (6301); every other w2v
service exposed ``w2v_cell_*`` only on its STRICT-mTLS API port, which the
cluster's plaintext Prometheus cannot scrape. These fail on 0.91.1:
``w2v_common.observability.metrics_server`` did not exist.
"""

from __future__ import annotations

import socket
import urllib.request
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from w2v_common.auth import cell_jwt  # noqa: F401 — registers w2v_cell_* families
from w2v_common.auth.middleware import AuthMiddleware
from w2v_common.observability import metrics_server
from w2v_common.observability.metrics_server import (
    METRICS_PORT_ENV,
    start_metrics_server_from_env,
    stop_metrics_server,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(METRICS_PORT_ENV, raising=False)
    stop_metrics_server()
    yield
    stop_metrics_server()


def _scrape(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as resp:  # nosec B310 — loopback test
        return resp.read().decode()


def test_unset_env_starts_nothing() -> None:
    assert start_metrics_server_from_env() is None
    assert metrics_server.active_port() is None


def test_env_port_serves_the_default_registry_including_cell_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = _free_port()
    monkeypatch.setenv(METRICS_PORT_ENV, str(port))
    assert start_metrics_server_from_env() == port
    body = _scrape(port)
    assert "w2v_cell_jwt_rejected_total" in body
    assert "w2v_cell_dual_both_total" in body
    assert "w2v_cell_legacy_delegation_total" in body


def test_start_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    port = _free_port()
    monkeypatch.setenv(METRICS_PORT_ENV, str(port))
    assert start_metrics_server_from_env() == port
    assert start_metrics_server_from_env() == port  # no second bind, no OSError


@pytest.mark.parametrize("raw", ["abc", "0", "70000", "8000", "-1"])
def test_invalid_port_is_refused_not_raised(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv(METRICS_PORT_ENV, raw)
    assert start_metrics_server_from_env() is None
    assert metrics_server.active_port() is None


def test_bind_failure_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("0.0.0.0", 0))  # nosec B104 — test occupies the port on all ifaces
        taken.listen()
        monkeypatch.setenv(METRICS_PORT_ENV, str(taken.getsockname()[1]))
        assert start_metrics_server_from_env() is None


async def test_auth_middleware_startup_opens_the_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every w2v service installs AuthMiddleware — that is the generic hook."""
    port = _free_port()
    monkeypatch.setenv(METRICS_PORT_ENV, str(port))
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await c.get("/health")  # builds the middleware stack, as app startup does
    assert metrics_server.active_port() == port
    assert "w2v_cell_" in _scrape(port)


async def test_auth_middleware_without_env_opens_nothing() -> None:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await c.get("/health")
    assert metrics_server.active_port() is None
