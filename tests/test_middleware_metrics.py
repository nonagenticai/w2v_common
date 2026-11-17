"""Tests for w2v_common.middleware.metrics + observability.metrics.

The canonical counters live on the prometheus_client default REGISTRY and
accumulate across the whole test process, so every assertion measures a
before/after delta rather than an absolute value.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY
from pydantic import Field

from w2v_common.asgi import create_app
from w2v_common.auth.middleware import PUBLIC_ENDPOINTS
from w2v_common.config import APISettings, MetricsSettings
from w2v_common.middleware.metrics import MetricsMiddleware
from w2v_common.observability.metrics import metrics_router, render_latest


def _requests_total(method: str, endpoint: str, status: str) -> float:
    value = REGISTRY.get_sample_value(
        "w2v_http_requests_total",
        {"method": method, "endpoint": endpoint, "status_code": status},
    )
    return value or 0.0


def _duration_count(method: str, endpoint: str) -> float:
    value = REGISTRY.get_sample_value(
        "w2v_http_request_duration_seconds_count",
        {"method": method, "endpoint": endpoint},
    )
    return value or 0.0


def test_metrics_middleware_records_count_and_duration() -> None:
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)

    @app.get("/widget")
    async def widget() -> dict[str, str]:
        return {"ok": "true"}

    before_count = _requests_total("GET", "/widget", "200")
    before_duration = _duration_count("GET", "/widget")

    with TestClient(app) as client:
        resp = client.get("/widget")

    assert resp.status_code == 200
    assert _requests_total("GET", "/widget", "200") == before_count + 1
    assert _duration_count("GET", "/widget") == before_duration + 1


def test_metrics_middleware_labels_status_code() -> None:
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)

    @app.get("/missing")
    async def missing() -> dict[str, str]:
        raise HTTPException(status_code=404, detail="nope")

    before = _requests_total("GET", "/missing", "404")
    with TestClient(app) as client:
        resp = client.get("/missing")
    assert resp.status_code == 404
    assert _requests_total("GET", "/missing", "404") == before + 1


def test_render_latest_returns_exposition() -> None:
    payload, content_type = render_latest()
    assert isinstance(payload, bytes)
    assert content_type == CONTENT_TYPE_LATEST
    assert b"w2v_http_requests_total" in payload


def test_metrics_router_endpoint_serves_exposition() -> None:
    app = FastAPI()
    app.include_router(metrics_router)
    with TestClient(app) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "w2v_http_request_duration_seconds" in resp.text


def test_metrics_endpoint_is_auth_exempt() -> None:
    # The Prometheus scraper carries no license token; /metrics must be public.
    assert "/metrics" in PUBLIC_ENDPOINTS


# ---------------------------------------------------------------------------
# create_app opt-in gate via MetricsSettings.
# ---------------------------------------------------------------------------


class _MetricsSettings(APISettings):
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)


def _router() -> APIRouter:
    router = APIRouter()

    @router.get("/ping")
    async def ping() -> dict[str, str]:
        return {"ok": "true"}

    return router


def test_create_app_metrics_enabled_records_and_exposes() -> None:
    settings = _MetricsSettings(port=8080, host="127.0.0.1", metrics=MetricsSettings(enabled=True))
    app = create_app(settings, routers=[_router()], auth_enabled=False)

    before = _requests_total("GET", "/ping", "200")
    with TestClient(app) as client:
        ping = client.get("/ping")
        metrics = client.get("/metrics")

    assert ping.status_code == 200
    assert _requests_total("GET", "/ping", "200") == before + 1
    assert metrics.status_code == 200
    assert "w2v_http_requests_total" in metrics.text


def test_create_app_metrics_disabled_has_no_endpoint() -> None:
    settings = _MetricsSettings(port=8081, host="127.0.0.1")  # enabled defaults False
    app = create_app(settings, routers=[_router()], auth_enabled=False)
    with TestClient(app) as client:
        assert client.get("/ping").status_code == 200
        assert client.get("/metrics").status_code == 404


def test_create_app_metrics_absent_setting_has_no_endpoint() -> None:
    settings = APISettings(port=8082, host="127.0.0.1")  # no metrics attr at all
    app = create_app(settings, routers=[_router()], auth_enabled=False)
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 404


# ``W2V_METRICS_PORT`` (0.91.2) exposes this registry to Prometheus. Labels must
# be BOUNDED: the raw path embeds run/project/tenant UUIDs and any caller (even
# an unauthenticated scanner) picks the path and the method token, so labelling
# by them minted one series per request — a scrape-side cardinality DoS.


def test_metrics_middleware_labels_by_route_template_not_raw_path() -> None:
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)

    @app.get("/runs/{run_id}")
    async def run(run_id: str) -> dict[str, str]:
        return {"id": run_id}

    before = _requests_total("GET", "/runs/{run_id}", "200")
    with TestClient(app) as client:
        assert client.get("/runs/0198f1d2-0000-7000-8000-0000000c3111").status_code == 200
        assert client.get("/runs/0198f1d2-0000-7000-8000-0000000c3222").status_code == 200

    assert _requests_total("GET", "/runs/{run_id}", "200") == before + 2
    assert _requests_total("GET", "/runs/0198f1d2-0000-7000-8000-0000000c3111", "200") == 0.0


def test_metrics_middleware_collapses_unmatched_paths_and_unknown_methods() -> None:
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)

    before_404 = _requests_total("GET", "<unmatched>", "404")
    before_other = sum(_requests_total("OTHER", "<unmatched>", code) for code in ("404", "405"))
    with TestClient(app) as client:
        assert client.get("/scanner/aaaa-1").status_code == 404
        assert client.get("/scanner/aaaa-2").status_code == 404
        client.request("XYZZY1", "/scanner/x")

    assert _requests_total("GET", "<unmatched>", "404") == before_404 + 2
    assert _requests_total("GET", "/scanner/aaaa-1", "404") == 0.0
    after_other = sum(_requests_total("OTHER", "<unmatched>", code) for code in ("404", "405"))
    assert after_other == before_other + 1
    assert not any(
        s.labels.get("method") == "XYZZY1" for m in REGISTRY.collect() for s in m.samples
    )
