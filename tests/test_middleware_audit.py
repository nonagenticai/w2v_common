"""Tests for w2v_common.middleware.audit.AuditMiddleware."""

from __future__ import annotations

import json
import logging
from typing import Any, cast

import pytest
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import Field

from w2v_common.asgi import create_app
from w2v_common.config import APISettings, AuditSettings
from w2v_common.middleware.audit import (
    DEFAULT_AUDIT_LOGGER,
    AuditMiddleware,
    fingerprint_api_key,
)

_TENANT = "0195f48b-0000-7000-8000-00000000f042"
_HEADER_TENANT = "0195f48b-0000-7000-8000-00000000f0ad"


def _audit_app(logger_name: str = DEFAULT_AUDIT_LOGGER) -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuditMiddleware, logger_name=logger_name)

    @app.get("/api/widgets")
    async def list_widgets() -> dict[str, str]:
        return {"ok": "true"}

    @app.post("/api/widgets")
    async def create_widget(request: Request) -> dict[str, str]:
        request.state.tenant_id = _TENANT
        return {"id": "w1"}

    @app.post("/api/forbidden")
    async def forbidden() -> JSONResponse:
        return JSONResponse({"detail": "nope"}, status_code=403)

    return app


def _audit_records(caplog: pytest.LogCaptureFixture, logger_name: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == logger_name]


def _payload(record: logging.LogRecord) -> dict[str, Any]:
    """Extract the single dict the middleware logs as a positional arg.

    ``logging`` collapses a lone Mapping positional into ``record.args``
    itself (not a 1-tuple), so the dict is ``record.args`` directly.
    """
    args = record.args
    assert isinstance(args, dict)
    return cast("dict[str, Any]", args)


def test_fingerprint_api_key_none_returns_none() -> None:
    assert fingerprint_api_key(None) is None
    assert fingerprint_api_key("") is None


def test_fingerprint_api_key_is_deterministic_16_hex() -> None:
    fp1 = fingerprint_api_key("secret-token")
    fp2 = fingerprint_api_key("secret-token")
    assert fp1 == fp2
    assert fp1 is not None
    assert len(fp1) == 16
    assert fingerprint_api_key("other-token") != fp1


def test_get_success_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    app = _audit_app()
    with caplog.at_level(logging.INFO, logger=DEFAULT_AUDIT_LOGGER), TestClient(app) as client:
        resp = client.get("/api/widgets")
    assert resp.status_code == 200
    assert _audit_records(caplog, DEFAULT_AUDIT_LOGGER) == []


def test_post_mutation_logged_at_info(caplog: pytest.LogCaptureFixture) -> None:
    app = _audit_app()
    with caplog.at_level(logging.INFO, logger=DEFAULT_AUDIT_LOGGER), TestClient(app) as client:
        resp = client.post("/api/widgets", json={"x": 1})
    assert resp.status_code == 200
    records = _audit_records(caplog, DEFAULT_AUDIT_LOGGER)
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.INFO
    assert record.getMessage().startswith("AUDIT: ")
    payload = _payload(record)
    assert payload["method"] == "POST"
    assert payload["resource"] == "widget"
    assert payload["tenant_id"] == _TENANT  # set on request.state by the handler
    assert payload["status_code"] == 200


def test_auth_failure_logged_at_warning(caplog: pytest.LogCaptureFixture) -> None:
    app = _audit_app()
    with caplog.at_level(logging.INFO, logger=DEFAULT_AUDIT_LOGGER), TestClient(app) as client:
        resp = client.post("/api/forbidden")
    assert resp.status_code == 403
    records = _audit_records(caplog, DEFAULT_AUDIT_LOGGER)
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].getMessage().startswith("AUTH_FAILURE: ")


def test_tenant_id_from_header_when_state_absent(caplog: pytest.LogCaptureFixture) -> None:
    app = _audit_app()
    with caplog.at_level(logging.INFO, logger=DEFAULT_AUDIT_LOGGER), TestClient(app) as client:
        # /api/forbidden never sets request.state.tenant_id → header fallback.
        client.post("/api/forbidden", headers={"x-tenant-id": _HEADER_TENANT})
    assert _payload(_audit_records(caplog, DEFAULT_AUDIT_LOGGER)[0])["tenant_id"] == (
        _HEADER_TENANT
    )


def test_bearer_tenant_header_is_sanitised(caplog: pytest.LogCaptureFixture) -> None:
    app = _audit_app()
    with caplog.at_level(logging.INFO, logger=DEFAULT_AUDIT_LOGGER), TestClient(app) as client:
        client.post("/api/forbidden", headers={"x-tenant-id": "Bearer leak-me"})
    # CONTRACT CHANGE: this used to assert the literal "anonymous". That was a
    # SENTINEL in a tenant field — a value that reads as a tenant, aggregates as
    # a tenant, and belongs to nobody, so every unattributed 401 in the fleet
    # merged into one fake customer. A 401 legitimately has NO tenant; None is
    # the honest representation, and the token still never reaches the log.
    payload = _payload(_audit_records(caplog, DEFAULT_AUDIT_LOGGER)[0])
    assert payload["tenant_id"] is None
    assert "leak-me" not in json.dumps(payload)


def test_non_uuid_tenant_header_is_not_logged_as_a_tenant(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hand-crafted slug is not a tenant, so it is not recorded as one."""
    app = _audit_app()
    with caplog.at_level(logging.INFO, logger=DEFAULT_AUDIT_LOGGER), TestClient(app) as client:
        client.post("/api/forbidden", headers={"x-tenant-id": "n8n-dev"})
    assert _payload(_audit_records(caplog, DEFAULT_AUDIT_LOGGER)[0])["tenant_id"] is None


def test_api_key_fingerprint_recorded_not_raw(caplog: pytest.LogCaptureFixture) -> None:
    app = _audit_app()
    with caplog.at_level(logging.INFO, logger=DEFAULT_AUDIT_LOGGER), TestClient(app) as client:
        client.post("/api/widgets", json={"x": 1}, headers={"authorization": "Bearer my-secret"})
    payload = _payload(_audit_records(caplog, DEFAULT_AUDIT_LOGGER)[0])
    assert payload["api_key_fingerprint"] == fingerprint_api_key("my-secret")
    assert payload["api_key_fingerprint"] != "my-secret"  # pragma: allowlist secret


def test_api_key_fingerprint_from_license_token_in_x_tenant_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """License tokens ride in `X-Tenant-Id: Bearer <token>`, not Authorization.
    The fingerprint must be derived from that token, not left null."""
    app = _audit_app()
    with caplog.at_level(logging.INFO, logger=DEFAULT_AUDIT_LOGGER), TestClient(app) as client:
        client.post("/api/widgets", json={"x": 1}, headers={"X-Tenant-Id": "Bearer lic-secret"})
    payload = _payload(_audit_records(caplog, DEFAULT_AUDIT_LOGGER)[0])
    assert payload["api_key_fingerprint"] == fingerprint_api_key("lic-secret")
    assert payload["api_key_fingerprint"] is not None


def test_custom_logger_name_is_used(caplog: pytest.LogCaptureFixture) -> None:
    app = _audit_app(logger_name="myservice.audit")
    with caplog.at_level(logging.INFO, logger="myservice.audit"), TestClient(app) as client:
        client.post("/api/widgets", json={"x": 1})
    assert len(_audit_records(caplog, "myservice.audit")) == 1
    assert _audit_records(caplog, DEFAULT_AUDIT_LOGGER) == []


# ---------------------------------------------------------------------------
# create_app opt-in gate via AuditSettings.
# ---------------------------------------------------------------------------


class _AuditSettings(APISettings):
    audit: AuditSettings = Field(default_factory=AuditSettings)


def _router() -> APIRouter:
    router = APIRouter()

    @router.post("/api/things")
    async def create_thing() -> dict[str, str]:
        return {"id": "t1"}

    return router


def test_create_app_audit_enabled_logs(caplog: pytest.LogCaptureFixture) -> None:
    settings = _AuditSettings(
        port=8080,
        host="127.0.0.1",
        audit=AuditSettings(enabled=True, logger_name="thing.audit"),
    )
    app = create_app(settings, routers=[_router()], auth_enabled=False)
    with caplog.at_level(logging.INFO, logger="thing.audit"), TestClient(app) as client:
        resp = client.post("/api/things", json={"x": 1})
    assert resp.status_code == 200
    records = _audit_records(caplog, "thing.audit")
    assert len(records) == 1
    assert records[0].getMessage().startswith("AUDIT: ")


def test_create_app_audit_disabled_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    settings = _AuditSettings(port=8081, host="127.0.0.1")  # enabled defaults False
    app = create_app(settings, routers=[_router()], auth_enabled=False)
    with caplog.at_level(logging.INFO, logger=DEFAULT_AUDIT_LOGGER), TestClient(app) as client:
        resp = client.post("/api/things", json={"x": 1})
    assert resp.status_code == 200
    assert _audit_records(caplog, DEFAULT_AUDIT_LOGGER) == []
