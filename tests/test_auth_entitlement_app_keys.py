"""Licence gate: configurable app keys + the licence-token fast path.

Owner decision 2026-10-22: tts/asr become shared PLATFORM services licensed
under app key ``speech``, consumed by Studio (``w2v``) and the agent platform.

* ``W2V_LICENSE_GATE_APP_KEYS`` (default ``w2v``) — entitled if ANY key is.
* A request authenticated by its OWN licence token (introspected by
  ``AuthMiddleware``, stored on ``request.state.license``) whose app key is
  accepted and live is entitled WITHOUT ``/licenses/active`` — that lookup
  needs cross-tenant visibility the service's Keycloak client may lack.

The fast-path tests run the REAL ``AuthMiddleware`` in front of the gate, so
they prove the middleware actually populates what the gate reads.
"""

from __future__ import annotations

import ipaddress
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import APIRouter, Depends, FastAPI, WebSocket
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect

import w2v_common.auth.license_app as la
import w2v_common.auth.middleware as mw
from w2v_common.auth import entitlement
from w2v_common.auth.entitlement import (
    LICENSE_GATE,
    LICENSE_GATE_APP_KEYS_ENV,
    LICENSE_GATE_TOKEN_APP_KEYS_ENV,
    LICENSE_GATE_WS,
    clear_license_status_cache,
    license_gate_app_keys,
)
from w2v_common.auth.licenses_read_client import (
    ActiveLicenseView,
    AppInfoView,
    LicensesClientError,
    LicenseView,
    set_licenses_read_client,
)
from w2v_common.auth.middleware import AuthMiddleware

TENANT = "0195f48b-0000-7000-8000-0000000000aa"
OTHER_TENANT = "0195f48b-0000-7000-8000-0000000000cc"
SUBJECT = "0195f48b-0000-7000-8000-0000000000bb"
_TOKEN = {"X-Tenant-Id": "Bearer opaque-licence-token"}


class PerAppLicenses:
    """``/licenses/active`` stand-in answering per app key.

    ``modes[app_key]``: ``active`` | ``none`` | ``boom`` (502) | ``forbidden``
    (403 — the cross-tenant-visibility failure agent-api hit) | ``revoked``
    (404 + a revoked inventory row). Unlisted → none.
    """

    def __init__(self, modes: dict[str, str] | None = None) -> None:
        self.modes = modes or {}
        self.calls: list[str] = []

    async def get_active_license(self, *, tenant_id: str, app_key: str):
        self.calls.append(app_key)
        mode = self.modes.get(app_key, "none")
        if mode == "active":
            return ActiveLicenseView(
                license_id=f"lic-{app_key}",
                tenant_id=tenant_id,
                expires_at="2099-01-01T00:00:00+00:00",
                app=AppInfoView(key=app_key),
            )
        if mode == "boom":
            raise LicensesClientError("licenses unreachable", status_code=502)
        if mode == "forbidden":
            raise LicensesClientError("tenant_not_accessible", status_code=403)
        return None

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str):
        if any(m == "forbidden" for m in self.modes.values()):
            raise LicensesClientError("tenant_not_accessible", status_code=403)
        return [
            LicenseView(
                id=f"lic-{key}",
                tenant_id=tenant_id,
                status="revoked",
                revoked_at="2026-10-22T00:00:00+00:00",
                app=AppInfoView(key=key),
            )
            for key, mode in self.modes.items()
            if mode == "revoked"
        ]


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    clear_license_status_cache()
    set_licenses_read_client(None)
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    monkeypatch.delenv(LICENSE_GATE_APP_KEYS_ENV, raising=False)
    monkeypatch.delenv(LICENSE_GATE_TOKEN_APP_KEYS_ENV, raising=False)
    monkeypatch.delenv("AUTH_DELEGATING_SUBJECTS", raising=False)
    for name in (
        "_license_cache",
        "_license_neg_cache",
        "_license_stale_cache",
        "_license_token_not_found_since",
        "_license_tnf_probe",
    ):
        setattr(mw, name, None)
    mw._introspect_locks.clear()
    for var in (
        la.LICENSE_APP_KEYS_ENV,
        "AUTH_TRUST_MESH_LICENSE_IDENTITY",
        "AUTH_MESH_VOUCH_SECRET",
        "AUTH_TRUST_INCLUSTER_IDENTITY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mw, "TRUSTED_NETWORKS", [ipaddress.ip_network("127.0.0.1/32")])
    yield
    clear_license_status_cache()
    set_licenses_read_client(None)
    mw._introspect_locks.clear()


def _use(fake: PerAppLicenses) -> PerAppLicenses:
    set_licenses_read_client(fake)
    return fake


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_default_accepted_keys_is_w2v_only():
    assert license_gate_app_keys() == ("w2v",)


@pytest.mark.parametrize("raw", ["", " ", " , ,"])
def test_blank_env_falls_back_to_w2v(monkeypatch, raw):
    monkeypatch.setenv(LICENSE_GATE_APP_KEYS_ENV, raw)
    assert license_gate_app_keys() == ("w2v",)


def test_token_keys_default_empty_and_must_be_accepted(monkeypatch):
    assert entitlement.license_gate_token_app_keys(("w2v",)) == frozenset()
    monkeypatch.setenv(LICENSE_GATE_TOKEN_APP_KEYS_ENV, "speech, kag")
    # `kag` is not accepted by the gate → no effect.
    assert entitlement.license_gate_token_app_keys(("speech", "w2v")) == {"speech"}


def test_env_is_parsed_in_order_and_deduplicated(monkeypatch):
    monkeypatch.setenv(LICENSE_GATE_APP_KEYS_ENV, " speech , w2v,speech ")
    assert license_gate_app_keys() == ("speech", "w2v")


# ---------------------------------------------------------------------------
# Lookup path (identity stamped directly — no licence token on the request)
# ---------------------------------------------------------------------------


def _stamped_app(state: dict[str, Any] | None = None) -> FastAPI:
    app = FastAPI()
    gated = APIRouter(dependencies=list(LICENSE_GATE))

    @gated.get("/api/things")
    async def things() -> dict[str, str]:
        return {"ok": "yes"}

    app.include_router(gated)

    @app.middleware("http")
    async def _stamp(request, call_next):
        request.state.tenant_id = TENANT
        request.state.subject_id = SUBJECT
        for k, v in (state or {}).items():
            setattr(request.state, k, v)
        return await call_next(request)

    return app


def _get_stamped(state: dict[str, Any] | None = None):
    return TestClient(_stamped_app(state), raise_server_exceptions=False).get("/api/things")


def test_default_unchanged_w2v_licence_entitles_and_only_w2v_is_asked():
    fake = _use(PerAppLicenses({"w2v": "active"}))
    assert _get_stamped().status_code == 200
    assert fake.calls == ["w2v"]


def test_default_unchanged_a_speech_licence_does_not_entitle():
    fake = _use(PerAppLicenses({"speech": "active"}))
    resp = _get_stamped()
    assert resp.status_code == 402
    assert resp.json()["detail"]["error"] == "license_none"
    assert fake.calls == ["w2v"]


@pytest.mark.parametrize("entitled_key", ["speech", "w2v"])
def test_speech_w2v_accepts_either_key(monkeypatch, entitled_key):
    monkeypatch.setenv(LICENSE_GATE_APP_KEYS_ENV, "speech,w2v")
    _use(PerAppLicenses({entitled_key: "active"}))
    assert _get_stamped().status_code == 200


def test_speech_w2v_refuses_when_neither_is_licensed(monkeypatch):
    monkeypatch.setenv(LICENSE_GATE_APP_KEYS_ENV, "speech,w2v")
    fake = _use(PerAppLicenses({"inference-router": "active"}))
    resp = _get_stamped()
    assert resp.status_code == 402
    assert fake.calls == ["speech", "w2v"]


def test_an_indeterminate_key_is_503_not_402(monkeypatch):
    """One key refused, the other could not be checked → we do NOT know → 503."""
    monkeypatch.setenv(LICENSE_GATE_APP_KEYS_ENV, "speech,w2v")
    _use(PerAppLicenses({"speech": "none", "w2v": "boom"}))
    resp = _get_stamped()
    assert resp.status_code == 503
    assert resp.headers["Retry-After"]


def test_a_forged_state_is_not_a_client_header():
    """Headers never feed the fast path: an X-App-Key header changes nothing."""
    _use(PerAppLicenses({}))
    client = TestClient(_stamped_app(), raise_server_exceptions=False)
    resp = client.get("/api/things", headers={"x-app-key": "w2v", "x-license-status": "active"})
    assert resp.status_code == 402


# ---------------------------------------------------------------------------
# Fast path — through the REAL AuthMiddleware licence lane
# ---------------------------------------------------------------------------


def _introspection(app_key: str | None, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "active": True,
        "tenant_id": TENANT,
        "subject_id": SUBJECT,
        "license_id": "lic-agent",
        "status": "active",
        "expires_at": "2099-01-01T00:00:00+00:00",
    }
    if app_key is not None:
        payload["app"] = {"id": "app-1", "key": app_key, "name": app_key}
    payload.update(extra)
    return payload


async def _get_via_middleware(headers: dict[str, str] | None = None):
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    gated = APIRouter(dependencies=list(LICENSE_GATE))

    @gated.get("/api/data")
    async def data(request: Request) -> dict[str, str]:
        return {"tenant_id": request.state.tenant_id}

    app.include_router(gated)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await c.get("/api/data", headers={**_TOKEN, **(headers or {})})


def _arm(monkeypatch: pytest.MonkeyPatch, keys: str = "speech,w2v", token: str = "speech"):
    monkeypatch.setenv(LICENSE_GATE_APP_KEYS_ENV, keys)
    monkeypatch.setenv(LICENSE_GATE_TOKEN_APP_KEYS_ENV, token)


@pytest.mark.asyncio
@patch("w2v_common.auth.middleware._introspect_license")
async def test_fast_path_entitles_an_accepted_active_licence_without_lookup(
    introspect, monkeypatch
):
    """agent-api's repro: its tenant is invisible to the w2v client (403)."""
    _arm(monkeypatch)
    introspect.return_value = _introspection("speech")
    fake = _use(PerAppLicenses({"speech": "forbidden", "w2v": "forbidden"}))
    resp = await _get_via_middleware()
    assert resp.status_code == 200, resp.text
    assert fake.calls == []


@pytest.mark.asyncio
@patch("w2v_common.auth.middleware._introspect_license")
async def test_fast_path_is_off_by_default_even_for_an_active_w2v_token(introspect):
    """Both envs unset: an ACTIVE w2v licence token is still looked up, so a
    licence revoked since the token was introspected is refused (~60s), not
    served for the 300s/900s introspection window."""
    introspect.return_value = _introspection("w2v")
    fake = _use(PerAppLicenses({"w2v": "revoked"}))
    resp = await _get_via_middleware()
    assert resp.status_code == 402, resp.text
    assert resp.json()["detail"]["error"] == "license_revoked"
    assert fake.calls == ["w2v"]


@pytest.mark.asyncio
@patch("w2v_common.auth.middleware._introspect_license")
async def test_w2v_stays_on_the_lookup_when_only_speech_is_armed(introspect, monkeypatch):
    _arm(monkeypatch)  # APP_KEYS=speech,w2v  TOKEN_APP_KEYS=speech
    introspect.return_value = _introspection("w2v")
    fake = _use(PerAppLicenses({"w2v": "revoked"}))
    resp = await _get_via_middleware()
    assert resp.status_code == 402, resp.text
    assert fake.calls == ["speech", "w2v"]


@pytest.mark.asyncio
@patch("w2v_common.auth.middleware._introspect_license")
async def test_token_key_not_in_app_keys_has_no_effect(introspect, monkeypatch):
    _arm(monkeypatch, keys="w2v", token="speech")
    introspect.return_value = _introspection("speech")
    fake = _use(PerAppLicenses({}))
    assert (await _get_via_middleware()).status_code == 402
    assert fake.calls == ["w2v"]


@pytest.mark.asyncio
@patch("w2v_common.auth.middleware._introspect_license")
async def test_licence_for_another_app_gets_no_fast_path(introspect, monkeypatch):
    """An inference-router licence must not entitle tts/asr: falls to the lookup."""
    _arm(monkeypatch, token="speech,inference-router")
    introspect.return_value = _introspection("inference-router")
    fake = _use(PerAppLicenses({"inference-router": "active"}))
    resp = await _get_via_middleware()
    assert resp.status_code == 402, resp.text
    assert fake.calls == ["speech", "w2v"]


@pytest.mark.asyncio
@patch("w2v_common.auth.middleware._introspect_license")
async def test_speech_licence_is_not_accepted_by_a_default_studio_gate(introspect):
    introspect.return_value = _introspection("speech")
    fake = _use(PerAppLicenses({}))
    assert (await _get_via_middleware()).status_code == 402
    assert fake.calls == ["w2v"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"status": "suspended"},
        {"status": "revoked"},
        {"expires_at": "2020-01-01T00:00:00+00:00"},
        {"starts_at": "2099-01-01T00:00:00+00:00"},
        {"expires_at": "not-a-date"},
    ],
)
@patch("w2v_common.auth.middleware._introspect_license")
async def test_non_live_licence_gets_no_fast_path(introspect, monkeypatch, extra):
    _arm(monkeypatch, keys="speech")
    introspect.return_value = _introspection("speech", **extra)
    fake = _use(PerAppLicenses({}))
    assert (await _get_via_middleware()).status_code == 402
    assert fake.calls == ["speech"]


@pytest.mark.asyncio
@patch("w2v_common.auth.middleware._introspect_license")
async def test_introspection_without_app_gets_no_fast_path(introspect, monkeypatch):
    _arm(monkeypatch, keys="w2v", token="w2v")
    introspect.return_value = _introspection(None)
    fake = _use(PerAppLicenses({}))
    assert (await _get_via_middleware()).status_code == 402
    assert fake.calls == ["w2v"]


# ---------------------------------------------------------------------------
# Fast path guards that need a hand-built state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("w2v_common.auth.middleware._introspect_license")
async def test_delegated_tenant_gets_no_fast_path(introspect, monkeypatch):
    """A delegating SA's speech licence never entitles the tenant it acts for —
    through the REAL middleware's X-Effective-Tenant-Id re-scope."""
    _arm(monkeypatch)
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", SUBJECT)
    introspect.return_value = _introspection("speech")
    fake = _use(PerAppLicenses({}))
    resp = await _get_via_middleware({"X-Effective-Tenant-Id": OTHER_TENANT})
    assert resp.status_code == 402, resp.text
    assert fake.calls == ["speech", "w2v"]


def test_licence_for_another_tenant_gets_no_fast_path(monkeypatch):
    """Unreachable through the middleware (the licence lane serves the
    introspected tenant unless delegated — covered above); guarded anyway."""
    _arm(monkeypatch, keys="w2v", token="w2v")
    fake = _use(PerAppLicenses({}))
    resp = _get_stamped({"license": _introspection("w2v", tenant_id=OTHER_TENANT)})
    assert resp.status_code == 402
    assert fake.calls == ["w2v"]


def test_inactive_introspection_gets_no_fast_path(monkeypatch):
    _arm(monkeypatch, keys="w2v", token="w2v")
    fake = _use(PerAppLicenses({}))
    resp = _get_stamped({"license": _introspection("w2v", active=False)})
    assert resp.status_code == 402
    assert fake.calls == ["w2v"]


# ---------------------------------------------------------------------------
# WebSocket twin — same decisions. AuthMiddleware is a BaseHTTPMiddleware and
# never runs on a websocket scope, so these cannot go through it: a service
# stamps websocket.state itself, which is what _stamp models.
# ---------------------------------------------------------------------------


def _ws_client(license_payload: dict[str, Any] | None) -> TestClient:
    app = FastAPI()

    async def _stamp(websocket: WebSocket) -> None:
        websocket.state.tenant_id = TENANT
        websocket.state.subject_id = SUBJECT
        if license_payload is not None:
            websocket.state.license = license_payload

    @app.websocket("/api/stream", dependencies=[Depends(_stamp), *LICENSE_GATE_WS])
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text("hello")
        await websocket.close()

    return TestClient(app, raise_server_exceptions=False)


def test_ws_speech_w2v_accepts_speech_lookup(monkeypatch):
    monkeypatch.setenv(LICENSE_GATE_APP_KEYS_ENV, "speech,w2v")
    _use(PerAppLicenses({"speech": "active"}))
    with _ws_client(None).websocket_connect("/api/stream") as ws:
        assert ws.receive_text() == "hello"


def test_ws_default_refuses_speech_only_tenant():
    _use(PerAppLicenses({"speech": "active"}))
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        _ws_client(None).websocket_connect("/api/stream"),
    ):
        pass  # pragma: no cover
    assert exc.value.code == entitlement.LICENSE_WS_CLOSE_POLICY


def test_ws_fast_path_entitles_accepted_licence(monkeypatch):
    _arm(monkeypatch, keys="speech")
    fake = _use(PerAppLicenses({"speech": "forbidden"}))
    with _ws_client(_introspection("speech")).websocket_connect("/api/stream") as ws:
        assert ws.receive_text() == "hello"
    assert fake.calls == []


def test_ws_fast_path_refuses_other_app(monkeypatch):
    _arm(monkeypatch, keys="speech")
    _use(PerAppLicenses({}))
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        _ws_client(_introspection("inference-router")).websocket_connect("/api/stream"),
    ):
        pass  # pragma: no cover
    assert exc.value.code == entitlement.LICENSE_WS_CLOSE_POLICY
