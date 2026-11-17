"""#222: a VALID licence for another app must not open this service.

A ``kag`` licence token got 200 from tts/asr because ``AuthMiddleware`` read
only ``active`` and the tenant from introspection. ``AUTH_LICENSE_APP_KEYS``
declares the apps a service serves; when set, any other app is ``403 WRONG_APP``.
"""

from __future__ import annotations

import ipaddress
import logging
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

import w2v_common.auth.license_app as la
import w2v_common.auth.middleware as mw
from w2v_common.auth.middleware import AuthMiddleware

pytestmark = pytest.mark.asyncio

TENANT = "0195f48b-0000-7000-8000-0000000000aa"
SUBJECT_ID = "0195f48b-0000-7000-8000-0000000000bb"
ENV = la.LICENSE_APP_KEYS_ENV


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
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
        ENV,
        "AUTH_TRUST_MESH_LICENSE_IDENTITY",
        "AUTH_MESH_VOUCH_SECRET",
        "AUTH_TRUST_INCLUSTER_IDENTITY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mw, "TRUSTED_NETWORKS", [ipaddress.ip_network("127.0.0.1/32")])
    monkeypatch.setattr(la, "_warned_unchecked", False)
    yield
    mw._introspect_locks.clear()


def _client() -> AsyncClient:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/api/data")
    async def data(request: Request):  # pragma: no cover - exercised via HTTP
        return {"tenant_id": request.state.tenant_id}

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _active(app_key: str | None = "w2v") -> dict[str, object]:
    payload: dict[str, object] = {
        "active": True,
        "tenant_id": TENANT,
        "subject_id": SUBJECT_ID,
        "license_id": "lic-1",
    }
    if app_key is not None:
        payload["app"] = {"id": "app-1", "key": app_key, "name": app_key}
    return payload


def _count(app_key: str, outcome: str) -> float:
    return la.license_app_total.labels(app_key=app_key, outcome=outcome)._value.get()


async def _get(headers: dict[str, str]):
    async with _client() as client:
        return await client.get("/api/data", headers=headers)


_TOKEN = {"X-Tenant-Id": "Bearer opaque-licence-token"}


class TestArmed:
    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_valid_licence_for_another_app_is_403(self, introspect, monkeypatch):
        """PROVES the #222 repro: another app's licence on a w2v service is refused."""
        monkeypatch.setenv(ENV, "w2v")
        introspect.return_value = _active("workers")
        before = _count("workers", "refused")
        resp = await _get(_TOKEN)
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == "WRONG_APP"
        assert _count("workers", "refused") == before + 1

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_licence_for_the_right_app_passes(self, introspect, monkeypatch):
        monkeypatch.setenv(ENV, "w2v")
        introspect.return_value = _active("w2v")
        before = _count("w2v", "allowed")
        resp = await _get(_TOKEN)
        assert resp.status_code == 200, resp.text
        assert resp.json()["tenant_id"] == TENANT
        assert _count("w2v", "allowed") == before + 1

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_any_listed_app_passes(self, introspect, monkeypatch):
        monkeypatch.setenv(ENV, " w2v , w2v-tts ")
        introspect.return_value = _active("w2v-tts")
        assert (await _get(_TOKEN)).status_code == 200

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_introspection_naming_no_app_fails_closed(self, introspect, monkeypatch):
        monkeypatch.setenv(ENV, "w2v")
        introspect.return_value = _active(None)
        resp = await _get(_TOKEN)
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "WRONG_APP"

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_flat_app_key_field_is_honoured(self, introspect, monkeypatch):
        monkeypatch.setenv(ENV, "w2v")
        introspect.return_value = {**_active(None), "app_key": "w2v"}
        assert (await _get(_TOKEN)).status_code == 200

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_authorization_bearer_shape_is_checked_too(self, introspect, monkeypatch):
        monkeypatch.setenv(ENV, "w2v")
        introspect.return_value = _active("workers")
        resp = await _get({"Authorization": "Bearer opaque-licence-token"})
        assert resp.status_code == 403


class TestExistingBehaviourUnchanged:
    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_inactive_token_is_still_401(self, introspect, monkeypatch):
        """Expired/revoked stays 401 INVALID_TOKEN — the app check runs after it."""
        monkeypatch.setenv(ENV, "w2v")
        introspect.return_value = {"active": False, "reason": "expired"}
        resp = await _get(_TOKEN)
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "INVALID_TOKEN"

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_unreachable_introspection_is_still_401(self, introspect, monkeypatch):
        monkeypatch.setenv(ENV, "w2v")
        introspect.return_value = None
        resp = await _get(_TOKEN)
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "INVALID_TOKEN"

    async def test_missing_credential_is_still_401(self, monkeypatch):
        monkeypatch.setenv(ENV, "w2v")
        resp = await _get({})
        assert resp.status_code == 401


class TestUnarmed:
    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_unset_passes_but_is_counted_unchecked(self, introspect):
        """Unset = not armed: no fleet outage on a bump, but never silent."""
        introspect.return_value = _active("workers")
        before = _count("workers", "unchecked")
        resp = await _get(_TOKEN)
        assert resp.status_code == 200
        assert _count("workers", "unchecked") == before + 1

    async def test_unset_warns_once_and_gauge_reads_zero(self, caplog):
        with caplog.at_level(logging.WARNING, logger=la.__name__):
            la.warn_if_license_app_unchecked()
            la.warn_if_license_app_unchecked()
        hits = [r for r in caplog.records if ENV in r.getMessage()]
        assert len(hits) == 1
        assert la.license_app_check_armed._value.get() == 0

    async def test_set_publishes_armed_gauge_and_does_not_warn(self, monkeypatch, caplog):
        monkeypatch.setenv(ENV, "w2v")
        with caplog.at_level(logging.WARNING, logger=la.__name__):
            la.warn_if_license_app_unchecked()
        assert not [r for r in caplog.records if ENV in r.getMessage()]
        assert la.license_app_check_armed._value.get() == 1

    async def test_blank_value_counts_as_unset(self, monkeypatch):
        monkeypatch.setenv(ENV, " , ")
        assert la.allowed_license_app_keys() is None


class TestMeshVouchedLane:
    def _injected(self, app_key: str | None) -> dict[str, str]:
        headers = {"X-Tenant-Id": TENANT, "X-License-Id": "lic-1", "X-Subject-Id": SUBJECT_ID}
        if app_key is not None:
            headers["X-App-Key"] = app_key
        return headers

    async def test_injected_other_app_is_403(self, monkeypatch):
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        monkeypatch.setenv(ENV, "w2v")
        resp = await _get(self._injected("workers"))
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "WRONG_APP"

    async def test_injected_right_app_passes(self, monkeypatch):
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        monkeypatch.setenv(ENV, "w2v")
        assert (await _get(self._injected("w2v"))).status_code == 200

    async def test_injected_set_without_app_fails_closed_when_armed(self, monkeypatch):
        monkeypatch.setenv("AUTH_TRUST_MESH_LICENSE_IDENTITY", "true")
        monkeypatch.setenv(ENV, "w2v")
        assert (await _get(self._injected(None))).status_code == 403


class TestLabelBound:
    async def test_label_is_shape_checked_and_capped(self, monkeypatch):
        monkeypatch.setattr(la, "_seen_labels", set())
        assert la._label(None) == "none"
        assert la._label("Not A Key!") == "invalid"
        for i in range(la._MAX_LABELS):
            assert la._label(f"app-{i}") == f"app-{i}"
        assert la._label("one-too-many") == "other"
        assert la._label("app-0") == "app-0"
