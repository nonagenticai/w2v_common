"""TenantLicenceLane wired into pooled clients (BaseServiceClient).

The modality_gateway / billing hops are switched by ``<HOP>_AUTH_MODE``. These
tests assert on the request the mocked downstream actually received, with each
client's REAL event hooks installed, and on the mint counter labelled by app.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import httpx
import pytest
import respx
from prometheus_client import REGISTRY

from w2v_common.auth.bundle_tokens import (
    BundleTokenMinter,
    BundleTokenRefusedError,
    reset_bundle_token_minter,
    set_bundle_token_minter,
)
from w2v_common.auth.identity_context import (
    NoTenantForLicenceError,
    clear_identity_context,
    set_identity_context,
)
from w2v_common.auth.tenant_licence_lane import TenantLicenceLane
from w2v_common.clients.base import BaseServiceClient
from w2v_common.tenant import TenantSource

TENANT_A = "0199a1b2-0000-7000-8000-00000000000a"
SHARED = "shared-downstream-licence"  # pragma: allowlist secret
TOKEN_URL = "http://kc.test/realms/secure-apps/protocol/openid-connect/token"
BASE_URL = "http://modality.test:8890"
_ENV = (
    "BUNDLE_MINTER_CLIENT_ID",
    "BUNDLE_MINTER_CLIENT_SECRET",
    "BUNDLE_MINTER_TOKEN_URL",
    "BUNDLE_MINTER_BUNDLE",
    "MODALITY_GATEWAY_AUTH_MODE",
    "BILLING_AUTH_MODE",
)


def _metric(app: str, outcome: str) -> float:
    v = REGISTRY.get_sample_value("w2v_bundle_token_mint_total", {"app": app, "outcome": outcome})
    return 0.0 if v is None else v


class Licences:
    """Minter backend: each mint returns ``lic-<app>-<n>`` and records the app."""

    def __init__(self) -> None:
        self.apps: list[str] = []
        self.status = 201

    def handler(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return httpx.Response(200, json={"access_token": "minter-sa", "expires_in": 300})
        if self.status != 201:
            return httpx.Response(self.status, json={"detail": {"error": "no_bundle"}})
        app = request.url.params["app"]
        self.apps.append(app)
        return httpx.Response(
            201,
            json={
                "token": f"lic-{app}-{len(self.apps)}",
                "expires_at": datetime.fromtimestamp(4_000_000_000, tz=UTC).isoformat(),
                "tenant_id": request.url.path.split("/")[3],
                "app": app,
                "license_id": f"lid-{app}-{len(self.apps)}",
            },
        )


@pytest.fixture
def env(monkeypatch) -> Iterator[pytest.MonkeyPatch]:
    monkeypatch.setenv("DOWNSTREAM_LICENSE_TOKEN", SHARED)
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    reset_bundle_token_minter()
    clear_identity_context()
    yield monkeypatch
    clear_identity_context()
    reset_bundle_token_minter()


@pytest.fixture
def licences(env) -> Licences:
    lic = Licences()
    set_bundle_token_minter(
        BundleTokenMinter(
            licenses_url="http://lic.test",
            token_url=TOKEN_URL,
            client_id="w2v-bundle-minter",
            client_secret="s",  # pragma: allowlist secret
            transport=httpx.MockTransport(lic.handler),
        )
    )
    return lic


def _bind() -> None:
    set_identity_context(
        TENANT_A,
        "inbound-licence-id",
        "subject-1",
        tenant_source=TenantSource.LICENSE_TOKEN,
    )


def _client(lane: TenantLicenceLane | None, token: str | None = SHARED) -> BaseServiceClient:
    return BaseServiceClient(
        base_url=BASE_URL, service_name="tts", license_token=token, licence_lane=lane
    )


# -- BaseServiceClient -----------------------------------------------------------


async def test_shared_lane_is_byte_identical_to_no_lane(env) -> None:
    _bind()
    seen: list[httpx.Headers] = []
    for lane in (None, TenantLicenceLane("modality_gateway")):
        client = _client(lane)
        with respx.mock(base_url=BASE_URL) as mock:
            route = mock.get("/x").mock(return_value=httpx.Response(200, json={}))
            await client._request("GET", "/x")
        seen.append(route.calls.last.request.headers)
        await client.close()
    no_lane, shared_lane = seen
    assert shared_lane["X-Tenant-Id"] == f"Bearer {SHARED}"
    assert {k: v for k, v in shared_lane.items() if k.lower() != "x-request-id"} == {
        k: v for k, v in no_lane.items() if k.lower() != "x-request-id"
    }


async def test_tenant_lane_sends_minted_licence_not_shared(licences) -> None:
    _bind()
    before = _metric("w2v", "minted")
    client = _client(TenantLicenceLane("modality_gateway", mode="tenant_licence"))
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.post("/api/tts/synthesize").mock(return_value=httpx.Response(200, json={}))
        await client._request("POST", "/api/tts/synthesize", json={"text": "hi"})
    req = route.calls.last.request
    assert req.headers["X-Tenant-Id"] == "Bearer lic-w2v-1"
    assert req.headers["X-License-Id"] == "lid-w2v-1"
    assert "X-Effective-Tenant-Id" not in req.headers
    assert SHARED not in str(req.headers)
    assert licences.apps == ["w2v"]
    assert _metric("w2v", "minted") == before + 1
    await client.close()


async def test_tenant_lane_401_invalidates_so_next_call_remints(licences) -> None:
    _bind()
    client = _client(TenantLicenceLane("modality_gateway", mode="tenant_licence"))
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/x").mock(side_effect=[httpx.Response(401), httpx.Response(200, json={})])
        with pytest.raises(httpx.HTTPStatusError):
            await client._request("GET", "/x")
        await client._request("GET", "/x")
    sent = [c.request.headers["X-Tenant-Id"] for c in route.calls]
    assert sent == ["Bearer lic-w2v-1", "Bearer lic-w2v-2"]
    await client.close()


async def test_tenant_lane_cached_licence_is_reused(licences) -> None:
    _bind()
    client = _client(TenantLicenceLane("modality_gateway", mode="tenant_licence"))
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/x").mock(return_value=httpx.Response(200, json={}))
        await client._request("GET", "/x")
        await client._request("GET", "/x")
    assert [c.request.headers["X-Tenant-Id"] for c in route.calls] == ["Bearer lic-w2v-1"] * 2
    assert licences.apps == ["w2v"]
    await client.close()


async def test_tenant_lane_without_tenant_raises_and_sends_nothing(licences) -> None:
    client = _client(TenantLicenceLane("modality_gateway", mode="tenant_licence"))
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        route = mock.get("/x").mock(return_value=httpx.Response(200, json={}))
        with pytest.raises(NoTenantForLicenceError):
            await client._request("GET", "/x")
    assert not route.called
    assert licences.apps == []
    await client.close()


async def test_tenant_lane_refused_mint_raises_never_shared(licences) -> None:
    _bind()
    licences.status = 404
    client = _client(TenantLicenceLane("modality_gateway", mode="tenant_licence"))
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        route = mock.get("/x").mock(return_value=httpx.Response(200, json={}))
        with pytest.raises(BundleTokenRefusedError):
            await client._request("GET", "/x")
    assert not route.called
    await client.close()


async def test_tenant_lane_health_check_needs_no_tenant(licences) -> None:
    client = _client(TenantLicenceLane("modality_gateway", mode="tenant_licence"))
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/health").mock(return_value=httpx.Response(200, json={"ok": True}))
        result = await client.health_check()
    assert result["status"] == "healthy"
    assert "X-Tenant-Id" not in route.calls.last.request.headers
    assert licences.apps == []
    await client.close()


def test_shared_lane_event_hooks_refused(env) -> None:
    with pytest.raises(RuntimeError, match="MODALITY_GATEWAY_AUTH_MODE is shared"):
        TenantLicenceLane("modality_gateway").event_hooks()
