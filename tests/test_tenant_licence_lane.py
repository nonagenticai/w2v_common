"""TenantLicenceLane (<HOP>_AUTH_MODE) + its first wiring, ArtifactClient, on the wire.

Assertions are on the request the mocked artifact_service actually received, with the
client's REAL event hooks installed, and on the mint counter labelled by app.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import respx
from prometheus_client import REGISTRY

from w2v_common.auth.bundle_tokens import (
    BundleMinterNotConfiguredError,
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
from w2v_common.auth.tenant_licence_lane import (
    HOP_APP_KEYS,
    HopAuthMode,
    TenantLicenceLane,
    hop_auth_mode_env,
    resolve_hop_auth_mode,
)
from w2v_common.clients.artifact_client import ArtifactClient
from w2v_common.tenant import TenantSource

TENANT_A = "0199a1b2-0000-7000-8000-00000000000a"
SHARED = "shared-downstream-licence"  # pragma: allowlist secret
CALLER = "callers-own-licence"  # pragma: allowlist secret
TOKEN_URL = "http://kc.test/realms/secure-apps/protocol/openid-connect/token"
BASE_URL = "http://artifacts.test:8556"
PROJECT_ID = uuid4()
ARTIFACT_ID = uuid4()
UPLOAD_OK = {
    "artifact_id": str(ARTIFACT_ID),
    "version": 1,
    "storage_key": "k",
    "size": 1,
    "sha256": "0" * 64,
}
_MINTER_ENV = (
    "BUNDLE_MINTER_CLIENT_ID",
    "BUNDLE_MINTER_CLIENT_SECRET",
    "BUNDLE_MINTER_TOKEN_URL",
    "BUNDLE_MINTER_BUNDLE",
)
_MODE_ENVS = ("ARTIFACTS_AUTH_MODE", "MODALITY_GATEWAY_AUTH_MODE")


def _metric(app: str, outcome: str) -> float:
    v = REGISTRY.get_sample_value("w2v_bundle_token_mint_total", {"app": app, "outcome": outcome})
    return 0.0 if v is None else v


class Licences:
    """Minter backend; each mint returns ``lic-<app>-<n>`` and records the app asked for."""

    def __init__(self) -> None:
        self.apps: list[str] = []
        self.status = 201
        self.error: str | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return httpx.Response(200, json={"access_token": "minter-sa", "expires_in": 300})
        tenant = request.url.path.split("/")[3]
        app = request.url.params["app"]
        if self.status != 201:
            return httpx.Response(self.status, json={"detail": {"error": self.error}})
        self.apps.append(app)
        exp = datetime.fromtimestamp(4_000_000_000, tz=UTC).isoformat()
        return httpx.Response(
            201,
            json={
                "token": f"lic-{app}-{len(self.apps)}",
                "expires_at": exp,
                "tenant_id": tenant,
                "app": app,
                "license_id": f"lid-{app}-{len(self.apps)}",
            },
        )


@pytest.fixture
def env(monkeypatch) -> Iterator[pytest.MonkeyPatch]:
    monkeypatch.setenv("DOWNSTREAM_LICENSE_TOKEN", SHARED)
    for name in (*_MINTER_ENV, *_MODE_ENVS):
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


def _bind(caller_token: str | None = None) -> None:
    set_identity_context(
        TENANT_A,
        "inbound-licence-id",
        "subject-1",
        caller_license_token=caller_token,
        tenant_source=TenantSource.LICENSE_TOKEN,
    )


async def _upload(client: ArtifactClient) -> None:
    await client.upload_bytes(project_id=PROJECT_ID, filename="f.txt", content=b"x")


# -- mode resolution -----------------------------------------------------------


def test_mode_env_name_per_hop() -> None:
    assert hop_auth_mode_env("artifacts") == "ARTIFACTS_AUTH_MODE"
    assert hop_auth_mode_env("modality_gateway") == "MODALITY_GATEWAY_AUTH_MODE"


def test_mode_unset_or_empty_is_shared(env) -> None:
    assert resolve_hop_auth_mode("artifacts") is HopAuthMode.SHARED
    env.setenv("ARTIFACTS_AUTH_MODE", "  ")
    assert resolve_hop_auth_mode("artifacts") is HopAuthMode.SHARED
    env.setenv("ARTIFACTS_AUTH_MODE", "Tenant_Licence")
    assert resolve_hop_auth_mode("artifacts") is HopAuthMode.TENANT_LICENCE


def test_mode_typo_fails_loud(env) -> None:
    env.setenv("ARTIFACTS_AUTH_MODE", "tenant-licence")
    with pytest.raises(ValueError, match="ARTIFACTS_AUTH_MODE"):
        TenantLicenceLane("artifacts")


def test_app_keys_per_hop() -> None:
    assert HOP_APP_KEYS == {
        "artifacts": "artifacts",
        "modality_gateway": "w2v",
        "billing": "w2v",
        "webscraper": "w2v",
        "media_discovery": "w2v",
    }
    assert TenantLicenceLane("artifacts", mode="shared").app == "artifacts"
    assert TenantLicenceLane("webscraper", mode="shared").app == "w2v"
    assert TenantLicenceLane("media_discovery", mode="shared").app == "w2v"
    assert hop_auth_mode_env("webscraper") == "WEBSCRAPER_AUTH_MODE"
    assert hop_auth_mode_env("media_discovery") == "MEDIA_DISCOVERY_AUTH_MODE"
    assert TenantLicenceLane("x", app="custom", mode="shared").app == "custom"
    with pytest.raises(ValueError, match="no licences app key"):
        TenantLicenceLane("unknown_hop", mode="shared")


def test_kag_hop_is_gone() -> None:
    """W0b: KAG is retired, so it is no longer a hop with a licences app key."""
    assert "kag" not in HOP_APP_KEYS
    with pytest.raises(ValueError, match="no licences app key"):
        TenantLicenceLane("kag", mode="shared")


# -- lane: shared default unchanged; tenant_licence minted ----------------------


async def test_shared_lane_returns_the_shared_token(env) -> None:
    lane = TenantLicenceLane("artifacts")
    assert lane.mode is HopAuthMode.SHARED
    assert await lane.headers() == {"X-Tenant-Id": f"Bearer {SHARED}"}
    assert await lane.headers(authorization=True) == {
        "X-Tenant-Id": f"Bearer {SHARED}",
        "Authorization": f"Bearer {SHARED}",
    }
    env.setenv("DOWNSTREAM_LICENSE_TOKEN", "")
    assert await TenantLicenceLane("artifacts").headers() == {}


async def test_tenant_lane_mints_caches_and_counts_per_app(licences) -> None:
    _bind()
    lane = TenantLicenceLane("artifacts", mode="tenant_licence")
    minted_before = _metric("artifacts", "minted")
    hit_before = _metric("artifacts", "cache_hit")
    first = await lane.headers()
    second = await lane.headers()
    assert first == {"X-Tenant-Id": "Bearer lic-artifacts-1", "X-License-Id": "lid-artifacts-1"}
    assert second == first
    assert licences.apps == ["artifacts"]  # cached: ONE mint
    assert _metric("artifacts", "minted") == minted_before + 1
    assert _metric("artifacts", "cache_hit") == hit_before + 1
    assert SHARED not in str(first)


async def test_modality_and_billing_lanes_mint_w2v(licences) -> None:
    _bind()
    await TenantLicenceLane("modality_gateway", mode="tenant_licence").headers()
    await TenantLicenceLane("billing", mode="tenant_licence").headers()
    # Same (tenant, app=w2v) key: the second hop is served from the cache.
    assert licences.apps == ["w2v"]


def test_tenant_lane_without_minter_config_fails_at_construction(env) -> None:
    before = _metric("artifacts", "not_configured")
    with pytest.raises(BundleMinterNotConfiguredError):
        TenantLicenceLane("artifacts", mode="tenant_licence")
    assert _metric("artifacts", "not_configured") == before + 1


async def test_tenant_lane_refusal_raises_never_shared(licences) -> None:
    _bind()
    licences.status, licences.error = 404, "no_bundle"
    lane = TenantLicenceLane("artifacts", mode="tenant_licence")
    with pytest.raises(BundleTokenRefusedError) as exc:
        await lane.headers()
    assert exc.value.error_code == "no_bundle"
    assert exc.value.app == "artifacts"


async def test_tenant_lane_without_tenant_raises(licences) -> None:
    lane = TenantLicenceLane("artifacts", mode="tenant_licence")
    with pytest.raises(NoTenantForLicenceError):
        await lane.headers()
    assert licences.apps == []


# -- ArtifactClient ---------------------------------------------------------------


async def test_artifact_default_mode_sends_shared_token_unchanged(env) -> None:
    client = ArtifactClient(base_url=BASE_URL)
    assert client.auth_mode is HopAuthMode.SHARED
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.post(f"/projects/{PROJECT_ID}/artifacts/upload").mock(
            return_value=httpx.Response(200, json=UPLOAD_OK)
        )
        await _upload(client)
    req = route.calls.last.request
    assert req.headers["Authorization"] == f"Bearer {SHARED}"
    assert req.headers["X-Tenant-Id"] == f"Bearer {SHARED}"
    await client.aclose()


async def test_artifact_default_mode_still_forwards_caller_token(env) -> None:
    _bind(CALLER)
    client = ArtifactClient(base_url=BASE_URL)
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.post(f"/projects/{PROJECT_ID}/artifacts/upload").mock(
            return_value=httpx.Response(200, json=UPLOAD_OK)
        )
        await _upload(client)
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {CALLER}"
    await client.aclose()


async def test_artifact_tenant_mode_sends_minted_artifacts_licence(licences) -> None:
    _bind()  # background-task shape: tenant bound, no caller token in scope
    client = ArtifactClient(base_url=BASE_URL, auth_mode="tenant_licence")
    before = _metric("artifacts", "minted")
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.post(f"/projects/{PROJECT_ID}/artifacts/upload").mock(
            return_value=httpx.Response(200, json=UPLOAD_OK)
        )
        await _upload(client)
        await _upload(client)
    req = route.calls.last.request
    assert req.headers["Authorization"] == "Bearer lic-artifacts-1"
    assert req.headers["X-Tenant-Id"] == "Bearer lic-artifacts-1"
    assert req.headers["X-License-Id"] == "lid-artifacts-1"
    assert "X-Effective-Tenant-Id" not in req.headers
    for call in route.calls:
        assert SHARED not in str(call.request.headers)
    assert licences.apps == ["artifacts"]  # second upload served from cache
    assert _metric("artifacts", "minted") == before + 1
    await client.aclose()


def test_artifact_tenant_mode_env_switch(env, licences) -> None:
    env.setenv("ARTIFACTS_AUTH_MODE", "tenant_licence")
    assert ArtifactClient(base_url=BASE_URL).auth_mode is HopAuthMode.TENANT_LICENCE


async def test_artifact_tenant_mode_prefers_callers_own_licence(licences) -> None:
    _bind(CALLER)
    client = ArtifactClient(base_url=BASE_URL, auth_mode="tenant_licence")
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.post(f"/projects/{PROJECT_ID}/artifacts/upload").mock(
            return_value=httpx.Response(200, json=UPLOAD_OK)
        )
        await _upload(client)
    req = route.calls.last.request
    assert req.headers["Authorization"] == f"Bearer {CALLER}"
    assert req.headers["X-Tenant-Id"] == f"Bearer {CALLER}"
    assert req.headers["X-License-Id"] == "inbound-licence-id"
    assert licences.apps == []  # nothing minted when the caller's licence is in scope
    await client.aclose()


async def test_artifact_tenant_mode_remints_once_on_401(licences, tmp_path: Path) -> None:
    _bind()
    client = ArtifactClient(base_url=BASE_URL, auth_mode="tenant_licence")
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get(f"/projects/{PROJECT_ID}/artifacts/{ARTIFACT_ID}/download").mock(
            side_effect=[httpx.Response(401), httpx.Response(200, content=b"bytes")]
        )
        out = await client.download_to_path(
            project_id=PROJECT_ID, artifact_id=ARTIFACT_ID, dest=tmp_path / "o.bin"
        )
    assert out.read_bytes() == b"bytes"
    sent = [c.request.headers["X-Tenant-Id"] for c in route.calls]
    assert sent == ["Bearer lic-artifacts-1", "Bearer lic-artifacts-2"]
    await client.aclose()


async def test_artifact_tenant_mode_mint_404_raises_and_sends_nothing(licences) -> None:
    _bind()
    licences.status, licences.error = 404, "no_bundle"
    client = ArtifactClient(base_url=BASE_URL, auth_mode="tenant_licence")
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        route = mock.post(f"/projects/{PROJECT_ID}/artifacts/upload").mock(
            return_value=httpx.Response(200, json=UPLOAD_OK)
        )
        with pytest.raises(BundleTokenRefusedError) as exc:
            await _upload(client)
    assert exc.value.error_code == "no_bundle"
    assert not route.called  # never fell back to the shared token
    await client.aclose()


async def test_artifact_tenant_mode_without_tenant_raises(licences) -> None:
    client = ArtifactClient(base_url=BASE_URL, auth_mode="tenant_licence")
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        route = mock.post(f"/projects/{PROJECT_ID}/artifacts/upload").mock(
            return_value=httpx.Response(200, json=UPLOAD_OK)
        )
        with pytest.raises(NoTenantForLicenceError):
            await _upload(client)
    assert not route.called
    await client.aclose()


def test_artifact_tenant_mode_without_minter_config_fails_at_construction(env) -> None:
    with pytest.raises(BundleMinterNotConfiguredError):
        ArtifactClient(base_url=BASE_URL, auth_mode="tenant_licence")
