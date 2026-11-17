"""BundleTokenMinter: tenant-x-app licence minting, caching, single-flight, errors, metric."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx
import pytest
from prometheus_client import REGISTRY

from w2v_common.auth import bundle_tokens as bt
from w2v_common.auth.bundle_tokens import (
    NEGATIVE_TTL_S,
    REFRESH_LEAD_S,
    BundleMinterNotConfiguredError,
    BundleTokenMinter,
    BundleTokenRefusedError,
    BundleTokenUnavailableError,
    WMAuthMode,
    get_bundle_token_minter,
    reset_bundle_token_minter,
    resolve_wm_auth_mode,
)

TENANT_A = "0199a1b2-0000-7000-8000-00000000000a"
TENANT_B = "0199a1b2-0000-7000-8000-00000000000b"
TOKEN_URL = "http://kc.test/realms/secure-apps/protocol/openid-connect/token"
LIC_URL = "http://lic.test"
NOW = 1_800_000_000.0


def _metric(app: str, outcome: str) -> float:
    v = REGISTRY.get_sample_value("w2v_bundle_token_mint_total", {"app": app, "outcome": outcome})
    return 0.0 if v is None else v


class FakeLicences:
    """Keycloak token endpoint + licences bundle-token route over one MockTransport."""

    def __init__(self) -> None:
        self.sa_calls = 0
        self.mints: list[httpx.Request] = []
        self.status = 201
        self.error: str | None = None
        self.expires_in_s = 3600.0
        self.delay = 0.0
        self.body_override: dict | None = None
        self.raise_transport = False
        self.sa_status = 200
        self.sa_delay = 0.0

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            self.sa_calls += 1
            if self.sa_delay:
                await asyncio.sleep(self.sa_delay)
            if self.sa_status != 200:
                return httpx.Response(self.sa_status)
            return httpx.Response(
                200, json={"access_token": f"sa-{self.sa_calls}", "expires_in": 300}
            )
        assert request.url.path.endswith("/bundle-token")
        if self.raise_transport:
            raise httpx.ConnectError("boom", request=request)
        self.mints.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        tenant = request.url.path.split("/")[3]
        if self.status != 201:
            detail = {"error": self.error} if self.error else "x"
            return httpx.Response(self.status, json={"detail": detail})
        if self.body_override is not None:
            return httpx.Response(201, json=self.body_override)
        expires = datetime.fromtimestamp(NOW + self.expires_in_s, tz=UTC).isoformat()
        return httpx.Response(
            201,
            json={
                "token": f"tok-{tenant}-{len(self.mints)}",
                "expires_at": expires,
                "tenant_id": tenant,
                "bundle": "w2v",
                "app": request.url.params["app"],
                "license_id": "lic-1",
                "token_id": "tid-1",
            },
        )


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> float:
        return self.now


def _minter(fake: FakeLicences, clock: Clock, *, bundle: str | None = None) -> BundleTokenMinter:
    return BundleTokenMinter(
        licenses_url=LIC_URL,
        token_url=TOKEN_URL,
        client_id="w2v-bundle-minter",
        client_secret="s",  # pragma: allowlist secret
        bundle=bundle,
        transport=httpx.MockTransport(fake.handler),
        clock=clock,
    )


@pytest.mark.asyncio
async def test_mint_request_shape_and_token() -> None:
    fake = FakeLicences()
    m = _minter(fake, Clock())
    before = _metric("workers", "minted")
    tok = await m.token(TENANT_A)
    assert tok == f"tok-{TENANT_A}-1"
    req = fake.mints[0]
    assert req.method == "POST"
    assert req.url.path == f"/internal/tenants/{TENANT_A}/bundle-token"
    assert req.url.params["app"] == "workers"
    assert req.url.params["ttl_minutes"] == "60"
    assert "bundle" not in req.url.params
    assert req.headers["Authorization"] == "Bearer sa-1"
    assert _metric("workers", "minted") == before + 1


@pytest.mark.asyncio
async def test_bundle_setting_is_sent_as_query_param() -> None:
    fake = FakeLicences()
    await _minter(fake, Clock(), bundle="w2v").token(TENANT_A)
    assert fake.mints[0].url.params["bundle"] == "w2v"


@pytest.mark.asyncio
async def test_cache_hit_makes_no_second_licences_call() -> None:
    fake = FakeLicences()
    m = _minter(fake, Clock())
    hits = _metric("workers", "cache_hit")
    first = await m.token(TENANT_A)
    second = await m.token(TENANT_A)
    assert first == second
    assert len(fake.mints) == 1
    assert fake.sa_calls == 1
    assert _metric("workers", "cache_hit") == hits + 1


@pytest.mark.asyncio
async def test_refresh_at_expiry_minus_five_minutes() -> None:
    fake = FakeLicences()
    clock = Clock()
    m = _minter(fake, clock)
    await m.token(TENANT_A)
    clock.now = NOW + 3600 - REFRESH_LEAD_S - 1  # still inside the window
    assert await m.token(TENANT_A) == f"tok-{TENANT_A}-1"
    assert len(fake.mints) == 1
    clock.now = NOW + 3600 - REFRESH_LEAD_S + 1  # inside the 5-minute lead: re-mint
    assert await m.token(TENANT_A) == f"tok-{TENANT_A}-2"
    assert len(fake.mints) == 2


@pytest.mark.asyncio
async def test_token_expiring_inside_lead_is_used_but_not_cached() -> None:
    fake = FakeLicences()
    fake.expires_in_s = 120  # licence expiry caps the token below the lead
    m = _minter(fake, Clock())
    await m.token(TENANT_A)
    await m.token(TENANT_A)
    assert len(fake.mints) == 2


@pytest.mark.asyncio
async def test_single_flight_under_concurrency_per_key() -> None:
    fake = FakeLicences()
    fake.delay = 0.02
    m = _minter(fake, Clock())
    results = await asyncio.gather(
        *[m.token(TENANT_A) for _ in range(20)], *[m.token(TENANT_B) for _ in range(20)]
    )
    assert len(fake.mints) == 2  # one per tenant, not 40
    assert len(set(results[:20])) == 1
    assert len(set(results[20:])) == 1
    assert results[0] != results[20]


@pytest.mark.parametrize(
    ("status", "error", "exc_type", "outcome", "code"),
    [
        (404, "tenant_not_found", BundleTokenRefusedError, "not_found", "tenant_not_found"),
        (404, "app_not_in_bundle", BundleTokenRefusedError, "not_found", "app_not_in_bundle"),
        (404, "no_app_licence", BundleTokenRefusedError, "not_found", "no_app_licence"),
        (403, "unknown_api_client", BundleTokenRefusedError, "forbidden", "unknown_api_client"),
        (409, "ambiguous_bundle", BundleTokenRefusedError, "conflict", "ambiguous_bundle"),
        (401, None, BundleTokenRefusedError, "unauthenticated", "unauthenticated"),
        (422, None, BundleTokenRefusedError, "rejected", "rejected"),
        (503, None, BundleTokenUnavailableError, "server_error", "server_error"),
    ],
)
@pytest.mark.asyncio
async def test_error_codes_are_mapped(status, error, exc_type, outcome, code) -> None:
    fake = FakeLicences()
    fake.status, fake.error = status, error
    m = _minter(fake, Clock())
    before = _metric("workers", outcome)
    with pytest.raises(exc_type) as ei:
        await m.token(TENANT_A)
    assert type(ei.value) is exc_type
    assert ei.value.error_code == code
    assert ei.value.status_code == status
    assert ei.value.outcome == outcome
    assert ei.value.tenant_id == TENANT_A
    assert _metric("workers", outcome) == before + 1


@pytest.mark.parametrize(("status", "error"), [(404, "no_app_licence"), (403, None), (409, None)])
@pytest.mark.asyncio
async def test_refusals_are_negative_cached(status, error) -> None:
    fake = FakeLicences()
    fake.status, fake.error = status, error
    clock = Clock()
    m = _minter(fake, clock)
    neg = _metric("workers", "negative_cache_hit")
    with pytest.raises(BundleTokenRefusedError):
        await m.token(TENANT_A)
    with pytest.raises(BundleTokenRefusedError):
        await m.token(TENANT_A)
    assert len(fake.mints) == 1  # second refusal served from the negative cache
    assert _metric("workers", "negative_cache_hit") == neg + 1
    # A literal, not the constant: a refusal is remembered for AT MOST 60 s, so a
    # licence provisioned after the refusal is picked up within a minute.
    assert NEGATIVE_TTL_S <= 60
    clock.now = NOW + 61
    fake.status = 201
    assert await m.token(TENANT_A) == f"tok-{TENANT_A}-2"


@pytest.mark.parametrize("status", [401, 503])
@pytest.mark.asyncio
async def test_transient_and_sa_failures_are_not_negative_cached(status) -> None:
    fake = FakeLicences()
    fake.status = status
    m = _minter(fake, Clock())
    for _ in range(2):
        with pytest.raises(bt.BundleTokenError):
            await m.token(TENANT_A)
    assert len(fake.mints) == 2


@pytest.mark.asyncio
async def test_transport_error() -> None:
    fake = FakeLicences()
    fake.raise_transport = True
    before = _metric("workers", "transport_error")
    with pytest.raises(BundleTokenUnavailableError) as ei:
        await _minter(fake, Clock()).token(TENANT_A)
    assert ei.value.error_code == "transport_error"
    assert _metric("workers", "transport_error") == before + 1


@pytest.mark.asyncio
async def test_keycloak_failure_is_sa_error() -> None:
    fake = FakeLicences()
    fake.sa_status = 401
    before = _metric("workers", "sa_error")
    with pytest.raises(BundleTokenUnavailableError) as ei:
        await _minter(fake, Clock()).token(TENANT_A)
    assert ei.value.error_code == "minter_auth_rejected"
    assert fake.mints == []
    assert _metric("workers", "sa_error") == before + 1


@pytest.mark.parametrize(
    "body",
    [
        {"expires_at": "2027-01-01T00:00:00+00:00", "tenant_id": TENANT_A, "app": "workers"},
        {"token": "t", "tenant_id": TENANT_A, "app": "workers"},
        {
            "token": "t",
            "expires_at": "2027-01-01T00:00:00+00:00",
            "tenant_id": TENANT_B,  # minted for ANOTHER tenant
            "app": "workers",
        },
        {
            "token": "t",
            "expires_at": "2027-01-01T00:00:00+00:00",
            "tenant_id": TENANT_A,
            "app": "w2v",  # minted for another app
        },
    ],
)
@pytest.mark.asyncio
async def test_malformed_or_mismatched_body_is_bad_response(body) -> None:
    fake = FakeLicences()
    fake.body_override = body
    before = _metric("workers", "bad_response")
    with pytest.raises(BundleTokenUnavailableError) as ei:
        await _minter(fake, Clock()).token(TENANT_A)
    assert ei.value.error_code == "malformed_response"
    assert _metric("workers", "bad_response") == before + 1


@pytest.mark.asyncio
async def test_invalidate_forces_a_remint() -> None:
    fake = FakeLicences()
    m = _minter(fake, Clock())
    await m.token(TENANT_A)
    m.invalidate(TENANT_A)
    assert await m.token(TENANT_A) == f"tok-{TENANT_A}-2"


@pytest.mark.asyncio
async def test_forbidden_drops_the_cached_sa_token() -> None:
    fake = FakeLicences()
    m = _minter(fake, Clock())
    await m.token(TENANT_A)
    fake.status, fake.error = 403, "unknown_api_client"
    with pytest.raises(BundleTokenRefusedError):
        await m.token(TENANT_B)
    fake.status = 201
    await m.token("0199a1b2-0000-7000-8000-00000000000c")
    assert fake.sa_calls == 2


@pytest.fixture
def _clean_minter_env(monkeypatch):
    for name in (
        "BUNDLE_MINTER_CLIENT_ID",
        "BUNDLE_MINTER_CLIENT_SECRET",
        "BUNDLE_MINTER_TOKEN_URL",
        "BUNDLE_MINTER_BUNDLE",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_bundle_token_minter()
    yield monkeypatch
    reset_bundle_token_minter()


def test_unconfigured_minter_raises_and_counts(_clean_minter_env) -> None:
    mp = _clean_minter_env
    # The w2v-backend Keycloak env must NOT configure the minter (no silent fallback).
    mp.setenv("KEYCLOAK_URL", "http://kc.test")
    mp.setenv("KEYCLOAK_CLIENT_ID", "w2v-backend")
    mp.setenv("KEYCLOAK_CLIENT_SECRET", "x")  # pragma: allowlist secret
    mp.setenv("BUNDLE_MINTER_CLIENT_ID", "w2v-bundle-minter")
    mp.setenv("BUNDLE_MINTER_CLIENT_SECRET", "y")  # pragma: allowlist secret
    before = _metric("workers", "not_configured")
    with pytest.raises(BundleMinterNotConfiguredError):
        get_bundle_token_minter()
    assert _metric("workers", "not_configured") == before + 1


@pytest.mark.asyncio
async def test_env_minter_uses_its_own_token_url_and_bundle(_clean_minter_env) -> None:
    mp = _clean_minter_env
    mp.setenv("BUNDLE_MINTER_CLIENT_ID", "w2v-bundle-minter")
    mp.setenv("BUNDLE_MINTER_CLIENT_SECRET", "y")  # pragma: allowlist secret
    mp.setenv("BUNDLE_MINTER_TOKEN_URL", TOKEN_URL)
    mp.setenv("BUNDLE_MINTER_BUNDLE", "w2v")
    mp.setenv("W2V_LICENSES_URL", LIC_URL)
    m = get_bundle_token_minter()
    assert m._token_url == TOKEN_URL
    assert m._bundle == "w2v"
    assert m._licenses_url == LIC_URL
    assert m._client_id == "w2v-bundle-minter"
    assert get_bundle_token_minter() is m


def test_wm_auth_mode_default_and_values(monkeypatch) -> None:
    monkeypatch.delenv("WM_AUTH_MODE", raising=False)
    assert resolve_wm_auth_mode() is WMAuthMode.DELEGATION
    monkeypatch.setenv("WM_AUTH_MODE", "tenant_licence")
    assert resolve_wm_auth_mode() is WMAuthMode.TENANT_LICENCE
    assert resolve_wm_auth_mode("delegation") is WMAuthMode.DELEGATION


def test_wm_auth_mode_typo_fails_loud(monkeypatch) -> None:
    monkeypatch.setenv("WM_AUTH_MODE", "tenant_license")
    with pytest.raises(ValueError, match="WM_AUTH_MODE"):
        resolve_wm_auth_mode()


@pytest.mark.asyncio
async def test_sa_token_is_refreshed_30s_before_keycloak_expiry() -> None:
    """Keycloak says expires_in=300: the SA token is reused until 270 s, not 300 s."""
    fake = FakeLicences()
    clock = Clock()
    m = _minter(fake, clock)
    await m.token(TENANT_A)
    assert fake.sa_calls == 1
    clock.now = NOW + 269
    m.invalidate(TENANT_A)
    await m.token(TENANT_A)
    assert fake.sa_calls == 1  # still inside its early-refresh window
    clock.now = NOW + 271  # past expires_in - 30, before expires_in
    m.invalidate(TENANT_A)
    await m.token(TENANT_A)
    assert fake.sa_calls == 2


@pytest.mark.asyncio
async def test_sa_token_fetch_is_single_flight() -> None:
    fake = FakeLicences()
    fake.sa_delay = 0.02
    m = _minter(fake, Clock())
    tenants = [f"0199a1b2-0000-7000-8000-0000000001{i:02d}" for i in range(10)]
    await asyncio.gather(*[m.token(t) for t in tenants])
    assert len(fake.mints) == 10
    assert fake.sa_calls == 1


@pytest.mark.asyncio
async def test_lock_map_is_bounded_and_loop_keyed() -> None:
    import gc

    fake = FakeLicences()
    m = _minter(fake, Clock())
    tenants = [f"0199a1b2-0000-7000-8000-0000000002{i:02d}" for i in range(50)]
    await asyncio.gather(*[m.token(t) for t in tenants])
    gc.collect()
    loop = asyncio.get_running_loop()
    # Keyed by the loop object itself, and no lock outlives its users.
    assert list(m._locks.keys()) == [loop]
    assert len(m._locks[loop]) == 0


def test_lock_map_drops_a_closed_loop() -> None:
    import gc

    fake = FakeLicences()
    m = _minter(fake, Clock())
    asyncio.run(m.token(TENANT_A))
    m.invalidate(TENANT_A)
    asyncio.run(m.token(TENANT_A))  # a second loop works, and the first is gone
    gc.collect()
    assert len(m._locks) == 0
    assert len(m._sa_locks) == 0


@pytest.mark.asyncio
async def test_reprs_never_show_tokens() -> None:
    fake = FakeLicences()
    m = _minter(fake, Clock())
    licence = await m.licence(TENANT_A)
    assert licence.license_id == "lic-1"
    assert licence.token_id == "tid-1"
    assert licence.token not in repr(licence)
    assert all(licence.token not in repr(e) for e in m._entries.values())
    assert m._sa is not None
    assert m._sa.token not in repr(m._sa)
