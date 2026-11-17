"""Which credential the entitlement gate presents to the licenses service.

The gate resolves entitlement for the tenant it is SERVING, which need not be
the tenant of whoever called. Before this, the only credentials on offer were
opaque licence tokens — the caller's own, or the never-configured static
``W2V_LICENSES_SERVICE_TOKEN``. The licenses service pins every introspected
licence token to exactly ``["license-reader"]`` on its own tenant with no
``azp``, so any lookup for a different tenant answered 403 and the gate resolved
``unknown`` -> 503 ``license_unavailable``. That happened identically for a
LICENSED tenant and an unlicensed one, i.e. the 402 ``license_none`` path never
ran in production.

The fix presents the ``w2v-backend`` client-credentials JWT, which the licenses
service already allowlists as a trusted multi-tenant proxy. The constraint these
tests hold down is that this is DELEGATION and not a widening: an ordinary tenant
caller asking about a different tenant must still get 403. That half is enforced
by the licenses service and pinned there
(``tests/auth/test_authz_idor.py::TestGetActiveLicenseIdor``); what is pinned
HERE is that the gate keeps presenting a credential whose scope the licenses
service can still evaluate — and that no fallback silently upgrades.
"""

from __future__ import annotations

import httpx
import pytest

from w2v_common.auth import licenses_read_client as lrc


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    """No ambient credential leaks in from the developer's shell or a peer test."""
    for name in (
        "KEYCLOAK_URL",
        "KEYCLOAK_REALM",
        "KEYCLOAK_SERVICE_REALM",
        "KEYCLOAK_CLIENT_ID",
        "KEYCLOAK_CLIENT_SECRET",
        "W2V_LICENSES_SERVICE_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    lrc.reset_service_token_minter()
    yield
    lrc.reset_service_token_minter()


def _configure_sa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEYCLOAK_URL", "https://auth.example.test")
    monkeypatch.setenv("KEYCLOAK_REALM", "portal")
    monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "w2v-backend")
    monkeypatch.setenv("KEYCLOAK_CLIENT_SECRET", "s3cret")  # pragma: allowlist secret
    lrc.reset_service_token_minter()


class TestServiceTokenPreference:
    async def test_sa_jwt_is_presented_when_configured(self, monkeypatch):
        """The SA JWT wins over the caller's opaque licence token.

        It has to: the caller token can only ever resolve the CALLER'S OWN
        tenant, which is the bug.
        """
        _configure_sa(monkeypatch)
        monkeypatch.setattr(lrc, "get_caller_license_token", lambda: "opaque-caller-token")

        seen: list[dict] = []

        async def fake_post(self, url, data=None, **kw):
            seen.append({"url": url, "data": data})
            return httpx.Response(
                200,
                json={"access_token": "sa-jwt", "expires_in": 300},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        assert await lrc._default_token() == "sa-jwt"
        assert seen[0]["url"] == (
            "https://auth.example.test/realms/portal/protocol/openid-connect/token"
        )
        assert seen[0]["data"]["grant_type"] == "client_credentials"
        assert seen[0]["data"]["client_id"] == "w2v-backend"

    async def test_token_is_cached_between_calls(self, monkeypatch):
        """One mint per pod per lifetime, not one per entitlement check."""
        _configure_sa(monkeypatch)
        calls = 0

        async def fake_post(self, url, data=None, **kw):
            nonlocal calls
            calls += 1
            return httpx.Response(
                200,
                json={"access_token": "sa-jwt", "expires_in": 3600},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        assert await lrc._default_token() == "sa-jwt"
        assert await lrc._default_token() == "sa-jwt"
        assert calls == 1

    async def test_falls_back_to_caller_token_when_mint_fails(self, monkeypatch):
        """A Keycloak outage must not take the gate from 'degraded' to 'broken'.

        The caller's token is still correct whenever the served tenant IS the
        caller's own, which is the majority of traffic.
        """
        _configure_sa(monkeypatch)
        monkeypatch.setattr(lrc, "get_caller_license_token", lambda: "opaque-caller-token")

        async def fake_post(self, url, data=None, **kw):
            raise httpx.ConnectError("keycloak down")

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        assert await lrc._default_token() == "opaque-caller-token"

    async def test_unconfigured_pod_keeps_the_previous_behaviour(self, monkeypatch):
        """No SA env block -> caller token, then the static token, then None.

        Most w2v deployments do not carry KEYCLOAK_CLIENT_SECRET yet, so this is
        the path they stay on until gitops lands it. It must not regress.
        """
        monkeypatch.setattr(lrc, "get_caller_license_token", lambda: "opaque-caller-token")
        assert await lrc._default_token() == "opaque-caller-token"

        monkeypatch.setattr(lrc, "get_caller_license_token", lambda: None)
        monkeypatch.setenv("W2V_LICENSES_SERVICE_TOKEN", "static-token")
        assert await lrc._default_token() == "static-token"

        monkeypatch.delenv("W2V_LICENSES_SERVICE_TOKEN")
        assert await lrc._default_token() is None

    async def test_partial_sa_config_does_not_mint(self, monkeypatch):
        """A half-filled env block must not produce a broken token request every
        call — it is indistinguishable from 'not rolled out here yet'."""
        monkeypatch.setenv("KEYCLOAK_URL", "https://auth.example.test")
        monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "w2v-backend")  # no secret
        lrc.reset_service_token_minter()
        monkeypatch.setattr(lrc, "get_caller_license_token", lambda: "opaque-caller-token")

        async def fake_post(self, url, data=None, **kw):
            raise AssertionError("must not attempt a mint without a client secret")

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        assert await lrc._default_token() == "opaque-caller-token"


class TestDelegationIsStillTenantScoped:
    """The gate must keep NAMING the tenant it is acting for.

    The licenses service grants the trusted-proxy retarget only for the tenant in
    ``X-Tenant-Id``; a request without it falls back to the SA's own claim and
    resolves the wrong tenant (or 401s). If this header ever stops being sent,
    presenting the SA JWT would be inert in exactly the same way the opaque token
    was — so the header is part of the fix, not an incidental detail.
    """

    async def test_get_active_license_sends_the_served_tenant_header(self, monkeypatch):
        captured: dict = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            captured["url"] = str(request.url)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="http://licenses") as http:
            client = lrc.LicensesReadClient(
                base_url="http://licenses",
                client=http,
                token_provider=lambda: _resolved("sa-jwt"),
            )
            assert await client.get_active_license(tenant_id="tenant-b", app_key="w2v") is None

        assert captured["headers"]["authorization"] == "Bearer sa-jwt"
        assert captured["headers"]["x-tenant-id"] == "tenant-b"
        assert "tenant_id=tenant-b" in captured["url"]

    async def test_list_user_licenses_sends_the_served_tenant_header(self, monkeypatch):
        captured: dict = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json=[])

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="http://licenses") as http:
            client = lrc.LicensesReadClient(
                base_url="http://licenses",
                client=http,
                token_provider=lambda: _resolved("sa-jwt"),
            )
            await client.list_user_licenses("subject-1", tenant_id="tenant-b")

        assert captured["headers"]["x-tenant-id"] == "tenant-b"


async def _resolved(value: str) -> str:
    return value


class TestServiceRealm:
    """Realm-per-app P7: the S2S token moves to KEYCLOAK_SERVICE_REALM (secure-apps)
    while KEYCLOAK_REALM (the inbound validator's primary issuer realm) stays put."""

    async def _mint_url(self, monkeypatch) -> str:
        seen: list[str] = []

        async def fake_post(self, url, data=None, **kw):
            seen.append(url)
            return httpx.Response(
                200,
                json={"access_token": "sa-jwt", "expires_in": 300},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        assert await lrc._default_token() == "sa-jwt"
        return seen[0]

    async def test_service_realm_wins_over_keycloak_realm(self, monkeypatch):
        _configure_sa(monkeypatch)
        monkeypatch.setenv("KEYCLOAK_SERVICE_REALM", "secure-apps")
        lrc.reset_service_token_minter()
        assert await self._mint_url(monkeypatch) == (
            "https://auth.example.test/realms/secure-apps/protocol/openid-connect/token"
        )

    async def test_blank_service_realm_is_keycloak_realm(self, monkeypatch):
        _configure_sa(monkeypatch)
        monkeypatch.setenv("KEYCLOAK_SERVICE_REALM", "  ")
        lrc.reset_service_token_minter()
        assert await self._mint_url(monkeypatch) == (
            "https://auth.example.test/realms/portal/protocol/openid-connect/token"
        )

    def test_service_token_realm_defaults(self, monkeypatch):
        assert lrc.service_token_realm() == "portal"
        monkeypatch.setenv("KEYCLOAK_REALM", "w2v")
        assert lrc.service_token_realm() == "w2v"
        monkeypatch.setenv("KEYCLOAK_SERVICE_REALM", "secure-apps")
        assert lrc.service_token_realm() == "secure-apps"

    def test_service_realm_does_not_move_the_validator_issuer(self, monkeypatch):
        """KEYCLOAK_SERVICE_REALM must not touch the inbound validator's issuer."""
        from w2v_common.auth import keycloak as kc

        monkeypatch.setenv("KEYCLOAK_URL", "https://auth.example.test")
        monkeypatch.setenv("KEYCLOAK_REALM", "w2v")
        monkeypatch.setenv("KEYCLOAK_SERVICE_REALM", "secure-apps")
        monkeypatch.delenv("KEYCLOAK_ISSUER", raising=False)
        monkeypatch.delenv("KEYCLOAK_TRUSTED_ISSUERS", raising=False)
        kc.reset_jwt_validator()
        try:
            v = kc.get_jwt_validator()
            assert v is not None
            assert v.issuer == "https://auth.example.test/realms/w2v"
        finally:
            kc.reset_jwt_validator()
