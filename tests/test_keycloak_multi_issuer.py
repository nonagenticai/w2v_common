"""Realm-per-app P0: multi-issuer Keycloak verification.

KEYCLOAK_TRUSTED_ISSUERS (CSV of issuer URLs, default = the single primary
issuer) and KEYCLOAK_SERVICE_ONLY_REALMS (default ``secure-apps``). Every
test here drives a REAL signature check against an in-memory keypair; the only
thing mocked is the HTTP GET of each realm's JWKS, keyed by URL so a token
verified against the wrong realm's keys fails exactly as it would live.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from w2v_common.auth import keycloak as kc
from w2v_common.auth.middleware import AuthMiddleware

BASE = "http://keycloak.example.test:8080"
PUBLIC = "https://auth.example.test"
PORTAL = f"{PUBLIC}/realms/portal"
W2V = f"{PUBLIC}/realms/w2v"
SECURE = f"{PUBLIC}/realms/secure-apps"
EVIL = f"{PUBLIC}/realms/evil"
TENANT = "0195f48b-0000-7000-8000-000000000abc"


def _b64(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8 or 1, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _keypair(kid: str) -> tuple[bytes, dict[str, Any]]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    nums = key.public_key().public_numbers()
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return pem, {
        "kty": "RSA",
        "kid": kid,
        "alg": "RS256",
        "use": "sig",
        "n": _b64(nums.n),
        "e": _b64(nums.e),
    }


@pytest.fixture(scope="module")
def realms() -> dict[str, tuple[bytes, dict[str, Any]]]:
    """One distinct keypair per realm (distinct kids, so a cross-realm key is unknown)."""
    return {name: _keypair(f"kid-{name}") for name in ("portal", "w2v", "secure-apps", "evil")}


def _jwks_url(realm: str) -> str:
    return f"{BASE}/realms/{realm}/protocol/openid-connect/certs"


@pytest.fixture
def fetched() -> list[str]:
    return []


@pytest.fixture
def jwks_http(
    realms: dict[str, tuple[bytes, dict[str, Any]]], fetched: list[str]
) -> Iterator[None]:
    """Serve each realm's JWKS at its in-cluster URL; record every fetch."""

    async def _get(url: str) -> MagicMock:
        fetched.append(url)
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        payload = {"keys": []}
        for name, (_, jwk) in realms.items():
            if url == _jwks_url(name):
                payload = {"keys": [jwk]}
        resp.json.return_value = payload
        return resp

    client = MagicMock()
    client.get = AsyncMock(side_effect=_get)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with patch("w2v_common.auth.keycloak.httpx.AsyncClient", return_value=client):
        yield


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("KEYCLOAK_TRUSTED_ISSUERS", "KEYCLOAK_SERVICE_ONLY_REALMS", "KEYCLOAK_AUDIENCES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KEYCLOAK_URL", BASE)
    monkeypatch.setenv("KEYCLOAK_REALM", "portal")
    monkeypatch.setenv("KEYCLOAK_ISSUER", PORTAL)
    kc.reset_jwt_validator()
    yield
    kc.reset_jwt_validator()


def _token(
    realms: dict[str, tuple[bytes, dict[str, Any]]], *, signer: str, iss: str, **extra: Any
) -> str:
    pem, jwk = realms[signer]
    now = int(time.time())
    claims = {
        "iss": iss,
        "sub": "user-1",
        "tenant_id": TENANT,
        "aud": "w2v-backend",
        "azp": "w2v-frontend",
        "iat": now,
        "exp": now + 300,
        "realm_access": {"roles": ["member"]},
        **extra,
    }
    return jwt.encode(claims, pem, algorithm="RS256", headers={"kid": jwk["kid"]})


# --------------------------------------------------------------------- parsing


class TestParsing:
    def test_realm_of_issuer(self) -> None:
        assert kc.realm_of_issuer(W2V) == "w2v"
        assert kc.realm_of_issuer(W2V + "/") == "w2v"
        assert kc.realm_of_issuer("https://auth.example.test") is None
        assert kc.realm_of_issuer(None) is None
        assert kc.realm_of_issuer(f"{PUBLIC}/realms/") is None

    def test_trusted_issuers_strip_dedupe(self) -> None:
        raw = f" {PORTAL}/ ,{W2V},{PORTAL},, "
        assert kc.parse_trusted_issuers(raw) == [PORTAL, W2V]
        assert kc.parse_trusted_issuers(None) == []
        assert kc.parse_trusted_issuers("") == []

    def test_service_only_default_and_explicit_empty(self) -> None:
        assert kc.parse_service_only_realms(None) == ["secure-apps"]
        assert kc.parse_service_only_realms("") == []
        assert kc.parse_service_only_realms("secure-apps, ops") == ["secure-apps", "ops"]

    def test_non_realm_trusted_issuer_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="not a Keycloak realm issuer"):
            kc.KeycloakJWTValidator(
                BASE, "portal", [], issuer=PORTAL, trusted_issuers=[PORTAL, "https://evil.test"]
            )


# ------------------------------------------------------------------- defaults


class TestDefaultIsUnchanged:
    def test_default_trusts_only_the_primary_issuer(self) -> None:
        v = kc.get_jwt_validator()
        assert v is not None
        assert v.trusted_issuers == (PORTAL,)
        # Even a secure-apps issuer is not "service-only" without explicit trust.
        assert v.is_service_only_issuer(SECURE) is False

    def test_primary_secure_apps_still_accepted_without_env(
        self, monkeypatch: pytest.MonkeyPatch, realms: Any, jwks_http: None
    ) -> None:
        """A consumer whose PRIMARY issuer is secure-apps keeps working when the env is unset."""
        monkeypatch.setenv("KEYCLOAK_REALM", "secure-apps")
        monkeypatch.setenv("KEYCLOAK_ISSUER", SECURE)
        v = kc.get_jwt_validator()
        assert v is not None
        assert v.is_service_only_issuer(SECURE) is False

    async def test_default_other_realm_token_rejected_on_issuer_pin(
        self, realms: Any, jwks_http: None, fetched: list[str]
    ) -> None:
        v = kc.get_jwt_validator()
        assert v is not None
        claims, reason = await v.validate(_token(realms, signer="w2v", iss=W2V))
        assert claims is None
        assert reason is not None
        # Default path is byte-identical: only the PRIMARY realm's JWKS is consulted.
        assert set(fetched) == {_jwks_url("portal")}

    async def test_default_primary_token_valid(self, realms: Any, jwks_http: None) -> None:
        v = kc.get_jwt_validator()
        assert v is not None
        claims, reason = await v.validate(_token(realms, signer="portal", iss=PORTAL))
        assert reason is None
        assert claims is not None and claims["iss"] == PORTAL


# --------------------------------------------------------------- multi-issuer


class TestMultiIssuer:
    @pytest.fixture(autouse=True)
    def _trusted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KEYCLOAK_TRUSTED_ISSUERS", f"{PORTAL},{W2V}/,{SECURE}")
        kc.reset_jwt_validator()

    async def test_each_issuer_verifies_against_its_own_jwks(
        self, realms: Any, jwks_http: None, fetched: list[str]
    ) -> None:
        v = kc.get_jwt_validator()
        assert v is not None
        assert v.trusted_issuers == (PORTAL, W2V, SECURE)
        for realm, iss in (("portal", PORTAL), ("w2v", W2V), ("secure-apps", SECURE)):
            claims, reason = await v.validate(_token(realms, signer=realm, iss=iss))
            assert reason is None, (realm, reason)
            assert claims is not None and claims["iss"] == iss
        assert set(fetched) == {_jwks_url("portal"), _jwks_url("w2v"), _jwks_url("secure-apps")}

    async def test_iss_is_pinned_cross_realm_key_refused(
        self, realms: Any, jwks_http: None
    ) -> None:
        """A token claiming the w2v realm but signed with the portal key never verifies."""
        v = kc.get_jwt_validator()
        assert v is not None
        claims, reason = await v.validate(_token(realms, signer="portal", iss=W2V))
        assert claims is None
        assert reason is not None and reason.startswith("unknown_kid")

    async def test_untrusted_issuer_falls_to_primary_pin(
        self, realms: Any, jwks_http: None, fetched: list[str]
    ) -> None:
        v = kc.get_jwt_validator()
        assert v is not None
        claims, reason = await v.validate(_token(realms, signer="evil", iss=EVIL))
        assert claims is None and reason is not None
        assert _jwks_url("evil") not in fetched

    async def test_primary_dropped_from_list_is_refused_without_fetch(
        self, monkeypatch: pytest.MonkeyPatch, realms: Any, jwks_http: None, fetched: list[str]
    ) -> None:
        monkeypatch.setenv("KEYCLOAK_TRUSTED_ISSUERS", W2V)
        kc.reset_jwt_validator()
        v = kc.get_jwt_validator()
        assert v is not None
        claims, reason = await v.validate(_token(realms, signer="portal", iss=PORTAL))
        assert claims is None
        assert reason == f"untrusted_issuer: {PORTAL}"
        assert fetched == []
        ok, _ = await v.validate(_token(realms, signer="w2v", iss=W2V))
        assert ok is not None

    async def test_audience_check_still_applies_to_extra_issuers(
        self, monkeypatch: pytest.MonkeyPatch, realms: Any, jwks_http: None
    ) -> None:
        monkeypatch.setenv("KEYCLOAK_AUDIENCES", "w2v-backend")
        kc.reset_jwt_validator()
        v = kc.get_jwt_validator()
        assert v is not None
        claims, reason = await v.validate(_token(realms, signer="w2v", iss=W2V, aud="other"))
        assert claims is None and reason is not None and "audience" in reason.lower()

    def test_service_only_flags(self) -> None:
        v = kc.get_jwt_validator()
        assert v is not None
        assert v.is_service_only_issuer(SECURE) is True
        assert v.is_service_only_issuer(W2V) is False

    async def test_prefetch_tolerates_a_missing_extra_realm(
        self, realms: Any, jwks_http: None
    ) -> None:
        v = kc.get_jwt_validator()
        assert v is not None
        await v.prefetch_jwks()
        assert v.is_ready()


# ----------------------------------------------------------------- middleware


async def _whoami(request: Request) -> dict[str, Any]:
    return {
        "sub": request.state.user.sub,
        "realm": request.state.user.realm,
        "state_realm": request.state.auth_realm,
    }


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.add_api_route("/api/whoami", _whoami, methods=["GET"])
    return app


async def _call(token: str) -> tuple[int, dict[str, Any]]:
    transport = ASGITransport(app=_app())
    with patch("w2v_common.auth.middleware._introspect_license", new_callable=AsyncMock) as intro:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/whoami", headers={"Authorization": f"Bearer {token}"})
        intro.assert_not_called()
    return resp.status_code, resp.json()


class TestMiddleware:
    async def test_default_exposes_portal_realm(self, realms: Any, jwks_http: None) -> None:
        status, body = await _call(_token(realms, signer="portal", iss=PORTAL))
        assert status == 200, body
        assert body == {"sub": "user-1", "realm": "portal", "state_realm": "portal"}

    async def test_app_realm_accepted_and_exposed(
        self, monkeypatch: pytest.MonkeyPatch, realms: Any, jwks_http: None
    ) -> None:
        monkeypatch.setenv("KEYCLOAK_TRUSTED_ISSUERS", f"{PORTAL},{W2V},{SECURE}")
        status, body = await _call(_token(realms, signer="w2v", iss=W2V))
        assert status == 200, body
        assert body["realm"] == "w2v" and body["state_realm"] == "w2v"

    async def test_service_only_realm_refused_as_user_session(
        self, monkeypatch: pytest.MonkeyPatch, realms: Any, jwks_http: None
    ) -> None:
        monkeypatch.setenv("KEYCLOAK_TRUSTED_ISSUERS", f"{PORTAL},{W2V},{SECURE}")
        status, body = await _call(_token(realms, signer="secure-apps", iss=SECURE))
        assert status == 401
        assert body["error"]["code"] == "JWT_SERVICE_REALM"

    async def test_service_only_realms_env_can_be_emptied(
        self, monkeypatch: pytest.MonkeyPatch, realms: Any, jwks_http: None
    ) -> None:
        monkeypatch.setenv("KEYCLOAK_TRUSTED_ISSUERS", f"{PORTAL},{SECURE}")
        monkeypatch.setenv("KEYCLOAK_SERVICE_ONLY_REALMS", "")
        status, body = await _call(_token(realms, signer="secure-apps", iss=SECURE))
        assert status == 200, body
        assert body["realm"] == "secure-apps"

    async def test_untrusted_realm_still_401(self, realms: Any, jwks_http: None) -> None:
        status, body = await _call(_token(realms, signer="w2v", iss=W2V))
        assert status == 401
        assert body["error"]["code"] == "JWT_INVALID"
