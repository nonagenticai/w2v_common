"""Tests for the Keycloak JWT branch of AuthMiddleware."""

from __future__ import annotations

import base64
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from w2v_common.auth import keycloak as kc_module
from w2v_common.auth.keycloak import _looks_like_jwt
from w2v_common.auth.middleware import AuthMiddleware

KEYCLOAK_URL = "https://auth.example.test"
REALM = "portal"
ISSUER = f"{KEYCLOAK_URL}/realms/{REALM}"
AUDIENCES = ["w2v-backend", "w2v-frontend"]
KID = "test-kid-1"


def _int_to_base64url(value: int) -> str:
    """RFC 7518 base64url encoding of a non-negative integer."""
    length = (value.bit_length() + 7) // 8 or 1
    raw = value.to_bytes(length, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


@pytest.fixture(scope="module")
def rsa_keypair() -> tuple[rsa.RSAPrivateKey, dict[str, Any]]:
    """Generate an in-memory RSA keypair and matching JWK dict for JWKS mocking."""
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    public_numbers = private_key.public_key().public_numbers()
    jwk = {
        "kty": "RSA",
        "kid": KID,
        "alg": "RS256",
        "use": "sig",
        "n": _int_to_base64url(public_numbers.n),
        "e": _int_to_base64url(public_numbers.e),
    }
    return private_key, jwk


@pytest.fixture(autouse=True)
def reset_module_state() -> Iterator[None]:
    """Reset cached validator + license cache between tests."""
    import w2v_common.auth.middleware as mw_mod

    kc_module.reset_jwt_validator()
    mw_mod._license_cache = None
    mw_mod._license_neg_cache = None
    mw_mod._license_stale_cache = None
    yield
    kc_module.reset_jwt_validator()
    mw_mod._license_cache = None
    mw_mod._license_neg_cache = None
    mw_mod._license_stale_cache = None


@pytest.fixture
def enable_keycloak(
    monkeypatch: pytest.MonkeyPatch, rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]]
) -> dict[str, Any]:
    """Configure env + mock JWKS endpoint to return our in-memory public key."""
    _, jwk = rsa_keypair
    monkeypatch.setenv("KEYCLOAK_URL", KEYCLOAK_URL)
    monkeypatch.setenv("KEYCLOAK_REALM", REALM)
    monkeypatch.setenv("KEYCLOAK_AUDIENCES", ",".join(AUDIENCES))
    return {"keys": [jwk]}


def _make_jwks_mock_client(jwks_payload: dict[str, Any]) -> MagicMock:
    """httpx.AsyncClient mock whose .get() returns the JWKS payload."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = jwks_payload
    resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


def _sign_token(private_key: rsa.RSAPrivateKey, *, claims: dict[str, Any], kid: str = KID) -> str:
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return jwt.encode(claims, pem, algorithm="RS256", headers={"kid": kid})


async def _data_endpoint(request: Request) -> dict[str, Any]:
    user = request.state.user
    return {
        "sub": user.sub,
        "email": user.email,
        "roles": user.roles,
        "tenant_id": request.state.tenant_id,
        "license": request.state.license,
    }


async def _state_endpoint(request: Request) -> dict[str, Any]:
    """Reads the FLAT identity claims off request.state (the /auth/me contract)."""
    return {
        "roles": getattr(request.state, "roles", None),
        "email": getattr(request.state, "email", None),
        "name": getattr(request.state, "name", None),
    }


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.add_api_route("/api/data", _data_endpoint, methods=["GET"])
    app.add_api_route("/api/state", _state_endpoint, methods=["GET"])
    return app


def _default_claims() -> dict[str, Any]:
    now = int(time.time())
    return {
        "iss": ISSUER,
        "aud": "w2v-backend",
        "sub": "user-123",
        # Every one of the 29 real portal users carries a non-empty tenant_id
        # attribute, injected by the `tenant` client scope's
        # oidc-usermodel-attribute-mapper. The default fixture models a REAL
        # token, so the tenant is deliberately DIFFERENT from `sub` — a test
        # where they coincide cannot tell "read the claim" apart from
        # "fell back to sub".
        "tenant_id": "0195f48b-0000-7000-8000-000000000abc",
        "preferred_username": "alice",
        "email": "alice@example.test",
        "realm_access": {"roles": ["admin", "user"]},
        "iat": now,
        "exp": now + 300,
    }


class TestLooksLikeJwt:
    def test_three_segments(self) -> None:
        assert _looks_like_jwt("aaa.bbb.ccc") is True

    def test_two_segments_is_not_jwt(self) -> None:
        assert _looks_like_jwt("aaa.bbb") is False

    def test_empty_segment_is_not_jwt(self) -> None:
        assert _looks_like_jwt("aaa..ccc") is False

    def test_empty_string(self) -> None:
        assert _looks_like_jwt("") is False

    def test_non_jwt_license_token(self) -> None:
        # License tokens are opaque strings without dots.
        assert _looks_like_jwt("license_tok_abc123xyz") is False


class TestValidatorDisabledByDefault:
    def test_returns_none_when_keycloak_url_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KEYCLOAK_URL", raising=False)
        assert kc_module.get_jwt_validator() is None


class TestKeycloakJwtBranch:
    async def test_valid_jwt_populates_user_and_skips_license(
        self,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],
        enable_keycloak: dict[str, Any],
    ) -> None:
        private_key, _ = rsa_keypair
        token = _sign_token(private_key, claims=_default_claims())

        mock_httpx = _make_jwks_mock_client(enable_keycloak)
        app = _make_app()
        transport = ASGITransport(app=app)

        with (
            patch("w2v_common.auth.keycloak.httpx.AsyncClient", return_value=mock_httpx),
            patch(
                "w2v_common.auth.middleware._introspect_license", new_callable=AsyncMock
            ) as mock_introspect,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/data", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["sub"] == "user-123"
        assert body["email"] == "alice@example.test"
        assert body["roles"] == ["admin", "user"]
        assert body["tenant_id"] == "0195f48b-0000-7000-8000-000000000abc"
        assert body["tenant_id"] != body["sub"]
        assert body["license"] is None
        mock_introspect.assert_not_called()

    async def test_expired_jwt_hard_fails_with_reason(
        self,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],
        enable_keycloak: dict[str, Any],
    ) -> None:
        """JWT-shaped but invalid → hard-fail 401 with the reason in the body.

        Rationale: license introspection would just re-reject a Keycloak JWT
        with "invalid_token", masking the real cause. Hard-failing here and
        surfacing the reason is what lets an operator debug issuer/kid/
        signature/expiry problems in prod.
        """
        private_key, _ = rsa_keypair
        now = int(time.time())
        claims = _default_claims()
        claims["iat"] = now - 3600
        claims["exp"] = now - 60
        token = _sign_token(private_key, claims=claims)

        mock_httpx = _make_jwks_mock_client(enable_keycloak)
        app = _make_app()
        transport = ASGITransport(app=app)

        with (
            patch("w2v_common.auth.keycloak.httpx.AsyncClient", return_value=mock_httpx),
            patch(
                "w2v_common.auth.middleware._introspect_license",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_introspect,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/data", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["code"] == "JWT_INVALID"
        assert "JWT validation failed" in body["error"]["message"]
        assert "expired" in body["error"]["message"].lower()
        mock_introspect.assert_not_called()

    async def test_wrong_audience_hard_fails_with_reason(
        self,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],
        enable_keycloak: dict[str, Any],
    ) -> None:
        """When KEYCLOAK_AUDIENCES is configured and aud mismatches → hard 401.

        Note: production w2v services leave KEYCLOAK_AUDIENCES unset, which
        disables aud checking entirely (matches fr/jose issuer-only pattern).
        This test runs with audiences explicitly configured via the
        _enable_keycloak fixture to exercise the enforcement path.
        """
        private_key, _ = rsa_keypair
        claims = _default_claims()
        claims["aud"] = "some-other-client"
        token = _sign_token(private_key, claims=claims)

        mock_httpx = _make_jwks_mock_client(enable_keycloak)
        app = _make_app()
        transport = ASGITransport(app=app)

        with (
            patch("w2v_common.auth.keycloak.httpx.AsyncClient", return_value=mock_httpx),
            patch(
                "w2v_common.auth.middleware._introspect_license",
                new_callable=AsyncMock,
            ) as mock_introspect,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/data", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["code"] == "JWT_INVALID"
        assert "JWT validation failed" in body["error"]["message"]
        assert "audience" in body["error"]["message"].lower()
        mock_introspect.assert_not_called()

    async def test_jwt_shaped_unknown_kid_hard_fails_with_reason(
        self, enable_keycloak: dict[str, Any]
    ) -> None:
        """Three-segment token with a kid Keycloak's JWKS doesn't know → 401.

        This covers the common mis-signed-token case (wrong realm, rotated
        key that never reached this JWKS cache, or a forged token).
        """
        # Sanity-check the fixture: keycloak env is configured and the JWKS
        # payload it built contains our test KID (we then deliberately mock
        # the JWKS endpoint with an EMPTY key set below to trigger 'unknown kid').
        assert [k["kid"] for k in enable_keycloak["keys"]] == [KID]
        # Three dot-separated base64 segments but not a real signed token.
        junk = (
            base64.urlsafe_b64encode(b'{"alg":"RS256","kid":"nope"}').rstrip(b"=").decode()
            + "."
            + base64.urlsafe_b64encode(b'{"iss":"x"}').rstrip(b"=").decode()
            + ".xxxxxxxx"
        )
        mock_httpx = _make_jwks_mock_client({"keys": []})
        app = _make_app()
        transport = ASGITransport(app=app)

        with (
            patch("w2v_common.auth.keycloak.httpx.AsyncClient", return_value=mock_httpx),
            patch(
                "w2v_common.auth.middleware._introspect_license",
                new_callable=AsyncMock,
            ) as mock_introspect,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/data", headers={"Authorization": f"Bearer {junk}"})
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["code"] == "JWT_INVALID"
        assert "JWT validation failed" in body["error"]["message"]
        # Reason surfaces the specific problem (unknown kid)
        assert "kid" in body["error"]["message"].lower()
        mock_introspect.assert_not_called()

    async def test_opaque_license_token_via_x_tenant_id_goes_to_introspection(
        self, enable_keycloak: dict[str, Any]
    ) -> None:
        """License tokens travel in `X-Tenant-Id: Bearer <token>` and reach
        license introspection without touching the JWT validator.

        2026-04-26 change: license tokens via `Authorization: Bearer` are no
        longer accepted (that fallback was misreading FR's Keycloak JWT as a
        license token and 401'ing).
        """
        # Fixture configured Keycloak env even though this test exercises the
        # opaque-license branch; assert the JWKS payload is well-formed so a
        # silent regression in the fixture would surface here.
        assert enable_keycloak["keys"][0]["kty"] == "RSA"
        app = _make_app()
        transport = ASGITransport(app=app)

        with (
            patch("w2v_common.auth.keycloak.httpx.AsyncClient") as mock_httpx_cls,
            patch(
                "w2v_common.auth.middleware._introspect_license",
                new_callable=AsyncMock,
                return_value={
                    "active": True,
                    "tenant_id": "0195f48b-0000-7000-8000-0000000001ac",
                    "subject_id": "lic-tenant",
                },
            ) as mock_introspect,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/data",
                    headers={"X-Tenant-Id": "Bearer license_tok_opaque_no_dots"},
                )
        assert resp.status_code == 200
        body = resp.json()
        assert body["tenant_id"] == "0195f48b-0000-7000-8000-0000000001ac"
        assert body["license"] is not None
        mock_introspect.assert_awaited_once()
        # JWKS client should never have been constructed for a non-JWT token.
        mock_httpx_cls.assert_not_called()

    async def test_opaque_token_in_authorization_is_introspected_not_jwt_validated(
        self, enable_keycloak: dict[str, Any]
    ) -> None:
        """CONTRACT CHANGED (platform licensing): an opaque bearer in
        ``Authorization`` is a LICENCE token and goes to introspection — even
        with Keycloak fully configured.

        What this test still guards, and why it keeps the JWKS assertion: an
        opaque bearer must never touch the Keycloak validator, and (sibling
        test) a JWT-shaped bearer must never touch licence introspection. The
        two credential kinds are told apart BY SHAPE before either lane runs —
        that separation is the 2026-04-26 fix and it is intact. What changed is
        only the disposition of the non-JWT branch: previously a dead end (401),
        now the platform's licence lane, because ``license-authz`` reads the
        licence token from ``Authorization`` only.
        """
        # Fixture sets KEYCLOAK_URL etc; verify the JWKS dict it returned is
        # the in-memory single-key set we expect for this test session.
        assert len(enable_keycloak["keys"]) == 1
        assert enable_keycloak["keys"][0]["alg"] == "RS256"
        app = _make_app()
        transport = ASGITransport(app=app)

        with (
            patch("w2v_common.auth.keycloak.httpx.AsyncClient") as mock_httpx_cls,
            patch(
                "w2v_common.auth.middleware._introspect_license",
                new_callable=AsyncMock,
                return_value={
                    "active": True,
                    "tenant_id": "0195f48b-0000-7000-8000-000000000abc",
                    "subject_id": "user-1",
                },
            ) as mock_introspect,
            patch(
                "w2v_common.auth.middleware._is_trusted_network",
                return_value=False,
            ),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/data",
                    headers={"Authorization": "Bearer opaque_license_token"},
                )
        assert resp.status_code == 200, resp.text
        assert resp.json()["tenant_id"] == "0195f48b-0000-7000-8000-000000000abc"
        mock_introspect.assert_awaited_once_with("opaque_license_token")
        # The Keycloak validator was never consulted: an opaque bearer is not
        # JWT-shaped, so it must not cause a JWKS fetch.
        mock_httpx_cls.assert_not_called()

    async def test_empty_audiences_accepts_any_aud_matching_fr_behaviour(
        self,
        monkeypatch: pytest.MonkeyPatch,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],
    ) -> None:
        """With KEYCLOAK_AUDIENCES unset, aud is not validated (matches fr/jose behaviour).

        Why: service-account tokens from Keycloak default to ``aud: account`` and
        fr-backend (jose) validates only ``iss``. Our middleware mirrors that when
        the operator hasn't configured an explicit audience list.
        """
        monkeypatch.setenv("KEYCLOAK_URL", KEYCLOAK_URL)
        monkeypatch.setenv("KEYCLOAK_REALM", REALM)
        monkeypatch.delenv("KEYCLOAK_AUDIENCES", raising=False)
        private_key, jwk = rsa_keypair
        claims = _default_claims()
        claims["aud"] = "account"  # Keycloak default for client-credentials tokens
        token = _sign_token(private_key, claims=claims)

        mock_httpx = _make_jwks_mock_client({"keys": [jwk]})
        app = _make_app()
        transport = ASGITransport(app=app)

        with (
            patch("w2v_common.auth.keycloak.httpx.AsyncClient", return_value=mock_httpx),
            patch(
                "w2v_common.auth.middleware._introspect_license", new_callable=AsyncMock
            ) as mock_introspect,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/data", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["sub"] == "user-123"
        assert body["email"] == "alice@example.test"
        assert body["license"] is None
        mock_introspect.assert_not_called()

    async def test_keycloak_issuer_override_decouples_from_keycloak_url(
        self,
        monkeypatch: pytest.MonkeyPatch,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],
    ) -> None:
        """KEYCLOAK_ISSUER overrides the ``iss`` value the validator expects.

        Lets KEYCLOAK_URL be an in-cluster Service URL (where JWKS is fetched)
        while ``iss`` verification still uses the public URL embedded in tokens.
        """
        internal_url = "http://keycloak.keycloak:8080"
        public_issuer = f"{KEYCLOAK_URL}/realms/{REALM}"  # https://auth.example.test/realms/portal
        monkeypatch.setenv("KEYCLOAK_URL", internal_url)
        monkeypatch.setenv("KEYCLOAK_REALM", REALM)
        monkeypatch.setenv("KEYCLOAK_ISSUER", public_issuer)
        monkeypatch.delenv("KEYCLOAK_AUDIENCES", raising=False)

        private_key, jwk = rsa_keypair
        # Token is minted with the PUBLIC issuer (as a real Keycloak would do).
        claims = _default_claims()
        claims["iss"] = public_issuer
        claims["aud"] = "account"
        token = _sign_token(private_key, claims=claims)

        mock_httpx = _make_jwks_mock_client({"keys": [jwk]})
        app = _make_app()
        transport = ASGITransport(app=app)

        with (
            patch("w2v_common.auth.keycloak.httpx.AsyncClient", return_value=mock_httpx),
            patch(
                "w2v_common.auth.middleware._introspect_license", new_callable=AsyncMock
            ) as mock_introspect,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/data", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["sub"] == "user-123"
        # JWKS was fetched from the INTERNAL URL.
        mock_httpx.get.assert_awaited()
        called_url = mock_httpx.get.await_args.args[0]
        assert called_url.startswith(internal_url), called_url
        assert "/realms/portal/protocol/openid-connect/certs" in called_url
        mock_introspect.assert_not_called()

    async def test_keycloak_issuer_defaults_to_keycloak_url_when_unset(
        self,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],
        enable_keycloak: dict[str, Any],
    ) -> None:
        """Backward compat: with KEYCLOAK_ISSUER unset, validator behaves identically.

        Token issuer = ``${KEYCLOAK_URL}/realms/${KEYCLOAK_REALM}`` still verifies.
        """
        private_key, _ = rsa_keypair
        token = _sign_token(private_key, claims=_default_claims())

        mock_httpx = _make_jwks_mock_client(enable_keycloak)
        app = _make_app()
        transport = ASGITransport(app=app)

        with (
            patch("w2v_common.auth.keycloak.httpx.AsyncClient", return_value=mock_httpx),
            patch("w2v_common.auth.middleware._introspect_license", new_callable=AsyncMock),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/data", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    async def test_frontend_jwt_with_plain_x_tenant_id_marker_succeeds(
        self,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],
        enable_keycloak: dict[str, Any],
    ) -> None:
        """Regression: frontend sends `Authorization: Bearer <jwt>` together
        with a plain (non-bearer-prefixed) `x-tenant-id: <tenant>` header.

        Contract:
          * The JWT branch wins. tenant_id is taken from the JWT's
            ``tenant_id`` claim, NOT from the x-tenant-id header.
          * The plain x-tenant-id header is treated as a no-op marker (the
            license-extraction path requires the literal `Bearer ` prefix on
            X-Tenant-Id, so a plain value is ignored — never submitted to
            license introspection).

        See frontend/src/lib/api/client.ts: defaultHeaders sets
        `'x-tenant-id': VITE_TENANT_ID` (plain) and per-request injects
        `Authorization: Bearer <kc_token>`. This test locks that combo so
        it can never silently break.
        """
        private_key, _ = rsa_keypair
        token = _sign_token(private_key, claims=_default_claims())

        mock_httpx = _make_jwks_mock_client(enable_keycloak)
        app = _make_app()
        transport = ASGITransport(app=app)

        with (
            patch("w2v_common.auth.keycloak.httpx.AsyncClient", return_value=mock_httpx),
            patch(
                "w2v_common.auth.middleware._introspect_license", new_callable=AsyncMock
            ) as mock_introspect,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/data",
                    headers={
                        "Authorization": f"Bearer {token}",
                        # Frontend sends this as a plain marker — NO 'Bearer ' prefix.
                        # Mixed-case header name is intentional: matches what the
                        # browser actually emits (lowercase via fetch defaults).
                        "x-tenant-id": "frontend-tenant-marker",
                    },
                )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # tenant_id MUST come from the JWT's tenant_id claim, not from the
        # plain x-tenant-id header value and not from `sub`.
        assert body["sub"] == "user-123"
        assert body["tenant_id"] == "0195f48b-0000-7000-8000-000000000abc"
        assert body["tenant_id"] != "frontend-tenant-marker"
        assert body["tenant_id"] != body["sub"]
        assert body["license"] is None
        # License introspection MUST NOT have been called — the plain
        # x-tenant-id value is not a license token candidate.
        mock_introspect.assert_not_called()

    async def test_plain_x_tenant_id_alone_without_jwt_is_unauthenticated(
        self, enable_keycloak: dict[str, Any]
    ) -> None:
        """Regression: a request with ONLY a plain (non-bearer) x-tenant-id
        and no Authorization header must be rejected as unauthenticated.

        This guards against a regression where the middleware might
        accidentally treat a plain x-tenant-id as proof of identity. The
        frontend's plain marker is harmless precisely BECAUSE the JWT in
        Authorization is what carries identity; without the JWT the request
        carries no credential at all.
        """
        # Even though this test sends NO Authorization header, the fixture
        # has Keycloak configured — assert that so a regression where the
        # fixture stops setting env (which would mask the real assertion
        # path under "validator unconfigured") is caught immediately.
        assert enable_keycloak["keys"][0]["use"] == "sig"
        app = _make_app()
        transport = ASGITransport(app=app)

        with (
            patch("w2v_common.auth.keycloak.httpx.AsyncClient") as mock_httpx_cls,
            patch(
                "w2v_common.auth.middleware._introspect_license", new_callable=AsyncMock
            ) as mock_introspect,
            patch(
                "w2v_common.auth.middleware._is_trusted_network",
                return_value=False,
            ),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/data",
                    headers={"x-tenant-id": "frontend-tenant-marker"},
                )
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["code"] == "AUTHENTICATION_REQUIRED"
        assert "Missing authentication token" in body["error"]["message"]
        mock_introspect.assert_not_awaited()
        mock_httpx_cls.assert_not_called()

    async def test_jwt_in_authorization_rejected_when_keycloak_url_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],
    ) -> None:
        """A JWT-shaped token in Authorization with no Keycloak validator must
        be hard-rejected, NOT submitted to license introspection.

        Pre-2026-04-26 behaviour fell through to license introspection — that
        was the bug that surfaced as FR's prod 401s on /api/assets/procedures
        and /api/assets/secrets (a Keycloak JWT was being submitted to the
        license endpoint, which returned token_not_found).
        """
        monkeypatch.delenv("KEYCLOAK_URL", raising=False)
        private_key, _ = rsa_keypair
        token = _sign_token(private_key, claims=_default_claims())

        app = _make_app()
        transport = ASGITransport(app=app)

        with (
            patch("w2v_common.auth.keycloak.httpx.AsyncClient") as mock_httpx_cls,
            patch(
                "w2v_common.auth.middleware._introspect_license",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_introspect,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/data", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["code"] == "JWT_VALIDATOR_UNAVAILABLE"
        assert "Keycloak validator is not configured" in body["error"]["message"]
        mock_introspect.assert_not_awaited()
        mock_httpx_cls.assert_not_called()


class TestRequestStateIdentityForwarding:
    """AuthMiddleware must stash roles/email/name as FLAT request.state attrs.

    Without this, request.state.roles is never set and every consumer that
    reads it (orchestrator /auth/me, future role guards) silently sees []
    even when the JWT carries realm roles. Regression guard for Phase A.
    """

    async def test_jwt_path_forwards_roles_email_name_to_state(
        self,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],
        enable_keycloak: dict[str, Any],
    ) -> None:
        private_key, _ = rsa_keypair
        token = _sign_token(private_key, claims=_default_claims())

        mock_httpx = _make_jwks_mock_client(enable_keycloak)
        app = _make_app()
        transport = ASGITransport(app=app)

        with (
            patch("w2v_common.auth.keycloak.httpx.AsyncClient", return_value=mock_httpx),
            patch("w2v_common.auth.middleware._introspect_license", new_callable=AsyncMock),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/state", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["roles"] == ["admin", "user"]
        assert body["email"] == "alice@example.test"
        assert body["name"] == "alice"

    async def test_license_path_forwards_empty_roles_to_state(
        self, enable_keycloak: dict[str, Any]
    ) -> None:
        """The opaque-license path has no realm roles → roles=[] (empty-safe)."""
        app = _make_app()
        transport = ASGITransport(app=app)

        with (
            patch("w2v_common.auth.keycloak.httpx.AsyncClient"),
            patch(
                "w2v_common.auth.middleware._introspect_license",
                new_callable=AsyncMock,
                return_value={
                    "active": True,
                    "tenant_id": "0195f48b-0000-7000-8000-0000000001ac",
                    "subject_id": "lic-subject",
                },
            ),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/state",
                    headers={"X-Tenant-Id": "Bearer license_tok_opaque_no_dots"},
                )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["roles"] == []
        assert body["email"] is None
        # preferred_username on the license path is the subject_id.
        assert body["name"] == "lic-subject"


class TestJwtLaneRequiresATenantClaim:
    """The JWT lane is fail-CLOSED on a missing/blank ``tenant_id`` claim.

    Before this class existed, ``tenant_id = claims.get("tenant_id") or sub``
    authenticated a tenant-less token as a SELF-NAMED tenant (tenant == sub).
    That is reachable in prod whenever ``KEYCLOAK_AUDIENCES`` does not pin the
    accepted clients: eight portal clients (``admin-cli``,
    ``realm-management``, ``account``, ...) lack the ``tenant`` client scope
    and therefore mint tokens with no ``tenant_id`` claim at all. The
    orchestrator's ``/api/auth/me`` WRITES on this path, so the fallback
    minted a ``tenant_members`` row for a tenant nobody provisioned.

    These tests assert BEHAVIOUR: whether the request reaches the application
    at all, and which tenant the application is told to act as.
    """

    @staticmethod
    def _recording_app() -> tuple[FastAPI, list[str | None]]:
        """App whose handler records the tenant it was actually invoked with.

        An empty list after a request is positive proof the request never
        reached application code — i.e. nothing downstream could have written
        a row for a self-named tenant.
        """
        seen: list[str | None] = []

        async def _record(request: Request) -> dict[str, Any]:
            tenant = getattr(request.state, "tenant_id", None)
            seen.append(tenant)
            return {"tenant_id": tenant}

        app = FastAPI()
        app.add_middleware(AuthMiddleware)
        app.add_api_route("/api/data", _record, methods=["GET"])
        return app, seen

    async def _call(
        self,
        private_key: rsa.RSAPrivateKey,
        jwks: dict[str, Any],
        claims: dict[str, Any],
    ) -> tuple[int, dict[str, Any], list[str | None], AsyncMock]:
        token = _sign_token(private_key, claims=claims)
        mock_httpx = _make_jwks_mock_client(jwks)
        app, seen = self._recording_app()
        transport = ASGITransport(app=app)
        with (
            patch("w2v_common.auth.keycloak.httpx.AsyncClient", return_value=mock_httpx),
            patch(
                "w2v_common.auth.middleware._introspect_license", new_callable=AsyncMock
            ) as mock_introspect,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/data", headers={"Authorization": f"Bearer {token}"})
        return resp.status_code, resp.json(), seen, mock_introspect

    async def test_jwt_with_tenant_claim_authenticates_as_that_tenant(
        self,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],
        enable_keycloak: dict[str, Any],
    ) -> None:
        """No regression for the 29 real portal users: a token WITH the claim
        still authenticates, and the app acts as the CLAIMED tenant."""
        private_key, _ = rsa_keypair
        claims = _default_claims()
        claims["tenant_id"] = "0198f1d2-0000-7000-8000-00000000abcd"

        status, body, seen, mock_introspect = await self._call(private_key, enable_keycloak, claims)

        assert status == 200, body
        # The handler ran, and it ran as the CLAIMED tenant — not as `sub`.
        assert seen == ["0198f1d2-0000-7000-8000-00000000abcd"]
        assert body["tenant_id"] == "0198f1d2-0000-7000-8000-00000000abcd"
        assert body["tenant_id"] != claims["sub"]
        mock_introspect.assert_not_called()

    async def test_jwt_without_tenant_claim_is_rejected_not_self_named(
        self,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],
        enable_keycloak: dict[str, Any],
    ) -> None:
        """A token from an off-scope client (no ``tenant_id`` claim) is 401.

        This is the exact shape an ``admin-cli`` / ``realm-management`` token
        has: valid signature, valid issuer, no tenant attribute mapper.
        """
        private_key, _ = rsa_keypair
        claims = _default_claims()
        del claims["tenant_id"]
        claims["azp"] = "admin-cli"

        status, body, seen, mock_introspect = await self._call(private_key, enable_keycloak, claims)

        assert status == 401, body
        # The decisive behaviour: application code was NEVER reached, so
        # nothing could have minted a tenant_members row for "user-123".
        assert seen == []
        assert body["error"]["code"] == "JWT_INVALID"
        assert "tenant" in body["error"]["message"].lower()
        # Same envelope as the license lane's empty-tenant rejection.
        assert body["error"]["status"] == 401
        assert body["error"]["request_id"]
        # A JWT is never handed to license introspection.
        mock_introspect.assert_not_called()

    @pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
    async def test_blank_tenant_claim_is_rejected(
        self,
        blank: str,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],
        enable_keycloak: dict[str, Any],
    ) -> None:
        """A present-but-blank claim must not authenticate either.

        ``"   "`` is truthy, so a bare ``or sub`` fallback would not even fire
        — it would authenticate tenant_id="   ", which the RLS GUC treats as
        unset (``NULLIF(current_setting(...), '')``) only after stripping.
        """
        private_key, _ = rsa_keypair
        claims = _default_claims()
        claims["tenant_id"] = blank

        status, body, seen, _ = await self._call(private_key, enable_keycloak, claims)

        assert status == 401, body
        assert seen == []
        assert body["error"]["code"] == "JWT_INVALID"

    async def test_rejection_leaves_no_identity_context_behind(
        self,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],
        enable_keycloak: dict[str, Any],
    ) -> None:
        """The reject path must not leak a half-set identity to the next request.

        The success path sets the identity ContextVars and clears them in a
        ``finally``; the reject path returns BEFORE setting them, so the
        outbound S2S headers must be empty afterwards.
        """
        from w2v_common.auth.identity_context import (
            clear_identity_context,
            get_identity_headers,
            get_tenant_id_from_context,
        )
        from w2v_common.tenant import MissingTenantError

        clear_identity_context()
        private_key, _ = rsa_keypair
        claims = _default_claims()
        del claims["tenant_id"]

        status, _body, seen, _ = await self._call(private_key, enable_keycloak, claims)

        assert status == 401
        assert seen == []
        # A rejected caller must not be able to have its identity stamped on
        # any outbound S2S call. CONTRACT CHANGE: this used to assert
        # get_identity_headers() returned BLANK values. Blank is a sentinel;
        # the header builder now REFUSES to produce a dict at all when no
        # tenant is bound, which is a stronger form of the same guarantee.
        with pytest.raises(MissingTenantError):
            get_identity_headers()
        assert get_tenant_id_from_context() is None
