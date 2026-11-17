"""P2 tenancy (WP2): JWT tenant opt-out, ``email_verified``, tenant API keys.

Every test here fails on 0.89.0:
- ``AUTH_JWT_TENANT_REQUIRED`` did not exist, so a claim-less JWT was always 401;
- ``email_verified`` was dropped on the floor;
- a ``w2v_sk_`` bearer went to licence introspection;
- ``TenantSource.MEMBERSHIP`` / ``TENANT_API_KEY`` did not exist.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from w2v_common.auth import (
    TENANT_API_KEY_PREFIX,
    build_downstream_tenant_headers,
    get_api_key_verifier,
    register_api_key_verifier,
)
from w2v_common.auth.identity_context import (
    EFFECTIVE_TENANT_HEADER,
    clear_identity_context,
    get_caller_license_token,
    get_subject_id_from_context,
    get_tenant_id_from_context,
    get_tenant_source_from_context,
    identity_event_hook,
    is_inbound_request,
)
from w2v_common.auth.middleware import AuthMiddleware
from w2v_common.middleware.audit import DEFAULT_AUDIT_LOGGER, AuditMiddleware
from w2v_common.tenant import VOUCHED_TENANT_SOURCES, MissingTenantError, TenantSource

from .test_auth_middleware_jwt import (
    _default_claims,
    _make_jwks_mock_client,
    _sign_token,
    enable_keycloak,  # noqa: F401 — pytest fixture
    reset_module_state,  # noqa: F401 — autouse pytest fixture
    rsa_keypair,  # noqa: F401 — pytest fixture
)

KEY_TENANT = uuid.UUID("0198f1d2-0000-7000-8000-0000000a91e1")
API_KEY = f"{TENANT_API_KEY_PREFIX}ABCDEFGH_c2VjcmV0LXNlY3JldC1zZWNyZXQtc2VjcmV0"


@pytest.fixture(autouse=True)
def _no_verifier() -> Iterator[None]:
    register_api_key_verifier(None)
    yield
    register_api_key_verifier(None)


async def _identity(request: Request) -> dict[str, Any]:
    """Echo what the middleware stashed AND what it put in the ContextVars."""
    user = request.state.user
    ctx_tenant = get_tenant_id_from_context()
    ctx_source = get_tenant_source_from_context()
    return {
        "state_tenant_id": request.state.tenant_id,
        "user_tenant_id": str(user.tenant_id) if user.tenant_id else None,
        "sub": user.sub,
        "email": request.state.email,
        "email_verified": request.state.email_verified,
        "user_email_verified": user.email_verified,
        "api_key_id": request.state.api_key_id,
        "license_token": request.state.license_token,
        "ctx_tenant": str(ctx_tenant) if ctx_tenant else None,
        "ctx_source": ctx_source.value if ctx_source else None,
        "ctx_caller_token": get_caller_license_token(),
        "ctx_subject": get_subject_id_from_context(),
    }


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.add_api_route("/api/whoami", _identity, methods=["GET"])
    return app


async def _jwt_call(
    private_key: rsa.RSAPrivateKey, jwks: dict[str, Any], claims: dict[str, Any]
) -> tuple[int, dict[str, Any], AsyncMock]:
    token = _sign_token(private_key, claims=claims)
    with (
        patch(
            "w2v_common.auth.keycloak.httpx.AsyncClient",
            return_value=_make_jwks_mock_client(jwks),
        ),
        patch(
            "w2v_common.auth.middleware._introspect_license", new_callable=AsyncMock
        ) as introspect,
    ):
        async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://t") as c:
            resp = await c.get("/api/whoami", headers={"Authorization": f"Bearer {token}"})
    return resp.status_code, resp.json(), introspect


async def _headers_call(headers: dict[str, str]) -> tuple[int, dict[str, Any], AsyncMock]:
    with patch(
        "w2v_common.auth.middleware._introspect_license", new_callable=AsyncMock, return_value=None
    ) as introspect:
        async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://t") as c:
            resp = await c.get("/api/whoami", headers=headers)
    return resp.status_code, resp.json(), introspect


# --------------------------------------------------------------------------- #
# AUTH_JWT_TENANT_REQUIRED
# --------------------------------------------------------------------------- #


class TestJwtTenantRequired:
    @pytest.mark.parametrize("value", [None, "true", "1", "", "garbage"])
    async def test_default_and_unrecognised_values_still_reject_a_claimless_jwt(
        self,
        value: str | None,
        monkeypatch: pytest.MonkeyPatch,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],  # noqa: F811
        enable_keycloak: dict[str, Any],  # noqa: F811
    ) -> None:
        if value is None:
            monkeypatch.delenv("AUTH_JWT_TENANT_REQUIRED", raising=False)
        else:
            monkeypatch.setenv("AUTH_JWT_TENANT_REQUIRED", value)
        claims = _default_claims()
        del claims["tenant_id"]

        status, body, introspect = await _jwt_call(rsa_keypair[0], enable_keycloak, claims)

        assert status == 401, body
        assert body["error"]["code"] == "JWT_INVALID"
        introspect.assert_not_called()

    @pytest.mark.parametrize("value", ["false", "0", "no", "OFF"])
    @pytest.mark.parametrize("absent", ["missing", "", "   "])
    async def test_opt_out_yields_a_tenantless_identity(
        self,
        value: str,
        absent: str,
        monkeypatch: pytest.MonkeyPatch,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],  # noqa: F811
        enable_keycloak: dict[str, Any],  # noqa: F811
    ) -> None:
        monkeypatch.setenv("AUTH_JWT_TENANT_REQUIRED", value)
        claims = _default_claims()
        if absent == "missing":
            del claims["tenant_id"]
        else:
            claims["tenant_id"] = absent
        claims["email_verified"] = True

        status, body, introspect = await _jwt_call(rsa_keypair[0], enable_keycloak, claims)

        assert status == 200, body
        # Authenticated as the subject, with NO tenant anywhere — not a
        # fallback to sub, not a sentinel.
        assert body["sub"] == "user-123"
        assert body["state_tenant_id"] is None
        assert body["user_tenant_id"] is None
        assert body["ctx_tenant"] is None
        # No tenant → no provenance; the membership middleware sets both.
        assert body["ctx_source"] is None
        assert body["email"] == "alice@example.test"
        assert body["email_verified"] is True
        introspect.assert_not_called()

    async def test_opt_out_still_rejects_a_present_but_invalid_claim(
        self,
        monkeypatch: pytest.MonkeyPatch,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],  # noqa: F811
        enable_keycloak: dict[str, Any],  # noqa: F811
    ) -> None:
        monkeypatch.setenv("AUTH_JWT_TENANT_REQUIRED", "false")
        claims = _default_claims()
        claims["tenant_id"] = "n8n-dev"

        status, body, _ = await _jwt_call(rsa_keypair[0], enable_keycloak, claims)

        assert status == 401, body
        assert body["error"]["code"] == "JWT_INVALID"

    async def test_opt_out_with_a_claim_is_the_claim_lane(
        self,
        monkeypatch: pytest.MonkeyPatch,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],  # noqa: F811
        enable_keycloak: dict[str, Any],  # noqa: F811
    ) -> None:
        monkeypatch.setenv("AUTH_JWT_TENANT_REQUIRED", "false")
        claims = _default_claims()

        status, body, _ = await _jwt_call(rsa_keypair[0], enable_keycloak, claims)

        assert status == 200, body
        assert body["state_tenant_id"] == claims["tenant_id"]
        assert body["ctx_tenant"] == claims["tenant_id"]
        assert body["ctx_source"] == TenantSource.JWT_CLAIM.value


class TestEmailVerified:
    @pytest.mark.parametrize(
        ("claim", "expected"),
        [(True, True), (False, False), ("true", False), (1, False), ("missing", False)],
    )
    async def test_email_verified_is_true_only_for_a_literal_true_claim(
        self,
        claim: object,
        expected: bool,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],  # noqa: F811
        enable_keycloak: dict[str, Any],  # noqa: F811
    ) -> None:
        claims = _default_claims()
        if claim != "missing":
            claims["email_verified"] = claim

        status, body, _ = await _jwt_call(rsa_keypair[0], enable_keycloak, claims)

        assert status == 200, body
        assert body["email_verified"] is expected
        assert body["user_email_verified"] is expected


# --------------------------------------------------------------------------- #
# Tenant API keys (w2v_sk_)
# --------------------------------------------------------------------------- #


class TestTenantApiKey:
    @pytest.mark.parametrize(
        "headers",
        [
            {"Authorization": f"Bearer {API_KEY}"},
            {"X-Tenant-Id": f"Bearer {API_KEY}"},
        ],
        ids=["authorization", "legacy-x-tenant-id"],
    )
    async def test_registered_verifier_authenticates_without_introspection(
        self, headers: dict[str, str]
    ) -> None:
        verifier = AsyncMock(return_value=(str(KEY_TENANT), "key-01"))
        register_api_key_verifier(verifier)

        status, body, introspect = await _headers_call(headers)

        assert status == 200, body
        verifier.assert_awaited_once_with(API_KEY)
        # The decisive property: the secret never reached the licenses service.
        introspect.assert_not_called()
        assert body["state_tenant_id"] == str(KEY_TENANT)
        assert body["ctx_tenant"] == str(KEY_TENANT)
        assert body["ctx_source"] == TenantSource.TENANT_API_KEY.value
        assert body["api_key_id"] == "key-01"
        assert body["ctx_subject"] == "api-key:key-01"
        # The key is never forwarded downstream or kept on request.state.
        assert body["ctx_caller_token"] == ""
        assert body["license_token"] is None

    async def test_no_verifier_registered_is_401_without_introspection(self) -> None:
        assert get_api_key_verifier() is None

        status, body, introspect = await _headers_call({"Authorization": f"Bearer {API_KEY}"})

        assert status == 401, body
        assert body["error"]["code"] == "API_KEY_UNSUPPORTED"
        introspect.assert_not_called()

    async def test_verifier_returning_none_is_401(self) -> None:
        register_api_key_verifier(AsyncMock(return_value=None))

        status, body, introspect = await _headers_call({"Authorization": f"Bearer {API_KEY}"})

        assert status == 401, body
        assert body["error"]["code"] == "INVALID_API_KEY"
        introspect.assert_not_called()

    async def test_verifier_returning_a_non_uuid_tenant_is_401(self) -> None:
        register_api_key_verifier(AsyncMock(return_value=("n8n-dev", "key-01")))

        status, body, _ = await _headers_call({"Authorization": f"Bearer {API_KEY}"})

        assert status == 401, body
        assert body["error"]["code"] == "INVALID_API_KEY"

    async def test_verifier_failure_is_503_and_never_logs_the_token(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def _boom(token: str) -> tuple[str, str] | None:
            raise RuntimeError(f"db down while looking up {token}")

        register_api_key_verifier(_boom)
        caplog.set_level(logging.DEBUG)

        status, body, introspect = await _headers_call({"Authorization": f"Bearer {API_KEY}"})

        assert status == 503, body
        assert body["error"]["code"] == "API_KEY_VERIFIER_UNAVAILABLE"
        assert body["error"]["status"] == 503
        introspect.assert_not_called()
        assert "RuntimeError" in caplog.text
        assert API_KEY not in caplog.text

    async def test_non_key_bearer_still_goes_to_introspection(self) -> None:
        """Regression guard: only the w2v_sk_ prefix is diverted."""
        verifier = AsyncMock()
        register_api_key_verifier(verifier)

        status, _body, introspect = await _headers_call({"Authorization": "Bearer lic_opaque"})

        assert status == 401  # introspection mock reports the token invalid
        introspect.assert_awaited_once_with("lic_opaque")
        verifier.assert_not_called()


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source", [TenantSource.MEMBERSHIP, TenantSource.TENANT_API_KEY])
def test_membership_and_api_key_are_vouched_and_delegate(source: TenantSource) -> None:
    assert source in VOUCHED_TENANT_SOURCES
    headers = build_downstream_tenant_headers(
        KEY_TENANT, "worker_management", source, downstream_license_token="svc-token"
    )
    assert headers[EFFECTIVE_TENANT_HEADER] == str(KEY_TENANT)


# --------------------------------------------------------------------------- #
# Review findings (WP2 follow-up)
# --------------------------------------------------------------------------- #


class TestApiKeyPrefixIsCaseInsensitive:
    """A mis-cased ``W2V_SK_`` must never fall through to licence introspection."""

    @pytest.mark.parametrize("prefix", ["W2V_SK_", "W2v_Sk_"])
    @pytest.mark.parametrize("header", ["Authorization", "X-Tenant-Id"])
    async def test_mixed_case_key_goes_to_the_verifier_not_licences(
        self, prefix: str, header: str
    ) -> None:
        key = prefix + API_KEY[len(TENANT_API_KEY_PREFIX) :]
        verifier = AsyncMock(return_value=None)
        register_api_key_verifier(verifier)

        status, body, introspect = await _headers_call({header: f"Bearer {key}"})

        assert status == 401, body
        assert body["error"]["code"] == "INVALID_API_KEY"
        verifier.assert_awaited_once_with(key)
        introspect.assert_not_called()

    async def test_mixed_case_key_without_verifier_is_401_without_introspection(self) -> None:
        key = "W2V_SK_" + API_KEY[len(TENANT_API_KEY_PREFIX) :]

        status, body, introspect = await _headers_call({"Authorization": f"Bearer {key}"})

        assert status == 401, body
        assert body["error"]["code"] == "API_KEY_UNSUPPORTED"
        introspect.assert_not_called()


class TestApiKeyCannotSwitchTenant:
    async def test_effective_tenant_header_is_ignored_on_the_api_key_lane(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even a key whose subject is (mis)listed as delegating cannot re-scope."""
        other = "0198f1d2-0000-7000-8000-00000000beef"
        monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", "api-key:key-01,key-01")
        register_api_key_verifier(AsyncMock(return_value=(str(KEY_TENANT), "key-01")))

        status, body, introspect = await _headers_call(
            {"Authorization": f"Bearer {API_KEY}", EFFECTIVE_TENANT_HEADER: other}
        )

        assert status == 200, body
        assert body["state_tenant_id"] == str(KEY_TENANT)
        assert body["user_tenant_id"] == str(KEY_TENANT)
        assert body["ctx_tenant"] == str(KEY_TENANT)
        assert body["ctx_source"] == TenantSource.TENANT_API_KEY.value
        introspect.assert_not_called()


async def _tenantless_outbound(request: Request) -> dict[str, Any]:
    """From inside a tenant-less JWT request, try an SA-credentialed outbound hop."""
    outbound = httpx.Request(
        "POST",
        "http://worker-management.w2v/api/jobs",
        headers={"X-Tenant-Id": "Bearer svc-account-token"},
    )
    try:
        await identity_event_hook(outbound)
    except MissingTenantError as exc:
        return {"refused": True, "error": str(exc)}
    return {"refused": False, "x_tenant_id": outbound.headers.get("X-Tenant-Id")}


class TestTenantlessJwtRequest:
    """``AUTH_JWT_TENANT_REQUIRED=false``: an authenticated principal with no tenant."""

    async def test_audit_never_records_the_client_supplied_tenant(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],  # noqa: F811
        enable_keycloak: dict[str, Any],  # noqa: F811
    ) -> None:
        monkeypatch.setenv("AUTH_JWT_TENANT_REQUIRED", "false")
        claims = _default_claims()
        del claims["tenant_id"]
        token = _sign_token(rsa_keypair[0], claims=claims)
        chosen = "0198f1d2-0000-7000-8000-00000000c0de"

        app = FastAPI()
        app.add_middleware(AuthMiddleware)
        app.add_middleware(AuditMiddleware)  # outermost, as in create_app
        app.add_api_route("/api/widgets", _identity, methods=["POST"])

        with (
            patch(
                "w2v_common.auth.keycloak.httpx.AsyncClient",
                return_value=_make_jwks_mock_client(enable_keycloak),
            ),
            caplog.at_level(logging.INFO, logger=DEFAULT_AUDIT_LOGGER),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                resp = await c.post(
                    "/api/widgets",
                    headers={"Authorization": f"Bearer {token}", "X-Tenant-Id": chosen},
                )

        assert resp.status_code == 200, resp.text
        assert resp.json()["state_tenant_id"] is None
        records = [r for r in caplog.records if r.name == DEFAULT_AUDIT_LOGGER]
        assert len(records) == 1
        payload = records[0].args
        assert isinstance(payload, dict)
        assert payload["tenant_id"] is None

    async def test_outbound_sa_bearer_hop_fails_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        rsa_keypair: tuple[rsa.RSAPrivateKey, dict[str, Any]],  # noqa: F811
        enable_keycloak: dict[str, Any],  # noqa: F811
    ) -> None:
        monkeypatch.setenv("AUTH_JWT_TENANT_REQUIRED", "false")
        claims = _default_claims()
        del claims["tenant_id"]
        token = _sign_token(rsa_keypair[0], claims=claims)

        app = FastAPI()
        app.add_middleware(AuthMiddleware)
        app.add_api_route("/api/outbound", _tenantless_outbound, methods=["GET"])

        with patch(
            "w2v_common.auth.keycloak.httpx.AsyncClient",
            return_value=_make_jwks_mock_client(enable_keycloak),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                resp = await c.get("/api/outbound", headers={"Authorization": f"Bearer {token}"})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["refused"] is True, body
        assert "no tenant bound" in body["error"]

    async def test_service_originated_sa_hop_is_unchanged(self) -> None:
        """No inbound request bound (CronJob / startup task): the SA Bearer still goes out."""
        clear_identity_context()
        assert not is_inbound_request()
        outbound = httpx.Request(
            "POST",
            "http://worker-management.w2v/api/jobs",
            headers={"X-Tenant-Id": "Bearer svc-account-token"},
        )

        await identity_event_hook(outbound)

        assert outbound.headers["X-Tenant-Id"] == "Bearer svc-account-token"
        assert EFFECTIVE_TENANT_HEADER not in outbound.headers
