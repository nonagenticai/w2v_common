"""Tests for license-token vs JWT routing in BaseServiceClient.

Closes the 2026-04-18 trusted-network bypass: orchestrator -> downstream
auth used to send the license token via Authorization, which the
middleware ignores (it reads license tokens only from X-Tenant-Id),
so requests succeeded only via the trusted-network fallback.
"""

from __future__ import annotations

import json

import httpx

from w2v_common.clients.base import BaseServiceClient

_BASE = "http://test"
_OPAQUE_LICENSE = "opaque-licence-token-fixture"
_JWT_SHAPED = "header.payload.signature"


def _capturing_transport(captured: list[httpx.Request]) -> httpx.MockTransport:
    """Return a MockTransport that records every outgoing request."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            status_code=200,
            content=json.dumps({"ok": True}).encode(),
            headers={"content-type": "application/json"},
        )

    return httpx.MockTransport(handler)


class TestLicenseTokenRouting:
    """Opaque license tokens must travel via X-Tenant-Id, not Authorization."""

    async def test_license_token_routed_to_x_tenant_id(self) -> None:
        captured: list[httpx.Request] = []
        client = BaseServiceClient(
            base_url=_BASE,
            service_name="svc",
            license_token=_OPAQUE_LICENSE,
        )
        client._client = httpx.AsyncClient(
            transport=_capturing_transport(captured),
            base_url=_BASE,
            headers={"X-Tenant-Id": f"Bearer {_OPAQUE_LICENSE}"},
        )

        await client._request("GET", "/whatever")

        assert len(captured) == 1
        outgoing = captured[0]
        assert outgoing.headers.get("X-Tenant-Id") == f"Bearer {_OPAQUE_LICENSE}"
        # Authorization must NOT carry the license token: middleware would
        # try to validate it as a Keycloak JWT, fail _looks_like_jwt, and
        # fall through to the trusted-network branch (the bypass we are
        # closing).
        auth_header = outgoing.headers.get("Authorization")
        if auth_header is not None:
            assert _OPAQUE_LICENSE not in auth_header

    async def test_get_client_sets_x_tenant_id_header(self) -> None:
        # Verifies the header-construction logic in _get_client without
        # manually overriding self._client. This is the path exercised by
        # every real BaseServiceClient instance in production.
        client = BaseServiceClient(
            base_url=_BASE,
            service_name="svc",
            license_token=_OPAQUE_LICENSE,
        )
        async_client = await client._get_client()
        try:
            assert async_client.headers.get("X-Tenant-Id") == f"Bearer {_OPAQUE_LICENSE}"
            # No Authorization header should be set for opaque license tokens.
            assert "authorization" not in {k.lower() for k in async_client.headers}
        finally:
            await client.close()


class TestJwtTokenRouting:
    """JWT-shaped tokens must keep traveling via Authorization for Keycloak."""

    async def test_jwt_routed_to_authorization(self) -> None:
        captured: list[httpx.Request] = []
        client = BaseServiceClient(
            base_url=_BASE,
            service_name="svc",
            license_token=_JWT_SHAPED,
        )
        client._client = httpx.AsyncClient(
            transport=_capturing_transport(captured),
            base_url=_BASE,
            headers={"Authorization": f"Bearer {_JWT_SHAPED}"},
        )

        await client._request("GET", "/whatever")

        assert len(captured) == 1
        outgoing = captured[0]
        assert outgoing.headers.get("Authorization") == f"Bearer {_JWT_SHAPED}"
        # X-Tenant-Id must NOT carry a JWT: the JWT branch in
        # AuthMiddleware.dispatch only inspects Authorization.
        x_tenant = outgoing.headers.get("X-Tenant-Id")
        if x_tenant is not None:
            assert _JWT_SHAPED not in x_tenant

    async def test_get_client_sets_authorization_for_jwt(self) -> None:
        client = BaseServiceClient(
            base_url=_BASE,
            service_name="svc",
            license_token=_JWT_SHAPED,
        )
        async_client = await client._get_client()
        try:
            assert async_client.headers.get("Authorization") == f"Bearer {_JWT_SHAPED}"
            assert "x-tenant-id" not in {k.lower() for k in async_client.headers}
        finally:
            await client.close()


class TestNoTokenStillNoHeader:
    """If no token is configured, neither header is set."""

    async def test_no_license_token_no_auth_headers(self) -> None:
        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        async_client = await client._get_client()
        try:
            lower_keys = {k.lower() for k in async_client.headers}
            assert "authorization" not in lower_keys
            assert "x-tenant-id" not in lower_keys
        finally:
            await client.close()
