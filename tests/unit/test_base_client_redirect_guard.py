"""Redirect-guard tests for ``BaseServiceClient``.

``BaseServiceClient`` deliberately does NOT follow redirects: a 3xx from a
microservice almost always indicates a route-shape bug in the callee
(e.g. ``POST /api/foo`` -> 307 ``/api/foo/`` because the callee declared
``@router.post("/")`` under a ``/api/foo`` prefix). Following the
redirect would silently mask the bug. We surface every 3xx as a
``BaseServiceClientRedirectError`` instead.

These tests use ``httpx.MockTransport`` to verify:

* a 307 response is converted into ``BaseServiceClientRedirectError``;
* successful 2xx responses still return the parsed JSON body;
* 4xx/5xx behaviour is unchanged (HTTPStatusError after retry exhaustion);
* ``_request_raw`` (binary downloads) gets the same redirect guard.
"""

from __future__ import annotations

import json

import httpx
import pytest

from w2v_common.clients import BaseServiceClient, BaseServiceClientRedirectError

_BASE = "http://test"


def _client_with_transport(transport: httpx.MockTransport) -> BaseServiceClient:
    """Build a ``BaseServiceClient`` whose underlying httpx client uses ``transport``.

    Mirrors the production header/follow-redirects setup so the test
    exercises the same call surface as a real downstream call.
    """
    client = BaseServiceClient(base_url=_BASE, service_name="svc")
    client._client = httpx.AsyncClient(
        transport=transport,
        base_url=_BASE,
        follow_redirects=False,
    )
    return client


# ---------------------------------------------------------------------------
# 307 -> BaseServiceClientRedirectError
# ---------------------------------------------------------------------------


class TestRedirectIsFatal:
    """Any 3xx must raise ``BaseServiceClientRedirectError`` immediately."""

    @pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
    async def test_3xx_raises_redirect_error(self, status_code: int) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                status_code=status_code,
                headers={"location": "/api/inspect/"},
            )

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(BaseServiceClientRedirectError) as excinfo:
                await client._request("POST", "/api/inspect")

            err = excinfo.value
            assert err.service_name == "svc"
            assert err.method == "POST"
            assert err.endpoint == "/api/inspect"
            assert err.status_code == status_code
            assert err.location == "/api/inspect/"
            # The redirect must NOT trigger retry-loop behaviour: exactly
            # one outbound request was made.
            assert len(captured) == 1
        finally:
            await client.close()

    async def test_redirect_error_is_runtimeerror_subclass(self) -> None:
        # Callers that want to broadly catch infrastructure errors via
        # RuntimeError must still match this exception.
        assert issubclass(BaseServiceClientRedirectError, RuntimeError)

    async def test_request_raw_also_guards_redirects(self) -> None:
        """``_request_raw`` (binary downloads) must apply the same guard."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                status_code=307,
                headers={"location": "/audio/123/"},
            )

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(BaseServiceClientRedirectError) as excinfo:
                await client._request_raw("GET", "/audio/123")
            assert excinfo.value.endpoint == "/audio/123"
            assert excinfo.value.location == "/audio/123/"
            assert len(captured) == 1
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# 2xx happy path is unaffected
# ---------------------------------------------------------------------------


class TestSuccessUnchanged:
    """Non-redirect responses behave exactly as before."""

    async def test_200_returns_parsed_json_body(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                content=json.dumps({"job_id": "abc"}).encode(),
                headers={"content-type": "application/json"},
            )

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            result = await client._request("POST", "/api/inspect")
            assert result == {"job_id": "abc"}
        finally:
            await client.close()

    async def test_request_raw_200_returns_response(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                content=b"\x00\x01binary",
                headers={"content-type": "application/octet-stream"},
            )

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            response = await client._request_raw("GET", "/audio/abc")
            assert response.status_code == 200
            assert response.content == b"\x00\x01binary"
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# 4xx/5xx behaviour is unchanged
# ---------------------------------------------------------------------------


class TestErrorBehaviourUnchanged:
    """Adding the redirect guard must not regress existing 4xx/5xx handling."""

    async def test_4xx_raises_httpstatuserror(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=404,
                content=json.dumps({"detail": "not found"}).encode(),
                headers={"content-type": "application/json"},
            )

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError) as excinfo:
                await client._request("GET", "/missing")
            assert excinfo.value.response.status_code == 404
        finally:
            await client.close()

    async def test_5xx_eventually_raises_after_retries(self) -> None:
        """500 is retryable; after retry exhaustion the final response raises."""
        attempts: list[int] = []

        def handler(_request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            return httpx.Response(
                status_code=500,
                content=json.dumps({"detail": "boom"}).encode(),
                headers={"content-type": "application/json"},
            )

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError) as excinfo:
                await client._request("GET", "/explode")
            assert excinfo.value.response.status_code == 500
            # 3 attempts is the configured retry budget for retryable codes.
            assert len(attempts) == 3
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Real httpx.AsyncClient (no MockTransport) honours follow_redirects=False
# ---------------------------------------------------------------------------


class TestRealClientFollowRedirectsFalse:
    """Verify the production code-path actually configures follow_redirects=False."""

    async def test_get_client_disables_redirect_following(self) -> None:
        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        try:
            async_client = await client._get_client()
            # httpx exposes the option as ``follow_redirects`` on the client.
            assert async_client.follow_redirects is False
        finally:
            await client.close()
