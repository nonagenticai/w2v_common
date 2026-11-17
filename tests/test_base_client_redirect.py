"""Defense-in-depth: BaseServiceClient must surface 3xx as a clean error.

History (2026-04-28): app_inspector_service POST /api/inspect/ trailing-slash
mismatch returned 307; BaseServiceClient previously called .json() on the
empty redirect body and raised JSONDecodeError, hiding the route-shape bug.

The fix landed in PR #2 (commit 1115800) as ``BaseServiceClientRedirectError``
which subclasses ``RuntimeError``. This regression test pins both:

    1. 3xx responses raise ``BaseServiceClientRedirectError`` with the
       redirect target visible in the message.
    2. Non-redirect (2xx) responses still parse JSON normally -- behavior
       preservation.

The test exercises the public ``set_transport`` seam rather than poking the
private ``_client`` attribute, so no ``# type: ignore`` is required.
"""

from __future__ import annotations

import httpx
import pytest

from w2v_common.clients.base import (
    BaseServiceClient,
    BaseServiceClientRedirectError,
)


@pytest.mark.asyncio
async def test_3xx_response_raises_with_meaningful_message() -> None:
    """A 307 from a downstream service must surface the redirect target."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            307,
            headers={"Location": "http://localhost/api/inspect/"},
        )

    client = BaseServiceClient(
        base_url="http://localhost",
        service_name="app_inspector",
        license_token="opaque-token",
    )
    client.set_transport(httpx.MockTransport(handler))

    with pytest.raises(BaseServiceClientRedirectError) as exc_info:
        await client._request("POST", "/api/inspect", {})

    err = exc_info.value
    assert err.status_code == 307
    assert err.location == "http://localhost/api/inspect/"
    assert err.method == "POST"
    assert err.endpoint == "/api/inspect"

    msg = str(err)
    assert "307" in msg
    assert "/api/inspect/" in msg  # surfaces the redirect target
    assert "app_inspector" in msg  # service name in message

    await client.close()


@pytest.mark.asyncio
async def test_2xx_response_still_returns_json() -> None:
    """Behavior preservation: non-redirect responses unchanged."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    client = BaseServiceClient(
        base_url="http://localhost",
        service_name="app_inspector",
        license_token="opaque-token",
    )
    client.set_transport(httpx.MockTransport(handler))

    result = await client._request("POST", "/api/inspect", {})
    assert result == {"ok": True}

    await client.close()


@pytest.mark.asyncio
async def test_set_transport_disables_follow_redirects() -> None:
    """The set_transport seam must preserve follow_redirects=False semantics.

    Otherwise a test using set_transport would silently follow redirects and
    miss the very route-shape bug this guard exists to catch.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            301,
            headers={"Location": "http://localhost/elsewhere"},
        )

    client = BaseServiceClient(
        base_url="http://localhost",
        service_name="svc",
    )
    client.set_transport(httpx.MockTransport(handler))

    with pytest.raises(BaseServiceClientRedirectError):
        await client._request("GET", "/api/whatever")

    await client.close()
