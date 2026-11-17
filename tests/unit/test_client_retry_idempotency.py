"""Retry-method gate + Idempotency-Key plumbing for ``BaseServiceClient``.

POST and PATCH must not be retried by default — replaying a non-idempotent
write after a 503 / transport error can silently double-create resources.
The gate opens only when the caller passes ``idempotency_key=``, which the
client mirrors as the ``Idempotency-Key`` header on every attempt so the
downstream service can dedupe replays.
"""

from __future__ import annotations

import json

import httpx
import pytest

from w2v_common.clients import BaseServiceClient

_BASE = "http://test"


def _client_with_transport(transport: httpx.MockTransport) -> BaseServiceClient:
    client = BaseServiceClient(base_url=_BASE, service_name="svc")
    client._client = httpx.AsyncClient(
        transport=transport,
        base_url=_BASE,
        follow_redirects=False,
    )
    return client


class TestPostRetryGate:
    """POST without ``idempotency_key`` must NOT retry; with it, must retry."""

    async def test_post_503_not_retried_without_key(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(status_code=503, content=b'{"error":"upstream"}')

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client._request("POST", "/api/foo", json={"x": 1})
        finally:
            await client.close()

        # Without idempotency_key, the gate slams shut after the first
        # 503 — no retry. Bare POST replay on transient errors is exactly
        # the double-create vector this gate is designed to prevent.
        assert len(calls) == 1

    async def test_post_503_retries_with_key(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) < 3:
                return httpx.Response(status_code=503, content=b'{"error":"upstream"}')
            return httpx.Response(status_code=200, content=b'{"ok":true}')

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            result = await client._request(
                "POST", "/api/foo", json={"x": 1}, idempotency_key="job-42"
            )
        finally:
            await client.close()

        assert result == {"ok": True}
        assert len(calls) == 3
        # Every attempt must carry the same Idempotency-Key so the
        # downstream service can dedupe replays.
        for req in calls:
            assert req.headers.get("idempotency-key") == "job-42"

    async def test_post_read_timeout_not_retried_without_key(self) -> None:
        # A ReadTimeout means the server may already be processing the request,
        # so replaying a non-idempotent POST could double-act — it must NOT be
        # retried without an idempotency key. (Connection-*reset* errors are a
        # different case: the request never reached the server, so those ARE
        # retried for POST — see TestPostRetriesOnConnectionReset.)
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            raise httpx.ReadTimeout("server slow")

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.ReadTimeout):
                await client._request("POST", "/api/foo", json={"x": 1})
        finally:
            await client.close()

        assert len(calls) == 1

    async def test_post_connection_reset_retried_without_key(self) -> None:
        # Stale keepalive after a downstream restart: the request never reached
        # the application, so retrying on a fresh connection is safe even for a
        # bare POST. This is the orchestrator->inference-router regression.
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) < 2:
                raise httpx.ConnectError("connection reset")
            return httpx.Response(status_code=200, content=b'{"ok":true}')

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            result = await client._request("POST", "/api/foo", json={"x": 1})
        finally:
            await client.close()

        assert result == {"ok": True}
        assert len(calls) == 2

    async def test_post_transport_error_retries_with_key(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) < 2:
                raise httpx.ConnectError("boom")
            return httpx.Response(status_code=200, content=b'{"ok":true}')

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            result = await client._request(
                "POST", "/api/foo", json={"x": 1}, idempotency_key="job-7"
            )
        finally:
            await client.close()

        assert result == {"ok": True}
        assert len(calls) == 2

    async def test_get_503_retries_without_key(self) -> None:
        """GET is idempotent per RFC 9110 — retried unconditionally."""

        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) < 2:
                return httpx.Response(status_code=503)
            return httpx.Response(status_code=200, content=b'{"ok":true}')

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            result = await client._request("GET", "/api/foo")
        finally:
            await client.close()

        assert result == {"ok": True}
        assert len(calls) == 2

    async def test_submit_job_threads_idempotency_key(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(status_code=200, content=b'{"job_id":"jid-1"}')

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            job_id = await client.submit_job(
                "/api/jobs", {"payload": "x"}, idempotency_key="dedupe-1"
            )
        finally:
            await client.close()

        assert job_id == "jid-1"
        assert calls[0].headers.get("idempotency-key") == "dedupe-1"


class TestRequestRawRetryGate:
    """``_request_raw`` shares the gate with ``_request``."""

    async def test_raw_post_503_not_retried_without_key(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(status_code=503)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client._request_raw("POST", "/api/foo")
        finally:
            await client.close()

        assert len(calls) == 1


class TestLogRedaction:
    """``response.text`` containing bearer tokens must NOT reach the log line."""

    async def test_4xx_body_with_bearer_is_redacted(self, caplog) -> None:
        body = json.dumps(
            {"error": "token_not_valid", "msg": "Authorization: Bearer abc123xyz"}
        ).encode()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=401, content=body)

        client = _client_with_transport(httpx.MockTransport(handler))
        try:
            with caplog.at_level("ERROR"), pytest.raises(httpx.HTTPStatusError):
                await client._request("GET", "/api/foo")
        finally:
            await client.close()

        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "abc123xyz" not in joined
        assert "[REDACTED]" in joined
