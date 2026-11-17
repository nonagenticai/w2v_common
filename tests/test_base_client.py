"""Tests for w2v_common.clients.base.BaseServiceClient."""

from __future__ import annotations

import logging
import ssl
from typing import Any

import httpx
import pytest

from w2v_common.clients.base import (
    _MAX_RETRIES,
    BaseServiceClient,
    JobCancelledError,
    JobSubmissionError,
)

_BASE = "http://test"


def _mock_client(handler: Any) -> httpx.AsyncClient:
    """Create an httpx.AsyncClient with a MockTransport and proper base_url."""
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=_BASE,
    )


def _json_response(data: dict, status_code: int = 200) -> httpx.Response:
    """Build an httpx.Response with JSON content."""
    import json

    return httpx.Response(
        status_code=status_code,
        content=json.dumps(data).encode(),
        headers={"content-type": "application/json"},
    )


class TestRequestRetryOn503ThenSuccess:
    async def test_retries_on_503_then_succeeds(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return _json_response({"error": "unavailable"}, status_code=503)
            return _json_response({"ok": True})

        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        client._client = _mock_client(handler)

        result = await client._request("GET", "/api/data")
        assert result == {"ok": True}
        assert call_count == 3
        await client.close()


class TestRequestRaisesAfterMaxRetries:
    async def test_raises_after_max_retries(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"error": "bad"}, status_code=503)

        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        client._client = _mock_client(handler)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client._request("GET", "/api/data")
        assert exc_info.value.response.status_code == 503
        await client.close()


class TestRequestSurfacesBodyInRaisedError:
    """A non-retryable >=400 must embed the response body in the raised error.

    Regression for the audiobook post-promote 500: a bare raise_for_status()
    dropped the body, so a caller that cannot read this pod's logs (the E2E
    validator) saw only "Server error '500'" with no cause. The body must ride
    on the exception message AND on .response.
    """

    async def test_request_embeds_body(self) -> None:
        # 403 is non-retryable, so this exercises the direct-raise path without
        # the retry backoff a 500 would incur; the body-surfacing is identical.
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response(
                {"detail": "minio upload failed: AccessDenied w2v-audiobooks"},
                status_code=403,
            )

        client = BaseServiceClient(base_url=_BASE, service_name="video_compile")
        client._client = _mock_client(handler)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client._request("POST", "/v1/concat-audio", json={"clips": []})
        msg = str(exc_info.value)
        assert "video_compile" in msg
        assert "/v1/concat-audio" in msg
        assert "403" in msg
        assert "AccessDenied w2v-audiobooks" in msg
        assert exc_info.value.response.status_code == 403
        await client.close()

    async def test_request_raw_embeds_body(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"detail": "boom-raw"}, status_code=403)

        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        client._client = _mock_client(handler)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client._request_raw("GET", "/api/data")
        assert "boom-raw" in str(exc_info.value)
        assert exc_info.value.response.status_code == 403
        await client.close()


class TestRequestRetriesOnTransportError:
    async def test_retries_on_transport_error(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < _MAX_RETRIES:
                raise httpx.ConnectError("connection refused")
            return _json_response({"recovered": True})

        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        client._client = _mock_client(handler)

        result = await client._request("GET", "/api/data")
        assert result == {"recovered": True}
        assert call_count == _MAX_RETRIES
        await client.close()


class TestPostRetriesOnConnectionReset:
    """POST is retried on connection-reset errors (stale keepalive / connect).

    Regression test for the orchestrator->inference-router failure: a long-lived
    pooled client hit a server-closed keepalive connection after a downstream
    restart and got a ``ReadError``; POST was non-retryable, so a recoverable
    blip became a hard stage failure.
    """

    async def test_post_retried_on_read_error_then_succeeds(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ReadError("")  # stale keepalive: server reset, no response
            return _json_response({"content": "ok"})

        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        client._client = _mock_client(handler)

        result = await client._request("POST", "/api/chat/completions", json={"x": 1})
        assert result == {"content": "ok"}
        assert call_count == 2
        await client.close()

    async def test_post_retried_on_connect_error_then_succeeds(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("connection refused")
            return _json_response({"ok": True})

        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        client._client = _mock_client(handler)

        result = await client._request("POST", "/api/decks", json={"x": 1})
        assert result == {"ok": True}
        assert call_count == 2
        await client.close()

    async def test_post_not_retried_on_read_timeout(self) -> None:
        # ReadTimeout means the server may be mid-process — replaying a
        # non-idempotent POST could double-act, so it must NOT be retried
        # without an idempotency key.
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            raise httpx.ReadTimeout("server slow")

        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        client._client = _mock_client(handler)

        with pytest.raises(httpx.ReadTimeout):
            await client._request("POST", "/api/decks", json={"x": 1})
        assert call_count == 1
        await client.close()


class TestSubmitJobReturnsJobId:
    async def test_submit_job_returns_job_id(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"job_id": "abc-123"})

        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        client._client = _mock_client(handler)

        job_id = await client.submit_job("/api/jobs", {"input": "data"})
        assert job_id == "abc-123"
        await client.close()


class TestPollJobReturnsOnCompleted:
    async def test_poll_returns_on_completed(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return _json_response({"status": "running"})
            return _json_response({"status": "completed", "result": "done"})

        client = BaseServiceClient(
            base_url=_BASE,
            service_name="svc",
            poll_interval=0.01,
            max_poll_attempts=10,
        )
        client._client = _mock_client(handler)

        result = await client.poll_job("/api/jobs", "job-1", timeout=30.0)
        assert result["status"] == "completed"
        assert result["result"] == "done"
        assert call_count == 3
        await client.close()


class TestPollJobRaisesOnFailed:
    async def test_poll_raises_on_failed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"status": "failed", "error": "out of memory"})

        client = BaseServiceClient(
            base_url=_BASE,
            service_name="svc",
            poll_interval=0.01,
        )
        client._client = _mock_client(handler)

        with pytest.raises(RuntimeError, match="out of memory"):
            await client.poll_job("/api/jobs", "job-2", timeout=30.0)
        await client.close()


class TestPollJobRaisesOnCancelled:
    async def test_poll_raises_promptly_on_cancelled(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _json_response({"status": "cancelled"})

        client = BaseServiceClient(
            base_url=_BASE,
            service_name="svc",
            poll_interval=0.01,
            max_poll_attempts=50,
        )
        client._client = _mock_client(handler)

        with pytest.raises(JobCancelledError, match="was cancelled"):
            await client.poll_job("/api/jobs", "job-cancelled", timeout=30.0)
        # Must raise on the FIRST poll that sees "cancelled" -- not spin until
        # max_poll_attempts (50) or timeout. This is the assertion that would
        # fail if the "cancelled" handling were removed: without it the status
        # falls through to asyncio.sleep() and the loop keeps polling.
        assert call_count == 1
        await client.close()

    async def test_cancelled_is_a_runtime_error_subclass(self) -> None:
        """Existing ``except RuntimeError:`` callers must still catch this."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"status": "cancelled"})

        client = BaseServiceClient(
            base_url=_BASE,
            service_name="svc",
            poll_interval=0.01,
        )
        client._client = _mock_client(handler)

        with pytest.raises(RuntimeError):
            await client.poll_job("/api/jobs", "job-cancelled-2", timeout=30.0)
        await client.close()


class TestPollJobUnknownStatusStillPolls:
    async def test_unrecognised_status_keeps_polling_not_terminal(self) -> None:
        """Regression guard: an unrecognised/in-progress status must NOT be
        treated as terminal. Only recognised statuses (completed family,
        failed family, cancelled) should stop the loop early.
        """
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return _json_response({"status": "queued"})
            return _json_response({"status": "completed", "result": "done"})

        client = BaseServiceClient(
            base_url=_BASE,
            service_name="svc",
            poll_interval=0.01,
            max_poll_attempts=10,
        )
        client._client = _mock_client(handler)

        result = await client.poll_job("/api/jobs", "job-queued", timeout=30.0)
        assert result["status"] == "completed"
        assert call_count == 3
        await client.close()


class TestPollJobRaisesTimeout:
    async def test_poll_raises_timeout_on_max_attempts(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"status": "running"})

        client = BaseServiceClient(
            base_url=_BASE,
            service_name="svc",
            poll_interval=0.01,
            max_poll_attempts=3,
        )
        client._client = _mock_client(handler)

        with pytest.raises(TimeoutError, match="exceeded 3 poll attempts"):
            await client.poll_job("/api/jobs", "job-3", timeout=300.0)
        await client.close()


class TestHealthCheckHealthy:
    async def test_health_check_healthy(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"version": "1.0"})

        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        client._client = _mock_client(handler)

        result = await client.health_check()
        assert result["status"] == "healthy"
        assert result["service"] == "svc"
        assert result["version"] == "1.0"
        await client.close()


class TestHealthCheckUnhealthy:
    async def test_health_check_unhealthy(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"error": "down"}, status_code=500)

        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        client._client = _mock_client(handler)

        result = await client.health_check()
        assert result["status"] == "unhealthy"
        assert result["service"] == "svc"
        assert "error" in result
        await client.close()


class TestCloseClosesClient:
    async def test_close_closes_client(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({})

        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        client._client = _mock_client(handler)
        assert client._client is not None
        assert not client._client.is_closed

        await client.close()
        assert client._client is None


class TestSslContextPassedToHttpx:
    async def test_ssl_context_passed_to_httpx(self) -> None:
        ctx = ssl.create_default_context()
        client = BaseServiceClient(
            base_url="http://test",
            service_name="svc",
            ssl_context=ctx,
        )
        # Trigger lazy client creation
        httpx_client = await client._get_client()
        assert httpx_client is not None
        assert client.ssl_context is ctx
        await client.close()


class TestRequestRawReturnsResponse:
    async def test_request_raw_returns_response(self) -> None:
        """_request_raw returns the raw httpx.Response object."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                content=b"audio data",
                headers={"content-type": "application/octet-stream"},
            )

        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        client._client = _mock_client(handler)

        response = await client._request_raw("GET", "/api/audio/download")
        assert isinstance(response, httpx.Response)
        assert response.content == b"audio data"
        assert response.status_code == 200
        await client.close()


class TestConstructorDefaults:
    def test_constructor_defaults(self) -> None:
        client = BaseServiceClient(base_url="http://example.com/", service_name="test-svc")
        assert client.base_url == "http://example.com"  # trailing slash stripped
        assert client.service_name == "test-svc"
        assert client.timeout == 30.0
        assert client.license_token is None
        assert client.ssl_context is True
        assert client.poll_interval == 2.0
        assert client.max_poll_attempts == 150
        assert client._client is None

    def test_constructor_custom_values(self) -> None:
        ctx = ssl.create_default_context()
        client = BaseServiceClient(
            base_url="http://svc:8000",
            service_name="custom",
            timeout=60.0,
            license_token="tok-123",
            ssl_context=ctx,
            poll_interval=5.0,
            max_poll_attempts=50,
        )
        assert client.base_url == "http://svc:8000"
        assert client.timeout == 60.0
        assert client.license_token == "tok-123"
        assert client.ssl_context is ctx
        assert client.poll_interval == 5.0
        assert client.max_poll_attempts == 50


class TestGetClientAttachesIdentityEventHook:
    async def test_get_client_attaches_identity_event_hook(self) -> None:
        """``_get_client()`` must attach ``identity_event_hook`` so cached
        clients propagate X-Tenant-Id / X-License-Id / X-Subject-Id from
        the per-request ContextVars on every outbound call.
        """
        from w2v_common.auth.identity_context import identity_event_hook

        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        underlying = await client._get_client()
        try:
            request_hooks = underlying.event_hooks.get("request", [])
            assert identity_event_hook in request_hooks, (
                "BaseServiceClient._get_client() must attach identity_event_hook "
                "so X-Tenant-Id / X-License-Id / X-Subject-Id flow on every call."
            )
        finally:
            await client.close()


class TestPollJobExtraSuccessStatuses:
    """A service may park a FINISHED job in a status the shared set cannot know.

    ``app_inspector`` returns ``awaiting_selection`` when a job was submitted with
    ``require_selection=True``, and serves ``/result`` in exactly that state. The
    shared poller must let a caller declare it terminal, or the poll spins to the
    deadline on a job whose result is already readable.
    """

    async def test_declared_status_is_terminal_and_returns_the_body(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _json_response({"status": "awaiting_selection", "job_id": "j1"})

        client = BaseServiceClient(
            base_url=_BASE,
            service_name="app_inspector",
            poll_interval=0.01,
            max_poll_attempts=50,
        )
        client._client = _mock_client(handler)

        result = await client.poll_job(
            "/api/jobs",
            "j1",
            timeout=30.0,
            extra_success_statuses=frozenset({"awaiting_selection"}),
        )
        assert result["status"] == "awaiting_selection"
        # Terminal on the FIRST poll. Without the extra-status branch the status
        # falls through to sleep() and the loop runs to max_poll_attempts.
        assert call_count == 1
        await client.close()

    async def test_undeclared_status_is_still_not_terminal(self) -> None:
        """The default set must stay honest -- the opt-in is per caller."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _json_response({"status": "awaiting_selection"})

        client = BaseServiceClient(
            base_url=_BASE,
            service_name="svc",
            poll_interval=0.01,
            max_poll_attempts=3,
        )
        client._client = _mock_client(handler)

        with pytest.raises(TimeoutError):
            await client.poll_job("/api/jobs", "j2", timeout=30.0)
        assert call_count == 3
        await client.close()

    async def test_unrecognised_status_is_logged_once_not_silently_absorbed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The gap must be EXPRESSIBLE. A status in neither set is otherwise
        indistinguishable from "still working" until the deadline fires."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"status": "awaiting_selection"})

        client = BaseServiceClient(
            base_url=_BASE,
            service_name="svc",
            poll_interval=0.01,
            max_poll_attempts=4,
        )
        client._client = _mock_client(handler)

        with (
            caplog.at_level(logging.WARNING, logger="w2v_common.clients.base"),
            pytest.raises(TimeoutError),
        ):
            await client.poll_job("/api/jobs", "j3", timeout=30.0)

        warnings = [r for r in caplog.records if "awaiting_selection" in r.getMessage()]
        # Exactly one across FOUR polls of the same status -- loud, not a flood.
        assert len(warnings) == 1, [r.getMessage() for r in warnings]

    async def test_in_progress_statuses_do_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """A normal ``processing`` poll must stay quiet, or the signal is worthless."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"status": "processing"})

        client = BaseServiceClient(
            base_url=_BASE,
            service_name="svc",
            poll_interval=0.01,
            max_poll_attempts=3,
        )
        client._client = _mock_client(handler)

        with (
            caplog.at_level(logging.WARNING, logger="w2v_common.clients.base"),
            pytest.raises(TimeoutError),
        ):
            await client.poll_job("/api/jobs", "j4", timeout=30.0)

        assert [r.getMessage() for r in caplog.records if "processing" in r.getMessage()] == []


class TestSubmitJobRefusesUnusableHandle:
    """A 2xx envelope with no usable ``job_id`` must NOT yield a blank handle.

    Every caller polls ``<endpoint>/{job_id}``, so a ``""`` handle polls the
    collection route and the run dies as a 404/timeout naming the POLL, never
    the submit -- while the accepted job runs on unsupervised.
    """

    @pytest.mark.parametrize(
        "envelope",
        [
            {},  # istio/cloudflared answering an empty 2xx
            {"id": "abc-123"},  # renamed field
            {"job": {"job_id": "abc-123"}},  # nested
            {"job_id": ""},  # present but blank
            {"job_id": "   "},  # whitespace-only
            {"job_id": None},
            {"job_id": True},  # bool is an int subclass but is not a handle
        ],
    )
    async def test_missing_job_id_raises_and_names_the_endpoint(self, envelope) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response(envelope)

        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        client._client = _mock_client(handler)

        with pytest.raises(JobSubmissionError) as excinfo:
            await client.submit_job("/api/jobs", {"input": "data"})
        message = str(excinfo.value)
        assert "/api/jobs" in message, "error must name the SUBMIT endpoint"
        assert "svc" in message
        await client.close()

    async def test_int_job_id_is_coerced_not_refused(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"job_id": 42})

        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        client._client = _mock_client(handler)

        assert await client.submit_job("/api/jobs", {"input": "data"}) == "42"
        await client.close()
