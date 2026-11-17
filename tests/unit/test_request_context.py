"""Tests for the request-id ContextVar, trace event hook, and log filter."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, cast

import httpx
import pytest

from w2v_common.clients import BaseServiceClient
from w2v_common.observability.request_context import (
    RequestIdLogFilter,
    clear_request_id,
    get_request_id,
    set_request_id,
    trace_event_hook,
)

_BASE = "http://test"


@pytest.fixture(autouse=True)
def _isolate_request_id() -> Iterator[None]:
    clear_request_id()
    yield
    clear_request_id()


class TestRequestIdContextVar:
    def test_default_is_empty_string(self) -> None:
        assert get_request_id() == ""

    def test_set_then_get_roundtrips(self) -> None:
        set_request_id("abc-123")
        assert get_request_id() == "abc-123"

    def test_set_none_normalises_to_empty(self) -> None:
        set_request_id("seed")
        set_request_id(None)
        assert get_request_id() == ""


class TestTraceEventHook:
    async def test_stamps_request_id_when_var_set(self) -> None:
        set_request_id("trace-xyz")
        req = httpx.Request("GET", f"{_BASE}/healthz")
        await trace_event_hook(req)
        assert req.headers["x-request-id"] == "trace-xyz"

    async def test_noop_when_var_empty(self) -> None:
        req = httpx.Request("GET", f"{_BASE}/healthz")
        await trace_event_hook(req)
        assert "x-request-id" not in req.headers

    async def test_caller_supplied_header_wins(self) -> None:
        set_request_id("from-context")
        req = httpx.Request(
            "GET",
            f"{_BASE}/healthz",
            headers={"X-Request-Id": "from-caller"},
        )
        await trace_event_hook(req)
        assert req.headers["x-request-id"] == "from-caller"


class TestBaseServiceClientPropagation:
    async def test_outbound_request_carries_request_id(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"status": "ok"})

        set_request_id("req-from-test")
        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        # Match production wiring: include the event_hook (set_transport does
        # not, by design) so we exercise the real propagation path.
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url=_BASE,
            follow_redirects=False,
            event_hooks={"request": [trace_event_hook]},
        )
        try:
            await client._request("GET", "/healthz")
        finally:
            await client.close()
        assert captured, "request was never made"
        assert captured[0].headers["x-request-id"] == "req-from-test"


class TestRequestIdLogFilter:
    def test_filter_attaches_request_id_to_record(self) -> None:
        set_request_id("log-test-id")
        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname="t.py",
            lineno=1,
            msg="hi",
            args=(),
            exc_info=None,
        )
        assert RequestIdLogFilter().filter(record) is True
        assert cast(Any, record).request_id == "log-test-id"

    def test_filter_preserves_explicit_record_attribute(self) -> None:
        set_request_id("ctxvar-id")
        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname="t.py",
            lineno=1,
            msg="hi",
            args=(),
            exc_info=None,
        )
        cast(Any, record).request_id = "explicit"
        RequestIdLogFilter().filter(record)
        assert cast(Any, record).request_id == "explicit"


class TestClientConstructionBounds:
    def test_timeout_none_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be a positive number, not None"):
            BaseServiceClient(base_url=_BASE, service_name="svc", timeout=cast(Any, None))

    def test_timeout_above_ceiling_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must be <= 600\.0s"):
            BaseServiceClient(base_url=_BASE, service_name="svc", timeout=9999.0)

    def test_timeout_at_ceiling_is_accepted(self) -> None:
        # Boundary: exactly _MAX_TIMEOUT (env unset) must NOT raise.
        client = BaseServiceClient(base_url=_BASE, service_name="svc", timeout=600.0)
        assert client.timeout == 600.0

    def test_env_raises_ceiling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # CLIENT_MAX_TIMEOUT_S is read at call time — a raised ceiling admits
        # a value that would be rejected under the default.
        monkeypatch.setenv("CLIENT_MAX_TIMEOUT_S", "800")
        client = BaseServiceClient(base_url=_BASE, service_name="svc", timeout=750.0)
        assert client.timeout == 750.0

    def test_env_lowers_ceiling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLIENT_MAX_TIMEOUT_S", "300")
        with pytest.raises(ValueError, match=r"must be <= 300\.0s"):
            BaseServiceClient(base_url=_BASE, service_name="svc", timeout=400.0)

    def test_env_invalid_ceiling_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A non-float env value is ignored → ceiling falls back to 600.0.
        monkeypatch.setenv("CLIENT_MAX_TIMEOUT_S", "abc")
        with pytest.raises(ValueError, match=r"must be <= 600\.0s"):
            BaseServiceClient(base_url=_BASE, service_name="svc", timeout=700.0)
        client = BaseServiceClient(base_url=_BASE, service_name="svc", timeout=600.0)
        assert client.timeout == 600.0

    def test_timeout_below_floor_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must be >= 1\.0s"):
            BaseServiceClient(base_url=_BASE, service_name="svc", timeout=0.0)

    def test_poll_interval_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="poll_interval must be > 0"):
            BaseServiceClient(base_url=_BASE, service_name="svc", poll_interval=0.0)

    def test_poll_interval_above_ceiling_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"poll_interval must be <= 60\.0s"):
            BaseServiceClient(base_url=_BASE, service_name="svc", poll_interval=120.0)

    def test_max_poll_attempts_above_ceiling_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_poll_attempts must be <= 500"):
            BaseServiceClient(base_url=_BASE, service_name="svc", max_poll_attempts=10_000)

    def test_max_poll_attempts_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_poll_attempts must be > 0"):
            BaseServiceClient(base_url=_BASE, service_name="svc", max_poll_attempts=0)

    def test_defaults_remain_valid(self) -> None:
        # Sanity: the existing defaults must satisfy the clamps so existing
        # callers don't break on 0.16.2.
        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        assert client.timeout == 30.0
        assert client.poll_interval == 2.0
        assert client.max_poll_attempts == 150


class TestPollJobBounds:
    async def test_poll_job_rejects_unbounded_max_attempts(self) -> None:
        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        try:
            with pytest.raises(ValueError, match="max_poll_attempts must be <= 500"):
                await client.poll_job("/jobs", job_id="j1", max_attempts=10_000)
        finally:
            await client.close()

    async def test_poll_job_rejects_none_timeout(self) -> None:
        client = BaseServiceClient(base_url=_BASE, service_name="svc")
        try:
            with pytest.raises(ValueError, match="must be a positive number"):
                await client.poll_job("/jobs", job_id="j1", timeout=cast(Any, None))
        finally:
            await client.close()
