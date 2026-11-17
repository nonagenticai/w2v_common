"""Tests for the outbound (downstream) call metrics on BaseServiceClient.

Every logical ``_request`` / ``_request_raw`` emits exactly ONE
``w2v_downstream_calls_total`` increment (after the client's internal retries
resolve) plus one ``w2v_downstream_latency_seconds`` observation, labelled by
the callee ``service_name``, method, and a classified ``outcome``. A unique
``service_name`` per test isolates the global default-REGISTRY counters.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from prometheus_client import REGISTRY

from w2v_common.clients.base import BaseServiceClient

_BASE = "http://test"


def _mock_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=_BASE)


def _json_response(data: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(data).encode(),
        headers={"content-type": "application/json"},
    )


def _calls(target: str, method: str, outcome: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "w2v_downstream_calls_total",
            {"target": target, "method": method, "outcome": outcome},
        )
        or 0.0
    )


def _latency_count(target: str, method: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "w2v_downstream_latency_seconds_count",
            {"target": target, "method": method},
        )
        or 0.0
    )


class TestDownstreamMetricsOk:
    async def test_success_records_ok_once(self) -> None:
        target = "svc_ok"

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"ok": True})

        client = BaseServiceClient(base_url=_BASE, service_name=target)
        client._client = _mock_client(handler)
        try:
            assert await client._request("GET", "/api/data") == {"ok": True}
        finally:
            await client.close()

        assert _calls(target, "GET", "ok") == 1.0
        # latency observed exactly once for the logical call
        assert _latency_count(target, "GET") == 1.0


class TestDownstreamMetricsHttpError:
    async def test_4xx_records_http_error_once(self) -> None:
        target = "svc_4xx"

        def handler(request: httpx.Request) -> httpx.Response:
            # 404 is non-retryable → single attempt, raises HTTPStatusError
            return _json_response({"error": "missing"}, status_code=404)

        client = BaseServiceClient(base_url=_BASE, service_name=target)
        client._client = _mock_client(handler)
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client._request("GET", "/api/data")
        finally:
            await client.close()

        assert _calls(target, "GET", "http_error") == 1.0
        # the retried 503 path also resolves to exactly one observation
        assert _calls(target, "GET", "ok") == 0.0


class TestDownstreamMetricsRetriedThenOneObservation:
    async def test_retries_collapse_to_single_observation(self) -> None:
        target = "svc_retry"
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return _json_response({"error": "bad"}, status_code=503)

        client = BaseServiceClient(base_url=_BASE, service_name=target)
        client._client = _mock_client(handler)
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client._request("GET", "/api/data")
        finally:
            await client.close()

        assert attempts > 1  # the client retried internally
        assert _calls(target, "GET", "http_error") == 1.0  # but metered once
        assert _latency_count(target, "GET") == 1.0


class TestDownstreamMetricsTimeout:
    async def test_read_timeout_records_timeout(self) -> None:
        target = "svc_timeout"

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        client = BaseServiceClient(base_url=_BASE, service_name=target)
        client._client = _mock_client(handler)
        try:
            with pytest.raises(httpx.TimeoutException):
                await client._request("GET", "/api/data")
        finally:
            await client.close()

        assert _calls(target, "GET", "timeout") == 1.0


class TestDownstreamMetricsTransportError:
    async def test_connect_error_records_transport_error(self) -> None:
        target = "svc_transport"

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        client = BaseServiceClient(base_url=_BASE, service_name=target)
        client._client = _mock_client(handler)
        try:
            with pytest.raises(httpx.TransportError):
                await client._request("GET", "/api/data")
        finally:
            await client.close()

        assert _calls(target, "GET", "transport_error") == 1.0


class TestRecordDownstreamCallNeverRaises:
    """``record_downstream_call`` is invoked from INSIDE ``_observe_downstream``'s
    except handlers, immediately before the bare ``raise`` that re-propagates the
    real downstream failure. If the metrics call itself raised, it would replace
    that failure with its own exception — same defect class as
    ``cost_bus._redis_endpoint`` (w2v_common 0.67.9 -> 0.67.10 #141): a helper
    called from inside an ``except`` block must be total, or it masks the real
    error instead of letting the bare ``raise`` re-propagate it.
    """

    def test_non_numeric_elapsed_does_not_raise(self) -> None:
        from w2v_common.observability.metrics import record_downstream_call

        # A non-numeric elapsed_seconds (e.g. a future refactor bug upstream)
        # used to blow up inside Histogram.observe()'s `+=`, uncaught.
        record_downstream_call(
            "svc_bad_elapsed",
            "GET",
            "ok",
            "not-a-float",  # ty: ignore[invalid-argument-type]
        )

    async def test_broken_metrics_call_does_not_mask_the_http_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from w2v_common.observability import metrics as metrics_mod

        target = "svc_metrics_broken"

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"error": "missing"}, status_code=404)

        class BrokenLatency:
            def labels(self, *args: Any, **kwargs: Any) -> Any:
                raise TypeError("simulated metrics failure (e.g. bad label value)")

        monkeypatch.setattr(metrics_mod, "downstream_latency_seconds", BrokenLatency())

        client = BaseServiceClient(base_url=_BASE, service_name=target)
        client._client = _mock_client(handler)
        try:
            # The real downstream fault (HTTPStatusError) must still surface,
            # NOT the TypeError from the broken metrics call.
            with pytest.raises(httpx.HTTPStatusError):
                await client._request("GET", "/api/data")
        finally:
            await client.close()


class TestDownstreamMetricsRawRequest:
    async def test_request_raw_is_metered_too(self) -> None:
        target = "svc_raw"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=200, content=b"binarydata")

        client = BaseServiceClient(base_url=_BASE, service_name=target)
        client._client = _mock_client(handler)
        try:
            resp = await client._request_raw("GET", "/api/blob")
            assert resp.content == b"binarydata"
        finally:
            await client.close()

        assert _calls(target, "GET", "ok") == 1.0
