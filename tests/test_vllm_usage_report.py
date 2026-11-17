"""vLLM-direct usage reports to the orchestrator's ``POST /api/runs/{run_id}/usage``."""

from __future__ import annotations

import asyncio
import gc
import json
from collections.abc import Callable, Iterator

import httpx
import pytest
from prometheus_client import REGISTRY

from w2v_common.auth.identity_context import (
    clear_identity_context,
    set_correlation_context,
    set_identity_context,
    set_w2v_job_context,
)
from w2v_common.clients import vllm_direct, vllm_usage_report
from w2v_common.clients.vllm_usage_report import (
    ENABLED_ENV,
    VLLMUsageReporter,
    drain_pending_usage_reports,
    usage_report_enabled,
)
from w2v_common.clients.wm_client import WMClient
from w2v_common.discovery import service_url
from w2v_common.tenant import TenantSource

TOKEN = "vllm-licence-SECRET-abc123"
SA_LICENCE = "sa-licence-SECRET-xyz"
BASE = "http://vllm.test:8001/v1"
TENANT = "0195f48b-0000-7000-8000-000000000001"
JOB = "0195f48b-0000-7000-8000-0000000000aa"
CORR = "0195f48b-0000-7000-8000-0000000000bb"

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        "LLM_VLLM_BASE_URL",
        "LLM_VLLM_MODEL_DEFAULT",
        "LLM_VLLM_MODEL_FAST",
        "LLM_VLLM_FAST_WORKERS",
        "WM_AUTH_MODE",
        "DOWNSTREAM_LICENSE_TOKEN",
        "CELL_SERVICE_NAME",
        "OTEL_SERVICE_NAME",
        ENABLED_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    monkeypatch.setenv("LLM_VLLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_VLLM_TOKEN", TOKEN)
    monkeypatch.setenv("CELL_SERVICE_NAME", "analysis-service")

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(vllm_direct, "_sleep", _no_sleep)
    monkeypatch.setattr(vllm_usage_report, "RETRY_DELAY_S", 0.0)
    clear_identity_context()
    yield
    clear_identity_context()


def _completion() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


def _is_report(request: httpx.Request) -> bool:
    return request.url.path.startswith("/api/runs/")


def _client(report: Handler | None = None) -> tuple[WMClient, list[httpx.Request]]:
    """WMClient on the vLLM backend; vLLM answers, ``report`` answers the orchestrator."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if _is_report(request):
            return report(request) if report else httpx.Response(202, json={"recorded": True})
        return _completion()

    client = WMClient(
        base_url="http://wm.test",
        license_token=SA_LICENCE,
        transport=httpx.MockTransport(handler),
    )
    return client, captured


def _reports(captured: list[httpx.Request]) -> list[httpx.Request]:
    return [r for r in captured if _is_report(r)]


def _bind_run(*, job: str | None = JOB, corr: str | None = None, stage: str | None = "script"):
    set_identity_context(TENANT, "lic-1", "user-42", tenant_source=TenantSource.LICENSE_TOKEN)
    if job:
        set_w2v_job_context(job)
    set_correlation_context(corr, stage)


def _count(outcome: str) -> float:
    value = REGISTRY.get_sample_value("w2v_vllm_usage_report_total", {"outcome": outcome})
    return value or 0.0


# ── enablement ──────────────────────────────────────────────────────────────


def test_enabled_by_default() -> None:
    assert usage_report_enabled() is True


@pytest.mark.parametrize("value", ["false", "0", "no", "OFF", " False "])
def test_env_switch_disables(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ENABLED_ENV, value)
    assert usage_report_enabled() is False


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("CELL_SERVICE_NAME", "w2v-orchestrator"),
        ("CELL_SERVICE_NAME", "orchestrator"),
    ],
)
def test_orchestrator_never_reports(monkeypatch: pytest.MonkeyPatch, var: str, value: str) -> None:
    monkeypatch.setenv(var, value)
    assert usage_report_enabled() is False


def test_orchestrator_by_otel_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CELL_SERVICE_NAME")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "w2v-orchestrator")
    assert usage_report_enabled() is False


# ── the report ──────────────────────────────────────────────────────────────


async def test_report_sent_with_body_and_headers_when_job_bound() -> None:
    client, captured = _client()
    _bind_run(corr=CORR)
    result = await client.message("brand-analysis", "hello")
    await drain_pending_usage_reports()
    assert result.output == "answer"
    (report,) = _reports(captured)
    assert report.method == "POST"
    assert str(report.url) == f"{service_url('orchestrator')}/api/runs/{JOB}/usage"
    body = json.loads(report.read())
    assert set(body) == {
        "call_id",
        "stage",
        "model",
        "provider",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "service",
    }
    assert body["stage"] == "script"
    assert body["model"] == "local-llm"
    assert body["provider"] == "vllm"
    assert (body["input_tokens"], body["output_tokens"], body["total_tokens"]) == (11, 7, 18)
    assert body["service"] == "analysis-service"
    assert len(body["call_id"]) == 36
    # Canonical S2S shape: SA licence in X-Tenant-Id Bearer + identity hook headers.
    assert report.headers["X-Tenant-Id"] == f"Bearer {SA_LICENCE}"
    assert report.headers["X-License-Id"] == "lic-1"
    assert report.headers["X-Subject-Id"] == "user-42"
    assert report.headers["X-Correlation-Id"] == CORR
    assert report.headers["X-Stage"] == "script"
    assert report.headers["X-W2V-Job-Id"] == JOB
    assert report.headers["X-Effective-Tenant-Id"] == TENANT
    # The vLLM licence never travels to the orchestrator.
    assert TOKEN not in str(report.headers)
    assert "authorization" not in report.headers
    await client.close()


async def test_falls_back_to_correlation_id() -> None:
    client, captured = _client()
    _bind_run(job=None, corr=CORR, stage=None)
    await client.message("brand-analysis", "hello")
    await drain_pending_usage_reports()
    (report,) = _reports(captured)
    assert report.url.path == f"/api/runs/{CORR}/usage"
    assert json.loads(report.read())["stage"] is None
    await client.close()


async def test_not_sent_without_run_id() -> None:
    client, captured = _client()
    set_identity_context(TENANT, "lic-1", "user-42")
    result = await client.message("brand-analysis", "hello")
    await drain_pending_usage_reports()
    assert result.output == "answer"
    assert _reports(captured) == []
    await client.close()


async def test_not_sent_when_run_id_malformed() -> None:
    client, captured = _client()
    _bind_run(job=None, corr="../../admin?x=1", stage=None)
    await client.message("brand-analysis", "hello")
    await drain_pending_usage_reports()
    assert _reports(captured) == []
    await client.close()


async def test_not_sent_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENABLED_ENV, "false")
    client, captured = _client()
    _bind_run()
    await client.message("brand-analysis", "hello")
    await drain_pending_usage_reports()
    assert _reports(captured) == []
    await client.close()


async def test_not_sent_from_the_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CELL_SERVICE_NAME", "w2v-orchestrator")
    client, captured = _client()
    _bind_run()
    await client.message("brand-analysis", "hello")
    await drain_pending_usage_reports()
    assert _reports(captured) == []
    await client.close()


async def test_not_sent_when_llm_call_fails() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if _is_report(request):
            return httpx.Response(202, json={"recorded": True})
        return httpx.Response(400, json={"detail": "bad"})

    client = WMClient(base_url="http://wm.test", transport=httpx.MockTransport(handler))
    _bind_run()
    with pytest.raises(httpx.HTTPStatusError):
        await client.message("brand-analysis", "hello")
    await drain_pending_usage_reports()
    assert _reports(captured) == []
    await client.close()


# ── failure handling ────────────────────────────────────────────────────────


async def test_http_error_swallowed_and_retried_once_with_same_call_id() -> None:
    before = _count("rejected")
    client, captured = _client(lambda _r: httpx.Response(503, json={"detail": "down"}))
    _bind_run()
    result = await client.message("brand-analysis", "hello")
    await drain_pending_usage_reports()
    assert result.output == "answer"
    reports = _reports(captured)
    assert len(reports) == 2  # one retry, no more
    ids = {json.loads(r.read())["call_id"] for r in reports}
    assert len(ids) == 1
    assert _count("rejected") == before + 1
    await client.close()


async def test_4xx_not_retried() -> None:
    client, captured = _client(lambda _r: httpx.Response(404, json={"detail": "no run"}))
    _bind_run()
    await client.message("brand-analysis", "hello")
    await drain_pending_usage_reports()
    assert len(_reports(captured)) == 1
    await client.close()


async def test_timeout_swallowed_and_logged_without_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    before = _count("failed")

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client, captured = _client(boom)
    _bind_run()
    with caplog.at_level("WARNING", logger=vllm_usage_report.__name__):
        result = await client.message("brand-analysis", "hello")
        await drain_pending_usage_reports()
    assert result.output == "answer"
    reports = _reports(captured)
    assert len(reports) == 2
    assert len({json.loads(r.read())["call_id"] for r in reports}) == 1
    assert _count("failed") == before + 1
    text = caplog.text
    assert "ReadTimeout" in text
    assert SA_LICENCE not in text
    assert TOKEN not in text
    assert "http://" not in text
    await client.close()


async def test_each_llm_call_gets_its_own_call_id() -> None:
    client, captured = _client()
    _bind_run()
    await client.message("brand-analysis", "one")
    await client.message("brand-analysis", "two")
    await drain_pending_usage_reports()
    ids = [json.loads(r.read())["call_id"] for r in _reports(captured)]
    assert len(ids) == 2
    assert ids[0] != ids[1]
    await client.close()


async def test_report_does_not_delay_the_llm_call() -> None:
    gate = asyncio.Event()
    seen: list[httpx.Request] = []

    class SlowReport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if _is_report(request):
                await gate.wait()
                return httpx.Response(202, json={"recorded": True})
            return _completion()

    client = WMClient(base_url="http://wm.test", transport=SlowReport())
    _bind_run()
    result = await asyncio.wait_for(client.message("brand-analysis", "hello"), timeout=2)
    assert result.output == "answer"
    gate.set()
    await drain_pending_usage_reports()
    assert len(_reports(seen)) == 1
    await client.close()


async def test_pending_task_is_strongly_referenced() -> None:
    gate = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        await gate.wait()
        return httpx.Response(202, json={"recorded": True})

    reporter = VLLMUsageReporter(transport=httpx.MockTransport(handler), enabled=True)
    _bind_run()
    before = _count("sent")
    assert reporter.schedule(call_id="c-1", model="m", input_tokens=1, output_tokens=2)
    gc.collect()
    pending = [t for t in vllm_usage_report._background_tasks if not t.done()]
    assert len(pending) == 1
    gate.set()
    await drain_pending_usage_reports()
    assert _count("sent") == before + 1
    assert not vllm_usage_report._background_tasks  # discarded on completion
    await reporter.close()


def test_schedule_without_loop_is_dropped_not_raised() -> None:
    reporter = VLLMUsageReporter(enabled=True)
    _bind_run()
    before = _count("dropped_no_loop")
    assert reporter.schedule(call_id="c", model="m", input_tokens=1, output_tokens=1) is False
    assert _count("dropped_no_loop") == before + 1


async def test_schedule_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> str:
        raise RuntimeError("context exploded")

    monkeypatch.setattr(vllm_usage_report, "resolve_run_id", broken)
    reporter = VLLMUsageReporter(enabled=True)
    assert reporter.schedule(call_id="c", model="m", input_tokens=1, output_tokens=1) is False


async def test_not_sent_without_bound_tenant() -> None:
    before = _count("dropped_no_tenant")
    client, captured = _client()
    set_w2v_job_context(JOB)  # a run id but no tenant
    result = await client.message("brand-analysis", "hello")
    await drain_pending_usage_reports()
    assert result.output == "answer"
    assert _reports(captured) == []
    assert _count("dropped_no_tenant") == before + 1
    await client.close()


async def test_not_sent_for_unvouched_tenant() -> None:
    before = _count("dropped_unvouched")
    client, captured = _client()
    set_identity_context(TENANT, "lic-1", "user-42", tenant_source=None)
    set_w2v_job_context(JOB)
    await client.message("brand-analysis", "hello")
    await drain_pending_usage_reports()
    assert _reports(captured) == []
    assert _count("dropped_unvouched") == before + 1
    await client.close()
