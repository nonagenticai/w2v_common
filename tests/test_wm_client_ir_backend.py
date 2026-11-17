"""WMClient with ``LLM_BACKEND=ir`` (the default): inference-router's contract, on the wire."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from w2v_common.auth.identity_context import (
    clear_identity_context,
    set_correlation_context,
    set_identity_context,
    set_w2v_job_context,
)
from w2v_common.clients import ir_direct, vllm_direct
from w2v_common.clients import vllm_usage_report as report_mod
from w2v_common.clients.ir_direct import IRDirectClient, IRNotConfiguredError, IRSettings
from w2v_common.clients.wm_client import WMClient
from w2v_common.discovery import service_url
from w2v_common.tenant import TenantSource, UnvouchedTenantError

TOKEN = "ir-licence-SECRET-abc123"  # pragma: allowlist secret
SA_TOKEN = "sa-w2v-licence"  # pragma: allowlist secret
TENANT = "0199a1b2-0000-7000-8000-00000000000a"
RUN = "0199a1b2-1111-7000-8000-000000000001"

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _ir_env(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "LLM_BACKEND",
        "LLM_IR_BASE_URL",
        "LLM_IR_TOKEN",
        "LLM_IR_TOKEN_FILE",
        "LLM_IR_MODEL_DEFAULT",
        "LLM_IR_USAGE_REPORT_ENABLED",
        "W2V_INFERENCE_ROUTER_URL",
        "W2V_LOCAL",
        "CELL_SERVICE_NAME",
        "OTEL_SERVICE_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_IR_TOKEN", TOKEN)
    monkeypatch.setenv("DOWNSTREAM_LICENSE_TOKEN", SA_TOKEN)

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(vllm_direct, "_sleep", _no_sleep)
    clear_identity_context()
    yield
    clear_identity_context()


def _completion(
    content: str = "hi", *, finish_reason: str = "stop", usage: dict | None = None
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage if usage is not None else {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


def _client(captured: list[httpx.Request], handler: Handler | None = None) -> WMClient:
    def recording(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request) if handler is not None else _completion()

    return WMClient(transport=httpx.MockTransport(recording))


def _body(request: httpx.Request) -> dict:
    return json.loads(request.read())


def _bind(source: TenantSource | None = TenantSource.LICENSE_TOKEN) -> None:
    set_identity_context(TENANT, "lic-1", "subject-1", tenant_source=source)


# ── configuration ───────────────────────────────────────────────────────────


def test_ir_is_the_default_backend() -> None:
    client = WMClient()
    assert client.llm_backend == "ir"
    assert isinstance(client._llm, IRDirectClient)
    assert client._llm.settings.base_url == service_url("inference_router") + "/v1"
    assert client._llm.settings.model_for("anything") == "local-llm"


def test_wm_backend_is_refused_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "wm")
    with pytest.raises(ValueError, match=r"removed in w2v_common 0\.109\.0"):
        WMClient()


def test_explicit_ir_without_token_fails_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "ir")
    monkeypatch.setenv("LLM_IR_TOKEN", "  ")
    with pytest.raises(ValueError, match="LLM_IR_TOKEN"):
        WMClient()


async def test_defaulted_ir_without_token_fails_the_call_not_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_IR_TOKEN")
    client = WMClient()  # a consumer test building a client with no LLM env must not crash
    with pytest.raises(IRNotConfiguredError, match="LLM_IR_TOKEN"):
        await client.message("w", "hi")
    health = await client.health_check()
    assert health["status"] == "unhealthy"
    assert "LLM_IR_TOKEN" in health["error"]


def test_token_file_is_read_and_env_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    secret = tmp_path / "ir-token"
    secret.write_text("from-file-token\n", encoding="utf-8")
    monkeypatch.delenv("LLM_IR_TOKEN")
    monkeypatch.setenv("LLM_IR_TOKEN_FILE", str(secret))
    assert IRSettings.from_env().token == "from-file-token"
    monkeypatch.setenv("LLM_IR_TOKEN", "from-env")
    assert IRSettings.from_env().token == "from-env"


def test_unreadable_token_file_names_the_path_not_a_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("LLM_IR_TOKEN")
    monkeypatch.setenv("LLM_IR_TOKEN_FILE", str(tmp_path / "missing"))
    monkeypatch.setenv("LLM_BACKEND", "ir")
    with pytest.raises(ValueError, match="LLM_IR_TOKEN_FILE"):
        WMClient()


def test_a_base_url_outside_the_inference_router_host_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegation keys on the discovery host; another host would bill the SA silently."""
    monkeypatch.setenv("LLM_BACKEND", "ir")
    monkeypatch.setenv("LLM_IR_BASE_URL", "http://ir.elsewhere.test:8150/v1")
    with pytest.raises(ValueError, match="inference_router host"):
        WMClient()


def test_discovery_override_moves_the_default_and_stays_metered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("W2V_INFERENCE_ROUTER_URL", "http://ir.port-forward.test:18150")
    client = WMClient()
    assert client._llm is not None
    assert client._llm.settings.base_url == "http://ir.port-forward.test:18150/v1"


def test_settings_repr_hides_token() -> None:
    client = WMClient()
    assert client._llm is not None
    assert TOKEN not in repr(client._llm.settings)


# ── the request ─────────────────────────────────────────────────────────────


async def test_request_shape_auth_and_attribution_headers() -> None:
    captured: list[httpx.Request] = []
    _bind()
    set_w2v_job_context(RUN)
    set_correlation_context(RUN, "script")

    result = await _client(captured).message("promo_script", "hello", max_tokens=9000)

    assert result.output == "hi"
    req = captured[0]
    assert req.method == "POST"
    assert req.url.host == "inference-router.inference-router"
    assert req.url.path == "/v1/chat/completions"
    h = req.headers
    assert h["X-Tenant-Id"] == f"Bearer {TOKEN}"  # the IR licence, never the w2v SA one
    assert "authorization" not in h
    assert h["X-Effective-Tenant-Id"] == TENANT
    assert h["X-App-Id"] == "w2v"
    assert h["X-Worker-Name"] == "promo_script"
    assert h["X-Job-Id"] == RUN
    assert h["X-Subject-Id"] == "subject-1"
    assert h["X-License-Id"] == "lic-1"
    assert h["X-Correlation-Id"] == RUN
    assert h["X-Stage"] == "script"
    assert "X-W2V-Cell-Token" not in h  # IR is out of the cell
    body = _body(req)
    assert body["model"] == "local-llm"
    assert body["max_tokens"] == 9000


async def test_no_run_bound_means_no_job_id_header() -> None:
    captured: list[httpx.Request] = []
    await _client(captured).message("w", "hi")
    assert "X-Job-Id" not in captured[0].headers


@pytest.mark.parametrize(("flag", "sent"), [(None, False), (False, False), (True, True)])
async def test_enable_reasoning_is_top_level_and_chat_template_kwargs_never_sent(
    flag: bool | None, sent: bool
) -> None:
    captured: list[httpx.Request] = []
    await _client(captured).message("w", "hi", enable_reasoning=flag)
    body = _body(captured[0])
    assert body["enable_reasoning"] is sent
    assert "chat_template_kwargs" not in body


async def test_max_tokens_is_clamped_to_the_ir_ceiling() -> None:
    captured: list[httpx.Request] = []
    await _client(captured).message("w", "hi", max_tokens=100_000)
    assert _body(captured[0])["max_tokens"] == ir_direct.IR_MAX_OUTPUT_TOKENS == 16_384


async def test_schema_is_sent_as_json_schema_and_reply_validated_with_reask() -> None:
    captured: list[httpx.Request] = []
    replies = iter(['{"title": 3}', '{"title": "ok"}'])
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }

    result = await _client(captured, lambda r: _completion(next(replies))).message(
        "w", "hi", output_schema=schema
    )

    assert result.output == {"title": "ok"}
    assert len(captured) == 2
    first = _body(captured[0])
    assert first["response_format"]["type"] == "json_schema"
    assert first["response_format"]["json_schema"]["schema"]["required"] == ["title"]
    # The re-ask carries the rejected reply + the violations, and the same headers.
    second = _body(captured[1])
    assert second["messages"][-2] == {"role": "assistant", "content": '{"title": 3}'}
    assert "does not match the required JSON schema" in second["messages"][-1]["content"]
    assert captured[1].headers["X-Worker-Name"] == "w"


async def test_usage_comes_from_the_openai_usage_block_labelled_inference_router() -> None:
    captured: list[httpx.Request] = []
    replies = iter(["not json", '{"a": 1}'])
    result = await _client(
        captured,
        lambda r: _completion(next(replies), usage={"prompt_tokens": 10, "completion_tokens": 4}),
    ).message("w", "hi", output_schema={"type": "object"})

    usage = result.usage_metrics
    assert usage is not None
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (
        20,
        8,
        28,
    )
    assert usage["requests"] == 2
    assert usage["provider"] == usage["served_provider"] == "inference_router"
    assert usage["metered_by"] == "inference_router"
    assert "funding" not in usage  # IR meters it; this client claims nothing
    assert result.metadata is not None and result.metadata["backend"] == "ir"


async def test_unvouched_tenant_is_never_delegated_and_nothing_is_sent() -> None:
    captured: list[httpx.Request] = []
    _bind(source=None)
    with pytest.raises(UnvouchedTenantError):
        await _client(captured).message("w", "hi")
    assert captured == []


async def test_no_tenant_outside_a_request_rides_the_ir_licence_alone() -> None:
    captured: list[httpx.Request] = []
    await _client(captured).message("w", "hi")
    assert captured[0].headers["X-Tenant-Id"] == f"Bearer {TOKEN}"
    assert "X-Effective-Tenant-Id" not in captured[0].headers


# ── errors ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("status", [401, 403])
async def test_auth_rejection_surfaces_the_reason_and_never_the_token(
    status: int, caplog: pytest.LogCaptureFixture
) -> None:
    captured: list[httpx.Request] = []
    reason = "Caller licence could not be introspected; caller is not authorised"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": f"{reason} (token {TOKEN})"})

    with caplog.at_level(logging.ERROR), pytest.raises(httpx.HTTPStatusError) as ei:
        await _client(captured, handler).message("w", "hi")

    assert len(captured) == 1  # auth is not retried
    err = ei.value
    assert err.response.status_code == status
    assert f"inference_router returned {status}" in str(err)
    assert reason in str(err)
    assert TOKEN not in str(err)
    assert TOKEN not in err.response.text
    assert "x-tenant-id" not in err.request.headers
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert reason in logged
    assert "app `inference-router`" in logged
    assert TOKEN not in logged


async def test_transport_error_carries_no_credential() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused {TOKEN}", request=request)

    with pytest.raises(httpx.ConnectError) as ei:
        await _client([], handler).message("w", "hi")
    assert TOKEN not in str(ei.value)
    assert "x-tenant-id" not in ei.value.request.headers


async def test_finish_reason_length_is_a_422() -> None:
    with pytest.raises(httpx.HTTPStatusError) as ei:
        await _client([], lambda r: _completion("cut", finish_reason="length")).message("w", "hi")
    assert ei.value.response.status_code == 422
    assert "finish_reason=length" in str(ei.value)


async def test_health_check_probes_models_with_the_ir_licence() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "local-llm"}]})

    health = await _client(captured, handler).health_check()
    assert health == {"status": "healthy", "service": "inference_router", "models": ["local-llm"]}
    assert captured[0].url.path == "/v1/models"
    assert captured[0].headers["X-Tenant-Id"] == f"Bearer {TOKEN}"


# ── run usage report ────────────────────────────────────────────────────────


def test_no_run_usage_report_by_default() -> None:
    """IR's own cost event reaches the run via the billing mirror; a report double counts."""
    client = WMClient()
    assert client._llm is not None
    assert client._llm.usage_reporter is None


async def test_opt_in_report_is_labelled_inference_router(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_IR_USAGE_REPORT_ENABLED", "true")
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/usage"):
            return httpx.Response(202, json={"recorded": True})
        return _completion()

    client = _client(captured, handler)
    assert client._llm is not None and client._llm.usage_reporter is not None
    _bind()
    set_w2v_job_context(RUN)
    await client.message("w", "hi")
    await report_mod.drain_pending_usage_reports(timeout=2.0)

    reports = [r for r in captured if r.url.path.endswith("/usage")]
    assert len(reports) == 1
    body = _body(reports[0])
    assert body["provider"] == "inference_router"
    assert "funding" not in body  # the orchestrator's RunUsageReport is extra="forbid"
    assert reports[0].headers["X-Tenant-Id"] == f"Bearer {SA_TOKEN}"  # never the IR licence
