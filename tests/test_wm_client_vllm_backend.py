"""WMClient with ``LLM_BACKEND=vllm``: direct pgx vLLM calls, WM-compatible surface."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import cast

import httpx
import pytest

from w2v_common.clients import vllm_direct
from w2v_common.clients.vllm_direct import (
    VLLM_MAX_OUTPUT_TOKENS,
    get_worker_defaults,
    known_workers,
    output_contract,
)
from w2v_common.clients.wm_client import WM_MIN_OUTPUT_TOKENS, WMClient, WMResponse

TOKEN = "vllm-licence-SECRET-abc123"
BASE = "http://vllm.test:8001/v1"

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _vllm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LLM_BACKEND",
        "LLM_VLLM_BASE_URL",
        "LLM_VLLM_TOKEN",
        "LLM_VLLM_MODEL_DEFAULT",
        "LLM_VLLM_MODEL_FAST",
        "LLM_VLLM_FAST_WORKERS",
        "WM_AUTH_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_VLLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_VLLM_TOKEN", TOKEN)

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(vllm_direct, "_sleep", _no_sleep)


def _completion(content: str, *, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


def _client(
    handler: Handler, captured: list[httpx.Request], monkeypatch: pytest.MonkeyPatch
) -> WMClient:
    monkeypatch.setenv("LLM_BACKEND", "vllm")

    def recording(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    return WMClient(base_url="http://wm.test", transport=httpx.MockTransport(recording))


def _body(request: httpx.Request) -> dict:
    return json.loads(request.read())


# ── backend selection ───────────────────────────────────────────────────────


def test_vllm_backend_is_selected_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    client = WMClient(base_url="http://wm.test")
    assert client.llm_backend == "vllm"
    assert type(client._llm) is vllm_direct.VLLMDirectClient


def test_unknown_backend_fails_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "vlm")
    with pytest.raises(ValueError, match="LLM_BACKEND"):
        WMClient(base_url="http://wm.test")


def test_vllm_without_token_fails_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    monkeypatch.setenv("LLM_VLLM_TOKEN", "  ")
    with pytest.raises(ValueError, match="LLM_VLLM_TOKEN"):
        WMClient(base_url="http://wm.test")


def test_settings_repr_hides_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    client = WMClient(base_url="http://wm.test")
    assert client._llm is not None
    assert TOKEN not in repr(client._llm.settings)


def test_default_base_url_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_VLLM_BASE_URL")
    settings = vllm_direct.VLLMSettings.from_env()
    assert settings.base_url == "http://localhost:8001/v1"
    assert settings.model_for("anything") == "local-llm"


def test_fast_workers_route_to_fast_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_VLLM_FAST_WORKERS", " critique-summary , ,hook-doctor")
    monkeypatch.setenv("LLM_VLLM_MODEL_FAST", "fast-llm")
    settings = vllm_direct.VLLMSettings.from_env()
    assert settings.fast_workers == frozenset({"critique-summary", "hook-doctor"})
    assert settings.model_for("hook-doctor") == "fast-llm"
    assert settings.model_for("brand-analysis") == "local-llm"
    monkeypatch.setenv("LLM_VLLM_MODEL_FAST", "")
    assert vllm_direct.VLLMSettings.from_env().model_for("hook-doctor") == "local-llm"


# ── the request ─────────────────────────────────────────────────────────────


async def test_routes_to_chat_completions_with_bearer_and_no_tenant_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[httpx.Request] = []
    client = _client(lambda _r: _completion("plain answer"), captured, monkeypatch)
    result = await client.message(
        "some-unseeded-worker",
        "hello",
        model_override="gpt-x",
        work_tier="high",
        product="w2v",
        project_id="not-a-uuid-and-ignored",
    )
    assert isinstance(result, WMResponse)
    assert result.output == "plain answer"
    assert result.output_type == "string"
    req = captured[0]
    assert str(req.url) == f"{BASE}/chat/completions"
    assert req.headers["Authorization"] == f"Bearer {TOKEN}"
    assert "x-tenant-id" not in req.headers
    assert "x-effective-tenant-id" not in req.headers
    body = _body(req)
    assert body["model"] == "local-llm"
    assert body["temperature"] == 0.1
    assert "response_format" not in body
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert result.metadata is not None
    assert result.metadata["backend"] == "vllm"
    assert result.metadata["seed_prompt_applied"] is False
    assert result.metadata["overrides_ignored"] == {
        "model_override": "gpt-x",
        "work_tier": "high",
        "product": "w2v",
    }
    assert result.usage_metrics is not None
    assert result.usage_metrics["funding"] == "platform_direct"
    assert result.usage_metrics["input_tokens"] == 11


async def test_usage_metrics_carry_wm_key_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """WM's TokenUsageMetrics names: readers use prompt/completion_tokens, never input/output."""
    captured: list[httpx.Request] = []
    client = _client(lambda _r: _completion("x"), captured, monkeypatch)
    usage = (await client.message("brand-analysis", "hello")).usage_metrics
    assert usage is not None
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 7
    assert usage["total_tokens"] == 18
    assert usage["served_model"] == "local-llm"
    assert usage["served_provider"] == "vllm"
    assert usage["requested_model"] == "local-llm"
    assert (usage["input_tokens"], usage["output_tokens"]) == (11, 7)


async def test_seed_prompt_then_call_prompt_as_system_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[httpx.Request] = []
    client = _client(lambda _r: _completion("ok"), captured, monkeypatch)
    result = await client.message("critique-summary", "the page", system_prompt="Be brief.")
    seed = get_worker_defaults("critique-summary")
    assert seed.seeded
    assert seed.system_prompt
    body = _body(captured[0])
    assert body["temperature"] == 0.0  # seeded at 0, not the 0.1 auto-create default
    assert body["messages"][0] == {
        "role": "system",
        "content": f"{seed.system_prompt}\n\nBe brief.",
    }
    # The per-call prompt is NOT folded into the user content on this backend.
    assert body["messages"][1] == {"role": "user", "content": "the page"}
    assert result.metadata is not None
    assert result.metadata["seed_prompt_applied"] is True


def test_seed_only_workers_are_vendored() -> None:
    needed = {
        "analysis-site-summary",
        "brand-analysis",
        "icp-analysis",
        "audit-claim-grounding",
        "audit-recommendation-writer",
        "audit-report-critic",
        "audit-exec-summary",
        "redesign-accessibility-scorer",
        "slide-bullet-extractor",
        "edit-plan-generator",
        "critique-summary",
        "critique-ux-review",
        "redesign-html",
        "course-curriculum-generator",
    }
    assert needed <= known_workers()
    for name in needed:
        assert get_worker_defaults(name).system_prompt, name


async def test_schema_grammar_is_sent_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[httpx.Request] = []
    client = _client(lambda _r: _completion('{"title": "x", "n": 3}'), captured, monkeypatch)
    schema = {
        "type": "object",
        "title": "Out",
        "properties": {
            "title": {"type": "string", "maxLength": 80},
            "n": {"type": "integer", "minimum": 1},
        },
        "required": ["title", "n"],
    }
    result = await client.message("brand-analysis", "site", output_schema=schema)
    assert result.output == {"title": "x", "n": 3}
    assert result.output_type == "object"
    assert result.output_schema_used == schema
    rf = _body(captured[0])["response_format"]
    assert rf["type"] == "json_schema"
    grammar = rf["json_schema"]["schema"]
    assert "title" not in grammar  # keyword dropped …
    assert set(grammar["properties"]) == {"title", "n"}  # … but not the property NAME
    assert grammar["properties"]["title"] == {"type": "string"}
    assert grammar["properties"]["n"] == {"type": "integer"}


def test_prefix_items_survive_the_strip() -> None:
    schema = {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "prefixItems": [{"type": "string", "minLength": 1}],
                "items": {"type": "string"},
                "minItems": 1,
            }
        },
        "required": ["lines"],
    }
    grammar = output_contract(schema).grammar
    assert grammar is not None
    props = cast("dict[str, dict[str, object]]", grammar["properties"])
    lines = props["lines"]
    assert lines["prefixItems"] == [{"type": "string"}]
    assert "minItems" not in lines


async def test_non_object_root_is_wrapped_and_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[httpx.Request] = []
    client = _client(lambda _r: _completion('{"response": ["a", "b"]}'), captured, monkeypatch)
    result = await client.message(
        "w", "x", output_schema={"type": "array", "items": {"type": "string"}, "maxItems": 1}
    )
    assert result.output == ["a", "b"]
    assert result.output_type == "array"
    assert result.usage_metrics is not None
    assert result.usage_metrics["bound_violations"][0]["keyword"] == "maxItems"


async def test_reask_on_invalid_json_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter([_completion("not json {"), _completion('{"a": 1}')])
    captured: list[httpx.Request] = []
    client = _client(lambda _r: next(replies), captured, monkeypatch)
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    result = await client.message("w", "x", output_schema=schema)
    assert result.output == {"a": 1}
    assert len(captured) == 2
    second = _body(captured[1])["messages"]
    assert second[-2] == {"role": "assistant", "content": "not json {"}
    assert "not valid JSON" in second[-1]["content"]
    assert result.usage_metrics is not None
    assert result.usage_metrics["requests"] == 2


async def test_schema_retries_exhausted_raise_422(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[httpx.Request] = []
    client = _client(lambda _r: _completion('{"a": "nope"}'), captured, monkeypatch)
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    with pytest.raises(httpx.HTTPStatusError) as info:
        await client.message("w", "x", output_schema=schema)
    assert info.value.response.status_code == 422
    assert "Exceeded maximum output retries (3)" in str(info.value)
    assert "a: 'nope' is not of type 'integer'" in info.value.response.text
    assert len(captured) == vllm_direct.OUTPUT_RETRIES + 1


async def test_finish_reason_length_raises_422(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[httpx.Request] = []
    client = _client(lambda _r: _completion("", finish_reason="length"), captured, monkeypatch)
    with pytest.raises(httpx.HTTPStatusError) as info:
        await client.message("w", "x")
    assert info.value.response.status_code == 422
    assert "token limit" in str(info.value)
    assert len(captured) == 1


async def test_401_raises_http_status_error_without_token(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    captured: list[httpx.Request] = []
    echo = f'{{"detail": "invalid licence {TOKEN}", "auth": "Bearer {TOKEN}"}}'
    client = _client(lambda _r: httpx.Response(401, text=echo), captured, monkeypatch)
    with caplog.at_level(logging.DEBUG), pytest.raises(httpx.HTTPStatusError) as info:
        await client.message("w", "x")
    exc = info.value
    assert exc.response.status_code == 401
    assert TOKEN not in str(exc)
    assert TOKEN not in exc.response.text
    assert "authorization" not in exc.request.headers
    assert TOKEN not in caplog.text
    assert len(captured) == 1  # auth is never retried


async def test_5xx_is_retried_then_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[httpx.Request] = []
    client = _client(lambda _r: httpx.Response(503, text="overloaded"), captured, monkeypatch)
    with pytest.raises(httpx.HTTPStatusError) as info:
        await client.message("w", "x")
    assert info.value.response.status_code == 503
    assert "overloaded" in str(info.value)
    assert len(captured) == 3


async def test_429_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter([httpx.Response(429, headers={"Retry-After": "1"}), _completion("ok")])
    captured: list[httpx.Request] = []
    client = _client(lambda _r: next(replies), captured, monkeypatch)
    assert (await client.message("w", "x")).output == "ok"
    assert len(captured) == 2


async def test_transport_error_is_retried_then_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[httpx.Request] = []

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = _client(boom, captured, monkeypatch)
    with pytest.raises(httpx.ConnectError) as exc_info:
        await client.message("w", "x")
    assert len(captured) == 3
    # The re-raised error carries a token-free request (a consumer logging
    # exc.request.headers must not leak the bearer).
    assert "authorization" not in exc_info.value.request.headers
    assert TOKEN not in repr(exc_info.value.request.headers)


async def test_non_json_200_raises_http_status_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(lambda _r: httpx.Response(200, text="<html>oops"), [], monkeypatch)
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.message("w", "x")
    assert exc_info.value.response.status_code == 502


async def test_content_blocks_become_image_url_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[httpx.Request] = []
    client = _client(lambda _r: _completion("a cat"), captured, monkeypatch)
    await client.message(
        "w",
        content_blocks=[
            {"type": "text", "text": "describe"},
            {"type": "image", "data": "QUJD", "media_type": "image/png"},
            {"type": "image", "data": "REVG"},
        ],
    )
    content = _body(captured[0])["messages"][-1]["content"]
    assert content == [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,REVG"}},
    ]


@pytest.mark.parametrize(("flag", "expected"), [(None, False), (False, False), (True, True)])
async def test_enable_reasoning_maps_to_enable_thinking(
    monkeypatch: pytest.MonkeyPatch, flag: bool | None, expected: bool
) -> None:
    captured: list[httpx.Request] = []
    client = _client(lambda _r: _completion("ok"), captured, monkeypatch)
    await client.message("w", "x", enable_reasoning=flag)
    assert _body(captured[0])["chat_template_kwargs"] == {"enable_thinking": expected}


@pytest.mark.parametrize(
    ("asked", "sent"),
    [
        (None, WM_MIN_OUTPUT_TOKENS),
        (100, WM_MIN_OUTPUT_TOKENS),
        (12_000, 12_000),
        (64_000, VLLM_MAX_OUTPUT_TOKENS),
    ],
)
async def test_max_tokens_floor_and_clamp(
    monkeypatch: pytest.MonkeyPatch, asked: int | None, sent: int
) -> None:
    captured: list[httpx.Request] = []
    client = _client(lambda _r: _completion("ok"), captured, monkeypatch)
    await client.message("w", "x", max_tokens=asked)
    assert _body(captured[0])["max_tokens"] == sent


async def test_message_and_blocks_together_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(lambda _r: _completion("ok"), [], monkeypatch)
    with pytest.raises(ValueError, match="exactly one"):
        await client.message("w", "x", content_blocks=[{"type": "text", "text": "y"}])


# ── health ──────────────────────────────────────────────────────────────────


async def test_health_check_lists_models_with_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[httpx.Request] = []
    client = _client(
        lambda _r: httpx.Response(200, json={"data": [{"id": "local-llm"}]}), captured, monkeypatch
    )
    health = await client.health_check()
    assert health == {"status": "healthy", "service": "vllm", "models": ["local-llm"]}
    assert str(captured[0].url) == f"{BASE}/models"
    assert captured[0].headers["Authorization"] == f"Bearer {TOKEN}"
    await client.aclose()


async def test_health_check_reports_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(lambda _r: httpx.Response(403, text=f"bad {TOKEN}"), [], monkeypatch)
    health = await client.health_check()
    assert health["status"] == "unhealthy"
    assert TOKEN not in str(health)
