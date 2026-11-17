"""Tests for w2v_common.clients.wm_client — the surface every backend shares.

Exercised over the DEFAULT backend (``ir``, inference-router) with the REAL pooled
client and event hooks around a mock transport. IR-specific contract tests live
in ``test_wm_client_ir_backend.py``; the rollback backend in
``test_wm_client_vllm_backend.py``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from w2v_common.auth.identity_context import (
    clear_identity_context,
    set_identity_context,
)
from w2v_common.clients import vllm_direct
from w2v_common.clients.wm_client import WM_MIN_OUTPUT_TOKENS, WMClient, WMResponse
from w2v_common.tenant import TenantSource

IR_TOKEN = "ir-licence-SECRET-xyz789"  # pragma: allowlist secret

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "LLM_BACKEND",
        "LLM_IR_BASE_URL",
        "LLM_IR_TOKEN_FILE",
        "LLM_IR_MODEL_DEFAULT",
        "LLM_IR_USAGE_REPORT_ENABLED",
        "W2V_INFERENCE_ROUTER_URL",
        "W2V_LOCAL",
        "WM_AUTH_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_IR_TOKEN", IR_TOKEN)

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(vllm_direct, "_sleep", _no_sleep)
    clear_identity_context()
    yield
    clear_identity_context()


def _ok(content: str = '{"x": 1}') -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    )


def _client(
    captured: list[httpx.Request],
    handler: Handler | None = None,
    **kwargs: Any,
) -> WMClient:
    def recording(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request) if handler is not None else _ok("hi")

    return WMClient(transport=httpx.MockTransport(recording), **kwargs)


def _body(request: httpx.Request) -> dict:
    return json.loads(request.read())


def _bind(tenant: str, licence: str | None = "lic-1", subject: str | None = "user-42") -> None:
    set_identity_context(tenant, licence, subject, tenant_source=TenantSource.LICENSE_TOKEN)


async def test_message_returns_a_wm_response():
    captured: list[httpx.Request] = []
    client = _client(captured, lambda r: _ok('{"x": 1}'))

    result = await client.message(
        worker_name="course-curriculum-generator",
        message="Plan a course about Foo",
        system_prompt="You are a curriculum architect",
        output_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
    )

    assert isinstance(result, WMResponse)
    assert result.output == {"x": 1}
    assert result.output_type == "object"
    assert len(captured) == 1
    assert captured[0].url.path == "/v1/chat/completions"
    body = _body(captured[0])
    assert body["messages"][-1] == {"role": "user", "content": "Plan a course about Foo"}
    assert "curriculum architect" in body["messages"][0]["content"]


async def test_identity_headers_injected_per_call_from_contextvars():
    captured: list[httpx.Request] = []
    client = _client(captured)
    _bind("0195f48b-0000-7000-8000-000000000001")

    await client.message(worker_name="w", message="hello")

    headers = captured[0].headers
    assert headers["x-tenant-id"] == f"Bearer {IR_TOKEN}"
    assert headers["x-effective-tenant-id"] == "0195f48b-0000-7000-8000-000000000001"
    assert headers["x-license-id"] == "lic-1"
    assert headers["x-subject-id"] == "user-42"


async def test_raises_on_http_error():
    captured: list[httpx.Request] = []
    client = _client(captured, lambda r: httpx.Response(400, json={"detail": "boom"}))

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.message(worker_name="w", message="hello")

    # The response body must survive into the raised error.
    assert "boom" in str(exc_info.value)
    assert "400" in str(exc_info.value)
    assert exc_info.value.response.status_code == 400


async def test_error_body_redacts_bearer_token():
    """A token echoed in an error body must not leak into the raised message."""
    captured: list[httpx.Request] = []
    client = _client(
        captured,
        lambda r: httpx.Response(401, json={"detail": f"rejected Bearer {IR_TOKEN} abc.def"}),
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.message(worker_name="w", message="hello")

    assert IR_TOKEN not in str(exc_info.value)
    assert IR_TOKEN not in exc_info.value.response.text
    assert "x-tenant-id" not in exc_info.value.request.headers


async def test_wm_base_url_still_resolves_for_the_tenant_config_mirror(monkeypatch):
    """The inherited WM client (orchestrator tenant-config mirror only) keeps its URL rule."""
    from w2v_common.discovery import service_url

    monkeypatch.delenv("WORKER_MANAGEMENT_URL", raising=False)
    monkeypatch.delenv("W2V_WORKER_MANAGEMENT_URL", raising=False)
    assert WMClient().base_url == service_url("worker_management")
    monkeypatch.setenv("WORKER_MANAGEMENT_URL", "http://wm-override.test:9999")
    assert WMClient().base_url == "http://wm-override.test:9999"


async def test_message_with_content_blocks_multimodal():
    captured: list[httpx.Request] = []
    client = _client(captured)

    await client.message(
        worker_name="vision",
        content_blocks=[
            {"type": "text", "text": "describe"},
            {"type": "image", "data": "QUJD", "media_type": "image/png"},
        ],
    )

    content = _body(captured[0])["messages"][-1]["content"]
    assert content == [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
    ]


async def test_wm_routing_knobs_are_recorded_not_sent():
    captured: list[httpx.Request] = []
    client = _client(captured)

    result = await client.message(
        worker_name="w", message="hi", work_tier="high", product="promo", model_override="x"
    )

    body = _body(captured[0])
    assert "work_tier" not in body
    assert "product" not in body
    assert body["model"] == "local-llm"
    assert result.metadata is not None
    assert result.metadata["overrides_ignored"] == {
        "model_override": "x",
        "work_tier": "high",
        "product": "promo",
    }


async def test_no_budget_sends_the_floor():
    captured: list[httpx.Request] = []
    await _client(captured).message(worker_name="w", message="hi")
    assert _body(captured[0])["max_tokens"] == WM_MIN_OUTPUT_TOKENS


async def test_an_explicit_budget_above_the_floor_is_sent_unchanged():
    """The floor is a floor, not a clamp — a caller that asks for more gets more."""
    captured: list[httpx.Request] = []
    await _client(captured).message(worker_name="w", message="hi", max_tokens=12_000)
    assert _body(captured[0])["max_tokens"] == 12_000


async def test_an_explicit_budget_below_the_floor_is_raised_and_logged(caplog):
    """A caller reasoning from the REFUTED pre-flight mechanism must not be silently obeyed."""
    captured: list[httpx.Request] = []
    client = _client(captured)

    with caplog.at_level(logging.WARNING, logger="w2v_common.clients.wm_client"):
        await client.message(worker_name="segment-selector", message="hi", max_tokens=1760)

    assert _body(captured[0])["max_tokens"] == WM_MIN_OUTPUT_TOKENS
    assert any("below the" in r.message and "floor" in r.message for r in caplog.records), (
        "raising a caller's explicit budget must be logged, not silent"
    )


def test_the_floor_is_above_the_value_that_actually_fails():
    """6000 was WM's own default AND the observed failure — the floor must clear it."""
    assert WM_MIN_OUTPUT_TOKENS > 6_000


async def test_message_requires_one_of_message_or_content_blocks():
    with pytest.raises(ValueError, match="must be provided"):
        await _client([]).message(worker_name="w")


async def test_message_rejects_both_message_and_content_blocks():
    with pytest.raises(ValueError, match="exactly one"):
        await _client([]).message(
            worker_name="w",
            message="hi",
            content_blocks=[{"type": "text", "text": "hi"}],
        )


def test_default_timeout_is_600_when_env_unset(monkeypatch):
    monkeypatch.delenv("WM_CLIENT_TIMEOUT_S", raising=False)
    assert WMClient().timeout == 600.0


def test_env_overrides_default_timeout(monkeypatch):
    monkeypatch.setenv("WM_CLIENT_TIMEOUT_S", "300")
    assert WMClient().timeout == 300.0


def test_explicit_timeout_above_600_is_not_clamped(monkeypatch):
    """REGRESSION GUARD: the orchestrator passes 660s; it must NOT be clamped."""
    monkeypatch.delenv("WM_CLIENT_TIMEOUT_S", raising=False)
    client = WMClient(timeout=660.0)
    assert client.timeout == 660.0
    assert client._llm is not None and client._llm.timeout == 660.0


def test_explicit_none_timeout_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("WM_CLIENT_TIMEOUT_S", raising=False)
    assert WMClient(timeout=None).timeout == 600.0


def test_invalid_env_timeout_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("WM_CLIENT_TIMEOUT_S", "abc")
    assert WMClient().timeout == 600.0


def test_the_wm_message_signature_is_kept():
    """Callers pass these kwargs today; dropping one would TypeError across the fleet."""
    import inspect

    params = inspect.signature(WMClient.message).parameters
    for name in (
        "content_blocks",
        "context",
        "system_prompt",
        "output_schema",
        "model_override",
        "max_tokens",
        "work_tier",
        "product",
        "enable_reasoning",
        "disable_state_storage",
        "project_id",
    ):
        assert name in params
    assert "instruction" not in params


# --- Caller attribution ------------------------------------------------------


async def test_every_call_carries_default_app_id_and_its_worker_name():
    captured: list[httpx.Request] = []
    client = _client(captured)

    await client.message(worker_name="course-outline", message="hi")
    await client.message(worker_name="segment_select", message="hi")

    assert [r.headers.get("X-App-Id") for r in captured] == ["w2v", "w2v"]
    # The worker follows THAT call — not the first one the pooled client served.
    assert [r.headers.get("X-Worker-Name") for r in captured] == [
        "course-outline",
        "segment_select",
    ]


async def test_app_id_is_overridable_per_client():
    captured: list[httpx.Request] = []
    await _client(captured, app_id="other-app").message(worker_name="w", message="hi")
    assert captured[0].headers["X-App-Id"] == "other-app"


def test_empty_app_id_is_refused():
    with pytest.raises(ValueError, match="app_id"):
        WMClient(app_id="")


async def test_attribution_headers_survive_retries():
    captured: list[httpx.Request] = []
    statuses = iter([503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "busy"}) if next(statuses) == 503 else _ok()

    await _client(captured, handler).message(worker_name="promo_script", message="hi")

    assert len(captured) == 2
    for req in captured:
        assert req.headers["X-App-Id"] == "w2v"
        assert req.headers["X-Worker-Name"] == "promo_script"
        assert req.headers["X-Tenant-Id"] == f"Bearer {IR_TOKEN}"


# --- X-Project-Id: the W2V project, else tenant-level project ----------------
# Owner ruling 2026-10-07: w2v projects (orchestrator projects.id) and fr
# projects are separate concepts. A call carries the W2V project of its work;
# an fr project is never a source. Only a call with no W2V project falls back
# to the tenant-level project (tenant hex reversed), and that is counted.

_TENANT = "d5d22a23-5503-5148-9f60-a70cd95260d7"
# d5d22a2355035148 9f60a70cd95260d7 reversed -> 7d06259dc07a06f9 8415305532a22d5d
_TENANT_PROJECT = "7d06259d-c07a-06f9-8415-305532a22d5d"
_FR_PROJECT = "0195f48b-1111-7000-8000-00000000abcd"
_W2V_PROJECT = "0195f48b-2222-7000-8000-00000000beef"


def test_tenant_level_project_id_reverses_the_hex():
    import uuid as _uuid

    from w2v_common.auth import tenant_level_project_id

    assert tenant_level_project_id(_TENANT) == _TENANT_PROJECT
    assert tenant_level_project_id(_uuid.UUID(_TENANT)) == _TENANT_PROJECT
    assert tenant_level_project_id(_TENANT.upper()) == _TENANT_PROJECT
    assert tenant_level_project_id(_TENANT_PROJECT) == _TENANT  # involution


@pytest.mark.parametrize(
    "bad", [None, "", "  ", "not-a-uuid", "00000000-0000-0000-0000-000000000000"]
)
def test_tenant_level_project_id_refuses_invalid_and_nil(bad):
    from w2v_common.auth import tenant_level_project_id

    assert tenant_level_project_id(bad) is None


def _fallback_count(reason: str, worker: str) -> float:
    from prometheus_client import REGISTRY

    return (
        REGISTRY.get_sample_value(
            "w2v_wm_project_fallback_total", {"reason": reason, "worker": worker}
        )
        or 0.0
    )


async def test_explicit_w2v_project_is_sent():
    captured: list[httpx.Request] = []
    _bind(_TENANT)
    await _client(captured).message(worker_name="w", message="hi", project_id=_W2V_PROJECT)
    assert captured[0].headers.get_list("X-Project-Id") == [_W2V_PROJECT]


async def test_invalid_explicit_project_is_refused_before_any_call():
    captured: list[httpx.Request] = []
    with pytest.raises(ValueError, match="project_id"):
        await _client(captured).message(worker_name="w", message="hi", project_id="not-a-uuid")
    assert captured == []


async def test_context_bound_w2v_project_is_sent_and_explicit_wins():
    from w2v_common.auth import set_w2v_project_context

    captured: list[httpx.Request] = []
    _bind(_TENANT)
    set_w2v_project_context(_W2V_PROJECT)
    client = _client(captured)
    other = "0195f48b-3333-7000-8000-000000000001"

    await client.message(worker_name="w", message="hi")
    await client.message(worker_name="w", message="hi", project_id=other)

    assert captured[0].headers.get_list("X-Project-Id") == [_W2V_PROJECT]
    assert captured[1].headers.get_list("X-Project-Id") == [other]


async def test_fr_project_never_displaces_the_w2v_project_even_with_flag_on(monkeypatch):
    from w2v_common.auth import bind_attribution_from_headers, set_w2v_project_context

    monkeypatch.setenv("AUTH_ATTRIBUTION_FORWARD_HEADERS", "true")
    captured: list[httpx.Request] = []
    _bind(_TENANT)
    bind_attribution_from_headers({"X-Project-Id": _FR_PROJECT}, trusted=False)
    set_w2v_project_context(_W2V_PROJECT)

    await _client(captured).message(worker_name="w", message="hi")

    assert captured[0].headers.get_list("X-Project-Id") == [_W2V_PROJECT]


@pytest.mark.parametrize("flag", ["true", "false"])
async def test_fr_project_alone_is_not_a_source_tenant_level_is_sent_and_counted(monkeypatch, flag):
    from w2v_common.auth import bind_attribution_from_headers

    monkeypatch.setenv("AUTH_ATTRIBUTION_FORWARD_HEADERS", flag)
    captured: list[httpx.Request] = []
    _bind(_TENANT)
    bind_attribution_from_headers({"X-Project-Id": _FR_PROJECT}, trusted=False)
    worker = f"fallback-{flag}"
    before = _fallback_count("tenant_level", worker)

    await _client(captured).message(worker_name=worker, message="hi")

    assert captured[0].headers.get_list("X-Project-Id") == [_TENANT_PROJECT]
    assert _fallback_count("tenant_level", worker) == before + 1


async def test_no_tenant_sends_no_project_counts_and_warns_once(monkeypatch, caplog):
    from w2v_common.clients import wm_client as wm_module

    monkeypatch.setattr(wm_module, "_warned_no_project", False)
    captured: list[httpx.Request] = []
    client = _client(captured)
    before = _fallback_count("no_tenant", "lonely")

    with caplog.at_level(logging.WARNING, logger=wm_module.__name__):
        await client.message(worker_name="lonely", message="hi")
        await client.message(worker_name="lonely", message="hi")

    assert all("X-Project-Id" not in r.headers for r in captured)
    assert _fallback_count("no_tenant", "lonely") == before + 2
    warnings = [r for r in caplog.records if "X-Project-Id" in r.getMessage()]
    assert len(warnings) == 1


# --- W2V project propagation between w2v services ----------------------------


def test_inbound_w2v_project_header_is_bound_and_bad_values_ignored():
    from w2v_common.auth import bind_attribution_from_headers, get_w2v_project_id_from_context

    set_identity_context(_TENANT, "lic-1", "user-42")
    bind_attribution_from_headers({"X-W2V-Project-Id": "garbage"}, trusted=False)
    assert get_w2v_project_id_from_context() == ""
    bind_attribution_from_headers({"X-W2V-Project-Id": _W2V_PROJECT.upper()}, trusted=False)
    assert get_w2v_project_id_from_context() == _W2V_PROJECT


def test_w2v_project_survives_same_tenant_reseed_and_clears_on_tenant_change():
    from w2v_common.auth import get_w2v_project_id_from_context, set_w2v_project_context

    set_identity_context(_TENANT, "lic-1", "user-42")
    set_w2v_project_context(_W2V_PROJECT)
    set_identity_context(_TENANT, None, None)
    assert get_w2v_project_id_from_context() == _W2V_PROJECT
    set_identity_context("0195f48b-0000-7000-8000-000000000009", None, None)
    assert get_w2v_project_id_from_context() == ""


def test_set_w2v_project_context_refuses_nil_and_garbage():
    from w2v_common.auth import set_w2v_project_context

    for bad in ("00000000-0000-0000-0000-000000000000", "nope"):
        with pytest.raises(ValueError):
            set_w2v_project_context(bad)


async def test_identity_hook_stamps_w2v_project_on_downstream_hops():
    from w2v_common.auth import identity_event_hook, set_w2v_project_context

    set_identity_context(_TENANT, "lic-1", "user-42")
    set_w2v_project_context(_W2V_PROJECT)
    request = httpx.Request("POST", "http://analysis.test/api/analyze")

    await identity_event_hook(request)

    assert request.headers["X-W2V-Project-Id"] == _W2V_PROJECT
    # Never on the fr header: the two concepts stay separate on the wire.
    assert "X-Project-Id" not in request.headers
