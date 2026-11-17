"""w2v_common.providers: run provider-slot resolution + tenant adapters (shipped dark)."""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import httpx
import jwt
import pytest
import respx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jwt.algorithms import OKPAlgorithm

from w2v_common.auth.cell_jwt import CELL_TOKEN_HEADER, reset_cell_state
from w2v_common.auth.identity_context import (
    clear_identity_context,
    is_provider_hop,
    mark_provider_hop,
    set_correlation_context,
    set_identity_context,
    set_w2v_job_context,
)
from w2v_common.clients import vllm_direct
from w2v_common.clients.wm_client import WMClient
from w2v_common.discovery import service_url
from w2v_common.providers import (
    ASRRequest,
    ProviderHopRefusedError,
    ProviderSlot,
    ProviderSlotRequiredError,
    ProviderSlotUnavailableError,
    TTSRequest,
    asr_transcribe,
    clear_provider_slot_cache,
    provider_slots_mode,
    resolve_provider_slot,
    tenant_asr,
    tenant_ocr,
    tenant_tts,
    tts_synthesize,
)
from w2v_common.providers import _http as provider_http
from w2v_common.providers import slots as slots_mod
from w2v_common.providers.adapters import TenantSpeech, pcm16_to_wav
from w2v_common.providers.slots import RESOLVE_PATH, set_provider_slot_client
from w2v_common.tenant import TenantSource, UnvouchedTenantError

TENANT_A = uuid.UUID("0199a1b2-0000-7000-8000-00000000000a")
TENANT_B = uuid.UUID("0199a1b2-0000-7000-8000-00000000000b")
RUN = "0199a1b2-1111-7000-8000-000000000001"
RUN_2 = "0199a1b2-1111-7000-8000-000000000002"
CONN = "0199a1b2-2222-7000-8000-000000000001"
IR_TOKEN = "ir-licence-SECRET-abc123"  # pragma: allowlist secret
SA_TOKEN = "sa-w2v-licence"  # pragma: allowlist secret
TENANT_SECRET = "sk-tenant-SECRET-xyz789"  # pragma: allowlist secret
CALLER = "analysis-service"
ORCH_AUD = "w2v-orchestrator"
TENANT_BASE = "https://llm.tenant.example/v1"
#: The fake resolver answers every tenant host with this public address.
PUBLIC_IP = "93.184.216.34"
PUBLIC_URL = f"https://{PUBLIC_IP}"

RESOLVE_URL = f"{service_url('orchestrator')}{RESOLVE_PATH}"
IR_URL = f"{service_url('inference_router')}/v1/chat/completions"

#: Headers that must NEVER reach a tenant provider.
IDENTITY_HEADERS = (
    "x-tenant-id",
    "x-effective-tenant-id",
    "x-license-id",
    "x-subject-id",
    "x-actor-id",
    "x-project-id",
    "x-w2v-project-id",
    "x-w2v-job-id",
    "x-correlation-id",
    "x-stage",
    "x-request-id",
    "x-app-id",
    "x-worker-name",
    "x-job-id",
    CELL_TOKEN_HEADER.lower(),
)


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def caller_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture(autouse=True)
def _env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caller_key: Ed25519PrivateKey
) -> Iterator[None]:
    for name in (
        "LLM_BACKEND",
        "LLM_IR_BASE_URL",
        "LLM_IR_TOKEN_FILE",
        "LLM_IR_USAGE_REPORT_ENABLED",
        "W2V_LOCAL",
        "W2V_ORCHESTRATOR_URL",
        "W2V_INFERENCE_ROUTER_URL",
        "OTEL_SERVICE_NAME",
        slots_mod.SWITCH_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_IR_TOKEN", IR_TOKEN)
    monkeypatch.setenv("DOWNSTREAM_LICENSE_TOKEN", SA_TOKEN)
    pem = caller_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    jwk = json.loads(OKPAlgorithm.to_jwk(caller_key.public_key()))
    jwk.update({"kid": "caller-1", "iss": CALLER})
    jwks = tmp_path / "jwks.json"
    jwks.write_text(json.dumps({"keys": [jwk]}))
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", pem)
    monkeypatch.setenv("CELL_JWT_KID", "caller-1")
    monkeypatch.setenv("CELL_SERVICE_NAME", CALLER)
    monkeypatch.setenv("CELL_DEPLOYMENT_ID", "strato-w2v")
    monkeypatch.setenv("CELL_JWKS_PATH", str(jwks))

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(vllm_direct, "_sleep", _no_sleep)
    monkeypatch.setattr(slots_mod, "_sleep", _no_sleep)
    monkeypatch.setattr(provider_http, "_sleep", _no_sleep)

    async def _public(_host: str, _port: int) -> list[str]:
        return [PUBLIC_IP]

    monkeypatch.setattr(provider_http, "_resolve", _public)
    reset_cell_state()
    clear_identity_context()
    clear_provider_slot_cache()
    set_provider_slot_client(None)
    yield
    reset_cell_state()
    clear_identity_context()
    clear_provider_slot_cache()
    set_provider_slot_client(None)


@pytest.fixture
def enforce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(slots_mod.SWITCH_ENV, "enforce")


def _bind(tenant: uuid.UUID = TENANT_A, *, run: str | None = RUN, vouched: bool = True) -> None:
    set_identity_context(
        tenant,
        "lic-1",
        "subject-1",
        tenant_source=TenantSource.LICENSE_TOKEN if vouched else None,
    )
    if run is not None:
        set_w2v_job_context(run)
    set_correlation_context(RUN, "script")


def _tenant_payload(
    slot: str = "llm",
    *,
    kind: str = "openai-compatible",
    base_url: str | None = TENANT_BASE,
    model: str | None = "gpt-4.1-mini",
    secret: str | None = TENANT_SECRET,
    options: dict | None = None,
    pinned_ip: str | None = None,
) -> dict:
    body: dict = {
        "source": "tenant",
        "slot": slot,
        "kind": kind,
        "base_url": base_url,
        "model": model,
        "options": options or {},
        "secret": secret,
        "ref": {"connection_id": CONN, "version": 3},
    }
    if pinned_ip is not None:
        body["pinned_ip"] = pinned_ip
    return body


def _deployment_payload(slot: str = "llm", kind: str = "nonagentic-wm") -> dict:
    return {
        "slot": slot,
        "source": "deployment",
        "kind": kind,
        "base_url": None,
        "model": None,
        "options": {},
        "secret": None,
        "ref": None,
    }


def _completion(content: str = "hi") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


def _wav() -> bytes:
    return pcm16_to_wav(b"\x01\x00" * 2400, 24_000)


# ── switch ──────────────────────────────────────────────────────────────────


def test_switch_defaults_off_and_rejects_typos(monkeypatch: pytest.MonkeyPatch) -> None:
    assert provider_slots_mode() == "off"
    monkeypatch.setenv(slots_mod.SWITCH_ENV, " Enforce ")
    assert provider_slots_mode() == "enforce"
    monkeypatch.setenv(slots_mod.SWITCH_ENV, "enforced")
    with pytest.raises(ValueError, match="W2V_PROVIDER_SLOTS"):
        provider_slots_mode()


@respx.mock(assert_all_called=False)
async def test_off_switch_makes_no_resolve_call(respx_mock: respx.MockRouter) -> None:
    resolve = respx_mock.post(RESOLVE_URL).mock(return_value=httpx.Response(200, json={}))
    ir = respx_mock.post(IR_URL).mock(return_value=_completion())
    _bind()

    client = WMClient()
    try:
        result = await client.message(
            "promo_script",
            content_blocks=[
                {"type": "text", "text": "look"},
                {"type": "image", "data": "aGk=", "media_type": "image/png"},
            ],
        )
    finally:
        await client.aclose()

    async def platform() -> str:
        return "platform"

    def tenant(_speech: TenantSpeech) -> str:
        raise AssertionError("tenant path taken with the switch off")

    assert await tts_synthesize(TTSRequest(text="x"), platform=platform, from_tenant=tenant) == (
        "platform"
    )
    assert result.output == "hi"
    assert ir.call_count == 1
    assert resolve.call_count == 0


# ── resolve: identity, cache, errors ────────────────────────────────────────


@respx.mock
async def test_resolve_sends_only_a_cell_token_for_the_bound_run(
    respx_mock: respx.MockRouter, caller_key: Ed25519PrivateKey
) -> None:
    route = respx_mock.post(RESOLVE_URL).mock(
        return_value=httpx.Response(200, json=_tenant_payload())
    )
    _bind()

    slot = await resolve_provider_slot("llm")

    assert slot.is_tenant and slot.kind == "openai-compatible"
    assert slot.secret_value() == TENANT_SECRET
    assert TENANT_SECRET not in repr(slot)
    assert slot.ref is not None and slot.ref.version == 3
    req = route.calls[0].request
    assert json.loads(req.content) == {"slot": "llm", "run_id": RUN}
    assert "x-tenant-id" not in req.headers
    assert "authorization" not in req.headers
    assert "x-effective-tenant-id" not in req.headers
    claims = jwt.decode(
        req.headers[CELL_TOKEN_HEADER],
        caller_key.public_key(),
        algorithms=["EdDSA"],
        audience=ORCH_AUD,
    )
    assert claims["iss"] == CALLER
    assert claims["tenant"] == str(TENANT_A)
    assert claims["run"] == RUN


@respx.mock
async def test_cache_is_per_tenant_and_per_run(respx_mock: respx.MockRouter) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        tenant = jwt.decode(
            request.headers[CELL_TOKEN_HEADER], options={"verify_signature": False}
        )["tenant"]
        secret = (
            "sk-A-secret" if tenant == str(TENANT_A) else "sk-B-secret"
        )  # pragma: allowlist secret
        return httpx.Response(200, json=_tenant_payload(secret=secret))

    route = respx_mock.post(RESOLVE_URL).mock(side_effect=answer)

    _bind(TENANT_A)
    a1 = await resolve_provider_slot("llm")
    a2 = await resolve_provider_slot("llm")
    assert route.call_count == 1  # cached
    assert a1 is a2

    clear_identity_context()
    _bind(TENANT_B)
    b = await resolve_provider_slot("llm")
    assert route.call_count == 2  # another tenant never reads A's entry
    assert b.secret_value() == "sk-B-secret"  # pragma: allowlist secret
    assert a1.secret_value() == "sk-A-secret"  # pragma: allowlist secret

    clear_identity_context()
    _bind(TENANT_A, run=RUN_2)
    await resolve_provider_slot("llm")
    assert route.call_count == 3  # another run of the same tenant resolves its own pin


@respx.mock(assert_all_called=False)
async def test_unvouched_or_unbound_tenant_raises_and_sends_nothing(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(RESOLVE_URL).mock(return_value=httpx.Response(200, json={}))
    # No tenant bound (cron / service-originated) ⇒ the platform, no call.
    unbound = await resolve_provider_slot("llm")
    assert unbound.source == "deployment" and unbound.kind == "platform"
    _bind(vouched=False)
    with pytest.raises(UnvouchedTenantError):
        await resolve_provider_slot("llm")
    assert route.call_count == 0


@respx.mock(assert_all_called=False)
async def test_hop_header_on_the_inbound_request_refuses(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(RESOLVE_URL).mock(return_value=httpx.Response(200, json={}))
    _bind()
    mark_provider_hop(True)
    with pytest.raises(ProviderHopRefusedError):
        await resolve_provider_slot("tts")
    assert route.call_count == 0
    clear_identity_context()
    assert is_provider_hop() is False


async def test_auth_middleware_marks_the_hop_header(monkeypatch: pytest.MonkeyPatch) -> None:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from w2v_common.auth.cell_jwt import mint_cell_token
    from w2v_common.auth.middleware import AuthMiddleware

    # This process is also the callee: the cell lane authenticates our own token.
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    monkeypatch.setenv("CELL_ALLOWED_CALLERS", "*")
    reset_cell_state()
    seen: list[bool] = []

    async def endpoint(_request: Request) -> JSONResponse:
        seen.append(is_provider_hop())
        return JSONResponse({})

    app = Starlette(routes=[Route("/api/thing", endpoint)])
    app.add_middleware(AuthMiddleware)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://svc"
    ) as client:
        for hop in (True, False):
            headers = {CELL_TOKEN_HEADER: mint_cell_token(CALLER, tenant=TENANT_A)}
            if hop:
                headers["X-W2V-Provider-Hop"] = "1"
            response = await client.get("/api/thing", headers=headers)
            assert response.status_code == 200, response.text
    assert seen == [True, False]


@respx.mock
async def test_503_is_retried_then_succeeds(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(RESOLVE_URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.ConnectError("rolling"),
            httpx.Response(200, json=_deployment_payload()),
        ]
    )
    _bind()
    slot = await resolve_provider_slot("llm")
    assert slot.source == "deployment" and slot.secret is None
    assert route.call_count == 3


@respx.mock
async def test_503_exhaustion_is_transient_unavailable(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(RESOLVE_URL).mock(return_value=httpx.Response(503))
    _bind()
    with pytest.raises(ProviderSlotUnavailableError) as err:
        await resolve_provider_slot("llm")
    assert err.value.transient is True
    assert 4 <= route.call_count <= slots_mod.RESOLVE_MAX_ATTEMPTS


@pytest.mark.parametrize(
    ("status", "body", "exc", "attr"),
    [
        (
            403,
            {"detail": {"code": "byo_provider_required", "modality": "tts", "message": "bring it"}},
            ProviderSlotRequiredError,
            403,
        ),
        (
            409,
            {"detail": {"code": "provider_unavailable", "message": "purged"}},
            ProviderSlotUnavailableError,
            409,
        ),
        (404, {"detail": {"code": "run_not_found"}}, ProviderSlotUnavailableError, 404),
        (403, {"detail": "cell lane only"}, ProviderSlotUnavailableError, 403),
    ],
)
@respx.mock
async def test_error_statuses_are_typed(
    respx_mock: respx.MockRouter, status: int, body: dict, exc: type[Exception], attr: int
) -> None:
    route = respx_mock.post(RESOLVE_URL).mock(return_value=httpx.Response(status, json=body))
    _bind()
    with pytest.raises(exc) as err:
        await resolve_provider_slot("tts")
    assert getattr(err.value, "status", None) == attr
    assert getattr(err.value, "slot", None) == "tts"
    assert route.call_count == 1  # never retried
    with pytest.raises(exc):
        await resolve_provider_slot("tts")
    # "unavailable" is negative-cached for 5 s; "byo required" is not cached
    assert route.call_count == (2 if exc is ProviderSlotRequiredError else 1)


def test_payload_parsing_is_strict() -> None:
    tenant = ProviderSlot.from_payload(
        _tenant_payload(pinned_ip="203.0.113.7"), expected_slot="llm"
    )
    assert tenant.source == "tenant" and tenant.pinned_ip == "203.0.113.7"
    assert dict(tenant.options) == {}
    dep = ProviderSlot.from_payload(_deployment_payload(), expected_slot="llm")
    assert dep.source == "deployment" and dep.ref is None
    for bad in (
        [],
        {**_tenant_payload(), "slot": "vlm"},
        {**_tenant_payload(), "kind": ""},
        {**_tenant_payload(), "source": "tenant", "ref": None},
        {**_deployment_payload(), "secret": "leak"},  # pragma: allowlist secret
        {**_tenant_payload(), "ref": {"connection_id": "nope", "version": 1}},
        {**_tenant_payload(), "source": "mystery"},
    ):
        with pytest.raises(ProviderSlotUnavailableError):
            ProviderSlot.from_payload(bad, expected_slot="llm")


async def test_no_cell_signer_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CELL_JWT_SIGNING_KEY")
    monkeypatch.delenv("CELL_JWT_KID")
    reset_cell_state()
    _bind()
    with pytest.raises(ProviderSlotUnavailableError, match="cell signing"):
        await resolve_provider_slot("llm")


# ── WMClient: tenant vs deployment ──────────────────────────────────────────


@respx.mock
async def test_tenant_llm_sends_the_tenant_bearer_only(
    respx_mock: respx.MockRouter, enforce: None
) -> None:
    respx_mock.post(RESOLVE_URL).mock(return_value=httpx.Response(200, json=_tenant_payload("vlm")))
    tenant = respx_mock.post(f"{PUBLIC_URL}/v1/chat/completions").mock(
        return_value=_completion("ok")
    )
    ir = respx_mock.post(IR_URL).mock(return_value=_completion())
    _bind()

    client = WMClient()
    try:
        result = await client.message(
            "promo_script",
            content_blocks=[
                {"type": "text", "text": "describe"},
                {"type": "image", "data": "aGk=", "media_type": "image/png"},
            ],
            enable_reasoning=True,
            project_id="0199a1b2-3333-7000-8000-000000000001",
        )
    finally:
        await client.aclose()

    assert result.output == "ok"
    assert ir.call_count == 0
    req = tenant.calls[0].request
    assert req.headers["authorization"] == f"Bearer {TENANT_SECRET}"
    assert req.headers["x-w2v-provider-hop"] == "1"
    for name in IDENTITY_HEADERS:
        assert name not in req.headers, name
    assert IR_TOKEN not in str(req.headers) and SA_TOKEN not in str(req.headers)
    body = json.loads(req.content)
    assert body["model"] == "gpt-4.1-mini"
    assert "chat_template_kwargs" not in body and "enable_reasoning" not in body
    parts = body["messages"][-1]["content"]
    assert parts[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}}
    assert result.usage_metrics is not None
    assert result.usage_metrics["funding"] == "tenant_byo"
    assert result.metadata is not None and result.metadata["source"] == "tenant"


@respx.mock
async def test_vlm_slot_is_resolved_when_images_are_present(
    respx_mock: respx.MockRouter, enforce: None
) -> None:
    def echo(request: httpx.Request) -> httpx.Response:
        slot = json.loads(request.content)["slot"]
        return httpx.Response(200, json=_deployment_payload(slot, "platform"))

    route = respx_mock.post(RESOLVE_URL).mock(side_effect=echo)
    respx_mock.post(IR_URL).mock(return_value=_completion())
    _bind()
    client = WMClient()
    try:
        await client.message(
            "w", content_blocks=[{"type": "image", "data": "aGk=", "media_type": "image/png"}]
        )
        await client.message("w", "text only")
    finally:
        await client.aclose()
    sent = [json.loads(c.request.content)["slot"] for c in route.calls]
    assert sent == ["vlm", "llm"]


@respx.mock(assert_all_called=False)
async def test_deployment_slot_is_byte_identical_to_off(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolve = respx_mock.post(RESOLVE_URL).mock(
        return_value=httpx.Response(200, json=_deployment_payload())
    )
    ir = respx_mock.post(IR_URL).mock(return_value=_completion())

    async def one_call() -> httpx.Request:
        clear_identity_context()
        _bind()
        client = WMClient()
        try:
            await client.message("promo_script", "hello", max_tokens=9000)
        finally:
            await client.aclose()
        return ir.calls[-1].request

    off = await one_call()
    assert resolve.call_count == 0
    monkeypatch.setenv(slots_mod.SWITCH_ENV, "enforce")
    on = await one_call()
    assert resolve.call_count == 1  # enforce really resolved, and chose the platform

    assert on.method == off.method and on.url == off.url
    assert on.content == off.content
    assert dict(on.headers) == dict(off.headers)
    assert on.headers["x-tenant-id"] == f"Bearer {IR_TOKEN}"
    assert on.headers["x-effective-tenant-id"] == str(TENANT_A)


@respx.mock
async def test_byo_required_surfaces_from_wmclient(
    respx_mock: respx.MockRouter, enforce: None
) -> None:
    respx_mock.post(RESOLVE_URL).mock(
        return_value=httpx.Response(403, json={"detail": {"code": "byo_provider_required"}})
    )
    _bind()
    client = WMClient()
    try:
        with pytest.raises(ProviderSlotRequiredError) as err:
            await client.message("w", "hi")
    finally:
        await client.aclose()
    assert err.value.failure_detail["code"] == "byo_provider_required"


# ── transport: pinned ip + origin lock ──────────────────────────────────────


@respx.mock
async def test_pinned_ip_keeps_host_and_sni(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post("https://203.0.113.7/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=_wav(), headers={"content-type": "audio/wav"})
    )
    slot = ProviderSlot.from_payload(
        _tenant_payload(
            "tts",
            kind="openai-compatible",
            pinned_ip="203.0.113.7",
            base_url="https://tts.tenant.example/v1",
        ),
        expected_slot="tts",
    )
    speech = await tenant_tts(slot, TTSRequest(text="hello"))
    req = route.calls[0].request
    assert req.headers["host"] == "tts.tenant.example"
    assert req.extensions["sni_hostname"] == "tts.tenant.example"
    assert speech.sample_rate == 24_000 and speech.duration_seconds == pytest.approx(0.1)


async def test_origin_lock_refuses_another_host() -> None:
    inner = httpx.MockTransport(lambda _r: httpx.Response(200))
    transport = provider_http.PinnedTransport("llm", "https://a.example/v1", None, inner)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.ConnectError, match="origin"):
            await client.get("https://b.example/v1/x")
        assert (await client.get("https://a.example/v1/x")).status_code == 200


# ── tts / asr / ocr adapters ────────────────────────────────────────────────


@respx.mock
async def test_elevenlabs_uses_its_key_header_not_bearer(respx_mock: respx.MockRouter) -> None:
    pcm = b"\x02\x00" * 240
    route = respx_mock.post(
        url__regex=r"https://93\.184\.216\.34/v1/text-to-speech/v-1/with-timestamps.*"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "audio_base64": base64.b64encode(pcm).decode(),
                "alignment": {
                    "characters": list("hi yo"),
                    "character_start_times_seconds": [0.0, 0.1, 0.2, 0.3, 0.4],
                    "character_end_times_seconds": [0.1, 0.2, 0.3, 0.4, 0.5],
                },
            },
        )
    )
    slot = ProviderSlot.from_payload(
        _tenant_payload(
            "tts", kind="elevenlabs", base_url=None, model=None, options={"voice_id": "v-1"}
        ),
        expected_slot="tts",
    )
    speech = await tenant_tts(slot, TTSRequest(text="<speak>hi yo</speak>", ssml=True))
    req = route.calls[0].request
    assert req.headers["xi-api-key"] == TENANT_SECRET
    assert "authorization" not in req.headers
    assert json.loads(req.content)["text"] == "hi yo"
    assert speech.word_timings == [
        {"word": "hi", "start": 0.0, "end": 0.2},
        {"word": "yo", "start": 0.3, "end": 0.5},
    ]


@respx.mock
async def test_nonagentic_tts_and_helper(respx_mock: respx.MockRouter, enforce: None) -> None:
    respx_mock.post(RESOLVE_URL).mock(
        return_value=httpx.Response(
            200,
            json=_tenant_payload(
                "tts",
                kind="nonagentic_endpoint",
                base_url="https://gpu.tenant.example",
                secret=None,
            ),
        )
    )
    tts = respx_mock.post("https://93.184.216.34/api/tts/synthesize").mock(
        return_value=httpx.Response(
            200,
            json={
                "audio_base64": base64.b64encode(_wav()).decode(),
                "sample_rate": 24000,
                "duration_sec": 0.1,
                "provider": "kokoro",
                "voice": "af",
                "word_timings": [{"word": "x", "start": 0.0, "end": 0.1}],
                "speech_seconds": 0.08,
                "pause_seconds": 0.02,
            },
            headers={"X-TTS-LUFS": "-16.0"},
        )
    )
    _bind()

    async def platform() -> str:
        raise AssertionError("platform path taken for a tenant slot")

    speech = await tts_synthesize(
        TTSRequest(text="x", voice="af", speed=1.1),
        platform=platform,
        from_tenant=lambda s: s,
    )
    assert isinstance(speech, TenantSpeech)
    assert speech.integrated_lufs == -16.0 and speech.provider == "kokoro"
    req = tts.calls[0].request
    assert "authorization" not in req.headers  # keyless connection
    assert json.loads(req.content) == {
        "text": "x",
        "speed": 1.1,
        "output_format": "wav",
        "voice": "af",
    }
    assert speech.audio_format == "wav"
    for name in IDENTITY_HEADERS:
        assert name not in req.headers, name


@respx.mock
async def test_asr_helper_deployment_takes_platform(
    respx_mock: respx.MockRouter, enforce: None
) -> None:
    respx_mock.post(RESOLVE_URL).mock(
        return_value=httpx.Response(200, json=_deployment_payload("asr", "platform"))
    )
    _bind()

    async def platform() -> str:
        return "platform"

    out = await asr_transcribe(
        ASRRequest(audio=b"x"), platform=platform, from_tenant=lambda _d: "tenant"
    )
    assert out == "platform"


@respx.mock
async def test_asr_kinds(respx_mock: respx.MockRouter) -> None:
    na = respx_mock.post("https://93.184.216.34/api/asr/transcribe").mock(
        return_value=httpx.Response(
            200, json={"text": "hi", "segments": [], "backend": "whisper", "model": "m"}
        )
    )
    oa = respx_mock.post("https://93.184.216.34/v1/audio/transcriptions").mock(
        return_value=httpx.Response(
            200,
            json={
                "text": "hello world",
                "language": "en",
                "duration": 1.0,
                "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hello world"}],
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.4},
                    {"word": "world", "start": 0.5, "end": 0.9},
                ],
            },
        )
    )
    native = ProviderSlot.from_payload(
        _tenant_payload("asr", kind="nonagentic_endpoint", base_url="https://asr.tenant.example"),
        expected_slot="asr",
    )
    out = await tenant_asr(native, ASRRequest(audio=b"RIFF", language="en"))
    assert out["text"] == "hi"
    req = na.calls[0].request
    assert req.headers["authorization"] == f"Bearer {TENANT_SECRET}"
    assert b'name="word_timestamps"' in req.content and b'name="file"' in req.content

    compat = ProviderSlot.from_payload(
        _tenant_payload(
            "asr", kind="openai-compatible", base_url="https://asr.tenant.example/v1", model=None
        ),
        expected_slot="asr",
    )
    mapped = await tenant_asr(compat, ASRRequest(audio=b"RIFF"))
    segments = cast("list[dict[str, list[dict[str, object]]]]", mapped["segments"])
    assert [w["word"] for w in segments[0]["words"]] == ["hello", "world"]
    assert b"whisper-1" in oa.calls[0].request.content


@respx.mock
async def test_ocr_and_tenant_errors(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://93.184.216.34/v1/ocr").mock(
        side_effect=[
            httpx.Response(
                200, json={"pages": [{"index": 1, "markdown": "b"}, {"index": 0, "markdown": "a"}]}
            ),
            httpx.Response(401, json={"error": {"message": f"bad key {TENANT_SECRET}"}}),
            httpx.Response(400, text=f"nope Bearer {TENANT_SECRET}"),
        ]
    )
    slot = ProviderSlot.from_payload(
        _tenant_payload(
            "ocr", kind="nonagentic_endpoint", base_url="https://ocr.tenant.example", model=None
        ),
        expected_slot="ocr",
    )
    pages = await tenant_ocr(slot, b"%PDF")
    assert [p.markdown for p in pages] == ["a", "b"]
    with pytest.raises(ProviderSlotUnavailableError) as auth:
        await tenant_ocr(slot, b"%PDF")
    assert auth.value.status == 401 and TENANT_SECRET not in str(auth.value)
    with pytest.raises(ProviderSlotUnavailableError) as bad:
        await tenant_ocr(slot, b"%PDF")
    assert bad.value.status == 400 and TENANT_SECRET not in str(bad.value)
    assert "nope" not in str(bad.value)  # the response body is never echoed
    assert bad.value.transient is False


async def test_adapters_refuse_a_deployment_slot() -> None:
    dep = ProviderSlot.from_payload(_deployment_payload("tts", "nonagentic"), expected_slot="tts")
    with pytest.raises(ValueError, match="tenant"):
        await tenant_tts(dep, TTSRequest(text="x"))


# ── review fixes: cache/run/single-flight/404 ───────────────────────────────


@respx.mock
async def test_run_id_falls_back_to_correlation_and_runless_is_short_lived(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(RESOLVE_URL).mock(
        return_value=httpx.Response(200, json=_deployment_payload())
    )
    set_identity_context(TENANT_A, "lic-1", "subject-1", tenant_source=TenantSource.LICENSE_TOKEN)
    set_correlation_context(RUN_2, "script")  # no X-W2V-Job-Id: WMClient's rule → correlation
    await resolve_provider_slot("llm")
    assert json.loads(route.calls[0].request.content)["run_id"] == RUN_2

    clear_identity_context()
    set_identity_context(TENANT_A, "lic-1", "subject-1", tenant_source=TenantSource.LICENSE_TOKEN)
    await resolve_provider_slot("llm")
    assert json.loads(route.calls[1].request.content)["run_id"] is None
    key = (TENANT_A, None, "llm")
    assert key not in slots_mod._cache  # never the 300 s cache
    assert key in slots_mod._runless_cache
    assert slots_mod._runless_cache.ttl == slots_mod.RUNLESS_CACHE_TTL_SECONDS == 30.0


@respx.mock
async def test_concurrent_resolves_share_one_call(respx_mock: respx.MockRouter) -> None:
    gate = asyncio.Event()

    async def slow(_request: httpx.Request) -> httpx.Response:
        await gate.wait()
        return httpx.Response(200, json=_tenant_payload())

    route = respx_mock.post(RESOLVE_URL).mock(side_effect=slow)
    _bind()
    tasks = [asyncio.create_task(resolve_provider_slot("llm")) for _ in range(5)]
    await asyncio.sleep(0.01)
    gate.set()
    results = await asyncio.gather(*tasks)
    assert route.call_count == 1
    assert all(r is results[0] for r in results)


@respx.mock
async def test_missing_route_404_is_transient_and_502_is_retried(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(RESOLVE_URL).mock(
        side_effect=[
            httpx.Response(502),
            httpx.Response(504),
            httpx.Response(404, json={"detail": "Not Found"}),
        ]
    )
    _bind()
    with pytest.raises(ProviderSlotUnavailableError) as err:
        await resolve_provider_slot("llm")
    assert route.call_count == 3
    assert err.value.status == 404 and err.value.transient is True
    assert "run was not found" not in str(err.value)


@respx.mock(assert_all_called=False)
async def test_enforce_without_a_tenant_takes_the_platform_path(
    respx_mock: respx.MockRouter, enforce: None
) -> None:
    resolve = respx_mock.post(RESOLVE_URL).mock(return_value=httpx.Response(500))
    ir = respx_mock.post(IR_URL).mock(return_value=_completion())
    client = WMClient()
    try:
        result = await client.message("promo_script", "hello")
    finally:
        await client.aclose()
    assert result.output == "hi"
    assert resolve.call_count == 0 and ir.call_count == 1


# ── review fixes: paid-call retries ─────────────────────────────────────────


def _tts_slot() -> ProviderSlot:
    return ProviderSlot.from_payload(
        _tenant_payload("tts", kind="openai-compatible", base_url="https://tts.tenant.example/v1"),
        expected_slot="tts",
    )


@pytest.mark.parametrize("status", [500, 502, 503, 504])
@respx.mock
async def test_tenant_5xx_is_never_retried(respx_mock: respx.MockRouter, status: int) -> None:
    route = respx_mock.post(f"{PUBLIC_URL}/v1/audio/speech").mock(
        return_value=httpx.Response(status, text="SECRET-CONTENT")
    )
    with pytest.raises(ProviderSlotUnavailableError) as err:
        await tenant_tts(_tts_slot(), TTSRequest(text="x"))
    assert route.call_count == 1
    assert err.value.transient is True and "SECRET-CONTENT" not in str(err.value)


@respx.mock
async def test_tenant_read_timeout_is_never_retried(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{PUBLIC_URL}/v1/audio/speech").mock(
        side_effect=httpx.ReadTimeout("slow")
    )
    with pytest.raises(ProviderSlotUnavailableError):
        await tenant_tts(_tts_slot(), TTSRequest(text="x"))
    assert route.call_count == 1


@respx.mock
async def test_tenant_connect_error_and_429_are_retried(respx_mock: respx.MockRouter) -> None:
    slept: list[float] = []

    async def record(seconds: float) -> None:
        slept.append(seconds)

    provider_http._sleep = record  # restored by monkeypatch in the autouse fixture
    route = respx_mock.post(f"{PUBLIC_URL}/v1/audio/speech").mock(
        side_effect=[
            httpx.ConnectError("refused"),
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, content=_wav(), headers={"content-type": "audio/wav"}),
        ]
    )
    speech = await tenant_tts(_tts_slot(), TTSRequest(text="x"))
    assert speech.sample_rate == 24_000
    assert route.call_count == 3
    assert slept[1] == 7.0  # Retry-After honoured


@respx.mock
async def test_tenant_llm_5xx_and_timeout_send_once_and_never_echo(
    respx_mock: respx.MockRouter, enforce: None, caplog: pytest.LogCaptureFixture
) -> None:
    respx_mock.post(RESOLVE_URL).mock(return_value=httpx.Response(200, json=_tenant_payload()))
    route = respx_mock.post(f"{PUBLIC_URL}/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                503, json={"error": {"message": "SECRET-CONTENT", "code": "overloaded"}}
            ),
            httpx.ReadTimeout("slow"),
            *[_completion("SECRET-REPLY-not-json") for _ in range(vllm_direct.OUTPUT_RETRIES + 1)],
        ]
    )
    _bind()
    client = WMClient()
    try:
        with pytest.raises(ProviderSlotUnavailableError) as first:
            await client.message("w", "hi")
        assert route.call_count == 1
        assert "overloaded" in str(first.value) and "SECRET-CONTENT" not in str(first.value)
        with pytest.raises(ProviderSlotUnavailableError):
            await client.message("w", "hi")
        assert route.call_count == 2
        schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
        with pytest.raises(httpx.HTTPStatusError) as reask:
            await client.message("w", "hi", output_schema=schema)
    finally:
        await client.aclose()
    assert "SECRET-REPLY" not in str(reask.value)
    assert "SECRET" not in caplog.text


# ── review fixes: SSRF / rebinding ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("address", "blocked"),
    [
        ("10.1.2.3", True),
        ("172.16.0.1", True),
        ("192.168.1.119", True),
        ("100.64.0.9", True),  # tailnet CGNAT
        ("127.0.0.1", True),
        ("169.254.169.254", True),
        ("::1", True),
        ("fd00::1", True),
        ("fe80::1", True),
        ("::ffff:10.0.0.1", True),
        ("0.0.0.0", True),
        ("not-an-ip", True),
        ("93.184.216.34", False),
        ("2606:4700::1111", False),
    ],
)
def test_blocked_address_table(address: str, blocked: bool) -> None:
    assert provider_http.is_blocked_address(address) is blocked


@pytest.mark.parametrize("answer", [["10.43.0.10"], ["93.184.216.34", "100.64.0.1"], []])
@respx.mock(assert_all_called=False)
async def test_unpinned_host_resolving_private_is_refused(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, answer: list[str]
) -> None:
    async def resolver(_host: str, _port: int) -> list[str]:
        return answer

    monkeypatch.setattr(provider_http, "_resolve", resolver)
    route = respx_mock.route().mock(return_value=httpx.Response(200))
    with pytest.raises(ProviderSlotUnavailableError):
        await tenant_tts(_tts_slot(), TTSRequest(text="x"))
    assert route.call_count == 0


@respx.mock
async def test_openai_llm_kind_defaults_to_the_public_api(
    respx_mock: respx.MockRouter, enforce: None
) -> None:
    respx_mock.post(RESOLVE_URL).mock(
        return_value=httpx.Response(200, json=_tenant_payload(kind="openai", base_url=None))
    )
    route = respx_mock.post(f"{PUBLIC_URL}/v1/chat/completions").mock(return_value=_completion())
    _bind()
    client = WMClient()
    try:
        await client.message("w", "hi")
    finally:
        await client.aclose()
    req = route.calls[0].request
    assert req.headers["host"] == "api.openai.com"
    assert req.extensions["sni_hostname"] == "api.openai.com"


# ── orchestrator #1150 contract ─────────────────────────────────────────────

_CONTRACT = json.loads(
    (Path(__file__).parent / "fixtures" / "provider_slot_contract.json").read_text()
)


def test_contract_fixture_keys_match_the_parser() -> None:
    assert set(_CONTRACT["tenant"]) == set(slots_mod.WIRE_KEYS)
    assert set(_CONTRACT["deployment"]) == set(slots_mod.WIRE_KEYS)
    tenant = ProviderSlot.from_payload(_CONTRACT["tenant"], expected_slot="llm")
    assert tenant.source == "tenant" and tenant.ref is not None and tenant.ref.version == 3
    # (the fixture's secret is null so the secret scanner stays quiet; a keyed one parses too)
    keyed = ProviderSlot.from_payload(
        {**_CONTRACT["tenant"], "secret": TENANT_SECRET}, expected_slot="llm"
    )
    assert keyed.secret_value() == TENANT_SECRET
    dep = ProviderSlot.from_payload(_CONTRACT["deployment"], expected_slot="tts")
    assert dep.source == "deployment" and dep.secret is None and dep.ref is None
    # source is preferred; the ref fallback still works without it
    legacy = {k: v for k, v in _CONTRACT["tenant"].items() if k != "source"}
    assert ProviderSlot.from_payload(legacy, expected_slot="llm").source == "tenant"
    for key in slots_mod.WIRE_KEYS - {"source"}:
        broken = {k: v for k, v in _CONTRACT["tenant"].items() if k != key}
        with pytest.raises(ProviderSlotUnavailableError, match="missing"):
            ProviderSlot.from_payload(broken, expected_slot="llm")


@pytest.mark.parametrize("case", _CONTRACT["errors"], ids=lambda c: c["body"]["detail"]["code"])
@respx.mock
async def test_contract_fixture_errors_are_typed(respx_mock: respx.MockRouter, case: dict) -> None:
    from w2v_common.providers import ProviderSlotConfigError

    route = respx_mock.post(RESOLVE_URL).mock(
        return_value=httpx.Response(case["status"], json=case["body"])
    )
    _bind()
    expected = {
        "config": ProviderSlotConfigError,
        "required": ProviderSlotRequiredError,
        "run_not_found": ProviderSlotUnavailableError,
        "unavailable": ProviderSlotUnavailableError,
    }[case["expect"]]
    with pytest.raises(expected) as err:
        await resolve_provider_slot("tts")
    assert route.call_count == 1  # none of them is retried
    if case["expect"] == "config":
        assert getattr(err.value, "code", None) == case["body"]["detail"]["code"]
        assert not isinstance(err.value, ProviderSlotUnavailableError)
    if case["expect"] == "run_not_found":
        assert err.value.transient is False and "run was not found" in str(err.value)


@respx.mock
async def test_ocr_kinds_route_and_model_rules(respx_mock: respx.MockRouter) -> None:
    ok = {"pages": [{"index": 0, "markdown": "a"}]}
    native = respx_mock.post(f"{PUBLIC_URL}/v1/ocr").mock(return_value=httpx.Response(200, json=ok))
    mistral = respx_mock.post(f"{PUBLIC_URL}/ocr").mock(return_value=httpx.Response(200, json=ok))
    na = ProviderSlot.from_payload(
        _tenant_payload(
            "ocr", kind="nonagentic_endpoint", base_url="https://ocr.t.example", model=None
        ),
        expected_slot="ocr",
    )
    await tenant_ocr(na, b"%PDF")
    assert "model" not in json.loads(native.calls[0].request.content)
    await tenant_ocr(na, b"%PDF", default_model="paddleocr-vl")
    assert json.loads(native.calls[1].request.content)["model"] == "paddleocr-vl"

    oc = ProviderSlot.from_payload(
        _tenant_payload(
            "ocr",
            kind="openai-compatible",
            base_url="https://api.mistral.example",
            model="mistral-ocr-latest",
        ),
        expected_slot="ocr",
    )
    await tenant_ocr(oc, b"%PDF")
    assert json.loads(mistral.calls[0].request.content)["model"] == "mistral-ocr-latest"

    no_model = ProviderSlot.from_payload(
        _tenant_payload(
            "ocr", kind="openai-compatible", base_url="https://api.mistral.example", model=None
        ),
        expected_slot="ocr",
    )
    with pytest.raises(ProviderSlotUnavailableError, match="model"):
        await tenant_ocr(no_model, b"%PDF")
