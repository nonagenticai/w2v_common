"""Tenant-kind HTTP adapters: llm/vlm, tts, asr, ocr on a TENANT provider slot.

Used only when :func:`~w2v_common.providers.slots.resolve_provider_slot` returned
``source="tenant"``; the deployment (platform) path is the caller's existing client,
unchanged. Every request goes through :func:`~w2v_common.providers._http.tenant_http_client`
(bare httpx, origin lock, pinned IP, ``X-W2V-Provider-Hop: 1``, the tenant's secret as
the only credential — never an identity header).

Provenance of the logic (orchestrator ``origin/main``):

* llm/vlm — reuses THIS library's :class:`~w2v_common.clients.vllm_direct.VLLMDirectClient`
  (seed prompts, xgrammar schema, validation + re-ask, retries) with a tenant credential
  and no reasoning field (OpenAI rejects ``chat_template_kwargs``). Kinds:
  ``openai-compatible`` (``{base_url}/chat/completions``, ``image_url`` parts for vlm) and
  ``anthropic`` through Anthropic's OpenAI-compatible endpoint (``{base}/v1``). The
  orchestrator's own in-process LLM is pydantic-ai based and was NOT lifted.
* tts — ported from ``src/services/providers/tts_inprocess.py`` (``httpx2`` → ``httpx``,
  pydantic schemas → :class:`TenantSpeech`): ``openai``, ``openai-compatible``
  (``POST {base}/audio/speech`` WAV), ``elevenlabs`` (``/v1/text-to-speech/{voice}/
  with-timestamps``, PCM 24 kHz → WAV + word timings; ElevenLabs' API takes the key as
  ``xi-api-key``, not Bearer) and ``nonagentic_endpoint`` (``POST {base}/api/tts/synthesize``).
  ``TenantSpeech.audio_format`` is magic-byte sniffed (see ``helpers`` for the per-kind
  table); WAV is requested everywhere because it is the platform format.
* asr — ported from ``src/services/providers/asr_inprocess.py``: ``nonagentic_endpoint``
  (``POST {base}/api/asr/transcribe`` multipart) and ``openai-compatible``
  (``POST {base}/audio/transcriptions``, verbose_json). The orchestrator's tenant
  adapter posts ``/audio/transcriptions`` under a base URL that already ends ``/v1``;
  that is the same wire as ``/v1/audio/transcriptions``.
* ocr — the provider_contracts ``POST {base}/v1/ocr`` (Mistral OCR shape) the
  orchestrator's ``src/clients/ocr_client.py`` speaks. There is no ``ocr`` slot in the
  orchestrator registry yet; the adapter exists so a consumer can adopt it when one is.

A tenant failure raises :class:`~w2v_common.providers.errors.ProviderSlotUnavailableError`
(``transient`` set for 429/5xx/unreachable) and NEVER falls back to the platform.
"""

from __future__ import annotations

import base64
import binascii
import html
import io
import logging
import re
import struct
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, cast
from urllib.parse import quote

import httpx

from w2v_common.clients.vllm_direct import JsonObject, VLLMDirectClient
from w2v_common.providers._http import tenant_headers, tenant_http_client
from w2v_common.providers.audio import (
    AudioFormat,
    audio_file_extension,
    audio_mime_type,
    detect_audio_format,
    sniff_audio_format,
)
from w2v_common.providers.errors import ProviderSlotUnavailableError
from w2v_common.providers.slots import ProviderSlot

logger = logging.getLogger(__name__)

KIND_OPENAI: Final = "openai"
KIND_OPENAI_COMPATIBLE: Final = "openai-compatible"
KIND_ANTHROPIC: Final = "anthropic"
KIND_ELEVENLABS: Final = "elevenlabs"
KIND_NONAGENTIC_ENDPOINT: Final = "nonagentic_endpoint"

OPENAI_DEFAULT_BASE_URL: Final = "https://api.openai.com/v1"
#: Anthropic's OpenAI SDK compatibility endpoint (Chat Completions shape).
ANTHROPIC_OPENAI_COMPAT_BASE_URL: Final = "https://api.anthropic.com/v1"
ELEVENLABS_DEFAULT_BASE_URL: Final = "https://api.elevenlabs.io"

NONAGENTIC_TTS_PATH: Final = "/api/tts/synthesize"
NONAGENTIC_ASR_PATH: Final = "/api/asr/transcribe"
OPENAI_ASR_PATH: Final = "/audio/transcriptions"
OCR_PATH: Final = "/v1/ocr"
OCR_COMPAT_PATH: Final = "/ocr"

OPENAI_TTS_DEFAULT_MODEL: Final = "gpt-4o-mini-tts"
OPENAI_TTS_DEFAULT_VOICE: Final = "alloy"
ELEVENLABS_DEFAULT_MODEL: Final = "eleven_multilingual_v2"
ELEVENLABS_PCM_RATE: Final = 24_000
ASR_DEFAULT_MODEL: Final = "whisper-1"
_ELEVENLABS_SPEED_RANGE: Final = (0.7, 1.2)
_OPENAI_SPEED_RANGE: Final = (0.25, 4.0)
_LUFS_HEADER: Final = "X-TTS-LUFS"
_SAMPLE_RATE_HEADER: Final = "X-TTS-Sample-Rate"
_DURATION_HEADER: Final = "X-TTS-Duration-Sec"
_AUTH_HTTP: Final = frozenset({401, 403})
_SSML_TAG: Final = re.compile(r"<[^>]+>")

DEFAULT_TIMEOUT_S: Final = 300.0


# ── shared error mapping ────────────────────────────────────────────────────


def _where(slot: ProviderSlot) -> str:
    return f"tenant {slot.slot} provider ({slot.kind}" + (
        f", model {slot.model})" if slot.model else ")"
    )


def _require_base_url(slot: ProviderSlot) -> str:
    if not slot.base_url:
        raise ProviderSlotUnavailableError(slot.slot, f"kind {slot.kind!r} requires a base URL")
    return slot.base_url.rstrip("/")


_CODE_SAFE: Final = re.compile(r"[^A-Za-z0-9_.:-]")


def provider_error_code(response: httpx.Response) -> str | None:
    """The provider's machine error code/type (``error.code`` / ``error.type`` /
    ``detail.code`` / ``code``), bounded + sanitised — NEVER the message or body."""
    try:
        body = response.json()
    except ValueError:
        return None
    candidates: list[object] = []
    if isinstance(body, Mapping):
        top = cast("Mapping[str, object]", body)
        candidates += [top.get("error"), top.get("detail"), top]
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        mapping = cast("Mapping[str, object]", item)
        for key in ("code", "type"):
            value = mapping.get(key)
            if isinstance(value, str | int) and not isinstance(value, bool):
                text = _CODE_SAFE.sub("", str(value))[:64]
                if text:
                    return text
    return None


def _status_error(slot: ProviderSlot, response: httpx.Response) -> ProviderSlotUnavailableError:
    """Status + provider error code only: the tenant's response content is never echoed."""
    status = response.status_code
    where = _where(slot)
    code = provider_error_code(response)
    suffix = f", code {code}" if code else ""
    if status in _AUTH_HTTP:
        return ProviderSlotUnavailableError(
            slot.slot,
            f"{where} rejected the credentials (HTTP {status}{suffix}); "
            "update the API key on the Providers page",
            status=status,
        )
    return ProviderSlotUnavailableError(
        slot.slot,
        f"{where} failed (HTTP {status}{suffix})",
        status=status,
        transient=status == 429 or status >= 500,
    )


def _malformed(slot: ProviderSlot, what: str) -> ProviderSlotUnavailableError:
    return ProviderSlotUnavailableError(
        slot.slot, f"{_where(slot)} returned an unusable response: {what}"
    )


async def _send(
    slot: ProviderSlot,
    base_url: str,
    url: str,
    *,
    timeout_s: float,
    headers: Mapping[str, str],
    json_body: Mapping[str, object] | None = None,
    data: Mapping[str, str | list[str]] | None = None,
    files: Mapping[str, tuple[str, bytes, str]] | None = None,
) -> httpx.Response:
    """POST to the tenant; non-2xx (redirects included — never followed) raises."""
    pinned = slot.pinned_ip if slot.base_url else None
    try:
        async with tenant_http_client(
            slot.slot, base_url, pinned, timeout_s, headers=headers
        ) as http:
            if files is not None:
                response = await http.post(url, data=dict(data or {}), files=dict(files))
            else:
                response = await http.post(url, json=dict(json_body or {}))
            await response.aread()
    except (httpx.TransportError, TimeoutError) as exc:
        raise ProviderSlotUnavailableError(
            slot.slot,
            f"{_where(slot)} unreachable ({type(exc).__name__})",
            transient=True,
        ) from None
    if response.status_code >= 300:
        err = _status_error(slot, response)
        logger.warning("tenant provider call failed slot=%s kind=%s: %s", slot.slot, slot.kind, err)
        raise err
    return response


def _json(slot: ProviderSlot, response: httpx.Response) -> Mapping[str, object]:
    try:
        payload = response.json()
    except ValueError:
        raise _malformed(slot, "body is not JSON") from None
    if not isinstance(payload, Mapping):
        raise _malformed(slot, "body is not a JSON object")
    return cast("Mapping[str, object]", payload)


def _option(slot: ProviderSlot, *names: str) -> str | None:
    for name in names:
        value = slot.options.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _require_tenant(slot: ProviderSlot) -> None:
    if not slot.is_tenant:
        raise ValueError("tenant adapters need a tenant provider slot")


# ── llm / vlm ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _TenantLLMSettings:
    """``DirectSettings`` for a tenant connection (one model for every worker)."""

    base_url: str
    token: str = field(repr=False)
    model: str

    def model_for(self, worker_name: str) -> str:
        del worker_name
        return self.model


#: llm/vlm kinds this client calls. The orchestrator registry (origin/main) allows
#: llm: openai-compatible, anthropic; vlm: openai-compatible. ``openai`` is accepted
#: too (public API default) so the registry can add it without a library release.
LLM_KINDS: Final = frozenset({KIND_OPENAI_COMPATIBLE, KIND_ANTHROPIC, KIND_OPENAI})


def _llm_base_url(slot: ProviderSlot) -> str:
    if slot.kind == KIND_OPENAI_COMPATIBLE:
        return _require_base_url(slot)
    if slot.kind == KIND_OPENAI:
        return (slot.base_url or OPENAI_DEFAULT_BASE_URL).rstrip("/")
    if slot.kind == KIND_ANTHROPIC:
        if not slot.base_url:
            return ANTHROPIC_OPENAI_COMPAT_BASE_URL
        base = slot.base_url.rstrip("/")
        return base if base.endswith("/v1") else f"{base}/v1"
    raise ProviderSlotUnavailableError(
        slot.slot, f"kind {slot.kind!r} cannot be called from this service"
    )


class TenantLLMClient(VLLMDirectClient):
    """:class:`VLLMDirectClient` against a tenant's OpenAI-compatible endpoint.

    Same call/validate/re-ask loop and error types as the platform path; the
    credential is the tenant's secret (Bearer, omitted when keyless), the body
    carries no reasoning field, ``usage_metrics`` say ``funding=tenant_byo``.
    """

    service_name = "tenant_llm"
    auth_hint = "the tenant's provider rejected its API key (Providers page)"
    echo_reply_problems = False

    def __init__(self, slot: ProviderSlot, *, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        _require_tenant(slot)
        if not slot.model:
            raise ProviderSlotUnavailableError(slot.slot, "the connection has no model configured")
        self.slot = slot
        self._base_url = _llm_base_url(slot)
        super().__init__(
            _TenantLLMSettings(
                base_url=self._base_url, token=slot.secret_value() or "", model=slot.model
            ),
            timeout=timeout,
        )

    def _build_client(self, transport: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
        del transport  # the tenant client owns its pinned/retry stack
        return tenant_http_client(
            self.slot.slot,
            self._base_url,
            self.slot.pinned_ip if self.slot.base_url else None,
            self.timeout,
            headers=tenant_headers(self.slot.secret_value()),
        )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = self._build_client(httpx.AsyncHTTPTransport())
        return self._client

    def _usage_labels(self, model: str) -> JsonObject:
        del model
        return {
            "served_provider": self.slot.kind,
            "provider": self.slot.kind,
            "funding": "tenant_byo",
            "source": "tenant",
        }

    async def _complete_once(
        self, body: JsonObject, headers: Mapping[str, str] | None = None
    ) -> tuple[str, int, int]:
        """One PAID round trip. A transport error / timeout / non-200 raises at once:
        nothing here (nor the re-ask loop, which only follows a schema-invalid 200)
        re-sends a call the provider may already be executing. Errors carry the
        status + provider error code only, never the tenant's response content."""
        del headers  # platform attribution never reaches a third party
        # OpenAI/Anthropic refuse unknown body keys: the vLLM reasoning knob never goes out.
        body.pop("chat_template_kwargs", None)
        body.pop("enable_reasoning", None)
        try:
            response = await self._get_client().post(self._completions_url, json=body)
        except ProviderSlotUnavailableError:
            raise
        except (httpx.TransportError, TimeoutError) as exc:
            raise ProviderSlotUnavailableError(
                self.slot.slot,
                f"{_where(self.slot)} unreachable ({type(exc).__name__})",
                transient=True,
            ) from None
        if response.status_code != 200:
            raise _status_error(self.slot, response)
        try:
            data = cast("JsonObject", response.json())
            choice = cast("list[JsonObject]", data.get("choices") or [{}])[0]
            usage = cast("JsonObject", data.get("usage") or {})
            message = cast("JsonObject", choice.get("message") or {})
            content = str(message.get("content") or "")
            prompt = int(cast("int", usage.get("prompt_tokens") or 0))
            completion = int(cast("int", usage.get("completion_tokens") or 0))
        except ValueError, TypeError, AttributeError, IndexError:
            raise _malformed(self.slot, "not a chat completion") from None
        if choice.get("finish_reason") == "length":
            raise ProviderSlotUnavailableError(
                self.slot.slot,
                f"{_where(self.slot)} hit its token limit ({body.get('max_tokens')}) "
                "before the reply finished",
                status=422,
            )
        return content, prompt, completion

    async def message(
        self,
        worker_name: str,
        *,
        message: str | None,
        content_blocks: Sequence[Mapping[str, object]] | None,
        system_prompt: str | None,
        output_schema: Mapping[str, object] | None,
        max_tokens: int,
        enable_reasoning: bool | None,
        context: Mapping[str, object] | None,
        overrides_ignored: Mapping[str, object],
        headers: Mapping[str, str] | None = None,
    ) -> JsonObject:
        del headers  # platform attribution headers never reach a third party
        payload = await super().message(
            worker_name,
            message=message,
            content_blocks=content_blocks,
            system_prompt=system_prompt,
            output_schema=output_schema,
            max_tokens=max_tokens,
            enable_reasoning=enable_reasoning,
            context=context,
            overrides_ignored=overrides_ignored,
            headers=None,
        )
        metadata = cast("JsonObject", payload.get("metadata") or {})
        metadata.update(
            {
                "source": "tenant",
                "backend": self.slot.kind,
                "slot": self.slot.slot,
            }
        )
        if self.slot.ref is not None:
            metadata["connection_id"] = str(self.slot.ref.connection_id)
            metadata["connection_version"] = self.slot.ref.version
        payload["metadata"] = metadata
        return payload


# ── tts ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TTSRequest:
    """The platform ``TTSClient.synthesize`` contract."""

    text: str
    voice: str | None = None
    speed: float = 1.0
    ssml: bool = False


@dataclass(frozen=True, slots=True)
class TenantSpeech:
    """A tenant synth, shaped like the orchestrator's ``TTSSynthesis``.

    ``audio_format`` is what the BYTES are (magic-byte sniff,
    :func:`~w2v_common.providers.audio.detect_audio_format`), never the provider's
    label. Every kind asks for WAV/PCM — the platform format (``tts_backend``
    answers WAV only) — so ``"wav"`` is the normal case; a tenant server that
    ignores the request yields its true format here, and ``sample_rate`` /
    ``duration_seconds`` are then ``None`` unless it reported them: nothing here
    transcodes (no decoder in this library), the consumer handles it.

    Fields the provider cannot report stay ``None`` so the caller's fallbacks run
    (``integrated_lufs`` → re-master, ``word_timings`` → ASR captions,
    ``speech/pause_seconds`` → authored-break-only pacing).
    """

    audio_bytes: bytes = field(repr=False)
    audio_format: AudioFormat
    sample_rate: int | None
    duration_seconds: float | None
    provider: str
    voice: str
    integrated_lufs: float | None = None
    #: ``[{"word", "start", "end"}]``
    word_timings: list[dict[str, object]] | None = None
    speech_seconds: float | None = None
    pause_seconds: float | None = None
    characters: int = 0
    model: str = ""

    @property
    def mime_type(self) -> str:
        """``audio/wav``, ``audio/mpeg``, … for :attr:`audio_format`."""
        return audio_mime_type(self.audio_format)


def plain_text(text: str, *, ssml: bool) -> str:
    """SSML markup stripped + entities decoded (no tenant kind parses ``<speak>``)."""
    if not ssml:
        return text
    return " ".join(html.unescape(_SSML_TAG.sub(" ", text)).split())


def pcm16_to_wav(pcm: bytes, sample_rate: int, *, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm)
    return buf.getvalue()


@dataclass(frozen=True, slots=True)
class _Wav:
    data: bytes
    sample_rate: int
    duration_seconds: float


def normalise_wav(raw: bytes) -> _Wav:
    """A provider WAV with CORRECT chunk sizes + its duration (``ValueError`` if not WAV).

    OpenAI(-compatible) servers stream WAV with placeholder sizes (``0xFFFFFFFF``); PCM
    is re-wrapped with true sizes, any other encoding passed through untouched.
    """
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("response is not a RIFF/WAVE file")
    pos = 12
    fmt: tuple[int, int, int, int] | None = None
    while pos + 8 <= len(raw):
        chunk_id = raw[pos : pos + 4]
        (size,) = struct.unpack("<I", raw[pos + 4 : pos + 8])
        body = pos + 8
        if chunk_id == b"fmt ":
            if size < 16 or body + 16 > len(raw):
                raise ValueError("WAV fmt chunk is truncated")
            tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", raw[body : body + 16])
            fmt = (tag, channels, rate, bits)
        elif chunk_id == b"data":
            if fmt is None:
                raise ValueError("WAV data chunk precedes its fmt chunk")
            tag, channels, rate, bits = fmt
            data = raw[body : body + min(size, len(raw) - body)]
            frame_bytes = channels * bits // 8
            if rate <= 0 or frame_bytes <= 0:
                raise ValueError("WAV fmt chunk declares no audio")
            data = data[: len(data) - len(data) % frame_bytes]
            duration = len(data) / (rate * frame_bytes)
            if tag == 1 and bits == 16:
                return _Wav(pcm16_to_wav(data, rate, channels=channels), rate, duration)
            return _Wav(raw, rate, duration)
        pos = body + size + (size & 1)
    raise ValueError("WAV has no data chunk")


def word_timings_from_alignment(alignment: object) -> list[dict[str, object]] | None:
    """ElevenLabs character alignment → ``[{word, start, end}]``; ``None`` if unusable."""
    if not isinstance(alignment, Mapping):
        return None
    align = cast("Mapping[str, object]", alignment)
    chars = align.get("characters")
    starts = align.get("character_start_times_seconds")
    ends = align.get("character_end_times_seconds")
    if not (
        isinstance(chars, Sequence)
        and isinstance(starts, Sequence)
        and isinstance(ends, Sequence)
        and not isinstance(chars, str)
        and len(chars) == len(starts) == len(ends)
        and chars
    ):
        return None
    words: list[dict[str, object]] = []
    current: list[str] = []
    first = last = 0.0
    try:
        for ch, start, end in zip(chars, starts, ends, strict=True):
            if not isinstance(ch, str):
                return None
            if ch.isspace() or not ch:
                if current:
                    words.append({"word": "".join(current), "start": first, "end": last})
                    current = []
                continue
            if not current:
                first = float(cast("float", start))
            current.append(ch)
            last = float(cast("float", end))
        if current:
            words.append({"word": "".join(current), "start": first, "end": last})
    except TypeError, ValueError:
        return None
    return words or None


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(max(value, bounds[0]), bounds[1])


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class _Audio:
    data: bytes
    audio_format: AudioFormat
    sample_rate: int | None
    duration_seconds: float | None


def _provider_audio(
    slot: ProviderSlot,
    raw: bytes,
    *,
    declared: str | None,
    sample_rate: int | None = None,
    duration_seconds: float | None = None,
) -> _Audio:
    """Provider bytes → their REAL format; WAV is re-wrapped with true sizes + timed.

    ``declared`` (Content-Type / JSON ``format``) is only cross-checked, and
    ``sample_rate`` / ``duration_seconds`` are what the provider REPORTED (used only
    when the bytes are not WAV — a WAV is measured, not believed).
    """
    try:
        audio_format = detect_audio_format(raw, declared=declared)
        if audio_format == "wav":
            wav = normalise_wav(raw)
            return _Audio(wav.data, "wav", wav.sample_rate, wav.duration_seconds)
    except ValueError as exc:
        raise _malformed(slot, str(exc)) from None
    return _Audio(raw, audio_format, sample_rate, duration_seconds)


async def _openai_speech(slot: ProviderSlot, request: TTSRequest, timeout_s: float) -> TenantSpeech:
    secret = slot.secret_value()
    if slot.kind == KIND_OPENAI_COMPATIBLE:
        base_url = _require_base_url(slot)
    else:
        if not secret:
            raise ProviderSlotUnavailableError(slot.slot, "the connection has no API key")
        base_url = (slot.base_url or OPENAI_DEFAULT_BASE_URL).rstrip("/")
    model = slot.model or OPENAI_TTS_DEFAULT_MODEL
    voice = _option(slot, "voice") or request.voice or OPENAI_TTS_DEFAULT_VOICE
    text = plain_text(request.text, ssml=request.ssml)
    body = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "wav",
        "speed": _clamp(request.speed, _OPENAI_SPEED_RANGE),
    }
    response = await _send(
        slot,
        base_url,
        f"{base_url}/audio/speech",
        timeout_s=timeout_s,
        headers=tenant_headers(secret, {"Accept": "audio/wav"}),
        json_body=body,
    )
    audio = _provider_audio(slot, response.content, declared=response.headers.get("content-type"))
    return TenantSpeech(
        audio_bytes=audio.data,
        audio_format=audio.audio_format,
        sample_rate=audio.sample_rate,
        duration_seconds=audio.duration_seconds,
        provider=slot.kind,
        voice=voice,
        characters=len(text),
        model=model,
    )


async def _elevenlabs_speech(
    slot: ProviderSlot, request: TTSRequest, timeout_s: float
) -> TenantSpeech:
    secret = slot.secret_value()
    if not secret:
        raise ProviderSlotUnavailableError(slot.slot, "the connection has no API key")
    voice_id = _option(slot, "voice_id", "voice") or request.voice
    if not voice_id:
        raise ProviderSlotUnavailableError(
            slot.slot, "the connection has no ElevenLabs voice id configured"
        )
    base_url = (slot.base_url or ELEVENLABS_DEFAULT_BASE_URL).rstrip("/")
    model = slot.model or ELEVENLABS_DEFAULT_MODEL
    text = plain_text(request.text, ssml=request.ssml)
    body: dict[str, object] = {"text": text, "model_id": model}
    if request.speed != 1.0:
        body["voice_settings"] = {"speed": _clamp(request.speed, _ELEVENLABS_SPEED_RANGE)}
    url = (
        f"{base_url}/v1/text-to-speech/{quote(voice_id, safe='')}/with-timestamps"
        f"?output_format=pcm_{ELEVENLABS_PCM_RATE}"
    )
    # ElevenLabs authenticates with ``xi-api-key`` (its API has no Bearer form).
    headers = tenant_headers(
        None, {"xi-api-key": secret, "Accept": "application/json"}, bearer=False
    )
    payload = _json(
        slot, await _send(slot, base_url, url, timeout_s=timeout_s, headers=headers, json_body=body)
    )
    audio_b64 = payload.get("audio_base64")
    if not isinstance(audio_b64, str):
        raise _malformed(slot, "no audio_base64")
    try:
        pcm = base64.b64decode(audio_b64, validate=True)
    except binascii.Error, ValueError:
        raise _malformed(slot, "audio_base64 is not valid base64") from None
    if not pcm:
        raise _malformed(slot, "empty audio")
    pcm = pcm[: len(pcm) - len(pcm) % 2]
    # Raw PCM has no container to sniff: the WAV is OURS, built around
    # ``output_format=pcm_24000`` (exact duration + the platform format).
    return TenantSpeech(
        audio_bytes=pcm16_to_wav(pcm, ELEVENLABS_PCM_RATE),
        audio_format="wav",
        sample_rate=ELEVENLABS_PCM_RATE,
        duration_seconds=len(pcm) / (2 * ELEVENLABS_PCM_RATE),
        provider=slot.kind,
        voice=voice_id,
        word_timings=word_timings_from_alignment(payload.get("alignment")),
        characters=len(text),
        model=model,
    )


def _timings(raw: object) -> list[dict[str, object]] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("word_timings is not a list")
    out: list[dict[str, object]] = []
    for item in cast("list[object]", raw):
        if not isinstance(item, Mapping):
            raise ValueError("word timing is not an object")
        entry = cast("Mapping[str, object]", item)
        start, end, word = _float(entry.get("start")), _float(entry.get("end")), entry.get("word")
        if start is None or end is None or not isinstance(word, str):
            raise ValueError("word timing lacks word/start/end")
        out.append({"word": word, "start": start, "end": end})
    return out


async def _nonagentic_speech(
    slot: ProviderSlot, request: TTSRequest, timeout_s: float
) -> TenantSpeech:
    base_url = _require_base_url(slot)
    voice = _option(slot, "voice") or request.voice
    model = slot.model or "platform"
    # ``output_format`` is the gateway/tts_backend ``SynthesizeRequest`` field; WAV
    # is the only value tts_backend accepts (mp3 → 422 since #55) and its default.
    body: dict[str, object] = {"text": request.text, "speed": request.speed, "output_format": "wav"}
    if voice is not None:
        body["voice"] = voice
    if request.ssml:
        body["ssml"] = True
    response = await _send(
        slot,
        base_url,
        f"{base_url}{NONAGENTIC_TTS_PATH}",
        timeout_s=timeout_s,
        headers=tenant_headers(slot.secret_value(), {"Accept": "application/json"}),
        json_body=body,
    )
    lufs = _float(response.headers.get(_LUFS_HEADER))
    characters = len(plain_text(request.text, ssml=request.ssml))
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("audio/"):
        rate = _float(response.headers.get(_SAMPLE_RATE_HEADER))
        audio = _provider_audio(
            slot,
            response.content,
            declared=content_type,
            sample_rate=int(rate) if rate is not None and rate > 0 else None,
            duration_seconds=_float(response.headers.get(_DURATION_HEADER)),
        )
        return TenantSpeech(
            audio_bytes=audio.data,
            audio_format=audio.audio_format,
            sample_rate=audio.sample_rate,
            duration_seconds=audio.duration_seconds,
            provider=slot.kind,
            voice=voice or "",
            integrated_lufs=lufs,
            characters=characters,
            model=model,
        )
    wire = _json(slot, response)
    try:
        audio_b64 = wire["audio_base64"]
        sample_rate = wire["sample_rate"]
        duration = _float(wire.get("duration_sec"))
        if not isinstance(audio_b64, str) or not isinstance(sample_rate, int) or duration is None:
            raise ValueError("audio_base64/sample_rate/duration_sec")
        audio = base64.b64decode(audio_b64, validate=True)
        timings = _timings(wire.get("word_timings"))
    except (KeyError, ValueError, binascii.Error) as exc:
        raise _malformed(slot, f"not a synthesize response ({type(exc).__name__})") from None
    if not audio:
        raise _malformed(slot, "empty audio")
    provider, wire_voice, fmt = wire.get("provider"), wire.get("voice"), wire.get("format")
    try:
        audio_format = detect_audio_format(audio, declared=fmt if isinstance(fmt, str) else None)
    except ValueError as exc:
        raise _malformed(slot, str(exc)) from None
    return TenantSpeech(
        audio_bytes=audio,
        audio_format=audio_format,
        sample_rate=sample_rate,
        duration_seconds=duration,
        provider=provider if isinstance(provider, str) else slot.kind,
        voice=wire_voice if isinstance(wire_voice, str) else (voice or ""),
        integrated_lufs=lufs,
        word_timings=timings,
        speech_seconds=_float(wire.get("speech_seconds")),
        pause_seconds=_float(wire.get("pause_seconds")),
        characters=characters,
        model=model,
    )


async def tenant_tts(
    slot: ProviderSlot, request: TTSRequest, *, timeout_s: float = DEFAULT_TIMEOUT_S
) -> TenantSpeech:
    """Synthesize ``request`` on the tenant ``tts`` connection ``slot``."""
    _require_tenant(slot)
    if slot.kind in {KIND_OPENAI, KIND_OPENAI_COMPATIBLE}:
        return await _openai_speech(slot, request, timeout_s)
    if slot.kind == KIND_ELEVENLABS:
        return await _elevenlabs_speech(slot, request, timeout_s)
    if slot.kind == KIND_NONAGENTIC_ENDPOINT:
        return await _nonagentic_speech(slot, request, timeout_s)
    raise ProviderSlotUnavailableError(slot.slot, f"kind {slot.kind!r} is not a supported TTS kind")


# ── asr ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ASRRequest:
    """The platform ``transcribe_file`` contract (audio already in memory).

    ``filename`` / ``content_type`` are the caller's LABELS; the upload is labelled
    from the bytes when they are a recognised container (see :func:`asr_upload`),
    so WAV sent as ``audio.mp3`` cannot make an OpenAI-compatible server — which
    picks its decoder from the file extension — refuse or mis-decode it.
    """

    audio: bytes = field(repr=False)
    filename: str | None = None
    content_type: str | None = None
    language: str | None = None
    word_timestamps: bool = True
    return_srt: bool = False


#: What OpenAI's ``/audio/transcriptions`` decodes (its documented upload list;
#: ``opus`` travels as Ogg). ``nonagentic_endpoint`` (asr_backend) decodes with
#: ffmpeg and takes any container.
OPENAI_ASR_FORMATS: Final[frozenset[AudioFormat]] = frozenset(
    {"wav", "mp3", "ogg", "opus", "flac", "m4a", "webm"}
)


def asr_upload(request: ASRRequest) -> tuple[str, bytes, str]:
    """``(filename, bytes, content_type)`` for the multipart ``file`` part.

    Recognised bytes win over the caller's labels (the stem of ``filename`` is
    kept); unrecognised bytes keep the labels, else ``audio`` /
    ``application/octet-stream`` — never a guessed format.
    """
    stem = (request.filename or "audio").rsplit(".", 1)[0] or "audio"
    sniffed = sniff_audio_format(request.audio)
    if sniffed is None:
        return (
            request.filename or "audio",
            request.audio,
            request.content_type or "application/octet-stream",
        )
    return f"{stem}.{audio_file_extension(sniffed)}", request.audio, audio_mime_type(sniffed)


def _words(raw: object) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        return out
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        entry = cast("Mapping[str, object]", item)
        start, end, word = _float(entry.get("start")), _float(entry.get("end")), entry.get("word")
        if start is None or end is None or not isinstance(word, str):
            continue
        out.append(
            {
                "start": start,
                "end": end,
                "word": word,
                "probability": _float(entry.get("probability")),
            }
        )
    return out


def _segments(raw: object) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        return out
    for i, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        seg = cast("Mapping[str, object]", item)
        start, end, text = _float(seg.get("start")), _float(seg.get("end")), seg.get("text")
        if start is None or end is None or not isinstance(text, str):
            continue
        raw_id = seg.get("id")
        out.append(
            {
                "id": raw_id if isinstance(raw_id, int) and not isinstance(raw_id, bool) else i,
                "start": start,
                "end": end,
                "text": text,
                "words": _words(seg.get("words")),
                "avg_logprob": _float(seg.get("avg_logprob")),
                "no_speech_prob": _float(seg.get("no_speech_prob")),
            }
        )
    return out


def _seg_float(seg: Mapping[str, object], key: str) -> float:
    return cast("float", seg[key])


def openai_transcript(
    payload: Mapping[str, object], *, model: str, backend: str
) -> dict[str, object]:
    """OpenAI ``verbose_json`` → the platform ``TranscriptionResponse`` dict shape.

    Top-level ``words`` go to the segment holding the word's midpoint (else the last
    segment starting before it); with no segments one synthetic segment spans them.
    """
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("no text")
    top_words = _words(payload.get("words"))
    segments = _segments(payload.get("segments"))
    duration = _float(payload.get("duration"))
    if not segments and (top_words or text.strip()):
        first = cast("float", top_words[0]["start"]) if top_words else 0.0
        last = cast("float", top_words[-1]["end"]) if top_words else (duration or 0.0)
        segments.append({"id": 0, "start": first, "end": last, "text": text, "words": []})
    if top_words and not any(seg.get("words") for seg in segments):
        for word in top_words:
            mid = (cast("float", word["start"]) + cast("float", word["end"])) / 2
            target = next(
                (s for s in segments if _seg_float(s, "start") <= mid <= _seg_float(s, "end")),
                None,
            )
            if target is None:
                before = [s for s in segments if _seg_float(s, "start") <= mid]
                target = before[-1] if before else segments[0]
            cast("list[object]", target["words"]).append(word)
    if duration is None:
        duration = _seg_float(segments[-1], "end") if segments else 0.0
    language = payload.get("language")
    return {
        "backend": backend,
        "model": model,
        "language": language if isinstance(language, str) else None,
        "duration": duration,
        "text": text,
        "segments": segments,
    }


async def tenant_asr(
    slot: ProviderSlot, request: ASRRequest, *, timeout_s: float = DEFAULT_TIMEOUT_S
) -> dict[str, object]:
    """Transcribe on the tenant ``asr`` connection; the platform transcript dict shape.

    ``nonagentic_endpoint`` answers ARE the platform shape and are returned as sent
    (the caller validates them into its own model); ``openai-compatible`` is mapped.
    """
    _require_tenant(slot)
    base_url = _require_base_url(slot)
    headers = tenant_headers(slot.secret_value(), {"Accept": "application/json"})
    files = {"file": asr_upload(request)}
    if slot.kind == KIND_NONAGENTIC_ENDPOINT:
        data: dict[str, str | list[str]] = {
            "word_timestamps": "true" if request.word_timestamps else "false",
            "return_srt": "true" if request.return_srt else "false",
        }
        if request.language is not None:
            data["language"] = request.language
        response = await _send(
            slot,
            base_url,
            f"{base_url}{NONAGENTIC_ASR_PATH}",
            timeout_s=timeout_s,
            headers=headers,
            data=data,
            files=files,
        )
        payload = dict(_json(slot, response))
        if not isinstance(payload.get("text"), str) or not isinstance(
            payload.get("segments"), list
        ):
            raise _malformed(slot, "not a transcription response (text/segments)")
        return payload
    if slot.kind == KIND_OPENAI_COMPATIBLE:
        sniffed = sniff_audio_format(request.audio)
        if sniffed is not None and sniffed not in OPENAI_ASR_FORMATS:
            raise ProviderSlotUnavailableError(
                slot.slot, f"an openai-compatible ASR connection does not accept {sniffed} audio"
            )
        model = slot.model or ASR_DEFAULT_MODEL
        form: dict[str, str | list[str]] = {"model": model, "response_format": "verbose_json"}
        if request.word_timestamps:
            form["timestamp_granularities[]"] = ["word", "segment"]
        if request.language is not None:
            form["language"] = request.language
        response = await _send(
            slot,
            base_url,
            f"{base_url}{OPENAI_ASR_PATH}",
            timeout_s=timeout_s,
            headers=headers,
            data=form,
            files=files,
        )
        try:
            return openai_transcript(_json(slot, response), model=model, backend=slot.kind)
        except ValueError as exc:
            raise _malformed(slot, str(exc)[:200]) from None
    raise ProviderSlotUnavailableError(slot.slot, f"kind {slot.kind!r} is not a supported ASR kind")


# ── ocr ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OcrPage:
    index: int
    markdown: str


async def tenant_ocr(
    slot: ProviderSlot,
    pdf: bytes,
    *,
    pages: Sequence[int] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    default_model: str | None = None,
) -> list[OcrPage]:
    """OCR ``pdf`` on the tenant ``ocr`` connection (Mistral OCR request/response shape).

    Kinds (orchestrator registry + ``src/clients/ocr_client.py``):

    * ``nonagentic_endpoint`` — a copy of the platform service: ``POST {base}/v1/ocr``;
      model optional (``default_model`` — the caller's platform OCR model — else omitted);
    * ``openai-compatible`` — a ``/v1`` base URL, Mistral-style: ``POST {base}/ocr``;
      model REQUIRED.
    """
    _require_tenant(slot)
    base_url = _require_base_url(slot)
    if slot.kind == KIND_NONAGENTIC_ENDPOINT:
        url = f"{base_url}{OCR_PATH}"
        model = slot.model or default_model
    elif slot.kind == KIND_OPENAI_COMPATIBLE:
        url = f"{base_url}{OCR_COMPAT_PATH}"
        model = slot.model
        if not model:
            raise ProviderSlotUnavailableError(slot.slot, "the connection has no model configured")
    else:
        raise ProviderSlotUnavailableError(
            slot.slot, f"kind {slot.kind!r} is not a supported OCR kind"
        )
    body: dict[str, object] = {
        "document": {
            "type": "document_url",
            "document_url": "data:application/pdf;base64," + base64.b64encode(pdf).decode("ascii"),
        },
        "include_image_base64": False,
    }
    if model:
        body["model"] = model
    if pages is not None:
        body["pages"] = list(pages)
    response = await _send(
        slot,
        base_url,
        url,
        timeout_s=timeout_s,
        headers=tenant_headers(slot.secret_value(), {"Accept": "application/json"}),
        json_body=body,
    )
    raw_pages = _json(slot, response).get("pages")
    if not isinstance(raw_pages, list):
        raise _malformed(slot, "no 'pages' list")
    out: list[OcrPage] = []
    for item in cast("list[object]", raw_pages):
        if not isinstance(item, Mapping):
            raise _malformed(slot, "page is not an object")
        page = cast("Mapping[str, object]", item)
        index, markdown = page.get("index"), page.get("markdown")
        if not isinstance(index, int) or isinstance(index, bool) or not isinstance(markdown, str):
            raise _malformed(slot, "page lacks an int 'index' / str 'markdown'")
        out.append(OcrPage(index=index, markdown=markdown))
    return sorted(out, key=lambda p: p.index)


__all__ = [
    "OPENAI_ASR_FORMATS",
    "ASRRequest",
    "OcrPage",
    "TTSRequest",
    "TenantLLMClient",
    "TenantSpeech",
    "asr_upload",
    "normalise_wav",
    "openai_transcript",
    "pcm16_to_wav",
    "plain_text",
    "tenant_asr",
    "tenant_ocr",
    "tenant_tts",
    "word_timings_from_alignment",
]
