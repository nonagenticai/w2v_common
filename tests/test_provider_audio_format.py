"""Tenant TTS/ASR audio formats: the magic bytes decide, never the provider's label."""

from __future__ import annotations

import base64
import json
import logging

import httpx
import pytest
import respx

from w2v_common.providers import (
    ASRRequest,
    ProviderSlot,
    ProviderSlotUnavailableError,
    TTSRequest,
    detect_audio_format,
    sniff_audio_format,
    tenant_asr,
    tenant_tts,
)
from w2v_common.providers import _http as provider_http
from w2v_common.providers.adapters import asr_upload, pcm16_to_wav

PUBLIC_IP = "93.184.216.34"
SECRET = "sk-tenant-audio-xyz"  # pragma: allowlist secret
WAV = pcm16_to_wav(b"\x01\x00" * 2400, 24_000)  # 0.1 s
MP3_ID3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x00" + b"\x00" * 64
MP3_FRAME = b"\xff\xfb\x90\x00" + b"\x00" * 64
OGG_OPUS = b"OggS\x00\x02" + b"\x00" * 22 + b"OpusHead" + b"\x00" * 16
OGG_VORBIS = b"OggS\x00\x02" + b"\x00" * 22 + b"\x01vorbis" + b"\x00" * 16


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _public(_host: str, _port: int) -> list[str]:
        return [PUBLIC_IP]

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(provider_http, "_resolve", _public)
    monkeypatch.setattr(provider_http, "_sleep", _no_sleep)


def _slot(kind: str, slot: str = "tts", **extra: object) -> ProviderSlot:
    payload: dict[str, object] = {
        "source": "tenant",
        "slot": slot,
        "kind": kind,
        "base_url": "https://audio.tenant.example/v1",
        "model": None,
        "options": {},
        "secret": SECRET,
        "ref": {"connection_id": "0199a1b2-2222-7000-8000-000000000009", "version": 1},
    }
    payload.update(extra)
    return ProviderSlot.from_payload(payload, expected_slot=slot)


# ── sniffing ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (WAV, "wav"),
        (MP3_ID3, "mp3"),
        (MP3_FRAME, "mp3"),
        (b"\xff\xf3\x64\xc4" + b"\x00" * 8, "mp3"),  # MPEG-2 Layer III
        (OGG_OPUS, "opus"),
        (OGG_VORBIS, "ogg"),
        (b"fLaC\x00\x00\x00\x22", "flac"),
        (b"\x00\x00\x00\x20ftypM4A ", "m4a"),
        (b"\x1a\x45\xdf\xa3\x01\x00", "webm"),
        (b"\xff\xf1\x50\x80\x00\x1f\xfc", "aac"),
        (b"RIFF\x00\x00\x00\x00AVI ", None),  # RIFF but not WAVE
        (b"\xff\xfd\x00\x00", None),  # MPEG Layer II, not mp3
        (b'{"error":"x"}', None),
        (b"", None),
    ],
)
def test_sniff_reads_the_container_from_the_bytes(data: bytes, expected: str | None) -> None:
    assert sniff_audio_format(data) == expected


def test_detect_never_trusts_a_label_alone(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(ValueError, match="not a recognised audio container"):
        detect_audio_format(b"<html>nope</html>", declared="audio/mpeg")
    with caplog.at_level(logging.WARNING, logger="w2v_common.providers.audio"):
        assert detect_audio_format(WAV, declared="audio/mpeg; charset=binary") == "wav"
    assert "labelled its audio mp3 but the bytes are wav" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="w2v_common.providers.audio"):
        assert detect_audio_format(OGG_OPUS, declared="audio/ogg") == "opus"
    assert caplog.text == ""  # same container family, not a disagreement


# ── tts: one test per kind ──────────────────────────────────────────────────


@respx.mock
async def test_openai_requests_wav_and_wav_labelled_mpeg_is_wav(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(f"https://{PUBLIC_IP}/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=WAV, headers={"content-type": "audio/mpeg"})
    )
    speech = await tenant_tts(_slot("openai"), TTSRequest(text="hello"))
    assert json.loads(route.calls[0].request.content)["response_format"] == "wav"
    assert speech.audio_format == "wav" and speech.mime_type == "audio/wav"
    assert speech.sample_rate == 24_000 and speech.duration_seconds == pytest.approx(0.1)


@respx.mock
async def test_openai_compatible_ignoring_the_request_returns_the_true_format(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post(f"https://{PUBLIC_IP}/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=MP3_ID3, headers={"content-type": "audio/wav"})
    )
    speech = await tenant_tts(_slot("openai-compatible"), TTSRequest(text="hello"))
    assert speech.audio_format == "mp3" and speech.mime_type == "audio/mpeg"
    assert speech.audio_bytes == MP3_ID3  # never transcoded
    # Not measurable without a decoder, and no provider report: absent, not guessed.
    assert speech.sample_rate is None and speech.duration_seconds is None


@respx.mock
async def test_openai_compatible_non_audio_labelled_audio_is_refused(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post(f"https://{PUBLIC_IP}/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"oops", headers={"content-type": "audio/mpeg"})
    )
    with pytest.raises(ProviderSlotUnavailableError, match="not a recognised audio container"):
        await tenant_tts(_slot("openai-compatible"), TTSRequest(text="hello"))


@respx.mock
async def test_elevenlabs_pcm_is_wrapped_as_wav(respx_mock: respx.MockRouter) -> None:
    pcm = b"\x02\x00" * 2400
    route = respx_mock.post(
        url__regex=rf"https://{PUBLIC_IP.replace('.', r'\.')}/v1/text-to-speech/v-9/with-timestamps.*"
    ).mock(return_value=httpx.Response(200, json={"audio_base64": base64.b64encode(pcm).decode()}))
    slot = _slot("elevenlabs", base_url=None, options={"voice_id": "v-9"})
    speech = await tenant_tts(slot, TTSRequest(text="hi"))
    assert route.calls[0].request.url.params["output_format"] == "pcm_24000"
    assert speech.audio_format == "wav" and sniff_audio_format(speech.audio_bytes) == "wav"
    assert speech.duration_seconds == pytest.approx(0.1)


@respx.mock
async def test_nonagentic_json_label_mp3_but_wav_bytes_is_wav(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(f"https://{PUBLIC_IP}/api/tts/synthesize").mock(
        return_value=httpx.Response(
            200,
            json={
                "audio_base64": base64.b64encode(WAV).decode(),
                "sample_rate": 24000,
                "duration_sec": 0.1,
                "provider": "kokoro",
                "voice": "af",
                "format": "mp3",
            },
        )
    )
    slot = _slot("nonagentic_endpoint", base_url="https://gpu.tenant.example")
    speech = await tenant_tts(slot, TTSRequest(text="x"))
    assert json.loads(route.calls[0].request.content)["output_format"] == "wav"
    assert speech.audio_format == "wav" and speech.mime_type == "audio/wav"


@respx.mock
async def test_nonagentic_json_garbage_audio_is_refused(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(f"https://{PUBLIC_IP}/api/tts/synthesize").mock(
        return_value=httpx.Response(
            200,
            json={
                "audio_base64": base64.b64encode(b"not audio").decode(),
                "sample_rate": 24000,
                "duration_sec": 0.1,
                "format": "wav",
            },
        )
    )
    slot = _slot("nonagentic_endpoint", base_url="https://gpu.tenant.example")
    with pytest.raises(ProviderSlotUnavailableError, match="not a recognised audio container"):
        await tenant_tts(slot, TTSRequest(text="x"))


@respx.mock
async def test_nonagentic_binary_mp3_keeps_its_reported_timing(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post(f"https://{PUBLIC_IP}/api/tts/synthesize").mock(
        return_value=httpx.Response(
            200,
            content=MP3_FRAME,
            headers={
                "content-type": "audio/wav",
                "X-TTS-Sample-Rate": "44100",
                "X-TTS-Duration-Sec": "1.250",
            },
        )
    )
    slot = _slot("nonagentic_endpoint", base_url="https://gpu.tenant.example")
    speech = await tenant_tts(slot, TTSRequest(text="x"))
    assert speech.audio_format == "mp3"
    assert speech.sample_rate == 44_100 and speech.duration_seconds == pytest.approx(1.25)


# ── asr input ───────────────────────────────────────────────────────────────


def test_asr_upload_is_labelled_from_the_bytes() -> None:
    assert asr_upload(ASRRequest(audio=WAV, filename="clip.mp3", content_type="audio/mpeg")) == (
        "clip.wav",
        WAV,
        "audio/wav",
    )
    assert asr_upload(ASRRequest(audio=MP3_FRAME))[::2] == ("audio.mp3", "audio/mpeg")
    assert asr_upload(ASRRequest(audio=OGG_OPUS))[::2] == ("audio.ogg", "audio/ogg")
    # Unrecognised bytes: the caller's labels, else a neutral one — never a guess.
    assert asr_upload(ASRRequest(audio=b"???", filename="a.mp4"))[::2] == (
        "a.mp4",
        "application/octet-stream",
    )
    assert asr_upload(ASRRequest(audio=b"???"))[::2] == ("audio", "application/octet-stream")


@respx.mock
async def test_asr_wav_labelled_mp3_is_uploaded_as_wav(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"https://{PUBLIC_IP}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "hi", "segments": [], "words": []})
    )
    await tenant_asr(
        _slot("openai-compatible", slot="asr"),
        ASRRequest(audio=WAV, filename="audio.mp3", content_type="audio/mpeg"),
    )
    body = route.calls[0].request.content
    assert b'filename="audio.wav"' in body and b"Content-Type: audio/wav" in body


@respx.mock
async def test_openai_compatible_asr_refuses_a_format_it_cannot_decode(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(f"https://{PUBLIC_IP}/v1/audio/transcriptions")
    with pytest.raises(ProviderSlotUnavailableError, match="does not accept aac"):
        await tenant_asr(
            _slot("openai-compatible", slot="asr"), ASRRequest(audio=b"\xff\xf1\x50\x80\x00")
        )
    assert route.call_count == 0  # refused before the paid call
