"""Thin entry points a consumer wraps around its EXISTING platform client.

    speech = await tts_synthesize(
        TTSRequest(text=..., voice=..., speed=..., ssml=...),
        platform=lambda: tts_client.synthesize(text, voice=..., speed=..., ssml=...),
        from_tenant=to_tts_synthesis,   # TenantSpeech -> the consumer's own result type
    )

``W2V_PROVIDER_SLOTS=off`` (default): ``platform()`` is awaited and returned, nothing
else happens — no resolve call, no import-time side effect, byte-identical to before.
``enforce``: the run's slot is resolved; ``source="tenant"`` → the tenant adapter and
``from_tenant``; ``source="deployment"`` → ``platform()`` unchanged (the platform client
keeps its service licence + ``X-Effective-Tenant-Id``; a ``byo_required`` tenant was
already refused by the resolve with :class:`ProviderSlotRequiredError`).

Audio formats — :attr:`TenantSpeech.audio_format` is sniffed from the bytes
(magic: ``RIFF…WAVE`` / ``ID3`` or an MPEG Layer III frame sync / ``OggS`` /
``fLaC`` / ``ftyp`` / EBML), never taken from a Content-Type or ``format`` label:

==================== ======================================= ====================
tenant kind          requested                               ``audio_format``
==================== ======================================= ====================
openai               ``response_format: "wav"``              ``wav`` (true format
openai-compatible    ``response_format: "wav"``              if the server ignored
                                                             the request)
elevenlabs           ``output_format=pcm_24000`` → wrapped   ``wav`` always
nonagentic_endpoint  ``output_format: "wav"``                ``wav`` (true format
                                                             otherwise)
==================== ======================================= ====================

WAV, not mp3, is requested because it IS the platform format: the platform
``POST /api/tts/synthesize`` (modality-gateway relaying ``tts_backend``) answers
``audio/wav`` only, and ``output_format: "mp3"`` is a 422 there since tts_backend
#55. A consumer whose platform call asks for mp3 is not getting mp3 — sniff its
platform bytes with :func:`~w2v_common.providers.audio.detect_audio_format` too.
Nothing is transcoded here (this library has no audio codec): a non-WAV tenant
answer comes back in its true format with ``sample_rate`` / ``duration_seconds``
``None`` unless the provider reported them, for the consumer to handle.

ASR input: the multipart upload is labelled from the bytes (WAV sent as
``audio.mp3`` becomes ``audio.wav`` / ``audio/wav``). ``openai-compatible``
accepts ``wav mp3 ogg opus flac m4a webm`` (anything else recognised is refused
before the paid call); ``nonagentic_endpoint`` decodes any container.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from w2v_common.providers.adapters import (
    DEFAULT_TIMEOUT_S,
    ASRRequest,
    TenantSpeech,
    TTSRequest,
    tenant_asr,
    tenant_tts,
)
from w2v_common.providers.slots import provider_slots_enforced, resolve_provider_slot


async def tts_synthesize[T](
    request: TTSRequest,
    *,
    platform: Callable[[], Awaitable[T]],
    from_tenant: Callable[[TenantSpeech], T],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> T:
    """The run's ``tts`` slot: tenant adapter or today's platform call."""
    if not provider_slots_enforced():
        return await platform()
    slot = await resolve_provider_slot("tts")
    if not slot.is_tenant:
        return await platform()
    return from_tenant(await tenant_tts(slot, request, timeout_s=timeout_s))


async def asr_transcribe[T](
    request: ASRRequest,
    *,
    platform: Callable[[], Awaitable[T]],
    from_tenant: Callable[[dict[str, object]], T],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> T:
    """The run's ``asr`` slot: tenant adapter (platform transcript dict) or today's call.

    ``request`` is only read on the tenant path; build it lazily if reading the
    audio is expensive by checking :func:`provider_slots_enforced` first.
    """
    if not provider_slots_enforced():
        return await platform()
    slot = await resolve_provider_slot("asr")
    if not slot.is_tenant:
        return await platform()
    return from_tenant(await tenant_asr(slot, request, timeout_s=timeout_s))


__all__ = ["asr_transcribe", "tts_synthesize"]
