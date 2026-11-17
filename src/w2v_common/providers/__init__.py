"""Run provider slots: the tenant's own provider or the platform default, per call.

Shipped DARK: ``W2V_PROVIDER_SLOTS=off`` (default) changes nothing anywhere.
See :mod:`w2v_common.providers.slots` for the orchestrator contract and
:mod:`w2v_common.providers.adapters` for the tenant-kind adapters.
"""

from __future__ import annotations

from w2v_common.providers.adapters import (
    ASRRequest,
    OcrPage,
    TenantLLMClient,
    TenantSpeech,
    TTSRequest,
    tenant_asr,
    tenant_ocr,
    tenant_tts,
)
from w2v_common.providers.audio import (
    AudioFormat,
    audio_mime_type,
    detect_audio_format,
    sniff_audio_format,
)
from w2v_common.providers.errors import (
    BYO_PROVIDER_REQUIRED,
    ProviderHopRefusedError,
    ProviderSlotConfigError,
    ProviderSlotError,
    ProviderSlotRequiredError,
    ProviderSlotUnavailableError,
)
from w2v_common.providers.helpers import asr_transcribe, tts_synthesize
from w2v_common.providers.slots import (
    SWITCH_ENV,
    ProviderSlot,
    ProviderSlotClient,
    ProviderSlotRef,
    clear_provider_slot_cache,
    provider_slots_enforced,
    provider_slots_mode,
    resolve_provider_slot,
)

__all__ = [
    "BYO_PROVIDER_REQUIRED",
    "SWITCH_ENV",
    "ASRRequest",
    "AudioFormat",
    "OcrPage",
    "ProviderHopRefusedError",
    "ProviderSlot",
    "ProviderSlotClient",
    "ProviderSlotConfigError",
    "ProviderSlotError",
    "ProviderSlotRef",
    "ProviderSlotRequiredError",
    "ProviderSlotUnavailableError",
    "TTSRequest",
    "TenantLLMClient",
    "TenantSpeech",
    "asr_transcribe",
    "audio_mime_type",
    "clear_provider_slot_cache",
    "detect_audio_format",
    "provider_slots_enforced",
    "provider_slots_mode",
    "resolve_provider_slot",
    "sniff_audio_format",
    "tenant_asr",
    "tenant_ocr",
    "tenant_tts",
    "tts_synthesize",
]
