"""Audio container detection: the MAGIC BYTES decide, a label never does.

A tenant provider's ``Content-Type`` and its JSON ``format`` field are claims — an
OpenAI-compatible server that ignores ``response_format`` still answers
``audio/wav``, a proxy relabels everything ``audio/mpeg``. Every tenant synth's
:attr:`~w2v_common.providers.adapters.TenantSpeech.audio_format` comes from
:func:`detect_audio_format`, which reads the bytes and only *logs* a disagreeing
label. Consumers can call it on platform bytes too (the platform TTS path is WAV
only — ``tts_backend`` rejects ``output_format="mp3"`` with 422 since #55).
"""

from __future__ import annotations

import logging
from typing import Final, Literal

logger = logging.getLogger(__name__)

type AudioFormat = Literal["wav", "mp3", "ogg", "opus", "flac", "aac", "m4a", "webm"]

_MIME: Final[dict[AudioFormat, str]] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "opus": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "m4a": "audio/mp4",
    "webm": "audio/webm",
}
_EXTENSION: Final[dict[AudioFormat, str]] = {
    "wav": "wav",
    "mp3": "mp3",
    "ogg": "ogg",
    "opus": "ogg",
    "flac": "flac",
    "aac": "aac",
    "m4a": "m4a",
    "webm": "webm",
}
#: Labels (MIME type or bare format name) → the format they CLAIM.
_LABELS: Final[dict[str, AudioFormat]] = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/vnd.wave": "wav",
    "wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mpeg3": "mp3",
    "mp3": "mp3",
    "audio/ogg": "ogg",
    "ogg": "ogg",
    "audio/opus": "opus",
    "opus": "opus",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "flac": "flac",
    "audio/aac": "aac",
    "audio/x-aac": "aac",
    "aac": "aac",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "m4a": "m4a",
    "audio/webm": "webm",
    "webm": "webm",
}
#: Same container, different name — not a disagreement worth a warning.
_OGG_FAMILY: Final = frozenset({"ogg", "opus"})


def sniff_audio_format(data: bytes) -> AudioFormat | None:
    """The container ``data`` actually is, from its leading bytes; ``None`` if unknown."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"OggS":
        # The first page carries the codec id header (``OpusHead`` at offset 28).
        return "opus" if b"OpusHead" in data[:64] else "ogg"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:3] == b"ID3":
        return "mp3"
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "m4a"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0:
        version, layer = (data[1] >> 3) & 0b11, (data[1] >> 1) & 0b11
        if layer == 0b01 and version != 0b01:  # MPEG audio Layer III (0xFFFB/F3/E3…)
            return "mp3"
        if layer == 0b00 and data[1] & 0xF0 == 0xF0:  # ADTS AAC (0xFFF1/F9)
            return "aac"
    return None


def label_audio_format(label: str | None) -> AudioFormat | None:
    """The format a ``Content-Type`` / ``format`` label claims (parameters ignored)."""
    if not label:
        return None
    return _LABELS.get(label.split(";", 1)[0].strip().lower())


def detect_audio_format(data: bytes, *, declared: str | None = None) -> AudioFormat:
    """``data``'s real container; ``declared`` (a label) is only cross-checked.

    Raises ``ValueError`` when the bytes are no recognised container — a label
    alone is never enough to call something audio.
    """
    actual = sniff_audio_format(data)
    if actual is None:
        raise ValueError("response is not a recognised audio container")
    claimed = label_audio_format(declared)
    if claimed is not None and claimed != actual and {claimed, actual} - _OGG_FAMILY:
        logger.warning(
            "provider labelled its audio %s but the bytes are %s; using %s",
            claimed,
            actual,
            actual,
        )
    return actual


def audio_mime_type(audio_format: AudioFormat) -> str:
    return _MIME[audio_format]


def audio_file_extension(audio_format: AudioFormat) -> str:
    return _EXTENSION[audio_format]


__all__ = [
    "AudioFormat",
    "audio_file_extension",
    "audio_mime_type",
    "detect_audio_format",
    "label_audio_format",
    "sniff_audio_format",
]
