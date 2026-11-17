"""Pure WCAG 2.x colour-contrast math shared across the w2v fleet.

Stdlib-only (no numpy, no httpx): parse colours, compute sRGB relative
luminance, contrast ratios, WCAG AA/AAA pass checks, and pick the
best-contrast text colour for a given background. Deterministic.

Reference: https://www.w3.org/TR/WCAG21/#dfn-relative-luminance and
https://www.w3.org/TR/WCAG21/#dfn-contrast-ratio
"""

from __future__ import annotations

import re

__all__ = [
    "best_text_color",
    "contrast_ratio",
    "parse_color",
    "passes_wcag",
    "relative_luminance",
]

RGB = tuple[int, int, int]

# rgb(r, g, b) / rgb(r g b) — integer channels only.
_RGB_FUNC_RE = re.compile(
    r"^rgb\(\s*(\d{1,3})\s*[,\s]\s*(\d{1,3})\s*[,\s]\s*(\d{1,3})\s*\)$",
    re.IGNORECASE,
)


def _clamp_channel(value: int) -> int:
    if not 0 <= value <= 255:
        raise ValueError(f"colour channel out of range [0, 255]: {value}")
    return value


def _parse_tuple(c: tuple[object, ...]) -> RGB:
    if len(c) != 3:
        raise ValueError(f"expected a 3-tuple (r, g, b), got: {c!r}")
    channels: list[int] = []
    for channel in c:
        if not isinstance(channel, int) or isinstance(channel, bool):
            raise ValueError(f"colour channels must be ints, got: {c!r}")
        channels.append(_clamp_channel(channel))
    return (channels[0], channels[1], channels[2])


def _parse_hex(text: str, original: object) -> RGB:
    hexpart = text[1:]
    if len(hexpart) == 3:
        hexpart = "".join(ch * 2 for ch in hexpart)
    if len(hexpart) != 6:
        raise ValueError(f"invalid hex colour: {original!r}")
    try:
        return (int(hexpart[0:2], 16), int(hexpart[2:4], 16), int(hexpart[4:6], 16))
    except ValueError as exc:
        raise ValueError(f"invalid hex colour: {original!r}") from exc


def parse_color(c: str | tuple[int, int, int]) -> RGB:
    """Parse a colour into an ``(r, g, b)`` tuple of 0-255 ints.

    Accepts ``#RGB``, ``#RRGGBB``, ``rgb(r, g, b)`` strings and raw
    ``(r, g, b)`` tuples. Raises ``ValueError`` on anything else.
    """
    # Runtime type guards: annotations are not enforced at call time.
    if isinstance(c, tuple):
        return _parse_tuple(c)
    if not isinstance(c, str):  # pragma: no cover - defensive
        raise ValueError(f"unsupported colour type: {type(c).__name__}")

    text = c.strip()
    if text.startswith("#"):
        return _parse_hex(text, c)

    match = _RGB_FUNC_RE.match(text)
    if match:
        r, g, b = (int(match.group(i)) for i in (1, 2, 3))
        return (_clamp_channel(r), _clamp_channel(g), _clamp_channel(b))

    raise ValueError(f"unparseable colour: {c!r}")


def _linearize(channel: int) -> float:
    c = channel / 255.0
    if c <= 0.03928:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: str | tuple[int, int, int]) -> float:
    """WCAG relative luminance of a colour, in ``[0.0, 1.0]``."""
    r, g, b = parse_color(rgb)
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def contrast_ratio(
    fg: str | tuple[int, int, int],
    bg: str | tuple[int, int, int],
) -> float:
    """WCAG contrast ratio between two colours, in ``[1.0, 21.0]``."""
    l1 = relative_luminance(fg)
    l2 = relative_luminance(bg)
    lighter, darker = (l1, l2) if l1 >= l2 else (l2, l1)
    return (lighter + 0.05) / (darker + 0.05)


def passes_wcag(ratio: float, *, large: bool = False, level: str = "AA") -> bool:
    """Whether ``ratio`` meets the WCAG threshold for ``level``/text size.

    AA: 4.5 normal / 3.0 large. AAA: 7.0 normal / 4.5 large.
    """
    normalized = level.strip().upper()
    if normalized == "AA":
        threshold = 3.0 if large else 4.5
    elif normalized == "AAA":
        threshold = 4.5 if large else 7.0
    else:
        raise ValueError(f"unknown WCAG level: {level!r} (expected 'AA' or 'AAA')")
    return ratio >= threshold


def best_text_color(
    bg: str | tuple[int, int, int],
    *,
    candidates: tuple[str, ...] = ("#000000", "#ffffff"),
) -> str:
    """Return the candidate colour with the highest contrast against ``bg``.

    Ties are resolved by candidate order (first wins).
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")
    return max(candidates, key=lambda candidate: contrast_ratio(candidate, bg))
