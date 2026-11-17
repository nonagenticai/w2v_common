"""Tests for the pure WCAG colour-contrast math (w2v_common.media.contrast)."""

from __future__ import annotations

import sys
from typing import cast

import pytest

from w2v_common.media import (
    best_text_color,
    contrast_ratio,
    parse_color,
    passes_wcag,
    relative_luminance,
)
from w2v_common.media import contrast as contrast_mod

# --------------------------------------------------------------------------- #
# contrast_ratio
# --------------------------------------------------------------------------- #


def test_white_on_black_is_exactly_21() -> None:
    assert contrast_ratio("#ffffff", "#000000") == pytest.approx(21.0)


def test_black_on_white_is_symmetric_21() -> None:
    # max-ordering makes the ratio order-independent.
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0)


def test_same_color_ratio_is_one() -> None:
    assert contrast_ratio("#777777", "#777777") == pytest.approx(1.0)


def test_mid_gray_boundary_vs_white() -> None:
    # #767676 is the canonical grey that just meets AA (4.5) against white.
    ratio_white = contrast_ratio("#767676", "#ffffff")
    assert ratio_white == pytest.approx(4.54, abs=0.05)
    assert passes_wcag(ratio_white, level="AA") is True
    # ...but not the stricter AAA normal threshold (7.0).
    assert passes_wcag(ratio_white, level="AAA") is False


def test_low_contrast_gray_pair_fails_aa() -> None:
    # Two close mid-greys have too little separation for AA text.
    ratio = contrast_ratio("#777777", "#999999")
    assert ratio < 4.5
    assert passes_wcag(ratio, level="AA") is False


# --------------------------------------------------------------------------- #
# relative_luminance
# --------------------------------------------------------------------------- #


def test_relative_luminance_extremes() -> None:
    assert relative_luminance("#000000") == pytest.approx(0.0)
    assert relative_luminance("#ffffff") == pytest.approx(1.0)


def test_relative_luminance_green_weight_dominates() -> None:
    # Pure green carries the largest coefficient (0.7152).
    assert relative_luminance((0, 255, 0)) == pytest.approx(0.7152)
    assert relative_luminance((255, 0, 0)) == pytest.approx(0.2126)
    assert relative_luminance((0, 0, 255)) == pytest.approx(0.0722)


# --------------------------------------------------------------------------- #
# passes_wcag thresholds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("ratio", "large", "level", "expected"),
    [
        (4.5, False, "AA", True),
        (4.49, False, "AA", False),
        (3.0, True, "AA", True),
        (2.99, True, "AA", False),
        (7.0, False, "AAA", True),
        (6.99, False, "AAA", False),
        (4.5, True, "AAA", True),
        (4.49, True, "AAA", False),
        (21.0, False, "aaa", True),  # level is case-insensitive
    ],
)
def test_passes_wcag_thresholds(ratio: float, large: bool, level: str, expected: bool) -> None:
    assert passes_wcag(ratio, large=large, level=level) is expected


def test_passes_wcag_unknown_level_raises() -> None:
    with pytest.raises(ValueError):
        passes_wcag(21.0, level="AAAA")


# --------------------------------------------------------------------------- #
# best_text_color
# --------------------------------------------------------------------------- #


def test_best_text_color_picks_black_on_light_bg() -> None:
    assert best_text_color("#ffffff") == "#000000"


def test_best_text_color_picks_white_on_dark_bg() -> None:
    assert best_text_color("#000000") == "#ffffff"


def test_best_text_color_custom_candidates() -> None:
    assert best_text_color("#000080", candidates=("#ffff00", "#000000")) == "#ffff00"


def test_best_text_color_empty_candidates_raises() -> None:
    with pytest.raises(ValueError):
        best_text_color("#ffffff", candidates=())


# --------------------------------------------------------------------------- #
# parse_color
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("#fff", (255, 255, 255)),
        ("#000", (0, 0, 0)),
        ("#FF8800", (255, 136, 0)),
        ("#ff8800", (255, 136, 0)),
        ("rgb(255, 136, 0)", (255, 136, 0)),
        ("rgb(0,0,0)", (0, 0, 0)),
        ("  #abc  ", (170, 187, 204)),
        ((12, 34, 56), (12, 34, 56)),
    ],
)
def test_parse_color_forms(
    value: str | tuple[int, int, int], expected: tuple[int, int, int]
) -> None:
    assert parse_color(value) == expected


@pytest.mark.parametrize(
    "garbage",
    [
        "not-a-color",
        "#12",
        "#12345",
        "#gggggg",
        "rgb(300, 0, 0)",  # channel out of range
        "rgb(0, 0)",  # too few channels
    ],
)
def test_parse_color_rejects_string_garbage(garbage: str) -> None:
    with pytest.raises(ValueError):
        parse_color(garbage)


@pytest.mark.parametrize(
    "garbage",
    [
        (1, 2),  # wrong tuple arity
        (1, 2, 3, 4),  # wrong tuple arity
        (256, 0, 0),  # channel out of range
        (1.0, 2.0, 3.0),  # non-int channels
        123,  # unsupported type
    ],
)
def test_parse_color_rejects_wrong_typed_input(garbage: object) -> None:
    # These are deliberately ill-typed values; cast keeps `ty` honest while
    # still exercising the runtime ValueError guards.
    with pytest.raises(ValueError):
        parse_color(cast("str | tuple[int, int, int]", garbage))


# --------------------------------------------------------------------------- #
# purity guard — no heavy deps pulled in by the module
# --------------------------------------------------------------------------- #


def test_module_is_stdlib_only() -> None:
    # The contrast module must not bind numpy/httpx — it is pure stdlib math.
    # (Checking the module namespace is side-effect free; popping sys.modules
    # would perturb httpx, which the wider suite legitimately imports.)
    src_globals = vars(contrast_mod)
    assert "numpy" not in src_globals
    assert "np" not in src_globals
    assert "httpx" not in src_globals
    # Only stdlib modules should back the public functions.
    assert contrast_mod.re.__name__ == "re"
    # The module loaded and is registered under its stdlib-only namespace.
    assert "w2v_common.media.contrast" in sys.modules
