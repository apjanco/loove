"""
Tests for script typology and denominator normalization.

Background
----------
The weighted score divides by the size of the CLDR exemplar set, and that size
is a property of how CLDR enumerated the script rather than of the tokenizer.
Across the 310 languages in the database, log10(exemplar set size) correlates
with mean weighted score at r = -0.60, so roughly a third of the variance in the
headline metric comes from the denominator alone.

Korean is the extreme: its exemplar set is all 11,172 precomposed Hangul
syllable blocks, giving a mean score of 0.15 across ingested models — lowest of
any major language — while the 67 conjoining Jamo that generate every one of
those blocks tell you whether the model can actually write Korean.
"""
from __future__ import annotations

import json

import pytest

from pipeline.analysis.script_types import (
    SCRIPT_TYPES,
    comparability,
    denominator_kind,
    is_comparable,
    normalize_exemplars,
    script_type,
)
from pipeline.analysis.tier_classifier import classify
from pipeline.tokenizers.base import ModelVocabData
from pipeline.unicode.cldr_loader import load_language_database

HANGUL_FIRST = 0xAC00   # 가
HANGUL_LAST = 0xD7A3    # 힣


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("script", "expected_type", "expected_kind"),
    [("Latn", "alphabet", "closed_inventory"),
     ("Arab", "abjad", "closed_inventory"),
     ("Deva", "abugida", "closed_inventory"),
     ("Ethi", "syllabary", "syllabic_derived"),
     ("Kore", "featural_syllabary", "combinatorial"),
     ("Hans", "logographic", "open_set"),
     ("Jpan", "mixed_logographic", "open_set")],
)
def test_script_classification(script: str, expected_type: str, expected_kind: str) -> None:
    assert script_type(script) == expected_type
    assert denominator_kind(script) == expected_kind


def test_every_script_in_the_database_is_classified() -> None:
    """An unclassified script would silently default to 'comparable'."""
    db = load_language_database()
    scripts = {entry["script"] for entry in db.values() if entry.get("script")}
    unclassified = sorted(s for s in scripts if s not in SCRIPT_TYPES)
    assert not unclassified, f"unclassified scripts: {unclassified}"


def test_comparability_flags_the_problem_scripts() -> None:
    assert is_comparable("Latn") and comparability("Latn") == "direct"
    for script in ("Kore", "Hans", "Hant", "Jpan", "Ethi"):
        assert not is_comparable(script), f"{script} must not be directly comparable"
        assert comparability(script) == "within_script_only"


def test_unknown_script_does_not_crash() -> None:
    assert script_type("Zzzz") == "unknown"
    assert denominator_kind("Zzzz") == "closed_inventory"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_hangul_normalizes_to_jamo() -> None:
    db = load_language_database()
    exemplars = set(db["ko"]["exemplar_main"])
    assert len(exemplars) > 10_000, "ko exemplar set should be the full syllable block"

    norm = normalize_exemplars("Kore", exemplars)
    assert norm is not None
    assert norm.kind == "hangul_jamo"
    # Three orders of magnitude smaller, and every element is a Jamo.
    assert len(norm.codepoints) < 100
    assert all(0x1100 <= cp <= 0x11FF for cp in norm.codepoints), (
        "generator set should be conjoining Jamo"
    )
    assert not any(HANGUL_FIRST <= cp <= HANGUL_LAST for cp in norm.codepoints), (
        "precomposed syllables must not survive normalization"
    )


def test_non_combinatorial_scripts_are_not_normalized() -> None:
    """
    Han does not decompose to a closed component set, and Ethiopic has no uniform
    canonical decomposition, so inventing a denominator for them would be a
    fudge rather than a normalization.
    """
    db = load_language_database()
    for locale_id, script in [("zh", "Hans"), ("ja", "Jpan"), ("am", "Ethi"), ("hi", "Deva")]:
        exemplars = set(db[locale_id]["exemplar_main"])
        assert normalize_exemplars(script, exemplars) is None, f"{script} must not normalize"


def test_normalization_is_not_merely_a_score_boost() -> None:
    """
    The normalized score must be able to fall as well as rise. A model that
    memorised syllable blocks but lacks Jamo should score *lower* against the
    generator set — otherwise the normalization is just inflating numbers.
    """
    db = load_language_database()
    exemplars = set(db["ko"]["exemplar_main"])
    norm = normalize_exemplars("Kore", exemplars)
    assert norm is not None

    blocks_only = ModelVocabData(
        model_id="synthetic-blocks", source="test", vocab_size=0,
        has_byte_fallback=False,
        codepoints_single=set(exemplars), codepoints_any=set(exemplars),
    )
    raw = classify(exemplars, blocks_only).weighted_score
    normalized = classify(norm.codepoints, blocks_only).weighted_score
    assert raw == pytest.approx(1.0)
    assert normalized == pytest.approx(0.0), (
        "a blocks-only vocabulary cannot compose Korean and must score 0 on Jamo"
    )

    jamo_only = ModelVocabData(
        model_id="synthetic-jamo", source="test", vocab_size=0,
        has_byte_fallback=False,
        codepoints_single=set(norm.codepoints), codepoints_any=set(norm.codepoints),
    )
    assert classify(exemplars, jamo_only).weighted_score == pytest.approx(0.0)
    assert classify(norm.codepoints, jamo_only).weighted_score == pytest.approx(1.0)


def test_coverage_output_carries_comparability_fields() -> None:
    """Consumers of the JSON must be able to tell which scores are comparable."""
    from pipeline.analysis.coverage import compute_coverage

    db = load_language_database()
    subset = {k: db[k] for k in ("en", "hi", "ko", "zh") if k in db}
    vocab = ModelVocabData(
        model_id="test", source="test", vocab_size=3,
        has_byte_fallback=True, codepoints_single={ord("a")}, codepoints_any={ord("a")},
    )
    result = compute_coverage(vocab, subset)
    langs = result["languages"]

    assert langs["en"]["comparability"] == "direct"
    assert langs["ko"]["comparability"] == "within_script_only"
    assert langs["zh"]["denominator_kind"] == "open_set"
    # Only Korean gets a second score.
    assert "main_normalized" in langs["ko"]
    assert langs["ko"]["main_normalized"]["normalization"] == "hangul_jamo"
    assert "main_normalized" not in langs["en"]
    assert "main_normalized" not in langs["zh"]
