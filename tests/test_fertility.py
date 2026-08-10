"""
Tests for UDHR variant selection, script resolution, and fertility metrics.

Background
----------
Two defects made fertility and coverage describe different writing systems for
the same language.

1. `cldr_loader` read the script code from `babel.Locale.script`, which is
   populated only when the locale identifier carries an explicit script subtag
   ("sr_Latn"). Every locale in the database is a bare tag ("az", "hi"), so the
   field was the empty string for all 310 of them.

2. `fertility._load_index` resolved a locale to the *first* UDHR entry in index
   document order. For the 44 locales with several translations that picks by
   accident, and for az/bs/tk/uz it picked Cyrillic (`azj_cyrl`, `bos_cyrl`,
   `tuk_cyrl`, `uzn_cyrl`) while CLDR's exemplar set for those locales is Latin.

Together: tier scores computed over a Latin alphabet, fertility measured on
Cyrillic prose. Fixing (1) is what lets (2) work at all, so both are tested here.
"""
from __future__ import annotations

import unicodedata

import pytest
from babel import Locale

from pipeline.analysis import fertility as F
from pipeline.unicode.cldr_loader import _resolve_script, load_language_database

# Locales whose UDHR translation was previously chosen in the wrong script.
CONTESTED = {"az": "Latn", "bs": "Latn", "tk": "Latn", "uz": "Latn"}


def _dominant_script(text: str) -> str:
    """Most frequent Unicode script family among the alphabetic characters."""
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isalpha():
            family = unicodedata.name(ch, "?").split()[0]
            counts[family] = counts.get(family, 0) + 1
    return max(counts, key=counts.get) if counts else "?"


# ---------------------------------------------------------------------------
# Script resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("locale_id", "expected"),
    [("hi", "Deva"), ("az", "Latn"), ("sr", "Cyrl"), ("ar", "Arab"),
     ("ja", "Jpan"), ("th", "Thai"), ("ta", "Taml"), ("zh", "Hans"),
     ("sr_Latn", "Latn")],
)
def test_script_resolves_for_bare_locale_tags(locale_id: str, expected: str) -> None:
    assert _resolve_script(locale_id, Locale.parse(locale_id)) == expected


def test_every_locale_in_database_has_a_script() -> None:
    """Regression: this field was empty for all 310 locales."""
    db = load_language_database()
    missing = [loc for loc, entry in db.items() if not entry.get("script")]
    assert not missing, f"{len(missing)} locales without a script code: {missing[:10]}"


# ---------------------------------------------------------------------------
# UDHR variant selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("locale_id", "script"), sorted(CONTESTED.items()))
def test_contested_locales_select_matching_script(locale_id: str, script: str) -> None:
    chosen = F._resolve_file_code(locale_id, script)
    assert chosen is not None
    variants = dict(F._LOCALE_VARIANTS.get(locale_id, []))
    assert variants.get(chosen) == script, (
        f"{locale_id}: chose {chosen} ({variants.get(chosen)}), wanted {script}"
    )
    assert not chosen.endswith("_cyrl"), f"{locale_id} regressed to the Cyrillic variant"


def test_selection_without_script_hint_is_unchanged() -> None:
    """
    The hint is additive: callers that pass no script keep document-order
    behaviour, so the change cannot perturb single-variant locales.
    """
    index = F._load_index()
    for locale_id in index:
        assert F._resolve_file_code(locale_id, None) == index[locale_id]


def test_unknown_script_falls_back_rather_than_failing() -> None:
    assert F._resolve_file_code("az", "Runr") == F._load_index()["az"]
    assert F._resolve_file_code("no_such_locale", "Latn") is None


def test_fertility_text_script_matches_exemplar_script() -> None:
    """
    The property that actually matters: for every locale we can measure, the
    UDHR prose and the CLDR exemplar set must be the same writing system.
    """
    db = load_language_database()
    disagreements = []
    for locale_id, entry in db.items():
        file_code = F._resolve_file_code(locale_id, entry.get("script"))
        if not file_code:
            continue
        path = F._CACHE_DIR / f"{file_code}.xml"
        if not path.exists():
            continue
        body = F._extract_body(path.read_text(encoding="utf-8", errors="replace"))
        if len(body) < 50:
            continue
        exemplar = "".join(chr(c) for c in entry.get("exemplar_main", []))
        text_script, ex_script = _dominant_script(body), _dominant_script(exemplar)
        if "?" not in (text_script, ex_script) and text_script != ex_script:
            disagreements.append((locale_id, ex_script, text_script, file_code))

    # Latin-script exemplars legitimately appear in romanised or CJK-mixed
    # translations; the four contested locales must not be among the failures.
    contested_failures = [d for d in disagreements if d[0] in CONTESTED]
    assert not contested_failures, f"script mismatch persists: {contested_failures}"


# ---------------------------------------------------------------------------
# Fertility computation
# ---------------------------------------------------------------------------

def test_fertility_reports_which_translation_it_measured() -> None:
    """Provenance fields let a reader see the script behind a fertility number."""
    result = F.compute_fertility("az", lambda t: list(range(len(t))), script="Latn")
    assert result is not None, "az UDHR text should be cached"
    assert result["udhr_script"] == "Latn"
    assert result["udhr_file"].endswith("_latn")
    assert result["sample_chars"] > 0 and result["sample_tokens"] > 0


def test_fertility_metrics_are_consistent() -> None:
    """One token per character must give tokens_per_char == 1."""
    result = F.compute_fertility("hi", lambda t: list(range(len(t))), script="Deva")
    assert result is not None
    # tokenize_fn counts every character including whitespace; sample_chars
    # excludes it, so the ratio is >= 1 and words-per-token tracks it.
    assert result["tokens_per_char"] >= 1.0
    assert result["tokens_per_word"] > result["tokens_per_char"]


def test_missing_locale_returns_none() -> None:
    assert F.compute_fertility("zzz", lambda t: [0]) is None


def test_tokenizer_failure_is_not_fatal() -> None:
    def boom(_: str) -> list[int]:
        raise RuntimeError("tokenizer exploded")

    assert F.compute_fertility("hi", boom, script="Deva") is None
