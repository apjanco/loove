"""
Tests for the GlotScript-R integration and the has_cldr verification fix.

Two defects motivate this file:

1. ``has_cldr`` was assigned from ``langcodes.standardize_tag()``, which
   returns a well-formed tag for any structurally valid ISO 639-3 code
   regardless of whether CLDR holds data for it.  7,497 of 8,238 languoids
   were flagged as CLDR-backed when only 277 were, and the 7,220 false
   positives were silently dropped from coverage output entirely.

2. GlotScript gives script identity, not exemplar characters.  Nothing in
   the integration may use it to synthesise a coverage score; a language
   without exemplars must stay unscored and say so.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.unicode import glotscript_loader as gs
from pipeline.unicode.glotscript_loader import (
    NON_SCRIPTS,
    TRANSCRIPTION_SCRIPTS,
    _split_scripts,
    load_glotscript,
    orthographic_main,
    scripts_for,
)

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def _clear():
    gs.clear_cache()
    yield
    gs.clear_cache()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_split_scripts_handles_separators_and_spacing():
    assert _split_scripts("Latn") == ["Latn"]
    assert _split_scripts("Arab, Latn") == ["Arab", "Latn"]
    assert _split_scripts("Arab,Latn") == ["Arab", "Latn"]
    assert _split_scripts("") == []


def test_split_scripts_drops_non_script_sentinels():
    """Zxxx means 'unwritten' — it is not a writing system."""
    for sentinel in NON_SCRIPTS:
        assert _split_scripts(sentinel) == []
    assert _split_scripts("Zxxx, Latn") == ["Latn"]


def test_split_scripts_deduplicates():
    assert _split_scripts("Latn, Latn") == ["Latn"]


def test_missing_data_file_degrades_to_empty():
    """A missing resource must yield 'no information', not an exception."""
    assert load_glotscript(path=ROOT / "data" / "does-not-exist.tsv") == {}


# ---------------------------------------------------------------------------
# Loaded resource
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (ROOT / "data" / "glotscript" / "GlotScript.tsv").exists(),
    reason="GlotScript.tsv not vendored",
)
class TestLoadedResource:

    def test_known_language_scripts(self):
        """Spot-check languages whose orthography is not in dispute."""
        assert "Ethi" in orthographic_main("amh")   # Amharic
        assert "Cyrl" in orthographic_main("rus")   # Russian
        assert "Deva" in orthographic_main("hin")   # Hindi
        assert "Jpan" in orthographic_main("jpn")   # Japanese
        assert "Latn" in orthographic_main("eng")   # English

    def test_braille_is_not_treated_as_an_orthography(self):
        """
        The regression that caught the ordering bug: GlotScript's main column
        is an unordered set, and Braille sorts first for both English
        ("Brai, Shaw, Latn, …") and Hindi ("Brai, Deva, …").  Collapsing to
        element zero reported English as a Braille-script language.
        """
        for code in ("eng", "hin", "rus", "amh"):
            assert "Brai" not in orthographic_main(code), code
            assert "Brai" not in scripts_for(code), code
        # …but it is still retrievable when explicitly asked for.
        assert "Brai" in scripts_for("eng", include_transcription=True)

    def test_multi_script_languages_return_all(self):
        """Digraphic languages must not be collapsed to one script."""
        aze = orthographic_main("aze")
        assert {"Arab", "Cyrl", "Latn"} <= set(aze), aze

    def test_unknown_code_returns_empty(self):
        assert scripts_for("zzzzz") == []
        assert scripts_for(None) == []
        assert scripts_for("") == []
        assert orthographic_main("zzzzz") == []

    def test_aux_scripts_exclude_main(self):
        """A script listed as main must not be repeated in aux."""
        for entry in load_glotscript().values():
            assert not (set(entry["main"]) & set(entry["aux"])), entry

    def test_no_sentinels_survive_into_entries(self):
        for entry in load_glotscript().values():
            assert not (set(entry["main"]) & NON_SCRIPTS)
            assert not (set(entry["aux"]) & NON_SCRIPTS)

    def test_unwritten_flag_requires_no_alternative(self):
        """
        'unwritten' must mean the source said so and offered nothing else.
        An empty cell means unknown, which is a different claim.
        """
        for entry in load_glotscript().values():
            if entry["unwritten"]:
                assert not entry["main"] and not entry["aux"], entry

    def test_resource_is_substantial(self):
        """Guards against a truncated or partially-downloaded file."""
        table = load_glotscript()
        assert len(table) > 6000, f"only {len(table)} languages loaded"


# ---------------------------------------------------------------------------
# has_cldr must reflect the real CLDR database
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (ROOT / "data" / "glottolog" / "languoids.json").exists()
    or not (ROOT / "data" / "cldr" / "languages.json").exists(),
    reason="Glottolog or CLDR data not fetched",
)
class TestHasCldrIsVerified:

    @staticmethod
    def _load():
        languoids = json.loads(
            (ROOT / "data" / "glottolog" / "languoids.json").read_text(encoding="utf-8")
        )["languoids"]
        cldr = json.loads(
            (ROOT / "data" / "cldr" / "languages.json").read_text(encoding="utf-8")
        )
        return languoids, set(cldr.keys())

    def test_no_false_positives(self):
        """
        The regression that motivated this file: every languoid claiming CLDR
        data must resolve to a locale the CLDR database actually contains.
        """
        languoids, cldr_ids = self._load()
        bad = [
            v["glottocode"] for v in languoids.values()
            if v["has_cldr"] and v["cldr_locale"] not in cldr_ids
        ]
        assert not bad, f"{len(bad)} languoids claim CLDR data they do not have"

    def test_has_cldr_count_is_bounded_by_cldr_size(self):
        """
        Cannot have more CLDR-backed languages than CLDR has locales.  This is
        the cheap invariant that would have caught the original bug: 7,497
        flagged against a 310-locale database is impossible on its face.
        """
        languoids, cldr_ids = self._load()
        n = sum(1 for v in languoids.values() if v["has_cldr"])
        assert n <= len(cldr_ids), f"{n} flagged vs {len(cldr_ids)} CLDR locales"

    def test_cldr_locale_only_set_when_has_cldr(self):
        """A locale ID that does not resolve must not be exposed at all."""
        languoids, _ = self._load()
        for v in languoids.values():
            if not v["has_cldr"]:
                assert v["cldr_locale"] is None, v["glottocode"]


# ---------------------------------------------------------------------------
# The integration must not manufacture coverage
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not list((ROOT / "data" / "coverage").glob("*.json")),
    reason="no coverage output to check",
)
class TestStubsStayUnscored:

    @staticmethod
    def _sample():
        f = sorted((ROOT / "data" / "coverage").glob("*.json"))[0]
        return json.loads(f.read_text(encoding="utf-8"))

    def test_script_never_implies_a_score(self):
        """
        A stub may carry a script from GlotScript, but must never carry a
        coverage score derived from it.  Script identity is not character data.
        """
        for entry in self._sample()["languages"].values():
            if entry.get("scored") is False:
                assert not entry.get("main"), "unscored language has coverage data"
                assert entry.get("weighted_score") is None

    def test_widely_covered_scripts_are_reported_represented(self):
        """
        Sanity floor on the reachability logic.  Any tokenizer that scores
        Latin-script languages at all must report Latin-script stubs as
        represented.  An earlier version inverted the tier ladder — tier 0 is
        the BEST tier, not the worst — and reported Greek and Armenian as
        unrepresented for a model that covers both perfectly.
        """
        langs = self._sample()["languages"]
        latin_stubs = [
            e for e in langs.values()
            if e.get("scored") is False and e.get("scripts") == ["Latn"]
        ]
        if not latin_stubs:
            pytest.skip("no Latin-only stubs in this sample")
        assert any(e["script_supported"] for e in latin_stubs)

    def test_perfectly_scored_script_is_never_unrepresented(self):
        """
        If a scored language in some script has a perfect weighted score, no
        stub in that script may be marked unrepresented.  This is the exact
        contradiction the tier inversion produced.
        """
        langs = self._sample()["languages"]
        perfect = {
            e["script"] for e in langs.values()
            if e.get("script") and (e.get("main") or {}).get("weighted_score") == 1.0
        }
        for e in langs.values():
            if e.get("scored") is False and e.get("script_supported") is False:
                assert not (set(e.get("scripts") or []) & perfect), (
                    f"{e['name']} marked unreachable in a fully-covered script"
                )

    def test_script_supported_is_three_valued(self):
        """None means 'cannot tell' and must not collapse to False."""
        vals = {
            e.get("script_supported")
            for e in self._sample()["languages"].values()
            if e.get("scored") is False
        }
        assert vals <= {True, False, None}
