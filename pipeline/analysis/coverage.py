"""
Coverage aggregator: combine tier classification and fertility into a
per-language, per-model coverage result ready for JSON storage and the API.

Output schema (data/coverage/{safe_model_id}.json):
{
  "model_id":         "gpt-4",
  "source":           "tiktoken",
  "vocab_size":       100256,
  "has_byte_fallback": true,
  "computed_at":      "2026-05-12T10:00:00Z",
  "languages": {
    "hi": {
      "name":        "Hindi",
      "script":      "Deva",
      // Glottolog enrichment (when available)
      "glottocode":  "hin1269",
      "iso639_3":    "hin",
      "macroarea":   "Eurasia",
      "family_id":   "indo1319",
      "family_name": "Indo-European",
      "latitude":    25.0,
      "longitude":   77.0,
      "main": {
        "total":          68,
        "weighted_score": 0.8823,
        "tier0_count":    55,
        "tier1_count":    8,
        "tier2_count":    5,
        "tier3_count":    0,
        "tier1": [2366, 2367],   // code points at each degraded tier
        "tier2": [2385],
        "tier3": []
      },
      "auxiliary": { ... },      // optional, only if exemplar data exists
      "fertility": {             // optional, only if --fertility flag used
        "tokens_per_char": 1.42,
        "tokens_per_word": 5.31,
        "sample_chars":    2847,
        "sample_tokens":   4041
      }
    },
    // Languages known from Glottolog but with no CLDR exemplar data:
    "aaa1241": {
      "name":       "Ghotuo",
      "glottocode": "aaa1241",
      "iso639_3":   "aaa",
      "macroarea":  "Africa",
      "family_id":  "atla1278",
      "family_name": "Atlantic-Congo",
      "has_cldr":   false        // no character-level analysis possible
    },
    ...
  }
}
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pipeline.tokenizers.base import ModelVocabData
from pipeline.analysis.tier_classifier import classify
from pipeline.analysis.fertility import compute_fertility

# Glottolog is optional — if the database hasn't been fetched yet we still
# produce a valid (unenriched) coverage result.
_GLOTTOLOG_CLDR_INDEX: dict[str, list[dict]] | None = None


def _get_glottolog_index() -> dict[str, list[dict]]:
    """
    Lazy-load a CLDR-locale → list[glottolog_entry] index.
    Returns an empty dict if Glottolog data hasn't been fetched.
    """
    global _GLOTTOLOG_CLDR_INDEX
    if _GLOTTOLOG_CLDR_INDEX is not None:
        return _GLOTTOLOG_CLDR_INDEX

    try:
        from pipeline.unicode.glottolog_loader import (
            load_languoid_database,
            cldr_to_glottolog_index,
        )
        languoids = load_languoid_database()
        raw_index = cldr_to_glottolog_index(languoids)
        # Expand: locale → list of full entry dicts
        _GLOTTOLOG_CLDR_INDEX = {
            locale: [languoids[gc] for gc in glottocodes if gc in languoids]
            for locale, glottocodes in raw_index.items()
        }
    except FileNotFoundError:
        _GLOTTOLOG_CLDR_INDEX = {}

    return _GLOTTOLOG_CLDR_INDEX

_ROOT = Path(__file__).parents[2]
_DATA_DIR = _ROOT / "data"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_coverage(
    vocab: ModelVocabData,
    language_db: dict[str, dict],
    tokenize_fn: Callable[[str], list[int]] | None = None,
    include_glottolog_stubs: bool = True,
) -> dict:
    """
    Compute tier-based coverage for `vocab` against every language in
    `language_db`, enriched with Glottolog metadata where available.

    Args:
        vocab:                   Extracted vocabulary data for a model.
        language_db:             CLDR language database as returned by cldr_loader.
        tokenize_fn:             Optional callable for fertility scoring.
                                 Signature: (text: str) -> list[int]
                                 If None, fertility metrics are omitted.
        include_glottolog_stubs: If True and the Glottolog database has been
                                 fetched, append stub entries for Glottolog
                                 languages that have no CLDR exemplar data.
                                 These entries carry metadata only (no tier
                                 analysis) and have has_cldr=False.

    Returns:
        A dict suitable for JSON serialisation (see module docstring).
    """
    glottolog_index = _get_glottolog_index()
    languages: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Languages with CLDR exemplar data → full tier analysis
    # ------------------------------------------------------------------
    for locale_id, lang_data in language_db.items():
        exemplar_main = set(lang_data.get("exemplar_main", []))
        exemplar_aux  = set(lang_data.get("exemplar_auxiliary", []))

        if not exemplar_main:
            continue

        main_result = classify(exemplar_main, vocab)
        entry: dict = {
            "name":     lang_data.get("name", locale_id),
            "script":   lang_data.get("script", ""),
            "has_cldr": True,
            "main":     main_result.to_dict(),
        }

        if exemplar_aux:
            entry["auxiliary"] = classify(exemplar_aux, vocab).to_dict()

        if tokenize_fn is not None:
            fertility = compute_fertility(locale_id, tokenize_fn)
            if fertility is not None:
                entry["fertility"] = fertility

        # Enrich with Glottolog metadata (use first matching entry)
        glottolog_entries = glottolog_index.get(locale_id, [])
        if glottolog_entries:
            gl = glottolog_entries[0]
            entry.update({
                "glottocode":  gl["glottocode"],
                "iso639_3":    gl["iso639_3"],
                "macroarea":   gl["macroarea"],
                "family_id":   gl["family_id"],
                "family_name": gl["family_name"],
                "latitude":    gl["latitude"],
                "longitude":   gl["longitude"],
            })

        languages[locale_id] = entry

    # ------------------------------------------------------------------
    # Glottolog-only languages → metadata stubs (no tier analysis)
    # ------------------------------------------------------------------
    if include_glottolog_stubs and glottolog_index:
        try:
            from pipeline.unicode.glottolog_loader import load_languoid_database
            all_languoids = load_languoid_database()
        except FileNotFoundError:
            all_languoids = {}

        cldr_locales_covered = set(language_db.keys())

        for glottocode, gl in all_languoids.items():
            cldr_locale = gl.get("cldr_locale")
            # Only add as stub if this language isn't already in the results
            if cldr_locale and cldr_locale in cldr_locales_covered:
                continue
            if gl.get("has_cldr"):
                continue  # Should have been covered above; skip duplicates.

            # Use glottocode as the key for stub entries
            languages[glottocode] = {
                "name":        gl["name"],
                "glottocode":  gl["glottocode"],
                "iso639_3":    gl["iso639_3"],
                "cldr_locale": cldr_locale,
                "macroarea":   gl["macroarea"],
                "family_id":   gl["family_id"],
                "family_name": gl["family_name"],
                "latitude":    gl["latitude"],
                "longitude":   gl["longitude"],
                "has_cldr":    False,
            }

    return {
        "model_id":          vocab.model_id,
        "source":            vocab.source,
        "vocab_size":        vocab.vocab_size,
        "has_byte_fallback": vocab.has_byte_fallback,
        "computed_at":       datetime.now(timezone.utc).isoformat(),
        "languages":         languages,
    }


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _safe_model_id(model_id: str) -> str:
    """Convert a model ID to a filesystem-safe filename stem."""
    return model_id.replace("/", "__").replace(":", "_").replace(" ", "_")


def save_coverage(result: dict, out_dir: Path | None = None) -> Path:
    """Serialise a coverage result to disk. Returns the written path."""
    if out_dir is None:
        out_dir = _DATA_DIR / "coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_model_id(result['model_id'])}.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def save_vocab(vocab: ModelVocabData, out_dir: Path | None = None) -> Path:
    """Serialise extracted vocabulary data to disk. Returns the written path."""
    if out_dir is None:
        out_dir = _DATA_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_model_id(vocab.model_id)}.json"
    out_path.write_text(
        json.dumps(vocab.to_json_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


def load_coverage(model_id: str, data_dir: Path | None = None) -> dict | None:
    """Load a previously computed coverage result, or None if not found."""
    if data_dir is None:
        data_dir = _DATA_DIR / "coverage"
    path = data_dir / f"{_safe_model_id(model_id)}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
