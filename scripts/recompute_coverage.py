#!/usr/bin/env python3
"""
Recompute coverage for every ingested model from cached vocab data.

Unlike ingest_model.py this needs no tokenizer download: it reconstructs each
model's derived code-point sets from data/models/*.json and re-runs the coverage
analysis.  Use it whenever the reference data changes (new CLDR release, new
historic-script exemplars, Glottolog refresh) to refresh all models at once.

Fertility is not stored in the vocab data, so any fertility block already
present in data/coverage/<id>.json is carried over unchanged.

Example:
  python scripts/recompute_coverage.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from pipeline.tokenizers.base import ModelVocabData
from pipeline.unicode.cldr_loader import build_language_database
from pipeline.unicode.historic_scripts import build_historic_language_db
from pipeline.analysis.coverage import compute_coverage, save_coverage, load_coverage

ROOT = Path(__file__).parents[1]
MODELS_DIR = ROOT / "data" / "models"


def _preserve_fertility(model_id: str, result: dict) -> int:
    """Copy fertility blocks from the existing coverage file into `result`."""
    old = load_coverage(model_id)
    if not old:
        return 0
    carried = 0
    langs = result["languages"]
    for key, entry in old.get("languages", {}).items():
        fert = entry.get("fertility")
        if fert is not None and key in langs:
            langs[key]["fertility"] = fert
            carried += 1
    return carried


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models", nargs="+", metavar="FILE",
        help="Specific model vocab JSON files to recompute (default: all).",
    )
    args = parser.parse_args()

    # Reference data is identical for every model — build it once.
    print("[*] Loading CLDR + historic-script language database…")
    language_db = build_language_database()
    historic_db = build_historic_language_db()
    for key, entry in historic_db.items():
        language_db.setdefault(key, entry)
    print(f"    {len(language_db)} languages "
          f"({len(historic_db)} historic-script synthetic exemplars)")

    files = (
        [Path(f) for f in args.models] if args.models
        else sorted(MODELS_DIR.glob("*.json"))
    )
    if not files:
        print(f"[!] No model vocab files found in {MODELS_DIR}")
        sys.exit(1)

    print(f"[*] Recomputing coverage for {len(files)} model(s)…\n")
    for path in files:
        try:
            vocab = ModelVocabData.from_json_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            print(f"    [warn] {path.name}: could not load vocab ({exc})")
            continue

        result = compute_coverage(vocab, language_db, tokenize_fn=None)
        carried = _preserve_fertility(vocab.model_id, result)

        langs = result["languages"]
        historic = sum(1 for v in langs.values()
                       if v.get("exemplar_source") == "unicode_block")
        no_script = sum(1 for v in langs.values()
                        if v.get("script_encoded") is False)
        save_coverage(result)
        print(f"    {vocab.model_id:<45} "
              f"+{historic} historic, {no_script} unencoded, "
              f"{carried} fertility carried")

    print("\n[✓] Done.")


if __name__ == "__main__":
    main()
