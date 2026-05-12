#!/usr/bin/env python3
"""
Pre-download UDHR corpus files for all languages in the CLDR database.

Run this after fetch_cldr.py to cache UDHR texts locally.  Fertility scoring
in ingest_model.py will then work offline (no per-language HTTP requests
during model ingestion).

Examples:
  python scripts/fetch_udhr.py
  python scripts/fetch_udhr.py --locales hi ar zh ja
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from pipeline.unicode.cldr_loader import load_language_database
from pipeline.analysis.fertility import prefetch_all, get_available_locales


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download UDHR corpus files.")
    parser.add_argument(
        "--locales", nargs="+", metavar="LOCALE",
        help="Specific locale IDs to fetch.  Default: all locales in languages.json.",
    )
    args = parser.parse_args()

    if args.locales:
        locales = args.locales
    else:
        try:
            db = load_language_database()
            locales = list(db.keys())
        except FileNotFoundError:
            print("CLDR language database not found.  Run: python scripts/fetch_cldr.py first.")
            sys.exit(1)

    available = set(get_available_locales())
    to_fetch = [loc for loc in locales if loc in available]
    skipped  = [loc for loc in locales if loc not in available]

    if skipped:
        print(f"No UDHR translation found for: {', '.join(skipped)}")

    print(f"Downloading UDHR texts for {len(to_fetch)} locales…")
    prefetch_all(to_fetch)
    print("Done.")


if __name__ == "__main__":
    main()
