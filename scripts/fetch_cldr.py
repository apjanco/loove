#!/usr/bin/env python3
"""
Download and process CLDR locale data into data/cldr/languages.json.

Run this once before ingesting any models.  Subsequent runs use the cached
CLDR XML files in data/cldr/raw/ and are fast.

Examples:
  python scripts/fetch_cldr.py                       # all locales
  python scripts/fetch_cldr.py --locales hi ar zh    # specific locales only
  python scripts/fetch_cldr.py --refresh             # re-download all XML
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable
sys.path.insert(0, str(Path(__file__).parents[1]))

from pipeline.unicode.cldr_loader import build_language_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache CLDR exemplar data.")
    parser.add_argument(
        "--locales", nargs="+", metavar="LOCALE",
        help="Specific locale IDs to fetch (e.g. hi ar zh).  Default: all.",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-download CLDR XML even when cached copies exist.",
    )
    args = parser.parse_args()

    print("Fetching CLDR locale data…")
    db = build_language_database(locales=args.locales, force_refresh=args.refresh)

    # Summary
    scripts: dict[str, int] = {}
    for lang in db.values():
        s = lang.get("script") or "(unknown)"
        scripts[s] = scripts.get(s, 0) + 1

    print(f"\n{len(db)} languages processed.")
    print("\nTop scripts by locale count:")
    for script, count in sorted(scripts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {script:<20} {count:>4}")


if __name__ == "__main__":
    main()
