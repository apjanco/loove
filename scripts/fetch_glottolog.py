#!/usr/bin/env python3
"""
Download Glottolog languoid data and build the language registry.

Run this once before ingesting models (after fetch_cldr.py).  It downloads
the Glottolog languoid CSV (~576 KB zip), parses ~8,000 language entries,
and cross-references them with the CLDR locale database.

Examples:
  python scripts/fetch_glottolog.py             # download + parse
  python scripts/fetch_glottolog.py --refresh   # re-download even if cached
  python scripts/fetch_glottolog.py --stats     # show breakdown by macroarea
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from pipeline.unicode.glottolog_loader import build_languoid_database, load_families


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and cache Glottolog language data."
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-download the CSV even if a cached copy exists.",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print a detailed breakdown after building the database.",
    )
    args = parser.parse_args()

    languoids = build_languoid_database(force_refresh=args.refresh)

    if args.stats:
        _print_stats(languoids)


def _print_stats(languoids: dict) -> None:
    families = {}
    try:
        families = load_families()
    except FileNotFoundError:
        pass

    # Macroarea breakdown
    by_area: dict[str, list] = defaultdict(list)
    for entry in languoids.values():
        area = entry.get("macroarea") or "(unknown)"
        by_area[area].append(entry)

    print("\n── Macroarea breakdown ──────────────────────────────────────")
    print(f"  {'Macroarea':<22} {'Total':>7}  {'With CLDR':>10}  {'Without CLDR':>13}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*10}  {'-'*13}")
    for area in sorted(by_area):
        entries   = by_area[area]
        with_cldr = sum(1 for e in entries if e["has_cldr"])
        print(f"  {area:<22}  {len(entries):>7,}  {with_cldr:>10,}  {len(entries)-with_cldr:>13,}")

    # Top families
    fam_counts: dict[str, int] = defaultdict(int)
    for entry in languoids.values():
        fid  = entry.get("family_id")
        name = entry.get("family_name") or fid or "(unknown)"
        fam_counts[name] += 1

    print("\n── Largest language families (top 20) ───────────────────────")
    for name, count in sorted(fam_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {name:<35}  {count:>5,} languages")

    total      = len(languoids)
    with_cldr  = sum(1 for e in languoids.values() if e["has_cldr"])
    print(f"\n  Total languages: {total:,}")
    print(f"  With CLDR data:  {with_cldr:,} ({with_cldr/total:.1%})")
    print(f"  Without CLDR:    {total-with_cldr:,} ({(total-with_cldr)/total:.1%})")
    print(
        "\nNote: languages without CLDR data will appear in the dashboard\n"
        "registry but cannot be scored for character-level coverage."
    )


if __name__ == "__main__":
    main()
