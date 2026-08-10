#!/usr/bin/env python3
"""
Fetch language → writing-system data from Wikidata to fill GlotScript gaps.

By default this targets the gap that motivated it: historical languages with no
script in GlotScript.  Widen with --scope to fetch every language lacking a
GlotScript script, or every language Wikidata knows.

The cache (data/wikidata/language_scripts.json) accumulates across runs, so you
can start narrow and expand later without refetching.

Examples:
  python scripts/fetch_wikidata_scripts.py                 # historical, script-less
  python scripts/fetch_wikidata_scripts.py --scope missing # any script-less lang
  python scripts/fetch_wikidata_scripts.py --scope all     # everything
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from pipeline.unicode.glottolog_loader import load_languoid_database
from pipeline.unicode.glotscript_loader import orthographic_main
from pipeline.unicode.wikidata_scripts import fetch_scripts, load_wikidata_scripts
from pipeline.unicode.historic_scripts import is_encoded, _UNENCODED_HISTORIC


def _target_isos(scope: str) -> list[str]:
    languoids = load_languoid_database()
    isos: list[str] = []
    for e in languoids.values():
        iso = e.get("iso639_3")
        if not iso:
            continue
        if scope == "all":
            isos.append(iso)
            continue
        # "missing" scopes: only languages GlotScript gives no script for.
        if orthographic_main(iso):
            continue
        if scope == "missing":
            isos.append(iso)
        elif scope == "historical-missing" and e.get("is_historical"):
            isos.append(iso)
    return sorted(set(isos))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scope",
        choices=["historical-missing", "missing", "all"],
        default="historical-missing",
        help="Which languages to query (default: historical-missing = the 122).",
    )
    args = parser.parse_args()

    isos = None if args.scope == "all" else _target_isos(args.scope)
    if isos is not None:
        print(f"[wikidata] Querying {len(isos)} language(s) for scope '{args.scope}'…")
    else:
        print("[wikidata] Querying ALL languages with an ISO 639-3 + script…")

    fetched = fetch_scripts(isos)
    print(f"[wikidata] Got scripts for {len(fetched)} language(s) this run.")

    # Report how much of the requested set was resolved, and how useful it is.
    if isos is not None:
        resolved = [i for i in isos if i in fetched]
        print(f"[wikidata] Resolved {len(resolved)}/{len(isos)} requested.")
        script_counts: Counter[str] = Counter()
        newly_scoreable = 0
        unencoded = 0
        for i in resolved:
            for s in fetched[i]:
                script_counts[s] += 1
            if any(is_encoded(s) for s in fetched[i]):
                newly_scoreable += 1
            elif all(s in _UNENCODED_HISTORIC for s in fetched[i]):
                unencoded += 1
        print(f"[wikidata] {newly_scoreable} have a Unicode-encoded historic "
              f"script → newly scoreable via block exemplars.")
        print(f"[wikidata] {unencoded} resolve only to unencoded scripts.")
        print("[wikidata] Top scripts found:")
        for s, n in script_counts.most_common(15):
            enc = "encoded" if is_encoded(s) else ""
            print(f"    {s:<6} {n:>4}  {enc}")

    total = len(load_wikidata_scripts())
    print(f"[wikidata] Cache now holds scripts for {total} language(s).")


if __name__ == "__main__":
    main()
