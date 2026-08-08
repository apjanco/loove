#!/usr/bin/env python3
"""
Relabel probe rows whose verdict came from a transport failure.

Why
---
`probe_locale` used to set ``echo_fidelity = 0.0`` and ``translation_refused =
True`` whenever an API call raised, then graded that row through the ordinary
verdict ladder. A timeout, a 403, or a rate-limit therefore produced the same
"Poor" as a model that genuinely mangled the script.

Every row in ``data/probes/`` was written under that behaviour, and all 108 of
them failed with ``HTTP Error 403: Forbidden`` — no model response text was ever
received. The dashboard nonetheless rendered them as 36 "Poor" verdicts per
frontier model, which reads as a finding about those models' language support
when it is a finding about an expired API key.

This script rewrites such rows to ``verdict = "Error"`` with null fidelity, and
records a file-level note. It does not delete anything: the original error
strings are preserved so a reader can see why each row is inconclusive.

Usage
-----
    python scripts/migrate_probe_verdicts.py            # report only
    python scripts/migrate_probe_verdicts.py --apply    # rewrite in place
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROBES_DIR = Path(__file__).resolve().parent.parent / "data" / "probes"


def migrate_row(row: dict) -> bool:
    """Relabel one row if its verdict rests on a failed call. True if changed."""
    if not row.get("echo_error"):
        return False
    row["verdict"] = "Error"
    row["echo_fidelity"] = None
    row["byte_artifacts"] = None
    # The old code set this True on a transport error; it was never a refusal.
    if row.get("translation_error"):
        row["translation_refused"] = None
    row["transport_failed"] = True
    row["n_transport_errors"] = sum(
        1 for k in ("echo_error", "script_error", "translation_error") if row.get(k)
    )
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="rewrite files in place")
    args = ap.parse_args()

    paths = sorted(PROBES_DIR.glob("*.json"))
    if not paths:
        print(f"No probe files in {PROBES_DIR}")
        return

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        results = data.get("results", {})
        changed = sum(migrate_row(row) for row in results.values())
        graded = sum(1 for r in results.values() if r.get("verdict") != "Error")

        print(f"{path.name}: {changed}/{len(results)} rows relabelled inconclusive, "
              f"{graded} still carry evidence")

        if changed:
            data["transport_failures"] = changed
            data["graded_locales"] = graded
            if graded == 0:
                data["warning"] = (
                    "Every call in this run failed at the transport layer, so this "
                    "file contains no evidence about the model's language support. "
                    "Re-run scripts/probe_model.py with a working API key."
                )
        if args.apply:
            shutil.copy2(path, path.with_suffix(".json.pre-migration"))
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print("\nApplied." if args.apply else "\nDry run — pass --apply to rewrite.")


if __name__ == "__main__":
    main()
