"""
Tests for probe verdict grading and transport-failure separation.

Background
----------
`probe_locale` set ``echo_fidelity = 0.0`` and ``translation_refused = True``
whenever an API call raised, then graded the row through the normal ladder. A
timeout, a 403, or a rate-limit therefore produced "Poor" — the same verdict as a
model that genuinely mangled the script.

All 108 rows in ``data/probes/`` were written under that behaviour and every one
failed with ``HTTP Error 403: Forbidden``. No model response text was ever
received, yet the dashboard rendered them as 36 "Poor" verdicts per frontier
model, which reads as a finding about those models rather than about an expired
key. ``scripts/migrate_probe_verdicts.py`` relabelled them.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROBES_DIR = ROOT / "data" / "probes"


def _load_probe_module():
    """scripts/ is not a package, so load probe_model.py by path."""
    spec = importlib.util.spec_from_file_location(
        "probe_model", ROOT / "scripts" / "probe_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe_model = _load_probe_module()
grade_verdict = probe_model.grade_verdict


# ---------------------------------------------------------------------------
# Verdict grading
# ---------------------------------------------------------------------------

def test_transport_failure_is_inconclusive_not_poor() -> None:
    """The regression this module exists for."""
    assert grade_verdict(
        fidelity=None, script_recognized=None,
        translation_refused=None, echo_failed=True,
    ) == "Error"


def test_echo_failure_cannot_be_graded_even_with_other_signals() -> None:
    """
    Without a round-trip there is no fidelity measurement, so the row is
    inconclusive regardless of what the other two tests returned.
    """
    assert grade_verdict(
        fidelity=None, script_recognized=True,
        translation_refused=False, echo_failed=True,
    ) == "Error"


def test_genuine_failure_is_still_poor() -> None:
    """A model that answered and got it wrong must not be excused as an error."""
    assert grade_verdict(
        fidelity=0.10, script_recognized=False,
        translation_refused=True, echo_failed=False,
    ) == "Poor"


@pytest.mark.parametrize(
    ("fidelity", "script_ok", "refused", "expected"),
    [(1.00, True, False, "Strong"),
     (0.96, True, False, "Strong"),
     (0.96, True, True, "Partial"),    # refusal downgrades a clean round-trip
     (0.80, False, False, "Partial"),
     (0.30, True, False, "Partial"),   # script recognised despite poor echo
     (0.30, False, False, "Poor"),
     (0.74, False, True, "Poor")],
)
def test_verdict_ladder(fidelity, script_ok, refused, expected) -> None:
    assert grade_verdict(
        fidelity=fidelity, script_recognized=script_ok,
        translation_refused=refused, echo_failed=False,
    ) == expected


def test_unknown_translation_outcome_does_not_award_strong() -> None:
    """
    A failed translation call leaves ``translation_refused = None``. That is
    absence of evidence, so the row cannot claim the top grade.
    """
    assert grade_verdict(
        fidelity=1.0, script_recognized=True,
        translation_refused=None, echo_failed=False,
    ) == "Partial"


# ---------------------------------------------------------------------------
# Stored data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", sorted(PROBES_DIR.glob("*.json")), ids=lambda p: p.name)
def test_stored_probe_rows_do_not_grade_failed_calls(path: Path) -> None:
    """No stored row may carry a graded verdict when its echo call failed."""
    data = json.loads(path.read_text(encoding="utf-8"))
    mislabelled = [
        locale for locale, row in data.get("results", {}).items()
        if row.get("echo_error") and row.get("verdict") in {"Strong", "Partial", "Poor"}
    ]
    assert not mislabelled, (
        f"{path.name}: {len(mislabelled)} rows grade a failed call: {mislabelled[:5]}"
    )


@pytest.mark.parametrize("path", sorted(PROBES_DIR.glob("*.json")), ids=lambda p: p.name)
def test_probe_files_with_no_usable_rows_say_so(path: Path) -> None:
    """A file where every call failed must be self-describing, not silently empty."""
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", {})
    graded = [r for r in results.values() if r.get("verdict") != "Error"]
    if results and not graded:
        assert data.get("warning"), (
            f"{path.name} has no usable rows but carries no warning field"
        )


def test_dashboard_reports_zero_graded_for_the_failed_runs() -> None:
    """End-to-end: the summary must not present an all-errors run as a result."""
    import app

    stems = app.list_probe_files()
    assert stems, "no probe files to render"
    summary = app.make_probe_summary_html(stems[0])
    assert "No usable results" in summary
    assert "0 of 36 languages graded" in summary

    table = app.make_probe_table(stems[0])
    assert set(table["Verdict"]) == {"Error"}
    assert (table["Fidelity"] == "").all(), "failed rows must not display a fidelity"
