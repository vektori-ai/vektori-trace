"""The report has to be able to say nothing was found."""

from __future__ import annotations

import json
from pathlib import Path

from vektori_trace.diagnose import Capability, DeficitScore
from vektori_trace.report import build_report, write_report


def _score(gap: float | None) -> DeficitScore:
    return DeficitScore(
        capability=Capability(id="c", name="Some capability", description="…"),
        baseline_rate=0.5,
        incident_rate=0.5,
        gap=gap,
        prevalence=0.4,
        priority=0.0,
        n_relevant_wins=3,
        n_relevant_losses=2,
    )


def test_no_deficit_report_is_written_without_a_task(tmp_path: Path) -> None:
    rejected = _score(-1.0)
    report = build_report(
        None, [rejected], None, None, {"min_gap": 0.2, "min_support": 3}
    )
    md_path = write_report(report, tmp_path)

    assert json.loads((tmp_path / "diagnosis.json").read_text())["chosen_deficit"] is None
    md = md_path.read_text()
    assert "No deficit found" in md
    # The rejected leader is still listed, but plainly marked as inspection-only.
    assert "Some capability" in md
    assert "rejected" in md


def test_report_prints_n_beside_every_rate(tmp_path: Path) -> None:
    """A capability measured on 3 traces must not read like one measured on 40."""
    chosen = _score(0.6)
    report = build_report(chosen, [chosen], tmp_path / "task", None, {})
    md = write_report(report, tmp_path).read_text()

    assert "N=3" in md
    assert "N=2" in md
    assert "N=3w/2l" in md
