"""The report has to be able to say nothing was found."""

from __future__ import annotations

import json
from pathlib import Path

from vektori_trace.diagnose import (
    Capability,
    DeficitScore,
    McNemarResult,
    ReplayDiagnosis,
)
from vektori_trace.report import build_replay_report, build_report, write_report


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


def _diagnosis(trainable: bool, *, discordant: tuple[int, int] = (6, 0)) -> ReplayDiagnosis:
    chosen = _score(0.6)
    within = DeficitScore(
        capability=chosen.capability,
        baseline_rate=None if not trainable else 0.0,
        incident_rate=1.0,
        gap=None if not trainable else 1.0,
        prevalence=1.0,
        priority=1.0,
        n_relevant_wins=0 if not trainable else 4,
        n_relevant_losses=5,
    )
    b, c = discordant
    return ReplayDiagnosis(
        frontier_model="gpt-5",
        candidate_model="small-8b",
        cross_model_scores=[chosen],
        chosen=chosen,
        within_model_score=within,
        mcnemar=McNemarResult(
            capability_id="c",
            frontier_only=b,
            candidate_only=c,
            concordant=2,
            discordant_n=b + c,
            p_value=0.03125,
        ),
        trainable=trainable,
    )


def test_replay_report_shows_both_contrasts_and_the_same_task_test(tmp_path: Path) -> None:
    report = build_replay_report(_diagnosis(trainable=True), tmp_path / "task", None, {})
    md = write_report(report, tmp_path).read_text()

    assert report["replay"]["trainable"] is True
    assert report["replay"]["mcnemar"]["discordant_n"] == 6
    # The reader has to be able to tell which contrast each number came from.
    assert "Cross-model contrast" in md
    assert "Within-model contrast" in md
    assert "McNemar" in md
    assert "0.0312" in md
    assert str(tmp_path / "task") in md


def test_replay_report_says_identified_not_trainable_and_names_no_task(tmp_path: Path) -> None:
    """The deficit is real and still not worth scaffolding — the report must
    say so prominently rather than printing a task path of None."""
    report = build_replay_report(_diagnosis(trainable=False), None, None, {})
    md = write_report(report, tmp_path).read_text()

    assert report["replay"]["trainable"] is False
    assert "Identified, not trainable" in md
    assert "Generated task" not in md
    assert "None" not in md


def test_replay_report_flags_an_underpowered_mcnemar(tmp_path: Path) -> None:
    report = build_replay_report(_diagnosis(trainable=True, discordant=(2, 1)), None, None, {})
    md = write_report(report, tmp_path).read_text()

    assert report["replay"]["mcnemar"]["underpowered"] is True
    assert "Underpowered" in md
    assert "no power" in md


def test_replay_report_with_no_deficit_still_writes_the_ranked_list(tmp_path: Path) -> None:
    diagnosis = ReplayDiagnosis(
        frontier_model="gpt-5",
        candidate_model="small-8b",
        cross_model_scores=[_score(-1.0)],
        chosen=None,
        within_model_score=None,
        mcnemar=None,
        trainable=None,
    )
    report = build_replay_report(diagnosis, None, None, {"min_gap": 0.2, "min_support": 3})
    md = write_report(report, tmp_path).read_text()

    assert report["replay"]["trainable"] is None
    assert "No deficit found" in md
    assert "Some capability" in md
    # Nothing was chosen, so there is no within-model or McNemar number to show.
    assert "Within-model contrast" not in md
