"""Unit tests for gap.py — the Step 4 headline number, computed as a pure
function over Trace objects (no Docker/subprocess/network)."""

from __future__ import annotations

from pathlib import Path

from vektori_trace.gap import (
    MIN_MEANINGFUL_GAP,
    MIN_TASKS_FOR_FRAMING,
    compute_gap,
    write_gap_report,
)
from vektori_trace.schema import Trace


def _trace(model: str, outcome: str, task: str) -> Trace:
    return Trace(
        run_id=f"{model}-{task}",
        status="x",
        turns=[],
        outcome=outcome,
        source_path=Path("unused"),
        model=model,
        task=task,
    )


def test_pass_rates_are_computed_over_the_paired_set_only() -> None:
    """Task 'c' was only attempted by the frontier — it must not inflate or
    deflate either rate. If a task's absence from the other arm silently
    widened the denominator on one side, comparing two different task sets
    would be exactly the bias Step 4 exists to avoid."""
    traces = [
        _trace("gpt-5", "win", "a"),
        _trace("gpt-5", "win", "b"),
        _trace("gpt-5", "loss", "c"),  # candidate never judged 'c'
        _trace("small", "loss", "a"),
        _trace("small", "win", "b"),
    ]

    result = compute_gap(traces, frontier_model="gpt-5", candidate_model="small", agent="claude-code")

    assert result.paired_n == 2  # only a, b
    assert result.frontier_wins == 2  # wins among {a, b} only, 'c' excluded
    assert result.candidate_wins == 1
    assert result.frontier_rate == 2 / 2
    assert result.candidate_rate == 1 / 2
    assert result.gap == 2 / 2 - 1 / 2
    # attempted counts still show the unpaired attempt, for visibility
    assert result.frontier_attempted == 3
    assert result.candidate_attempted == 2


def test_lopsided_infra_attrition_does_not_bias_the_rate() -> None:
    """The frontier attempted 10 tasks and won all of them; the candidate only
    got a trace on 2 of those 10 (the other 8 excluded upstream as
    InfraFailure) and lost both. The reported rate must reflect only the 2
    tasks both arms actually judged, not 10 frontier wins against 2 candidate
    losses."""
    traces = [_trace("gpt-5", "win", f"t{i}") for i in range(10)]
    traces += [_trace("small", "loss", "t0"), _trace("small", "loss", "t1")]

    result = compute_gap(traces, frontier_model="gpt-5", candidate_model="small", agent="claude-code")

    assert result.paired_n == 2
    assert result.frontier_wins == 2  # not 10
    assert result.frontier_rate == 1.0
    assert result.candidate_rate == 0.0


def test_empty_arm_yields_none_rate_not_a_crash() -> None:
    traces = [_trace("gpt-5", "win", "a")]

    result = compute_gap(traces, frontier_model="gpt-5", candidate_model="small", agent="claude-code")

    assert result.paired_n == 0
    assert result.candidate_attempted == 0
    assert result.frontier_rate is None
    assert result.candidate_rate is None
    assert result.gap is None


def test_write_gap_report_flags_small_paired_n(tmp_path: Path) -> None:
    traces = [_trace("gpt-5", "win", "a"), _trace("small", "loss", "a")]
    result = compute_gap(traces, frontier_model="gpt-5", candidate_model="small", agent="claude-code")
    assert result.paired_n < MIN_TASKS_FOR_FRAMING

    md_path = write_gap_report(result, tmp_path)
    text = md_path.read_text()

    assert f"≥{MIN_TASKS_FOR_FRAMING}" in text


def test_write_gap_report_flags_small_gap() -> None:
    """A gap under ~10 points should say 'change the candidate', independent of
    paired_n — build it directly via GapResult so a large N doesn't also flag
    the other advisory and confuse what's being tested."""
    from vektori_trace.gap import GapResult

    result = GapResult(
        agent="claude-code",
        frontier_model="gpt-5",
        candidate_model="small",
        frontier_attempted=100,
        candidate_attempted=100,
        paired_n=100,
        frontier_wins=55,
        candidate_wins=50,
    )
    assert abs(result.gap) < MIN_MEANINGFUL_GAP

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        text = write_gap_report(result, Path(d)).read_text()

    assert "change the candidate" in text


def test_write_gap_report_omits_advisories_when_gap_is_large_and_n_is_big(tmp_path: Path) -> None:
    traces = []
    for i in range(60):
        traces.append(_trace("gpt-5", "win", f"t{i}"))
        traces.append(_trace("small", "loss", f"t{i}"))
    result = compute_gap(traces, frontier_model="gpt-5", candidate_model="small", agent="claude-code")
    assert result.paired_n >= MIN_TASKS_FOR_FRAMING
    assert result.gap is not None and result.gap >= MIN_MEANINGFUL_GAP

    md_path = write_gap_report(result, tmp_path)
    text = md_path.read_text()

    assert "Advisories" not in text
