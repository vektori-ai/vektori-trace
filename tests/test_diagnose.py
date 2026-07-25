"""Scoring and deficit selection — the parts that decide what gets reported.

No LLM involved: `score_deficits` and `select_deficit` are pure functions over
labels, which is exactly why they're worth pinning.
"""

from __future__ import annotations

from pathlib import Path

from vektori_trace.diagnose import (
    Capability,
    TraceLabels,
    score_deficits,
    select_deficit,
)
from vektori_trace.schema import Trace

CAP = Capability(id="reads_traceback", name="Reads the traceback", description="…")


def _trace(run_id: str, outcome: str) -> Trace:
    return Trace(
        run_id=run_id, status="", turns=[], outcome=outcome, source_path=Path(run_id)
    )


def _labels(outcome: str, label: str | None, n: int, prefix: str) -> list[TraceLabels]:
    """n traces with `label` for CAP — or, when label is None, no entry at all."""
    return [
        TraceLabels(
            trace=_trace(f"{prefix}{i}", outcome),
            labels={} if label is None else {CAP.id: label},
            evidence={},
        )
        for i in range(n)
    ]


def _score(trace_labels):
    return score_deficits([CAP], trace_labels)[0]


def test_gap_is_measured_over_relevant_traces_only() -> None:
    labels = (
        _labels("win", "PRESENT", 4, "w")
        + _labels("loss", "LACKING", 3, "l")
        + _labels("loss", "NA", 5, "na")
    )
    s = _score(labels)
    assert s.baseline_rate == 0.0
    assert s.incident_rate == 1.0
    assert s.gap == 1.0
    assert s.n_relevant_wins == 4
    assert s.n_relevant_losses == 3
    # prevalence is over ALL losses, not just relevant ones
    assert s.prevalence == 3 / 8


def test_missing_label_counts_as_na_not_present() -> None:
    """A capability the labeller returned no verdict on is unmeasured.

    Reading silence as PRESENT drags the win-side baseline to 0 and
    manufactures a gap out of missing data — so the un-labelled wins here must
    drop out of the denominator entirely, leaving the one real win to set the
    baseline and the gap at 0.
    """
    labels = (
        _labels("win", None, 6, "w")
        + _labels("win", "LACKING", 1, "wl")
        + _labels("loss", "LACKING", 4, "l")
    )
    s = _score(labels)
    assert s.n_relevant_wins == 1
    assert s.baseline_rate == 1.0
    assert s.gap == 0.0


def test_select_deficit_rejects_an_inverted_gap() -> None:
    """gap = -1.0 means the capability was *more* present in losses. Evidence
    against the hypothesis must not come back as a confident diagnosis."""
    labels = _labels("win", "LACKING", 4, "w") + _labels("loss", "PRESENT", 4, "l")
    scores = score_deficits([CAP], labels)
    assert scores[0].gap == -1.0
    assert select_deficit(scores) is None


def test_select_deficit_rejects_a_gap_below_threshold() -> None:
    labels = (
        _labels("win", "PRESENT", 9, "w")
        + _labels("win", "LACKING", 1, "wl")
        + _labels("loss", "PRESENT", 8, "l")
        + _labels("loss", "LACKING", 2, "ll")
    )
    scores = score_deficits([CAP], labels)
    assert scores[0].gap == 0.1
    assert select_deficit(scores, min_gap=0.20) is None
    assert select_deficit(scores, min_gap=0.05) is scores[0]


def test_select_deficit_rejects_thin_support() -> None:
    """A perfect gap measured on two traces is not a finding."""
    labels = _labels("win", "PRESENT", 2, "w") + _labels("loss", "LACKING", 2, "l")
    scores = score_deficits([CAP], labels)
    assert scores[0].gap == 1.0
    assert select_deficit(scores, min_support=3) is None
    assert select_deficit(scores, min_support=2) is scores[0]


def test_select_deficit_skips_past_a_rejected_leader() -> None:
    """The ranked list is by priority; selection walks it and takes the first
    entry that actually clears the bar, rather than stopping at the top."""
    weak = Capability(id="weak", name="Weak", description="…")
    strong = Capability(id="strong", name="Strong", description="…")

    def tl(outcome, weak_label, strong_label, run_id):
        return TraceLabels(
            trace=_trace(run_id, outcome),
            labels={"weak": weak_label, "strong": strong_label},
            evidence={},
        )

    labels = (
        # `weak` looks perfect (gap 1.0, prevalence 1.0) but is relevant in
        # only 2 wins; `strong` is weaker but measured on every trace.
        [tl("win", "PRESENT", "PRESENT", f"w{i}") for i in range(2)]
        + [tl("win", "NA", "PRESENT", f"wna{i}") for i in range(2)]
        + [tl("loss", "LACKING", "LACKING", f"l{i}") for i in range(3)]
        + [tl("loss", "LACKING", "PRESENT", "lp0")]
    )
    scores = score_deficits([weak, strong], labels)
    assert scores[0].capability.id == "weak"
    assert scores[0].gap == 1.0
    assert scores[0].n_relevant_wins == 2
    chosen = select_deficit(scores, min_support=3)
    assert chosen is not None
    assert chosen.capability.id == "strong"


def test_select_deficit_returns_none_on_empty_scores() -> None:
    assert select_deficit([]) is None
