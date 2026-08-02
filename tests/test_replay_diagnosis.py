"""The two contrasts and the same-task correction.

`score_deficits` splits on outcome and knows nothing about models, so the whole
cross-model design rests on the input filter being right: if it leaks a
candidate win or a frontier loss, the "cross-model" gap is quietly the old
mixed-model average again. McNemar is here because it's the only number that
compares the models on the *same* task, and its exact tail is easy to get
subtly wrong. No LLM involved — all pure functions over labels.
"""

from __future__ import annotations

from pathlib import Path

from vektori_trace.evaluate.diagnose import (
    Capability,
    McNemarResult,
    TraceLabels,
    _exact_mcnemar_p,
    cross_model_trace_labels,
    diagnose_replay,
    mcnemar_test,
    within_model_trace_labels,
)
from vektori_trace.schema import Trace

CAP = Capability(id="reads_traceback", name="Reads the traceback", description="…")
FRONTIER = "gpt-5"
CANDIDATE = "small-8b"


def _tl(model: str, task: str, outcome: str, label: str | None) -> TraceLabels:
    return TraceLabels(
        trace=Trace(
            run_id=f"{model}-{task}",
            status="",
            turns=[],
            outcome=outcome,
            source_path=Path(f"{model}-{task}"),
            model=model,
            task=task,
        ),
        labels={} if label is None else {CAP.id: label},
        evidence={},
    )


def _mixed() -> list[TraceLabels]:
    """One trace per model per task: the frontier wins t1-t3 and loses t4, the
    candidate loses t1-t3 and wins t4."""
    return [
        _tl(FRONTIER, "t1", "win", "PRESENT"),
        _tl(FRONTIER, "t2", "win", "PRESENT"),
        _tl(FRONTIER, "t3", "win", "PRESENT"),
        _tl(FRONTIER, "t4", "loss", "LACKING"),
        _tl(CANDIDATE, "t1", "loss", "LACKING"),
        _tl(CANDIDATE, "t2", "loss", "LACKING"),
        _tl(CANDIDATE, "t3", "loss", "LACKING"),
        _tl(CANDIDATE, "t4", "win", "PRESENT"),
    ]


def test_cross_model_filter_keeps_only_frontier_wins_and_candidate_losses() -> None:
    kept = cross_model_trace_labels(_mixed(), FRONTIER, CANDIDATE)
    assert {(tl.trace.model, tl.trace.outcome) for tl in kept} == {
        (FRONTIER, "win"),
        (CANDIDATE, "loss"),
    }
    # The frontier's own loss and the candidate's own win are exactly what an
    # unfiltered `score_deficits` would have averaged the contrast away with.
    assert {tl.trace.task for tl in kept} == {"t1", "t2", "t3"}


def test_cross_model_filter_ignores_traces_from_a_third_model() -> None:
    labels = [*_mixed(), _tl("other-model", "t5", "win", "PRESENT")]
    assert all(
        tl.trace.model in (FRONTIER, CANDIDATE)
        for tl in cross_model_trace_labels(labels, FRONTIER, CANDIDATE)
    )


def test_within_model_filter_keeps_one_models_wins_and_losses() -> None:
    kept = within_model_trace_labels(_mixed(), CANDIDATE)
    assert {tl.trace.model for tl in kept} == {CANDIDATE}
    assert sorted(tl.trace.outcome for tl in kept) == ["loss", "loss", "loss", "win"]


def test_exact_mcnemar_p_on_a_symmetric_split_is_one() -> None:
    """b == c is the null hypothesis stated as data — it must not reject."""
    assert _exact_mcnemar_p(3, 3) == 1.0
    assert _exact_mcnemar_p(1, 1) == 1.0


def test_exact_mcnemar_p_cannot_reject_on_one_discordant_pair() -> None:
    """n=1: 2 * P(X=0) = 2 * 0.5 = 1.0. A single pair is a coin flip."""
    assert _exact_mcnemar_p(0, 1) == 1.0
    assert _exact_mcnemar_p(1, 0) == 1.0


def test_exact_mcnemar_p_matches_hand_computed_values() -> None:
    # n=8, k=0: 2 * C(8,0) * 0.5**8 = 2/256
    assert _exact_mcnemar_p(8, 0) == 2 / 256
    # n=10, k=1: 2 * (C(10,0) + C(10,1)) * 0.5**10 = 2 * 11 / 1024
    assert _exact_mcnemar_p(9, 1) == 2 * 11 / 1024
    # Symmetric in its arguments: which model is "b" cannot change a two-sided p.
    assert _exact_mcnemar_p(1, 9) == _exact_mcnemar_p(9, 1)


def test_the_support_floor_is_a_property_of_the_test() -> None:
    """Why MIN_DISCORDANT_PAIRS exists: under 6 pairs no possible split reaches
    p<0.05, and under 9 only a perfectly one-sided one does — a single pair the
    other way puts an 8-pair result back above the line."""
    assert _exact_mcnemar_p(5, 0) > 0.05  # the most extreme 5-pair result
    assert _exact_mcnemar_p(6, 0) < 0.05
    assert _exact_mcnemar_p(7, 1) > 0.05  # 8 pairs, one dissenting -> not significant
    assert _exact_mcnemar_p(8, 1) < 0.05
    assert mcnemar_test(
        [_tl(FRONTIER, "t1", "win", "PRESENT"), _tl(CANDIDATE, "t1", "loss", "LACKING")],
        CAP.id,
        FRONTIER,
        CANDIDATE,
    ).underpowered


def test_mcnemar_counts_b_c_and_concordant_pairs() -> None:
    labels = [
        # t1, t2: frontier had it, candidate didn't -> b
        _tl(FRONTIER, "t1", "win", "PRESENT"),
        _tl(CANDIDATE, "t1", "loss", "LACKING"),
        _tl(FRONTIER, "t2", "win", "PRESENT"),
        _tl(CANDIDATE, "t2", "loss", "LACKING"),
        # t3: candidate had it, frontier didn't -> c
        _tl(FRONTIER, "t3", "loss", "LACKING"),
        _tl(CANDIDATE, "t3", "win", "PRESENT"),
        # t4: both had it -> concordant
        _tl(FRONTIER, "t4", "win", "PRESENT"),
        _tl(CANDIDATE, "t4", "win", "PRESENT"),
        # t5: neither had it -> concordant
        _tl(FRONTIER, "t5", "loss", "LACKING"),
        _tl(CANDIDATE, "t5", "loss", "LACKING"),
    ]
    r = mcnemar_test(labels, CAP.id, FRONTIER, CANDIDATE)
    assert (r.frontier_only, r.candidate_only, r.concordant) == (2, 1, 2)
    assert r.discordant_n == 3
    assert r.p_value == _exact_mcnemar_p(2, 1)


def test_mcnemar_excludes_a_pair_that_is_na_on_either_side() -> None:
    """NA means the capability wasn't relevant to that trajectory, so the two
    sides aren't comparable. Counting it as agreement would pad `concordant`
    with pairs nothing was ever measured on."""
    labels = [
        _tl(FRONTIER, "t1", "win", "PRESENT"),
        _tl(CANDIDATE, "t1", "loss", "NA"),
        _tl(FRONTIER, "t2", "win", "NA"),
        _tl(CANDIDATE, "t2", "loss", "LACKING"),
        # An unlabelled trace is unmeasured too, not PRESENT.
        _tl(FRONTIER, "t3", "win", None),
        _tl(CANDIDATE, "t3", "loss", "LACKING"),
        _tl(FRONTIER, "t4", "win", "PRESENT"),
        _tl(CANDIDATE, "t4", "loss", "LACKING"),
    ]
    r = mcnemar_test(labels, CAP.id, FRONTIER, CANDIDATE)
    assert (r.frontier_only, r.candidate_only, r.concordant, r.discordant_n) == (1, 0, 0, 1)


def test_mcnemar_excludes_tasks_only_one_model_attempted() -> None:
    labels = [
        _tl(FRONTIER, "t1", "win", "PRESENT"),
        _tl(CANDIDATE, "t1", "loss", "LACKING"),
        _tl(FRONTIER, "t2", "win", "PRESENT"),  # candidate never got here
        _tl(CANDIDATE, "t3", "loss", "LACKING"),  # frontier never got here
    ]
    r = mcnemar_test(labels, CAP.id, FRONTIER, CANDIDATE)
    assert (r.frontier_only, r.candidate_only, r.concordant) == (1, 0, 0)


def test_mcnemar_ignores_traces_with_no_task_key() -> None:
    labels = [
        _tl(FRONTIER, "t1", "win", "PRESENT"),
        _tl(CANDIDATE, "t1", "loss", "LACKING"),
    ]
    labels[0].trace.task = None
    r = mcnemar_test(labels, CAP.id, FRONTIER, CANDIDATE)
    assert r.discordant_n == 0
    assert r.p_value is None


def _replay_labels(candidate_present_wins: int) -> list[TraceLabels]:
    """A clean cross-model deficit (frontier PRESENT on every win, candidate
    LACKING on every loss), plus `candidate_present_wins` candidate wins where
    it did demonstrate the capability — the only thing that decides
    trainability."""
    labels = []
    for i in range(6):
        labels.append(_tl(FRONTIER, f"t{i}", "win", "PRESENT"))
        labels.append(_tl(CANDIDATE, f"t{i}", "loss", "LACKING"))
    for i in range(candidate_present_wins):
        labels.append(_tl(FRONTIER, f"e{i}", "win", "PRESENT"))
        labels.append(_tl(CANDIDATE, f"e{i}", "win", "PRESENT"))
    return labels


def test_diagnose_replay_reports_a_trainable_deficit() -> None:
    d = diagnose_replay(_replay_labels(3), [CAP], FRONTIER, CANDIDATE)

    assert d.chosen is not None
    assert d.chosen.gap == 1.0  # frontier wins 0.0 lacking, candidate losses 1.0
    assert d.trainable is True
    # The within-model contrast is the candidate against itself: it demonstrated
    # the capability in all 3 of its own wins, and lacked it in all 6 losses.
    assert d.within_model_score.baseline_rate == 0.0
    assert d.within_model_score.n_relevant_wins == 3
    assert d.within_model_score.incident_rate == 1.0
    assert isinstance(d.mcnemar, McNemarResult)
    assert (d.mcnemar.frontier_only, d.mcnemar.candidate_only) == (6, 0)


def test_diagnose_replay_reports_identified_but_not_trainable() -> None:
    """The candidate never demonstrated the capability, so cross-model says fix
    it and within-model says there's nothing to rejection-sample."""
    d = diagnose_replay(_replay_labels(0), [CAP], FRONTIER, CANDIDATE)

    assert d.chosen is not None
    assert d.chosen.gap == 1.0
    assert d.trainable is False
    assert d.within_model_score.n_relevant_wins == 0
    assert d.within_model_score.baseline_rate is None


def test_diagnose_replay_is_not_trainable_on_thin_within_model_support() -> None:
    """Two demonstrations is the same degenerate case min_support exists to
    reject on the cross-model side; it doesn't become evidence here."""
    d = diagnose_replay(_replay_labels(2), [CAP], FRONTIER, CANDIDATE, min_support=3)
    assert d.chosen is not None
    assert d.within_model_score.n_relevant_wins == 2
    assert d.trainable is False

    assert diagnose_replay(_replay_labels(2), [CAP], FRONTIER, CANDIDATE, min_support=2).trainable


def test_diagnose_replay_returns_no_deficit_when_the_contrast_is_flat() -> None:
    """Nothing cleared the bar — and then there is nothing to test or scaffold
    against, so the second contrast and McNemar are not computed at all."""
    labels = [_tl(FRONTIER, f"t{i}", "win", "LACKING") for i in range(4)]
    labels += [_tl(CANDIDATE, f"t{i}", "loss", "LACKING") for i in range(4)]

    d = diagnose_replay(labels, [CAP], FRONTIER, CANDIDATE)
    assert d.chosen is None
    assert d.trainable is None
    assert d.within_model_score is None
    assert d.mcnemar is None
    assert d.cross_model_scores[0].gap == 0.0  # still reported for inspection


def test_cross_model_contrast_differs_from_the_mixed_average() -> None:
    """The reason the filter exists. Mixed, the candidate's own wins pull the
    baseline up and shrink the gap; filtered, the contrast is the intended one."""
    from vektori_trace.evaluate.diagnose import score_deficits

    labels = _replay_labels(3)
    mixed = score_deficits([CAP], labels)[0]
    cross = diagnose_replay(labels, [CAP], FRONTIER, CANDIDATE).cross_model_scores[0]

    assert mixed.n_relevant_wins == 12  # 9 frontier wins + 3 candidate wins
    assert cross.n_relevant_wins == 9  # frontier only
    assert cross.gap == 1.0
    assert mixed.gap == 1.0  # same here, but measured over a different population
    assert mixed.n_relevant_wins != cross.n_relevant_wins
