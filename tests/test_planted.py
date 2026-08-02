"""The planted-deficit corpus and the recovery harness.

Everything here except `test_recovery_on_an_easy_corpus` runs offline. The
corpus generator carries the load-bearing guarantees — if it leaks the answer
or fails to control a confound, a green recovery run proves nothing — so those
guarantees are tested directly rather than trusted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vektori_trace.evaluate.diagnose import Capability, score_deficits
from vektori_trace.evaluate.planted import (
    DISTRACTOR_MODES,
    MATCH_NEAR,
    MATCH_NONE,
    MATCH_STRICT,
    ORACLE_DISTRACTORS,
    ORACLE_PLANTED,
    SCENARIOS,
    TRUTH,
    VERDICT_BELOW_THRESHOLD,
    VERDICT_NOT_PROPOSED,
    VERDICT_OUTRANKED,
    VERDICT_RECOVERED,
    SweepConfig,
    build_corpus,
    estimate_calls,
    match_planted,
    oracle_labels,
    run_ceiling,
    run_recovery,
    run_sweep,
    write_corpus,
)
from vektori_trace.schema import Trace, load_manifest

# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------


def test_prevalence_is_honoured() -> None:
    corpus = build_corpus(n_wins=10, n_losses=10, prevalence=0.6)
    planted = [k for k in corpus.kinds.values() if k == "loss_planted"]
    assert len(planted) == 6
    assert corpus.prevalence == 0.6


def test_non_planted_losses_use_distractor_modes() -> None:
    """Losses that don't carry the deficit must fail for *some* other reason —
    otherwise prevalence < 1 just means fewer losses, not competing causes."""
    corpus = build_corpus(n_wins=6, n_losses=9, prevalence=1 / 3)
    loss_kinds = [k for rid, k in corpus.kinds.items() if k.startswith("loss")]
    assert sorted(loss_kinds).count("loss_planted") == 3
    others = {k.removeprefix("loss_") for k in loss_kinds if k != "loss_planted"}
    assert others <= set(DISTRACTOR_MODES)
    assert len(others) > 1  # more than one competing explanation on offer


def test_run_ids_do_not_leak_ground_truth() -> None:
    """Run ids go into the proposer's prompt verbatim. A run called
    `loss-planted-3` hands the model the answer key."""
    corpus = build_corpus(n_wins=6, n_losses=6, prevalence=0.5)
    for run_id in corpus.kinds:
        assert run_id.startswith("run-")
        for word in ("planted", "win", "loss", "clean", "distractor", "deficit"):
            assert word not in run_id


def test_trace_payloads_do_not_leak_ground_truth() -> None:
    corpus = build_corpus(n_wins=6, n_losses=6, prevalence=0.5)
    blob = json.dumps(corpus.traces).lower()
    for word in ("planted", "deficit", "ground_truth", "distractor", "capability"):
        assert word not in blob


def test_losing_traces_reason_as_much_as_winning_ones() -> None:
    """The confound control that matters most: if only winners produce thinking
    blocks, the ranker can separate wins from losses without ever engaging with
    the planted capability, and 'recovery' means nothing."""
    corpus = build_corpus(n_wins=8, n_losses=8, prevalence=1.0)
    by_id = {t["runId"]: t for t in corpus.traces}

    def has_thinking(run_id: str) -> bool:
        return any(t.get("thinking") for t in by_id[run_id]["turns"])

    planted = [r for r, k in corpus.kinds.items() if k == "loss_planted"]
    wins = [r for r, k in corpus.kinds.items() if k.startswith("win")]
    assert all(has_thinking(r) for r in planted)
    assert all(has_thinking(r) for r in wins)


def test_planted_losses_repeat_the_identical_call() -> None:
    """The deficit has to actually be in the data, not just in the label."""
    corpus = build_corpus(n_wins=4, n_losses=4, prevalence=1.0)
    by_id = {t["runId"]: t for t in corpus.traces}
    for run_id, kind in corpus.kinds.items():
        if kind != "loss_planted":
            continue
        calls = [tc for turn in by_id[run_id]["turns"] for tc in turn["toolCalls"]]
        assert len(calls) >= 2
        assert all(c["args"] == calls[0]["args"] for c in calls)


def test_recovering_wins_change_their_arguments() -> None:
    corpus = build_corpus(n_wins=8, n_losses=2, prevalence=1.0, clean_win_share=0.0)
    by_id = {t["runId"]: t for t in corpus.traces}
    for run_id, kind in corpus.kinds.items():
        if kind != "win_recovering":
            continue
        calls = [tc for turn in by_id[run_id]["turns"] for tc in turn["toolCalls"]]
        assert len(calls) == 2
        assert calls[0]["args"] != calls[1]["args"]


def test_some_wins_never_exercise_the_capability() -> None:
    """Those wins are genuinely NA, which is what puts the relevant-trace
    denominators under test instead of leaving them assumed."""
    corpus = build_corpus(n_wins=10, n_losses=4, prevalence=1.0, clean_win_share=0.3)
    assert sum(1 for k in corpus.kinds.values() if k == "win_clean") == 3
    assert sum(1 for v in corpus.truth.values() if v == "NA") >= 3


def test_ground_truth_labels_match_trace_kinds() -> None:
    corpus = build_corpus(n_wins=6, n_losses=6, prevalence=0.5)
    for run_id, kind in corpus.kinds.items():
        expected = {
            "win_recovering": "PRESENT",
            "win_clean": "NA",
            "loss_planted": "LACKING",
        }.get(kind, "NA")
        assert corpus.truth[run_id] == expected


def test_corpus_is_seeded() -> None:
    a = build_corpus(n_wins=6, n_losses=6, prevalence=0.5, seed=7)
    b = build_corpus(n_wins=6, n_losses=6, prevalence=0.5, seed=7)
    c = build_corpus(n_wins=6, n_losses=6, prevalence=0.5, seed=8)
    assert a.traces == b.traces
    assert a.traces != c.traces


@pytest.mark.parametrize("prevalence", [-0.1, 1.5])
def test_invalid_prevalence_raises(prevalence: float) -> None:
    with pytest.raises(ValueError):
        build_corpus(n_wins=4, n_losses=4, prevalence=prevalence)


# ---------------------------------------------------------------------------
# On-disk round trip
# ---------------------------------------------------------------------------


def test_corpus_round_trips_through_the_real_load_path(tmp_path: Path) -> None:
    """Recovery must be measured through the same loader a real diagnosis uses,
    not an in-memory shortcut that skips the manifest."""
    corpus = build_corpus(n_wins=5, n_losses=5, prevalence=0.8)
    manifest_path = write_corpus(corpus, tmp_path)

    entries = load_manifest(manifest_path)
    traces = [Trace.load(e.path, outcome=e.outcome) for e in entries]

    assert len(traces) == 10
    assert sum(t.outcome == "win" for t in traces) == 5
    assert all(t.turns for t in traces)
    assert {t.run_id for t in traces} == set(corpus.kinds)


def test_ground_truth_is_written_beside_the_traces_not_inside_them(tmp_path: Path) -> None:
    corpus = build_corpus(n_wins=4, n_losses=4, prevalence=1.0)
    write_corpus(corpus, tmp_path)

    truth = json.loads((tmp_path / "ground_truth.json").read_text())
    assert truth["expected_labels"] == corpus.truth
    for trace_file in (tmp_path / "traces").glob("*.json"):
        assert "expected" not in trace_file.read_text().lower()


# ---------------------------------------------------------------------------
# The matcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cap_id", "name", "description", "expected"),
    [
        (
            "error_recovery",
            "Adapts after a failed tool call",
            "Reads the error message and issues a corrected call.",
            MATCH_STRICT,
        ),
        (
            "no_blind_retry",
            "Avoids blind retries",
            "Does not repeat an identical failing call.",
            MATCH_STRICT,
        ),
        (
            "instruction_following",
            "Follows the requested output format",
            "Returns results in the shape the user asked for.",
            MATCH_NONE,
        ),
        (
            "autonomy",
            "Proceeds without confirmation",
            "Completes the task rather than asking the user to restate it.",
            MATCH_NONE,
        ),
    ],
)
def test_matcher_verdicts(cap_id: str, name: str, description: str, expected: str) -> None:
    assert match_planted(Capability(id=cap_id, name=name, description=description)) == expected


def test_vague_but_probably_right_capability_is_a_near_miss() -> None:
    """`error_handling` names the error signal and nothing about acting on it.

    It is probably the planted concept, stated too vaguely to be sure. Scoring
    it as a hit overstates the ranker and scoring it as a miss understates it,
    which is why the verdict is three-way and both rates get reported.
    """
    vague = Capability(id="error_handling", name="Error handling", description="Handles errors.")
    assert match_planted(vague) == MATCH_NEAR
    unrelated = Capability(id="planning", name="Planning", description="Plans ahead.")
    assert match_planted(unrelated) == MATCH_NONE


# ---------------------------------------------------------------------------
# Scoring the planted corpus under a perfect labeller
# ---------------------------------------------------------------------------


def test_perfect_labels_yield_the_expected_gap_and_prevalence() -> None:
    """A sanity check on the experiment's own arithmetic: if the labeller were
    perfect, the planted capability scores gap 1.0 at the planted prevalence.
    That is the ceiling every real run is measured against."""
    from vektori_trace.evaluate.diagnose import TraceLabels

    corpus = build_corpus(n_wins=10, n_losses=10, prevalence=0.6)
    cap = Capability(id="planted", name="planted", description="…")
    by_id = {t["runId"]: t for t in corpus.traces}

    labels = [
        TraceLabels(
            trace=Trace(
                run_id=rid,
                status="",
                turns=[],
                outcome="win" if kind.startswith("win") else "loss",
                source_path=Path(rid),
            ),
            labels={cap.id: corpus.truth[rid]},
            evidence={},
        )
        for rid, kind in corpus.kinds.items()
        if rid in by_id
    ]

    score = score_deficits([cap], labels)[0]
    assert score.baseline_rate == 0.0
    assert score.incident_rate == 1.0
    assert score.gap == 1.0
    assert score.prevalence == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# The ceiling
# ---------------------------------------------------------------------------


def test_ceiling_recovers_on_a_comfortable_corpus() -> None:
    """A perfect proposer and labeller must recover the deficit, or the corpus
    itself — not the ranker — is what's broken."""
    corpus = build_corpus(n_wins=12, n_losses=12, prevalence=1.0, seed=0)
    result = run_ceiling(corpus)

    assert result.recovered
    assert result.matched_rank == 0
    assert result.scores[0].gap == 1.0
    assert result.label_accuracy.accuracy == 1.0


def test_ceiling_beats_the_distractors_at_low_prevalence() -> None:
    """Even when most losses fail for other reasons, the planted capability
    should still rank first: it has the larger gap on the traces where it's
    relevant, and priority is gap × prevalence."""
    corpus = build_corpus(n_wins=20, n_losses=20, prevalence=0.3, seed=0)
    result = run_ceiling(corpus)

    assert result.recovered
    assert result.scores[0].gap == 1.0


def test_ceiling_is_zero_when_support_is_short_despite_a_perfect_signal() -> None:
    """The finding this whole ceiling exists to surface.

    At 6 losses and prevalence 0.3 there are 2 deficit-carrying losses. The
    capability still ranks first with a gap of 1.0 — the signal is perfect and
    unambiguous — and `min_support` rejects it anyway. Configs like this
    measure the threshold, not the ranker, and spending LLM calls on them tells
    you nothing.
    """
    corpus = build_corpus(n_wins=6, n_losses=6, prevalence=0.3, seed=0)
    result = run_ceiling(corpus)

    assert not result.recovered
    assert result.verdict == VERDICT_BELOW_THRESHOLD
    assert result.matched_rank == 0
    assert result.scores[0].gap == 1.0  # nothing wrong with the evidence
    assert result.scores[0].n_relevant_losses == 2  # there just isn't enough of it

    # Lower the support floor and the same corpus recovers.
    assert run_ceiling(corpus, min_support=2).recovered


def test_ceiling_support_floor_applies_to_wins_too() -> None:
    """Three wins minus the clean-win share leaves two that exercise the
    capability, which is below the floor however many losses there are."""
    corpus = build_corpus(n_wins=3, n_losses=12, prevalence=1.0, seed=0)
    result = run_ceiling(corpus)

    assert result.scores[0].n_relevant_wins == 2
    assert not result.recovered
    assert result.verdict == VERDICT_BELOW_THRESHOLD


def test_oracle_labels_cover_every_trace_kind() -> None:
    """A kind with no oracle label would silently score as NA and quietly
    deflate the ceiling."""
    for kind in TRUTH:
        labels = oracle_labels(kind)
        assert labels[ORACLE_PLANTED.id] == TRUTH[kind]
        assert len(labels) == 1 + len(DISTRACTOR_MODES)


def test_each_distractor_is_lacking_only_on_its_own_traces() -> None:
    for mode, cap in ORACLE_DISTRACTORS.items():
        for kind in TRUTH:
            label = oracle_labels(kind)[cap.id]
            if kind == f"loss_{mode}":
                assert label == "LACKING"
            else:
                assert label != "LACKING"


# ---------------------------------------------------------------------------
# Sweep bookkeeping
# ---------------------------------------------------------------------------


def test_sweep_computes_the_ceiling_without_any_llm(tmp_path: Path) -> None:
    """`--ceiling-only` must not touch the network. Nothing is stubbed here —
    if it tried to call OpenAI it would raise on the missing key."""
    cells = run_sweep(
        [SweepConfig(12, 12, 1.0), SweepConfig(3, 3, 1.0)],
        tmp_path,
        repeats=2,
        ceiling_only=True,
    )

    assert [c.ceiling_rate for c in cells] == [1.0, 0.0]
    assert all(c.results == [] for c in cells)
    assert all(c.recovery_rate == 0.0 for c in cells)
    assert (tmp_path / "corpora").exists()


def test_sweep_writes_every_corpus_and_its_ceiling(tmp_path: Path) -> None:
    """Nothing in the report may be underivable from what's on disk."""
    run_sweep([SweepConfig(6, 6, 1.0)], tmp_path, repeats=2, ceiling_only=True)

    run_dirs = sorted((tmp_path / "corpora").iterdir())
    assert len(run_dirs) == 2  # one per repeat, distinct seeds
    for d in run_dirs:
        assert (d / "manifest.json").exists()
        assert (d / "ground_truth.json").exists()
        assert (d / "ceiling.json").exists()
        assert len(list((d / "traces").glob("*.json"))) == 12


def test_estimate_calls_counts_proposer_plus_one_per_trace() -> None:
    configs = [SweepConfig(3, 3, 1.0), SweepConfig(6, 6, 0.5)]
    assert estimate_calls(configs, repeats=1) == (1 + 6) + (1 + 12)
    assert estimate_calls(configs, repeats=3) == 3 * ((1 + 6) + (1 + 12))


def test_scenarios_are_distinct() -> None:
    assert len({s.task for s in SCENARIOS}) == len(SCENARIOS)
    assert len(SCENARIOS) >= 6  # enough variety that the labeller isn't reading a template


# ---------------------------------------------------------------------------
# The recovery harness, with the model stubbed out
#
# What the live test can't check cheaply: that a recovery is scored correctly
# once the LLM has spoken. Each failure mode calls for a different response —
# a proposer that never names the capability is a prompt problem, a distractor
# outranking it is a labeller problem, a top-ranked match rejected by the
# threshold is a calibration problem — so the three must not collapse together.
# ---------------------------------------------------------------------------

PLANTED_CAP = Capability(
    id="error_adaptation",
    name="Adapts after a failed tool call",
    description="Reads the error message and issues a corrected call.",
)
DISTRACTOR_CAP = Capability(
    id="autonomy",
    name="Proceeds without asking",
    description="Completes the task rather than bouncing it back to the user.",
)


def _stub_diagnosis(monkeypatch, proposed, label_fn):
    """Replace both LLM calls in the module under test."""
    from vektori_trace.evaluate import planted as mod

    monkeypatch.setattr(mod, "propose_capabilities", lambda traces, model=None: proposed)

    def fake_label(trace, capabilities, model=None):
        from vektori_trace.evaluate.diagnose import TraceLabels

        return TraceLabels(trace=trace, labels=label_fn(trace), evidence={})

    monkeypatch.setattr(mod, "label_trace", fake_label)


def _corpus_on_disk(tmp_path: Path, **kwargs):
    corpus = build_corpus(**kwargs)
    return corpus, write_corpus(corpus, tmp_path)


def test_perfect_labeller_recovers_the_deficit(tmp_path: Path, monkeypatch) -> None:
    corpus, manifest_path = _corpus_on_disk(
        tmp_path, n_wins=8, n_losses=8, prevalence=1.0, seed=0
    )
    _stub_diagnosis(
        monkeypatch,
        [PLANTED_CAP, DISTRACTOR_CAP],
        lambda t: {PLANTED_CAP.id: corpus.truth[t.run_id], DISTRACTOR_CAP.id: "NA"},
    )

    result = run_recovery(manifest_path, corpus.truth)

    assert result.recovered
    assert result.verdict == VERDICT_RECOVERED
    assert result.matched is not None and result.matched.id == PLANTED_CAP.id
    assert result.matched_rank == 0
    assert result.label_accuracy.accuracy == 1.0
    assert result.label_accuracy.n == 16


def test_verdict_when_the_proposer_never_names_it(tmp_path: Path, monkeypatch) -> None:
    corpus, manifest_path = _corpus_on_disk(
        tmp_path, n_wins=6, n_losses=6, prevalence=1.0, seed=0
    )
    _stub_diagnosis(
        monkeypatch,
        [DISTRACTOR_CAP],
        lambda t: {DISTRACTOR_CAP.id: "LACKING" if t.outcome == "loss" else "PRESENT"},
    )

    result = run_recovery(manifest_path, corpus.truth)

    assert not result.recovered
    assert result.verdict == VERDICT_NOT_PROPOSED
    assert result.matched is None
    # No matched capability means no labels to score against ground truth.
    assert result.label_accuracy.n == 0
    assert result.label_accuracy.accuracy is None


def test_verdict_when_a_distractor_outranks_it(tmp_path: Path, monkeypatch) -> None:
    """The capability is named, but a competing explanation scores higher — a
    labelling problem, and a different thing from never proposing it."""
    corpus, manifest_path = _corpus_on_disk(
        tmp_path, n_wins=8, n_losses=8, prevalence=0.5, seed=0
    )

    def labels(t):
        # The distractor is (wrongly) marked LACKING in every loss, so it wins
        # on prevalence; the planted one is labelled correctly.
        return {
            PLANTED_CAP.id: corpus.truth[t.run_id],
            DISTRACTOR_CAP.id: "LACKING" if t.outcome == "loss" else "PRESENT",
        }

    _stub_diagnosis(monkeypatch, [PLANTED_CAP, DISTRACTOR_CAP], labels)
    result = run_recovery(manifest_path, corpus.truth)

    assert not result.recovered
    assert result.verdict == VERDICT_OUTRANKED
    assert result.matched is not None
    assert result.matched_rank is not None and result.matched_rank > 0


def test_verdict_when_the_threshold_rejects_a_top_ranked_match(
    tmp_path: Path, monkeypatch
) -> None:
    """Top of the list and still rejected: the pipeline worked and the
    threshold is what disagreed. The one failure mode that isn't a bug."""
    corpus, manifest_path = _corpus_on_disk(
        tmp_path, n_wins=8, n_losses=8, prevalence=1.0, seed=0
    )
    _stub_diagnosis(
        monkeypatch,
        [PLANTED_CAP],
        lambda t: {PLANTED_CAP.id: corpus.truth[t.run_id]},
    )

    result = run_recovery(manifest_path, corpus.truth, min_gap=0.99, min_support=999)

    assert not result.recovered
    assert result.verdict == VERDICT_BELOW_THRESHOLD
    assert result.matched_rank == 0
    assert result.selected is None


def test_label_accuracy_records_the_confusion(tmp_path: Path, monkeypatch) -> None:
    """Blur measured directly: which way the labeller is wrong matters, since
    LACKING-read-as-NA and NA-read-as-LACKING push the gap opposite ways."""
    corpus, manifest_path = _corpus_on_disk(
        tmp_path, n_wins=4, n_losses=4, prevalence=1.0, seed=0
    )
    # Every trace mislabelled NA.
    _stub_diagnosis(monkeypatch, [PLANTED_CAP], lambda t: {PLANTED_CAP.id: "NA"})

    result = run_recovery(manifest_path, corpus.truth)

    acc = result.label_accuracy
    assert acc.n == 8
    assert acc.confusion["LACKING->NA"] == 4
    assert acc.accuracy is not None and acc.accuracy < 1.0
    # Nothing was relevant anywhere, so there is no gap to find.
    assert result.scores[0].gap is None
    assert result.verdict == VERDICT_BELOW_THRESHOLD


def test_missing_label_is_scored_as_na_against_ground_truth(
    tmp_path: Path, monkeypatch
) -> None:
    """The labeller returning no entry at all is the same as saying NA — the
    convention scoring already uses, applied here too so accuracy doesn't
    silently count a non-answer as correct."""
    corpus, manifest_path = _corpus_on_disk(
        tmp_path, n_wins=4, n_losses=4, prevalence=1.0, seed=0
    )
    _stub_diagnosis(monkeypatch, [PLANTED_CAP], lambda t: {})

    result = run_recovery(manifest_path, corpus.truth)

    acc = result.label_accuracy
    assert acc.n == 8
    assert acc.confusion["LACKING->NA"] == 4
    assert acc.confusion["PRESENT->NA"] > 0


# ---------------------------------------------------------------------------
# The live regression test
# ---------------------------------------------------------------------------


@pytest.mark.llm
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY (costs money)"
)
def test_recovery_on_an_easy_corpus(tmp_path: Path) -> None:
    """The frozen regression: on the easiest corpus in the sweep — plenty of
    traces, deficit in every loss — the ranker must name the planted capability
    and select it.

    If this fails, nothing downstream is worth running. Deliberately the easy
    end of the sweep: it is a floor, not a benchmark.
    """
    corpus = build_corpus(n_wins=6, n_losses=6, prevalence=1.0, seed=0)
    manifest_path = write_corpus(corpus, tmp_path)

    result = run_recovery(manifest_path, corpus.truth)

    assert result.matched is not None, (
        "the proposer never named the planted capability; proposed: "
        f"{[c.name for c in result.proposed]}"
    )
    assert result.recovered, f"verdict={result.verdict}, ranked={[s.capability.name for s in result.scores]}"
