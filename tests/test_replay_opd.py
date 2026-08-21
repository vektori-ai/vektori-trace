"""`run_replay_chunk_opd` — the replay driver (plan §8).

The driver's job is joining existing parts, so these tests target the seams and
the invariants that make this replay OPD rather than something else wearing the
label:

- the stored DeepSeek action is never the supervised target;
- one frozen ck75 version produces every sample;
- a partial batch is refused, because a missing action silently changes the
  global denominator;
- the 256-token cap cannot come back.
"""

from __future__ import annotations

import pytest

from vektori_trace.chunk_opd import ChunkOPDError
from vektori_trace.replay_opd import (
    ReplayOPDError,
    SampledAction,
    assert_action_is_student_sampled,
    build_replay_batch,
    run_replay_chunk_opd,
)
from vektori_trace.replay_select import (
    REOPD_DEFAULT_KAPPA,
    ReplayPrefix,
    ReplaySelectionError,
    reopd_step_weights,
)

V0 = "ck75-v0"
CAP = 2048


def _prefix(task: str, trace: str, step: int = 2) -> ReplayPrefix:
    return ReplayPrefix(task=task, trace_id=trace, step_index=step, prefix_turns=[])


def _mk(task: str, trace: str, step: int) -> ReplayPrefix:
    return ReplayPrefix(task, trace, step, [])


def _split(text: str, chunk: int) -> list[bytes]:
    raw = text.encode()
    return [raw[i : i + chunk] for i in range(0, len(raw), chunk)]


def _action(prefix: ReplayPrefix, i: int, text: str = '{"cmd": "ls -la"}') -> SampledAction:
    toks = _split(text, 4)
    return SampledAction(
        prefix_id=prefix.prefix_id,
        sample_index=i,
        action_bytes=text.encode(),
        action_token_ids=list(range(1, len(toks) + 1)),
        action_token_bytes=toks,
        behavior_logprobs=[-0.5 - 0.01 * k for k in range(len(toks))],
        policy_version=V0,
    )


def _teacher(action: SampledAction) -> tuple[list[bytes], list[float]]:
    """Teacher tokenisation at a different granularity — the cross-tok case."""
    toks = _split(action.action_bytes.decode(), 3)
    return toks, [-0.3 - 0.02 * (k % 5) for k in range(len(toks))]


def _batch_of(n_prefixes: int = 8, per_prefix: int = 4):
    prefixes = [_prefix(f"task{i}", f"tr{i}") for i in range(n_prefixes)]
    actions, scores = [], {}
    for p in prefixes:
        for i in range(per_prefix):
            a = _action(p, i)
            actions.append(a)
            scores[a.key] = _teacher(a)
    return prefixes, actions, scores


def _noop_step(batch):
    return {"stepped": True, "tokens": batch.global_supervised_tokens}


# ---------------------------------------------------------------------------
# The defining invariant: the stored teacher action is never the target
# ---------------------------------------------------------------------------


def test_sampled_action_matching_the_stored_action_is_refused():
    """Otherwise this is replay SFT with an OPD label."""
    p = _prefix("t", "tr0")
    a = _action(p, 0)
    with pytest.raises(ReplayOPDError, match="byte-identical to the stored"):
        assert_action_is_student_sampled(a, a.action_bytes)


def test_no_stored_action_recorded_is_not_an_error():
    p = _prefix("t", "tr0")
    assert_action_is_student_sampled(_action(p, 0), None)


def test_stored_action_check_runs_inside_build():
    prefixes, actions, scores = _batch_of(8, 2)
    stored = {prefixes[0].prefix_id: actions[0].action_bytes}
    with pytest.raises(ReplayOPDError, match="byte-identical"):
        build_replay_batch(prefixes, actions, scores, stored_teacher_actions=stored)


def test_a_different_stored_action_passes():
    prefixes, actions, scores = _batch_of(8, 2)
    stored = {p.prefix_id: b'{"cmd": "something else entirely"}' for p in prefixes}
    batch = build_replay_batch(
        prefixes, actions, scores, stored_teacher_actions=stored
    )
    assert batch.global_supervised_tokens > 0


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------


def test_batch_aligns_every_action_and_counts_tokens_per_prefix():
    prefixes, actions, scores = _batch_of(8, 4)
    batch = build_replay_batch(prefixes, actions, scores)

    assert len(batch.advantages) == 32
    assert set(batch.supervised_tokens_by_prefix) == {p.prefix_id for p in prefixes}
    assert batch.global_supervised_tokens == sum(
        batch.supervised_tokens_by_prefix.values()
    )
    assert batch.policy_version == V0


def test_missing_teacher_score_is_refused_not_dropped():
    """A short batch still normalises and still trains — the invisible error."""
    prefixes, actions, scores = _batch_of(8, 2)
    scores.pop(actions[0].key)
    with pytest.raises(ReplayOPDError, match="no teacher score"):
        build_replay_batch(prefixes, actions, scores)


def test_mixed_policy_versions_are_refused():
    prefixes, actions, scores = _batch_of(8, 2)
    object.__setattr__(actions[-1], "policy_version", "ck75-v1")
    with pytest.raises(ReplayOPDError, match="mixes policy versions"):
        build_replay_batch(prefixes, actions, scores)


def test_action_referencing_an_unknown_prefix_is_refused():
    prefixes, actions, scores = _batch_of(8, 2)
    object.__setattr__(actions[0], "prefix_id", "ghost@9")
    with pytest.raises(ReplayOPDError, match="no such prefix"):
        build_replay_batch(prefixes, actions, scores)


def test_empty_action_is_refused_at_construction():
    p = _prefix("t", "tr0")
    with pytest.raises(ReplayOPDError, match="empty action"):
        SampledAction(
            prefix_id=p.prefix_id,
            sample_index=0,
            action_bytes=b"",
            action_token_ids=[],
            action_token_bytes=[],
            behavior_logprobs=[],
            policy_version=V0,
        )


def test_token_bytes_must_reconstruct_the_action():
    p = _prefix("t", "tr0")
    with pytest.raises(ReplayOPDError, match="do not reconstruct"):
        SampledAction(
            prefix_id=p.prefix_id,
            sample_index=0,
            action_bytes=b"hello",
            action_token_ids=[1, 2],
            action_token_bytes=[b"hel", b"LO"],
            behavior_logprobs=[-0.1, -0.2],
            policy_version=V0,
        )


def test_one_task_dominating_the_batch_is_refused():
    """§8.4 enforced through the driver, not just the selector."""
    prefixes = [_prefix("hot", "tr0"), _prefix("hot", "tr1")]
    actions, scores = [], {}
    for p in prefixes:
        a = _action(p, 0)
        actions.append(a)
        scores[a.key] = _teacher(a)

    with pytest.raises(ReplaySelectionError, match="of supervised tokens"):
        build_replay_batch(prefixes, actions, scores)


# ---------------------------------------------------------------------------
# run_replay_chunk_opd
# ---------------------------------------------------------------------------


def test_full_run_returns_a_report_and_calls_the_step_once():
    prefixes, actions, scores = _batch_of(8, 4)
    calls = []

    def step(batch):
        calls.append(batch)
        return {"ok": True}

    rep = run_replay_chunk_opd(
        prefixes, actions, scores, step, max_new_tokens=CAP, n_samples_per_prefix=4
    )

    assert len(calls) == 1, "exactly one optimizer step (§8.3)"
    assert rep["n_prefixes"] == 8
    assert rep["n_actions"] == 32
    assert rep["global_supervised_tokens"] > 0
    assert rep["policy_version"] == V0
    assert rep["optimizer"] == {"ok": True}
    assert set(rep["supervised_tokens_by_chunk_kind"]) <= {"1:1", "1:N", "N:1", "M:N"}
    assert len(rep["per_action_stats"]) == 32


def test_report_records_the_selection_policy():
    """§8.3's stratified sample and ReOPD's kappa^t are different experiments."""
    prefixes, actions, scores = _batch_of(8, 4)
    rep = run_replay_chunk_opd(
        prefixes, actions, scores, _noop_step, max_new_tokens=CAP
    )
    assert rep["selection_policy"] == "stratified-diagnostic"

    rep2 = run_replay_chunk_opd(
        prefixes,
        actions,
        scores,
        _noop_step,
        max_new_tokens=CAP,
        selection_policy="reopd-kappa-0.6",
    )
    assert rep2["selection_policy"] == "reopd-kappa-0.6"


def test_previous_256_cap_is_refused_by_the_driver():
    prefixes, actions, scores = _batch_of(8, 4)
    with pytest.raises(ChunkOPDError, match="previous"):
        run_replay_chunk_opd(prefixes, actions, scores, _noop_step, max_new_tokens=256)


def test_short_batch_is_refused():
    prefixes, actions, scores = _batch_of(8, 4)
    with pytest.raises(ReplayOPDError, match="expected 32"):
        run_replay_chunk_opd(
            prefixes,
            actions[:-1],
            scores,
            _noop_step,
            max_new_tokens=CAP,
            n_samples_per_prefix=4,
        )


def test_optimizer_step_is_not_called_when_the_batch_is_invalid():
    prefixes, actions, scores = _batch_of(8, 4)
    scores.pop(actions[0].key)
    calls = []

    with pytest.raises(ReplayOPDError):
        run_replay_chunk_opd(
            prefixes, actions, scores, lambda b: calls.append(b), max_new_tokens=CAP
        )
    assert not calls, "no step may run on a batch that failed validation"


# ---------------------------------------------------------------------------
# ReOPD's kappa^t schedule (borrowed semantics, not the stack)
# ---------------------------------------------------------------------------


def test_weights_decay_with_step_index():
    cands = [_prefix("t", f"tr{i}", step=i) for i in range(5)]
    w = reopd_step_weights(cands, kappa=REOPD_DEFAULT_KAPPA)

    vals = [w[c.prefix_id] for c in cands]
    assert vals == sorted(vals, reverse=True), "earlier steps must weigh more"
    assert sum(vals) == pytest.approx(1.0)


def test_kappa_one_is_uniform():
    cands = [_prefix("t", f"tr{i}", step=i) for i in range(4)]
    w = reopd_step_weights(cands, kappa=1.0)
    assert all(v == pytest.approx(0.25) for v in w.values())


def test_ratio_between_consecutive_steps_is_kappa():
    cands = [_prefix("t", "tr0", step=0), _prefix("t", "tr1", step=1)]
    w = reopd_step_weights(cands, kappa=0.6)
    assert w["tr1@1"] / w["tr0@0"] == pytest.approx(0.6)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_invalid_kappa_is_refused(bad):
    with pytest.raises(ReplaySelectionError, match="kappa must be"):
        reopd_step_weights([_prefix("t", "tr0", 1)], kappa=bad)


def test_empty_candidates_refused():
    with pytest.raises(ReplaySelectionError, match="no candidates"):
        reopd_step_weights([])


def test_kappa_mass_beyond_step_10_matches_the_documented_figures():
    """Locks the numbers the kappa choice is argued from.

    These were wrong once in review (28% claimed for kappa=0.95 over 0..24; the
    real figure is 44.5%), and the whole early-vs-late-prefix argument rests on
    them, so they are asserted rather than left in prose.
    """
    cands = [_prefix("t", f"tr{i}", step=i) for i in range(25)]

    steep = reopd_step_weights(cands, kappa=0.6)
    late_steep = sum(steep[c.prefix_id] for c in cands if c.step_index >= 10)
    assert late_steep == pytest.approx(0.0060, abs=5e-4)

    shallow = reopd_step_weights(cands, kappa=0.95)
    late_shallow = sum(shallow[c.prefix_id] for c in cands if c.step_index >= 10)
    assert late_shallow == pytest.approx(0.4447, abs=1e-3)


def test_pooled_weighting_gives_a_long_trace_more_total_mass():
    """The pooled form is a choice: long traces carry more mass before decay.

    Documented on `reopd_step_weights` because it is one origin of the
    single-trace domination §8.4 forbids.
    """
    long_trace = [_mk("t", "long", i) for i in range(12)]
    short_trace = [_mk("t", "short", i) for i in range(3)]
    w = reopd_step_weights(long_trace + short_trace, kappa=0.95)

    long_mass = sum(w[c.prefix_id] for c in long_trace)
    short_mass = sum(w[c.prefix_id] for c in short_trace)
    assert long_mass > short_mass


def test_report_carries_the_realized_step_histogram():
    """§10 by-trace-stage counts, and the only way to tell the two policies apart."""
    prefixes = [_mk(f"task{i}", f"tr{i}", step=i) for i in range(8)]
    actions, scores = [], {}
    for p in prefixes:
        for i in range(4):
            a = _action(p, i)
            actions.append(a)
            scores[a.key] = _teacher(a)

    rep = run_replay_chunk_opd(
        prefixes, actions, scores, _noop_step, max_new_tokens=CAP
    )

    hist = rep["realized_step_histogram"]
    assert sum(hist.values()) == 32
    assert set(hist) == {str(i) for i in range(8)}
    assert all(v == 4 for v in hist.values()), "4 samples at each prefix"
    assert sum(rep["supervised_tokens_by_step"].values()) == (
        rep["global_supervised_tokens"]
    )
