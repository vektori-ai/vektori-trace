"""Crash-and-resume behaviour for the ReOPD run directory.

These tests simulate dying at each point in the state machine. The property
under test throughout: a restart never re-buys a teacher score and never
mistakes a half-written update for a finished one.
"""

from __future__ import annotations

import json

import pytest

from vektori_trace.tau2.reopd_state import (
    ReOPDStateError,
    RunState,
    append_jsonl,
    atomic_write_json,
    read_jsonl,
)


def _run(tmp_path, n=4):
    return RunState(tmp_path / "run", n_updates=n)


def _complete(rs, i, n_actions=2):
    """Drive one update all the way to TRAINED."""
    u = rs.update(i)
    u.mark("PLANNED")
    for k in range(n_actions):
        append_jsonl(u.actions_path, {"key": f"{i}#{k}", "action": "x"})
    u.mark("SAMPLED")
    for k in range(n_actions):
        append_jsonl(u.scores_path, {"key": f"{i}#{k}", "logprobs": [-0.1]})
    u.mark("SCORED")
    u.checkpoint_path.mkdir(parents=True, exist_ok=True)
    u.mark("TRAINED")


# --- resume ---------------------------------------------------------------


def test_fresh_run_resumes_at_zero(tmp_path):
    assert _run(tmp_path).resume_point() == 0


def test_resumes_after_completed_updates(tmp_path):
    rs = _run(tmp_path)
    _complete(rs, 0)
    _complete(rs, 1)
    assert rs.resume_point() == 2


def test_finished_run_resumes_past_the_end(tmp_path):
    rs = _run(tmp_path, n=2)
    _complete(rs, 0)
    _complete(rs, 1)
    assert rs.resume_point() == 2


def test_partial_update_resumes_at_itself(tmp_path):
    """Crashed after scoring, before training: redo training, not scoring."""
    rs = _run(tmp_path)
    _complete(rs, 0)
    u = rs.update(1)
    u.mark("PLANNED")
    append_jsonl(u.actions_path, {"key": "1#0"})
    u.mark("SAMPLED")
    append_jsonl(u.scores_path, {"key": "1#0", "logprobs": [-0.2]})
    u.mark("SCORED")

    assert rs.resume_point() == 1
    assert u.stage() == "SCORED"
    assert u.reached("SAMPLED")
    assert not u.reached("TRAINED")


# --- never buy a score twice ---------------------------------------------


def test_paid_scores_are_reused(tmp_path):
    rs = _run(tmp_path)
    u = rs.update(0)
    append_jsonl(u.scores_path, {"key": "0#0", "logprobs": [-0.1]})
    append_jsonl(u.scores_path, {"key": "0#1", "logprobs": [-0.3]})
    paid = rs.paid_scores(0)
    assert set(paid) == {"0#0", "0#1"}


def test_partial_score_file_survives_a_crash(tmp_path):
    """Died at request 2 of 4: the two already billed must still be readable."""
    rs = _run(tmp_path)
    u = rs.update(0)
    u.mark("PLANNED")
    for k in range(4):
        append_jsonl(u.actions_path, {"key": f"0#{k}"})
    u.mark("SAMPLED")
    for k in range(2):
        append_jsonl(u.scores_path, {"key": f"0#{k}", "logprobs": [-0.1]})
    # no SCORED marker -- the stage never finished
    assert not u.reached("SCORED")
    assert set(rs.paid_scores(0)) == {"0#0", "0#1"}


def test_torn_final_line_is_dropped_not_fatal(tmp_path):
    """A crash mid-append leaves a partial line; the rest must still load."""
    rs = _run(tmp_path)
    u = rs.update(0)
    append_jsonl(u.scores_path, {"key": "0#0", "logprobs": [-0.1]})
    with u.scores_path.open("a") as fh:
        fh.write('{"key": "0#1", "logpro')      # torn
    assert set(rs.paid_scores(0)) == {"0#0"}


# --- markers must not outrun their outputs -------------------------------


def test_sampled_without_actions_is_refused(tmp_path):
    rs = _run(tmp_path)
    rs.update(0).mark("SAMPLED")
    with pytest.raises(ReOPDStateError, match="ahead of its outputs"):
        rs.resume_point()


def test_scored_with_short_score_file_is_refused(tmp_path):
    rs = _run(tmp_path)
    u = rs.update(0)
    for k in range(4):
        append_jsonl(u.actions_path, {"key": f"0#{k}"})
    u.mark("SAMPLED")
    append_jsonl(u.scores_path, {"key": "0#0"})
    u.mark("SCORED")                              # lies: 1 score for 4 actions
    with pytest.raises(ReOPDStateError, match="short batch"):
        rs.resume_point()


def test_trained_without_checkpoint_is_refused(tmp_path):
    rs = _run(tmp_path)
    u = rs.update(0)
    append_jsonl(u.actions_path, {"key": "0#0"})
    u.mark("SAMPLED")
    append_jsonl(u.scores_path, {"key": "0#0"})
    u.mark("SCORED")
    u.mark("TRAINED")                             # no checkpoint dir
    with pytest.raises(ReOPDStateError, match="no checkpoint"):
        rs.resume_point()


def test_reached_is_monotonic(tmp_path):
    """A later stage implies every earlier one, even if markers are missing."""
    rs = _run(tmp_path)
    u = rs.update(0)
    u.checkpoint_path.mkdir(parents=True)
    append_jsonl(u.actions_path, {"key": "0#0"})
    append_jsonl(u.scores_path, {"key": "0#0"})
    u.mark("TRAINED")
    assert all(u.reached(s) for s in ("PLANNED", "SAMPLED", "SCORED", "TRAINED"))


# --- the frozen manifest --------------------------------------------------


def test_manifest_freezes_once(tmp_path):
    rs = _run(tmp_path)
    payload = {"n_per_update": 16, "max_new_tokens": 2048}
    assert rs.freeze_manifest(payload) == payload
    assert rs.freeze_manifest(dict(payload)) == payload      # idempotent


def test_manifest_refuses_changed_settings(tmp_path):
    rs = _run(tmp_path)
    rs.freeze_manifest({"n_per_update": 16, "max_new_tokens": 2048})
    with pytest.raises(ReOPDStateError, match="two recipes"):
        rs.freeze_manifest({"n_per_update": 8, "max_new_tokens": 2048})


def test_manifest_ignores_start_time(tmp_path):
    rs = _run(tmp_path)
    rs.freeze_manifest({"n_per_update": 16, "started_at": "t0"})
    rs.freeze_manifest({"n_per_update": 16, "started_at": "t1"})   # no raise


# --- atomic write ---------------------------------------------------------


def test_atomic_write_leaves_no_partial_file(tmp_path):
    p = tmp_path / "a" / "b.json"
    atomic_write_json(p, {"x": 1})
    assert json.loads(p.read_text()) == {"x": 1}
    assert not list(p.parent.glob("*.tmp"))


def test_read_jsonl_missing_file_is_empty(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []


def test_update_index_bounds(tmp_path):
    rs = _run(tmp_path, n=2)
    with pytest.raises(ReOPDStateError):
        rs.update(2)
