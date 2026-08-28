"""Crash-and-resume behaviour for the ReOPD run directory.

These tests simulate dying at each point in the state machine. The property
under test throughout: a restart never re-buys a teacher score and never
mistakes a half-written update for a finished one.
"""

from __future__ import annotations

import json

import pytest

import json as _json

from vektori_trace.tau2.reopd_state import (
    ReOPDStateError,
    RunState,
    append_jsonl,
    atomic_write_json,
    read_jsonl,
)


def _checkpoint(u, *, reload_verified=True, update_index=None):
    """A complete, resumable checkpoint."""
    cp = u.checkpoint_path
    cp.mkdir(parents=True, exist_ok=True)
    (cp / "adapter_config.json").write_text("{}")
    (cp / "adapter_model.safetensors").write_text("w")
    (cp / "optimizer.pt").write_text("o")
    (cp / "state.json").write_text(_json.dumps({
        "update_index": u.index if update_index is None else update_index,
        "policy_version": f"update-{u.index}",
        "parent_policy_hash": "abc123",
        "rng_state": "deadbeef",
        "scheduler_state": {"last_epoch": u.index},
        "reload_verified": reload_verified,
    }))


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
    _checkpoint(u)
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
    with pytest.raises(ReOPDStateError, match="have no score"):
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


def test_empty_checkpoint_dir_is_refused(tmp_path):
    """The failure that turns a crash at update 20 into a silent restart."""
    rs = _run(tmp_path)
    u = rs.update(0)
    append_jsonl(u.actions_path, {"key": "0#0"})
    u.mark("SAMPLED")
    append_jsonl(u.scores_path, {"key": "0#0"})
    u.mark("SCORED")
    u.checkpoint_path.mkdir(parents=True)
    u.mark("TRAINED")
    with pytest.raises(ReOPDStateError, match="incomplete"):
        rs.resume_point()


def test_checkpoint_without_reload_proof_is_refused(tmp_path):
    rs = _run(tmp_path)
    u = rs.update(0)
    append_jsonl(u.actions_path, {"key": "0#0"})
    u.mark("SAMPLED")
    append_jsonl(u.scores_path, {"key": "0#0"})
    u.mark("SCORED")
    _checkpoint(u, reload_verified=False)
    u.mark("TRAINED")
    with pytest.raises(ReOPDStateError, match="reload-verified"):
        rs.resume_point()


def test_checkpoint_claiming_wrong_update_is_refused(tmp_path):
    rs = _run(tmp_path)
    u = rs.update(0)
    append_jsonl(u.actions_path, {"key": "0#0"})
    u.mark("SAMPLED")
    append_jsonl(u.scores_path, {"key": "0#0"})
    u.mark("SCORED")
    _checkpoint(u, update_index=7)
    u.mark("TRAINED")
    with pytest.raises(ReOPDStateError, match="claims update 7"):
        rs.resume_point()


def test_duplicate_scores_cannot_mask_a_missing_one(tmp_path):
    """A count check accepts this; exact key-set equality must not."""
    rs = _run(tmp_path)
    u = rs.update(0)
    for k in range(2):
        append_jsonl(u.actions_path, {"key": f"0#{k}"})
    u.mark("SAMPLED")
    append_jsonl(u.scores_path, {"key": "0#0"})
    append_jsonl(u.scores_path, {"key": "0#0"})   # dupe, not 0#1
    u.mark("SCORED")
    with pytest.raises(ReOPDStateError, match="duplicate score keys"):
        rs.resume_point()


def test_foreign_score_keys_are_refused(tmp_path):
    rs = _run(tmp_path)
    u = rs.update(0)
    append_jsonl(u.actions_path, {"key": "0#0"})
    u.mark("SAMPLED")
    append_jsonl(u.scores_path, {"key": "0#0"})
    append_jsonl(u.scores_path, {"key": "9#9"})
    u.mark("SCORED")
    with pytest.raises(ReOPDStateError, match="outside this batch"):
        rs.resume_point()


def test_stale_score_fingerprint_is_refused(tmp_path):
    """A score bought for a different action must not survive a resume."""
    rs = _run(tmp_path)
    u = rs.update(0)
    append_jsonl(u.actions_path, {"key": "0#0", "score_fingerprint": "aaa"})
    u.mark("SAMPLED")
    append_jsonl(u.scores_path, {"key": "0#0", "fingerprint": "bbb"})
    u.mark("SCORED")
    with pytest.raises(ReOPDStateError, match="different action or teacher"):
        rs.resume_point()


def test_missing_required_score_fingerprint_is_refused(tmp_path):
    rs = _run(tmp_path)
    u = rs.update(0)
    append_jsonl(u.actions_path, {"key": "0#0", "score_fingerprint": "aaa"})
    u.mark("SAMPLED")
    append_jsonl(u.scores_path, {"key": "0#0"})
    u.mark("SCORED")
    with pytest.raises(ReOPDStateError, match="different action or teacher"):
        rs.resume_point()


def test_duplicate_action_keys_are_refused(tmp_path):
    rs = _run(tmp_path)
    u = rs.update(0)
    append_jsonl(u.actions_path, {"key": "0#0"})
    append_jsonl(u.actions_path, {"key": "0#0"})
    u.mark("SAMPLED")
    append_jsonl(u.scores_path, {"key": "0#0"})
    u.mark("SCORED")
    with pytest.raises(ReOPDStateError, match="duplicate action keys"):
        rs.resume_point()


def test_malformed_middle_line_is_fatal(tmp_path):
    """Only a torn FINAL line is recoverable."""
    rs = _run(tmp_path)
    u = rs.update(0)
    append_jsonl(u.scores_path, {"key": "0#0"})
    with u.scores_path.open("a") as fh:
        fh.write("{not json}\n")
    append_jsonl(u.scores_path, {"key": "0#1"})
    with pytest.raises(ReOPDStateError, match="malformed JSON at line 2"):
        rs.paid_scores(0)


def test_reached_is_monotonic(tmp_path):
    """A later stage implies every earlier one, even if markers are missing."""
    rs = _run(tmp_path)
    u = rs.update(0)
    append_jsonl(u.actions_path, {"key": "0#0"})
    append_jsonl(u.scores_path, {"key": "0#0"})
    _checkpoint(u)
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
