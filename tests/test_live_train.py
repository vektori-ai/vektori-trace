"""The live SAMPLED -> SCORED -> TRAINED bridge.

No GPU and no paid call: the trainer and teacher pool are injected, which is
the same seam `run_replay_chunk_opd` was built with. What is tested here is the
bookkeeping that makes a live update trainable at all -- the share limit, the
stale-score filter, the provenance binding and the policy-version gate -- each
of which fails silently rather than loudly if it is wrong.
"""

from __future__ import annotations

import base64
import json

import pytest

from vektori_trace.tau2.live_train import (
    LiveTrainError,
    live_max_trace_share,
    load_live_update_inputs,
    train_live_update,
)
from vektori_trace.tau2.live_turns import live_score_fingerprint
from vektori_trace.tau2.reopd_state import (
    RunState,
    UpdateDir,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
)


# --- the share limit -------------------------------------------------------


def test_share_limit_admits_a_four_episode_batch():
    """The replay default of 0.35 would reject the Phase 1 proof outright."""
    share = live_max_trace_share(4)
    assert share == pytest.approx(0.375)
    assert share > 0.35


def test_share_limit_is_one_for_a_single_episode():
    """Honest rather than convenient: one trajectory has no concentration to
    measure, so the episode count is the thing to raise, not the limit."""
    assert live_max_trace_share(1) == 1.0


def test_share_limit_tightens_as_episodes_are_added():
    assert live_max_trace_share(8) < live_max_trace_share(4)


def test_share_limit_refuses_an_empty_batch():
    with pytest.raises(LiveTrainError, match="at least one episode"):
        live_max_trace_share(0)


# --- fixtures --------------------------------------------------------------


def _row(episode: str, turn: int, *, policy="live-u000", history=None, action=b"A"):
    history = history or [{"role": "user", "content": f"{episode}-{turn}"}]
    from vektori_trace.tau2.live_episode import semantic_hash

    row = {
        "prefix_id": f"{episode}@{turn}",
        "sample_index": 0,
        "key": f"{episode}@{turn}#0",
        "action_bytes_b64": base64.b64encode(action).decode(),
        "action_token_bytes_b64": [base64.b64encode(action).decode()],
        "action_token_ids": [65],
        "behavior_logprobs": [-0.2],
        "prompt_token_ids": [1, 2, 3],
        "policy_version": policy,
        "finish_reason": "stop",
        "episode_id": episode,
        "task_id": "57",
        "turn_index": turn,
        "semantic_history_hash": semantic_hash(history),
        "teacher_context_hash": "teacher-1",
    }
    row["score_fingerprint"] = live_score_fingerprint(row)
    return row, history


def _sampled_update(tmp_path, rows_and_histories, *, index=0):
    u = UpdateDir(tmp_path, index)
    u.path.mkdir(parents=True, exist_ok=True)
    rows = [r for r, _ in rows_and_histories]
    rendered = {r["prefix_id"]: h for r, h in rows_and_histories}
    atomic_write_jsonl(u.actions_path, rows)
    atomic_write_json(u.path / "rendered.json", rendered)
    u.mark("PLANNED", {"stage": "PLANNED"})
    u.mark("SAMPLED", {"stage": "SAMPLED", "actions": len(rows)})
    return u


# --- loading a sampled update ---------------------------------------------


def test_load_assembles_actions_prefixes_and_share(tmp_path):
    pairs = [
        _row("ep-1", 0), _row("ep-1", 1),
        _row("ep-2", 0), _row("ep-2", 1),
        _row("ep-3", 0), _row("ep-4", 0),
    ]
    u = _sampled_update(tmp_path, pairs)

    inputs = load_live_update_inputs(u, policy_version="live-u000")

    assert len(inputs.actions) == 6
    assert len(inputs.prefixes) == 6
    assert inputs.n_episodes == 4
    # Derived from the episode count, not inherited from replay.
    assert inputs.max_trace_share == pytest.approx(0.375)
    # The four fields the training path reads off a prefix.
    assert inputs.prefixes[0].task == "57"
    assert inputs.prefixes[0].prefix_id == "ep-1@0"


def test_load_refuses_an_update_that_never_sampled(tmp_path):
    u = UpdateDir(tmp_path, 0)
    u.path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(LiveTrainError, match="has not reached SAMPLED"):
        load_live_update_inputs(u, policy_version="live-u000")


def test_load_refuses_actions_from_another_policy(tmp_path):
    """The student policy is fixed for an entire batch. A turn sampled under
    the previous adapter makes its importance ratio compare two different
    distributions, with a finite loss and nothing in the logs."""
    u = _sampled_update(tmp_path, [_row("ep-1", 0, policy="live-u000")])
    with pytest.raises(LiveTrainError, match="sampled under"):
        load_live_update_inputs(u, policy_version="live-u001")


def test_load_refuses_a_turn_with_no_teacher_context(tmp_path):
    row, _ = _row("ep-1", 0)
    u = UpdateDir(tmp_path, 0)
    u.path.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(u.actions_path, [row])
    atomic_write_json(u.path / "rendered.json", {})
    u.mark("SAMPLED", {"stage": "SAMPLED"})
    with pytest.raises(LiveTrainError, match="no rendered teacher context"):
        load_live_update_inputs(u, policy_version="live-u000")


def test_load_proves_histories_reproduce_captured_prompt_ids(tmp_path):
    """The seam between 'the history we stored' and 'the state the student
    sampled in'. A history that does not re-render is one the teacher would
    score under a conversation that never happened."""
    u = _sampled_update(tmp_path, [_row("ep-1", 0)])

    with pytest.raises(Exception, match="does not reproduce captured prompt"):
        load_live_update_inputs(
            u, policy_version="live-u000", render_ids=lambda _m: [9, 9, 9]
        )

    inputs = load_live_update_inputs(
        u, policy_version="live-u000", render_ids=lambda _m: [1, 2, 3]
    )
    assert len(inputs.actions) == 1


# --- scoring and training --------------------------------------------------


class _Pool:
    """Stands in for Fireworks. Records what it was asked to pay for."""

    def __init__(self):
        self.scored_keys = []


class _Trainer:
    def __init__(self):
        self.steps = []

    def step(self, batch):
        self.steps.append(batch)
        return {"loss": 0.5, "grad_norm": 1.0, "n_examples": 1}

    def checkpoint(self, path, *, update_index, policy_version):
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "update_index": update_index,
            "policy_version": policy_version,
            "adapter_hash": "adapter-next",
            # Resuming with a fresh optimizer discards Adam's moments, which
            # silently changes the effective learning rate for every update
            # that follows. `validate_checkpoint` requires all of these.
            "parent_policy_hash": "parent-1",
            "rng_state": "rng-1",
            "scheduler_state": {"last_epoch": update_index},
            # A saved adapter that does not change logits on reload is the
            # failure that makes a whole run a no-op, so the real trainer
            # proves the reload before writing this.
            "reload_verified": True,
        }
        (path / "state.json").write_text(json.dumps(state))
        # `validate_checkpoint` requires every file a resume needs. Writing a
        # partial checkpoint here would make the test pass against a stub the
        # real run would reject.
        (path / "adapter_config.json").write_text("{}")
        (path / "optimizer.pt").write_bytes(b"")
        (path / "adapter_model.safetensors").write_bytes(b"")
        return state


def _fake_score(monkeypatch, seen):
    """Patch the scorer at the point `opd_stages` imports it."""
    import vektori_trace.replay_score as rs

    class _Scored:
        def __init__(self, key, action):
            self.key = key
            self.teacher_token_bytes = [action]
            self.teacher_logprobs = [-0.1]
            self.n_prefix_tokens = 3
            self.n_trailing_dropped = 0

    def fake(actions, rendered, tok, pool, *, on_scored=None,
             already_scored=None, **kw):
        already_scored = already_scored or {}
        out = {}
        for a in actions:
            if a.key in already_scored:
                out[a.key] = already_scored[a.key]
                continue
            seen.append(a.key)
            sc = _Scored(a.key, a.action_bytes)
            if on_scored:
                on_scored(sc)
            out[a.key] = (sc.teacher_token_bytes, sc.teacher_logprobs)
        return out, {"teacher_input_tokens": 100,
                     "n_newly_scored": len(seen),
                     "n_reused_from_disk": len(already_scored)}

    monkeypatch.setattr(rs, "score_replay_batch", fake)


def _fake_chunk_opd(monkeypatch, captured):
    import vektori_trace.replay_opd as ro

    def fake(prefixes, actions, scored, step, **kw):
        captured.update(kw)
        captured["n_prefixes"] = len(prefixes)
        step({"n": len(actions)})
        return {"optimizer": {"loss": 0.5}, "global_supervised_tokens": 4}

    monkeypatch.setattr(ro, "run_replay_chunk_opd", fake)


def test_train_scores_then_steps_and_checkpoints(tmp_path, monkeypatch):
    pairs = [_row("ep-1", 0), _row("ep-1", 1), _row("ep-2", 0), _row("ep-3", 0)]
    u = _sampled_update(tmp_path, pairs)
    run = RunState(tmp_path, n_updates=1)

    seen, kw = [], {}
    _fake_score(monkeypatch, seen)
    _fake_chunk_opd(monkeypatch, kw)

    inputs = load_live_update_inputs(u, policy_version="live-u000")
    trainer = _Trainer()
    state = train_live_update(
        u, inputs, run, teacher_tok=object(), pool=_Pool(), trainer=trainer,
        run_dir=tmp_path, policy_version="live-u000", max_new_tokens=4096,
    )

    assert len(seen) == 4
    assert len(trainer.steps) == 1
    assert state["adapter_hash"] == "adapter-next"
    assert u.reached("TRAINED")
    # The batch-derived share reached the loss, not the replay default.
    assert kw["max_trace_share"] == pytest.approx(0.5)
    assert kw["selection_policy"] == "tau2-live-episode-balanced"
    assert kw["n_samples_per_prefix"] == 1

    # Every persisted score carries the binding RunState.validate() checks.
    scores = read_jsonl(u.scores_path)
    assert len(scores) == 4
    by_key = {r["key"]: r for r in scores}
    for row in inputs.capture_rows:
        assert by_key[row["key"]]["fingerprint"] == row["score_fingerprint"]
        assert by_key[row["key"]]["semantic_history_hash"] == (
            row["semantic_history_hash"]
        )
    # And the update validates as a whole -- the resume-time backstop.
    u.validate()


def test_train_reuses_paid_scores_but_drops_stale_ones(tmp_path, monkeypatch):
    """A live turn key can recur inside one update after a discard-and-
    resample, where identical action bytes follow a *different* history. That
    paid score was bought for a trajectory that no longer exists."""
    fresh, fresh_hist = _row("ep-1", 0)
    keep, keep_hist = _row("ep-2", 0)
    u = _sampled_update(tmp_path, [(fresh, fresh_hist), (keep, keep_hist)])
    run = RunState(tmp_path, n_updates=1)

    # One cached score matches its history; one was bought under another.
    atomic_write_jsonl(u.scores_path, [
        {
            "key": keep["key"],
            "teacher_token_bytes_b64": [base64.b64encode(b"A").decode()],
            "teacher_logprobs": [-0.1],
            "n_prefix_tokens": 3,
            "n_trailing_dropped": 0,
            "fingerprint": keep["score_fingerprint"],
            "semantic_history_hash": keep["semantic_history_hash"],
        },
        {
            "key": fresh["key"],
            "teacher_token_bytes_b64": [base64.b64encode(b"A").decode()],
            "teacher_logprobs": [-0.1],
            "n_prefix_tokens": 3,
            "n_trailing_dropped": 0,
            "fingerprint": "stale",
            "semantic_history_hash": "a-history-that-no-longer-exists",
        },
    ])

    seen, kw = [], {}
    _fake_score(monkeypatch, seen)
    _fake_chunk_opd(monkeypatch, kw)

    inputs = load_live_update_inputs(u, policy_version="live-u000")
    train_live_update(
        u, inputs, run, teacher_tok=object(), pool=_Pool(), trainer=_Trainer(),
        run_dir=tmp_path, policy_version="live-u000", max_new_tokens=4096,
    )

    # The stale one was re-bought; the provable one was reused. Paying twice is
    # the cheap failure.
    assert seen == [fresh["key"]]
    rows = {r["key"]: r for r in read_jsonl(u.scores_path)}
    assert rows[fresh["key"]]["fingerprint"] == fresh["score_fingerprint"]
    assert rows[fresh["key"]]["semantic_history_hash"] == (
        fresh["semantic_history_hash"]
    )


def test_train_is_idempotent_across_a_resume(tmp_path, monkeypatch):
    """The optimizer step is the one stage that must never run twice for one
    batch: a second step would double this update's contribution."""
    u = _sampled_update(tmp_path, [_row("ep-1", 0), _row("ep-2", 0)])
    run = RunState(tmp_path, n_updates=1)
    seen, kw = [], {}
    _fake_score(monkeypatch, seen)
    _fake_chunk_opd(monkeypatch, kw)

    inputs = load_live_update_inputs(u, policy_version="live-u000")
    trainer = _Trainer()
    for _ in range(2):
        train_live_update(
            u, inputs, run, teacher_tok=object(), pool=_Pool(),
            trainer=trainer, run_dir=tmp_path, policy_version="live-u000",
            max_new_tokens=4096,
        )
    assert len(trainer.steps) == 1


# --- adapter-hash propagation ----------------------------------------------


def _checkpoint(path, adapter_hash="adapter-u000"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "state.json").write_text(json.dumps({
        "update_index": 0, "policy_version": "live-u000",
        "adapter_hash": adapter_hash, "parent_policy_hash": "p",
        "rng_state": "r", "scheduler_state": {}, "reload_verified": True,
    }))
    return path


class _Args:
    """The argparse namespace `refresh_live_policy` mutates."""

    def __init__(self):
        self.api_base = "http://student"
        self.initial_served_name = "a-sft-new"
        self.served_name = "a-sft-new"
        self.student_model = "a-sft-new"
        self.adapter_hash = "sft-parent-hash"
        self.probe_prompt_ids = [1, 2, 3]
        self.probe_logprobs = [-0.1]
        self.reload_url = "http://reload"


def _patch_refresh(monkeypatch):
    import vektori_trace.tau2.reopd_refresh as rr
    monkeypatch.setattr(
        rr, "refresh_policy",
        lambda *a, **kw: {"probe_logprobs": [-0.2], "max_logprob_delta": 0.4},
    )


def test_refresh_advances_the_adapter_hash(tmp_path, monkeypatch):
    """The hash is stamped on every archived episode. Leaving it at the parent
    after repointing the endpoint makes update 1 sample the new adapter while
    claiming the SFT one -- and every episode in the batch would agree, so
    nothing downstream catches it."""
    from vektori_trace.tau2.live_train import refresh_live_policy

    _patch_refresh(monkeypatch)
    cp = _checkpoint(tmp_path / "update-000" / "checkpoint")
    args = _Args()

    refresh_live_policy(args, 1, cp, run_dir=tmp_path)

    assert args.adapter_hash == "adapter-u000"
    assert args.student_model == "a-sft-new-u000"
    assert args.served_name == "a-sft-new-u000"
    assert args.probe_logprobs == [-0.2]


def test_refresh_refuses_a_checkpoint_with_no_hash(tmp_path, monkeypatch):
    from vektori_trace.tau2.live_train import refresh_live_policy

    _patch_refresh(monkeypatch)
    cp = tmp_path / "update-000" / "checkpoint"
    cp.mkdir(parents=True)
    (cp / "state.json").write_text(json.dumps({"update_index": 0}))

    with pytest.raises(LiveTrainError, match="records no adapter_hash"):
        refresh_live_policy(_Args(), 1, cp, run_dir=tmp_path)


def test_refresh_refuses_an_unreadable_checkpoint(tmp_path, monkeypatch):
    from vektori_trace.tau2.live_train import refresh_live_policy

    _patch_refresh(monkeypatch)
    with pytest.raises(LiveTrainError, match="cannot read"):
        refresh_live_policy(_Args(), 1, tmp_path / "nope", run_dir=tmp_path)
