"""End-to-end through train_live_update, on the PROJECTED path.

Requirement: the live driver must never reach `score_replay_batch`. These
tests drive the real `train_live_update`, not `live_score.py` in isolation.
"""

from __future__ import annotations

import base64
import json

import pytest

from vektori_trace.tau2.live_train import load_live_update_inputs, train_live_update
from vektori_trace.tau2.live_turns import live_score_fingerprint
from vektori_trace.tau2.reopd_state import (
    RunState, UpdateDir, atomic_write_json, atomic_write_jsonl, read_jsonl,
)

TOOLS = [{"type": "function", "function": {
    "name": "get_order_details",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
}}]


def _row(ep, turn, raw):
    from vektori_trace.tau2.live_episode import semantic_hash
    hist = [
        {"role": "system", "content": "You are a retail agent.", "tools": TOOLS},
        {"role": "user", "content": f"{ep} turn {turn}"},
    ]
    toks = [bytes([b]) for b in raw.encode()]
    r = {
        "prefix_id": f"{ep}@{turn}", "sample_index": 0, "key": f"{ep}@{turn}#0",
        "action_bytes_b64": base64.b64encode(raw.encode()).decode(),
        "action_token_bytes_b64": [base64.b64encode(t).decode() for t in toks],
        "action_token_ids": list(range(len(toks))),
        "behavior_logprobs": [-0.5] * len(toks),
        "prompt_token_ids": [1, 2, 3],
        "policy_version": "live-u000", "finish_reason": "stop",
        "episode_id": ep, "task_id": "57", "turn_index": turn,
        "semantic_history_hash": semantic_hash(hist),
        "teacher_context_hash": "t1",
    }
    r["score_fingerprint"] = live_score_fingerprint(r)
    return r, hist


class _Pool:
    def __init__(self):
        self.calls = 0

    def score_ids(self, prompt_ids, tokens):
        self.calls += 1
        return [-0.4] * len(tokens)


class _Trainer:
    def __init__(self):
        self.batches = []

    def step(self, batch):
        self.batches.append(batch)
        return {"loss": 0.42, "grad_norm": 0.9, "n_examples": len(batch.keys)}

    def checkpoint(self, path, *, update_index, policy_version):
        path.mkdir(parents=True, exist_ok=True)
        st = {"update_index": update_index, "policy_version": policy_version,
              "adapter_hash": "adapter-projected", "parent_policy_hash": "p",
              "rng_state": "r", "scheduler_state": {}, "reload_verified": True}
        (path / "state.json").write_text(json.dumps(st))
        (path / "adapter_config.json").write_text("{}")
        (path / "optimizer.pt").write_bytes(b"")
        (path / "adapter_model.safetensors").write_bytes(b"")
        return st


@pytest.fixture(scope="module")
def tokenizer():
    from vektori_trace.vocab_bridge import load_tokenizer
    return load_tokenizer("deepseek-ai/DeepSeek-V4-Flash-0731")


def _sampled(tmp_path, pairs):
    u = UpdateDir(tmp_path, 0)
    u.path.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(u.actions_path, [r for r, _ in pairs])
    atomic_write_json(u.path / "rendered.json",
                      {r["prefix_id"]: h for r, h in pairs})
    u.mark("PLANNED", {})
    u.mark("SAMPLED", {"actions": len(pairs)})
    return u


RAW_A = "<think>I should check the order status.</think>Let me look that up."
RAW_B = "<think>Now I will confirm the details.</think>Here is what I found."


def test_end_to_end_uses_projection_and_never_the_replay_scorer(
    tmp_path, tokenizer, monkeypatch
):
    """The whole point: driving the real driver must not touch
    `score_replay_batch`."""
    import vektori_trace.replay_score as rs

    def _boom(*a, **k):
        raise AssertionError("live path called score_replay_batch")

    monkeypatch.setattr(rs, "score_replay_batch", _boom)

    pairs = [_row("ep-1", 0, RAW_A), _row("ep-2", 0, RAW_B),
             _row("ep-3", 0, RAW_A), _row("ep-4", 0, RAW_B)]
    u = _sampled(tmp_path, pairs)
    run = RunState(tmp_path, n_updates=1)
    inputs = load_live_update_inputs(u, policy_version="live-u000")
    trainer, pool = _Trainer(), _Pool()

    state = train_live_update(
        u, inputs, run, teacher_tok=tokenizer, pool=pool, trainer=trainer,
        run_dir=tmp_path, policy_version="live-u000", max_new_tokens=4096,
    )

    assert state["adapter_hash"] == "adapter-projected"
    assert pool.calls == 4, "each action scored once through the projection"
    assert len(trainer.batches) == 1
    batch = trainer.batches[0]
    assert batch.global_supervised_tokens > 0
    assert batch.spread_report["projection"].startswith("semantic")


def test_markup_tokens_carry_zero_weight_in_the_real_batch(tmp_path, tokenizer):
    pairs = [_row("ep-1", 0, RAW_A), _row("ep-2", 0, RAW_B),
             _row("ep-3", 0, RAW_A), _row("ep-4", 0, RAW_B)]
    u = _sampled(tmp_path, pairs)
    run = RunState(tmp_path, n_updates=1)
    inputs = load_live_update_inputs(u, policy_version="live-u000")
    trainer = _Trainer()
    train_live_update(
        u, inputs, run, teacher_tok=tokenizer, pool=_Pool(), trainer=trainer,
        run_dir=tmp_path, policy_version="live-u000", max_new_tokens=4096,
    )
    ta = trainer.batches[0].advantages[0]
    raw = RAW_A
    # `<think>` occupies bytes 0..6; none of those may be supervised.
    for i in range(len("<think>")):
        assert ta.supervised_mask[i] is False, f"markup token {i} supervised"
        assert ta.advantages[i] == 0.0


def test_projected_scores_persist_and_resume_without_rescoring(
    tmp_path, tokenizer
):
    pairs = [_row("ep-1", 0, RAW_A), _row("ep-2", 0, RAW_B),
             _row("ep-3", 0, RAW_A), _row("ep-4", 0, RAW_B)]
    u = _sampled(tmp_path, pairs)
    run = RunState(tmp_path, n_updates=1)
    inputs = load_live_update_inputs(u, policy_version="live-u000")

    p1 = _Pool()
    train_live_update(
        u, inputs, run, teacher_tok=tokenizer, pool=p1, trainer=_Trainer(),
        run_dir=tmp_path, policy_version="live-u000", max_new_tokens=4096,
    )
    assert p1.calls == 4

    rows = read_jsonl(u.scores_path)
    assert len(rows) == 4
    assert all(r["projection"] == "semantic" for r in rows)
    assert all("teacher_logprob_by_index" in r for r in rows)
    # Fingerprint binding survives, so RunState.validate() still governs.
    by_key = {r["key"]: r for r in rows}
    for row, _ in pairs:
        assert by_key[row["key"]]["fingerprint"] == row["score_fingerprint"]

    # Second pass: TRAINED short-circuits, and nothing is re-scored.
    p2 = _Pool()
    train_live_update(
        u, inputs, run, teacher_tok=tokenizer, pool=p2, trainer=_Trainer(),
        run_dir=tmp_path, policy_version="live-u000", max_new_tokens=4096,
    )
    assert p2.calls == 0, "a resume must not re-buy teacher scores"
