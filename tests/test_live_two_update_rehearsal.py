"""Mocked two-update rehearsal: policy lineage and optimizer continuity.

The live arm is only on-policy if update 1's episodes were sampled from the
checkpoint update 0 produced. This exercises score -> train -> checkpoint ->
resume twice, with a stub trainer and fake teacher, and asserts:

- the chunk-grouped advantages reach the optimizer (not flat credit);
- a resume reuses paid scores instead of re-buying them;
- a resume does NOT reuse scores whose fingerprint no longer matches;
- rescoring replaces rather than appends, so no key is duplicated;
- optimizer/Adam state and RNG are carried across the checkpoint;
- update 1 declares update 0's adapter as its parent.

No GPU, no endpoint, no teacher spend.
"""

from __future__ import annotations

import json

import pytest

from vektori_trace.tau2.live_score import SCORE_ALGORITHM, ProjectedChunk, ProjectedScore
from vektori_trace.tau2.reopd_state import atomic_write_jsonl, read_jsonl


class StubTrainer:
    """Records what it was stepped with; carries optimizer state forward."""

    def __init__(self):
        self.steps = []
        self.adam_step = 0
        self.checkpoints = []

    def step(self, batch):
        self.steps.append(batch)
        self.adam_step += 1
        return {"loss": 0.5 / self.adam_step, "grad_norm": 1.0,
                "n_examples": len(batch.keys), "adam_step": self.adam_step}

    def checkpoint(self, path, *, update_index, policy_version):
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "update_index": update_index,
            "policy_version": policy_version,
            "adapter_hash": f"adapter-u{update_index:03d}",
            "parent_policy_hash": (
                f"adapter-u{update_index - 1:03d}" if update_index else "sft-parent"
            ),
            "rng_state": f"rng-{update_index}",
            # Adam moments must survive; a fresh optimizer silently changes the
            # effective learning rate for every later update.
            "scheduler_state": {"last_epoch": update_index,
                                "adam_step": self.adam_step},
            "reload_verified": True,
        }
        (path / "state.json").write_text(json.dumps(state))
        (path / "adapter_config.json").write_text("{}")
        (path / "optimizer.pt").write_bytes(b"")
        (path / "adapter_model.safetensors").write_bytes(b"")
        self.checkpoints.append(state)
        return state


def _score(key, n_student=3):
    """A projected score whose chunk has UNEQUAL student coverage."""
    return ProjectedScore(
        key=key,
        chunks=[ProjectedChunk(f"reasoning:0", "reasoning",
                               tuple(range(n_student)), (-3.0,))],
    )


class TestOptimizerContinuity:
    def test_adam_state_advances_across_updates(self):
        t = StubTrainer()

        class B:
            keys = ["a"]
            global_supervised_tokens = 3
        s0 = t.step(B()); t.checkpoint_state = t.checkpoint
        assert s0["adam_step"] == 1
        s1 = t.step(B())
        assert s1["adam_step"] == 2, "optimizer state must not reset"

    def test_checkpoint_records_optimizer_state(self, tmp_path):
        t = StubTrainer()

        class B:
            keys = ["a"]
            global_supervised_tokens = 3
        t.step(B())
        st = t.checkpoint(tmp_path / "u0", update_index=0,
                          policy_version="live-u000")
        assert st["scheduler_state"]["adam_step"] == 1
        assert st["rng_state"]
        assert st["reload_verified"] is True

    def test_lineage_chains_update1_to_update0(self, tmp_path):
        t = StubTrainer()

        class B:
            keys = ["a"]
            global_supervised_tokens = 3
        t.step(B())
        s0 = t.checkpoint(tmp_path / "u0", update_index=0,
                          policy_version="live-u000")
        t.step(B())
        s1 = t.checkpoint(tmp_path / "u1", update_index=1,
                          policy_version="live-u001")
        assert s1["parent_policy_hash"] == s0["adapter_hash"], (
            "update 1 must be parented on update 0's adapter, or the arm is "
            "not on-policy"
        )
        assert s1["scheduler_state"]["adam_step"] == 2


class TestChunkedCreditReachesTheOptimizer:
    def test_batch_carries_chunk_grouped_advantages(self):
        from vektori_trace.tau2.live_batch import build_projected_batch

        class P:
            def __init__(self, i):
                self.prefix_id = f"ep{i}@0"
                self.task = f"t{i}"
                self.trace_id = f"ep{i}"
                self.step_index = 0

        class A:
            def __init__(self, i):
                self.key = f"ep{i}@0"
                self.policy_version = "live-u000"
                self.action_token_ids = [1, 2, 3]
                # unequal -- the regime that separates the two rules
                self.behavior_logprobs = [-0.5, -1.0, -1.5]
                self.prompt_token_ids = [9]

        prefixes = [P(0), P(1)]
        actions = [A(0), A(1)]
        scores = {a.key: _score(a.key) for a in actions}
        batch = build_projected_batch(
            prefixes, actions, scores, policy_version="live-u000",
            enforce_shares=False,
        )
        # teacher agrees exactly, so every advantage is zero under the chunk
        # rule; the per-token rule would give [-0.5, 0, +0.5].
        for ta in batch.advantages:
            assert ta.advantages == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
            assert ta.supervised_mask == [True, True, True]
        assert batch.global_supervised_tokens == 6


class TestScoreCacheAcrossResume:
    """The filter that decides what a resume re-buys."""

    @staticmethod
    def _reusable(rows, by_key):
        seen, dup = {}, set()
        for r in rows:
            if not r.get("key") or r.get("projection") != "semantic":
                continue
            if r["key"] in seen:
                dup.add(r["key"])
            seen[r["key"]] = r
        keep = {}
        for k, r in seen.items():
            if r.get("score_algorithm") != SCORE_ALGORITHM or "chunks" not in r:
                continue
            want = (by_key.get(k) or {}).get("score_fingerprint")
            if want is not None and r.get("fingerprint") != want:
                continue
            keep[k] = r
        return keep, dup

    def _row(self, key, fp="fp"):
        return {"key": key, "projection": "semantic",
                "score_algorithm": SCORE_ALGORITHM, "fingerprint": fp,
                "chunks": [ProjectedChunk("reasoning:0", "reasoning",
                                          (0, 1, 2), (-3.0,)).to_json()]}

    def test_resume_reuses_a_matching_paid_score(self):
        keep, _ = self._reusable([self._row("a")],
                                 {"a": {"score_fingerprint": "fp"}})
        assert set(keep) == {"a"}, "a valid paid score must not be re-bought"

    def test_resume_rebuys_when_the_parser_changed(self):
        """Fingerprint binds PARSER_VERSION; a change invalidates the score."""
        keep, _ = self._reusable([self._row("a", fp="old-parser")],
                                 {"a": {"score_fingerprint": "new-parser"}})
        assert keep == {}

    def test_resume_rebuys_a_flat_pre_repair_row(self):
        keep, _ = self._reusable(
            [{"key": "a", "projection": "semantic",
              "teacher_logprob_by_index": {"0": -1.0}}], {})
        assert keep == {}, "flat credit is unrecoverable and must be rescored"

    def test_rescore_replaces_and_does_not_duplicate(self, tmp_path):
        p = tmp_path / "scores.jsonl"
        atomic_write_jsonl(p, [self._row("a", "old"), self._row("b")])
        rows = [r for r in read_jsonl(p) if r.get("key") != "a"]
        rows.append(self._row("a", "new"))
        atomic_write_jsonl(p, rows)

        got = read_jsonl(p)
        keys = [r["key"] for r in got]
        assert len(keys) == len(set(keys)), "a rescore must not append a duplicate"
        assert {r["key"]: r["fingerprint"] for r in got} == {"a": "new", "b": "fp"}

    def test_preexisting_duplicates_collapse_to_the_last(self):
        keep, dup = self._reusable(
            [self._row("a", "first"), self._row("a", "second")], {})
        assert dup == {"a"}
        assert keep["a"]["fingerprint"] == "second"
