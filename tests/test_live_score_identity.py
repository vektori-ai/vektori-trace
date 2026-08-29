"""A paid score is valid only for the algorithms that produced it.

The 2026-08-29 parser repair changes how identical raw bytes split into
reasoning/content/tools. A cached score bought under the old splitter is a
score for a different action, so the fingerprint must move when any of the
algorithm identities move -- otherwise a resume silently reuses it.
"""

from __future__ import annotations

import json

import pytest

from vektori_trace.tau2.live_score import SCORE_ALGORITHM
from vektori_trace.tau2.live_turns import (
    LiveTurnError,
    live_score_fingerprint,
    score_row_provenance,
)


def _row(**over):
    r = dict(
        key="ep1@0",
        policy_version="p0",
        semantic_history_hash="h0",
        teacher_context_hash="t0",
        action_bytes_b64="YWJj",
        prompt_token_ids=[1, 2, 3],
        episode_id="ep1",
        turn_index=0,
    )
    r.update(over)
    return r


class TestFingerprintBindsAlgorithms:
    def test_thinking_mode_changes_the_fingerprint(self):
        a = live_score_fingerprint(_row(thinking_mode="thinking"))
        b = live_score_fingerprint(_row(thinking_mode="non-thinking"))
        assert a != b

    def test_default_thinking_mode_is_stable(self):
        assert live_score_fingerprint(_row()) == live_score_fingerprint(
            _row(thinking_mode="thinking"))

    def test_tokenizer_and_teacher_identity_bind(self):
        base = live_score_fingerprint(_row())
        assert live_score_fingerprint(_row(student_tokenizer="qwen3")) != base
        assert live_score_fingerprint(_row(teacher_model="dsv4")) != base

    def test_parser_version_is_bound(self, monkeypatch):
        base = live_score_fingerprint(_row())
        import vektori_trace.tau2.live_agent as la
        monkeypatch.setattr(la, "PARSER_VERSION", "v-other")
        assert live_score_fingerprint(_row()) != base

    def test_projection_version_is_bound(self, monkeypatch):
        base = live_score_fingerprint(_row())
        import vektori_trace.tau2.live_projection as lp
        monkeypatch.setattr(lp, "PROJECTION_VERSION", "v-other")
        assert live_score_fingerprint(_row()) != base

    def test_score_algorithm_is_bound(self, monkeypatch):
        base = live_score_fingerprint(_row())
        import vektori_trace.tau2.live_score as ls
        monkeypatch.setattr(ls, "SCORE_ALGORITHM", "flat-v1")
        assert live_score_fingerprint(_row()) != base

    def test_history_still_binds(self):
        """The original property must survive the additions."""
        assert live_score_fingerprint(_row(semantic_history_hash="h1")) != \
            live_score_fingerprint(_row(semantic_history_hash="h2"))


class TestProvenance:
    def test_records_readable_algorithm_identity(self):
        class S:
            key = "ep1@0"
        prov = score_row_provenance(S(), _row())
        assert prov["score_algorithm"] == SCORE_ALGORITHM
        assert prov["parser_version"] == "v2"
        assert prov["projection_version"] == "v2"
        assert prov["thinking_mode"] == "thinking"
        assert prov["fingerprint"] == live_score_fingerprint(_row())

    def test_refuses_mismatched_score(self):
        class S:
            key = "other@0"
        with pytest.raises(LiveTurnError, match="does not belong"):
            score_row_provenance(S(), _row())


class TestCacheReuseRules:
    """The read filter in `live_train.score_live_batch`."""

    @staticmethod
    def _filter(rows, by_key):
        """Mirror of the production filter, exercised directly."""
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

    def test_flat_row_is_not_a_cache_hit(self):
        rows = [{"key": "a", "projection": "semantic",
                 "teacher_logprob_by_index": {"0": -1.0}}]
        keep, _ = self._filter(rows, {})
        assert keep == {}

    def test_wrong_algorithm_is_not_a_cache_hit(self):
        rows = [{"key": "a", "projection": "semantic",
                 "score_algorithm": "flat-v1", "chunks": []}]
        keep, _ = self._filter(rows, {})
        assert keep == {}

    def test_fingerprint_mismatch_is_not_a_cache_hit(self):
        rows = [{"key": "a", "projection": "semantic",
                 "score_algorithm": SCORE_ALGORITHM, "chunks": [],
                 "fingerprint": "old"}]
        keep, _ = self._filter(rows, {"a": {"score_fingerprint": "new"}})
        assert keep == {}

    def test_matching_row_is_reused(self):
        rows = [{"key": "a", "projection": "semantic",
                 "score_algorithm": SCORE_ALGORITHM, "chunks": [],
                 "fingerprint": "fp"}]
        keep, _ = self._filter(rows, {"a": {"score_fingerprint": "fp"}})
        assert set(keep) == {"a"}

    def test_duplicate_keys_detected_and_last_wins(self):
        rows = [
            {"key": "a", "projection": "semantic",
             "score_algorithm": SCORE_ALGORITHM, "chunks": [], "n": 1},
            {"key": "a", "projection": "semantic",
             "score_algorithm": SCORE_ALGORITHM, "chunks": [], "n": 2},
        ]
        keep, dup = self._filter(rows, {})
        assert dup == {"a"}
        assert keep["a"]["n"] == 2


def test_atomic_replace_leaves_one_row_per_key(tmp_path):
    from vektori_trace.tau2.reopd_state import atomic_write_jsonl, read_jsonl

    p = tmp_path / "scores.jsonl"
    atomic_write_jsonl(p, [{"key": "a", "v": 1}, {"key": "b", "v": 1}])
    rows = [r for r in read_jsonl(p) if r.get("key") != "a"]
    rows.append({"key": "a", "v": 2})
    atomic_write_jsonl(p, rows)

    got = read_jsonl(p)
    keys = [r["key"] for r in got]
    assert len(keys) == len(set(keys)), "a rescore must not duplicate a key"
    assert {r["key"]: r["v"] for r in got} == {"a": 2, "b": 1}
