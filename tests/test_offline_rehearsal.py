"""The offline rehearsal harness, as a permanent gate.

`scripts/tau2_offline_rehearsal.py` runs archived actions through the real
parser, projector and chunk-advantage code with a deterministic fake teacher.
This exercises it on a corpus covering every shape the pilot produced, so the
harness itself cannot rot between runs.

The fake teacher is what keeps this free. It also makes the rehearsal honest
about its limits: it proves the path executes and that chunk identity survives
a resume, not that any particular advantage is correct.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import random
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "tau2_offline_rehearsal",
    Path(__file__).parent.parent / "scripts" / "tau2_offline_rehearsal.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

TC = ('<tool_call>{"name": "get_order_details", "arguments": '
      '{"order_id": "#W1"}}</tool_call>')


def _action(key, raw, rng):
    toks, i = [], 0
    while i < len(raw):
        n = rng.choice([2, 3, 4, 5])
        toks.append(raw[i:i + n].encode())
        i += n
    assert b"".join(toks) == raw.encode()
    return {
        "key": key,
        "action_bytes_b64": base64.b64encode(raw.encode()).decode(),
        "action_token_bytes_b64": [base64.b64encode(t).decode() for t in toks],
        "action_token_ids": list(range(len(toks))),
        # Unequal logprobs: the only regime that separates the chunk rule
        # from the per-token one.
        "behavior_logprobs": [-(0.1 + rng.random() * 2.5) for _ in toks],
    }


@pytest.fixture
def corpus(tmp_path):
    rng = random.Random(7)
    rows = [
        _action("ep1@0", f"<think>Check status first.</think>{TC}", rng),
        _action("ep1@1", f"<think>I need the order details.{TC}", rng),
        _action("ep2@0", "<think>Simple.</think>Your order shipped.", rng),
        _action("ep2@1", f"Let me look that up.{TC}", rng),
        _action("ep3@0", f"<think>Need both records.{TC}{TC}", rng),
        _action("ep3@1", "<think>reasoning that simply stops", rng),
    ]
    p = tmp_path / "actions.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


class TestRehearsal:
    def test_runs_clean(self, corpus):
        rep = _MOD.rehearse(corpus)
        assert rep["failures"] == []
        assert rep["n_actions"] == 6
        assert rep["parse_refused"] == 0

    def test_recovers_the_pilot_failure_form(self, corpus):
        """The unclosed-think turns are supervised, not discarded."""
        rep = _MOD.rehearse(corpus)
        assert rep["implicit_boundary"] >= 2
        assert rep["reasoning_recovered"] >= 4

    def test_accounting_is_total(self, corpus):
        rep = _MOD.rehearse(corpus)
        assert rep["supervised_tokens"] > 0
        assert rep["excluded_tokens"] > 0
        assert sum(rep["excluded_by_reason"].values()) == rep["excluded_tokens"]

    def test_tool_serialization_is_the_bulk_of_exclusions(self, corpus):
        """v1 scope: tool JSON is conditioned on, never credited."""
        rep = _MOD.rehearse(corpus)
        assert rep["excluded_by_reason"].get("tool_serialization", 0) > 0

    def test_versions_are_recorded(self, corpus):
        rep = _MOD.rehearse(corpus)
        assert rep["parser_version"] == "v3"
        assert rep["projection_version"] == "v4"
        assert rep["score_algorithm"] == "chunk-v2"

    def test_declares_the_teacher_is_fake(self, corpus):
        """The report must never be mistaken for real advantages."""
        rep = _MOD.rehearse(corpus)
        assert "FAKE" in rep["teacher"]
        assert "not real" in rep["teacher"]

    def test_deterministic(self, corpus):
        a = _MOD.rehearse(corpus)
        b = _MOD.rehearse(corpus)
        assert a == b


class TestFakeTeacher:
    def test_logprobs_are_finite_and_negative(self):
        t = _MOD.FakeTeacher()
        lp = t.score_ids([1, 2], list(range(50)))
        assert len(lp) == 50
        assert all(-4.0 <= v < 0.0 for v in lp)

    def test_logprobs_are_unequal(self):
        """Equal logprobs would hide the very defect this rehearses."""
        t = _MOD.FakeTeacher()
        lp = t.score_ids([], list(range(20)))
        assert len(set(lp)) > 1

    def test_deterministic(self):
        assert _MOD.FakeTeacher().score_ids([], [1, 2, 3]) == \
            _MOD.FakeTeacher().score_ids([], [1, 2, 3])
