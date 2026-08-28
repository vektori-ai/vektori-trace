"""Projected scoring: teacher credit reaches semantics, never serialization."""

from __future__ import annotations

import pytest

from vektori_trace.tau2.live_score import (
    LiveScoreError,
    score_live_action,
)

TOOLS = [{"type": "function", "function": {
    "name": "get_order_details",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
}}]

HISTORY = [
    {"role": "system", "content": "You are a retail agent.", "tools": TOOLS},
    {"role": "user", "content": "Where is order #W123?"},
]


class _Pool:
    """Returns a deterministic logprob per position and records the call."""

    def __init__(self):
        self.calls = []

    def score_ids(self, prompt_ids, tokens):
        self.calls.append((list(prompt_ids), list(tokens)))
        return [-0.5] * len(tokens)


def _tok():
    from vektori_trace.vocab_bridge import load_tokenizer
    return load_tokenizer("deepseek-ai/DeepSeek-V4-Flash-0731")


def _bytes_for(text: str) -> list[bytes]:
    """One byte per token: a worst case for chunking, and exact."""
    return [bytes([b]) for b in text.encode("utf-8")]


@pytest.fixture(scope="module")
def tokenizer():
    return _tok()


def test_markup_and_tool_serialization_receive_no_credit(tokenizer):
    raw = ('<think>I should look up the order.</think>'
           'Checking now.'
           '<tool_call>\n{"name": "get_order_details", '
           '"arguments": {"order_id": "#W123"}}\n</tool_call>')
    toks = _bytes_for(raw)
    pool = _Pool()

    sc = score_live_action(
        key="ep@0#0", raw_text=raw, student_token_bytes=toks,
        semantic_history=HISTORY, teacher_tokenizer=tokenizer, pool=pool,
    )

    # Every supervised index must lie inside a semantic payload.
    reasoning_span = raw.index("I should look up")
    content_span = raw.index("Checking now.")
    tool_span = raw.index("<tool_call>")
    for i in sc.teacher_logprob_by_index:
        assert i < tool_span, f"token {i} is inside tool serialization"
    assert any(reasoning_span <= i < reasoning_span + 10
               for i in sc.teacher_logprob_by_index)
    # And the teacher was asked for the NATIVE render, not the raw Qwen bytes.
    assert pool.calls, "the teacher was never called"
    # `content` here abuts the DSML tool block in the native render, so a
    # DeepSeek token spans the boundary. The payload is skipped rather than
    # scored on a guessed fraction -- and says so.
    assert sc.payload_report["content"]["skipped"].startswith("teacher token")
    assert "teacher_boundary_straddle" in set(sc.excluded.values())


def test_content_is_supervised_when_no_tool_block_abuts_it(tokenizer):
    """Same shape without a trailing tool call: content now has a clean
    boundary, so it carries credit. This is the control for the skip above."""
    raw = "<think>I should look up the order.</think>Checking now."
    toks = _bytes_for(raw)
    sc = score_live_action(
        key="ep@5#0", raw_text=raw, student_token_bytes=toks,
        semantic_history=HISTORY, teacher_tokenizer=tokenizer, pool=_Pool(),
    )
    content_at = raw.index("Checking now.")
    assert any(i >= content_at for i in sc.teacher_logprob_by_index)
    assert "content" in sc.payload_report


def test_unterminated_think_yields_no_credit_at_all(tokenizer):
    """The update-1 regression: no closed reasoning block, so no payload."""
    raw = '<think>reasoning<tool_call>\n{"name": "get_order_details"}\n</tool_call>'
    pool = _Pool()
    sc = score_live_action(
        key="ep@1#0", raw_text=raw, student_token_bytes=_bytes_for(raw),
        semantic_history=HISTORY, teacher_tokenizer=tokenizer, pool=pool,
    )
    assert sc.n_supervised == 0
    assert sc.payload_report["n_payloads"] == 0
    assert not pool.calls, "no payload, so the teacher must not be billed"


def test_every_token_is_supervised_or_excluded(tokenizer):
    raw = "<think>why</think>Here you go."
    toks = _bytes_for(raw)
    sc = score_live_action(
        key="ep@2#0", raw_text=raw, student_token_bytes=toks,
        semantic_history=HISTORY, teacher_tokenizer=tokenizer, pool=_Pool(),
    )
    covered = set(sc.teacher_logprob_by_index) | set(sc.excluded)
    assert covered == set(range(len(toks)))


def test_non_finite_teacher_logprob_is_refused(tokenizer):
    class _Bad(_Pool):
        def score_ids(self, prompt_ids, tokens):
            return [float("-inf")] * len(tokens)

    raw = "<think>why</think>ok"
    with pytest.raises(LiveScoreError, match="non-finite"):
        score_live_action(
            key="ep@3#0", raw_text=raw, student_token_bytes=_bytes_for(raw),
            semantic_history=HISTORY, teacher_tokenizer=tokenizer, pool=_Bad(),
        )


def test_length_mismatch_is_refused(tokenizer):
    class _Short(_Pool):
        def score_ids(self, prompt_ids, tokens):
            return [-0.5] * (len(tokens) - 1)

    raw = "<think>why</think>ok"
    with pytest.raises(LiveScoreError, match="logprobs for"):
        score_live_action(
            key="ep@4#0", raw_text=raw, student_token_bytes=_bytes_for(raw),
            semantic_history=HISTORY, teacher_tokenizer=tokenizer, pool=_Short(),
        )
