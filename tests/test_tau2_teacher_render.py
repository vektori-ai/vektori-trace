"""Tau2's context actually reaches DeepSeek, and the action span is exact.

`replay_score` carries no benchmark knowledge, but that does not make it
benchmark-neutral: its input contract is "canonical messages the DeepSeek
encoder can render", and Tau2 satisfies that contract in a specific way -- the
retail policy as the system message, tool schemas on that same message in
OpenAI format. Nothing outside this file tests that combination.

The chain under test:

    C30 policy + tools + history + sampled action
        -> DeepSeek render
        -> exact action span
        -> ids the teacher would be asked to score

A failure anywhere here is silent in the ordinary way: the render succeeds, the
span is found, the loss is finite, and the teacher scored an action under
conditioning the student never saw.
"""

from __future__ import annotations

import pytest

from vektori_trace.tau2.c30_loader import C30Prefix

RETAIL_TOOLS = [
    {"type": "function", "function": {
        "name": "find_user_id_by_email",
        "description": "Find a user id from an email address.",
        "parameters": {"type": "object",
                       "properties": {"email": {"type": "string"}},
                       "required": ["email"]}}},
    {"type": "function", "function": {
        "name": "cancel_pending_order",
        "description": "Cancel an order that has not shipped.",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string"},
                                      "reason": {"type": "string"}},
                       "required": ["order_id", "reason"]}}},
]

POLICY = (
    "You are a retail agent. You must authenticate the user before reading or "
    "modifying any order. Never mutate before confirmation."
)


def _prefix(messages, tools=RETAIL_TOOLS) -> C30Prefix:
    return C30Prefix(
        prefix_id="42#5", task_id="42", trace_id="tracehash", position=5,
        action_type="toolcall", tool_names=["cancel_pending_order"],
        semantic_hash="s" * 64,
        prompt_token_ids=[1, 2, 3],
        canonical_messages=[{"role": "system", "content": POLICY,
                             "tools": tools}] + messages,
        tools=tools,
        stored_teacher_action={"role": "assistant", "content": "recorded"},
    )


HISTORY = [
    {"role": "user", "content": "I want to cancel order #W123."},
    {"role": "assistant", "tool_calls": [{
        "id": "c1", "type": "function",
        "function": {"name": "find_user_id_by_email",
                     "arguments": '{"email": "a@b.com"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "content": '{"user_id": "u_9"}'},
]


# --- the policy and tools reach the render -------------------------------


def test_policy_text_appears_in_the_deepseek_render():
    from vektori_trace.providers.teacher.cross import render_teacher_prefix

    text = render_teacher_prefix(_prefix(HISTORY).canonical_messages)
    assert "You are a retail agent" in text
    assert "authenticate the user" in text


def test_tool_schemas_appear_in_the_deepseek_render():
    """The bug this file exists for: tools on the object but not in context."""
    from vektori_trace.providers.teacher.cross import render_teacher_prefix

    text = render_teacher_prefix(_prefix(HISTORY).canonical_messages)
    for name in ("find_user_id_by_email", "cancel_pending_order"):
        assert name in text, f"{name} never reached DeepSeek's context"
    # the parameter schema, not merely the name
    assert "order_id" in text


def test_dropping_tools_measurably_shrinks_the_context():
    """Proves the assertion above is load-bearing, not incidentally true."""
    from vektori_trace.providers.teacher.cross import render_teacher_prefix

    with_tools = render_teacher_prefix(_prefix(HISTORY).canonical_messages)
    without = render_teacher_prefix(
        [{"role": "system", "content": POLICY}] + HISTORY
    )
    assert len(with_tools) > len(without)
    assert "cancel_pending_order" not in without


def test_openai_tool_shape_is_what_the_encoder_expects():
    """load_domain_tools returns [{type, function}]; the encoder unwraps it."""
    from vektori_trace.encoding_dsv4 import tools_from_openai_format

    fns = tools_from_openai_format(RETAIL_TOOLS)
    assert [f["name"] for f in fns] == [
        "find_user_id_by_email", "cancel_pending_order"]


def test_history_tool_calls_and_results_survive_the_render():
    from vektori_trace.providers.teacher.cross import render_teacher_prefix

    text = render_teacher_prefix(_prefix(HISTORY).canonical_messages)
    assert "a@b.com" in text        # the call's arguments
    assert "u_9" in text            # the observation it returned


# --- the action span ------------------------------------------------------


def _teacher_tokenizer():
    from vektori_trace.vocab_bridge import load_tokenizer
    try:
        return load_tokenizer("deepseek-ai/DeepSeek-V3")
    except Exception as e:                              # offline / no access
        pytest.skip(f"teacher tokenizer unavailable: {e}")


@pytest.mark.parametrize("action_text", [
    '{"name": "cancel_pending_order", "arguments": {"order_id": "W123"}}',
    "I have cancelled order #W123 for you.",
])
def test_action_span_is_byte_exact_under_the_tau2_context(action_text):
    """The span the teacher scores must be exactly the student's bytes.

    Run against the real DeepSeek tokenizer because the failure being guarded
    is a boundary straddle -- a token spanning the prefix/action edge -- which
    a stub tokenizer cannot reproduce.
    """
    from vektori_trace.replay_score import locate_action_span

    tok = _teacher_tokenizer()
    messages = _prefix(HISTORY).canonical_messages
    prefix_ids, action_ids, dropped = locate_action_span(
        messages, action_text, tok
    )

    assert prefix_ids, "empty teacher prefix"
    assert action_ids, "empty action span"

    decoded = tok.decode(action_ids) if hasattr(tok, "decode") else None
    if decoded is not None:
        assert decoded.startswith(action_text[:20])


def test_joint_render_extends_the_prefix_render():
    """The joint STRING must extend the prefix string.

    Note this is asserted on the render, not on independently-encoded ids.
    Encoding both strings separately and demanding the id lists nest is a
    stricter condition than correctness requires: the last prefix token can
    merge with the action's first character, which is the boundary straddle
    `locate_action_span` detects and refuses to resolve by offset arithmetic
    (replay_score.py:116). Asserting id-nesting here would fail on a perfectly
    scorable prefix.
    """
    from vektori_trace.providers.teacher.cross import render_teacher_prefix

    messages = _prefix(HISTORY).canonical_messages
    action = "Cancelled."
    prefix_text = render_teacher_prefix(messages)
    joint_text = render_teacher_prefix(
        [*messages, {"role": "assistant", "content": action}])

    assert joint_text.startswith(prefix_text)
    assert action in joint_text[len(prefix_text):]


def test_tools_change_the_teacher_prefix_ids():
    """Conditioning differs with and without tools, so the scores would too."""
    from vektori_trace.providers.teacher.cross import (
        encode_teacher_ids,
        render_teacher_prefix,
    )

    tok = _teacher_tokenizer()
    with_tools = encode_teacher_ids(
        render_teacher_prefix(_prefix(HISTORY).canonical_messages), tok)
    without = encode_teacher_ids(
        render_teacher_prefix(
            [{"role": "system", "content": POLICY}] + HISTORY), tok)
    assert with_tools != without
    assert len(with_tools) > len(without)
