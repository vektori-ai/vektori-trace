"""Unclosed `<think>` terminated by a valid tool call (vLLM `92762ed`).

Three of eight update-0 episodes on 2026-08-29 were discarded because the model
opened `<think>`, reasoned at length, emitted valid tool calls, and never wrote
`</think>`. `finish_reason` was `stop`, so this is neither truncation nor
absent reasoning -- it is a form the upstream Qwen3 parser accepts and our
client-side splitter did not.

The rule is deliberately narrow: a *complete, valid* tool call ends an unclosed
block, spans stay disjoint, no `</think>` bytes are invented, and anything
ambiguous is refused.
"""

from __future__ import annotations

import json

import pytest

from vektori_trace.tau2.live_agent import (
    LiveCaptureError,
    PARSER_VERSION,
    _reasoning_byte_span,
    _resolve_reasoning,
    split_generation,
)

TC = ('<tool_call>{"name": "get_order_details", "arguments": '
      '{"order_id": "#W0000000"}}</tool_call>')


def _span_text(raw):
    sp = _reasoning_byte_span(raw)
    if sp is None:
        return None
    return raw.encode("utf-8")[sp[0]:sp[1]].decode("utf-8")


class TestClosedThinkUnchanged:
    def test_closed_think_still_parses(self):
        raw = f"<think>reasoning here</think>\n{TC}"
        r, c, tools = split_generation(raw)
        assert r == "reasoning here"
        assert len(tools) == 1 and tools[0]["name"] == "get_order_details"
        assert _span_text(raw) == "reasoning here"

    def test_closed_think_with_visible_content(self):
        raw = "<think>thinking</think>Here is your answer."
        r, c, tools = split_generation(raw)
        assert r == "thinking"
        assert c == "Here is your answer."
        assert tools == []

    def test_no_think_at_all(self):
        r, c, tools = split_generation(f"Just an answer.\n{TC}")
        assert r is None
        assert c == "Just an answer."
        assert len(tools) == 1


class TestImplicitBoundary:
    """The pilot's actual failure form."""

    def test_unclosed_think_then_tool_call(self):
        raw = f"<think>I need the order details first.\n{TC}"
        r, c, tools = split_generation(raw)
        assert r == "I need the order details first.\n"
        assert len(tools) == 1
        assert tools[0]["arguments"] == {"order_id": "#W0000000"}
        # no content: everything was reasoning or tool call
        assert c is None

    def test_mode_is_recorded(self):
        span = _resolve_reasoning(f"<think>abc\n{TC}")
        assert span.mode == "implicit_tool_call"
        assert _resolve_reasoning("<think>abc</think>").mode == "closed"

    def test_spans_are_disjoint(self):
        """Reasoning must not swallow the tool call."""
        raw = f"<think>reason\n{TC}"
        assert "<tool_call>" not in _span_text(raw)
        assert "get_order_details" not in _span_text(raw)

    def test_no_closing_tag_invented(self):
        raw = f"<think>reason\n{TC}"
        assert "</think>" not in _span_text(raw)
        assert "</think>" not in raw

    def test_byte_span_matches_parse(self):
        """The span the teacher scores is exactly what split_generation returned."""
        raw = f"<think>multi\nline\treasoning ünïcode\n{TC}"
        r, _, _ = split_generation(raw)
        assert _span_text(raw) == r

    def test_multiple_tool_calls_first_one_is_the_boundary(self):
        second = ('<tool_call>{"name": "get_user_details", '
                  '"arguments": {"user_id": "u1"}}</tool_call>')
        raw = f"<think>reason\n{TC}\n{second}"
        r, c, tools = split_generation(raw)
        assert r == "reason\n"
        assert [t["name"] for t in tools] == [
            "get_order_details", "get_user_details"]

    def test_text_after_tool_call_is_content(self):
        raw = f"<think>reason\n{TC}\nAnything else?"
        r, c, tools = split_generation(raw)
        assert r == "reason\n"
        assert c == "Anything else?"
        assert len(tools) == 1


class TestRefusals:
    """Ambiguity yields no reasoning span; the run then refuses the turn."""

    def test_unclosed_think_with_no_tool_call(self):
        assert _resolve_reasoning("<think>reasoning that just stops") is None

    def test_unclosed_think_with_incomplete_tool_call(self):
        raw = '<think>reason\n<tool_call>{"name": "get_order'
        assert _resolve_reasoning(raw) is None

    def test_empty_reasoning_refused(self):
        assert _resolve_reasoning(f"<think>\n  \n{TC}") is None

    def test_multiple_unclosed_openers_refused(self):
        assert _resolve_reasoning(f"<think>a\n<think>b\n{TC}") is None

    def test_multiple_openers_with_closed_block_refused(self):
        assert _resolve_reasoning("<think>a</think><think>b") is None

    def test_unpaired_closer_refused(self):
        assert _resolve_reasoning(f"</think>stray\n<think>reason\n{TC}") is None

    def test_malformed_tool_json_still_raises(self):
        raw = '<think>reason\n<tool_call>{not json}</tool_call>'
        with pytest.raises(LiveCaptureError, match="not JSON"):
            split_generation(raw)

    def test_tool_call_without_name_raises(self):
        raw = '<think>reason\n<tool_call>{"arguments": {}}</tool_call>'
        with pytest.raises(LiveCaptureError, match="no 'name'"):
            split_generation(raw)


class TestArchivedPilotFixtures:
    """Shapes taken from the three preserved 2026-08-29 failed turns.

    The archived generations are long; what is reproduced here is their
    structure -- unclosed opener, substantial reasoning, one or more valid
    Hermes calls, no closer, ending at `stop`.
    """

    @pytest.mark.parametrize(
        "reasoning,calls",
        [
            # task 44 / seed 1 -- one call after a long deliberation
            ("The user wants to cancel order #W0000000. Policy says a pending "
             "order may be cancelled, but I must confirm the status first.\n",
             [("get_order_details", {"order_id": "#W0000000"})]),
            # task 68 / seed 0 -- shorter reasoning, one call
            ("I should look up the user before modifying anything.\n",
             [("find_user_id_by_email", {"email": "a@b.com"})]),
            # task 76 / seed 0 -- two calls, failed at turn 2
            ("Need both the order and the user profile to proceed.\n",
             [("get_order_details", {"order_id": "#W1111111"}),
              ("get_user_details", {"user_id": "sara_doe_496"})]),
        ],
        ids=["task44", "task68", "task76"],
    )
    def test_archived_form_now_captures(self, reasoning, calls):
        blocks = "".join(
            '<tool_call>' + json.dumps({"name": n, "arguments": a})
            + '</tool_call>' for n, a in calls
        )
        raw = f"<think>{reasoning}{blocks}"

        r, c, tools = split_generation(raw)
        assert r == reasoning, "reasoning must be recovered, not dropped"
        assert [t["name"] for t in tools] == [n for n, _ in calls]
        assert [t["arguments"] for t in tools] == [a for _, a in calls]
        # the gate that refused these turns is a non-empty reasoning span
        assert r.strip()
        assert _span_text(raw) == r
        assert "<tool_call>" not in r


def test_parser_version_is_v4():
    """Fingerprints bind this; a bump must be deliberate."""
    assert PARSER_VERSION == "v4"


# --- implicit_eof: an unclosed <think> that ends at EOF (parser v4) ---------
#
# Observed in update 5: the model wrote its customer-facing summary inside an
# unclosed <think> and stopped. vLLM's production qwen3 parser classifies those
# bytes as reasoning_content, so scoring them in that role matches the serving
# stack. The action is trainable but NOT format-valid -- Tau2 never renders
# reasoning_content, so the customer receives nothing.

def test_implicit_eof_accepted_when_finish_is_stop():
    raw = "<think>\nI've completed all three modifications successfully."
    span = _resolve_reasoning(raw, "stop")
    assert span is not None
    assert span.mode == "implicit_eof"
    assert span.text.strip().startswith("I've completed")
    # no bytes invented: the span must be a literal substring
    assert span.text in raw


def test_implicit_eof_refused_on_length_cap():
    """A truncated action is a fragment, not a decision."""
    raw = "<think>\nWait, let me check that again. Wait, let me check"
    assert _resolve_reasoning(raw, "length") is None


def test_implicit_eof_refused_without_finish_reason():
    """Absent provenance, the shape stays ambiguous and is refused."""
    raw = "<think>\nSome authored reasoning that never closes."
    assert _resolve_reasoning(raw, None) is None


def test_implicit_eof_refused_when_empty():
    assert _resolve_reasoning("<think>\n   \n", "stop") is None


def test_implicit_eof_does_not_populate_content():
    """The bytes are scored as reasoning only; never duplicated as content."""
    raw = "<think>\nHere is your summary of the three changes."
    reasoning, content, tools = split_generation(raw, "stop")
    assert reasoning is not None
    assert content is None
    assert tools == []


def test_closed_think_still_wins_over_eof():
    raw = "<think>\nreal reasoning\n</think>\nvisible answer"
    span = _resolve_reasoning(raw, "stop")
    assert span is not None and span.mode == "closed"


def test_tool_call_boundary_still_wins_over_eof():
    raw = ('<think>\nreasoning first\n'
           '<tool_call>\n{"name": "get_order_details", "arguments": {}}\n</tool_call>')
    span = _resolve_reasoning(raw, "stop")
    assert span is not None and span.mode == "implicit_tool_call"
