"""`verify_episode` must accept what the parser accepts.

2026-08-29, pilot update 0: two of eight episodes were failed by
`verify_episode()` for "raw action lacks complete reasoning delimiters" while
their reasoning had been captured, non-empty and index-mapped -- 2,193 and
2,056 characters, with 3 and 1 tool calls parsed. The gate tested for a literal
`</think>`, which is PARSER_VERSION v1 logic; v2 (vLLM `92762ed`) accepts an
unclosed `<think>` bounded by a complete valid `<tool_call>`.

The archive gate and the parser must ask the SAME question, or the pipeline
throws away work it correctly parsed.
"""

from __future__ import annotations

import pathlib

import pytest

from vektori_trace.tau2 import live_episode
from vektori_trace.tau2.live_agent import _resolve_reasoning

TC = '<tool_call>{"name": "get_order_details", "arguments": {"o": "1"}}</tool_call>'


class TestGateUsesTheSharedResolver:
    def test_no_literal_closing_tag_check_remains(self):
        src = pathlib.Path(live_episode.__file__).read_text()
        assert 'b"</think>" not in raw' not in src, (
            "the literal </think> requirement is v1 logic and rejects valid "
            "implicit-boundary generations"
        )

    def test_gate_calls_resolve_reasoning(self):
        src = pathlib.Path(live_episode.__file__).read_text()
        assert "_resolve_reasoning" in src


class TestResolverAgreesWithTheGate:
    """The shapes the pilot actually produced."""

    def test_task68_shape_resolves(self):
        """Unclosed think, reasoning, then several tool calls."""
        raw = f"<think>\nOkay, let's see. The user is asking...\n{TC}{TC}{TC}"
        sp = _resolve_reasoning(raw)
        assert sp is not None
        assert sp.mode == "implicit_tool_call"
        assert sp.text.strip()

    def test_task108_shape_resolves(self):
        raw = f"<think>\nOkay, I need to help Yusuf...\nFirst, retrieve.\n{TC}"
        sp = _resolve_reasoning(raw)
        assert sp is not None and sp.mode == "implicit_tool_call"

    def test_task95_shape_does_not_resolve(self):
        """Unclosed think, NO tool call -- reasoning drifts into the answer.

        925 chars: opens <think>, reasons about the order, then addresses the
        user directly and stops. There is no non-arbitrary boundary between
        reasoning and answer, so the gate must refuse rather than guess.
        """
        raw = ("<think>\nOkay, let me check the order again to see if there's "
               "a second laptop item.\n\nLooking at order #W2905754...\n"
               "Is your Visa ending in 1234 the one you'd like to use?")
        assert "</think>" not in raw and "<tool_call>" not in raw
        assert _resolve_reasoning(raw) is None

    def test_closed_block_still_resolves(self):
        assert _resolve_reasoning("<think>reasoned</think>answer").mode == "closed"

    @pytest.mark.parametrize("raw", [
        "no think block at all",
        "<think>   </think>answer",          # empty reasoning
        f"<think>a<think>b{TC}",             # ambiguous: two openers
    ], ids=["absent", "empty", "two-openers"])
    def test_unusable_shapes_still_refused(self, raw):
        assert _resolve_reasoning(raw) is None


def test_the_gate_would_have_passed_seven_of_eight():
    """With the fix, update 0 had ONE genuine failure, not three."""
    shapes = {
        "task44": f"<think>reasoning{TC}",
        "task68": f"<think>reasoning{TC}{TC}{TC}",
        "task108": f"<think>reasoning{TC}",
        "task95": "<think>reasoning that drifts into speaking to the user.",
    }
    resolved = {k: _resolve_reasoning(v) is not None for k, v in shapes.items()}
    assert resolved == {"task44": True, "task68": True,
                        "task108": True, "task95": False}
