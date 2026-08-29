"""Projection scope: only shared semantic payload bytes may carry credit.

Qwen serializes tool calls as Hermes JSON; DeepSeek renders DSML. They share no
bytes, so no correspondence exists to transfer a score across, and v1
deliberately supervises reasoning and visible content only. This file makes
that scope explicit and measured rather than assumed.
"""

from __future__ import annotations

import json

import pytest

from vektori_trace.tau2.live_projection import (
    EXCLUDE_MARKUP,
    EXCLUDE_OUTSIDE,
    EXCLUDE_STRADDLE,
    EXCLUDE_TOOL,
    PROJECTION_VERSION,
    project_action,
)

MARKUP = ("<think>", "</think>", "<tool_call>", "</tool_call>",
          "<|im_end|>", "<|endoftext|>")


def _toks(*pieces):
    return [p.encode("utf-8") for p in pieces]


class TestNoMarkupCarriesWeight:
    def test_closed_think_with_tool_call(self):
        raw = ('<think>reason</think>Visible.'
               '<tool_call>{"name": "f", "arguments": {}}</tool_call>')
        toks = _toks("<think>", "reason", "</think>", "Visible.",
                     "<tool_call>", '{"name": "f", "arguments": {}}',
                     "</tool_call>")
        p = project_action(raw, toks)
        for i, kind in p.supervised.items():
            assert kind in ("reasoning", "content")
            assert toks[i].decode() not in MARKUP

    def test_tool_json_is_never_supervised(self):
        raw = ('<think>r</think><tool_call>'
               '{"name": "get_order_details", "arguments": {"id": "W1"}}'
               '</tool_call>')
        payload = '{"name": "get_order_details", "arguments": {"id": "W1"}}'
        toks = _toks("<think>", "r", "</think>", "<tool_call>", payload,
                     "</tool_call>")
        p = project_action(raw, toks)
        assert p.excluded[4] == EXCLUDE_TOOL
        assert 4 not in p.supervised

    def test_im_end_is_stripped_before_projection(self):
        """`<|im_end|>` never reaches the projector.

        `live_train`/`tau2_live_opd_modal` strip the trailing special before
        calling `project_action`, because `split_generation` would otherwise
        fold it into visible content -- where it would be a literal substring
        of the raw bytes and could be scored as if the model had said it.
        """
        from vektori_trace.tau2.live_agent import split_generation

        raw_with = "<think>r</think>Answer.<|im_end|>"
        assert split_generation(raw_with)[1] == "Answer.<|im_end|>"

        raw = raw_with
        for special in ("<|im_end|>", "<|endoftext|>"):
            if raw.endswith(special):
                raw = raw[: -len(special)]
        toks = _toks("<think>", "r", "</think>", "Answer.")
        p = project_action(raw, toks)
        assert p.supervised[3] == "content"
        for i, kind in p.supervised.items():
            assert "<|im_end|>" not in toks[i].decode()

    def test_implicit_boundary_form_excludes_markup_too(self):
        """v2 parser shape: unclosed think, no closer to exclude."""
        raw = '<think>reason<tool_call>{"name": "f"}</tool_call>'
        toks = _toks("<think>", "reason", "<tool_call>", '{"name": "f"}',
                     "</tool_call>")
        p = project_action(raw, toks)
        assert p.supervised == {1: "reasoning"}
        for i in (0, 2, 3, 4):
            assert i not in p.supervised


class TestAccountingIsComplete:
    """Every token is either supervised or excluded with a stated reason."""

    @pytest.mark.parametrize("raw,pieces", [
        ("<think>r</think>C.", ("<think>", "r", "</think>", "C.")),
        ('<think>r</think><tool_call>{"name": "f"}</tool_call>',
         ("<think>", "r", "</think>", "<tool_call>", '{"name": "f"}',
          "</tool_call>")),
        ('<think>r<tool_call>{"name": "f"}</tool_call>',
         ("<think>", "r", "<tool_call>", '{"name": "f"}', "</tool_call>")),
        ("Plain answer.", ("Plain answer.",)),
    ], ids=["closed", "closed+tool", "implicit", "no-think"])
    def test_partition_is_total_and_disjoint(self, raw, pieces):
        toks = _toks(*pieces)
        p = project_action(raw, toks)
        sup, exc = set(p.supervised), set(p.excluded)
        assert sup & exc == set(), "a token cannot be both"
        assert sup | exc == set(range(len(toks))), "every token accounted for"
        rep = p.report()
        assert rep["n_supervised"] + rep["n_excluded"] == len(toks)

    def test_every_exclusion_has_a_known_reason(self):
        raw = '<think>r</think>C.<tool_call>{"name": "f"}</tool_call>'
        toks = _toks("<think>", "r", "</think>", "C.", "<tool_call>",
                     '{"name": "f"}', "</tool_call>")
        p = project_action(raw, toks)
        known = {EXCLUDE_MARKUP, EXCLUDE_TOOL, EXCLUDE_STRADDLE,
                 EXCLUDE_OUTSIDE}
        assert set(p.excluded.values()) <= known

    def test_report_quantifies_boundary_loss(self):
        """Straddlers are counted, not silently dropped."""
        # a token spanning the end of reasoning and the closer
        raw = "<think>ab</think>C."
        toks = _toks("<think>", "a", "b</think>", "C.")
        p = project_action(raw, toks)
        rep = p.report()
        assert "excluded_by_reason" in rep
        assert rep["n_excluded"] >= 1
        # retained_fraction is the measured coverage, reported every run
        assert 0.0 <= rep["retained_fraction"] <= 1.0


def test_projection_version_pinned():
    """Bound into score fingerprints; a change must invalidate caches.

    v2 (2026-08-29) additionally excludes the whitespace run hugging each
    payload boundary -- see test_projection_boundary_whitespace.py.
    """
    assert PROJECTION_VERSION == "v4"
