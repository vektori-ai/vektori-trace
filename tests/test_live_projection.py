"""Semantic projection: which student tokens may carry teacher credit."""

from __future__ import annotations

import pytest

from vektori_trace.tau2.live_projection import (
    EXCLUDE_OUTSIDE,
    EXCLUDE_STRADDLE,
    EXCLUDE_TOOL,
    ProjectionError,
    project_action,
)


def _toks(*parts: str) -> list[bytes]:
    return [p.encode() for p in parts]


def test_every_token_is_accounted_for():
    """No silent drops: the caller can assert the totals add up."""
    raw = "<think>checking</think>Here you go.<|im_end|>"
    toks = _toks("<think>", "checking", "</think>", "Here you go.", "<|im_end|>")
    p = project_action(raw, toks)
    assert p.n_supervised + p.n_excluded == len(toks)


def test_reasoning_and_content_are_supervised_markup_is_not():
    raw = "<think>checking</think>Here you go."
    toks = _toks("<think>", "checking", "</think>", "Here you go.")
    p = project_action(raw, toks)
    assert p.supervised == {1: "reasoning", 3: "content"}
    assert p.excluded[0] == EXCLUDE_OUTSIDE   # <think>
    assert p.excluded[2] == EXCLUDE_OUTSIDE   # </think>


def test_tool_serialization_is_excluded_with_its_own_reason():
    """`<tool_call>` scored -55 in the real batch: DeepSeek renders tool calls
    as a DSML block and never emits this wrapper, so the score judges notation,
    not the decision."""
    raw = '<think>look it up</think><tool_call>\n{"name": "get_order"}\n</tool_call>'
    toks = _toks("<think>", "look it up", "</think>",
                 "<tool_call>", '\n{"name": "get_order"}\n', "</tool_call>")
    p = project_action(raw, toks)
    assert p.supervised == {1: "reasoning"}
    for i in (3, 4, 5):
        assert p.excluded[i] == EXCLUDE_TOOL


def test_token_straddling_a_payload_boundary_is_dropped_whole():
    """Half a token cannot carry half an advantage."""
    raw = "<think>abc</think>tail"
    toks = _toks("<think>ab", "c</think>", "tail")
    p = project_action(raw, toks)
    assert p.excluded[0] == EXCLUDE_STRADDLE
    assert p.excluded[1] == EXCLUDE_STRADDLE
    assert p.supervised[2] == "content"


def test_unterminated_think_has_no_reasoning_payload():
    """The update-1 regression. split_generation finds no closed block, so
    there is no reasoning payload to transfer credit onto -- and the projection
    says so rather than inventing one."""
    raw = '<think>reasoning<tool_call>\n{"name": "get_order"}\n</tool_call>'
    toks = _toks("<think>", "reasoning", "<tool_call>",
                 '\n{"name": "get_order"}\n', "</tool_call>")
    p = project_action(raw, toks)
    assert p.supervised == {}
    assert p.report()["has_reasoning"] is False


def test_refuses_tokens_that_do_not_reconstruct_the_action():
    with pytest.raises(ProjectionError, match="do not reconstruct"):
        project_action("<think>a</think>b", _toks("something", "else"))


def test_report_states_retained_fraction_and_reasons():
    raw = '<think>why</think>ok<tool_call>\n{"name": "get_order"}\n</tool_call>'
    toks = _toks("<think>", "why", "</think>", "ok",
                 "<tool_call>", '\n{"name": "get_order"}\n', "</tool_call>")
    rep = project_action(raw, toks).report()
    assert rep["n_tokens"] == 7
    assert rep["supervised_by_kind"] == {"reasoning": 1, "content": 1}
    assert rep["excluded_by_reason"][EXCLUDE_TOOL] == 3
    assert 0.0 < rep["retained_fraction"] < 1.0
