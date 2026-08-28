"""Token classification for OPD masking decisions."""

from __future__ import annotations

from vektori_trace.tau2.live_token_classes import (
    TokenClass,
    class_report,
    classify_action,
)


def _toks(*parts: str) -> list[bytes]:
    return [p.encode() for p in parts]


def test_markup_tokens_are_isolated_from_payload():
    toks = _toks("<think>", "hello", "</think>", "answer")
    assert classify_action(toks) == [
        TokenClass.MARKUP, TokenClass.REASONING,
        TokenClass.MARKUP, TokenClass.CONTENT,
    ]


def test_tool_json_is_its_own_class():
    toks = _toks("<tool_call>", '{"name": "get_order"}', "</tool_call>")
    assert classify_action(toks) == [
        TokenClass.MARKUP, TokenClass.TOOL_JSON, TokenClass.MARKUP,
    ]


def test_unterminated_think_still_classifies_as_reasoning():
    """The exact update-1 regression: <think> opened, never closed, then a
    tool call. A malformed action must still be analysable -- silently
    skipping it would hide the very turns that need explaining."""
    toks = _toks("<think>", "checking the order", "<tool_call>", '{"a":1}',
                 "</tool_call>")
    assert classify_action(toks) == [
        TokenClass.MARKUP, TokenClass.REASONING,
        TokenClass.MARKUP, TokenClass.TOOL_JSON, TokenClass.MARKUP,
    ]


def test_token_straddling_markup_is_not_masked():
    """A token covering `</think>` AND real text carries content. Masking it
    wholesale deletes supervision, and the run still looks healthy."""
    assert classify_action([b"</think>Now"]) != [TokenClass.MARKUP]


def test_report_separates_markup_from_semantics():
    rows = [
        (TokenClass.MARKUP, -3.36, "</think>"),
        (TokenClass.MARKUP, -2.77, "</think>"),
        (TokenClass.REASONING, +0.5, "order"),
        (TokenClass.REASONING, -0.2, "the"),
        (TokenClass.TOOL_JSON, -55.289, '"'),
    ]
    rep = class_report(rows)
    assert rep["n_supervised_tokens"] == 5
    assert rep["by_class"][TokenClass.MARKUP]["n_negative"] == 2
    assert rep["by_class"][TokenClass.REASONING]["n_positive"] == 1
    # The -55 outlier must be attributable to a class, not lost in a global bin.
    assert rep["by_class"][TokenClass.TOOL_JSON]["min"] == -55.289
    assert rep["markup_share"] == 0.4
