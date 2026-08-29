"""Wrapper-adjacent whitespace is serialization; interior whitespace is not.

Measured on the one-episode canary (2026-08-29): every action opened
`<think>\nOkay`, Qwen scored that newline ~0 (its template requires it),
DeepSeek scored it -15 to -23. In 11/11 turns the most negative supervised
token was `'\n'` at action index 1 -- a family-format disagreement that would
have been the batch's dominant gradient.

The fix is narrow on purpose. Whitespace inside the reasoning body is authored
content -- paragraph breaks, list structure, the rhythm of the argument -- and
must keep carrying credit.
"""

from __future__ import annotations

import pytest

from vektori_trace.tau2.live_projection import (
    PROJECTION_VERSION,
    _trim_boundary_whitespace,
    project_action,
)


def _toks(*pieces):
    return [p.encode("utf-8") for p in pieces]


class TestTrim:
    def test_leading_newline_trimmed(self):
        b = b"\nOkay"
        assert _trim_boundary_whitespace(b, 0, len(b)) == (1, 5)

    def test_trailing_newline_trimmed(self):
        b = b"Okay\n"
        assert _trim_boundary_whitespace(b, 0, len(b)) == (0, 4)

    def test_both_ends_trimmed(self):
        b = b"\n\n Okay \n"
        s, e = _trim_boundary_whitespace(b, 0, len(b))
        assert b[s:e] == b"Okay"

    def test_interior_whitespace_untouched(self):
        """The load-bearing case: only the ends move."""
        b = b"\nFirst para.\n\nSecond para.\n"
        s, e = _trim_boundary_whitespace(b, 0, len(b))
        assert b[s:e] == b"First para.\n\nSecond para."
        assert b"\n\n" in b[s:e], "interior blank line must survive"

    def test_all_whitespace_collapses_to_empty(self):
        b = b"\n \t\n"
        s, e = _trim_boundary_whitespace(b, 0, len(b))
        assert s == e

    def test_no_whitespace_is_a_noop(self):
        b = b"Okay"
        assert _trim_boundary_whitespace(b, 0, len(b)) == (0, 4)

    def test_respects_the_given_window(self):
        b = b"xx\nOkay\nyy"
        s, e = _trim_boundary_whitespace(b, 2, 8)
        assert b[s:e] == b"Okay"

    def test_tabs_and_spaces_trimmed(self):
        b = b"\t  Okay  \t"
        s, e = _trim_boundary_whitespace(b, 0, len(b))
        assert b[s:e] == b"Okay"

    def test_carriage_return_trimmed(self):
        b = b"\r\nOkay\r\n"
        s, e = _trim_boundary_whitespace(b, 0, len(b))
        assert b[s:e] == b"Okay"


class TestProjection:
    def test_canary_shape_excludes_the_opening_newline(self):
        """`<think>\\nOkay...` -- the exact observed form."""
        raw = "<think>\nOkay, check the order.\n</think>Done."
        toks = _toks("<think>", "\n", "Okay, check the order.", "\n",
                     "</think>", "Done.")
        p = project_action(raw, toks)
        assert 1 not in p.supervised, "opening newline must not carry credit"
        assert 3 not in p.supervised, "closing newline must not carry credit"
        assert p.supervised[2] == "reasoning"
        assert p.supervised[5] == "content"

    def test_interior_newline_still_supervised(self):
        raw = "<think>\nFirst.\n\nSecond.\n</think>Done."
        toks = _toks("<think>", "\n", "First.", "\n\n", "Second.", "\n",
                     "</think>", "Done.")
        p = project_action(raw, toks)
        assert 1 not in p.supervised          # boundary
        assert 5 not in p.supervised          # boundary
        assert p.supervised[3] == "reasoning", "interior blank line is authored"
        assert p.supervised[2] == "reasoning"
        assert p.supervised[4] == "reasoning"

    def test_implicit_boundary_form_also_trimmed(self):
        tc = '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
        raw = f"<think>\nNeed details.\n{tc}"
        toks = _toks("<think>", "\n", "Need details.", "\n", tc)
        p = project_action(raw, toks)
        assert 1 not in p.supervised
        assert 3 not in p.supervised
        assert p.supervised[2] == "reasoning"

    def test_accounting_still_total(self):
        raw = "<think>\nOkay.\n</think>Done."
        toks = _toks("<think>", "\n", "Okay.", "\n", "</think>", "Done.")
        p = project_action(raw, toks)
        assert set(p.supervised) & set(p.excluded) == set()
        assert set(p.supervised) | set(p.excluded) == set(range(len(toks)))

    def test_whitespace_only_reasoning_supervises_nothing(self):
        raw = "<think>\n \n</think>Done."
        toks = _toks("<think>", "\n \n", "</think>", "Done.")
        p = project_action(raw, toks)
        assert "reasoning" not in p.supervised.values()

    def test_raw_bytes_unchanged(self):
        """Projection decides eligibility; it never rewrites the action."""
        raw = "<think>\nOkay.\n</think>Done."
        toks = _toks("<think>", "\n", "Okay.", "\n", "</think>", "Done.")
        project_action(raw, toks)
        assert raw == "<think>\nOkay.\n</think>Done."
        assert b"".join(toks) == raw.encode("utf-8")


def test_projection_version_bumped():
    """Cache identity must change loudly: old score rows must not be reused."""
    assert PROJECTION_VERSION == "v2"


def test_fingerprint_changes_with_projection_version(monkeypatch):
    from vektori_trace.tau2.live_turns import live_score_fingerprint

    row = dict(key="k", policy_version="p", semantic_history_hash="h",
               teacher_context_hash="t", action_bytes_b64="YQ==",
               prompt_token_ids=[1], episode_id="e", turn_index=0)
    now = live_score_fingerprint(row)
    import vektori_trace.tau2.live_projection as lp
    monkeypatch.setattr(lp, "PROJECTION_VERSION", "v1")
    assert live_score_fingerprint(row) != now, (
        "a v1 score row must not pass fingerprint reuse under v2"
    )
