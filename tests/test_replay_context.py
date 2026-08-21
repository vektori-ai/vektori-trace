"""Prefix context-budget checks (§4, §11)."""

from __future__ import annotations

import pytest

from vektori_trace.replay_context import (
    ContextBudget,
    ContextBudgetError,
    assert_prefix_fits,
    measure_prefix,
    summarize_budgets,
)


class _Tok:
    """Whitespace tokenizer; token counts are word counts."""

    def __call__(self, text, add_special_tokens=False):
        self.saw_add_special = add_special_tokens
        return {"input_ids": text.split()}


def _budget(tokens, cap=9216, window=40960):
    return ContextBudget("p@1", tokens, cap, window)


class TestBudget:
    def test_fits_with_headroom(self):
        b = _budget(30_000)
        assert b.required == 39_216
        assert b.headroom == 1744
        assert b.fits

    def test_exact_fit_is_allowed(self):
        """Boundary belongs to the fitting side: nothing is truncated at ==."""
        b = _budget(40_960 - 9216)
        assert b.headroom == 0
        assert b.fits

    def test_one_over_does_not_fit(self):
        assert not _budget(40_960 - 9216 + 1).fits

    def test_cap_counts_against_the_window(self):
        """A prefix that fits alone can still not fit with the action cap.

        This is the case that makes a prefix-only check useless: the server
        needs room for both.
        """
        b = _budget(35_000)
        assert b.prefix_tokens < b.max_model_len
        assert not b.fits


class TestAssert:
    def test_refuses_overflow_with_the_overage(self):
        with pytest.raises(ContextBudgetError, match="Over by 100"):
            assert_prefix_fits(_budget(40_960 - 9216 + 100))

    def test_passes_when_it_fits(self):
        assert_prefix_fits(_budget(1000))


class TestMeasure:
    def test_uses_pinned_tokenizer_without_special_tokens(self):
        """Special tokens would count tokens the server never sees."""
        tok = _Tok()
        b = measure_prefix("p@1", "a b c d", tok, max_new_tokens=10, max_model_len=100)
        assert b.prefix_tokens == 4
        assert tok.saw_add_special is False


class TestSummary:
    def test_reports_min_headroom_and_overflow(self):
        budgets = [_budget(1000), _budget(30_000), _budget(40_000)]
        s = summarize_budgets(budgets)
        assert s["n_prefixes"] == 3
        assert s["n_overflow"] == 1
        assert s["overflow_prefix_ids"] == ["p@1"]
        assert s["max_prefix_tokens"] == 40_000
        assert s["min_headroom"] < 0

    def test_empty_is_not_an_error(self):
        assert summarize_budgets([]) == {"n_prefixes": 0}
