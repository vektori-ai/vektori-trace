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


class _Cand:
    def __init__(self, pid, step, task="t0", trace="tr0", pc=False, tokens=100):
        self.prefix_id, self.step_index, self.task = pid, step, task
        self.trace_id, self.post_compaction, self._n = trace, pc, tokens


def _render(c):
    return " ".join(["w"] * c._n)


class TestBudgetFilter:
    """Filtering must happen before selection, and must report what it changed."""

    def test_overflowing_candidates_are_dropped(self):
        from vektori_trace.replay_context import filter_candidates_by_budget

        cands = [_Cand("a@1", 1, tokens=100), _Cand("b@2", 2, tokens=50_000)]
        kept, rep = filter_candidates_by_budget(
            cands, _render, _Tok(), max_new_tokens=9216, max_model_len=40960
        )
        assert [c.prefix_id for c in kept] == ["a@1"]
        assert rep["n_overflow"] == 1
        assert rep["overflow_rate"] == 0.5
        assert rep["dropped"][0]["reason"] == "context overflow"
        assert rep["dropped"][0]["over_by"] > 0

    def test_stage_distribution_reported_before_and_after(self):
        """Overflow is not uniform: it eats late stages first.

        A filter that silently converts a late-stage stratum into an
        early-stage one leaves every count looking right, which is why the
        before/after pair is reported rather than just the survivors.
        """
        from vektori_trace.replay_context import filter_candidates_by_budget

        cands = [
            _Cand("a@1", 1, tokens=100),
            _Cand("b@2", 2, tokens=100),
            _Cand("c@60", 60, tokens=50_000),
            _Cand("d@61", 61, tokens=50_000),
        ]
        _, rep = filter_candidates_by_budget(
            cands, _render, _Tok(), max_new_tokens=9216, max_model_len=40960
        )
        assert rep["stage_distribution_before"] == {"1": 1, "2": 1, "60": 1, "61": 1}
        assert rep["stage_distribution_after"] == {"1": 1, "2": 1}

    def test_task_and_trace_eligibility_reported(self):
        from vektori_trace.replay_context import filter_candidates_by_budget

        cands = [
            _Cand("a@1", 1, task="t0", trace="tr0", tokens=100),
            _Cand("b@1", 1, task="t1", trace="tr1", tokens=50_000),
        ]
        _, rep = filter_candidates_by_budget(
            cands, _render, _Tok(), max_new_tokens=9216, max_model_len=40960
        )
        assert rep["eligible_tasks_before"] == 2
        assert rep["eligible_tasks_after"] == 1
        assert rep["eligible_traces_before"] == 2
        assert rep["eligible_traces_after"] == 1

    def test_post_compaction_counts_survive_the_filter_report(self):
        """Removing 38% may eliminate the post-compaction states entirely."""
        from vektori_trace.replay_context import filter_candidates_by_budget

        cands = [
            _Cand("a@1", 1, pc=False, tokens=100),
            _Cand("b@40", 40, pc=True, tokens=50_000),
        ]
        _, rep = filter_candidates_by_budget(
            cands, _render, _Tok(), max_new_tokens=9216, max_model_len=40960
        )
        assert rep["post_compaction_before"] == 1
        assert rep["post_compaction_after"] == 0

    def test_render_failure_is_dropped_not_raised(self):
        from vektori_trace.replay_context import filter_candidates_by_budget

        def bad_render(c):
            if c.prefix_id == "b@2":
                raise ValueError("boom")
            return _render(c)

        cands = [_Cand("a@1", 1, tokens=10), _Cand("b@2", 2, tokens=10)]
        kept, rep = filter_candidates_by_budget(
            cands, bad_render, _Tok(), max_new_tokens=100, max_model_len=40960
        )
        assert [c.prefix_id for c in kept] == ["a@1"]
        assert rep["n_render_errors"] == 1
