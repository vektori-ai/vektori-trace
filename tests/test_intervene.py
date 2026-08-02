"""Step D — bisection localizer."""

from __future__ import annotations

import math

from vektori_trace.evaluate.intervene import bisect_forking_step
from vektori_trace.schema import ToolCall, Turn


def _n_steps(n: int) -> list[Turn]:
    turns = []
    for i in range(n):
        turns.append(
            Turn(
                index=i,
                role="assistant",
                tool_calls=[ToolCall(id=str(i), name="bash", args={"cmd": f"echo {i}"})],
            )
        )
    return turns


def test_bisection_finds_planted_fork() -> None:
    # Recoverable for T <= 3; forking step = 4.
    turns = _n_steps(10)

    def cont(_turns, T: int) -> bool:
        return T <= 3

    result = bisect_forking_step(turns, continue_with_teacher=cont, samples_per_probe=2)
    assert result.largest_recoverable_T == 3
    assert result.forking_step == 4
    # A binary search over `steps+1` candidates costs floor(log2(steps+1))+1
    # probes, which is always within ceil(log2 steps)+2 — so `budget_ok` alone
    # cannot fail. Pin the actual probe count instead.
    search = [p for p in result.probes if p.phase == "search"]
    assert len(search) <= math.ceil(math.log2(10 + 1))
    assert len(search) < 10  # logarithmic, not a linear scan
    assert result.budget_ok


def test_bisection_detects_non_monotone() -> None:
    turns = _n_steps(16)
    # Recoverable at a sparse, non-monotone set: the student wanders back into a
    # recoverable state at T=7 after being unrecoverable at 3..6.
    recoverable = {0, 1, 2, 7}

    def cont(_turns, T: int) -> bool:
        return T in recoverable

    result = bisect_forking_step(
        turns, continue_with_teacher=cont, samples_per_probe=1, verify_probes=4
    )
    assert result.verify_probes > 0, "monotonicity must be probed off the search path"
    assert not result.monotone
    assert result.monotonicity_violations > 0
    assert result.non_monotone_fraction > 0.0


def test_monotone_trajectory_reports_no_violations() -> None:
    turns = _n_steps(16)
    result = bisect_forking_step(
        turns,
        continue_with_teacher=lambda _t, T: T <= 5,
        samples_per_probe=1,
        verify_probes=4,
    )
    assert result.largest_recoverable_T == 5
    assert result.monotone
    assert result.non_monotone_fraction == 0.0
    assert result.verify_probes > 0


def test_probe_sample_disagreement_is_recorded_not_resolved() -> None:
    turns = _n_steps(8)
    calls: dict[int, int] = {}

    def flaky(_turns, T: int) -> bool:
        # T == 2 is genuinely recoverable but the teacher fails it on the first
        # sample. A majority rule over two samples is AND, which would turn this
        # into a forking step.
        calls[T] = calls.get(T, 0) + 1
        if T <= 2:
            return not (T == 2 and calls[T] == 1)
        return False

    result = bisect_forking_step(
        turns, continue_with_teacher=flaky, samples_per_probe=2, verify_probes=0
    )
    assert result.largest_recoverable_T == 2
    assert result.sample_disagreements >= 1
    assert result.sample_disagreement_fraction > 0.0
    assert any(p.disagreed for p in result.probes)


def test_budget_reported_in_both_units() -> None:
    turns = _n_steps(16)
    result = bisect_forking_step(
        turns,
        continue_with_teacher=lambda _t, T: T <= 3,
        samples_per_probe=2,
        verify_probes=0,
    )
    budget = math.ceil(math.log2(16)) + 2
    assert result.budget_probe_points == budget
    search_probes = [p for p in result.probes if p.phase == "search"]
    assert len(search_probes) <= math.ceil(math.log2(16 + 1))
    assert len(search_probes) < 16  # logarithmic, not a linear scan
    assert result.budget_ok
    # The continuation count is the thing PLAN.md's criterion 3 actually states,
    # and unlike budget_ok it can fail — dual sampling plus verify probes can
    # exceed it, which is the tension the two fields exist to expose.
    assert result.teacher_continuations == len(search_probes) * 2
    assert result.continuation_budget_ok

    over = bisect_forking_step(
        turns,
        continue_with_teacher=lambda _t, T: T <= 3,
        samples_per_probe=4,
        verify_probes=4,
    )
    assert not over.continuation_budget_ok
    assert over.budget_ok


def test_empty_trajectory() -> None:
    result = bisect_forking_step([], continue_with_teacher=lambda *_: True)
    assert result.steps == 0
    assert result.forking_step is None


def _resume_result(T: int, *, verified: bool):
    from vektori_trace.evaluate.resume import ResumeResult

    return ResumeResult(
        T=T,
        steps_replayed=T + 1,
        consistent=verified,
        expected_diff="",
        actual_diff="",
        verified=verified,
        checks_run=1 if verified else 0,
        desync_reason=None if verified else "unverified: nothing checkable",
    )


def test_unverified_resume_is_recorded_and_is_not_a_desync() -> None:
    """`replay_prefix` returns verified=False when nothing checkable could be
    compared. That is not a desync — nothing disagreed — so bisection continues,
    but the teacher continued from a workspace nobody checked and the result has
    to say so, distinctly from `dropped`."""
    turns = _n_steps(8)
    result = bisect_forking_step(
        turns,
        resume=lambda _t, T: _resume_result(T, verified=False),
        continue_with_teacher=lambda _t, T: T <= 3,
        samples_per_probe=1,
        verify_probes=0,
    )
    # Bisection ran to completion — an unverified prefix is not a hard fail.
    assert result.dropped is False
    assert result.drop_reason is None
    assert result.largest_recoverable_T == 3
    assert result.forking_step == 4
    # ...and every probe is flagged unverified, distinct from desync.
    assert result.resume_unverified is True
    assert result.unverified_probes == len(result.probes)
    assert result.unverified_fraction == 1.0
    assert all(p.unverified and not p.desync for p in result.probes)


def test_verified_resume_leaves_the_unverified_counters_at_zero() -> None:
    turns = _n_steps(8)
    result = bisect_forking_step(
        turns,
        resume=lambda _t, T: _resume_result(T, verified=True),
        continue_with_teacher=lambda _t, T: T <= 3,
        samples_per_probe=1,
        verify_probes=0,
    )
    assert result.resume_unverified is False
    assert result.unverified_probes == 0
    assert result.unverified_fraction == 0.0
    assert result.forking_step == 4


def test_desync_still_drops_and_is_not_counted_as_unverified() -> None:
    from vektori_trace.evaluate.resume import ResumeDesyncError

    def boom(_turns, T: int):
        raise ResumeDesyncError("git diff mismatch at T")

    result = bisect_forking_step(
        _n_steps(8),
        resume=boom,
        continue_with_teacher=lambda _t, T: True,
        samples_per_probe=1,
        verify_probes=0,
    )
    assert result.dropped is True
    assert result.drop_reason == "git diff mismatch at T"
    assert result.unverified_probes == 0
    assert result.resume_unverified is False
