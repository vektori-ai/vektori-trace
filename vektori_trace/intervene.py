"""Step D — verifier-guided bisection localizer (PLAN.md).

Binary search over prefix length T:
1. Replay student failed trajectory to step T (resume.py).
2. Teacher continues to termination.
3. Verifier grades the result.
4. Largest T where teacher-continuation still succeeds → forking step is T+1.

Each probe sampled twice; non-monotonic fraction reported, never silently resolved.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .resume import ResumeDesyncError, ResumeResult, replay_prefix
from .schema import Turn


@dataclass
class ProbeResult:
    T: int
    recoverable: bool  # teacher continuation succeeded
    samples: list[bool] = field(default_factory=list)
    desync: bool = False
    detail: str = ""
    phase: str = "search"  # "search" | "verify"
    disagreed: bool = False  # the repeated samples did not agree
    # The prefix replay ran but nothing checkable could be compared. Not a
    # desync — there was no disagreement — but not a verified prefix either.
    unverified: bool = False


@dataclass
class BisectionResult:
    forking_step: int | None  # T+1 where T is largest recoverable prefix; None if none
    largest_recoverable_T: int | None
    probes: list[ProbeResult]
    teacher_continuations: int
    monotone: bool
    non_monotone_fraction: float
    # NB for anyone checking acceptance criterion 3, which is worded in *teacher
    # continuations*: `budget_ok` is the probe-point reading and is an invariant
    # of binary search — it cannot fail, so it is not evidence of anything. The
    # field that can fail, and the one criterion 3 actually asks about, is
    # `continuation_budget_ok` below.
    budget_ok: bool  # search probe points ≤ ceil(log2(s)) + 2
    steps: int
    dropped: bool = False  # desync hard-fail dropped the task
    drop_reason: str | None = None
    # An *unverified* prefix replay is not a desync: nothing checkable could be
    # compared, so nothing disagreed. PLAN.md hard-fails on desync only, so the
    # bisection continues rather than dropping the task — but the teacher then
    # continued from a workspace nobody checked, and a forking step located that
    # way is not execution-grounded in the sense Stage 3 claims. Recorded here,
    # distinct from `dropped`/desync, so a caller can filter such tasks out of
    # the grounding set instead of trusting them silently.
    unverified_probes: int = 0
    unverified_fraction: float = 0.0
    resume_unverified: bool = False
    # Monotonicity is measured by probes the binary search did not itself visit;
    # the search's own probe set is consistent by construction and cannot falsify
    # the assumption. `verify_probes` is how many such points were checked.
    verify_probes: int = 0
    monotonicity_violations: int = 0
    # A probe whose repeated samples disagree is recorded, never silently resolved.
    sample_disagreements: int = 0
    sample_disagreement_fraction: float = 0.0
    # PLAN.md states the budget in teacher continuations but also requires each
    # probe point to be sampled twice; the two cannot both hold. Both are reported
    # so either reading is checkable. `budget_ok` is the probe-point reading.
    budget_probe_points: int = 0
    continuation_budget_ok: bool = True


ContinueFn = Callable[[list[Turn], int], bool]
"""Teacher continuation from prefix T → True if verifier passes."""

ResumeFn = Callable[[list[Turn], int], ResumeResult]
"""Replay student prefix to T; may raise ResumeDesyncError."""


def _resolve(samples: list[bool]) -> bool:
    """Recoverability is an existence claim: one passing continuation proves the
    prefix is still recoverable, a failure only fails to prove it. So repeated
    samples resolve by OR, not by majority — with the default two samples a
    majority rule would be AND, which turns teacher sampling noise into a
    "forking step". Disagreement is recorded separately and never resolved away.
    """
    return any(samples)


def bisect_forking_step(
    turns: list[Turn],
    *,
    n_action_steps: int | None = None,
    resume: ResumeFn | None = None,
    continue_with_teacher: ContinueFn,
    samples_per_probe: int = 2,
    verify_probes: int = 2,
) -> BisectionResult:
    """Find largest T where teacher continuation succeeds; forking step = T+1.

    Action-step count defaults to number of parent assistant tool calls.

    `verify_probes` extra probe points, chosen from prefixes the search never
    visited, test the monotonicity assumption. Without them the reported
    non-monotone fraction is identically zero: after a failure at T the search
    sets `hi = T-1`, so no later probe can contradict an earlier one.
    """
    from .resume import assistant_tool_steps

    steps = n_action_steps if n_action_steps is not None else len(assistant_tool_steps(turns))
    if steps <= 0:
        return BisectionResult(
            forking_step=None,
            largest_recoverable_T=None,
            probes=[],
            teacher_continuations=0,
            monotone=True,
            non_monotone_fraction=0.0,
            budget_ok=True,
            steps=0,
        )

    budget = math.ceil(math.log2(steps)) + 2
    probes: list[ProbeResult] = []
    continuations = 0

    def probe(T: int, phase: str = "search") -> ProbeResult:
        nonlocal continuations
        unverified = False
        detail = ""
        if resume is not None:
            try:
                res = resume(turns, T)
            except ResumeDesyncError as e:
                return ProbeResult(
                    T=T, recoverable=False, desync=True, detail=str(e), phase=phase
                )
            # `replay_prefix` returns verified=False when nothing checkable could
            # be compared. That is not a desync, so the search continues — but the
            # continuation started from an unchecked workspace and the probe says so.
            if res is not None and not getattr(res, "verified", True):
                unverified = True
                detail = getattr(res, "desync_reason", None) or "prefix replay unverified"
        samples: list[bool] = []
        for _ in range(samples_per_probe):
            ok = bool(continue_with_teacher(turns, T))
            samples.append(ok)
            continuations += 1
        return ProbeResult(
            T=T,
            recoverable=_resolve(samples),
            samples=samples,
            phase=phase,
            disagreed=len(set(samples)) > 1,
            unverified=unverified,
            detail=detail,
        )

    # Binary search for largest recoverable T in [-1, steps-1].
    # T=-1 means empty prefix (teacher from scratch).
    lo, hi = -1, steps - 1
    largest: int | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        pr = probe(mid)
        probes.append(pr)
        if pr.desync:
            n_probes = len([p for p in probes if not p.desync]) + 1
            return BisectionResult(
                forking_step=None,
                largest_recoverable_T=None,
                probes=probes,
                teacher_continuations=continuations,
                monotone=True,
                non_monotone_fraction=0.0,
                # Budget is on probe points (each may be dual-sampled per PLAN.md).
                budget_ok=n_probes <= budget,
                steps=steps,
                dropped=True,
                drop_reason=pr.detail,
                budget_probe_points=budget,
                continuation_budget_ok=continuations <= budget * samples_per_probe,
                **_unverified_stats(probes),
            )
        if pr.recoverable:
            largest = mid
            lo = mid + 1
        else:
            hi = mid - 1

    n_search_probes = len([p for p in probes if not p.desync])

    # Measure monotonicity on points the search never visited. If recoverability
    # is monotone in T, every T' <= largest is recoverable and every T' > largest
    # is not; each verification probe is a chance to falsify that.
    visited = {p.T for p in probes}
    verify_points = _unvisited_verification_points(
        largest, steps, visited, verify_probes
    )
    violations = 0
    for T in verify_points:
        pr = probe(T, phase="verify")
        probes.append(pr)
        if pr.desync:
            continue
        expected = largest is not None and largest >= pr.T
        if pr.recoverable != expected:
            violations += 1

    checked = len([p for p in probes if p.phase == "verify" and not p.desync])
    monotone = violations == 0
    non_mono_frac = (violations / checked) if checked else 0.0

    graded = [p for p in probes if not p.desync]
    n_disagreed = sum(1 for p in graded if p.disagreed)
    disagree_frac = (n_disagreed / len(graded)) if graded else 0.0

    forking = (largest + 1) if largest is not None else 0
    # If even empty prefix fails, forking_step=0 (broke before any student action).
    # If full prefix still recoverable, no forking step inside the trajectory.
    if largest == steps - 1:
        forking_step: int | None = None  # never became unrecoverable
    else:
        forking_step = forking

    return BisectionResult(
        forking_step=forking_step,
        largest_recoverable_T=largest,
        probes=probes,
        teacher_continuations=continuations,
        monotone=monotone,
        non_monotone_fraction=non_mono_frac,
        budget_ok=n_search_probes <= budget,
        steps=steps,
        verify_probes=checked,
        monotonicity_violations=violations,
        sample_disagreements=n_disagreed,
        sample_disagreement_fraction=disagree_frac,
        budget_probe_points=budget,
        continuation_budget_ok=continuations <= budget * samples_per_probe,
        **_unverified_stats(probes),
    )


def _unverified_stats(probes: list[ProbeResult]) -> dict[str, Any]:
    """Unverified-prefix bookkeeping, kept out of the desync/dropped accounting."""
    graded = [p for p in probes if not p.desync]
    n_unverified = sum(1 for p in graded if p.unverified)
    return {
        "unverified_probes": n_unverified,
        "unverified_fraction": (n_unverified / len(graded)) if graded else 0.0,
        "resume_unverified": n_unverified > 0,
    }


def _unvisited_verification_points(
    largest: int | None,
    steps: int,
    visited: set[int],
    want: int,
) -> list[int]:
    """Pick up to `want` unvisited prefixes, half below `largest` and half above.

    Deterministic (midpoint of each unvisited run) so a rerun probes the same
    points and the reported fraction is reproducible.
    """
    if want <= 0:
        return []
    below = [t for t in range(0, (largest if largest is not None else 0)) if t not in visited]
    above = [
        t
        for t in range((largest + 2) if largest is not None else 0, steps)
        if t not in visited
    ]
    picks: list[int] = []
    if below:
        picks.append(below[len(below) // 2])
    if above:
        picks.append(above[len(above) // 2])
    # Spend any remaining budget on whichever side still has unvisited points.
    for pool in (below, above):
        for t in pool:
            if len(picks) >= want:
                break
            if t not in picks:
                picks.append(t)
    return sorted(picks[:want])


def make_resume_fn(sandbox: Any, **kwargs: Any) -> ResumeFn:
    """Bind a sandbox into a ResumeFn for bisect_forking_step."""

    def _resume(turns: list[Turn], T: int) -> ResumeResult:
        return replay_prefix(turns, T, sandbox, **kwargs)

    return _resume


__all__ = [
    "BisectionResult",
    "ContinueFn",
    "ProbeResult",
    "ResumeFn",
    "bisect_forking_step",
    "make_resume_fn",
]
