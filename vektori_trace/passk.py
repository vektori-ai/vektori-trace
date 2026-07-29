"""Step C — pass@k support measurement (PLAN.md).

Unbiased estimator from one sample of n rollouts:
    pass@k = 1 − C(n−c, k) / C(n, k)

Two-stage escalation: n=8 everywhere; escalate to n=32 only for tasks at 0/8.
Never pool stage-1 and stage-2 into one estimate — report per stratum.
Aggregation unit is (capability, model). Luck control: only-passes-at-k>8 quarantined.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .validity import run_trial

K_VALUES = (1, 4, 8, 16, 32)
STAGE1_N = 8
STAGE2_N = 32

Stratum = Literal["stage1", "stage2"]


def _comb(n: int, k: int) -> float:
    """C(n, k); 0 when undefined (k < 0 or k > n)."""
    if k < 0 or k > n:
        return 0.0
    return float(math.comb(n, k))


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased pass@k estimator. None when k > n (undefined) or n == 0."""
    if n <= 0 or k <= 0 or k > n:
        return None
    if c < 0 or c > n:
        raise ValueError(f"c must be in [0, n], got c={c}, n={n}")
    # 1 − C(n−c, k) / C(n, k)
    denom = _comb(n, k)
    if denom == 0:
        return None
    return 1.0 - (_comb(n - c, k) / denom)


@dataclass
class RolloutOutcome:
    task: str
    passed: bool
    stratum: Stratum = "stage1"


@dataclass
class TaskPassK:
    task: str
    n: int
    c: int
    stratum: Stratum
    curves: dict[int, float | None] = field(default_factory=dict)
    luck_quarantine: bool = False  # only passes appear to need k > 8

    @classmethod
    def from_counts(
        cls,
        task: str,
        n: int,
        c: int,
        *,
        stratum: Stratum,
        k_values: Iterable[int] = K_VALUES,
    ) -> TaskPassK:
        # `luck_quarantine` is deliberately NOT set here: within a single sample
        # of n, c > 0 implies pass@1 = c/n > 0, so "passes only at k > 8" is not
        # expressible from one stratum. Luck is a cross-stratum fact — see
        # `luck_quarantined_tasks`.
        return cls(
            task=task,
            n=n,
            c=c,
            stratum=stratum,
            curves={k: pass_at_k(n, c, k) for k in k_values},
        )


@dataclass
class CapabilityPassK:
    capability: str
    model: str
    stratum: Stratum
    n_tasks: int
    mean_curves: dict[int, float | None]
    task_ids: list[str]
    # Luck-quarantined tasks are held out of the mean, and named so the exclusion
    # is auditable rather than an unexplained drop in N.
    luck_excluded: list[str] = field(default_factory=list)


def compute_task_passk(
    outcomes: list[RolloutOutcome],
    *,
    k_values: Iterable[int] = K_VALUES,
) -> dict[tuple[str, Stratum], TaskPassK]:
    """Group outcomes by (task, stratum); never merge strata."""
    buckets: dict[tuple[str, Stratum], list[RolloutOutcome]] = {}
    for o in outcomes:
        buckets.setdefault((o.task, o.stratum), []).append(o)
    out: dict[tuple[str, Stratum], TaskPassK] = {}
    for (task, stratum), rows in buckets.items():
        c = sum(1 for r in rows if r.passed)
        out[(task, stratum)] = TaskPassK.from_counts(
            task, len(rows), c, stratum=stratum, k_values=k_values
        )
    return out


def tasks_needing_escalation(stage1: Mapping[str, TaskPassK]) -> list[str]:
    """Escalate to n=32 only for tasks at 0/8 (PLAN.md two-stage design)."""
    return sorted(t for t, pr in stage1.items() if pr.n > 0 and pr.c == 0)


def luck_quarantined_tasks(
    stage1: Mapping[str, TaskPassK],
    stage2: Mapping[str, TaskPassK],
) -> list[str]:
    """Tasks whose only passes appear under escalation — 0/8 at stage 1, c > 0 at
    stage 2.

    PLAN.md: "any task whose only passes occur at k>8 is quarantined for review
    before it may set a routing decision." That is not a within-sample property —
    one sample of n gives pass@1 = c/n, which is > 0 whenever c > 0. It is the
    comparison between the n=8 stratum and the n=32 stratum. Escalation only ever
    happens for 0/8 tasks, so any stage-2 pass is by construction a pass the
    stage-1 sample never saw.
    """
    out: list[str] = []
    for task, s2 in stage2.items():
        if s2.c <= 0:
            continue
        s1 = stage1.get(task)
        if s1 is None or (s1.n > 0 and s1.c == 0):
            out.append(task)
    return sorted(out)


def mark_luck_quarantine(
    stage1: Mapping[str, TaskPassK],
    stage2: Mapping[str, TaskPassK],
) -> list[str]:
    """Set `luck_quarantine` on both strata of every luck task; return the ids."""
    tasks = luck_quarantined_tasks(stage1, stage2)
    for task in tasks:
        for block in (stage1, stage2):
            pr = block.get(task)
            if pr is not None:
                pr.luck_quarantine = True
    return tasks


@dataclass
class LuckResolution:
    """Outcome of the review PLAN.md requires before a lucky task may route.

    PLAN.md: "its passing patch is diffed against the gold patch, and an
    independent second sample of n=32 must reproduce the sign of the decision."
    Both are required — the diff is a human/automated judgement that the pass was
    real work rather than a reward hack, the re-sample is what shows the support
    was not one lucky rollout.
    """

    task: str
    resample_n: int
    resample_c: int
    patch_matches_gold: bool | None = None  # None = not yet inspected
    note: str = ""

    @property
    def reproduced(self) -> bool:
        """Did the independent sample reproduce the sign of the decision?"""
        return self.resample_n > 0 and self.resample_c > 0

    @property
    def resolved(self) -> bool:
        """Cleared to set a routing decision."""
        return self.reproduced and self.patch_matches_gold is True


def resolve_luck_quarantine(
    stage2: Mapping[str, TaskPassK],
    resolutions: Mapping[str, LuckResolution],
) -> dict[str, Any]:
    """Clear `luck_quarantine` on tasks whose review reproduced the decision.

    Runs between `passk` and `route`. A task with no resolution stays flagged —
    silence is not a clearance. Returns a report of what was cleared, what stayed
    held, and why, so the quarantine ledger is auditable rather than implicit.
    """
    cleared: list[str] = []
    held: dict[str, str] = {}
    for task, pr in stage2.items():
        if not pr.luck_quarantine:
            continue
        res = resolutions.get(task)
        if res is None:
            held[task] = "no independent re-sample recorded"
            continue
        if not res.reproduced:
            held[task] = (
                f"re-sample did not reproduce the pass ({res.resample_c}/{res.resample_n})"
            )
            continue
        if res.patch_matches_gold is None:
            held[task] = "patch-vs-gold diff not inspected"
            continue
        if res.patch_matches_gold is False:
            held[task] = "passing patch does not match gold — suspected reward hack"
            continue
        pr.luck_quarantine = False
        cleared.append(task)
    return {
        "cleared": sorted(cleared),
        "held": dict(sorted(held.items())),
        "n_cleared": len(cleared),
        "n_held": len(held),
    }


def aggregate_by_capability(
    task_curves: Mapping[str, TaskPassK],
    task_to_capability: Mapping[str, str],
    *,
    model: str,
    stratum: Stratum,
    k_values: Iterable[int] = K_VALUES,
) -> dict[str, CapabilityPassK]:
    """Aggregation unit is (capability, model). N printed beside every rate.

    Luck-quarantined tasks are excluded from the mean. PLAN.md quarantines them
    precisely so they cannot set a decision before an independent n=32 re-sample;
    a capability mean that still contains them lets one lucky rollout drive the
    per-capability decision through the back door. `n_tasks` (the reported N) is
    the count after exclusion, and the excluded ids are listed.
    """
    groups: dict[str, list[TaskPassK]] = {}
    excluded: dict[str, list[str]] = {}
    for task, pr in task_curves.items():
        if pr.stratum != stratum:
            continue
        cap = task_to_capability.get(task)
        if not cap:
            continue
        if pr.luck_quarantine:
            excluded.setdefault(cap, []).append(task)
            groups.setdefault(cap, [])
            continue
        groups.setdefault(cap, []).append(pr)

    ks = list(k_values)
    result: dict[str, CapabilityPassK] = {}
    for cap, rows in groups.items():
        mean_curves: dict[int, float | None] = {}
        for k in ks:
            vals = [r.curves.get(k) for r in rows if r.curves.get(k) is not None]
            mean_curves[k] = (sum(vals) / len(vals)) if vals else None  # type: ignore[arg-type]
        result[cap] = CapabilityPassK(
            capability=cap,
            model=model,
            stratum=stratum,
            n_tasks=len(rows),
            mean_curves=mean_curves,
            task_ids=[r.task for r in rows],
            luck_excluded=sorted(excluded.get(cap, [])),
        )
    return result


def pooled_estimate_is_biased(
    stage1: Mapping[str, TaskPassK],
    stage2: Mapping[str, TaskPassK],
) -> bool:
    """Synthetic-test helper: naive pooling of stage1+stage2 differs from per-stratum."""
    # Build naive pooled n,c per task then compare pass@1 to stage1-only.
    for task, s1 in stage1.items():
        s2 = stage2.get(task)
        if s2 is None:
            continue
        pooled_n, pooled_c = s1.n + s2.n, s1.c + s2.c
        pooled = pass_at_k(pooled_n, pooled_c, min(8, pooled_n))
        separate = s1.curves.get(min(8, s1.n))
        if pooled is not None and separate is not None and abs(pooled - separate) > 1e-12:
            return True
    return False


def measure_passk_stage(
    task_dirs: list[Path],
    agent: str,
    model: str,
    jobs_dir: Path,
    n: int,
    *,
    stratum: Stratum,
    api_base: str | None = None,
    model_info: dict[str, Any] | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    infra_failures: dict[str, int] | None = None,
    max_workers: int = 1,
) -> list[RolloutOutcome]:
    """Drive n harbor trials per task. Infra failures (passed is None) excluded
    from the denominator (V0_PLAN.md rule) but counted into `infra_failures`, so
    a task that produced no gradeable rollout at all is reportable rather than
    silently absent from the sweep.

    `max_workers > 1` runs trials concurrently. The sweep is ~1,300 containerised
    rollouts of minutes each (PLAN.md Stage 2); serially that is days, which does
    not fit the "nothing expensive precedes the gate" economics. Rollouts are
    independent — each gets its own container and its own job dir — so the only
    shared state is the result list, which is assembled after the fact rather
    than appended to from threads.
    """
    jobs = [(task_dir, i) for task_dir in task_dirs for i in range(n)]

    def _one(job: tuple[Path, int]) -> tuple[str, bool | None]:
        task_dir, i = job
        trial = run_trial(
            task_dir,
            agent=agent,
            # Per-rollout job dir: harbor picks the newest reward by mtime, and
            # concurrent trials sharing one dir race for it.
            jobs_dir=jobs_dir / f"{task_dir.name}-{i}" if max_workers > 1 else jobs_dir,
            model=model,
            api_base=api_base,
            model_info=model_info,
            agent_kwargs=agent_kwargs,
        )
        return task_dir.name, trial.passed

    if max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            outcomes = list(pool.map(_one, jobs))
    else:
        outcomes = [_one(job) for job in jobs]

    results: list[RolloutOutcome] = []
    for name, passed in outcomes:
        if passed is None:
            if infra_failures is not None:
                infra_failures[name] = infra_failures.get(name, 0) + 1
            continue
        results.append(RolloutOutcome(task=name, passed=passed, stratum=stratum))
    return results


def two_stage_sweep(
    task_dirs: list[Path],
    agent: str,
    model: str,
    jobs_dir: Path,
    *,
    stage1_n: int = STAGE1_N,
    stage2_n: int = STAGE2_N,
    api_base: str | None = None,
    model_info: dict[str, Any] | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    task_to_capability: Mapping[str, str] | None = None,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Stage 1 n=8 all tasks; stage 2 n=32 only for 0/8. Returns separate strata."""
    infra: dict[str, int] = {}
    stage1_outcomes = measure_passk_stage(
        task_dirs,
        agent,
        model,
        jobs_dir / "stage1",
        stage1_n,
        stratum="stage1",
        api_base=api_base,
        model_info=model_info,
        agent_kwargs=agent_kwargs,
        infra_failures=infra,
        max_workers=max_workers,
    )
    stage1 = {
        task: pr
        for (task, stratum), pr in compute_task_passk(stage1_outcomes).items()
        if stratum == "stage1"
    }
    escalate_ids = set(tasks_needing_escalation(stage1))
    escalate_dirs = [d for d in task_dirs if d.name in escalate_ids]
    stage2_outcomes = measure_passk_stage(
        escalate_dirs,
        agent,
        model,
        jobs_dir / "stage2",
        stage2_n,
        stratum="stage2",
        api_base=api_base,
        model_info=model_info,
        agent_kwargs=agent_kwargs,
        infra_failures=infra,
        max_workers=max_workers,
    )
    stage2 = {
        task: pr
        for (task, stratum), pr in compute_task_passk(stage2_outcomes).items()
        if stratum == "stage2"
    }
    luck = mark_luck_quarantine(stage1, stage2)

    report: dict[str, Any] = {
        "model": model,
        "agent": agent,
        "stage1_n": stage1_n,
        "stage2_n": stage2_n,
        "stage1": {t: _task_to_dict(pr) for t, pr in stage1.items()},
        "stage2": {t: _task_to_dict(pr) for t, pr in stage2.items()},
        "escalated": sorted(escalate_ids),
        "luck_quarantine": luck,
        # AC #2: every task carries a support classification, and tasks that
        # produced no gradeable rollout at all are named rather than missing.
        "support": {
            d.name: classify_support(stage1.get(d.name), stage2.get(d.name))
            for d in task_dirs
        },
        "infra_failures": dict(sorted(infra.items())),
        "no_gradeable_rollouts": sorted(
            d.name for d in task_dirs if d.name not in stage1 and d.name not in stage2
        ),
    }

    # PLAN.md's aggregation unit is (capability, model) — per-task curves are the
    # observation, not the reporting unit. N accompanies every rate.
    if task_to_capability:
        report["by_capability"] = {
            stratum: {
                cap: {
                    "capability": agg.capability,
                    "model": agg.model,
                    "stratum": agg.stratum,
                    "N": agg.n_tasks,
                    "mean_curves": {str(k): v for k, v in agg.mean_curves.items()},
                    "task_ids": agg.task_ids,
                    "luck_excluded": agg.luck_excluded,
                }
                for cap, agg in aggregate_by_capability(
                    block, task_to_capability, model=model, stratum=stratum
                ).items()
            }
            for stratum, block in (("stage1", stage1), ("stage2", stage2))
        }
    return report


def classify_support(
    stage1: TaskPassK | None,
    stage2: TaskPassK | None,
) -> str:
    """`in_support` | `outside_support` | `undetermined` | `no_rollouts`.

    `outside_support` requires the n=32 escalation to have actually run: 0/8 on
    its own is underpowered, not evidence of absent support.
    """
    if stage1 is None and stage2 is None:
        return "no_rollouts"
    if (stage1 is not None and stage1.c > 0) or (stage2 is not None and stage2.c > 0):
        return "in_support"
    if stage2 is not None and stage2.n > 0 and stage2.c == 0:
        return "outside_support"
    return "undetermined"


def _task_to_dict(pr: TaskPassK) -> dict[str, Any]:
    return {
        "task": pr.task,
        "n": pr.n,
        "c": pr.c,
        "stratum": pr.stratum,
        "curves": {str(k): v for k, v in pr.curves.items()},
        "luck_quarantine": pr.luck_quarantine,
    }


__all__ = [
    "K_VALUES",
    "STAGE1_N",
    "STAGE2_N",
    "CapabilityPassK",
    "LuckResolution",
    "RolloutOutcome",
    "TaskPassK",
    "aggregate_by_capability",
    "classify_support",
    "compute_task_passk",
    "luck_quarantined_tasks",
    "mark_luck_quarantine",
    "measure_passk_stage",
    "pass_at_k",
    "pooled_estimate_is_biased",
    "resolve_luck_quarantine",
    "tasks_needing_escalation",
    "two_stage_sweep",
]
