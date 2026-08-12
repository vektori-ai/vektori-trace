"""Step C — pass@k support measurement (PLAN.md).

Unbiased estimator from one sample of n rollouts:
    pass@k = 1 − C(n−c, k) / C(n, k)

Two-stage escalation: n=8 everywhere; escalate to n=32 only for tasks at 0/8.
Never pool stage-1 and stage-2 into one estimate — report per stratum.
Aggregation unit is (capability, model). Luck control: only-passes-at-k>8 quarantined.
"""

from __future__ import annotations

import json
import logging
import math
import threading as _threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .validity import run_trial

_log = logging.getLogger(__name__)

K_VALUES = (1, 4, 8, 16, 32)
STAGE1_N = 8
STAGE2_N = 32

# One line per rollout, flushed as it completes. Separate from the report, which
# is only written once the sweep finishes: a sweep killed at rollout 27 of 32
# still leaves 27 usable records here.
PASSK_LOG_FILENAME = "passk_log.jsonl"

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
    # Everything below is evidence, not estimator input. Without it a sweep
    # reports a rate and throws away the run that produced it: which directory
    # holds the trajectory, how long it took, how far the agent actually got.
    rollout_index: int | None = None
    reward: float | None = None
    jobs_dir: Path | None = None
    elapsed_sec: float | None = None
    started_at: float | None = None
    n_turns: int | None = None
    n_tool_calls: int | None = None


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


def _trajectory_shape(job_dir: Path) -> tuple[int | None, int | None]:
    """(n_turns, n_tool_calls) from a harbor job dir's ATIF trajectory.

    (None, None) when there is no parseable trajectory — which is itself worth
    recording, since a rollout that produced no trajectory failed differently
    from one that produced a trajectory ending in a wrong patch.
    """
    try:
        from ..mining.atif import TrajectoryParseError, parse_job_trajectory
    except ImportError:  # pragma: no cover - harbor extra not installed
        return None, None
    try:
        turns = parse_job_trajectory(job_dir)
    except (TrajectoryParseError, FileNotFoundError, OSError):
        return None, None
    return len(turns), sum(len(t.tool_calls or []) for t in turns)


def _outcome_record(outcome: RolloutOutcome) -> dict[str, Any]:
    return {
        "task": outcome.task,
        "stratum": outcome.stratum,
        "rollout_index": outcome.rollout_index,
        "passed": outcome.passed,
        "reward": outcome.reward,
        "jobs_dir": str(outcome.jobs_dir) if outcome.jobs_dir else None,
        "started_at": outcome.started_at,
        "elapsed_sec": outcome.elapsed_sec,
        "n_turns": outcome.n_turns,
        "n_tool_calls": outcome.n_tool_calls,
    }


def _append_log(log_path: Path, record: dict[str, Any]) -> None:
    """One JSON line, flushed. A killed sweep keeps everything up to the kill."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


#: Wall-clock ceiling for a single harbor rollout. `validity.run_trial`'s own
#: default is 30 min, which a 40-step agentic trajectory on a slow model does not
#: fit -- the first Qwen3-8B sweep lost every rollout to it. Overridable per run
#: via `passk --timeout-sec`.
DEFAULT_TRIAL_TIMEOUT_SEC = 5400


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
    log_path: Path | None = None,
    timeout_sec: int = DEFAULT_TRIAL_TIMEOUT_SEC,
) -> list[RolloutOutcome]:
    """Drive n harbor trials per task. Infra failures (passed is None) excluded
    from the denominator (V0_PLAN.md rule) but counted into `infra_failures`, so
    a task that produced no gradeable rollout at all is reportable rather than
    silently absent from the sweep.

    Every rollout gets **its own job dir**, at any `max_workers`. This used to be
    conditional on `max_workers > 1`, on the reasoning that only concurrent
    trials race for harbor's newest-by-mtime reward. The scalar did survive
    serially — but all n rollouts of a task then wrote into one directory and
    overwrote each other's trajectories, so a serial sweep kept the rate and
    destroyed the evidence behind it. `rollout.collect_rollouts` had it right.

    `max_workers > 1` runs trials concurrently. The full sweep is ~1,300
    containerised rollouts of minutes each (PLAN.md Stage 2); serially that is
    days. Rollouts are independent, so the only shared state is the result list,
    assembled after the fact rather than appended to from threads.

    `log_path` receives one JSON line per rollout as it completes.
    """
    jobs = [(task_dir, i) for task_dir in task_dirs for i in range(n)]
    log_lock = _threading.Lock()

    def _one(job: tuple[Path, int]) -> RolloutOutcome | tuple[str, None]:
        task_dir, i = job
        rollout_jobs_dir = jobs_dir / f"{task_dir.name}-{i}"
        # `run_trial` catches TimeoutExpired itself and scores the rollout as a
        # loss (a timeout is a model failure -- "never finished in budget" --
        # not an infra problem), reporting it back on `TrialResult.timed_out`.
        # This function used to carry its own `except TimeoutExpired` block that
        # marked the rollout ungradeable instead; it could never fire, and the
        # two policies contradicted each other. `timed_out` was therefore
        # unconditionally false in every log line ever written.
        trial = run_trial(
            task_dir,
            agent=agent,
            jobs_dir=rollout_jobs_dir,
            model=model,
            timeout_sec=timeout_sec,
            api_base=api_base,
            model_info=model_info,
            agent_kwargs=agent_kwargs,
        )
        timed_out = trial.timed_out
        if timed_out:
            _log.warning(
                "rollout %s[%d] exceeded timeout_sec=%d -- scored as a loss",
                task_dir.name,
                i,
                timeout_sec,
            )
        n_turns, n_tool_calls = _trajectory_shape(trial.jobs_dir)
        outcome = RolloutOutcome(
            task=task_dir.name,
            passed=bool(trial.passed),
            stratum=stratum,
            rollout_index=i,
            reward=trial.reward,
            jobs_dir=trial.jobs_dir,
            elapsed_sec=trial.elapsed_sec,
            started_at=trial.started_at,
            n_turns=n_turns,
            n_tool_calls=n_tool_calls,
        )
        if log_path is not None:
            record = _outcome_record(outcome)
            # `passed` is None for an infra failure, which the outcome's bool
            # would flatten to False. The log keeps the distinction the
            # denominator rule depends on.
            record["passed"] = trial.passed
            record["infra_failure"] = trial.passed is None
            # Always present, so a reader never has to infer absence-means-false.
            record["timed_out"] = timed_out
            # Why a verdict was withheld. Without it, `infra_failure: true` is
            # indistinguishable between "harbor never started" and "the verifier
            # ran but disclaimed its own reward".
            record["parse_status"] = trial.parse_status
            with log_lock:
                _append_log(log_path, record)
        if trial.passed is None:
            return task_dir.name, None
        return outcome

    if max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            outcomes = list(pool.map(_one, jobs))
    else:
        outcomes = [_one(job) for job in jobs]

    results: list[RolloutOutcome] = []
    for item in outcomes:
        if isinstance(item, tuple):
            name, _ = item
            if infra_failures is not None:
                infra_failures[name] = infra_failures.get(name, 0) + 1
            continue
        results.append(item)
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
    escalate: bool = True,
    log_path: Path | None = None,
    timeout_sec: int = DEFAULT_TRIAL_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Stage 1 n=8 all tasks; stage 2 n=32 only for 0/8. Returns separate strata.

    `escalate=False` runs stage 1 alone. Escalation triggers on `c == 0` with no
    regard for how large stage 1 was, so a deliberately small `stage1_n` — a
    4-rollout smoke test, say — turns into a 32-rollout stage 2 the moment the
    model fails, which for an untested model is the expected outcome. Opting out
    is how a small run stays small.
    """
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
        log_path=log_path,
        timeout_sec=timeout_sec,
    )
    stage1 = {
        task: pr
        for (task, stratum), pr in compute_task_passk(stage1_outcomes).items()
        if stratum == "stage1"
    }
    escalate_ids = set(tasks_needing_escalation(stage1)) if escalate else set()
    escalate_dirs = [d for d in task_dirs if d.name in escalate_ids]
    if escalate_dirs:
        _log.info(
            "escalating %d task(s) to stage 2 at n=%d: %s",
            len(escalate_dirs),
            stage2_n,
            ", ".join(sorted(escalate_ids)),
        )
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
        log_path=log_path,
        timeout_sec=timeout_sec,
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
        # Provenance: an infra_failures count means nothing without the wall the
        # rollouts were run against.
        "timeout_sec": timeout_sec,
        "stage1": {t: _task_to_dict(pr) for t, pr in stage1.items()},
        "stage2": {t: _task_to_dict(pr) for t, pr in stage2.items()},
        "escalated": sorted(escalate_ids),
        "escalation_enabled": escalate,
        # The rollouts behind the rates: one entry per gradeable rollout, naming
        # the job dir that holds its trajectory and token captures.
        "rollouts": [
            _outcome_record(o) for o in (*stage1_outcomes, *stage2_outcomes)
        ],
        "luck_quarantine": luck,
        # AC #2: every task carries a support classification, and tasks that
        # produced no gradeable rollout at all are named rather than missing.
        "support": {
            d.name: classify_support(
                stage1.get(d.name), stage2.get(d.name), stage2_n=stage2_n
            )
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
    *,
    stage2_n: int = STAGE2_N,
) -> str:
    """`in_support` | `outside_support` | `undetermined` | `no_rollouts`.

    `outside_support` requires the n=32 escalation to have actually run: 0/8 on
    its own is underpowered, not evidence of absent support.

    "Actually run" means the escalation produced (nearly) the sample it asked
    for, not merely one gradeable rollout. Testing `stage2.n > 0` let 0/2 —
    thirty of thirty-two rollouts lost to infra failures — be reported as
    evidence that a task is unsolvable, which is exactly the underpowered
    conclusion this function exists to prevent.
    """
    if stage1 is None and stage2 is None:
        return "no_rollouts"
    if (stage1 is not None and stage1.c > 0) or (stage2 is not None and stage2.c > 0):
        return "in_support"
    if stage2 is not None and stage2.c == 0 and stage2.n >= _min_stage2_n(stage2_n):
        return "outside_support"
    return "undetermined"


def _min_stage2_n(stage2_n: int) -> int:
    """How much of the escalation must survive before its zero counts as evidence.

    Three quarters: enough tolerance that a couple of infra failures don't void
    an otherwise complete escalation, tight enough that a mostly-failed stage 2
    stays `undetermined`.
    """
    return max(1, (stage2_n * 3) // 4)


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
