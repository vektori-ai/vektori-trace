"""Step 6's first question: on the tasks a diagnosed deficit is lacking in, how
often does the candidate actually pass? Rejection sampling needs the band, not
the extremes — measured with repeated rollouts on diagnosis-selected tasks only,
never the whole mined set (that's Step 4's job, once, cheaply)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .validity import run_trial

# Below this, rejection sampling keeps nothing: an all-fail rollout group has no
# passing sample to sample from. Above the top, GRPO's (r - mean) / std is zero
# for an all-pass group too — both ends carry no training signal, only the
# middle does (V0_PLAN.md Step 6).
PASSRATE_MIN = 0.10
PASSRATE_MAX = 0.40
# Plan says 8-16 rollouts per task; 8 is the floor that still resolves the band
# at single-task granularity without paying 16x for diagnostic power nothing
# downstream needs yet.
DEFAULT_ROLLOUTS = 8


@dataclass
class RolloutResult:
    task: str
    passed: bool


@dataclass
class PassRate:
    task: str
    passed: int
    n: int

    @property
    def rate(self) -> float | None:
        return (self.passed / self.n) if self.n else None

    def in_band(self, band: tuple[float, float] = (PASSRATE_MIN, PASSRATE_MAX)) -> bool:
        lo, hi = band
        return self.rate is not None and lo <= self.rate <= hi

    def majority_pass(self) -> bool:
        """Pre-declared McNemar binarization for A0–A4 (V0 Step 6).

        `measure_pass_rates` yields 8–16 Bernoulli trials per task; McNemar needs
        one bit per task per arm. Rule: strict majority of judgeable rollouts
        passed (`passed * 2 > n`). Ties count as fail. Declared before any arm
        runs — not chosen after seeing p-values.
        """
        return self.n > 0 and self.passed * 2 > self.n


def compute_pass_rates(results: list[RolloutResult]) -> dict[str, PassRate]:
    """Group rollout outcomes by task into a rate + N. A task with zero
    rollouts recorded simply doesn't appear — there's nothing to divide."""
    by_task: dict[str, list[RolloutResult]] = {}
    for r in results:
        by_task.setdefault(r.task, []).append(r)
    return {
        task: PassRate(task=task, passed=sum(1 for r in rs if r.passed), n=len(rs))
        for task, rs in by_task.items()
    }


def measure_pass_rates(
    task_dirs: list[Path],
    agent: str,
    model: str,
    jobs_dir: Path,
    rollouts: int = DEFAULT_ROLLOUTS,
    *,
    api_base: str | None = None,
    model_info: dict[str, Any] | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    extra_instruction_path: Path | None = None,
) -> dict[str, PassRate]:
    """Drive `rollouts` real `harbor run` trials per task and aggregate. The I/O
    boundary `compute_pass_rates` is deliberately kept clear of: this function is
    the only piece of the module that touches Docker or an LLM API, so it's the
    only piece not covered by an offline test."""
    results: list[RolloutResult] = []
    for task_dir in task_dirs:
        for _ in range(rollouts):
            trial = run_trial(
                task_dir,
                agent=agent,
                jobs_dir=jobs_dir,
                model=model,
                api_base=api_base,
                model_info=model_info,
                agent_kwargs=agent_kwargs,
                extra_instruction_path=extra_instruction_path,
            )
            if trial.passed is None:
                continue  # unjudgeable trial; not a pass, not a fail, not counted
            results.append(RolloutResult(task=task_dir.name, passed=trial.passed))
    return compute_pass_rates(results)


__all__ = [
    "DEFAULT_ROLLOUTS",
    "PASSRATE_MAX",
    "PASSRATE_MIN",
    "PassRate",
    "RolloutResult",
    "compute_pass_rates",
    "measure_pass_rates",
]
