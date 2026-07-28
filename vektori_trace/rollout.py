"""Rejection-sampling rollout collection for Step 6 training.

Reuses `validity.run_trial` the same way `passrate.measure_pass_rates` does,
but keeps the full ATIF trajectory (not just the scalar reward). Only passing
rollouts are retained — that's the rejection. Docker-touching; not unit-tested
beyond mocking, matching `measure_pass_rates`.

Collecting candidate rollouts requires the candidate already served (Modal
vLLM) — one phase earlier than A2/A3 eval. Sequence per trained arm:
serve → rollout collect → train → serve adapter → measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .mining.atif import TrajectoryParseError, parse_job_trajectory
from .schema import Turn
from .validity import run_trial


@dataclass
class CollectedRollout:
    task: str
    passed: bool
    reward: float | None
    turns: list[Turn] = field(default_factory=list)
    jobs_dir: Path | None = None


def collect_rollouts(
    task_dirs: list[Path],
    agent: str,
    model: str,
    jobs_dir: Path,
    rollouts: int,
    *,
    api_base: str | None = None,
    model_info: dict[str, Any] | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    extra_instruction_path: Path | None = None,
    keep_failures: bool = False,
) -> list[CollectedRollout]:
    """Run `rollouts` trials per task; keep only passes unless `keep_failures`.

    Trajectory parse failures are dropped (infra), not treated as agent losses —
    same discipline as mining: a missing ATIF must not poison the training set.
    """
    kept: list[CollectedRollout] = []
    for task_dir in task_dirs:
        for i in range(rollouts):
            # Unique job dir per attempt so a prior result.json can't shadow us.
            attempt_dir = jobs_dir / f"{task_dir.name}-r{i}"
            trial = run_trial(
                task_dir,
                agent=agent,
                jobs_dir=attempt_dir,
                model=model,
                api_base=api_base,
                model_info=model_info,
                agent_kwargs=agent_kwargs,
                extra_instruction_path=extra_instruction_path,
            )
            if trial.passed is None:
                continue
            if not trial.passed and not keep_failures:
                continue
            try:
                turns = parse_job_trajectory(trial.jobs_dir)
            except TrajectoryParseError:
                continue
            kept.append(
                CollectedRollout(
                    task=task_dir.name,
                    passed=trial.passed,
                    reward=trial.reward,
                    turns=turns,
                    jobs_dir=trial.jobs_dir,
                )
            )
    return kept


__all__ = ["CollectedRollout", "collect_rollouts"]
