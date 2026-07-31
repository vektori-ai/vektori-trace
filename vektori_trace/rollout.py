"""Rejection-sampling rollout collection for Step 6 training.

Reuses `validity.run_trial` the same way `passrate.measure_pass_rates` does,
but keeps the full ATIF trajectory (not just the scalar reward). Only passing
rollouts are retained — that's the rejection. Docker-touching; not unit-tested
beyond mocking, matching `measure_pass_rates`.

Collecting candidate rollouts requires the candidate already served (Modal
vLLM) — one phase earlier than A2/A3 eval. Sequence per trained arm:
serve → rollout collect → train → serve adapter → measure.

Phase 0.5: when `capture_tokens=True`, each attempt is driven through a
`CaptureProxy` that injects `return_token_ids` and writes sampled ids next to
the harbor job dir. Training then consumes those ids instead of re-tokenizing.
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
    #: True when sampled token ids were persisted next to this job.
    tokens_captured: bool = False


def _merge_capture_agent_kwargs(
    agent_kwargs: dict[str, Any] | None,
    *,
    capture_logprobs: bool,
) -> dict[str, Any]:
    from .token_capture import token_capture_agent_kwargs

    base = dict(agent_kwargs or {})
    for k, v in token_capture_agent_kwargs(
        return_token_ids=True, logprobs=capture_logprobs
    ).items():
        if k == "extra_body" and isinstance(base.get("extra_body"), dict):
            merged = dict(base["extra_body"])
            merged.update(v)
            base["extra_body"] = merged
        else:
            base[k] = v
    return base


def _persist_captures_beside_job(
    proxy_capture_dir: Path,
    job_dir: Path,
    *,
    upstream: str | None,
) -> bool:
    """Copy proxy-written captures into the harbor job dir. Returns whether any
    completions were stored there."""
    from .token_capture import (
        CAPTURE_FILENAME,
        append_capture,
        dump_capture_manifest,
        load_captures,
    )

    captures = load_captures(proxy_capture_dir)
    if not captures:
        return False
    job_dir.mkdir(parents=True, exist_ok=True)
    target = job_dir / CAPTURE_FILENAME
    if not target.exists():
        for cap in captures:
            append_capture(job_dir, cap)
    dump_capture_manifest(
        job_dir,
        captures,
        extra={"via": "capture_proxy", "upstream": upstream},
    )
    return True


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
    capture_tokens: bool = False,
    capture_logprobs: bool = False,
) -> list[CollectedRollout]:
    """Run `rollouts` trials per task; keep only passes unless `keep_failures`.

    Trajectory parse failures are dropped (infra), not treated as agent losses —
    same discipline as mining: a missing ATIF must not poison the training set.

    `capture_tokens`: wrap `api_base` in a Phase 0.5 capture proxy (and merge
    litellm `extra_body` into agent_kwargs) so sampled token ids land next to
    the harbor job. No-ops the proxy when `api_base` is None — there is nothing
    to wrap; agent_kwargs still request ids if the harness forwards them.
    """
    from .token_capture import capture_proxy, load_captures

    kept: list[CollectedRollout] = []
    base_ak: dict[str, Any] | None = dict(agent_kwargs or {}) or None
    if capture_tokens:
        # Belt and suspenders: agent kwargs ask litellm for ids; the proxy
        # injects the flag even if the harness drops unknown kwargs.
        base_ak = _merge_capture_agent_kwargs(
            agent_kwargs, capture_logprobs=capture_logprobs
        )

    for task_dir in task_dirs:
        for i in range(rollouts):
            # Unique job dir per attempt so a prior result.json can't shadow us.
            attempt_dir = jobs_dir / f"{task_dir.name}-r{i}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            proxy_dir = attempt_dir / "_capture_proxy"
            tokens_captured = False

            if capture_tokens and api_base:
                with capture_proxy(
                    api_base,
                    proxy_dir,
                    inject_logprobs=capture_logprobs,
                ) as proxy:
                    trial = run_trial(
                        task_dir,
                        agent=agent,
                        jobs_dir=attempt_dir,
                        model=model,
                        api_base=proxy.api_base,
                        model_info=model_info,
                        agent_kwargs=base_ak,
                        extra_instruction_path=extra_instruction_path,
                    )
                    tokens_captured = _persist_captures_beside_job(
                        proxy_dir, trial.jobs_dir, upstream=api_base
                    )
            else:
                trial = run_trial(
                    task_dir,
                    agent=agent,
                    jobs_dir=attempt_dir,
                    model=model,
                    api_base=api_base,
                    model_info=model_info,
                    agent_kwargs=base_ak,
                    extra_instruction_path=extra_instruction_path,
                )
                if capture_tokens:
                    tokens_captured = bool(load_captures(trial.jobs_dir))

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
                    tokens_captured=tokens_captured,
                )
            )
    return kept


__all__ = ["CollectedRollout", "collect_rollouts"]
