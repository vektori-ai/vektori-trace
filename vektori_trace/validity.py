"""Run a generated Harbor task through `harbor run` to produce the validity proof:
the oracle solution passes, and a real off-the-shelf agent (without the diagnosed
capability) fails."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrialResult:
    agent: str
    passed: bool | None  # None if the reward couldn't be determined
    reward: float | None
    jobs_dir: Path
    raw_stdout: str


def _newest_first(paths: list[Path]) -> list[Path]:
    """Most recently modified first, so a stale artifact from an earlier run in
    the same directory never shadows this run's result. `rglob` returns
    directory order, which has already handed us a stale 0.0 over a fresh 1.0."""

    def mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    return sorted(paths, key=mtime, reverse=True)


def _find_reward(jobs_dir: Path) -> float | None:
    """Harbor writes a per-trial result.json with verifier_result.rewards.reward,
    nested under jobs_dir/<timestamp>/<trial_name>/result.json. Fall back to the
    raw verifier/reward.txt if the JSON shape doesn't match."""
    for result_file in _newest_first(list(jobs_dir.rglob("result.json"))):
        try:
            data = json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rewards = (data.get("verifier_result") or {}).get("rewards") or {}
        if "reward" in rewards:
            return float(rewards["reward"])
    for reward_file in _newest_first(list(jobs_dir.rglob("reward.txt"))):
        try:
            return float(reward_file.read_text().strip())
        except (ValueError, OSError):
            continue
    return None


# Statuses the graded verifier writes when it could not evaluate the agent's
# work at all. The 0.0 beside them is a placeholder for "no verdict", not a
# measurement of the agent — reading it as a loss puts a trace in the corpus
# that the diagnosis then has to explain, and it will.
UNJUDGEABLE_STATUSES = frozenset({"no_model_patch"})


def _find_parse_status(jobs_dir: Path) -> str | None:
    """The graded verifier's own account of whether its 0.0 means anything."""
    for details in _newest_first(list(jobs_dir.rglob("reward-details.json"))):
        try:
            data = json.loads(details.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        status = data.get("parse_status")
        if isinstance(status, str):
            return status
    return None


def run_trial(
    task_dir: Path,
    agent: str,
    jobs_dir: Path,
    model: str | None = None,
    timeout_sec: int = 1800,
) -> TrialResult:
    """Invoke `harbor run` for a single agent against a task and parse the reward.

    Harbor's agent names are hyphenated and it rejects an underscore outright,
    before any container starts — so `claude_code` never runs at all. Normalised
    here as well as in `HarborTraceRunner`, since `--base-agent` reaches this
    function without passing through that one.
    """
    task_dir = task_dir.resolve()
    agent = agent.replace("_", "-")
    job_dir = (jobs_dir / f"{task_dir.name}-{agent}").resolve()
    cmd = [
        "harbor",
        "run",
        "-p",
        str(task_dir),
        "-a",
        agent,
        "--env",
        "docker",
        "--yes",
        "-o",
        str(job_dir),
    ]
    if model:
        cmd += ["--model", model]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    reward = _find_reward(job_dir)
    passed = None if reward is None else reward >= 1.0
    return TrialResult(
        agent=agent,
        passed=passed,
        reward=reward,
        jobs_dir=job_dir,
        raw_stdout=proc.stdout[-4000:] + proc.stderr[-2000:],
    )


def prove_validity(
    task_dir: Path,
    jobs_dir: Path,
    base_agent: str | None = None,
    base_model: str | None = None,
) -> dict:
    """Run the oracle (should pass) and, if base_agent is given, a real agent
    (should fail) against the generated task."""
    oracle = run_trial(task_dir, agent="oracle", jobs_dir=jobs_dir)
    result = {
        "oracle": oracle,
        "base": None,
        "valid": oracle.passed is True,
    }
    if base_agent:
        base = run_trial(task_dir, agent=base_agent, jobs_dir=jobs_dir, model=base_model)
        result["base"] = base
        result["valid"] = (oracle.passed is True) and (base.passed is False)
    return result
