"""Orchestrator: mine a repo's real PR history into sandbox-verified tasks,
run an agent against each, and adapt the results into Run/Turn traces + a
manifest — the exact input `diagnose` already consumes.

Win/loss comes from the task's own F2P/P2P verifier (deterministic, real test
execution), never from an LLM guessing at outcome. Everything downstream
(diagnose.py, taskgen.py, validity.py) is unchanged — a mined trace is
indistinguishable from a pasted one once it's written to disk.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from vektori_trace.mining import (
    AuthSpec,
    BootstrapSpec,
    LLMSpec,
    OutputSpec,
    PipelineInput,
    PRRuntimeOptions,
    PRRuntimePipeline,
    RepoSpec,
    ensure_bootstrap,
)
from vektori_trace.schema import Turn
from vektori_trace.validity import _find_reward


def mine_tasks(
    repo: str,
    out_dir: Path,
    *,
    llm: LLMSpec,
    user_dockerfile: Path | None = None,
    org: str = "vektori",
    options: PRRuntimeOptions | None = None,
) -> list[Path]:
    """Mine `repo`'s merged PR history into sandbox-verified Harbor tasks
    under out_dir. Returns the per-task directories written.

    If `user_dockerfile` is given, the sandbox is built directly from it (no
    LLM involved). Otherwise `ensure_bootstrap` runs its ReAct agent loop to
    discover how to build/test the repo — needed when there's no working
    Dockerfile, or when a mined PR's base commit predates what the current
    Dockerfile/CI can actually build.
    """
    pipeline_input = PipelineInput(
        repo=RepoSpec(url=repo),
        llm=llm,
        output=OutputSpec(org=org),
        auth=AuthSpec(),
        bootstrap=BootstrapSpec(user_dockerfile=user_dockerfile),
    )
    bootstrap = ensure_bootstrap(pipeline_input.repo, pipeline_input.bootstrap, llm)
    pipeline = PRRuntimePipeline(pipeline_input, options or PRRuntimeOptions(), bootstrap=bootstrap)
    pipeline.run(out_dir)
    return [p for p in sorted(out_dir.iterdir()) if p.is_dir() and (p / "task.toml").exists()]


@dataclass
class MinedRun:
    """What running an agent against one mined task produces."""

    turns: list[Turn]
    passed: bool  # from the task's own F2P/P2P verifier, never LLM-judged


class TraceRunner(Protocol):
    """The agent under test. Real implementations shell out to `harbor run`
    against the mined task dir and parse its trajectory log into Turns."""

    def run(self, task_dir: Path) -> MinedRun: ...


def _turn_to_dict(t: Turn) -> dict:
    return {
        "index": t.index,
        "role": t.role,
        "thinking": t.thinking,
        "content": t.content,
        "toolCalls": [{"id": tc.id, "name": tc.name, "args": tc.args} for tc in t.tool_calls],
        "toolCallId": t.tool_call_id,
    }


def collect_traces(task_dirs: list[Path], runner: TraceRunner, traces_dir: Path) -> list[dict]:
    """Run `runner` against every mined task, write each as a Run/Turn trace
    JSON file, and return manifest entries ready for `load_manifest`."""
    traces_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for task_dir in task_dirs:
        result = runner.run(task_dir)
        run_id = f"{task_dir.name}-{uuid.uuid4().hex[:8]}"
        payload = {
            "runId": run_id,
            "status": "success" if result.passed else "failure",
            "turns": [_turn_to_dict(t) for t in result.turns],
        }
        trace_path = traces_dir / f"{run_id}.json"
        trace_path.write_text(json.dumps(payload, indent=2))
        manifest.append({"path": str(trace_path), "outcome": "win" if result.passed else "loss"})
    return manifest


class HarborTraceRunner:
    """Runs `harbor run` against a mined task and parses the resulting turns.

    Harbor's per-harness trajectory log format isn't standardized across
    agents (codex/claude_code/aider/...), so this does a best-effort parse:
    any `trajectory.json` under the job dir (a turn-shaped array, same shape
    as our own `Turn`) is used directly; otherwise falls back to a single
    condensed Turn built from raw stdout, so a harness we can't parse still
    produces a usable (if coarse) trace instead of failing outright. Passed
    comes from `_find_reward` — the task's own verifier, never LLM-judged.
    """

    def __init__(
        self, agent: str, jobs_dir: Path, model: str | None = None, timeout_sec: int = 1800
    ):
        self.agent = agent
        self.jobs_dir = jobs_dir
        self.model = model
        self.timeout_sec = timeout_sec

    def _parse_turns(self, job_dir: Path, raw_stdout: str) -> list[Turn]:
        for traj_file in job_dir.rglob("trajectory.json"):
            try:
                raw_turns = json.loads(traj_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(raw_turns, list):
                return [Turn.from_dict(t) for t in raw_turns]
        return [Turn(index=0, role="assistant", content=raw_stdout[-4000:])]

    def run(self, task_dir: Path) -> MinedRun:
        task_dir = task_dir.resolve()
        job_dir = (self.jobs_dir / f"{task_dir.name}-{self.agent}").resolve()
        cmd = [
            "harbor",
            "run",
            "-p",
            str(task_dir),
            "-a",
            self.agent,
            "--env",
            "docker",
            "--yes",
            "-o",
            str(job_dir),
        ]
        if self.model:
            cmd += ["--model", self.model]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_sec)
        reward = _find_reward(job_dir)
        passed = reward is not None and reward >= 1.0
        turns = self._parse_turns(job_dir, proc.stdout + proc.stderr)
        return MinedRun(turns=turns, passed=passed)


__all__ = [
    "HarborTraceRunner",
    "MinedRun",
    "TraceRunner",
    "collect_traces",
    "mine_tasks",
]
