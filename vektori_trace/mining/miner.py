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
import sys
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
from vektori_trace.mining.atif import TrajectoryParseError, parse_job_trajectory
from vektori_trace.mining.result import PipelineResult
from vektori_trace.schema import Turn
from vektori_trace.validity import UNJUDGEABLE_STATUSES, _find_parse_status, _find_reward


def mine_tasks(
    repo: str,
    out_dir: Path,
    *,
    llm: LLMSpec,
    user_dockerfile: Path | None = None,
    test_cmds: list[str] | None = None,
    language: str | None = None,
    org: str = "vektori",
    options: PRRuntimeOptions | None = None,
) -> tuple[list[Path], PipelineResult]:
    """Mine `repo`'s merged PR history into sandbox-verified Harbor tasks
    under out_dir.

    Returns the per-task directories written *and* the pipeline result. The
    result carries the per-reason skip counts, which is the only thing that
    says whether a small yield means the repo is unsuitable or a filter is
    wrong — the two call for opposite responses, and the number of tasks alone
    can't tell them apart.

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
        bootstrap=BootstrapSpec(
            user_dockerfile=user_dockerfile,
            user_test_cmds=test_cmds or [],
            user_language=language,
        ),
    )
    bootstrap = ensure_bootstrap(pipeline_input.repo, pipeline_input.bootstrap, llm)
    pipeline = PRRuntimePipeline(pipeline_input, options or PRRuntimeOptions(), bootstrap=bootstrap)
    result = pipeline.run(out_dir)
    task_dirs = [p for p in sorted(out_dir.iterdir()) if p.is_dir() and (p / "task.toml").exists()]
    return task_dirs, result


class InfraFailure(RuntimeError):
    """The run never produced a judgeable attempt — Docker died, the harness
    crashed, the container OOMed, the verifier never emitted a reward.

    This is emphatically *not* a loss. A loss means the agent tried and failed,
    which is evidence about the agent; an infra failure is evidence about our
    machine. Recording one as the other feeds diagnosis a trajectory whose
    ending it must explain, and it obliges by inventing a capability deficit.
    Excluded from the manifest entirely.
    """


@dataclass
class MinedRun:
    """What running an agent against one mined task produces."""

    turns: list[Turn]
    passed: bool  # from the task's own F2P/P2P verifier, never LLM-judged


class TraceRunner(Protocol):
    """The agent under test. Real implementations shell out to `harbor run`
    against the mined task dir and parse its trajectory log into Turns.

    Raises InfraFailure when the attempt couldn't be judged at all."""

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


def collect_traces(
    task_dirs: list[Path],
    runner: TraceRunner,
    traces_dir: Path,
    *,
    manifest_path: Path | None = None,
) -> list[dict]:
    """Run `runner` against every mined task, write each as a Run/Turn trace
    JSON file, and return manifest entries ready for `load_manifest`.

    Tasks whose run couldn't be judged (InfraFailure, or any unexpected error
    from the runner) are skipped, not recorded as losses. If `manifest_path` is
    given the manifest is rewritten after every task, so a crash on task 40 of
    50 leaves 39 usable traces behind instead of nothing.
    """
    traces_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for task_dir in task_dirs:
        try:
            result = runner.run(task_dir)
        except InfraFailure as e:
            print(f"  skip {task_dir.name}: infra failure — {e}", file=sys.stderr)
            continue
        except Exception as e:  # a bug in one task's run must not kill the sweep
            print(f"  skip {task_dir.name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        run_id = f"{task_dir.name}-{uuid.uuid4().hex[:8]}"
        payload = {
            "runId": run_id,
            "status": "success" if result.passed else "failure",
            "turns": [_turn_to_dict(t) for t in result.turns],
        }
        trace_path = traces_dir / f"{run_id}.json"
        trace_path.write_text(json.dumps(payload, indent=2))
        manifest.append({"path": str(trace_path), "outcome": "win" if result.passed else "loss"})
        if manifest_path is not None:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


class HarborTraceRunner:
    """Runs `harbor run` against a mined task and parses the resulting turns.

    Harbor writes ATIF, which `mining.atif` parses using Harbor's own models.
    There is no best-effort fallback: a trajectory we cannot parse is an infra
    failure and the task leaves the dataset. The previous fallback — the last
    4,000 characters of stdout as one Turn — produced traces that looked
    ordinary on disk and were a stderr tail, and every label derived from one
    was fiction. Passed comes from `_find_reward`, the task's own verifier,
    never LLM-judged.
    """

    def __init__(
        self, agent: str, jobs_dir: Path, model: str | None = None, timeout_sec: int = 1800
    ):
        # Harbor's AgentName values are hyphenated (`claude-code`, `terminus-2`,
        # `mini-swe-agent`). An underscore is rejected outright — the run dies
        # with "Agent name X is not valid" before any container starts, which
        # `run` reads as a non-zero exit and reports as an InfraFailure, so
        # every task in the sweep is silently excluded and the manifest comes
        # back empty. The underscore form is the natural thing to type and was
        # this CLI's own default, so accept it and normalise.
        self.agent = agent.replace("_", "-")
        self.jobs_dir = jobs_dir
        self.model = model
        self.timeout_sec = timeout_sec

    def _parse_turns(self, job_dir: Path) -> list[Turn]:
        try:
            return parse_job_trajectory(job_dir)
        except TrajectoryParseError as e:
            raise InfraFailure(f"could not parse trajectory: {e}") from e

    def run(self, task_dir: Path) -> MinedRun:
        task_dir = task_dir.resolve()
        # Unique per invocation: a reused job dir lets a previous run's
        # result.json be picked up as this run's reward.
        job_dir = (
            self.jobs_dir / f"{task_dir.name}-{self.agent}-{uuid.uuid4().hex[:8]}"
        ).resolve()
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

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_sec)
        except subprocess.TimeoutExpired as e:
            raise InfraFailure(f"harbor run exceeded {self.timeout_sec}s") from e
        except OSError as e:
            raise InfraFailure(f"could not exec harbor: {e}") from e

        if proc.returncode != 0:
            raise InfraFailure(
                f"harbor run exited {proc.returncode}: {(proc.stderr or proc.stdout)[-500:]}"
            )

        reward = _find_reward(job_dir)
        if reward is None:
            # The verifier never emitted a reward, so nothing here says the
            # agent failed — only that we can't tell.
            raise InfraFailure(f"no reward found under {job_dir}")

        # A reward the verifier itself disclaims. In separate-verifier mode the
        # agent's work travels as a collected diff, and collection is a step
        # that can fail on its own — an agent shadowing `git`, an unwritable
        # /logs, a failed artifact upload. The verifier writes 0.0 because it
        # must write something, and says in parse_status that the number means
        # nothing. Same rule as a Docker OOM: not knowing what happened is not
        # the agent failing.
        status = _find_parse_status(job_dir)
        if status in UNJUDGEABLE_STATUSES:
            raise InfraFailure(f"verifier could not judge this run ({status})")

        turns = self._parse_turns(job_dir)
        return MinedRun(turns=turns, passed=reward >= 1.0)


__all__ = [
    "HarborTraceRunner",
    "InfraFailure",
    "MinedRun",
    "TraceRunner",
    "collect_traces",
    "mine_tasks",
]
