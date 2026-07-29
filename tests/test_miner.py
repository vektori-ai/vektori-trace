"""Unit tests for mining/miner.py — no Docker/network/harbor involved.

`mine_tasks` itself (real PR mining) needs Docker + a real repo, so it's left
for manual/integration testing. This covers the part that's pure and fast:
turning agent-run results into Run/Turn traces + a manifest.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from vektori_trace.mining.miner import (
    HarborTraceRunner,
    InfraFailure,
    MinedRun,
    collect_traces,
)
from vektori_trace.schema import Trace, Turn, load_manifest
from vektori_trace.validity import _find_reward


class FakeTraceRunner:
    """Values are MinedRuns to return, or exceptions to raise."""

    def __init__(self, results: dict[str, MinedRun | Exception]):
        self.results = results
        self.seen: list[str] = []

    def run(self, task_dir: Path) -> MinedRun:
        self.seen.append(task_dir.name)
        result = self.results[task_dir.name]
        if isinstance(result, Exception):
            raise result
        return result


def _tasks(tmp_path: Path, *names: str) -> list[Path]:
    dirs = []
    for n in names:
        d = tmp_path / "tasks" / n
        d.mkdir(parents=True)
        dirs.append(d)
    return dirs


def _run(turns: str, passed: bool) -> MinedRun:
    return MinedRun(turns=[Turn(index=0, role="assistant", content=turns)], passed=passed)


def test_collect_traces_writes_win_and_loss(tmp_path: Path) -> None:
    task_a = tmp_path / "tasks" / "vektori__widget-1"
    task_b = tmp_path / "tasks" / "vektori__widget-2"
    task_a.mkdir(parents=True)
    task_b.mkdir(parents=True)

    runner = FakeTraceRunner(
        {
            "vektori__widget-1": MinedRun(
                turns=[Turn(index=0, role="assistant", content="fixed it")], passed=True
            ),
            "vektori__widget-2": MinedRun(
                turns=[Turn(index=0, role="assistant", content="gave up")], passed=False
            ),
        }
    )

    traces_dir = tmp_path / "traces"
    manifest = collect_traces([task_a, task_b], runner, traces_dir)

    assert {m["outcome"] for m in manifest} == {"win", "loss"}
    assert len(list(traces_dir.glob("*.json"))) == 2

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    entries = load_manifest(manifest_path)
    traces = [Trace.load(e.path, outcome=e.outcome) for e in entries]
    outcomes = {t.outcome for t in traces}
    assert outcomes == {"win", "loss"}
    win_trace = next(t for t in traces if t.outcome == "win")
    assert win_trace.turns[0].content == "fixed it"


def test_infra_failure_is_excluded_not_recorded_as_a_loss(tmp_path: Path) -> None:
    """A Docker OOM says nothing about the agent. Recorded as a loss it becomes
    a trajectory diagnosis has to explain, and it explains it with an invented
    deficit — so the task has to leave the dataset entirely."""
    ok, broken = _tasks(tmp_path, "t-ok", "t-broken")
    runner = FakeTraceRunner(
        {
            "t-ok": _run("fixed it", passed=False),
            "t-broken": InfraFailure("harbor run exited 137"),
        }
    )

    manifest = collect_traces([ok, broken], runner, tmp_path / "traces")

    assert len(manifest) == 1
    assert manifest[0]["outcome"] == "loss"
    assert "t-ok" in manifest[0]["path"]
    assert len(list((tmp_path / "traces").glob("*.json"))) == 1


def test_one_bad_task_does_not_kill_the_sweep(tmp_path: Path) -> None:
    a, b, c = _tasks(tmp_path, "t-a", "t-b", "t-c")
    runner = FakeTraceRunner(
        {
            "t-a": _run("a", passed=True),
            "t-b": RuntimeError("something unforeseen"),
            "t-c": _run("c", passed=True),
        }
    )

    manifest = collect_traces([a, b, c], runner, tmp_path / "traces")

    assert runner.seen == ["t-a", "t-b", "t-c"]  # c still attempted after b blew up
    assert len(manifest) == 2


def test_manifest_is_written_after_every_task(tmp_path: Path) -> None:
    """A crash on task 40 of 50 must leave 39 usable traces behind, not zero."""
    a, b = _tasks(tmp_path, "t-a", "t-b")
    manifest_path = tmp_path / "manifest.json"

    class CrashingRunner:
        def run(self, task_dir: Path) -> MinedRun:
            if task_dir.name == "t-b":
                raise KeyboardInterrupt
            return _run("a", passed=True)

    try:
        collect_traces([a, b], CrashingRunner(), tmp_path / "traces", manifest_path=manifest_path)
    except KeyboardInterrupt:
        pass

    written = json.loads(manifest_path.read_text())
    assert len(written) == 1
    assert written[0]["outcome"] == "win"


def test_find_reward_prefers_the_newest_result(tmp_path: Path) -> None:
    """rglob returns directory order, which has already handed us a stale 0.0
    over a fresh 1.0 in the same job dir."""
    stale = tmp_path / "aaa-old"
    fresh = tmp_path / "zzz-new"
    for d in (stale, fresh):
        d.mkdir()
    (stale / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 0.0}}}))
    (fresh / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}}))
    os.utime(stale / "result.json", (1_000, 1_000))
    os.utime(fresh / "result.json", (2_000, 2_000))

    assert _find_reward(tmp_path) == 1.0


def test_find_reward_is_none_when_nothing_was_written(tmp_path: Path) -> None:
    """None means 'we can't tell', and the caller must turn that into an
    exclusion rather than a zero reward."""
    assert _find_reward(tmp_path) is None


# ---------------------------------------------------------------------------
# HarborTraceRunner.run — the boundary between "a loss" and "we can't tell"
#
# collect_traces turns InfraFailure into an excluded task rather than a loss,
# so which branch fires here decides whether a Docker crash ends up in the
# diagnosis corpus as an agent failure. Only FakeTraceRunner was covered; the
# real runner's branches were not.
# ---------------------------------------------------------------------------


def _harbor_stub(
    monkeypatch,
    *,
    returncode: int = 0,
    reward: float | None = None,
    raises: Exception | None = None,
):
    """Stand in for `harbor run`, optionally writing a result.json first.

    Trajectory parsing is stubbed out alongside it: what's under test here is
    which branch a given (exit code, reward) lands in, and the parser has its
    own tests. Leaving the real parser in place would make these fail for a
    second, unrelated reason — no trajectory on disk.
    """
    monkeypatch.setattr(
        HarborTraceRunner,
        "_parse_turns",
        lambda self, *a, **k: [Turn(index=0, role="assistant", content="did a thing")],
    )

    def fake_run(cmd, **kwargs):
        if raises is not None:
            raise raises
        # `-o <job_dir>` is where harbor would write its results.
        job_dir = Path(cmd[cmd.index("-o") + 1])
        job_dir.mkdir(parents=True, exist_ok=True)
        if reward is not None:
            (job_dir / "result.json").write_text(
                json.dumps({"verifier_result": {"rewards": {"reward": reward}}})
            )
        return subprocess.CompletedProcess(cmd, returncode, stdout="out", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_reward_zero_is_a_loss_not_an_infra_failure(tmp_path: Path, monkeypatch) -> None:
    """The contract this whole class hangs on: harbor exits 0 for a valid
    verifier outcome, and reward=0.0 is a valid outcome. Reading it as an infra
    failure would silently drop every genuine loss — and losses are the half of
    the corpus the diagnosis is computed from."""
    _harbor_stub(monkeypatch, returncode=0, reward=0.0)
    runner = HarborTraceRunner(agent="codex", jobs_dir=tmp_path / "jobs")

    run = runner.run(tmp_path)

    assert run.passed is False


def test_reward_one_is_a_win(tmp_path: Path, monkeypatch) -> None:
    _harbor_stub(monkeypatch, returncode=0, reward=1.0)
    runner = HarborTraceRunner(agent="codex", jobs_dir=tmp_path / "jobs")

    assert runner.run(tmp_path).passed is True


def test_nonzero_exit_is_an_infra_failure(tmp_path: Path, monkeypatch) -> None:
    """harbor only exits non-zero on CLI/infrastructure errors, never for a
    low reward — so a non-zero exit says nothing about the agent."""
    _harbor_stub(monkeypatch, returncode=1, reward=1.0)
    runner = HarborTraceRunner(agent="codex", jobs_dir=tmp_path / "jobs")

    with pytest.raises(InfraFailure, match="exited 1"):
        runner.run(tmp_path)


def test_timeout_is_an_infra_failure(tmp_path: Path, monkeypatch) -> None:
    _harbor_stub(
        monkeypatch, raises=subprocess.TimeoutExpired(cmd="harbor", timeout=1800)
    )
    runner = HarborTraceRunner(agent="codex", jobs_dir=tmp_path / "jobs")

    with pytest.raises(InfraFailure, match="exceeded"):
        runner.run(tmp_path)


def test_missing_harbor_binary_is_an_infra_failure(tmp_path: Path, monkeypatch) -> None:
    _harbor_stub(monkeypatch, raises=FileNotFoundError("harbor"))
    runner = HarborTraceRunner(agent="codex", jobs_dir=tmp_path / "jobs")

    with pytest.raises(InfraFailure, match="could not exec harbor"):
        runner.run(tmp_path)


def test_exit_zero_with_no_reward_is_an_infra_failure(tmp_path: Path, monkeypatch) -> None:
    """Clean exit, no reward file: the verifier never rendered a verdict, so
    there is no evidence the agent failed — only that we can't tell."""
    _harbor_stub(monkeypatch, returncode=0, reward=None)
    runner = HarborTraceRunner(agent="codex", jobs_dir=tmp_path / "jobs")

    with pytest.raises(InfraFailure, match="no reward found"):
        runner.run(tmp_path)


def test_each_run_gets_its_own_job_dir(tmp_path: Path, monkeypatch) -> None:
    """A reused job dir lets a previous run's result.json be read as this
    run's reward."""
    seen: list[str] = []

    def fake_run(cmd, **kwargs):
        job_dir = Path(cmd[cmd.index("-o") + 1])
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "result.json").write_text(
            json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}})
        )
        seen.append(str(job_dir))
        return subprocess.CompletedProcess(cmd, 0, stdout="out", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        HarborTraceRunner,
        "_parse_turns",
        lambda self, *a, **k: [Turn(index=0, role="assistant", content="did a thing")],
    )
    runner = HarborTraceRunner(agent="codex", jobs_dir=tmp_path / "jobs")
    runner.run(tmp_path)
    runner.run(tmp_path)

    assert len(set(seen)) == 2


def test_a_disclaimed_reward_is_an_infra_failure_not_a_loss(
    tmp_path: Path, monkeypatch
) -> None:
    """The separate verifier writes 0.0 when the agent's diff never arrived,
    because it has to write something — and says so in parse_status. Reading
    that 0.0 as a loss puts a trace in the corpus that the diagnosis then has
    to explain, and it will invent a capability to do it."""

    def fake_run(cmd, **kwargs):
        job_dir = Path(cmd[cmd.index("-o") + 1])
        (job_dir / "verifier").mkdir(parents=True, exist_ok=True)
        (job_dir / "result.json").write_text(
            json.dumps({"verifier_result": {"rewards": {"reward": 0.0}}})
        )
        (job_dir / "verifier" / "reward-details.json").write_text(
            json.dumps({"reward": 0.0, "resolved": False, "parse_status": "no_model_patch"})
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="out", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        HarborTraceRunner,
        "_parse_turns",
        lambda self, *a, **k: [Turn(index=0, role="assistant", content="did a thing")],
    )
    runner = HarborTraceRunner(agent="codex", jobs_dir=tmp_path / "jobs")

    with pytest.raises(InfraFailure, match="could not judge"):
        runner.run(tmp_path)


def test_an_ordinary_zero_is_still_a_loss(tmp_path: Path, monkeypatch) -> None:
    """The mirror image: parse_status="ok" with reward 0.0 is a real agent
    failure, and dropping those would empty the loss half of the corpus."""

    def fake_run(cmd, **kwargs):
        job_dir = Path(cmd[cmd.index("-o") + 1])
        (job_dir / "verifier").mkdir(parents=True, exist_ok=True)
        (job_dir / "result.json").write_text(
            json.dumps({"verifier_result": {"rewards": {"reward": 0.0}}})
        )
        (job_dir / "verifier" / "reward-details.json").write_text(
            json.dumps({"reward": 0.0, "resolved": False, "parse_status": "ok"})
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="out", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        HarborTraceRunner,
        "_parse_turns",
        lambda self, *a, **k: [Turn(index=0, role="assistant", content="did a thing")],
    )
    runner = HarborTraceRunner(agent="codex", jobs_dir=tmp_path / "jobs")

    assert runner.run(tmp_path).passed is False


def test_underscored_agent_names_are_normalised(tmp_path: Path) -> None:
    """Harbor's AgentName values are hyphenated and it rejects an underscore
    before any container starts. `claude_code` — this CLI's own former default
    — therefore ran nothing: the non-zero exit read as an InfraFailure, every
    task in the sweep was excluded, and the manifest came back empty with no
    indication why."""
    runner = HarborTraceRunner(agent="claude_code", jobs_dir=tmp_path)
    assert runner.agent == "claude-code"

    assert HarborTraceRunner(agent="mini_swe_agent", jobs_dir=tmp_path).agent == "mini-swe-agent"
    # Already-correct names must survive untouched.
    assert HarborTraceRunner(agent="terminus-2", jobs_dir=tmp_path).agent == "terminus-2"
    assert HarborTraceRunner(agent="codex", jobs_dir=tmp_path).agent == "codex"


# ---------------------------------------------------------------------------
# Endpoint overrides. `run_trial` has accepted api_base/model_info since the
# Modal work, but this runner — the one `replay` uses for both arms — built its
# own harbor command and dropped them. A candidate served on our own vLLM was
# therefore unreachable from the command that produces the headline gap number.
# ---------------------------------------------------------------------------


def _captured_harbor_cmd(monkeypatch, runner: HarborTraceRunner, task_dir: Path) -> list[str]:
    seen: list[list[str]] = []
    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        seen.append(list(cmd))
        job_dir = Path(cmd[cmd.index("-o") + 1])
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "result.json").write_text(
            json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}})
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="out", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        HarborTraceRunner,
        "_parse_turns",
        lambda self, *a, **k: [Turn(index=0, role="assistant", content="did a thing")],
    )
    runner.run(task_dir)
    monkeypatch.setattr(subprocess, "run", real_run)
    return seen[0]


def test_api_base_and_model_info_reach_the_harbor_command(tmp_path: Path, monkeypatch) -> None:
    """Harbor has no top-level --api-base; both travel as --ak key=value, and
    model_info must arrive as JSON or harbor refuses to start a hosted_vllm run."""
    task = tmp_path / "t1"
    task.mkdir()
    runner = HarborTraceRunner(
        agent="terminus-2",
        jobs_dir=tmp_path / "jobs",
        model="hosted_vllm/qwen3-8b",
        api_base="https://example.modal.run/v1",
        model_info={"max_input_tokens": 32768},
    )
    cmd = _captured_harbor_cmd(monkeypatch, runner, task)

    assert "--ak" in cmd
    aks = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--ak"]
    assert "api_base=https://example.modal.run/v1" in aks
    assert 'model_info={"max_input_tokens": 32768}' in aks


def test_no_ak_args_when_no_endpoint_override(tmp_path: Path, monkeypatch) -> None:
    """The default path is a public API model; a stray empty --ak is a harbor
    parse error, not a no-op."""
    task = tmp_path / "t1"
    task.mkdir()
    runner = HarborTraceRunner(agent="claude-code", jobs_dir=tmp_path / "jobs", model="gpt-5")
    cmd = _captured_harbor_cmd(monkeypatch, runner, task)
    assert "--ak" not in cmd
