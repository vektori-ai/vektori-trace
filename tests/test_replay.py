"""Unit tests for collect_paired_traces — the multi-model, one-scaffold replay
that Step 4 needs. No Docker/network/harbor involved."""

from __future__ import annotations

import json
from pathlib import Path

from vektori_trace.mining.miner import InfraFailure, MinedRun, collect_paired_traces, discover_tasks
from vektori_trace.schema import Trace, Turn


class FakeTraceRunner:
    """Values are MinedRuns to return, or exceptions to raise, per task name."""

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
        (d / "task.toml").write_text("x = 1")
        dirs.append(d)
    return dirs


def _run(content: str, passed: bool) -> MinedRun:
    return MinedRun(turns=[Turn(index=0, role="assistant", content=content)], passed=passed)


def test_discover_tasks_only_returns_dirs_with_task_toml(tmp_path: Path) -> None:
    (tmp_path / "tasks").mkdir()
    real = tmp_path / "tasks" / "real-task"
    real.mkdir()
    (real / "task.toml").write_text("x = 1")
    (tmp_path / "tasks" / "not-a-task").mkdir()  # no task.toml

    assert discover_tasks(tmp_path / "tasks") == [real]


def test_every_arm_runs_against_every_task(tmp_path: Path) -> None:
    t1, t2 = _tasks(tmp_path, "t1", "t2")
    frontier = FakeTraceRunner({"t1": _run("f1", True), "t2": _run("f2", False)})
    candidate = FakeTraceRunner({"t1": _run("c1", False), "t2": _run("c2", True)})

    manifest = collect_paired_traces(
        [t1, t2], [("gpt-5", frontier), ("small-model", candidate)], tmp_path / "traces"
    )

    assert frontier.seen == ["t1", "t2"]
    assert candidate.seen == ["t1", "t2"]
    assert len(manifest) == 4
    assert {m["model"] for m in manifest} == {"gpt-5", "small-model"}
    assert {m["task"] for m in manifest} == {"t1", "t2"}


def test_manifest_entries_pair_by_task_across_models(tmp_path: Path) -> None:
    (t1,) = _tasks(tmp_path, "t1")
    frontier = FakeTraceRunner({"t1": _run("f", True)})
    candidate = FakeTraceRunner({"t1": _run("c", False)})

    manifest = collect_paired_traces(
        [t1], [("gpt-5", frontier), ("small-model", candidate)], tmp_path / "traces"
    )

    by_model = {m["model"]: m for m in manifest}
    assert by_model["gpt-5"]["task"] == "t1"
    assert by_model["gpt-5"]["outcome"] == "win"
    assert by_model["small-model"]["task"] == "t1"
    assert by_model["small-model"]["outcome"] == "loss"

    traces = [
        Trace.load(Path(m["path"]), outcome=m["outcome"], model=m["model"], task=m["task"])
        for m in manifest
    ]
    frontier_trace = next(t for t in traces if t.model == "gpt-5")
    assert frontier_trace.task == "t1"
    assert frontier_trace.turns[0].content == "f"


def test_one_arms_infra_failure_does_not_exclude_the_other_arms_trace(tmp_path: Path) -> None:
    """A Docker OOM on the frontier's attempt at a task must not take the
    candidate's (perfectly judgeable) attempt at that same task down with it."""
    (t1,) = _tasks(tmp_path, "t1")
    frontier = FakeTraceRunner({"t1": InfraFailure("harbor run exited 137")})
    candidate = FakeTraceRunner({"t1": _run("c", True)})

    manifest = collect_paired_traces(
        [t1], [("gpt-5", frontier), ("small-model", candidate)], tmp_path / "traces"
    )

    assert len(manifest) == 1
    assert manifest[0]["model"] == "small-model"
    assert manifest[0]["outcome"] == "win"


def test_one_bad_arm_does_not_kill_the_sweep(tmp_path: Path) -> None:
    t1, t2 = _tasks(tmp_path, "t1", "t2")
    frontier = FakeTraceRunner({"t1": RuntimeError("boom"), "t2": _run("f", True)})
    candidate = FakeTraceRunner({"t1": _run("c1", True), "t2": _run("c2", True)})

    manifest = collect_paired_traces(
        [t1, t2], [("gpt-5", frontier), ("small-model", candidate)], tmp_path / "traces"
    )

    assert frontier.seen == ["t1", "t2"]  # t2 still attempted after t1 blew up
    assert len(manifest) == 3


def test_manifest_is_written_after_every_arm(tmp_path: Path) -> None:
    (t1,) = _tasks(tmp_path, "t1")
    manifest_path = tmp_path / "manifest.json"

    class CrashingSecondArm:
        def run(self, task_dir: Path) -> MinedRun:
            raise KeyboardInterrupt

    frontier = FakeTraceRunner({"t1": _run("f", True)})
    try:
        collect_paired_traces(
            [t1],
            [("gpt-5", frontier), ("small-model", CrashingSecondArm())],
            tmp_path / "traces",
            manifest_path=manifest_path,
        )
    except KeyboardInterrupt:
        pass

    written = json.loads(manifest_path.read_text())
    assert len(written) == 1
    assert written[0]["model"] == "gpt-5"
