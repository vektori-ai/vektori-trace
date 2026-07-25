"""Unit tests for mining/miner.py — no Docker/network/harbor involved.

`mine_tasks` itself (real PR mining) needs Docker + a real repo, so it's left
for manual/integration testing. This covers the part that's pure and fast:
turning agent-run results into Run/Turn traces + a manifest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from vektori_trace.mining.miner import InfraFailure, MinedRun, collect_traces
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
