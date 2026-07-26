"""End-to-end tests for `cmd_replay` with `harbor` stubbed out — the level
where the path-resolution bug (manifest paths that don't survive Trace.load)
and the total-failure exit code only show up as a whole. No Docker/network."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from vektori_trace.cli import build_parser
from vektori_trace.mining.miner import HarborTraceRunner
from vektori_trace.schema import Turn


def _task_dirs(tmp_path: Path, *names: str) -> Path:
    tasks_dir = tmp_path / "tasks"
    for n in names:
        d = tasks_dir / n
        d.mkdir(parents=True)
        (d / "task.toml").write_text("x = 1")
    return tasks_dir


def _stub_harbor(monkeypatch, reward_by_model: dict[str, float]):
    def fake_run(cmd, **kwargs):
        job_dir = Path(cmd[cmd.index("-o") + 1])
        job_dir.mkdir(parents=True, exist_ok=True)
        model = cmd[cmd.index("--model") + 1] if "--model" in cmd else None
        reward = reward_by_model[model]
        (job_dir / "result.json").write_text(
            json.dumps({"verifier_result": {"rewards": {"reward": reward}}})
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="out", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        HarborTraceRunner,
        "_parse_turns",
        lambda self, *a, **k: [Turn(index=0, role="assistant", content="did a thing")],
    )


def test_replay_end_to_end_writes_a_loadable_manifest_and_correct_gap(tmp_path, monkeypatch) -> None:
    tasks_dir = _task_dirs(tmp_path, "t1", "t2")
    _stub_harbor(monkeypatch, {"gpt-5": 1.0, "small": 0.0})

    parser = build_parser()
    args = parser.parse_args(
        [
            "replay",
            "--tasks-dir",
            str(tasks_dir),
            "--frontier-model",
            "gpt-5",
            "--candidate-model",
            "small",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    rc = args.func(args)

    assert rc == 0
    gap = json.loads((tmp_path / "out" / "gap.json").read_text())
    assert gap["paired_n"] == 2
    assert gap["frontier_rate"] == 1.0
    assert gap["candidate_rate"] == 0.0
    assert gap["gap"] == 1.0

    # The manifest's own paths must actually resolve — this is what a relative,
    # non-absolute trace_path would silently break.
    manifest = json.loads((tmp_path / "out" / "replay-manifest.json").read_text())
    assert len(manifest) == 4
    for entry in manifest:
        assert Path(entry["path"]).is_file()


def test_replay_with_a_relative_out_dir_still_produces_loadable_paths(tmp_path, monkeypatch) -> None:
    """The bug this guards against: a relative --out plus collect_paired_traces
    writing relative trace paths, joined a second time by load_manifest's
    'resolve relative to the manifest's own directory' rule -> a path that
    doesn't exist."""
    tasks_dir = _task_dirs(tmp_path, "t1")
    _stub_harbor(monkeypatch, {"gpt-5": 1.0, "small": 1.0})

    parser = build_parser()
    args = parser.parse_args(
        [
            "replay",
            "--tasks-dir",
            str(tasks_dir),
            "--frontier-model",
            "gpt-5",
            "--candidate-model",
            "small",
            "--out",
            "relative-out",
        ]
    )

    monkeypatch.chdir(tmp_path)
    rc = args.func(args)

    assert rc == 0
    manifest = json.loads((tmp_path / "relative-out" / "replay-manifest.json").read_text())
    for entry in manifest:
        assert Path(entry["path"]).is_absolute()
        assert Path(entry["path"]).is_file()


def test_replay_exits_nonzero_when_no_task_was_judged_by_both_arms(tmp_path, monkeypatch) -> None:
    """Every candidate attempt fails infra-side (e.g. a bad --candidate-model
    name) -> paired_n is 0 -> this must not report success."""
    tasks_dir = _task_dirs(tmp_path, "t1", "t2")

    def fake_run(cmd, **kwargs):
        job_dir = Path(cmd[cmd.index("-o") + 1])
        job_dir.mkdir(parents=True, exist_ok=True)
        model = cmd[cmd.index("--model") + 1] if "--model" in cmd else None
        if model == "small":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unknown model")
        (job_dir / "result.json").write_text(
            json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}})
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="out", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        HarborTraceRunner,
        "_parse_turns",
        lambda self, *a, **k: [Turn(index=0, role="assistant", content="did a thing")],
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "replay",
            "--tasks-dir",
            str(tasks_dir),
            "--frontier-model",
            "gpt-5",
            "--candidate-model",
            "small",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    rc = args.func(args)

    assert rc == 1


def test_replay_rerun_against_the_same_out_dir_makes_no_new_harbor_calls(tmp_path, monkeypatch) -> None:
    """Simulates the scenario the resume fix exists for: the sweep already
    finished once (or was interrupted after finishing), and gets invoked again
    against the same --out. Nothing already in the manifest should cost a
    second harbor run."""
    tasks_dir = _task_dirs(tmp_path, "t1", "t2")
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        job_dir = Path(cmd[cmd.index("-o") + 1])
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "result.json").write_text(
            json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}})
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="out", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        HarborTraceRunner,
        "_parse_turns",
        lambda self, *a, **k: [Turn(index=0, role="assistant", content="did a thing")],
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "replay",
            "--tasks-dir",
            str(tasks_dir),
            "--frontier-model",
            "gpt-5",
            "--candidate-model",
            "small",
            "--out",
            str(tmp_path / "out"),
        ]
    )

    assert args.func(args) == 0
    assert calls["n"] == 4  # 2 tasks x 2 arms

    assert args.func(args) == 0  # rerun against the identical --out
    assert calls["n"] == 4  # no new harbor invocations at all


def test_replay_rejects_a_nonexistent_tasks_dir(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "replay",
            "--tasks-dir",
            str(tmp_path / "does-not-exist"),
            "--frontier-model",
            "gpt-5",
            "--candidate-model",
            "small",
            "--out",
            str(tmp_path / "out"),
        ]
    )

    assert args.func(args) == 2
