"""run_trial: a `harbor run` timeout is a genuine model failure, not an infra
failure -- it must not crash the sweep, and must count in the pass@k
denominator instead of being excluded as passed=None."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vektori_trace.validity import run_trial


def test_timeout_does_not_raise(tmp_path: Path, monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"), output="partial stdout", stderr="partial stderr")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_trial(
        task_dir=tmp_path,
        agent="terminus-2",
        jobs_dir=tmp_path / "jobs",
        timeout_sec=1800,
    )

    assert result.passed is False
    assert result.reward == 0.0


def test_timeout_persists_partial_output(tmp_path: Path, monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"), output="partial stdout", stderr="partial stderr")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_trial(
        task_dir=tmp_path,
        agent="terminus-2",
        jobs_dir=tmp_path / "jobs",
        timeout_sec=1800,
    )

    job_dir = tmp_path / "jobs" / f"{tmp_path.name}-terminus-2"
    assert (job_dir / "harbor_stdout.txt").read_text() == "partial stdout"
    assert (job_dir / "harbor_stderr.txt").read_text() == "partial stderr"


def test_a_real_reward_file_beats_the_timeout_default(tmp_path: Path, monkeypatch) -> None:
    """If harbor somehow wrote a result before being killed, trust it over
    the 0.0 timeout default."""
    job_dir = tmp_path / "jobs" / f"{tmp_path.name}-terminus-2"
    job_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text('{"verifier_result": {"rewards": {"reward": 1.0}}}')

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_trial(
        task_dir=tmp_path,
        agent="terminus-2",
        jobs_dir=tmp_path / "jobs",
        timeout_sec=1800,
    )

    assert result.passed is True
    assert result.reward == 1.0
