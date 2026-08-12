"""run_trial: a reward the verifier itself disclaims is not a measurement of the
agent.

The Aug-3 Qwen3-14B sweep reported pass@4 = 0/4 on a task where pytest died at
config load and never collected a single test. The verifier wrote
`parse_status: "fallback_exitcode"` and `eval_trustworthy: false` beside its
0.0; `run_trial` ignored both and returned passed=False, so four ungradeable
rollouts landed in the pass@k denominator as genuine model failures. One of them
had submitted an empty patch and "failed" identically.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from vektori_trace.evaluate.validity import run_trial


def _fake_harbor(monkeypatch, returncode: int = 1) -> None:
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def _job_dir(tmp_path: Path) -> Path:
    d = tmp_path / "jobs" / f"{tmp_path.name}-terminus-2"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_verdict(job_dir: Path, /, scored: float, **details) -> None:
    (job_dir / "result.json").write_text(
        json.dumps({"verifier_result": {"rewards": {"reward": scored}}})
    )
    if details:
        (job_dir / "reward-details.json").write_text(json.dumps(details))


def test_fallback_exitcode_withholds_the_verdict(tmp_path: Path, monkeypatch) -> None:
    """The exact shape of the Aug-3 artifacts."""
    _write_verdict(
        _job_dir(tmp_path),
        scored=0.0,
        reward=0.0,
        resolved=False,
        parse_status="fallback_exitcode",
        eval_trustworthy=False,
        runner="pytest",
        exit_code=1,
    )
    _fake_harbor(monkeypatch)

    result = run_trial(task_dir=tmp_path, agent="terminus-2", jobs_dir=tmp_path / "jobs")

    assert result.passed is None, "ungradeable must not be reported as a loss"
    assert result.parse_status == "fallback_exitcode"


def test_eval_trustworthy_false_alone_is_enough(tmp_path: Path, monkeypatch) -> None:
    """A future verifier may drop parse_status but keep the trust flag."""
    _write_verdict(_job_dir(tmp_path), scored=0.0, eval_trustworthy=False, runner="pytest")
    _fake_harbor(monkeypatch)

    result = run_trial(task_dir=tmp_path, agent="terminus-2", jobs_dir=tmp_path / "jobs")

    assert result.passed is None


def test_a_trustworthy_zero_is_still_a_loss(tmp_path: Path, monkeypatch) -> None:
    """The fix must not launder genuine failures into infra failures."""
    _write_verdict(
        _job_dir(tmp_path),
        scored=0.0,
        reward=0.0,
        resolved=False,
        parse_status="resolved",
        eval_trustworthy=True,
        runner="pytest",
    )
    _fake_harbor(monkeypatch)

    result = run_trial(task_dir=tmp_path, agent="terminus-2", jobs_dir=tmp_path / "jobs")

    assert result.passed is False
    assert result.reward == 0.0


def test_no_reward_details_at_all_is_still_a_loss(tmp_path: Path, monkeypatch) -> None:
    """Absence of the file means the verifier made no disclaimer."""
    _write_verdict(_job_dir(tmp_path), scored=0.0)
    _fake_harbor(monkeypatch)

    result = run_trial(task_dir=tmp_path, agent="terminus-2", jobs_dir=tmp_path / "jobs")

    assert result.passed is False


def test_a_pass_is_never_withheld(tmp_path: Path, monkeypatch) -> None:
    """Only a 0.0 can be disclaimed; a 1.0 stands regardless."""
    _write_verdict(_job_dir(tmp_path), scored=1.0, parse_status="fallback_exitcode")
    _fake_harbor(monkeypatch, returncode=0)

    result = run_trial(task_dir=tmp_path, agent="terminus-2", jobs_dir=tmp_path / "jobs")

    assert result.passed is True


def test_a_timeout_keeps_its_loss(tmp_path: Path, monkeypatch) -> None:
    """A timeout IS a model failure ("never finished in budget"), so it keeps
    its 0.0 even though no verifier ran to disclaim it. Guards the ordering of
    the two branches in run_trial."""

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_trial(
        task_dir=tmp_path,
        agent="terminus-2",
        jobs_dir=tmp_path / "jobs",
        timeout_sec=1800,
    )

    assert result.passed is False
    assert result.timed_out is True
