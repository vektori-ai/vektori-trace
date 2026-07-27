"""Unit tests for passrate.py — pure aggregation of per-task rollout outcomes
into a rate + N. No Docker/subprocess: `measure_pass_rates` is the only I/O
boundary and is exercised separately with a monkeypatched `run_trial`."""

from __future__ import annotations

from pathlib import Path

from vektori_trace.passrate import (
    PASSRATE_MAX,
    PASSRATE_MIN,
    PassRate,
    RolloutResult,
    compute_pass_rates,
    measure_pass_rates,
)


def test_compute_pass_rates_groups_by_task() -> None:
    results = [
        RolloutResult(task="a", passed=True),
        RolloutResult(task="a", passed=False),
        RolloutResult(task="a", passed=True),
        RolloutResult(task="b", passed=False),
    ]
    rates = compute_pass_rates(results)

    assert rates["a"].passed == 2
    assert rates["a"].n == 3
    assert rates["a"].rate == 2 / 3
    assert rates["b"].rate == 0.0


def test_compute_pass_rates_omits_tasks_with_no_results() -> None:
    rates = compute_pass_rates([])
    assert rates == {}


def test_in_band_uses_default_band_boundaries_inclusive() -> None:
    assert PassRate(task="a", passed=1, n=10).in_band()  # 0.10, the floor
    assert PassRate(task="a", passed=4, n=10).in_band()  # 0.40, the ceiling
    assert not PassRate(task="a", passed=0, n=10).in_band()  # 0.0, all-fail
    assert not PassRate(task="a", passed=5, n=10).in_band()  # 0.50, above band


def test_in_band_with_zero_rollouts_is_false_not_a_crash() -> None:
    assert PassRate(task="a", passed=0, n=0).rate is None
    assert not PassRate(task="a", passed=0, n=0).in_band()


def test_in_band_respects_a_custom_band() -> None:
    pr = PassRate(task="a", passed=5, n=10)  # 0.5, outside the default band
    assert pr.in_band(band=(0.4, 0.6))
    assert not pr.in_band(band=(PASSRATE_MIN, PASSRATE_MAX))


def test_measure_pass_rates_drives_run_trial_and_aggregates(tmp_path: Path, monkeypatch) -> None:
    """The orchestrator: N `run_trial` calls per task, aggregated by task name.
    Not real Docker — `run_trial` is monkeypatched to a scripted sequence."""
    import vektori_trace.passrate as passrate_mod

    task_a = tmp_path / "task-a"
    task_a.mkdir()
    task_b = tmp_path / "task-b"
    task_b.mkdir()

    calls: list[tuple[str, str]] = []
    # task-a: 2 pass, 1 fail, 1 unjudgeable (not counted). task-b: all fail.
    script = {"task-a": iter([True, False, True, None]), "task-b": iter([False, False, False, False])}

    class _Trial:
        def __init__(self, passed):
            self.passed = passed

    def fake_run_trial(task_dir, agent, jobs_dir, model=None, timeout_sec=1800, **_kw):
        calls.append((task_dir.name, model))
        return _Trial(next(script[task_dir.name]))

    monkeypatch.setattr(passrate_mod, "run_trial", fake_run_trial)

    rates = measure_pass_rates(
        [task_a, task_b], agent="claude-code", model="small", jobs_dir=tmp_path / "jobs", rollouts=4
    )

    assert len(calls) == 8  # 4 rollouts x 2 tasks
    assert all(model == "small" for _, model in calls)
    assert rates["task-a"].passed == 2
    assert rates["task-a"].n == 3  # the None trial isn't counted
    assert rates["task-b"].passed == 0
    assert rates["task-b"].n == 4
