"""Step C — pass@k estimator, escalation, aggregation, luck controls."""

from __future__ import annotations

import math

import pytest

from vektori_trace.evaluate.passk import (
    K_VALUES,
    RolloutOutcome,
    TaskPassK,
    aggregate_by_capability,
    compute_task_passk,
    luck_quarantined_tasks,
    mark_luck_quarantine,
    pass_at_k,
    pooled_estimate_is_biased,
    tasks_needing_escalation,
)


def _closed_form(n: int, c: int, k: int) -> float:
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def test_pass_at_k_matches_closed_form() -> None:
    for n, c in [(8, 0), (8, 3), (8, 8), (32, 1), (16, 7)]:
        for k in (1, 4, 8, 16, 32):
            if k > n:
                assert pass_at_k(n, c, k) is None
            else:
                assert pass_at_k(n, c, k) == pytest.approx(_closed_form(n, c, k))


def test_pass_at_k_boundaries() -> None:
    assert pass_at_k(0, 0, 1) is None
    assert pass_at_k(5, 0, 1) == 0.0
    assert pass_at_k(5, 5, 1) == 1.0
    assert pass_at_k(5, 2, 8) is None


def test_escalation_only_zeros() -> None:
    stage1 = {
        "a": TaskPassK.from_counts("a", 8, 0, stratum="stage1"),
        "b": TaskPassK.from_counts("b", 8, 1, stratum="stage1"),
        "c": TaskPassK.from_counts("c", 8, 0, stratum="stage1"),
    }
    assert tasks_needing_escalation(stage1) == ["a", "c"]


def test_strata_never_merged() -> None:
    outcomes = [
        RolloutOutcome("t", False, "stage1"),
        RolloutOutcome("t", False, "stage1"),
        RolloutOutcome("t", True, "stage2"),
    ]
    got = compute_task_passk(outcomes)
    assert ("t", "stage1") in got and ("t", "stage2") in got
    assert got[("t", "stage1")].c == 0
    assert got[("t", "stage2")].c == 1


def test_pooled_vs_per_stratum_diverge() -> None:
    """Escalating only zeros biases naive pooling upward: pooling 0/8 with the
    32-sample that the zero triggered reports support the stage-1 stratum never
    saw. PLAN.md: report per stratum, never pool."""
    stage1 = {"t": TaskPassK.from_counts("t", 8, 0, stratum="stage1")}
    stage2 = {"t": TaskPassK.from_counts("t", 32, 4, stratum="stage2")}
    assert pooled_estimate_is_biased(stage1, stage2)

    pooled = pass_at_k(8 + 32, 0 + 4, 8)
    per_stratum = stage1["t"].curves[8]
    assert per_stratum == 0.0
    assert pooled is not None and pooled > 0.35  # the upward bias, quantified

    # A task that stayed at zero after escalation is not biased by pooling.
    clean1 = {"t": TaskPassK.from_counts("t", 8, 0, stratum="stage1")}
    clean2 = {"t": TaskPassK.from_counts("t", 32, 0, stratum="stage2")}
    assert not pooled_estimate_is_biased(clean1, clean2)


def test_luck_is_cross_stratum_not_within_sample() -> None:
    """Within one sample, c > 0 ⇒ pass@1 > 0, so no (n, c) can express
    "passes only at k > 8". Any within-sample luck rule is dead code."""
    for n in (8, 16, 32):
        for c in range(n + 1):
            pr = TaskPassK.from_counts("t", n, c, stratum="stage1")
            assert pr.luck_quarantine is False
            if c > 0:
                assert (pr.curves[1] or 0.0) > 0.0
    assert 8 in K_VALUES  # the k the "only passes at k>8" rule is stated against


def test_luck_flagged_when_passes_only_appear_under_escalation() -> None:
    stage1 = {
        "lucky": TaskPassK.from_counts("lucky", 8, 0, stratum="stage1"),
        "dead": TaskPassK.from_counts("dead", 8, 0, stratum="stage1"),
        "fine": TaskPassK.from_counts("fine", 8, 3, stratum="stage1"),
    }
    stage2 = {
        "lucky": TaskPassK.from_counts("lucky", 32, 1, stratum="stage2"),
        "dead": TaskPassK.from_counts("dead", 32, 0, stratum="stage2"),
    }
    assert luck_quarantined_tasks(stage1, stage2) == ["lucky"]
    assert mark_luck_quarantine(stage1, stage2) == ["lucky"]
    # Marked on both strata so either one can gate a routing decision.
    assert stage1["lucky"].luck_quarantine
    assert stage2["lucky"].luck_quarantine
    assert not stage2["dead"].luck_quarantine
    assert not stage1["fine"].luck_quarantine


def test_aggregate_by_capability_includes_n() -> None:
    curves = {
        "t1": TaskPassK.from_counts("t1", 8, 2, stratum="stage1"),
        "t2": TaskPassK.from_counts("t2", 8, 4, stratum="stage1"),
    }
    agg = aggregate_by_capability(
        curves, {"t1": "capA", "t2": "capA"}, model="m", stratum="stage1"
    )
    assert agg["capA"].n_tasks == 2
    assert agg["capA"].task_ids == ["t1", "t2"]
    # mean of pass@1 over the group: (2/8 + 4/8) / 2
    assert agg["capA"].mean_curves[1] == pytest.approx(0.375)
    assert agg["capA"].mean_curves[16] is None  # k > n is undefined, not 0


def test_support_classification_needs_the_escalation() -> None:
    from vektori_trace.evaluate.passk import classify_support

    s1_zero = TaskPassK.from_counts("t", 8, 0, stratum="stage1")
    s1_pass = TaskPassK.from_counts("t", 8, 2, stratum="stage1")
    s2_zero = TaskPassK.from_counts("t", 32, 0, stratum="stage2")
    s2_pass = TaskPassK.from_counts("t", 32, 1, stratum="stage2")

    assert classify_support(s1_pass, None) == "in_support"
    assert classify_support(s1_zero, s2_pass) == "in_support"
    assert classify_support(s1_zero, s2_zero) == "outside_support"
    # 0/8 alone is underpowered, not proof of absent support.
    assert classify_support(s1_zero, None) == "undetermined"
    assert classify_support(None, None) == "no_rollouts"


def test_capability_means_exclude_luck_quarantined_tasks() -> None:
    """PLAN.md quarantines only-passes-at-k>8 tasks so they cannot set a decision
    before an independent n=32 re-sample. Per-task routing honours that; a
    capability mean that still averages them in lets one lucky rollout drive the
    per-capability decision anyway."""
    stage2 = {
        "lucky": TaskPassK.from_counts("lucky", 32, 16, stratum="stage2"),
        "solid": TaskPassK.from_counts("solid", 32, 4, stratum="stage2"),
    }
    stage1 = {
        "lucky": TaskPassK.from_counts("lucky", 8, 0, stratum="stage1"),
        "solid": TaskPassK.from_counts("solid", 8, 1, stratum="stage1"),
    }
    assert mark_luck_quarantine(stage1, stage2) == ["lucky"]

    agg = aggregate_by_capability(
        stage2, {"lucky": "capA", "solid": "capA"}, model="m", stratum="stage2"
    )
    cap = agg["capA"]
    # N is the count *after* exclusion, and the excluded id is named.
    assert cap.n_tasks == 1
    assert cap.task_ids == ["solid"]
    assert cap.luck_excluded == ["lucky"]
    # The mean is the surviving task's own curve, not the average of the two.
    solid_p1 = stage2["solid"].curves[1]
    assert cap.mean_curves[1] == pytest.approx(solid_p1)
    assert cap.mean_curves[1] != pytest.approx(
        (solid_p1 + stage2["lucky"].curves[1]) / 2
    )


def test_capability_group_of_only_luck_tasks_reports_zero_n_not_a_rate() -> None:
    curves = {"lucky": TaskPassK.from_counts("lucky", 32, 8, stratum="stage2")}
    curves["lucky"].luck_quarantine = True
    agg = aggregate_by_capability(
        curves, {"lucky": "capA"}, model="m", stratum="stage2"
    )
    assert agg["capA"].n_tasks == 0
    assert agg["capA"].mean_curves[1] is None
    assert agg["capA"].luck_excluded == ["lucky"]


def test_luck_quarantine_has_a_resolve_path() -> None:
    """PLAN.md holds a lucky task "pending patch-vs-gold diff and independent
    re-sample" — pending, not forever. Both conditions must be met to clear."""
    from vektori_trace.evaluate.passk import LuckResolution, resolve_luck_quarantine

    stage2 = {
        "confirmed": TaskPassK.from_counts("confirmed", 32, 2, stratum="stage2"),
        "not_reproduced": TaskPassK.from_counts("not_reproduced", 32, 1, stratum="stage2"),
        "hacked": TaskPassK.from_counts("hacked", 32, 1, stratum="stage2"),
        "uninspected": TaskPassK.from_counts("uninspected", 32, 1, stratum="stage2"),
        "silent": TaskPassK.from_counts("silent", 32, 1, stratum="stage2"),
        "never_lucky": TaskPassK.from_counts("never_lucky", 32, 8, stratum="stage2"),
    }
    for name in ("confirmed", "not_reproduced", "hacked", "uninspected", "silent"):
        stage2[name].luck_quarantine = True

    report = resolve_luck_quarantine(
        stage2,
        {
            "confirmed": LuckResolution("confirmed", 32, 3, patch_matches_gold=True),
            "not_reproduced": LuckResolution(
                "not_reproduced", 32, 0, patch_matches_gold=True
            ),
            "hacked": LuckResolution("hacked", 32, 4, patch_matches_gold=False),
            "uninspected": LuckResolution("uninspected", 32, 4),
        },
    )
    assert report["cleared"] == ["confirmed"]
    assert not stage2["confirmed"].luck_quarantine
    # Everything else stays held, each with a stated reason.
    assert set(report["held"]) == {"not_reproduced", "hacked", "uninspected", "silent"}
    assert "did not reproduce" in report["held"]["not_reproduced"]
    assert "reward hack" in report["held"]["hacked"]
    assert "not inspected" in report["held"]["uninspected"]
    # Silence is not a clearance.
    assert "no independent re-sample" in report["held"]["silent"]
    assert all(stage2[t].luck_quarantine for t in report["held"])
    # A task that was never quarantined is untouched, not reported.
    assert "never_lucky" not in report["held"]


def test_luck_resolution_requires_both_conditions() -> None:
    from vektori_trace.evaluate.passk import LuckResolution

    assert LuckResolution("t", 32, 2, patch_matches_gold=True).resolved
    assert not LuckResolution("t", 32, 0, patch_matches_gold=True).resolved
    assert not LuckResolution("t", 32, 2, patch_matches_gold=False).resolved
    assert not LuckResolution("t", 32, 2).resolved


# ---------------------------------------------------------------------------
# Sweep mechanics: the evidence behind the rate, not the rate itself.
# ---------------------------------------------------------------------------


def _fake_trials(monkeypatch, *, passed: bool | None, calls: list):
    """Replace run_trial with a recorder. No Docker, no harbor, no network."""
    from pathlib import Path

    from vektori_trace.evaluate import passk as passk_mod
    from vektori_trace.evaluate.validity import TrialResult

    def fake_run_trial(task_dir, agent, jobs_dir, model=None, **kwargs):
        calls.append(Path(jobs_dir))
        return TrialResult(
            agent=agent,
            passed=passed,
            reward=None if passed is None else (1.0 if passed else 0.0),
            jobs_dir=Path(jobs_dir),
            raw_stdout="",
            elapsed_sec=0.5,
            started_at=1_700_000_000.0,
        )

    monkeypatch.setattr(passk_mod, "run_trial", fake_run_trial)


def _task_dirs(tmp_path, names):
    dirs = []
    for name in names:
        d = tmp_path / name
        d.mkdir()
        (d / "task.toml").write_text("[task]\n")
        dirs.append(d)
    return dirs


def test_each_rollout_gets_its_own_job_dir_when_serial(tmp_path, monkeypatch):
    """Serial sweeps used to funnel every rollout into one dir, so the n
    trajectories overwrote each other and only the last survived."""
    from vektori_trace.evaluate.passk import measure_passk_stage

    calls: list = []
    _fake_trials(monkeypatch, passed=False, calls=calls)
    measure_passk_stage(
        _task_dirs(tmp_path, ["t1", "t2"]),
        "terminus-2",
        "qwen3-8b",
        tmp_path / "jobs",
        4,
        stratum="stage1",
        max_workers=1,
    )
    assert len(calls) == 8
    assert len(set(calls)) == 8, "rollouts shared a job dir and clobbered each other"


def test_no_escalate_keeps_a_small_run_small(tmp_path, monkeypatch):
    """c == 0 triggers escalation regardless of stage-1 size, so a 4-rollout
    smoke test would silently become 4 + 32."""
    from vektori_trace.evaluate.passk import two_stage_sweep

    calls: list = []
    _fake_trials(monkeypatch, passed=False, calls=calls)
    report = two_stage_sweep(
        _task_dirs(tmp_path, ["t1"]),
        agent="terminus-2",
        model="qwen3-8b",
        jobs_dir=tmp_path / "jobs",
        stage1_n=4,
        stage2_n=32,
        escalate=False,
    )
    assert len(calls) == 4
    assert report["escalated"] == []
    assert report["escalation_enabled"] is False


def test_escalation_still_fires_when_enabled(tmp_path, monkeypatch):
    from vektori_trace.evaluate.passk import two_stage_sweep

    calls: list = []
    _fake_trials(monkeypatch, passed=False, calls=calls)
    report = two_stage_sweep(
        _task_dirs(tmp_path, ["t1"]),
        agent="terminus-2",
        model="qwen3-8b",
        jobs_dir=tmp_path / "jobs",
        stage1_n=4,
        stage2_n=6,
    )
    assert len(calls) == 10
    assert report["escalated"] == ["t1"]


def test_rollout_log_is_written_per_rollout(tmp_path, monkeypatch):
    """The log must survive a killed sweep, so it is flushed per rollout rather
    than written with the report at the end."""
    import json

    from vektori_trace.evaluate.passk import PASSK_LOG_FILENAME, measure_passk_stage

    calls: list = []
    _fake_trials(monkeypatch, passed=True, calls=calls)
    log_path = tmp_path / PASSK_LOG_FILENAME
    measure_passk_stage(
        _task_dirs(tmp_path, ["t1"]),
        "terminus-2",
        "qwen3-8b",
        tmp_path / "jobs",
        3,
        stratum="stage1",
        max_workers=1,
        log_path=log_path,
    )
    lines = [json.loads(x) for x in log_path.read_text().splitlines()]
    assert len(lines) == 3
    assert {x["rollout_index"] for x in lines} == {0, 1, 2}
    assert all(x["elapsed_sec"] == 0.5 for x in lines)
    assert all(x["started_at"] == 1_700_000_000.0 for x in lines)
    assert all(x["jobs_dir"] for x in lines)
    assert all(x["infra_failure"] is False for x in lines)


def test_infra_failures_stay_out_of_the_denominator_but_reach_the_log(
    tmp_path, monkeypatch
):
    """passed=None is a harness failure, not a model failure. Counting it as a
    loss is the easiest way to report a wrongly-pessimistic baseline."""
    import json

    from vektori_trace.evaluate.passk import PASSK_LOG_FILENAME, measure_passk_stage

    calls: list = []
    _fake_trials(monkeypatch, passed=None, calls=calls)
    log_path = tmp_path / PASSK_LOG_FILENAME
    infra: dict = {}
    outcomes = measure_passk_stage(
        _task_dirs(tmp_path, ["t1"]),
        "terminus-2",
        "qwen3-8b",
        tmp_path / "jobs",
        2,
        stratum="stage1",
        max_workers=1,
        infra_failures=infra,
        log_path=log_path,
    )
    assert outcomes == []
    assert infra == {"t1": 2}
    lines = [json.loads(x) for x in log_path.read_text().splitlines()]
    assert len(lines) == 2
    assert all(x["passed"] is None and x["infra_failure"] is True for x in lines)
