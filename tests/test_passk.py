"""Step C — pass@k estimator, escalation, aggregation, luck controls."""

from __future__ import annotations

import math

import pytest

from vektori_trace.passk import (
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
    from vektori_trace.passk import classify_support

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
    from vektori_trace.passk import LuckResolution, resolve_luck_quarantine

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
    from vektori_trace.passk import LuckResolution

    assert LuckResolution("t", 32, 2, patch_matches_gold=True).resolved
    assert not LuckResolution("t", 32, 0, patch_matches_gold=True).resolved
    assert not LuckResolution("t", 32, 2, patch_matches_gold=False).resolved
    assert not LuckResolution("t", 32, 2).resolved
