"""Step F — routing rule table + boundaries."""

from __future__ import annotations

from vektori_trace.routing import (
    PASS1_HIGH,
    PASS1_LOW,
    CurveSummary,
    per_cell_counts,
    route_cell,
)


def test_route_rl_in_support_unreliable() -> None:
    d = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.1, pass32=0.4, n32=32, c32=10),
        CurveSummary(pass1=0.9, pass32=0.95, n32=32, c32=30),
    )
    assert d.route == "RL"


def test_route_opd_outside_student_support() -> None:
    d = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.0, pass32=0.0, n32=32, c32=0),
        CurveSummary(pass1=0.8, pass32=0.9, n32=32, c32=28),
    )
    assert d.route == "OPD"


def test_route_quarantine_teacher_also_zero() -> None:
    d = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.0, pass32=0.0, n32=32, c32=0),
        CurveSummary(pass1=0.0, pass32=0.0, n32=32, c32=0),
        quarantine_cause="frontier_limited",
    )
    assert d.route == "QUARANTINE"
    assert d.quarantine_cause == "frontier_limited"


def test_route_none_no_deficit() -> None:
    d = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.9, pass32=0.95, n32=32, c32=30),
        CurveSummary(pass1=0.95, pass32=0.99, n32=32, c32=31),
    )
    assert d.route == "NONE"


def test_threshold_boundaries() -> None:
    # pass@1 exactly at low boundary → RL path (≤ low)
    d = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=PASS1_LOW, pass32=0.5, n32=32, c32=10),
        CurveSummary(pass1=0.9, pass32=0.9, n32=32, c32=28),
    )
    assert d.route == "RL"

    d2 = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=PASS1_HIGH, pass32=0.9, n32=32, c32=28),
        CurveSummary(pass1=PASS1_HIGH, pass32=0.9, n32=32, c32=28),
    )
    assert d2.route == "NONE"


def test_luck_holds_decision() -> None:
    d = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.0, pass32=0.1, n32=32, c32=2, luck_quarantine=True),
        CurveSummary(pass1=0.9, pass32=0.9, n32=32, c32=28),
    )
    assert d.route == "QUARANTINE"
    assert d.held_for_luck
    assert d.quarantine_cause == "luck_pending"


def test_teacher_near_zero_uses_counts_not_the_rate() -> None:
    """pass@32 at k=n is degenerate: c=1/32 gives 1.0, not 1/32. Comparing it to
    a 1/32 threshold inverts the teacher-support test."""
    d = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.0, pass32=0.0, n32=32, c32=0),
        # Teacher scraped one pass in 32 → ≈ no support → QUARANTINE, not OPD.
        CurveSummary(pass1=1 / 32, pass32=1.0, n32=32, c32=1),
    )
    assert d.route == "QUARANTINE"

    d2 = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.0, pass32=0.0, n32=32, c32=0),
        CurveSummary(pass1=2 / 32, pass32=1.0, n32=32, c32=2),
    )
    assert d2.route == "OPD"


def test_unmeasured_support_is_underpowered_not_a_route() -> None:
    # 0/8 with no escalation: nothing is known about pass@32 either way.
    d = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.0, pass32=None, n32=0, c32=0),
        CurveSummary(pass1=0.9, pass32=None, n32=0, c32=0),
    )
    assert d.route == "QUARANTINE"
    assert d.quarantine_cause == "underpowered"

    # Teacher side unmeasured.
    d2 = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.0, pass32=0.0, n32=32, c32=0),
        CurveSummary(pass1=0.0, pass32=None, n32=0, c32=0),
    )
    assert d2.route == "QUARANTINE"
    assert d2.quarantine_cause == "underpowered"


def test_stage1_pass_proves_support_without_escalation() -> None:
    # One pass at n=8 is proof of support; no n=32 sample is owed.
    d = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.125, pass32=None, n32=0, c32=0),
        CurveSummary(pass1=0.875, pass32=None, n32=0, c32=0),
    )
    assert d.route == "RL"


def test_per_cell_counts() -> None:
    cells = [
        route_cell(
            f"t{i}",
            "cap",
            CurveSummary(pass1=0.0, pass32=0.0, n32=32, c32=0),
            CurveSummary(pass1=0.9, pass32=0.9, n32=32, c32=28),
        )
        for i in range(3)
    ]
    counts = per_cell_counts(cells)
    assert counts["N"] == 3
    assert counts["by_route"]["OPD"] == 3


def test_student_high_teacher_low_is_no_deficit_not_quarantine() -> None:
    """A task the student reliably solves has nothing to train, whatever the
    teacher scores. Requiring the teacher to be high too labelled it unfixable."""
    d = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.9, pass32=1.0, n32=32, c32=30),
        CurveSummary(pass1=0.0, pass32=0.0, n32=32, c32=0),
    )
    assert d.route == "NONE"
    assert d.rule == "R4_no_deficit"
    assert "teacher" in d.evidence


def test_student_in_support_teacher_absent_is_quarantine() -> None:
    d = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.1, pass32=0.5, n32=32, c32=4),
        CurveSummary(pass1=0.0, pass32=0.0, n32=32, c32=1),
    )
    assert d.route == "QUARANTINE"
    assert d.rule == "R5_teacher_no_support"
    assert d.quarantine_cause == "frontier_limited"


def test_mid_band_is_flagged_as_not_preregistered() -> None:
    """PLAN.md's table names only pass@1 ≤ 0.25 and ≥ 0.75. The band between
    them is an extension and every cell decided by it must say so."""
    d = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.5, pass32=0.8, n32=32, c32=16),
        CurveSummary(pass1=0.9, pass32=1.0, n32=32, c32=30),
    )
    assert d.route == "RL"
    assert d.rule == "R7_mid_band_extension"
    assert d.preregistered is False

    registered = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.1, pass32=0.8, n32=32, c32=16),
        CurveSummary(pass1=0.9, pass32=1.0, n32=32, c32=30),
    )
    assert registered.rule == "R1_in_support_unreliable"
    assert registered.preregistered is True


def test_every_rule_id_is_documented() -> None:
    from vektori_trace.routing import PREREGISTERED_RULES, ROUTING_RULES

    cells = [
        CurveSummary(pass1=0.1, pass32=0.8, n32=32, c32=16),
        CurveSummary(pass1=0.0, pass32=0.0, n32=32, c32=0),
        CurveSummary(pass1=0.9, pass32=1.0, n32=32, c32=30),
        CurveSummary(pass1=0.5, pass32=0.8, n32=32, c32=16),
        CurveSummary(pass1=0.0, pass32=None, n32=0, c32=0),
        CurveSummary(pass1=0.0, pass32=0.1, n32=32, c32=2, luck_quarantine=True),
    ]
    seen = set()
    for s in cells:
        for t in cells:
            seen.add(route_cell("t", "cap", s, t).rule)
    # Equality, not subset: a subset assertion still passes if a rule goes dead.
    # R3 needs a caller-supplied cause and R6 an unmeasured teacher, both of
    # which the cross-product above reaches.
    assert seen == set(ROUTING_RULES)
    assert "R7_mid_band_extension" not in PREREGISTERED_RULES
    assert set(ROUTING_RULES) > PREREGISTERED_RULES


def test_counts_expose_rule_and_cause_breakdown() -> None:
    decisions = [
        route_cell(
            "t1",
            "cap",
            CurveSummary(pass1=0.5, pass32=0.8, n32=32, c32=16),
            CurveSummary(pass1=0.9, pass32=1.0, n32=32, c32=30),
        ),
        route_cell(
            "t2",
            "cap",
            CurveSummary(pass1=0.0, pass32=0.0, n32=32, c32=0),
            CurveSummary(pass1=0.9, pass32=1.0, n32=32, c32=30),
        ),
    ]
    counts = per_cell_counts(decisions)
    assert counts["not_preregistered"] == 1
    assert counts["by_rule"]["R2_outside_student_support"] == 1


def test_task_capability_map_joins_diagnosis_to_tasks() -> None:
    from vektori_trace.routing import task_capability_map

    diagnosis = {
        "all_deficits_ranked": [
            {
                "capability": {"id": "cap_a", "name": "A"},
                "lacking_loss_run_ids": ["r1", "r2"],
            },
            {
                "capability": {"id": "cap_b", "name": "B"},
                "lacking_loss_run_ids": ["r2", "r3"],
            },
        ],
        "chosen_deficit": {
            "capability": {"id": "cap_a", "name": "A"},
            "lacking_loss_run_ids": ["r1", "r2"],
        },
    }
    run_to_task = {"r1": "task1", "r2": "task2", "r3": "task3"}
    assert task_capability_map(diagnosis, run_to_task) == {
        "task1": ["cap_a"],
        "task2": ["cap_a", "cap_b"],
        "task3": ["cap_b"],
    }
    assert task_capability_map(diagnosis, run_to_task, only_chosen=True) == {
        "task1": ["cap_a"],
        "task2": ["cap_a"],
    }
    # Run ids with no task in the manifest are dropped, not invented.
    assert task_capability_map(diagnosis, {"r1": "task1"}) == {"task1": ["cap_a"]}


def test_teacher_near_zero_is_a_rate_not_a_raw_count_at_other_stratum_sizes() -> None:
    """PLAN.md's rule is "teacher pass@32 ≈ 0 is ≤ 1/32". `--stage2-n` is a flag,
    so the stratum is not always 32; a bare `c32 <= 1` test at n=16 applies a
    1/16 rule, twice as permissive as the pre-registration."""
    from vektori_trace.routing import TEACHER_PASS32_NEAR_ZERO_RATE

    assert TEACHER_PASS32_NEAR_ZERO_RATE == 1 / 32

    # 1 pass in 16 is 1/16 > 1/32 → the teacher HAS support → OPD, not QUARANTINE.
    d = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.0, pass32=0.0, n32=16, c32=0),
        CurveSummary(pass1=1 / 16, pass32=1.0, n32=16, c32=1),
    )
    assert d.route == "OPD", d.evidence

    # 1 pass in 64 is 1/64 ≤ 1/32 → still ≈ no teacher support.
    d2 = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.0, pass32=0.0, n32=64, c32=0),
        CurveSummary(pass1=1 / 64, pass32=1.0, n32=64, c32=1),
    )
    assert d2.route == "QUARANTINE"

    # The n=32 boundary the rule was written against is unchanged in both directions.
    at32 = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.0, pass32=0.0, n32=32, c32=0),
        CurveSummary(pass1=1 / 32, pass32=1.0, n32=32, c32=1),
    )
    assert at32.route == "QUARANTINE"
    just_over = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.0, pass32=0.0, n32=32, c32=0),
        CurveSummary(pass1=2 / 32, pass32=1.0, n32=32, c32=2),
    )
    assert just_over.route == "OPD"


def test_underpowered_evidence_names_every_undetermined_side() -> None:
    """Both sides unmeasured: naming only the student understates the cell."""
    both = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.0, pass32=None, n32=0, c32=0),
        CurveSummary(pass1=0.0, pass32=None, n32=0, c32=0),
    )
    assert both.quarantine_cause == "underpowered"
    assert "student" in both.evidence
    assert "teacher" in both.evidence

    student_only = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.0, pass32=None, n32=0, c32=0),
        CurveSummary(pass1=0.9, pass32=None, n32=0, c32=0),
    )
    assert "student" in student_only.evidence
    assert "teacher" not in student_only.evidence

    teacher_only = route_cell(
        "t",
        "cap",
        CurveSummary(pass1=0.0, pass32=0.0, n32=32, c32=0),
        CurveSummary(pass1=0.0, pass32=None, n32=0, c32=0),
    )
    assert "teacher" in teacher_only.evidence
    assert "student" not in teacher_only.evidence
