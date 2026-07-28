"""Step F — capability routing rule (PLAN.md).

Pre-registered thresholds (fixed before data):
- pass@1 low  ≤ 0.25
- pass@1 high ≥ 0.75
- pass@32 = 0 → 0/32 after luck controls
- teacher pass@32 ≈ 0 → ≤ 1/32

Every (task × capability) gets exactly one of {RL, OPD, QUARANTINE, NONE}.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

Route = Literal["RL", "OPD", "QUARANTINE", "NONE"]
QuarantineCause = Literal[
    "broken_task", "frontier_limited", "luck_pending", "underpowered", "unspecified"
]

PASS1_LOW = 0.25
PASS1_HIGH = 0.75
# A *count*, stated against a reference stratum of 32: at most one of 32 rollouts
# passed. It is deliberately NOT the `pass32` curve — at k=n the estimator is
# degenerate (c=0 → 0.0, c≥1 → 1.0), so pass@32 cannot express "≤ 1/32" and
# comparing the two is a unit error. Only the raw counts decide this rule.
TEACHER_PASS32_NEAR_ZERO_COUNT = 1
TEACHER_PASS32_REFERENCE_N = 32
# The rule PLAN.md states is "≤ 1/32", and the sweep does not always take exactly
# 32 samples (`--stage2-n` is a flag). Comparing `c32 <= 1` against a stratum of
# 16 silently applies a 1/16 rule — twice as permissive as the pre-registration.
# Choice made here: generalise to the *rate* 1/32, evaluated in exact integer
# arithmetic (c32 * 32 <= 1 * n32). At n32 = 32 this is identical to the count
# rule, so the pre-registered threshold is unchanged where it was measured as
# written; at any other stratum size it means what PLAN.md says rather than what
# the sample size happened to be.
TEACHER_PASS32_NEAR_ZERO_RATE = TEACHER_PASS32_NEAR_ZERO_COUNT / TEACHER_PASS32_REFERENCE_N


@dataclass(frozen=True)
class CurveSummary:
    """Support curve summary for one model on one task (or cell)."""

    pass1: float | None
    pass32: float | None
    n32: int = 0
    c32: int = 0
    luck_quarantine: bool = False


ROUTING_RULES = {
    "R0_luck_pending": "only passes at k>8 — held for an independent n=32 re-sample",
    "R1_in_support_unreliable": "pass@1 ≤ low, pass@32 > 0, teacher has support → RL",
    "R2_outside_student_support": "pass@32 = 0, teacher has support → OPD",
    "R3_frontier_limited": "pass@32 = 0 and teacher pass@32 ≈ 0 → QUARANTINE",
    "R4_no_deficit": "student pass@1 ≥ high → NONE",
    "R5_teacher_no_support": "student in support, teacher pass@32 ≈ 0 → QUARANTINE",
    "R6_underpowered": "support never measured for this cell → QUARANTINE, no claim",
    "R7_mid_band_extension": (
        "low < pass@1 < high with support → RL. NOT pre-registered: PLAN.md's "
        "table names only the ≤ low and ≥ high cases."
    ),
}
PREREGISTERED_RULES = frozenset(ROUTING_RULES) - {"R7_mid_band_extension"}


@dataclass
class RoutingDecision:
    task: str
    capability: str
    route: Route
    student: CurveSummary
    teacher: CurveSummary
    thresholds: dict[str, float]
    quarantine_cause: QuarantineCause | None = None
    evidence: str = ""
    held_for_luck: bool = False
    # Which row of the decision table produced this, and whether that row was
    # part of the pre-registration. Acceptance criterion 5 asks for the rule and
    # thresholds in the manifest; a rule invented after the table was written has
    # to be visible as such rather than reading like the registered rule.
    rule: str = ""
    preregistered: bool = True

    def __post_init__(self) -> None:
        # Derived, never hand-set: a literal at each return site drifts silently
        # from ROUTING_RULES the moment a rule is added or reclassified.
        if self.rule:
            self.preregistered = self.rule in PREREGISTERED_RULES


def _pass32_is_zero(curve: CurveSummary) -> bool | None:
    """True = proven outside support, False = proven inside, None = not measured.

    Escalation to n=32 is what settles this. Without it, one passing rollout at
    any k still proves support (c > 0 ⇒ pass@1 > 0), but *no* passes at n=8 prove
    nothing about n=32 — that cell is underpowered, not out-of-support.
    """
    if curve.n32 > 0:
        return curve.c32 == 0
    if curve.pass1 is not None and curve.pass1 > 0:
        return False
    return None


def _teacher_near_zero(curve: CurveSummary) -> bool | None:
    """True = teacher support ≈ absent (≤1/32), False = present, None = unmeasured.

    Rate comparison, not a bare count — the escalation stratum is not guaranteed
    to be 32 rollouts (see `TEACHER_PASS32_NEAR_ZERO_RATE`). Integer arithmetic so
    the n32 = 32 boundary is exact.
    """
    if curve.n32 > 0:
        return (
            curve.c32 * TEACHER_PASS32_REFERENCE_N
            <= TEACHER_PASS32_NEAR_ZERO_COUNT * curve.n32
        )
    if curve.pass1 is not None and curve.pass1 > 0:
        # The teacher solved it at least once in the stage-1 sample, so it is not
        # the ≤1/32 case; no escalation was owed.
        return False
    return None


def route_cell(
    task: str,
    capability: str,
    student: CurveSummary,
    teacher: CurveSummary,
    *,
    pass1_low: float = PASS1_LOW,
    pass1_high: float = PASS1_HIGH,
    quarantine_cause: QuarantineCause | None = None,
) -> RoutingDecision:
    """Assign exactly one route by the pre-registered table."""
    thresholds = {
        "pass1_low": pass1_low,
        "pass1_high": pass1_high,
        "teacher_pass32_near_zero_count": float(TEACHER_PASS32_NEAR_ZERO_COUNT),
        "teacher_pass32_near_zero_reference_n": float(TEACHER_PASS32_REFERENCE_N),
        "teacher_pass32_near_zero_rate": TEACHER_PASS32_NEAR_ZERO_RATE,
    }

    # Luck control: only-passes-at-k>8 held until independent n=32 re-sample.
    if student.luck_quarantine:
        return RoutingDecision(
            task=task,
            capability=capability,
            route="QUARANTINE",
            student=student,
            teacher=teacher,
            thresholds=thresholds,
            quarantine_cause="luck_pending",
            rule="R0_luck_pending",
            evidence="only passes at k>8 — held pending independent n=32 re-sample",
            held_for_luck=True,
        )

    s_p1 = student.pass1
    t_p1 = teacher.pass1
    s_zero32 = _pass32_is_zero(student)
    t_near0 = _teacher_near_zero(teacher)

    # R4 NONE: the student already solves it reliably. The teacher's rate is
    # irrelevant here — a task the student passes at ≥ pass1_high has no deficit
    # to train, whatever the teacher does. (Requiring the teacher to be high too
    # sent student-high/teacher-weak cells to QUARANTINE, i.e. labelled a task
    # the student reliably solves as unfixable by any intervention.)
    if s_p1 is not None and s_p1 >= pass1_high:
        note = (
            ""
            if t_p1 is None or t_p1 >= pass1_high
            else f"; note teacher pass@1={t_p1:.3f} < {pass1_high}"
        )
        return RoutingDecision(
            task=task,
            capability=capability,
            route="NONE",
            student=student,
            teacher=teacher,
            thresholds=thresholds,
            rule="R4_no_deficit",
            evidence=f"student pass@1={s_p1:.3f} ≥ {pass1_high}; no deficit{note}",
        )

    # Underpowered: the support question was never measured for this cell. PLAN.md
    # error table — reported as underpowered, no claim made from it.
    if s_zero32 is None or t_near0 is None:
        # Every undetermined side is named. When both are undetermined, reporting
        # only the student understates what is missing from the cell.
        missing = " and ".join(
            name
            for name, determined in (("student", s_zero32), ("teacher", t_near0))
            if determined is None
        )
        return RoutingDecision(
            task=task,
            capability=capability,
            route="QUARANTINE",
            student=student,
            teacher=teacher,
            thresholds=thresholds,
            quarantine_cause="underpowered",
            rule="R6_underpowered",
            evidence=(
                f"{missing} support undetermined: 0 passes at n=8 and no n=32 "
                "escalation — no routing claim from this cell"
            ),
        )

    # QUARANTINE: student pass@32=0 and teacher also ≈0. The measurement itself
    # establishes frontier-limitation; `broken_task` is the label a human applies
    # on inspection when the task turns out to be impossible or malformed, so it
    # only ever arrives through the caller-supplied `quarantine_cause`.
    if s_zero32 and t_near0:
        cause = quarantine_cause or "frontier_limited"
        return RoutingDecision(
            task=task,
            capability=capability,
            route="QUARANTINE",
            student=student,
            teacher=teacher,
            thresholds=thresholds,
            quarantine_cause=cause,
            rule="R3_frontier_limited",
            evidence=(
                f"student pass@32=0 and teacher pass@32≈0 "
                f"(c32={teacher.c32}/{teacher.n32}); cause={cause}"
            ),
        )

    # R2 OPD: student pass@32=0, teacher has support.
    if s_zero32:
        return RoutingDecision(
            task=task,
            capability=capability,
            route="OPD",
            student=student,
            teacher=teacher,
            thresholds=thresholds,
            rule="R2_outside_student_support",
            evidence="student outside support (pass@32=0); teacher has support",
        )

    # Student is in support from here on (s_zero32 is False).

    # R5 QUARANTINE: student has support but the teacher does not. There is
    # nothing to distil from, and RL on a task the teacher cannot demonstrate is
    # the frontier-limited case with a luckier student.
    if t_near0:
        return RoutingDecision(
            task=task,
            capability=capability,
            route="QUARANTINE",
            student=student,
            teacher=teacher,
            thresholds=thresholds,
            quarantine_cause=quarantine_cause or "frontier_limited",
            rule="R5_teacher_no_support",
            evidence=(
                "student in support but teacher pass@32≈0 "
                f"(c32={teacher.c32}/{teacher.n32}) — nothing to distil from"
            ),
        )

    # R1 RL: in support, unreliable.
    if s_p1 is not None and s_p1 <= pass1_low:
        return RoutingDecision(
            task=task,
            capability=capability,
            route="RL",
            student=student,
            teacher=teacher,
            thresholds=thresholds,
            rule="R1_in_support_unreliable",
            evidence=(
                f"in support but unreliable: student pass@1={s_p1:.3f} ≤ {pass1_low}, "
                f"pass@32>0"
            ),
        )

    # R7 RL (mid band). PLAN.md's table names only pass@1 ≤ low and pass@1 ≥ high;
    # (low, high) is genuinely outside the pre-registration. It is resolved the
    # same way as R1 — the deciding fact is support, and a mid-band student has
    # it — but the decision is tagged so a reader can filter every cell that
    # relied on an extension of the registered rule rather than the rule itself.
    return RoutingDecision(
        task=task,
        capability=capability,
        route="RL",
        student=student,
        teacher=teacher,
        thresholds=thresholds,
        rule="R7_mid_band_extension",
        evidence=(
            f"student pass@1={s_p1 if s_p1 is None else round(s_p1, 3)} in "
            f"({pass1_low}, {pass1_high}) with support — routed RL by extension, "
            "not by the pre-registered table"
        ),
    )


def route_all(
    cells: list[tuple[str, str, CurveSummary, CurveSummary]],
    *,
    quarantine_causes: dict[str, QuarantineCause] | None = None,
) -> list[RoutingDecision]:
    """Route every (task, capability) cell."""
    causes = quarantine_causes or {}
    return [
        route_cell(
            task,
            cap,
            student,
            teacher,
            quarantine_cause=causes.get(task),
        )
        for task, cap, student, teacher in cells
    ]


def task_capability_map(
    diagnosis: dict[str, Any],
    run_id_to_task: Mapping[str, str],
    *,
    only_chosen: bool = False,
) -> dict[str, list[str]]:
    """task id → capability ids `diagnose.py` marked LACKING on that task's losses.

    PLAN.md's aggregation unit is (capability, model), and routing is per
    (task × capability) — so the cells have to come from the diagnosis report,
    not from one label applied to every task. Joins `lacking_loss_run_ids` from
    each scored capability against the replay manifest's run_id → task mapping.
    """
    scores: list[dict[str, Any]]
    if only_chosen:
        chosen = diagnosis.get("chosen_deficit")
        scores = [chosen] if chosen else []
    else:
        scores = list(diagnosis.get("all_deficits_ranked") or [])
        if not scores and diagnosis.get("chosen_deficit"):
            scores = [diagnosis["chosen_deficit"]]

    out: dict[str, list[str]] = {}
    for score in scores:
        cap = (score.get("capability") or {}).get("id") or (
            score.get("capability") or {}
        ).get("name")
        if not cap:
            continue
        for run_id in score.get("lacking_loss_run_ids") or []:
            task = run_id_to_task.get(run_id)
            if task is None:
                continue
            caps = out.setdefault(task, [])
            if cap not in caps:
                caps.append(cap)
    return {task: sorted(caps) for task, caps in out.items()}


def per_cell_counts(decisions: list[RoutingDecision]) -> dict[str, Any]:
    """Per-route counts with N — underpowered cells are reportable."""
    counts: dict[str, int] = {"RL": 0, "OPD": 0, "QUARANTINE": 0, "NONE": 0}
    by_cap: dict[str, dict[str, int]] = {}
    by_rule: dict[str, int] = {}
    by_cause: dict[str, int] = {}
    for d in decisions:
        counts[d.route] = counts.get(d.route, 0) + 1
        by_cap.setdefault(d.capability, {"RL": 0, "OPD": 0, "QUARANTINE": 0, "NONE": 0})
        by_cap[d.capability][d.route] += 1
        by_rule[d.rule] = by_rule.get(d.rule, 0) + 1
        if d.quarantine_cause:
            by_cause[d.quarantine_cause] = by_cause.get(d.quarantine_cause, 0) + 1
    return {
        "N": len(decisions),
        "by_route": counts,
        "by_capability": by_cap,
        "by_rule": by_rule,
        "quarantine_by_cause": by_cause,
        # Cells decided by a rule that was not pre-registered, so a reader can
        # re-run the comparison without them.
        "not_preregistered": sum(1 for d in decisions if not d.preregistered),
    }


def decision_to_dict(d: RoutingDecision) -> dict[str, Any]:
    return {
        "task": d.task,
        "capability": d.capability,
        "route": d.route,
        "student": asdict(d.student),
        "teacher": asdict(d.teacher),
        "thresholds": d.thresholds,
        "quarantine_cause": d.quarantine_cause,
        "evidence": d.evidence,
        "held_for_luck": d.held_for_luck,
        "rule": d.rule,
        "preregistered": d.preregistered,
    }


__all__ = [
    "PASS1_HIGH",
    "PASS1_LOW",
    "PREREGISTERED_RULES",
    "ROUTING_RULES",
    "TEACHER_PASS32_NEAR_ZERO_COUNT",
    "TEACHER_PASS32_NEAR_ZERO_RATE",
    "TEACHER_PASS32_REFERENCE_N",
    "CurveSummary",
    "QuarantineCause",
    "Route",
    "RoutingDecision",
    "decision_to_dict",
    "per_cell_counts",
    "route_all",
    "route_cell",
    "task_capability_map",
]
