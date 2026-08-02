"""`route` and `plan-b-arms` — capability routing decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .._shared import _load_traces


def cmd_route(args: argparse.Namespace) -> int:
    from ...routing import (
        ROUTING_RULES,
        CurveSummary,
        decision_to_dict,
        per_cell_counts,
        route_cell,
    )

    student = json.loads(Path(args.student_passk).read_text())
    teacher = json.loads(Path(args.teacher_passk).read_text())

    def _curve(block: dict, task: str) -> CurveSummary:
        # Prefer stage2 if present for this task, else stage1.
        row = (block.get("stage2") or {}).get(task) or (block.get("stage1") or {}).get(task)
        if not row:
            return CurveSummary(pass1=None, pass32=None)
        curves = row.get("curves") or {}
        # n32/c32 are the *actual* stage-2 stratum, which is `--stage2-n` and not
        # necessarily 32. routing.py compares them as a rate against 1/32, so the
        # pre-registered rule holds whatever sample size the sweep took.
        return CurveSummary(
            pass1=curves.get("1"),
            pass32=curves.get("32"),
            n32=int(row["n"]) if row.get("stratum") == "stage2" else 0,
            c32=int(row["c"]) if row.get("stratum") == "stage2" else 0,
            luck_quarantine=bool(row.get("luck_quarantine")),
        )

    tasks = sorted(
        set(student.get("stage1") or {})
        | set(student.get("stage2") or {})
        | set(teacher.get("stage1") or {})
        | set(teacher.get("stage2") or {})
    )

    # Cells are (task × capability). The capabilities come from the diagnosis
    # report joined to the replay manifest — one blanket label across every task
    # would make the per-capability counts meaningless.
    if args.diagnosis:
        from ...routing import task_capability_map

        if not args.manifest:
            print(
                "error: --diagnosis needs --manifest to map lacking-loss run ids "
                "back to tasks",
                file=sys.stderr,
            )
            return 2
        diagnosis = json.loads(Path(args.diagnosis).read_text())
        run_to_task = {
            t.run_id: t.task for t in _load_traces(Path(args.manifest)) if t.task
        }
        caps_by_task = task_capability_map(
            diagnosis, run_to_task, only_chosen=args.chosen_deficit_only
        )
        unlabelled = [t for t in tasks if not caps_by_task.get(t)]
        if unlabelled:
            print(
                f"  {len(unlabelled)} task(s) carry no LACKING capability in "
                f"{args.diagnosis}; excluded from routing",
                file=sys.stderr,
            )
    else:
        caps_by_task = {t: [args.capability] for t in tasks}

    decisions = [
        route_cell(t, cap, _curve(student, t), _curve(teacher, t))
        for t in tasks
        for cap in caps_by_task.get(t, [])
    ]
    if not decisions:
        print("error: no (task × capability) cells to route", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "thresholds": decisions[0].thresholds if decisions else {},
        "rules": ROUTING_RULES,
        "counts": per_cell_counts(decisions),
        "decisions": [decision_to_dict(d) for d in decisions],
    }
    path = out / "routing.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"routing report: {path}")
    print(json.dumps(report["counts"], indent=2))
    return 0


def _reload_routing_decisions(raw: dict) -> list:
    """Rebuild `RoutingDecision`s from a routing.json, provenance intact."""
    from ...routing import CurveSummary, RoutingDecision

    decisions = []
    for d in raw.get("decisions") or []:
        decisions.append(
            RoutingDecision(
                task=d["task"],
                capability=d["capability"],
                route=d["route"],
                student=CurveSummary(**d["student"]),
                teacher=CurveSummary(**d["teacher"]),
                thresholds=d.get("thresholds") or {},
                quarantine_cause=d.get("quarantine_cause"),
                evidence=d.get("evidence") or "",
                held_for_luck=bool(d.get("held_for_luck")),
                # Carry the rule through: without it every reloaded cell reads as
                # pre-registered, and the mid-band extension becomes invisible to
                # exactly the filtering it was tagged for.
                rule=d.get("rule") or "",
            )
        )
    return decisions

def cmd_plan_b_arms(args: argparse.Namespace) -> int:
    from ...arms import plan_b_arms

    decisions = _reload_routing_decisions(json.loads(Path(args.routing).read_text()))
    holdout = None
    if args.holdout:
        holdout = [
            line.strip()
            for line in Path(args.holdout).read_text().splitlines()
            if line.strip()
        ]
    if args.resolvable_effect_size is None:
        print(
            "warning: no --resolvable-effect-size recorded. PLAN.md requires the "
            "smallest resolvable effect to be stated before training; without it "
            "B1-vs-B2 cannot be interpreted afterwards.",
            file=sys.stderr,
        )
    plans = plan_b_arms(
        decisions,
        holdout=holdout,
        resolvable_effect_size=args.resolvable_effect_size,
        pilot=args.pilot,
        seed=args.seed,
        exclude_not_preregistered=args.preregistered_only,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        arm: {
            "arm": p.arm,
            "description": p.description,
            # Keyed by "<task>::<capability>" — the routing unit is the cell, not
            # the task, so the mix B2 has to hold identical is counted per cell.
            "assignments": p.assignments,
            "task_ids": p.task_ids,
            "cells": [list(c) for c in p.cells],
            "n_cells": len(p.cells),
            "method_mix": p.method_mix,
            "resolvable_effect_size": p.resolvable_effect_size,
        }
        for arm, p in plans.items()
    }
    path = out / "b_arms_plan.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"B-arm plan: {path}")
    return 0
