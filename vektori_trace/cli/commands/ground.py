"""`ground` — grounding report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ...evaluate.diagnose import DEFAULT_MIN_GAP
from .._args import _min_gap_arg


def cmd_ground(args: argparse.Namespace) -> int:
    """Step E — compare diagnose labels against execution-located forking steps."""
    from ...grounding import GroundingPair, ground_diagnosis, report_to_dict

    bisection = json.loads(Path(args.bisection).read_text())
    diagnosis = json.loads(Path(args.diagnosis).read_text())
    judgments: dict[str, bool] = {}
    if args.judgments:
        judgments = {
            k: bool(v) for k, v in json.loads(Path(args.judgments).read_text()).items()
        }

    gap_by_cap = {}
    for score in diagnosis.get("all_deficits_ranked") or []:
        cap = (score.get("capability") or {}).get("id")
        if cap:
            gap_by_cap[cap] = score.get("gap")
    run_to_cap: dict[str, str] = {}
    for score in diagnosis.get("all_deficits_ranked") or []:
        cap = (score.get("capability") or {}).get("id")
        for rid in score.get("lacking_loss_run_ids") or []:
            run_to_cap.setdefault(rid, cap)

    pairs = []
    for row in bisection.get("trajectories") or []:
        if row.get("dropped") or row.get("forking_step") is None:
            continue
        rid = row["run_id"]
        cap = run_to_cap.get(rid)
        if cap is None:
            continue
        pairs.append(
            GroundingPair(
                task=str(row.get("task")),
                capability=cap,
                forking_step=row["forking_step"],
                label_agrees=judgments.get(rid),
                diagnose_gap=gap_by_cap.get(cap),
            )
        )
    if not pairs:
        print(
            "error: no (forking step, capability) pair — is this the diagnosis "
            "and bisection from the same replay run?",
            file=sys.stderr,
        )
        return 1

    report = ground_diagnosis(pairs, current_min_gap=args.min_gap)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "grounding.json"
    path.write_text(json.dumps(report_to_dict(report), indent=2) + "\n")
    print(f"grounding report: {path}")
    print(
        f"pairs={report.n} agreement={report.agreement_rate} "
        f"blur={report.blur:.3f} suggested_min_gap={report.suggested_min_gap}"
    )
    if report.agreement_rate is None:
        print(
            "note: no human judgments supplied (--judgments), so agreement is "
            "unmeasured. PLAN.md acceptance criterion 4 needs 10 hand-inspected "
            "forking steps; the ids to inspect are in the report.",
            file=sys.stderr,
        )
    elif report.underpowered:
        print("warning: fewer than 10 judged pairs — underpowered", file=sys.stderr)
    return 0

def register_ground(sub: argparse._SubParsersAction) -> None:
    """Register the `ground` subcommand on `sub`."""
    p_ground = sub.add_parser(
        "ground",
        help="Step E: compare diagnose labels against execution-located forking steps",
    )
    p_ground.add_argument("--bisection", required=True, help="bisection.json from `bisect`")
    p_ground.add_argument("--diagnosis", required=True, help="report.json from `diagnose`")
    p_ground.add_argument(
        "--judgments",
        default=None,
        help='JSON {run_id: true|false} of hand-inspected agreement (AC #4)',
    )
    p_ground.add_argument("--min-gap", type=_min_gap_arg, default=DEFAULT_MIN_GAP)
    p_ground.add_argument("--out", default="./vektori-out")
    p_ground.set_defaults(func=cmd_ground)
