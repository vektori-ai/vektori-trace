"""`passk` and `import-gym` — evaluation entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .._args import _add_endpoint_args, _positive_int_arg
from .._shared import _load_traces


def cmd_passk(args: argparse.Namespace) -> int:
    from ...evaluate.passk import PASSK_LOG_FILENAME, two_stage_sweep

    tasks_dir = Path(args.tasks_dir)
    task_dirs = sorted(
        p for p in tasks_dir.iterdir() if p.is_dir() and (p / "task.toml").exists()
    )
    if not task_dirs:
        print(f"error: no tasks in {tasks_dir}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # PLAN.md aggregates by (capability, model). Without the diagnosis join the
    # sweep can only report per task, which is the observation unit, not the
    # reporting unit — so say so rather than printing a bare per-task dump.
    task_to_capability: dict[str, str] | None = None
    if args.diagnosis and args.manifest:
        from ...routing import task_capability_map

        diagnosis = json.loads(Path(args.diagnosis).read_text())
        run_to_task = {
            t.run_id: t.task for t in _load_traces(Path(args.manifest)) if t.task
        }
        caps = task_capability_map(diagnosis, run_to_task)
        # One capability per task for aggregation; a task lacking several is
        # aggregated under its top-ranked one (the report keeps the full map).
        task_to_capability = {t: c[0] for t, c in caps.items() if c}
    elif args.diagnosis or args.manifest:
        print(
            "error: --diagnosis and --manifest must be given together",
            file=sys.stderr,
        )
        return 2

    report = two_stage_sweep(
        task_dirs,
        agent=args.agent,
        model=args.model,
        jobs_dir=out / "passk_jobs",
        stage1_n=args.stage1_n,
        stage2_n=args.stage2_n,
        api_base=args.api_base,
        model_info=args.model_info,
        task_to_capability=task_to_capability,
        max_workers=args.max_workers,
        escalate=not args.no_escalate,
        log_path=out / PASSK_LOG_FILENAME,
        **(
            {"timeout_sec": args.timeout_sec}
            if getattr(args, "timeout_sec", None)
            else {}
        ),
    )
    path = out / "passk.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"passk report: {path}")
    print(f"rollout log:  {out / PASSK_LOG_FILENAME}")
    print(
        f"escalated: {len(report['escalated'])}  "
        f"luck_quarantine: {len(report['luck_quarantine'])}"
    )
    support_counts: dict[str, int] = {}
    for cls in report["support"].values():
        support_counts[cls] = support_counts.get(cls, 0) + 1
    print("support: " + json.dumps(support_counts))
    if report["no_gradeable_rollouts"]:
        print(
            f"warning: {len(report['no_gradeable_rollouts'])} task(s) produced no "
            "gradeable rollout (all infra failures)",
            file=sys.stderr,
        )
    if task_to_capability is None:
        print(
            "note: no --diagnosis/--manifest, so no (capability, model) aggregation "
            "— per-task curves only",
            file=sys.stderr,
        )
    return 0


def cmd_import_gym(args: argparse.Namespace) -> int:
    from ...gym_import import import_gym

    result = import_gym(Path(args.source), Path(args.out), limit=args.limit)
    print(f"imported {len(result.tasks)} tasks → {args.out}")
    if result.skipped:
        print(f"skipped {len(result.skipped)} record(s) with no runnable oracle:")
        for iid, reason in sorted(result.skipped.items())[:10]:
            print(f"  {iid}: {reason}")
        if len(result.skipped) > 10:
            print(f"  ... and {len(result.skipped) - 10} more")
    if not result.tasks:
        print(
            "error: nothing importable — every record lacked an image, F2P set, "
            "base_commit or test_patch",
            file=sys.stderr,
        )
        return 1
    return 0

def register_passk(sub: argparse._SubParsersAction) -> None:
    """Register the `passk` subcommand on `sub`."""
    p_passk = sub.add_parser(
        "passk",
        help="Step C: two-stage pass@k sweep (n=8, escalate zeros to n=32); never pools strata",
    )
    p_passk.add_argument("--tasks-dir", required=True)
    p_passk.add_argument("--agent", required=True)
    p_passk.add_argument("--model", required=True)
    p_passk.add_argument("--out", default="./vektori-out")
    p_passk.add_argument("--stage1-n", type=_positive_int_arg, default=8)
    p_passk.add_argument("--stage2-n", type=_positive_int_arg, default=32)
    # ~1,300 containerised rollouts of minutes each; serially that is days,
    # which does not fit "nothing expensive precedes the gate".
    p_passk.add_argument("--max-workers", type=_positive_int_arg, default=1)
    p_passk.add_argument(
        "--no-escalate",
        action="store_true",
        help=(
            "run stage 1 only. Escalation fires on c == 0 regardless of how big "
            "stage 1 was, so a small --stage1-n silently becomes --stage2-n "
            "rollouts the moment a task fails"
        ),
    )
    p_passk.add_argument(
        "--timeout-sec",
        type=_positive_int_arg,
        default=None,
        help=(
            "wall-clock ceiling per rollout (default 5400). A rollout that "
            "exceeds it is logged with timed_out=true and excluded from the "
            "denominator, instead of aborting the sweep"
        ),
    )
    _add_endpoint_args(p_passk)
    p_passk.add_argument(
        "--diagnosis",
        default=None,
        help="report.json from `diagnose`; with --manifest, aggregates by (capability, model)",
    )
    p_passk.add_argument(
        "--manifest", default=None, help="replay manifest.json (run_id → task)"
    )
    p_passk.set_defaults(func=cmd_passk)

def register_import_gym(sub: argparse._SubParsersAction) -> None:
    """Register the `import-gym` subcommand on `sub`."""
    p_gym = sub.add_parser(
        "import-gym",
        help="Step B: import R2E-Gym/SWE-smith JSONL into harbor task dirs",
    )
    p_gym.add_argument("--source", required=True, help="JSONL of gym instances")
    p_gym.add_argument("--out", required=True, help="output tasks directory")
    p_gym.add_argument("--limit", type=int, default=None)
    p_gym.set_defaults(func=cmd_import_gym)
