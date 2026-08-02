"""`passk` and `import-gym` — evaluation entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .._shared import _load_traces


def cmd_passk(args: argparse.Namespace) -> int:
    from ...passk import PASSK_LOG_FILENAME, two_stage_sweep

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
