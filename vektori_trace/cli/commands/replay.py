"""`replay` — re-run a manifest and report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ...gap import compute_gap, format_rate, write_gap_report
from ...mining.miner import (
    HarborTraceRunner,
    collect_paired_traces,
    discover_tasks,
)
from ...schema import Trace


def cmd_replay(args: argparse.Namespace) -> int:
    """Run a frontier and a candidate model over the same mined tasks, on one
    pinned scaffold, and report the gap number — before any diagnosis runs."""
    if args.frontier_model == args.candidate_model:
        print(
            "error: --frontier-model and --candidate-model are the same "
            f"({args.frontier_model!r}) — there is no gap to measure between a "
            "model and itself.",
            file=sys.stderr,
        )
        return 2

    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.is_dir():
        print(f"error: --tasks-dir {tasks_dir} does not exist or is not a directory", file=sys.stderr)
        return 2
    task_dirs = discover_tasks(tasks_dir)
    if not task_dirs:
        print(f"error: no task.toml found under {tasks_dir}", file=sys.stderr)
        return 2
    print(f"Replaying {len(task_dirs)} mined task(s) from {tasks_dir}")
    print(f"Scaffold (pinned across both arms): {args.agent}")

    out_dir = Path(args.out)
    jobs_dir = out_dir / "jobs"
    # Only the candidate takes endpoint overrides. The frontier arm is the
    # ceiling being measured against, always a public API model — pointing it at
    # a self-hosted URL would mean the "frontier" number came from our own
    # server, which is the one thing it must not be.
    arms = [
        (args.frontier_model, HarborTraceRunner(agent=args.agent, jobs_dir=jobs_dir, model=args.frontier_model)),
        (
            args.candidate_model,
            HarborTraceRunner(
                agent=args.agent,
                jobs_dir=jobs_dir,
                model=args.candidate_model,
                api_base=args.candidate_api_base,
                model_info=args.candidate_model_info,
            ),
        ),
    ]

    traces_dir = out_dir / "replay_traces"
    manifest_path = out_dir / "replay-manifest.json"
    print(f"Running frontier ({args.frontier_model}) and candidate ({args.candidate_model}) against each task...")
    manifest = collect_paired_traces(task_dirs, arms, traces_dir, manifest_path=manifest_path)

    traces = [
        Trace.load(Path(m["path"]), outcome=m["outcome"], model=m["model"], task=m["task"])
        for m in manifest
    ]
    result = compute_gap(
        traces, frontier_model=args.frontier_model, candidate_model=args.candidate_model, agent=args.agent
    )

    frontier_skipped = len(task_dirs) - result.frontier_attempted
    candidate_skipped = len(task_dirs) - result.candidate_attempted
    print(
        f"\nfrontier ({args.frontier_model}): {format_rate(result.frontier_rate)} "
        f"({result.frontier_wins}/{result.paired_n} paired; "
        f"{result.frontier_attempted} attempted, {frontier_skipped} skipped)"
    )
    print(
        f"candidate ({args.candidate_model}): {format_rate(result.candidate_rate)} "
        f"({result.candidate_wins}/{result.paired_n} paired; "
        f"{result.candidate_attempted} attempted, {candidate_skipped} skipped)"
    )
    print(f"gap: {format_rate(result.gap)}  (paired tasks: {result.paired_n})")

    md_path = write_gap_report(result, out_dir)
    print(f"\nGap report written to {md_path}")

    if result.paired_n == 0:
        print(
            "\nerror: no task was judged by both arms — nothing to compare. Check the "
            "skip lines above for why each arm's attempts were excluded.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Next: vektori-trace diagnose --manifest {manifest_path} --out {args.out} "
        f"--frontier-model {args.frontier_model} --candidate-model {args.candidate_model}"
    )
    return 0
