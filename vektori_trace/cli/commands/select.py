"""`select` — pick training tasks and write the held-out split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ...passrate import measure_pass_rates
from ...select import held_out_split, select_training_tasks, write_selection_report
from .._shared import _load_traces


def cmd_select(args: argparse.Namespace) -> int:
    """Step 6's first question: of the tasks the diagnosed deficit was lacking
    in, which does the candidate pass 10-40% of the time — the band rejection
    sampling and GRPO both need, neither empty nor already-solved."""
    diagnosis_path = Path(args.diagnosis)
    diagnosis = json.loads(diagnosis_path.read_text())
    chosen = diagnosis.get("chosen_deficit")
    if chosen is None:
        print(
            f"error: {diagnosis_path} has no chosen_deficit — nothing to select "
            "training tasks for. Run `diagnose` again with thresholds that clear a "
            "deficit, or accept there's none to train against yet.",
            file=sys.stderr,
        )
        return 1

    replay = diagnosis.get("replay")
    if not replay:
        print(
            f"error: {diagnosis_path} wasn't produced with --frontier-model/"
            "--candidate-model — `select` needs the cross-model/within-model split "
            "to know whose losses the deficit was measured on.",
            file=sys.stderr,
        )
        return 2
    frontier_model, candidate_model = replay["frontier_model"], replay["candidate_model"]

    traces = _load_traces(Path(args.manifest))
    trace_by_run_id = {t.run_id: t for t in traces}
    lacking_loss_tasks = [
        trace_by_run_id[rid].task
        for rid in chosen["lacking_loss_run_ids"]
        if rid in trace_by_run_id and trace_by_run_id[rid].task is not None
    ]
    if not lacking_loss_tasks:
        print(
            "error: none of the chosen deficit's lacking-loss run ids resolve to a "
            f"task in {args.manifest} — is this the same manifest `diagnose` used?",
            file=sys.stderr,
        )
        return 1

    tasks_dir = Path(args.tasks_dir)
    unique_tasks = list(dict.fromkeys(lacking_loss_tasks))
    task_dirs = [tasks_dir / t for t in unique_tasks if (tasks_dir / t).is_dir()]
    missing = set(unique_tasks) - {p.name for p in task_dirs}
    if missing:
        print(
            f"  {len(missing)} lacking-loss task(s) not found under {tasks_dir}, "
            "skipping: " + ", ".join(sorted(missing)[:5]) + (" ..." if len(missing) > 5 else ""),
            file=sys.stderr,
        )

    out_dir = Path(args.out)
    print(
        f"Measuring pass rate for {candidate_model} on {len(task_dirs)} lacking-loss "
        f"task(s), {args.rollouts} rollout(s) each..."
    )
    pass_rates = measure_pass_rates(
        task_dirs, agent=args.agent, model=candidate_model, jobs_dir=out_dir / "jobs", rollouts=args.rollouts
    )

    band = (args.passrate_min, args.passrate_max)
    selected = select_training_tasks(lacking_loss_tasks, pass_rates, band=band)
    print(f"  {len(selected)}/{len(unique_tasks)} task(s) land in band {band}")

    exclude: set[str] = set()
    if args.exclude:
        exclude = {
            line.strip() for line in Path(args.exclude).read_text().splitlines() if line.strip()
        }
    train_ids, holdout_ids = held_out_split(
        selected, exclude=exclude, holdout_frac=args.holdout_frac, seed=args.seed
    )

    md_path = write_selection_report(
        out_dir,
        frontier_model=frontier_model,
        candidate_model=candidate_model,
        agent=args.agent,
        band=band,
        rollouts=args.rollouts,
        seed=args.seed,
        holdout_frac=args.holdout_frac,
        pass_rates=pass_rates,
        lacking_loss_tasks=lacking_loss_tasks,
        selected=selected,
        train_ids=train_ids,
        holdout_ids=holdout_ids,
        exclude=exclude,
    )
    print(f"train={len(train_ids)}  holdout={len(holdout_ids)} (frac={args.holdout_frac}, seed={args.seed})")
    print(f"Selection report written to {md_path}")

    if not selected:
        print(
            "\nEmpty band: nothing trainable at this model+scaffold with the current "
            "deficit (V0_PLAN.md Step 6 stop condition).",
            file=sys.stderr,
        )
        return 1
    return 0
