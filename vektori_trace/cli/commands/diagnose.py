"""`diagnose` — label traces, score deficits, propose capabilities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ...evaluate.diagnose import (
    DEFAULT_MIN_GAP,
    DEFAULT_MIN_SUPPORT,
    diagnose_replay,
    label_trace,
    propose_capabilities,
    score_deficits,
    select_deficit,
)
from ...evaluate.report import build_replay_report, build_report, write_report
from ...evaluate.validity import prove_validity
from ...taskgen import scaffold_task
from .._args import _min_gap_arg, _min_support_arg
from .._shared import _check_replay_models, _load_traces


def cmd_diagnose(args: argparse.Namespace) -> int:
    traces = _load_traces(Path(args.manifest))
    wins = [t for t in traces if t.outcome == "win"]
    losses = [t for t in traces if t.outcome == "loss"]
    if not wins or not losses:
        print(
            "error: need at least one 'win' and one 'loss' trace to compute the "
            "contrastive scoring to work.",
            file=sys.stderr,
        )
        return 1

    problem = _check_replay_models(args, traces)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    replay_mode = args.frontier_model is not None

    print(f"Loaded {len(wins)} win(s) and {len(losses)} loss(es).")
    if replay_mode:
        print(
            f"Two contrasts: cross-model (frontier {args.frontier_model} wins vs "
            f"candidate {args.candidate_model} losses) and within-model ({args.candidate_model})."
        )
    print("Proposing candidate capabilities...")
    capabilities = propose_capabilities(traces, model=args.model)
    for c in capabilities:
        print(f"  - {c.id}: {c.name}")

    print("Labeling each trace against candidate capabilities...")
    trace_labels = [label_trace(t, capabilities, model=args.model) for t in traces]

    diagnosis = None
    if replay_mode:
        diagnosis = diagnose_replay(
            trace_labels,
            capabilities,
            frontier_model=args.frontier_model,
            candidate_model=args.candidate_model,
            min_gap=args.min_gap,
            min_support=args.min_support,
        )
        scores, top = diagnosis.cross_model_scores, diagnosis.chosen
    else:
        scores = score_deficits(capabilities, trace_labels)
        top = select_deficit(scores, min_gap=args.min_gap, min_support=args.min_support)

    thresholds = {"min_gap": args.min_gap, "min_support": args.min_support}
    out_dir = Path(args.out)

    def write(task_dir: Path | None, validity: dict | None) -> Path:
        report = (
            build_replay_report(diagnosis, task_dir, validity, thresholds)
            if diagnosis is not None
            else build_report(top, scores, task_dir, validity, thresholds)
        )
        return write_report(report, out_dir)

    print("\nRanked candidates:")
    for s in scores:
        print(
            f"  {s.capability.name}: priority={_fmt(s.priority)}, gap={_fmt(s.gap)}, "
            f"prevalence={s.prevalence:.2f}, N={s.n_relevant_wins}w/{s.n_relevant_losses}l"
        )

    if top is None:
        # A clean, honest exit — not an error. Nothing here separates wins from
        # losses well enough to build a task around.
        print(
            f"\nNo deficit found: nothing cleared min_gap={args.min_gap} with at least "
            f"{args.min_support} relevant traces on each side."
        )
        print(f"Report written to {write(None, None)}")
        return 0

    print(
        f"\nTop deficit: {top.capability.name} "
        f"(gap={_fmt(top.gap)}, prevalence={top.prevalence:.2f}, "
        f"N={top.n_relevant_wins}w/{top.n_relevant_losses}l)"
    )

    if diagnosis is not None:
        w, m = diagnosis.within_model_score, diagnosis.mcnemar
        print(
            f"Within-model ({args.candidate_model}): lacking in {_fmt(w.baseline_rate)} of its "
            f"own wins (N={w.n_relevant_wins}), {_fmt(w.incident_rate)} of its own losses "
            f"(N={w.n_relevant_losses})"
        )
        print(
            f"Same-task McNemar: b={m.frontier_only} (frontier only), c={m.candidate_only} "
            f"(candidate only), {m.discordant_n} discordant, "
            f"p={'n/a' if m.p_value is None else f'{m.p_value:.4f}'}"
            + ("  [underpowered]" if m.underpowered else "")
        )
        if not diagnosis.trainable:
            # A real answer, not a failure — and the point of the second
            # contrast. Scaffolding a task here would produce a training set
            # rejection sampling can never fill.
            print(
                f"\nIdentified, not trainable: {args.candidate_model} never demonstrated "
                "this capability often enough in its own wins for rejection sampling to "
                "have anything to keep. Not generating a task."
            )
            print(f"Report written to {write(None, None)}")
            return 0

    tasks_dir = out_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Harbor task (env + verifier + oracle solution)...")
    task_dir = scaffold_task(top, tasks_dir, model=args.model)
    print(f"  task written to {task_dir}")

    validity = None
    if args.prove:
        print(f"Running validity proof (oracle{', ' + args.base_agent if args.base_agent else ''})...")
        validity = prove_validity(
            task_dir,
            jobs_dir=out_dir / "jobs",
            base_agent=args.base_agent,
            base_model=args.base_model,
        )
        print(f"  oracle passed: {validity['oracle'].passed}")
        if validity["base"]:
            print(f"  {args.base_agent} passed: {validity['base'].passed}")
        print(f"  valid: {validity['valid']}")

    print(f"\nReport written to {write(task_dir, validity)}")
    return 0


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"

def register_diagnose(sub: argparse._SubParsersAction) -> None:
    """Register the `diagnose` subcommand on `sub`."""
    p_diag = sub.add_parser(
        "diagnose", help="diagnose a capability deficit from win/loss traces and generate a task"
    )
    p_diag.add_argument("--manifest", required=True, help="JSON manifest of trace files + outcomes")
    p_diag.add_argument("--out", default="./vektori-out", help="output directory")
    p_diag.add_argument("--model", default=None, help="OpenAI model override")
    p_diag.add_argument(
        "--prove", action="store_true", help="also run harbor to produce the validity proof"
    )
    p_diag.add_argument(
        "--base-agent",
        default=None,
        help="harbor agent name to run as the 'base' attempt, e.g. codex, claude-code",
    )
    p_diag.add_argument("--base-model", default=None, help="model name for --base-agent")
    p_diag.add_argument(
        "--min-gap",
        type=_min_gap_arg,
        default=DEFAULT_MIN_GAP,
        help="minimum win/loss gap for a capability to be reported as a deficit (uncalibrated)",
    )
    p_diag.add_argument(
        "--min-support",
        type=_min_support_arg,
        default=DEFAULT_MIN_SUPPORT,
        help="minimum relevant traces on each side of the gap",
    )
    # Both or neither. Given both, the manifest is read as a `replay` manifest
    # and scored as two contrasts (cross-model, within-model) plus a same-task
    # McNemar test; given neither, the manifest is one undifferentiated win/loss
    # set exactly as before.
    p_diag.add_argument(
        "--frontier-model",
        default=None,
        help=(
            "the frontier model in a `replay` manifest. With --candidate-model, scores "
            "the cross-model contrast (frontier wins vs candidate losses) instead of "
            "mixing both models into one win/loss set"
        ),
    )
    p_diag.add_argument(
        "--candidate-model",
        default=None,
        help="the candidate model under test; required alongside --frontier-model",
    )
    p_diag.set_defaults(func=cmd_diagnose)
