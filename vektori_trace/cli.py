from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .diagnose import label_trace, propose_capabilities, score_deficits
from .report import build_report, write_report
from .schema import Trace, load_manifest
from .taskgen import scaffold_task
from .validity import prove_validity


def _load_traces(manifest_path: Path) -> list[Trace]:
    entries = load_manifest(manifest_path)
    return [Trace.load(e.path, outcome=e.outcome) for e in entries]


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

    print(f"Loaded {len(wins)} win(s) and {len(losses)} loss(es).")
    print("Proposing candidate capabilities...")
    capabilities = propose_capabilities(traces, model=args.model)
    for c in capabilities:
        print(f"  - {c.id}: {c.name}")

    print("Labeling each trace against candidate capabilities...")
    trace_labels = [label_trace(t, capabilities, model=args.model) for t in traces]

    scores = score_deficits(capabilities, trace_labels)
    top = scores[0]
    print(
        f"\nTop deficit: {top.capability.name} "
        f"(gap={top.gap}, prevalence={top.prevalence:.2f})"
    )

    out_dir = Path(args.out)
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

    report = build_report(top, scores, task_dir, validity)
    md_path = write_report(report, out_dir)
    print(f"\nReport written to {md_path}")
    return 0


def cmd_prove(args: argparse.Namespace) -> int:
    task_dir = Path(args.task_dir)
    validity = prove_validity(
        task_dir,
        jobs_dir=Path(args.out) / "jobs",
        base_agent=args.base_agent,
        base_model=args.base_model,
    )
    print(f"oracle passed: {validity['oracle'].passed}")
    if validity["base"]:
        print(f"{args.base_agent} passed: {validity['base'].passed}")
    print(f"valid: {validity['valid']}")
    return 0 if validity["valid"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vektori-trace")
    sub = parser.add_subparsers(dest="command", required=True)

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
        help="harbor agent name to run as the 'base' attempt, e.g. codex, claude_code",
    )
    p_diag.add_argument("--base-model", default=None, help="model name for --base-agent")
    p_diag.set_defaults(func=cmd_diagnose)

    p_prove = sub.add_parser("prove", help="run the validity proof for an already-generated task")
    p_prove.add_argument("task_dir")
    p_prove.add_argument("--out", default="./vektori-out")
    p_prove.add_argument("--base-agent", default=None)
    p_prove.add_argument("--base-model", default=None)
    p_prove.set_defaults(func=cmd_prove)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
