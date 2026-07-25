from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .diagnose import (
    DEFAULT_MIN_GAP,
    DEFAULT_MIN_SUPPORT,
    label_trace,
    propose_capabilities,
    score_deficits,
    select_deficit,
)
from .mining.miner import HarborTraceRunner, collect_traces, mine_tasks
from .mining.spec import LLMSpec
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
    thresholds = {"min_gap": args.min_gap, "min_support": args.min_support}
    out_dir = Path(args.out)

    print("\nRanked candidates:")
    for s in scores:
        print(
            f"  {s.capability.name}: priority={_fmt(s.priority)}, gap={_fmt(s.gap)}, "
            f"prevalence={s.prevalence:.2f}, N={s.n_relevant_wins}w/{s.n_relevant_losses}l"
        )

    top = select_deficit(scores, min_gap=args.min_gap, min_support=args.min_support)
    if top is None:
        # A clean, honest exit — not an error. Nothing here separates wins from
        # losses well enough to build a task around.
        print(
            f"\nNo deficit found: nothing cleared min_gap={args.min_gap} with at least "
            f"{args.min_support} relevant traces on each side."
        )
        report = build_report(None, scores, None, None, thresholds)
        md_path = write_report(report, out_dir)
        print(f"Report written to {md_path}")
        return 0

    print(
        f"\nTop deficit: {top.capability.name} "
        f"(gap={_fmt(top.gap)}, prevalence={top.prevalence:.2f}, "
        f"N={top.n_relevant_wins}w/{top.n_relevant_losses}l)"
    )

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

    report = build_report(top, scores, task_dir, validity, thresholds)
    md_path = write_report(report, out_dir)
    print(f"\nReport written to {md_path}")
    return 0


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def cmd_mine(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    tasks_dir = out_dir / "mined_tasks"
    traces_dir = out_dir / "mined_traces"

    print(f"Mining {args.repo}'s merged PR history into sandbox-verified tasks...")
    task_dirs = mine_tasks(
        args.repo,
        tasks_dir,
        llm=LLMSpec(provider=args.llm_provider, model=args.llm_model),
        user_dockerfile=Path(args.dockerfile) if args.dockerfile else None,
    )
    print(f"  {len(task_dirs)} task(s) mined to {tasks_dir}")

    runner = HarborTraceRunner(agent=args.agent, jobs_dir=out_dir / "jobs", model=args.model)
    print(f"Running {args.agent} against each mined task to collect traces...")
    manifest_path = out_dir / "manifest.json"
    manifest = collect_traces(task_dirs, runner, traces_dir, manifest_path=manifest_path)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    wins = sum(1 for m in manifest if m["outcome"] == "win")
    skipped = len(task_dirs) - len(manifest)
    print(f"  {wins} win(s), {len(manifest) - wins} loss(es) — manifest written to {manifest_path}")
    if skipped:
        print(f"  {skipped} task(s) skipped (unjudgeable — see the skip lines above)")
    print(f"\nNext: vektori-trace diagnose --manifest {manifest_path} --out {args.out}")
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
    p_diag.add_argument(
        "--min-gap",
        type=float,
        default=DEFAULT_MIN_GAP,
        help="minimum win/loss gap for a capability to be reported as a deficit (uncalibrated)",
    )
    p_diag.add_argument(
        "--min-support",
        type=int,
        default=DEFAULT_MIN_SUPPORT,
        help="minimum relevant traces on each side of the gap",
    )
    p_diag.set_defaults(func=cmd_diagnose)

    p_prove = sub.add_parser("prove", help="run the validity proof for an already-generated task")
    p_prove.add_argument("task_dir")
    p_prove.add_argument("--out", default="./vektori-out")
    p_prove.add_argument("--base-agent", default=None)
    p_prove.add_argument("--base-model", default=None)
    p_prove.set_defaults(func=cmd_prove)

    p_mine = sub.add_parser(
        "mine",
        help=(
            "mine a repo's real PR history into sandbox-verified tasks, run an agent "
            "against each, and write win/loss traces + a manifest for `diagnose`"
        ),
    )
    p_mine.add_argument("--repo", required=True, help="'owner/name' or a full GitHub URL")
    p_mine.add_argument(
        "--dockerfile",
        default=None,
        help=(
            "path to the repo's own working Dockerfile (skips the bootstrap agent). "
            "Omit to let the agent auto-discover the build/test setup instead — needed "
            "when there's no Dockerfile yet, or a mined PR predates what the current "
            "one can build."
        ),
    )
    p_mine.add_argument(
        "--agent", default="claude_code", help="harbor agent name to run against each task"
    )
    p_mine.add_argument("--model", default=None, help="model name for --agent")
    p_mine.add_argument(
        "--llm-provider", default="openai", help="provider for the bootstrap agent's LLM calls"
    )
    p_mine.add_argument(
        "--llm-model", default="gpt-5-nano", help="model for the bootstrap agent's LLM calls"
    )
    p_mine.add_argument("--out", default="./vektori-out", help="output directory")
    p_mine.set_defaults(func=cmd_mine)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
