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
from .envcheck import (
    build_probe_task,
    build_reward_hack_task,
    evaluate_reward_hack,
    run_probe,
)
from .mining.miner import HarborTraceRunner, collect_traces, mine_tasks
from .mining.spec import LLMSpec
from .planted import (
    DEFAULT_SWEEP,
    DISTRACTOR_MODES,
    PLANTED_NAME,
    SweepConfig,
    estimate_calls,
    run_sweep,
    write_sweep_report,
)
from .report import build_report, write_report
from .schema import Trace, load_manifest
from .taskgen import scaffold_task
from .validity import _find_reward, prove_validity


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


def cmd_selftest(args: argparse.Namespace) -> int:
    """Plant a deficit in synthetic traces and measure whether we recover it."""
    configs = DEFAULT_SWEEP
    if args.quick:
        configs = (SweepConfig(n_wins=6, n_losses=6, prevalence=1.0),)

    out_dir = Path(args.out)
    calls = 0 if args.ceiling_only else estimate_calls(configs, args.repeats)
    print(
        f"Planted-deficit self-test: {len(configs)} config(s) × {args.repeats} repeat(s) "
        f"≈ {calls} LLM calls to {args.model or 'the default model'}."
    )
    print(f"Planted capability: {PLANTED_NAME}")
    print(f"Distractor failure modes: {', '.join(DISTRACTOR_MODES)}\n")

    def report_cell(cell) -> None:
        acc = cell.mean_label_accuracy
        ceiling = cell.ceiling_rate
        line = f"  {cell.config.label:>20}  ceiling {'n/a' if ceiling is None else f'{ceiling:.0%}'}"
        if cell.results:
            line += (
                f"  recovered {cell.recovery_rate:.0%}"
                f"  proposed {cell.proposed_rate:.0%}"
                f"  label acc {'n/a' if acc is None else f'{acc:.0%}'}"
                f"  ({', '.join(f'{k}×{v}' for k, v in sorted(cell.verdicts.items()))})"
            )
        else:
            line += f"  ({', '.join(f'{k}×{v}' for k, v in sorted(cell.ceiling_verdicts.items()))})"
        print(line)

    cells = run_sweep(
        configs,
        out_dir,
        repeats=args.repeats,
        model=args.model,
        min_gap=args.min_gap,
        min_support=args.min_support,
        seed=args.seed,
        ceiling_only=args.ceiling_only,
        on_cell=report_cell,
    )

    md_path = write_sweep_report(
        cells,
        out_dir,
        model=args.model,
        min_gap=args.min_gap,
        min_support=args.min_support,
        seed=args.seed,
    )
    ceiling = sum(c.ceiling_rate or 0.0 for c in cells) / len(cells)
    print(f"\nMean ceiling (perfect labeller) across configs: {ceiling:.0%}")
    if args.ceiling_only:
        print(f"Report written to {md_path}")
        return 0

    overall = sum(c.recovery_rate for c in cells) / len(cells)
    print(f"Mean recovery rate across configs:              {overall:.0%}")
    print(f"Report written to {md_path}")
    # Non-zero only when the ranker recovered nothing anywhere *that the
    # ceiling says was recoverable* — configs the thresholds rule out by
    # construction are not the ranker's failure.
    recoverable = [c for c in cells if (c.ceiling_rate or 0.0) > 0]
    if not recoverable:
        print("No config was recoverable even in principle — check the thresholds.")
        return 1
    return 0 if any(c.recovery_rate > 0 for c in recoverable) else 1


def cmd_checkenv(args: argparse.Namespace) -> int:
    """Verify inside a real container that an emitted task's environment holds."""
    out_dir = Path(args.out)
    task_dir = build_probe_task(out_dir / "task")
    print(f"Probe task written to {task_dir}")
    print("Running it through harbor (builds an image; first run is slow)...\n")

    findings, probe, output = run_probe(task_dir, out_dir / "jobs")
    if probe is None:
        print("error: the probe produced no report — harbor output follows.", file=sys.stderr)
        print(output[-3000:], file=sys.stderr)
        return 2

    for f in findings:
        print(f"  [{f.mark}] {f.name}: {f.detail}")

    (out_dir / "envcheck.json").write_text(
        json.dumps(
            {
                "probe": probe,
                "findings": [{"name": f.name, "ok": f.ok, "detail": f.detail} for f in findings],
            },
            indent=2,
        )
    )
    print(f"\nReport written to {out_dir / 'envcheck.json'}")

    if args.reward_hack:
        print("\nReward-hack probe: an agent that fixes nothing and forges its score...")
        hack_dir = build_reward_hack_task(out_dir / "task")
        run_probe(hack_dir, out_dir / "jobs-rewardhack")
        reward = _find_reward(out_dir / "jobs-rewardhack")
        hack_finding = evaluate_reward_hack(reward)
        print(f"  [{hack_finding.mark}] {hack_finding.name}: {hack_finding.detail}")
        findings = [*findings, hack_finding]

    failed = [f for f in findings if not f.ok]
    if failed:
        print(f"\n{len(failed)} check(s) failed — emitted tasks are not sound as shipped.")
        return 1
    print("\nAll checks passed: the Dockerfile and the compose overlay both take effect.")
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

    p_self = sub.add_parser(
        "selftest",
        help=(
            "plant a known capability deficit in synthetic traces and measure how "
            "often the ranker recovers it, across trace counts and prevalences"
        ),
    )
    p_self.add_argument("--out", default="./vektori-selftest", help="output directory")
    p_self.add_argument("--model", default=None, help="OpenAI model override")
    p_self.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="runs per config; the proposer and labeller are sampled, so one run is one draw",
    )
    p_self.add_argument(
        "--quick",
        action="store_true",
        help="a single easy config (6w/6l, prevalence 1.0) instead of the full sweep",
    )
    p_self.add_argument(
        "--ceiling-only",
        action="store_true",
        help=(
            "skip the LLM entirely and report only what a perfect proposer and "
            "labeller would recover — free, offline, and an upper bound on any real run"
        ),
    )
    p_self.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP)
    p_self.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    p_self.add_argument("--seed", type=int, default=0)
    p_self.set_defaults(func=cmd_selftest)

    p_env = sub.add_parser(
        "check-env",
        help=(
            "verify inside a real container that an emitted task's Dockerfile "
            "(base commit + git scrub) and compose overlay (egress guard) both take effect"
        ),
    )
    p_env.add_argument("--out", default="./vektori-envcheck", help="output directory")
    p_env.add_argument(
        "--reward-hack",
        action="store_true",
        help=(
            "also run an agent that fixes nothing and forges its own reward, to "
            "measure whether the shared-container verifier can be gamed"
        ),
    )
    p_env.set_defaults(func=cmd_checkenv)

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
