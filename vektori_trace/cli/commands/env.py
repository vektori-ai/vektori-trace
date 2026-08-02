"""`selftest`, `checkenv`, `prove` — environment and validity checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ...evaluate.planted import (
    DEFAULT_SWEEP,
    DISTRACTOR_MODES,
    PLANTED_NAME,
    SweepConfig,
    estimate_calls,
    run_sweep,
    write_sweep_report,
)
from ...evaluate.validity import _find_reward, prove_validity
from ...runtime.envcheck import (
    build_committing_task,
    build_honest_task,
    build_probe_task,
    build_reward_hack_task,
    evaluate_committing,
    evaluate_honest,
    evaluate_reward_hack,
    run_probe,
)


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

        # The control: a blocked hack means nothing if honest work also scores
        # zero, which is what a broken isolated verifier would look like.
        print("Control: an agent that actually fixes the bug...")
        honest_dir = build_honest_task(out_dir / "task")
        run_probe(honest_dir, out_dir / "jobs-honest")
        honest_finding = evaluate_honest(_find_reward(out_dir / "jobs-honest"))
        print(f"  [{honest_finding.mark}] {honest_finding.name}: {honest_finding.detail}")
        # The collection control. Both probes above leave their work in the
        # worktree, so neither exercises the path a real agent takes when it
        # commits — and a diff taken against the index comes out empty there.
        print("Control: an agent that commits its fix...")
        committing_dir = build_committing_task(out_dir / "task")
        run_probe(committing_dir, out_dir / "jobs-committing")
        committing_finding = evaluate_committing(_find_reward(out_dir / "jobs-committing"))
        print(f"  [{committing_finding.mark}] {committing_finding.name}: {committing_finding.detail}")
        findings = [*findings, hack_finding, honest_finding, committing_finding]

    failed = [f for f in findings if not f.ok]
    if failed:
        print(f"\n{len(failed)} check(s) failed — emitted tasks are not sound as shipped.")
        return 1
    print(
        "\nAll checks passed: the image build, the history scrub and the network "
        "policy all take effect."
    )
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
