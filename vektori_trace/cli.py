from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .diagnose import (
    DEFAULT_MIN_GAP,
    DEFAULT_MIN_SUPPORT,
    diagnose_replay,
    label_trace,
    propose_capabilities,
    score_deficits,
    select_deficit,
)
from .envcheck import (
    build_committing_task,
    build_honest_task,
    build_probe_task,
    build_reward_hack_task,
    evaluate_committing,
    evaluate_honest,
    evaluate_reward_hack,
    run_probe,
)
from .gap import compute_gap, format_rate, write_gap_report
from .mining.inspect import audit_tasks, failure_histogram
from .mining.miner import (
    HarborTraceRunner,
    collect_paired_traces,
    collect_traces,
    discover_tasks,
    mine_tasks,
)
from .mining.spec import LLMSpec, PRRuntimeOptions
from .planted import (
    DEFAULT_SWEEP,
    DISTRACTOR_MODES,
    PLANTED_NAME,
    SweepConfig,
    estimate_calls,
    run_sweep,
    write_sweep_report,
)
from .report import build_replay_report, build_report, write_report
from .schema import Trace, load_manifest
from .taskgen import scaffold_task
from .validity import _find_reward, prove_validity


def _min_gap_arg(value: str) -> float:
    """`--min-gap` as a threshold that can actually reject something.

    `float("nan")` parses fine, and every `gap < nan` comparison in
    `select_deficit` is then False — so a NaN threshold doesn't loosen the
    filter, it *removes* it, and inverted evidence (gap = -1.0) gets reported
    as a confident diagnosis again. That is the exact failure this flag was
    added to prevent, so it's rejected at the boundary rather than downstream.
    """
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(
            f"must be a finite number, got {value!r} (a non-finite threshold rejects nothing)"
        )
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"must be >= 0, got {parsed} (a negative gap is evidence against the hypothesis)"
        )
    return parsed


def _min_support_arg(value: str) -> int:
    """`--min-support` below 1 would admit capabilities with no evidence at all."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            f"must be >= 1, got {parsed} (0 admits capabilities backed by no traces)"
        )
    return parsed


def _load_traces(manifest_path: Path) -> list[Trace]:
    entries = load_manifest(manifest_path)
    return [Trace.load(e.path, outcome=e.outcome, model=e.model, task=e.task) for e in entries]


def _check_replay_models(args: argparse.Namespace, traces: list[Trace]) -> str | None:
    """Validate `--frontier-model`/`--candidate-model`, or None if they're fine.

    Checked before the proposer runs: every failure here is knowable from the
    manifest alone, and discovering one after labelling has cost an LLM call per
    trace is pure waste.
    """
    frontier, candidate = args.frontier_model, args.candidate_model
    if (frontier is None) != (candidate is None):
        missing = "--candidate-model" if frontier else "--frontier-model"
        return (
            f"{missing} is required alongside the other — the two contrasts are "
            "defined by which model produced which trace, so one name on its own "
            "names nothing."
        )
    if frontier is None:
        return None
    if frontier == candidate:
        return (
            f"--frontier-model and --candidate-model are the same ({frontier!r}) — "
            "there is no cross-model contrast between a model and itself."
        )
    present = {t.model for t in traces}
    for flag, model in (("--frontier-model", frontier), ("--candidate-model", candidate)):
        if model not in present:
            known = ", ".join(sorted(m for m in present if m)) or "none (no 'model' field set)"
            return (
                f"{flag} {model!r} has no traces in the manifest. Models present: {known}. "
                "A manifest without models is a `mine` manifest; the two-contrast path "
                "needs one from `replay`."
            )
    return None


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


def cmd_mine(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    tasks_dir = out_dir / "mined_tasks"
    traces_dir = out_dir / "mined_traces"

    # Fail before the bootstrap, not after every PR has been silently
    # discarded. A supplied Dockerfile skips the agent that discovers the test
    # command, and F2P/P2P are derived by *running* the suite — so without one
    # validation can't derive anything and every candidate skips as
    # `no_fail_to_pass`, which reads as "this repo has no minable PRs".
    if args.dockerfile and not [c for c in (args.test_cmd or []) if c.strip()]:
        print(
            "error: --dockerfile needs --test-cmd. Supplying a Dockerfile skips the "
            "bootstrap agent, so nothing discovers how to run the suite, and F2P/P2P "
            "come from running it. Without one every PR skips as no_fail_to_pass.\n"
            "  e.g. --test-cmd 'python -m pytest -q'",
            file=sys.stderr,
        )
        return 2

    options = PRRuntimeOptions(
        limit=args.limit,
        require_linked_issue=not args.no_require_linked_issue,
        skip_validation=args.skip_validation,
    )
    print(f"Mining {args.repo}'s merged PR history into sandbox-verified tasks...")
    print(
        f"  limit={options.limit}  require_linked_issue={options.require_linked_issue}  "
        f"skip_validation={options.skip_validation}"
    )
    task_dirs, result = mine_tasks(
        args.repo,
        tasks_dir,
        llm=LLMSpec(provider=args.llm_provider, model=args.llm_model),
        user_dockerfile=Path(args.dockerfile) if args.dockerfile else None,
        test_cmds=args.test_cmd or None,
        language=args.language,
        options=options,
    )
    print(f"  {len(task_dirs)} task(s) mined to {tasks_dir}")

    # The yield alone can't say whether a small number means the repo is
    # unsuitable or a filter is wrong, and those call for opposite responses.
    print(f"\nWhere the {result.candidates} candidate PR(s) went:")
    print(f"  {'emitted':<28} {result.emitted}")
    for reason, n in sorted(result.skip_reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<28} {n}")

    audits = audit_tasks(task_dirs)
    if audits:
        bad = [a for a in audits if not a.ok]
        print(f"\nStatic audit of {len(audits)} emitted task(s):")
        hist = failure_histogram(audits)
        if not hist:
            print("  every task agrees with itself on all checks")
        for check, n in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f"  [FAIL] {check:<34} {n}/{len(audits)} task(s)")
        for a in bad[:3]:
            print(f"\n  {a.task}:")
            for name in a.failures:
                print(f"    [FAIL] {name}: {a.details[name]}")

    (out_dir / "mine-report.json").write_text(
        json.dumps(
            {
                "repo": args.repo,
                "candidates": result.candidates,
                "emitted": result.emitted,
                "skipped": result.skipped,
                "skip_reasons": result.skip_reasons,
                "audits": [
                    {"task": a.task, "ok": a.ok, "checks": a.checks, "details": a.details}
                    for a in audits
                ],
            },
            indent=2,
        )
    )
    print(f"\nMine report written to {out_dir / 'mine-report.json'}")

    # A failed audit stops the replay, not just --no-replay's exit code.
    # `git_history_scrubbed=False` means an agent can read the fix out of
    # `.git`, so every trace collected from that task is contaminated — and a
    # contaminated win is worse than no win, because nothing downstream can
    # tell it apart from a real one. The other checks mean the task is
    # unwinnable, which poisons the loss half just as effectively.
    bad_audits = [a for a in audits if not a.ok]
    if bad_audits:
        print(
            f"\nerror: {len(bad_audits)} of {len(audits)} task(s) failed the audit. "
            "Not running an agent against them — traces collected from a task that "
            "fails these checks are contaminated or unwinnable, and neither is "
            "distinguishable from the real thing once written to disk.",
            file=sys.stderr,
        )
        return 1

    if args.no_replay:
        print(
            "\n--no-replay: stopping before the agent runs. "
            f"Next: vektori-trace mine --repo {args.repo} (without --no-replay), "
            "or inspect the tasks above."
        )
        return 0

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
    arms = [
        (args.frontier_model, HarborTraceRunner(agent=args.agent, jobs_dir=jobs_dir, model=args.frontier_model)),
        (args.candidate_model, HarborTraceRunner(agent=args.agent, jobs_dir=jobs_dir, model=args.candidate_model)),
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


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, separated from running it so argument parsing —
    notably the threshold validators — is testable without dispatching."""
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
    # Same validators as `diagnose` — the sweep scores recovery against these
    # thresholds, so a NaN here silently reports 100% recovery.
    p_self.add_argument("--min-gap", type=_min_gap_arg, default=DEFAULT_MIN_GAP)
    p_self.add_argument("--min-support", type=_min_support_arg, default=DEFAULT_MIN_SUPPORT)
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
        "--agent",
        default="claude-code",
        help=(
            "harbor agent name to run against each task. Harbor's names are hyphenated "
            "(claude-code, codex, terminus-2); underscores are normalised"
        ),
    )
    p_mine.add_argument("--model", default=None, help="model name for --agent")
    p_mine.add_argument(
        "--llm-provider", default="openai", help="provider for the bootstrap agent's LLM calls"
    )
    p_mine.add_argument(
        "--llm-model", default="gpt-5-nano", help="model for the bootstrap agent's LLM calls"
    )
    p_mine.add_argument("--out", default="./vektori-out", help="output directory")
    p_mine.add_argument(
        "--limit", type=int, default=50, help="how many merged PRs to consider (default 50)"
    )
    p_mine.add_argument(
        "--test-cmd",
        action="append",
        default=[],
        help=(
            "how to run the suite, repeatable. Required with --dockerfile: skipping the "
            "bootstrap agent means nothing discovers the test command, and F2P/P2P are derived "
            "by running the suite, so without it every PR skips as no_fail_to_pass"
        ),
    )
    p_mine.add_argument(
        "--language",
        default=None,
        choices=["python", "node", "go", "rust", "java", "c_cpp"],
        help="language hint for --dockerfile runs (selects the toolchain PATH prelude)",
    )
    p_mine.add_argument(
        "--no-require-linked-issue",
        action="store_true",
        help=(
            "keep PRs with no 'Fixes #N' trailer. The linked issue is what gives the task a "
            "problem statement written before the fix existed; without one the PR body is the "
            "only source and it describes the solution"
        ),
    )
    p_mine.add_argument(
        "--skip-validation",
        action="store_true",
        help=(
            "emit tasks without running the suite twice to derive F2P/P2P. Fast, and the result "
            "is an UNVERIFIED task graded on exit code alone — never train on these"
        ),
    )
    p_mine.add_argument(
        "--no-replay",
        action="store_true",
        help="mine, audit and stop, without running an agent to collect traces",
    )
    p_mine.set_defaults(func=cmd_mine)

    p_replay = sub.add_parser(
        "replay",
        help=(
            "run a frontier and a candidate model over the same already-mined tasks, "
            "on one pinned scaffold, and report the pass-rate gap number"
        ),
    )
    p_replay.add_argument(
        "--tasks-dir", required=True, help="a previously-mined tasks dir (e.g. from `mine --no-replay`)"
    )
    p_replay.add_argument(
        "--agent",
        default="claude-code",
        help=(
            "harbor agent name — the ONE scaffold shared by both arms, since the gap is a "
            "property of model x scaffold, not just the model (Harbor's names are hyphenated: "
            "claude-code, codex, terminus-2; underscores are normalised)"
        ),
    )
    p_replay.add_argument("--frontier-model", required=True, help="the frontier model, e.g. gpt-5")
    p_replay.add_argument(
        "--candidate-model", required=True, help="the candidate model under test, e.g. a 4B-8B open model"
    )
    p_replay.add_argument("--out", default="./vektori-out", help="output directory")
    p_replay.set_defaults(func=cmd_replay)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
