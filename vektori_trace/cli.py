from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

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
    mine_commits,
    mine_tasks,
)
from .mining.spec import CommitRuntimeOptions, LLMSpec, PRRuntimeOptions
from .passrate import DEFAULT_ROLLOUTS, PASSRATE_MAX, PASSRATE_MIN, measure_pass_rates
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
from .select import held_out_split, select_training_tasks, write_selection_report
from .taskgen import scaffold_task
from .tokenizer_check import DEFAULT_STUDENT, DEFAULT_TEACHER
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


def _positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}")
    return parsed


def _model_info_arg(value: str) -> dict[str, Any]:
    """`--model-info` as inline JSON or `@path.json`.

    Harbor refuses to start a `hosted_vllm/<name>` run without it, and the
    literal dict is long enough that pasting it into a shell is where the typo
    goes, so the file form exists too.
    """
    raw = Path(value[1:]).read_text() if value.startswith("@") else value
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"not valid JSON: {e}") from None
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"must be a JSON object, got {type(parsed).__name__}")
    return parsed


def _add_endpoint_args(parser: argparse.ArgumentParser, prefix: str = "") -> None:
    """Flags for pointing an arm at an OpenAI-compatible endpoint we host.

    `run_trial` has taken `api_base`/`model_info` since the Modal work, but no
    command exposed them, so every measurement was silently restricted to models
    a public provider already serves — which excludes the served candidate.
    """
    flag = f"--{prefix}api-base" if prefix else "--api-base"
    info = f"--{prefix}model-info" if prefix else "--model-info"
    dest = f"{prefix.replace('-', '_')}api_base"
    info_dest = f"{prefix.replace('-', '_')}model_info"
    parser.add_argument(
        flag,
        dest=dest,
        default=None,
        help="OpenAI-compatible base URL (e.g. a Modal vLLM /v1); reaches the agent via harbor --ak",
    )
    parser.add_argument(
        info,
        dest=info_dest,
        type=_model_info_arg,
        default=None,
        help="JSON (or @file.json) of litellm model_info; required by harbor for hosted_vllm/ models",
    )


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


def cmd_mine_commits(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    tasks_dir = out_dir / "mined_tasks"

    if args.dockerfile and not [c for c in (args.test_cmd or []) if c.strip()]:
        print(
            "error: --dockerfile needs --test-cmd. Supplying a Dockerfile skips the "
            "bootstrap agent, so nothing discovers how to run the suite, and F2P/P2P "
            "come from running it. Without one every commit skips as no_fail_to_pass.\n"
            "  e.g. --test-cmd 'python -m pytest -q'",
            file=sys.stderr,
        )
        return 2

    # git log cannot walk past the clone. Silently returning fewer candidates
    # than asked for would look like "this repo has no fixes", which is the
    # exact misreading the skip histogram exists to prevent.
    if args.clone_depth <= args.limit:
        print(
            f"error: --clone-depth ({args.clone_depth}) must exceed --limit ({args.limit}); "
            "git log cannot walk past the end of a shallow clone, so the walk would stop "
            "early and report a yield that looks like a property of the repo.\n"
            f"  try --clone-depth {max(args.limit * 4, 200)}",
            file=sys.stderr,
        )
        return 2

    options = CommitRuntimeOptions(
        limit=args.limit,
        branch=args.branch,
        clone_depth=args.clone_depth,
        skip_validation=args.skip_validation,
        synthesize_with_llm=not args.no_synthesis,
        max_pass_to_pass=args.max_pass_to_pass,
    )
    print(f"Mining {args.repo}'s commit history into sandbox-verified tasks...")
    print(
        f"  limit={options.limit}  branch={options.branch}  "
        f"synthesis={options.synthesize_with_llm}  max_p2p={options.max_pass_to_pass}"
    )
    if not options.synthesize_with_llm:
        print(
            "  warning: --no-synthesis emits raw commit text as the problem statement. "
            "Those instructions can name the fix, which inflates every pass@k downstream.",
            file=sys.stderr,
        )

    task_dirs, result = mine_commits(
        args.repo,
        tasks_dir,
        llm=LLMSpec(provider=args.llm_provider, model=args.llm_model),
        user_dockerfile=Path(args.dockerfile) if args.dockerfile else None,
        test_cmds=args.test_cmd or None,
        language=args.language,
        options=options,
    )
    print(f"  {len(task_dirs)} task(s) mined to {tasks_dir}")

    print(f"\nWhere the {result.candidates} candidate commit(s) went:")
    print(f"  {'emitted':<28} {result.emitted}")
    for reason, n in sorted(result.skip_reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<28} {n}")

    audits = audit_tasks(task_dirs)
    if audits:
        print(f"\nStatic audit of {len(audits)} emitted task(s):")
        hist = failure_histogram(audits)
        if not hist:
            print("  every task agrees with itself on all checks")
        for check, n in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f"  [FAIL] {check:<34} {n}/{len(audits)} task(s)")
        for a in [a for a in audits if not a.ok][:3]:
            print(f"\n  {a.task}:")
            for name in a.failures:
                print(f"    [FAIL] {name}: {a.details[name]}")

    (out_dir / "mine-commits-report.json").write_text(
        json.dumps(
            {
                "repo": args.repo,
                "pipeline": "commit_runtime",
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
    print(f"\nMine report written to {out_dir / 'mine-commits-report.json'}")

    bad_audits = [a for a in audits if not a.ok]
    if bad_audits:
        print(
            f"\nerror: {len(bad_audits)} of {len(audits)} task(s) failed the audit.",
            file=sys.stderr,
        )
        return 1
    return 0


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


def cmd_train(args: argparse.Namespace) -> int:
    """One arm: serve → rejection-sample rollouts → LoRA SFT → adapter + report.

    Train extras are imported lazily so a base install never pays for torch.
    """
    # Lazy: torch/transformers/peft/modal must not be imported at cli.py top-level.
    from .dataset import tokenize_sft_example
    from .rollout import collect_rollouts
    from .serve import serve_model, served_to_harbor_kwargs
    from .train import TrainConfig, run_training, write_train_report

    tasks_dir = Path(args.tasks_dir)
    if not args.tasks:
        print("error: pass at least one --task id", file=sys.stderr)
        return 2
    task_dirs = [tasks_dir / t for t in args.tasks]
    missing = [p.name for p in task_dirs if not p.is_dir()]
    if missing:
        print(f"error: task dir(s) not found under {tasks_dir}: {', '.join(missing)}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_base = getattr(args, "api_base", None)
    if api_base:
        from .endpoint import endpoint_serve_cm

        serve_cm = endpoint_serve_cm(
            api_base, model_name=getattr(args, "served_model_name", None)
        )
    else:
        serve_cm = serve_model

    try:
        with serve_cm(args.model, gpu=args.modal_gpu) as served:
            hk = served_to_harbor_kwargs(served)
            rollouts = collect_rollouts(
                task_dirs,
                agent=args.agent,
                jobs_dir=out_dir / "jobs",
                rollouts=args.rollouts,
                **hk,
            )
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not rollouts:
        print(
            "error: rejection sampling kept nothing — no passing trajectories to train on",
            file=sys.stderr,
        )
        return 1

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = []
    for r in rollouts:
        ex = tokenize_sft_example(r.turns, tokenizer)
        if ex is not None:
            examples.append(ex)
    if not examples:
        print("error: no tokenizable parent-assistant turns in passing rollouts", file=sys.stderr)
        return 1

    cfg = TrainConfig(
        base_model=args.model,
        output_dir=out_dir,
        task_ids=list(args.tasks),
        max_steps=args.max_steps,
        seed=args.seed,
        # An endpoint we manage means this box owns training too — no Modal.
        use_modal=not (args.local or bool(api_base)),
        modal_gpu=args.modal_gpu,
        stage_to_volume=not api_base,
    )
    result = run_training(examples, cfg, tokenizer=tokenizer)
    md = write_train_report(result, out_dir)
    print(f"adapter: {result.adapter_dir}")
    print(f"final loss: {result.final_loss}")
    print(f"report: {md}")
    return 0


def cmd_run_arms(args: argparse.Namespace) -> int:
    """Full A0–A4 orchestrator from selection.json (+ diagnosis.json for A1)."""
    from .arms import DEFAULT_CANDIDATE_MODEL, ArmsConfig, run_arms

    selection = Path(args.selection)
    diagnosis = Path(args.diagnosis)
    if not selection.is_file():
        print(f"error: selection.json not found: {selection}", file=sys.stderr)
        return 2
    if not diagnosis.is_file():
        print(f"error: diagnosis.json not found: {diagnosis}", file=sys.stderr)
        return 2
    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.is_dir():
        print(f"error: tasks dir not found: {tasks_dir}", file=sys.stderr)
        return 2

    cfg = ArmsConfig(
        selection_path=selection,
        diagnosis_path=diagnosis,
        tasks_dir=tasks_dir,
        out_dir=Path(args.out),
        agent=args.agent,
        candidate_model=args.candidate_model or DEFAULT_CANDIDATE_MODEL,
        frontier_model=args.frontier_model,
        rollouts=args.rollouts,
        seed=args.seed,
        pilot=args.pilot,
        use_modal=not (args.local or bool(args.api_base)),
        modal_gpu=args.modal_gpu,
        max_train_steps=args.max_steps,
        skip_nonregression=args.skip_nonregression,
        api_base=args.api_base,
        served_model_name=args.served_model_name,
    )
    try:
        report = run_arms(cfg)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    paths = report.get("_paths") or {}
    print(f"arms report: {paths.get('md', Path(args.out) / 'arms.md')}")
    return 0


def _load_teacher_trajectories(source: Path) -> list[tuple[str, list[Any]]]:
    """Teacher trajectories from harbor job dirs or ATIF JSON traces.

    Two shapes, because two things produce them: `replay` leaves harbor job
    directories, and mined/example traces are JSON. Anything unreadable is
    reported rather than skipped — a silently smaller corpus changes what the run
    measured.
    """
    from .mining.atif import TrajectoryParseError, parse_job_trajectory
    from .schema import Trace

    trajectories: list[tuple[str, list[Any]]] = []
    if not source.is_dir():
        raise FileNotFoundError(f"teacher trace source not found: {source}")

    for path in sorted(source.iterdir()):
        if path.is_dir():
            try:
                trajectories.append((path.name, parse_job_trajectory(path)))
            except TrajectoryParseError as e:
                raise ValueError(f"{path}: not a readable harbor job dir ({e})") from e
        elif path.suffix == ".json":
            try:
                trace = Trace.load(path)
            except Exception as e:
                # Match the job-dir branch above: name the file that failed.
                # Trace.load raises schema/decode errors the caller's
                # FileNotFoundError/ValueError handler does not catch, so without
                # this an unreadable trace escapes as a bare traceback.
                raise ValueError(f"{path}: not a readable ATIF trace ({e})") from e
            trajectories.append((trace.task or path.stem, trace.turns))
    if not trajectories:
        raise ValueError(
            f"no trajectories under {source} — expected harbor job dirs or ATIF .json traces"
        )
    return trajectories


def _teacher_pool_for(args: argparse.Namespace) -> Any:
    """The teacher named by `--teacher-backend`, checked before any GPU time.

    Every backend fails fast here rather than on the first scoring call halfway
    into a run: whether the teacher can score supplied tokens at all is the
    precondition for the entire method (`docs/OPD.md`), and finding out late costs
    whatever the student instance has already billed.
    """
    backend = getattr(args, "teacher_backend", "vllm")
    if backend == "vllm":
        from .teacher import teacher_pool_from_endpoint

        if not args.teacher_api_base:
            raise ValueError("--teacher-api-base is required for --teacher-backend vllm")
        return teacher_pool_from_endpoint(
            args.teacher_api_base, model=args.teacher_served_name
        )
    if backend == "fireworks":
        from .teacher_fireworks import fireworks_pool_from_env

        return fireworks_pool_from_env(
            model=args.teacher_model_id, api_base=args.teacher_api_base
        )
    from .teacher_bedrock import BedrockTeacherPool

    if not args.teacher_model_id:
        raise ValueError(
            "--teacher-model-id is required for --teacher-backend bedrock "
            "(the imported model's ARN)"
        )
    pool = BedrockTeacherPool(model_id=args.teacher_model_id, region=args.teacher_region)
    # Bedrock's ability to score supplied tokens is the repo's open question
    # (docs/HOSTED_TEACHERS.md), so it gets the same one-request check the other
    # two backends get rather than being taken on trust.
    pool.score_ids([9707, 11], [1879, 0])
    return pool


def cmd_distill(args: argparse.Namespace) -> int:
    """OPD: teacher prefix → student samples → teacher scores → reverse-KL step.

    Same-vocab path: student and teacher share a vocabulary; student ids are
    sent directly to the teacher for scoring.

    Cross-tokenizer path (--cross-tokenizer): student and teacher have different
    vocabularies; byte alignment maps the two token streams (FINAL-PLAN.md).
    """
    from .distill import OPDTrainConfig, run_opd_training, write_opd_report
    from .endpoint import EndpointError
    from .reopd import iter_reopd_examples
    from .teacher import TeacherScoringError
    from .tokenizer_check import DEFAULT_STUDENT, DEFAULT_TEACHER, TokenizerMismatchError

    try:
        trajectories = _load_teacher_trajectories(Path(args.teacher_traces))
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    examples = list(
        iter_reopd_examples(trajectories, steps_per_traj=args.steps_per_trajectory)
    )
    if not examples:
        print(
            "error: teacher trajectories contain no parent assistant turns — "
            "nothing for the student to act on",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out)
    try:
        # Fails before any GPU time if the teacher cannot score supplied tokens.
        pool = _teacher_pool_for(args)
    except (EndpointError, TeacherScoringError, ValueError) as e:
        print(f"error: teacher endpoint unusable for OPD: {e}", file=sys.stderr)
        return 2

    cross_tokenizer = getattr(args, "cross_tokenizer", False)
    bridge_path = getattr(args, "bridge", None)
    thinking_mode = getattr(args, "thinking_mode", "chat")
    min_granularity = getattr(args, "min_granularity", 0.5)
    max_span_student_tokens = getattr(args, "max_span_student_tokens", 8)
    cross_top_k = getattr(args, "cross_top_k", 5)
    teacher_tok_id = getattr(args, "teacher_tokenizer_id", None)

    # For cross-tokenizer mode, load bridge + teacher tokenizer here so errors
    # surface before any GPU allocation.
    _bridge = None
    _teacher_tok = None
    if cross_tokenizer:
        if not bridge_path:
            print(
                "error: --cross-tokenizer requires --bridge PATH (run "
                "`vektori-trace build-bridge` first)",
                file=sys.stderr,
            )
            return 2
        from .vocab_bridge import CrossTokenizerBridge
        try:
            _bridge = CrossTokenizerBridge.load(bridge_path)
        except Exception as e:
            print(f"error: cannot load bridge {bridge_path}: {e}", file=sys.stderr)
            return 2

        if teacher_tok_id:
            from .vocab_bridge import load_tokenizer
            _teacher_tok = load_tokenizer(teacher_tok_id)

        # Wrap pool in CrossTokenizerTeacherPool for provenance recording.
        from .teacher_cross import CrossTokenizerTeacherPool
        pool = CrossTokenizerTeacherPool(
            pool=pool,
            teacher_tokenizer=_teacher_tok,
            thinking_mode=thinking_mode,
            bridge=_bridge,
        )

    cfg = OPDTrainConfig(
        student_model=args.student or DEFAULT_STUDENT,
        teacher_model=args.teacher or DEFAULT_TEACHER,
        output_dir=out_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        examples_per_step=args.examples_per_step,
        seed=args.seed,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        gradient_checkpointing=args.gradient_checkpointing,
        cross_tokenizer=cross_tokenizer,
        bridge_path=bridge_path,
        thinking_mode=thinking_mode,
        min_alignment_granularity=min_granularity,
        max_span_student_tokens=max_span_student_tokens,
        cross_top_k=cross_top_k,
    )
    prov = pool.provenance()
    print(
        f"OPD: {len(examples)} step-examples from {len(trajectories)} trajectories · "
        f"teacher {prov.get('teacher_model', '?')} @ {prov.get('teacher_api_base', '?')}"
    )
    try:
        result = run_opd_training(
            examples, pool, cfg,
            bridge=_bridge,
            teacher_tokenizer=_teacher_tok,
        )
    except TokenizerMismatchError as e:
        print(f"error: teacher/student tokenizers differ: {e}", file=sys.stderr)
        return 2
    except (RuntimeError, TeacherScoringError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    md = write_opd_report(result, out_dir)
    print(f"adapter: {result.adapter_dir}")
    print(f"steps: {result.steps}  final loss: {result.final_loss}")
    print(f"mean log ratio: {result.mean_log_ratio_final}")
    if result.skipped_empty_samples:
        print(f"note: {result.skipped_empty_samples} example(s) sampled no tokens")
    print(f"report: {md}")
    return 0


def cmd_probe_teacher(args: argparse.Namespace) -> int:
    """Does this hosted teacher return per-token logprobs for tokens we supply?

    The whole hosted-teacher question reduces to this one request, and neither
    vendor's documentation settles it: Fireworks documents `echo_last` and the
    integer-array prompt separately without an example combining them, and AWS
    documents `prompt_logprobs` on the chat schema while claiming completion-schema
    support it never demonstrates. So: send it, print what came back.

    Exit 0 means OPD can run against this teacher. Exit 1 means it cannot, and the
    message is the reason — which is a result worth recording, not a failure.
    """
    from .teacher import TeacherScoringError

    # Arbitrary-but-valid ids. The check is on the shape of the response, not on
    # what the tokens mean, and there is no server-side tokenizer to ask.
    prefix_ids = [9707, 11, 1879]
    tokens = [0, 1986, 374]

    pool: Any
    if args.backend == "fireworks":
        from .teacher_fireworks import (
            DEFAULT_FIREWORKS_BASE,
            DEFAULT_FIREWORKS_TEACHER,
            FireworksTeacherPool,
        )

        try:
            pool = FireworksTeacherPool(
                model=args.model or DEFAULT_FIREWORKS_TEACHER,
                api_base=args.api_base or DEFAULT_FIREWORKS_BASE,
            )
        except TeacherScoringError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        target = f"{pool.model} @ {pool.api_base}"
    else:
        if not args.model:
            print(
                "error: --model is required for bedrock (the imported model's ARN)",
                file=sys.stderr,
            )
            return 2
        from .teacher_bedrock import BedrockTeacherPool

        try:
            pool = BedrockTeacherPool(model_id=args.model, region=args.region)
        except TeacherScoringError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        target = f"{pool.model_id} @ bedrock:{pool.region}"

    print(f"probing {args.backend}: {target}")
    result: dict[str, Any] = {
        "backend": args.backend,
        "target": target,
        "prefix_ids": prefix_ids,
        "tokens": tokens,
    }
    try:
        scored = pool.score_ids(prefix_ids, tokens)
    except TeacherScoringError as e:
        result.update({"score_ids": "failed", "error": str(e)})
        _write_probe(args, result)
        print(f"score_ids: FAILED — {e}", file=sys.stderr)
        print(
            "OPD cannot run against this teacher. This is the documented outcome "
            "to record, not something to work around.",
            file=sys.stderr,
        )
        return 1

    if len(scored) != len(tokens):
        result.update({"score_ids": "failed", "error": f"{len(scored)} logprobs for {len(tokens)} tokens"})
        _write_probe(args, result)
        print(f"score_ids: FAILED — {result['error']}", file=sys.stderr)
        return 1

    result.update({"score_ids": "ok", "logprobs": scored})
    print(f"score_ids: OK — {len(scored)} logprobs, {[round(x, 4) for x in scored]}")

    if args.top_k > 0:
        try:
            rows = pool.score_ids_topk(prefix_ids, tokens, args.top_k)
            widths = [len(r) for r in rows]
            result.update({"score_ids_topk": "ok", "row_widths": widths})
            print(f"score_ids_topk(K={args.top_k}): OK — row widths {widths}")
        except (TeacherScoringError, ValueError) as e:
            # Not fatal: the sampled-token objective is the declared one and it
            # works. Top-K is the lower-variance variant, and losing it costs
            # variance, not correctness.
            result.update({"score_ids_topk": "failed", "topk_error": str(e)})
            print(f"score_ids_topk(K={args.top_k}): FAILED — {e}")
            print("note: top_k=0 (`reverse_kl_surrogate`) is unaffected.")

    result["provenance"] = pool.provenance()
    _write_probe(args, result)
    print("this teacher can run OPD.")

    if getattr(args, "echo", False):
        # P0 echo probe. FireworksTeacherPool does not expose probe_echo_support;
        # wrap it so the same check CrossTokenizerTeacherPool uses is available.
        echo_pool = pool
        if not hasattr(echo_pool, "probe_echo_support"):
            from .teacher_cross import CrossTokenizerTeacherPool

            echo_pool = CrossTokenizerTeacherPool(
                pool=pool,
                teacher_tokenizer=None,
                thinking_mode="chat",
            )
        echo_result = echo_pool.probe_echo_support()
        result["echo"] = echo_result
        print(f"echo support: {'OK' if echo_result.get('ok') else 'FAILED'}")
        if not echo_result.get("ok"):
            print(f"  error: {echo_result.get('error')}", file=sys.stderr)
            _write_probe(args, result)
            return 1
    return 0


def _write_probe(args: argparse.Namespace, result: dict[str, Any]) -> None:
    if not args.out:
        return
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"probe result: {path}")


def cmd_check_tokenizers(args: argparse.Namespace) -> int:
    from .tokenizer_check import (
        DEFAULT_STUDENT,
        DEFAULT_TEACHER,
        TokenizerMismatchError,
        check_tokenizers,
    )

    teacher = args.teacher or DEFAULT_TEACHER
    student = args.student or DEFAULT_STUDENT
    try:
        t_fp, s_fp = check_tokenizers(teacher, student)
    except TokenizerMismatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "teacher": {
                    "name": t_fp.name,
                    "vocab_size": t_fp.vocab_size,
                    "merges_sha256": t_fp.merges_sha256,
                    "vocab_sha256": t_fp.vocab_sha256,
                },
                "student": {
                    "name": s_fp.name,
                    "vocab_size": s_fp.vocab_size,
                    "merges_sha256": s_fp.merges_sha256,
                    "vocab_sha256": s_fp.vocab_sha256,
                },
            },
            indent=2,
        )
    )
    return 0


def cmd_build_bridge(args: argparse.Namespace) -> int:
    """Build a CrossTokenizerBridge JSON from a teacher/student tokenizer pair.

    The bridge maps every teacher token id to the byte-identical student token id
    (when one exists), and stores the byte tables for both tokenizers so that
    run_opd_training can align sampled student tokens with teacher re-tokenisation
    without loading either tokenizer at training time.
    """
    from .vocab_bridge import CrossTokenizerError, check_cross_tokenizer

    try:
        bridge = check_cross_tokenizer(
            args.teacher_tokenizer,
            args.student_tokenizer,
            thinking_mode=args.thinking_mode,
        )
    except CrossTokenizerError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bridge.save(out_path)
    print(f"bridge: {out_path}")
    print(f"exact-map size: {len(bridge.exact_map)} byte-identical token pairs")
    print(
        f"teacher vocab: {bridge.teacher_table.vocab_size}  "
        f"student vocab: {bridge.student_table.vocab_size}"
    )
    print(
        f"coverage: {len(bridge.exact_map) / bridge.teacher_table.vocab_size:.1%} "
        "of teacher tokens map exactly"
    )
    return 0


def cmd_align_report(args: argparse.Namespace) -> int:
    """Offline granularity report: encode text samples with both tokenizers and
    align by bytes, reporting granularity (spans / student tokens) per sample.

    Uses the bridge byte tables so no network is required; tokenizers are loaded
    locally to encode the input text.
    """
    import sys

    from .align import AlignmentError, align_by_bytes
    from .vocab_bridge import CrossTokenizerBridge

    bridge = CrossTokenizerBridge.load(args.bridge)

    # §10.7 — refuse a drifted encoder even for offline reporting.
    from .encoding_dsv4 import ENCODING_DSV4_SHA256, verify_encoding_dsv4_pin

    try:
        verify_encoding_dsv4_pin()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if bridge.encoding_dsv4_hash != ENCODING_DSV4_SHA256:
        print(
            f"error: bridge encoding_dsv4 hash mismatch: "
            f"bridge={bridge.encoding_dsv4_hash!r} current={ENCODING_DSV4_SHA256!r}",
            file=sys.stderr,
        )
        return 2

    # Load both tokenizers to encode the samples. `load_tokenizer` rather than
    # AutoTokenizer: align-report is an offline stage and must not fail because
    # `transformers` cannot parse a teacher's *model* config.
    from .vocab_bridge import encode_ids, load_tokenizer

    teacher_name = bridge.teacher_fingerprint.name
    student_name = bridge.student_fingerprint.name
    teacher_tok = load_tokenizer(teacher_name)
    student_tok = load_tokenizer(student_name)

    raw = Path(args.text).read_text() if args.text else sys.stdin.read()
    samples = [line for line in raw.splitlines() if line.strip()]
    if not samples:
        print("error: no samples to align (empty input)", file=sys.stderr)
        return 1

    rows = []
    for sample in samples:
        s_ids = encode_ids(student_tok, sample)
        t_ids = encode_ids(teacher_tok, sample)
        s_bytes = [bridge.student_table.table.get(i, b"") for i in s_ids]
        t_bytes = [bridge.teacher_table.table.get(i, b"") for i in t_ids]
        s_bytes = [b for b in s_bytes if b]
        t_bytes = [b for b in t_bytes if b]
        if not s_bytes or not t_bytes:
            rows.append({"sample": sample[:40], "error": "empty after EOS stripping"})
            continue
        try:
            al = align_by_bytes(
                s_bytes,
                t_bytes,
                max_span_student_tokens=getattr(args, "max_span_student_tokens", 8),
            )
            # §4 — log granularity with a coarse content-type tag (numeric-heavy
            # vs other); do not normalize numbers out of the data.
            from .cross_kl import _content_type_of_bytes

            joined = b"".join(s_bytes)
            content_type = _content_type_of_bytes(joined)
            rows.append({
                "sample": sample[:40],
                "granularity": round(al.granularity, 4),
                "content_type": content_type,
                "n_student": al.n_student_tokens,
                "n_teacher": al.n_teacher_tokens,
                "n_spans": len(al.spans),
            })
        except AlignmentError as e:
            rows.append({"sample": sample[:40], "error": str(e)})

    print(json.dumps(rows, indent=2))
    gran = [r["granularity"] for r in rows if "granularity" in r]
    if gran:
        print(
            f"\nmean granularity: {sum(gran) / len(gran):.4f}  "
            f"(n={len(gran)}, min={min(gran):.4f})"
        )
        by_type: dict[str, list[float]] = {}
        for r in rows:
            if "granularity" not in r:
                continue
            by_type.setdefault(r.get("content_type", "other"), []).append(r["granularity"])
        for ctype, vals in sorted(by_type.items()):
            print(
                f"  {ctype}: mean={sum(vals)/len(vals):.4f} n={len(vals)} "
                f"min={min(vals):.4f}"
            )
    return 0


def cmd_passk(args: argparse.Namespace) -> int:
    from .passk import two_stage_sweep

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
        from .routing import task_capability_map

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
    )
    path = out / "passk.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"passk report: {path}")
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
    from .gym_import import import_gym

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


def cmd_route(args: argparse.Namespace) -> int:
    from .routing import (
        ROUTING_RULES,
        CurveSummary,
        decision_to_dict,
        per_cell_counts,
        route_cell,
    )

    student = json.loads(Path(args.student_passk).read_text())
    teacher = json.loads(Path(args.teacher_passk).read_text())

    def _curve(block: dict, task: str) -> CurveSummary:
        # Prefer stage2 if present for this task, else stage1.
        row = (block.get("stage2") or {}).get(task) or (block.get("stage1") or {}).get(task)
        if not row:
            return CurveSummary(pass1=None, pass32=None)
        curves = row.get("curves") or {}
        # n32/c32 are the *actual* stage-2 stratum, which is `--stage2-n` and not
        # necessarily 32. routing.py compares them as a rate against 1/32, so the
        # pre-registered rule holds whatever sample size the sweep took.
        return CurveSummary(
            pass1=curves.get("1"),
            pass32=curves.get("32"),
            n32=int(row["n"]) if row.get("stratum") == "stage2" else 0,
            c32=int(row["c"]) if row.get("stratum") == "stage2" else 0,
            luck_quarantine=bool(row.get("luck_quarantine")),
        )

    tasks = sorted(
        set(student.get("stage1") or {})
        | set(student.get("stage2") or {})
        | set(teacher.get("stage1") or {})
        | set(teacher.get("stage2") or {})
    )

    # Cells are (task × capability). The capabilities come from the diagnosis
    # report joined to the replay manifest — one blanket label across every task
    # would make the per-capability counts meaningless.
    if args.diagnosis:
        from .routing import task_capability_map

        if not args.manifest:
            print(
                "error: --diagnosis needs --manifest to map lacking-loss run ids "
                "back to tasks",
                file=sys.stderr,
            )
            return 2
        diagnosis = json.loads(Path(args.diagnosis).read_text())
        run_to_task = {
            t.run_id: t.task for t in _load_traces(Path(args.manifest)) if t.task
        }
        caps_by_task = task_capability_map(
            diagnosis, run_to_task, only_chosen=args.chosen_deficit_only
        )
        unlabelled = [t for t in tasks if not caps_by_task.get(t)]
        if unlabelled:
            print(
                f"  {len(unlabelled)} task(s) carry no LACKING capability in "
                f"{args.diagnosis}; excluded from routing",
                file=sys.stderr,
            )
    else:
        caps_by_task = {t: [args.capability] for t in tasks}

    decisions = [
        route_cell(t, cap, _curve(student, t), _curve(teacher, t))
        for t in tasks
        for cap in caps_by_task.get(t, [])
    ]
    if not decisions:
        print("error: no (task × capability) cells to route", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "thresholds": decisions[0].thresholds if decisions else {},
        "rules": ROUTING_RULES,
        "counts": per_cell_counts(decisions),
        "decisions": [decision_to_dict(d) for d in decisions],
    }
    path = out / "routing.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"routing report: {path}")
    print(json.dumps(report["counts"], indent=2))
    return 0


def _reload_routing_decisions(raw: dict) -> list:
    """Rebuild `RoutingDecision`s from a routing.json, provenance intact."""
    from .routing import CurveSummary, RoutingDecision

    decisions = []
    for d in raw.get("decisions") or []:
        decisions.append(
            RoutingDecision(
                task=d["task"],
                capability=d["capability"],
                route=d["route"],
                student=CurveSummary(**d["student"]),
                teacher=CurveSummary(**d["teacher"]),
                thresholds=d.get("thresholds") or {},
                quarantine_cause=d.get("quarantine_cause"),
                evidence=d.get("evidence") or "",
                held_for_luck=bool(d.get("held_for_luck")),
                # Carry the rule through: without it every reloaded cell reads as
                # pre-registered, and the mid-band extension becomes invisible to
                # exactly the filtering it was tagged for.
                rule=d.get("rule") or "",
            )
        )
    return decisions


def _sandbox_for_task(task_dir: Path, platform: str = "linux/amd64"):
    """Start a fresh container for one task's environment image.

    Step A replays into a *fresh* container per prefix — reusing one would let an
    earlier probe's writes leak into the next, which is the desync the assertion
    is supposed to detect.
    """
    import tempfile

    from .mining.bootstrap.docker import DockerSandbox

    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        raise FileNotFoundError(f"no environment/Dockerfile in {task_dir}")
    image = ""
    for line in dockerfile.read_text().splitlines():
        if line.strip().upper().startswith("FROM "):
            image = line.split(None, 1)[1].strip()
            break
    if not image:
        raise ValueError(f"no FROM line in {dockerfile}")
    marker = Path(tempfile.mkdtemp(prefix="r2e-resume-"))
    (marker / ".keep").write_text("")
    return DockerSandbox.start(base_image=image, repo_dir=marker, platform=platform)


def cmd_resume_check(args: argparse.Namespace) -> int:
    """Step A — replay trajectory prefixes into fresh containers, report desync.

    PLAN.md calls this the spike that gates the design: "High desync makes ReOPD
    mandatory rather than merely preferable — that must be known in week 1."
    """
    from .resume import assistant_tool_steps, measure_desync_rate, replay_prefix

    traces = _load_traces(Path(args.manifest))
    if args.model:
        traces = [t for t in traces if t.model == args.model]
    traces = [t for t in traces if t.task]
    if not traces:
        print("error: no traces with a task id in the manifest", file=sys.stderr)
        return 2
    if args.limit:
        traces = traces[: args.limit]

    tasks_dir = Path(args.tasks_dir)
    results = []
    per_trace: list[dict[str, Any]] = []
    for trace in traces:
        task_dir = tasks_dir / str(trace.task)
        if not task_dir.is_dir():
            print(f"  skip {trace.run_id}: no task dir {task_dir}", file=sys.stderr)
            continue
        n_steps = len(assistant_tool_steps(trace.turns))
        if n_steps == 0:
            continue
        # Probe prefixes across the trajectory rather than only the full one:
        # desync is a function of how far in you replay.
        fractions = [f for f in (0.25, 0.5, 1.0) if int(n_steps * f) > 0]
        for frac in fractions:
            T = int(n_steps * frac) - 1
            sandbox = None
            try:
                sandbox = _sandbox_for_task(task_dir, platform=args.platform)
                res = replay_prefix(trace.turns, T, sandbox, hard_fail=False)
            except Exception as e:
                print(f"  {trace.run_id} T={T}: infra failure: {e}", file=sys.stderr)
                continue
            finally:
                if sandbox is not None:
                    with contextlib.suppress(Exception):
                        sandbox.cleanup()
            results.append(res)
            per_trace.append(
                {
                    "run_id": trace.run_id,
                    "task": trace.task,
                    "T": T,
                    "steps": n_steps,
                    "verified": res.verified,
                    "consistent": res.consistent,
                    "checks_run": res.checks_run,
                    "unsupported_skipped": res.unsupported_skipped,
                    "readonly_skipped": res.readonly_skipped,
                    "desync_reason": res.desync_reason,
                }
            )

    if not results:
        print("error: no prefix was replayed", file=sys.stderr)
        return 1
    stats = measure_desync_rate(results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "resume-check.json"
    path.write_text(json.dumps({"stats": stats, "replays": per_trace}, indent=2) + "\n")
    print(f"resume check: {path}")
    print(json.dumps(stats, indent=2))
    if stats["verified"] == 0:
        print(
            "warning: nothing was verifiable — the desync rate is undefined, not 0",
            file=sys.stderr,
        )
    return 0


def cmd_bisect(args: argparse.Namespace) -> int:
    """Step D — verifier-guided bisection to the forking step."""
    import shlex
    import subprocess

    from .intervene import bisect_forking_step, make_resume_fn

    # The teacher must continue *from the replayed prefix*. `validity.run_trial`
    # starts its own fresh container and cannot be handed one, so using it here
    # would run the teacher from scratch and ignore T entirely — every probe
    # would return the same answer and the located "forking step" would be an
    # artifact of the search, not of the trajectory. Until a scaffold that
    # accepts a seeded container exists, the continuation backend is supplied by
    # the caller: a command receiving the task dir and a JSON prefix, exiting 0
    # when the mined verifier passes.
    if not args.continuation_cmd:
        print(
            "error: --continuation-cmd is required. Bisection needs a teacher "
            "continuation that starts from the replayed prefix at step T; there "
            "is no harbor entrypoint for that yet, so the backend is external. "
            "The command receives {task_dir} and {prefix_json} and must exit 0 "
            "iff the verifier passes.",
            file=sys.stderr,
        )
        return 2

    traces = _load_traces(Path(args.manifest))
    traces = [t for t in traces if t.task and t.outcome == "loss"]
    if args.model:
        traces = [t for t in traces if t.model == args.model]
    if not traces:
        print("error: no failed traces with task ids in the manifest", file=sys.stderr)
        return 2
    if args.limit:
        traces = traces[: args.limit]

    tasks_dir = Path(args.tasks_dir)
    jobs = Path(args.out) / "bisect_jobs"
    rows: list[dict[str, Any]] = []
    for trace in traces:
        task_dir = tasks_dir / str(trace.task)
        if not task_dir.is_dir():
            continue

        def continue_with_teacher(turns, T, _task_dir=task_dir, _trace=trace):
            from .resume import assistant_tool_steps

            jobs.mkdir(parents=True, exist_ok=True)
            prefix_path = jobs / f"{_trace.run_id}-T{T}.json"
            prefix_path.write_text(
                json.dumps(
                    {
                        "task": str(_trace.task),
                        "run_id": _trace.run_id,
                        "T": T,
                        "n_action_steps": len(assistant_tool_steps(turns)),
                        "teacher_model": args.teacher_model,
                        "prefix_turns": [
                            {
                                "index": t.index,
                                "role": t.role,
                                "content": t.content,
                                "tool_calls": [
                                    {"id": tc.id, "name": tc.name, "args": tc.args}
                                    for tc in t.tool_calls
                                ],
                            }
                            for t in turns
                        ][: T + 1 if T >= 0 else 0],
                    },
                    indent=2,
                )
                + "\n"
            )
            cmd = args.continuation_cmd.format(
                task_dir=shlex.quote(str(_task_dir)),
                prefix_json=shlex.quote(str(prefix_path)),
            )
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return proc.returncode == 0

        resume_fn = None
        sandbox = None
        if args.replay_prefix:
            try:
                sandbox = _sandbox_for_task(task_dir, platform=args.platform)
                resume_fn = make_resume_fn(sandbox, hard_fail=True)
            except Exception as e:
                print(f"  {trace.run_id}: sandbox unavailable: {e}", file=sys.stderr)
        try:
            result = bisect_forking_step(
                trace.turns,
                resume=resume_fn,
                continue_with_teacher=continue_with_teacher,
                samples_per_probe=args.samples_per_probe,
                verify_probes=args.verify_probes,
            )
        finally:
            if sandbox is not None:
                with contextlib.suppress(Exception):
                    sandbox.cleanup()
        rows.append(
            {
                "run_id": trace.run_id,
                "task": trace.task,
                "steps": result.steps,
                "forking_step": result.forking_step,
                "largest_recoverable_T": result.largest_recoverable_T,
                "monotone": result.monotone,
                "non_monotone_fraction": result.non_monotone_fraction,
                "sample_disagreements": result.sample_disagreements,
                "teacher_continuations": result.teacher_continuations,
                "budget_ok": result.budget_ok,
                "continuation_budget_ok": result.continuation_budget_ok,
                "dropped": result.dropped,
                "drop_reason": result.drop_reason,
                "resume_unverified": result.resume_unverified,
            }
        )

    if not rows:
        print("error: no trajectory was bisected", file=sys.stderr)
        return 1
    located = [r for r in rows if r["forking_step"] is not None and not r["dropped"]]
    summary = {
        "n": len(rows),
        "located": len(located),
        "dropped": sum(1 for r in rows if r["dropped"]),
        "non_monotone": sum(1 for r in rows if not r["monotone"]),
        "resume_unverified": sum(1 for r in rows if r["resume_unverified"]),
        "total_teacher_continuations": sum(r["teacher_continuations"] for r in rows),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bisection.json"
    path.write_text(json.dumps({"summary": summary, "trajectories": rows}, indent=2) + "\n")
    print(f"bisection report: {path}")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_ground(args: argparse.Namespace) -> int:
    """Step E — compare diagnose labels against execution-located forking steps."""
    from .grounding import GroundingPair, ground_diagnosis, report_to_dict

    bisection = json.loads(Path(args.bisection).read_text())
    diagnosis = json.loads(Path(args.diagnosis).read_text())
    judgments: dict[str, bool] = {}
    if args.judgments:
        judgments = {
            k: bool(v) for k, v in json.loads(Path(args.judgments).read_text()).items()
        }

    gap_by_cap = {}
    for score in diagnosis.get("all_deficits_ranked") or []:
        cap = (score.get("capability") or {}).get("id")
        if cap:
            gap_by_cap[cap] = score.get("gap")
    run_to_cap: dict[str, str] = {}
    for score in diagnosis.get("all_deficits_ranked") or []:
        cap = (score.get("capability") or {}).get("id")
        for rid in score.get("lacking_loss_run_ids") or []:
            run_to_cap.setdefault(rid, cap)

    pairs = []
    for row in bisection.get("trajectories") or []:
        if row.get("dropped") or row.get("forking_step") is None:
            continue
        rid = row["run_id"]
        cap = run_to_cap.get(rid)
        if cap is None:
            continue
        pairs.append(
            GroundingPair(
                task=str(row.get("task")),
                capability=cap,
                forking_step=row["forking_step"],
                label_agrees=judgments.get(rid),
                diagnose_gap=gap_by_cap.get(cap),
            )
        )
    if not pairs:
        print(
            "error: no (forking step, capability) pair — is this the diagnosis "
            "and bisection from the same replay run?",
            file=sys.stderr,
        )
        return 1

    report = ground_diagnosis(pairs, current_min_gap=args.min_gap)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "grounding.json"
    path.write_text(json.dumps(report_to_dict(report), indent=2) + "\n")
    print(f"grounding report: {path}")
    print(
        f"pairs={report.n} agreement={report.agreement_rate} "
        f"blur={report.blur:.3f} suggested_min_gap={report.suggested_min_gap}"
    )
    if report.agreement_rate is None:
        print(
            "note: no human judgments supplied (--judgments), so agreement is "
            "unmeasured. PLAN.md acceptance criterion 4 needs 10 hand-inspected "
            "forking steps; the ids to inspect are in the report.",
            file=sys.stderr,
        )
    elif report.underpowered:
        print("warning: fewer than 10 judged pairs — underpowered", file=sys.stderr)
    return 0


def cmd_plan_b_arms(args: argparse.Namespace) -> int:
    from .arms import plan_b_arms

    decisions = _reload_routing_decisions(json.loads(Path(args.routing).read_text()))
    holdout = None
    if args.holdout:
        holdout = [
            line.strip()
            for line in Path(args.holdout).read_text().splitlines()
            if line.strip()
        ]
    if args.resolvable_effect_size is None:
        print(
            "warning: no --resolvable-effect-size recorded. PLAN.md requires the "
            "smallest resolvable effect to be stated before training; without it "
            "B1-vs-B2 cannot be interpreted afterwards.",
            file=sys.stderr,
        )
    plans = plan_b_arms(
        decisions,
        holdout=holdout,
        resolvable_effect_size=args.resolvable_effect_size,
        pilot=args.pilot,
        seed=args.seed,
        exclude_not_preregistered=args.preregistered_only,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        arm: {
            "arm": p.arm,
            "description": p.description,
            # Keyed by "<task>::<capability>" — the routing unit is the cell, not
            # the task, so the mix B2 has to hold identical is counted per cell.
            "assignments": p.assignments,
            "task_ids": p.task_ids,
            "cells": [list(c) for c in p.cells],
            "n_cells": len(p.cells),
            "method_mix": p.method_mix,
            "resolvable_effect_size": p.resolvable_effect_size,
        }
        for arm, p in plans.items()
    }
    path = out / "b_arms_plan.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"B-arm plan: {path}")
    return 0


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

    p_select = sub.add_parser(
        "select",
        help=(
            "measure candidate pass rate on the diagnosed deficit's lacking-loss tasks "
            "and select the ones in the trainable band (V0_PLAN.md Step 6)"
        ),
    )
    p_select.add_argument("--manifest", required=True, help="the replay manifest `diagnose` was run against")
    p_select.add_argument("--diagnosis", required=True, help="path to a diagnosis.json produced with --frontier-model/--candidate-model")
    p_select.add_argument("--tasks-dir", required=True, help="mined tasks directory (each task has a task.toml)")
    p_select.add_argument("--agent", required=True, help="the scaffold pinned across replay — reused for pass-rate rollouts")
    p_select.add_argument("--out", default="./vektori-out", help="output directory")
    p_select.add_argument(
        "--rollouts", type=_positive_int_arg, default=DEFAULT_ROLLOUTS,
        help="rollouts per lacking-loss task to measure candidate pass rate (plan: 8-16)",
    )
    p_select.add_argument("--passrate-min", type=float, default=PASSRATE_MIN)
    p_select.add_argument("--passrate-max", type=float, default=PASSRATE_MAX)
    p_select.add_argument("--holdout-frac", type=float, default=0.2, help="fraction of selected tasks carved out as held-out before training")
    p_select.add_argument("--seed", type=int, default=0, help="held-out split seed — written to the report, re-derivable")
    p_select.add_argument(
        "--exclude", default=None,
        help="file of task ids (one per line) to drop before splitting, e.g. SWE-bench Verified tasks",
    )
    p_select.set_defaults(func=cmd_select)

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

    p_mine_commits = sub.add_parser(
        "mine-commits",
        help=(
            "mine a repo's COMMIT history into sandbox-verified tasks. Sibling of `mine`: "
            "reaches fixes that never became a PR with a linked issue (54% of candidates in "
            "the prefect pilot), at the cost of an LLM-synthesized problem statement"
        ),
    )
    p_mine_commits.add_argument("--repo", required=True, help="'owner/name' or a full GitHub URL")
    p_mine_commits.add_argument(
        "--dockerfile", default=None, help="the repo's own Dockerfile (skips the bootstrap agent)"
    )
    p_mine_commits.add_argument("--out", default="./vektori-out", help="output directory")
    p_mine_commits.add_argument(
        "--limit", type=int, default=50, help="how many commits to walk (default 50)"
    )
    p_mine_commits.add_argument(
        "--branch", default="HEAD", help="branch to walk (default HEAD)"
    )
    p_mine_commits.add_argument(
        "--clone-depth",
        type=int,
        default=200,
        help=(
            "clone depth for the log walk (default 200). Must exceed --limit or git log "
            "runs out of history before the limit is reached"
        ),
    )
    p_mine_commits.add_argument(
        "--test-cmd",
        action="append",
        default=[],
        help="how to run the suite, repeatable. Required with --dockerfile (see `mine`)",
    )
    p_mine_commits.add_argument(
        "--language",
        default=None,
        choices=["python", "node", "go", "rust", "java", "c_cpp"],
        help="language hint for --dockerfile runs",
    )
    p_mine_commits.add_argument(
        "--llm-provider", default="openai", help="provider for bootstrap + synthesis LLM calls"
    )
    p_mine_commits.add_argument(
        "--llm-model", default="gpt-5-nano", help="model for bootstrap + synthesis LLM calls"
    )
    p_mine_commits.add_argument(
        "--no-synthesis",
        action="store_true",
        help=(
            "use raw commit text as the problem statement instead of an LLM rewrite. "
            "NOT RECOMMENDED: commit messages are written after the fix and routinely name "
            "the changed function, so an agent can score 1.0 by reading the prompt"
        ),
    )
    p_mine_commits.add_argument(
        "--max-pass-to-pass",
        type=int,
        default=50,
        help=(
            "cap the P2P regression set (default 50, 0 disables). The graded reward is "
            "f2p_rate*p2p_rate, so a whole-suite P2P scales correct solves down on any flake"
        ),
    )
    p_mine_commits.add_argument(
        "--skip-validation",
        action="store_true",
        help="emit without deriving F2P/P2P. UNVERIFIED tasks — never train on these",
    )
    p_mine_commits.set_defaults(func=cmd_mine_commits)

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
    _add_endpoint_args(p_replay, prefix="candidate-")
    p_replay.set_defaults(func=cmd_replay)

    p_train = sub.add_parser(
        "train",
        help=(
            "serve the candidate, rejection-sample passing rollouts on a task set, "
            "and LoRA-SFT an adapter (V0_PLAN.md Step 6)"
        ),
    )
    p_train.add_argument("--tasks-dir", required=True, help="mined tasks directory")
    p_train.add_argument(
        "--task",
        dest="tasks",
        action="append",
        default=[],
        help="task id to train on (repeatable)",
    )
    p_train.add_argument("--agent", required=True, help="harbor scaffold, pinned across the run")
    p_train.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="base/candidate model to serve + train (placeholder until Step 4 gap exists)",
    )
    p_train.add_argument("--out", default="./vektori-out", help="output directory")
    p_train.add_argument(
        "--rollouts",
        type=_positive_int_arg,
        default=DEFAULT_ROLLOUTS,
        help="rejection-sampling rollouts per task",
    )
    p_train.add_argument("--max-steps", type=_positive_int_arg, default=50)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--modal-gpu", default="A10G")
    p_train.add_argument(
        "--local",
        action="store_true",
        help="run LoRA on this machine instead of Modal (for tiny CPU smoke tests)",
    )
    p_train.add_argument(
        "--api-base",
        default=None,
        help=(
            "attach to a vLLM server you already run (EC2/local) instead of "
            "spawning Modal; implies --local for training"
        ),
    )
    p_train.add_argument(
        "--served-model-name",
        default=None,
        help="name the endpoint serves under (default: discovered from /v1/models)",
    )
    p_train.set_defaults(func=cmd_train)

    p_arms = sub.add_parser(
        "run-arms",
        help=(
            "run A0–A4 from selection.json: prompt baseline, random-task control, "
            "deficit-selected LoRA, frontier ceiling (V0_PLAN.md Step 6)"
        ),
    )
    p_arms.add_argument("--selection", required=True, help="path to selection.json from `select`")
    p_arms.add_argument(
        "--diagnosis",
        required=True,
        help="path to diagnosis.json (A1 templates its prompt from stored evidence)",
    )
    p_arms.add_argument("--tasks-dir", required=True, help="mined tasks directory")
    p_arms.add_argument("--agent", required=True, help="harbor scaffold pinned across all arms")
    p_arms.add_argument(
        "--candidate-model",
        default="Qwen/Qwen3-8B",
        help="placeholder default — swap once Step 4 produces a real gap number",
    )
    p_arms.add_argument(
        "--frontier-model",
        default=None,
        help="defaults to selection.json's frontier_model",
    )
    p_arms.add_argument("--out", default="./vektori-out", help="output directory")
    p_arms.add_argument("--rollouts", type=_positive_int_arg, default=DEFAULT_ROLLOUTS)
    p_arms.add_argument("--max-steps", type=_positive_int_arg, default=50)
    p_arms.add_argument("--seed", type=int, default=0)
    p_arms.add_argument("--modal-gpu", default="A10G")
    p_arms.add_argument(
        "--pilot",
        action="store_true",
        help="cap each arm at ~10 tasks before any full run (V0_PLAN.md)",
    )
    p_arms.add_argument(
        "--local",
        action="store_true",
        help="run LoRA locally instead of Modal (orchestration tests / tiny models)",
    )
    p_arms.add_argument(
        "--api-base",
        default=None,
        help=(
            "run every arm against a vLLM server you already run (EC2/local) "
            "instead of spawning Modal containers; implies --local for training. "
            "The server needs --enable-lora and VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 "
            "so A2/A3 adapters can be loaded without a restart."
        ),
    )
    p_arms.add_argument(
        "--served-model-name",
        default=None,
        help="name the endpoint serves under (default: discovered from /v1/models)",
    )
    p_arms.add_argument(
        "--skip-nonregression",
        action="store_true",
        help="skip the IFEval non-regression pass (still records the pre-declared tolerance)",
    )
    p_arms.set_defaults(func=cmd_run_arms)

    # --- FINAL-PLAN.md cross-tokenizer OPD path ---
    p_distill = sub.add_parser(
        "distill",
        help=(
            "OPD: student samples at a teacher prefix, the teacher scores those "
            "tokens, reverse-KL step. Same-vocab path: teacher/student share a "
            "tokenizer. Cross-tokenizer path (--cross-tokenizer): byte-alignment "
            "of Qwen3 + DeepSeek-V4-Flash (FINAL-PLAN.md)."
        ),
    )
    p_distill.add_argument(
        "--teacher-traces",
        required=True,
        help="dir of harbor job dirs and/or ATIF .json traces from the teacher",
    )
    p_distill.add_argument(
        "--teacher-backend",
        choices=("vllm", "fireworks", "bedrock"),
        default="vllm",
        help=(
            "where the teacher runs. vllm (default) is the reference path and the "
            "only unquantised one; fireworks and bedrock need no GPU but should "
            "pass `probe-teacher` first (docs/HOSTED_TEACHERS.md)"
        ),
    )
    p_distill.add_argument(
        "--teacher-api-base",
        default=None,
        help=(
            "vllm: the self-hosted server (required, needs prompt_logprobs — "
            "PLAN.md C1). fireworks: overrides the default gateway. bedrock: unused"
        ),
    )
    p_distill.add_argument(
        "--teacher-model-id",
        default=None,
        help=(
            "fireworks: `accounts/.../models/<id>` or a deployment path. "
            "bedrock: the imported model's ARN (required)"
        ),
    )
    p_distill.add_argument(
        "--teacher-region",
        default="us-east-1",
        help="bedrock only; must be a Custom Model Import region",
    )
    p_distill.add_argument(
        "--teacher-served-name",
        default=None,
        help="name the teacher endpoint serves under (default: discovered)",
    )
    p_distill.add_argument("--teacher", default=None, help="teacher HF id for the tokenizer check")
    p_distill.add_argument("--student", default=None, help="student HF id to train")
    p_distill.add_argument("--out", default="./vektori-out/opd", help="output directory")
    p_distill.add_argument("--max-steps", type=_positive_int_arg, default=200)
    p_distill.add_argument("--learning-rate", type=float, default=1e-5)
    p_distill.add_argument(
        "--examples-per-step",
        type=_positive_int_arg,
        default=4,
        help="examples accumulated per optimizer step (one teacher round-trip each)",
    )
    p_distill.add_argument(
        "--steps-per-trajectory",
        type=_positive_int_arg,
        default=None,
        help="cap ReOPD step-examples taken from each trajectory (default: all)",
    )
    p_distill.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="sampling temperature; 1.0 keeps the sample on-policy (see distill.py)",
    )
    p_distill.add_argument("--max-new-tokens", type=_positive_int_arg, default=256)
    p_distill.add_argument(
        "--top-k",
        type=int,
        default=0,
        help=(
            "0 (default) = reverse-KL surrogate over sampled tokens, the objective "
            "PLAN.md declares. >0 = analytic top-K KL (thunlp/OPD uses 16): same "
            "teacher cost, lower variance, but a different objective — pre-register "
            "before switching. Recorded in the run's provenance either way."
        ),
    )
    p_distill.add_argument("--seed", type=int, default=0)
    p_distill.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="trade step time for activation memory on a smaller card",
    )
    # ── Cross-tokenizer flags (FINAL-PLAN.md) ──────────────────────────────
    p_distill.add_argument(
        "--cross-tokenizer",
        action="store_true",
        dest="cross_tokenizer",
        help=(
            "enable cross-tokenizer OPD: student and teacher have different "
            "vocabularies; byte alignment maps both token streams. Requires "
            "--bridge and typically --teacher-backend fireworks."
        ),
    )
    p_distill.add_argument(
        "--bridge",
        default=None,
        metavar="PATH",
        help="CrossTokenizerBridge JSON artifact from `vektori-trace build-bridge`",
    )
    p_distill.add_argument(
        "--thinking-mode",
        default="chat",
        choices=("chat", "thinking"),
        dest="thinking_mode",
        help=(
            "teacher deployment inference mode; must match the bridge's thinking_mode "
            "(default: chat)"
        ),
    )
    p_distill.add_argument(
        "--min-granularity",
        type=float,
        default=0.5,
        dest="min_granularity",
        help=(
            "hard-fail if alignment granularity (spans/student tokens) is below this "
            "for any example (pre-registered floor: 0.5, FINAL-PLAN.md §4)"
        ),
    )
    p_distill.add_argument(
        "--max-span-student-tokens",
        type=int,
        default=8,
        dest="max_span_student_tokens",
        help=(
            "hard-fail if any aligned span covers more than this many student tokens "
            "(FINAL-PLAN.md §10.5; default: 8)"
        ),
    )
    p_distill.add_argument(
        "--cross-top-k",
        type=int,
        default=5,
        dest="cross_top_k",
        help=(
            "request top-K teacher logprobs per position for Estimator A "
            "(Fireworks caps at 5; set 0 to disable Estimator A entirely)"
        ),
    )
    p_distill.add_argument(
        "--teacher-tokenizer",
        default=None,
        dest="teacher_tokenizer_id",
        metavar="HF_ID",
        help=(
            "HF model id for the teacher tokenizer, used to re-tokenise the "
            "student's sampled action text on the teacher side. Required for "
            "--cross-tokenizer unless the teacher model id is already set."
        ),
    )
    p_distill.set_defaults(func=cmd_distill)

    p_tok = sub.add_parser(
        "check-tokenizers",
        help="Step 0: verify teacher/student share a tokenizer (hard-fail on mismatch)",
    )
    p_tok.add_argument(
        "--teacher", default=None, help=f"defaults to the pilot teacher ({DEFAULT_TEACHER})"
    )
    p_tok.add_argument(
        "--student", default=None, help=f"defaults to the pilot student ({DEFAULT_STUDENT})"
    )
    p_tok.set_defaults(func=cmd_check_tokenizers)

    p_bridge = sub.add_parser(
        "build-bridge",
        help=(
            "build a CrossTokenizerBridge JSON artifact from a teacher/student "
            "tokenizer pair (required for --cross-tokenizer distillation)"
        ),
    )
    p_bridge.add_argument(
        "--teacher-tokenizer",
        required=True,
        metavar="HF_ID",
        help="HF model id for the teacher tokenizer, e.g. deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    p_bridge.add_argument(
        "--student-tokenizer",
        required=True,
        metavar="HF_ID",
        help="HF model id for the student tokenizer, e.g. Qwen/Qwen3-8B",
    )
    p_bridge.add_argument(
        "--thinking-mode",
        default="chat",
        choices=("chat", "thinking"),
        help="teacher deployment inference mode (default: chat)",
    )
    p_bridge.add_argument(
        "--out",
        default="bridge.json",
        help="output path for the bridge artifact (default: bridge.json)",
    )
    p_bridge.set_defaults(func=cmd_build_bridge)

    p_align = sub.add_parser(
        "align-report",
        help=(
            "offline granularity report: encode text samples with both tokenizers "
            "and align by bytes, printing granularity per sample"
        ),
    )
    p_align.add_argument(
        "--bridge",
        required=True,
        metavar="PATH",
        help="CrossTokenizerBridge JSON artifact from `build-bridge`",
    )
    p_align.add_argument(
        "--text",
        default=None,
        metavar="FILE",
        help="file of text samples (one per line); defaults to stdin",
    )
    p_align.add_argument(
        "--max-span-student-tokens",
        type=int,
        default=8,
        dest="max_span_student_tokens",
        help="hard-fail threshold for span width (default: 8)",
    )
    p_align.set_defaults(func=cmd_align_report)

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

    p_gym = sub.add_parser(
        "import-gym",
        help="Step B: import R2E-Gym/SWE-smith JSONL into harbor task dirs",
    )
    p_gym.add_argument("--source", required=True, help="JSONL of gym instances")
    p_gym.add_argument("--out", required=True, help="output tasks directory")
    p_gym.add_argument("--limit", type=int, default=None)
    p_gym.set_defaults(func=cmd_import_gym)

    p_route = sub.add_parser(
        "route",
        help="Step F: apply routing rule to a passk JSON report → RL|OPD|QUARANTINE|NONE",
    )
    p_route.add_argument("--student-passk", required=True, help="passk JSON for the student")
    p_route.add_argument("--teacher-passk", required=True, help="passk JSON for the teacher")
    p_route.add_argument(
        "--diagnosis",
        default=None,
        help="report.json from `diagnose` — cells become (task × LACKING capability)",
    )
    p_route.add_argument(
        "--manifest",
        default=None,
        help="replay manifest.json, required with --diagnosis (run_id → task)",
    )
    p_route.add_argument(
        "--chosen-deficit-only",
        action="store_true",
        help="route only the chosen deficit instead of every ranked capability",
    )
    p_route.add_argument(
        "--capability",
        default="default",
        help="single label for every task; ignored when --diagnosis is given",
    )
    p_route.add_argument("--out", default="./vektori-out")
    p_route.set_defaults(func=cmd_route)

    p_bplan = sub.add_parser(
        "plan-b-arms",
        help="Step I: build B1–B4 assignment plans from routing.json (pilot caps at 10)",
    )
    p_bplan.add_argument("--routing", required=True, help="routing.json from `route`")
    p_bplan.add_argument("--out", default="./vektori-out")
    p_bplan.add_argument("--resolvable-effect-size", type=float, default=None)
    p_bplan.add_argument("--pilot", action="store_true")
    p_bplan.add_argument("--seed", type=int, default=0)
    p_bplan.add_argument(
        "--holdout",
        default=None,
        help="file of held-out task ids, one per line — removed from every training arm",
    )
    p_bplan.add_argument(
        "--preregistered-only",
        action="store_true",
        help="drop cells decided by a rule outside the pre-registration (mid band)",
    )
    p_bplan.set_defaults(func=cmd_plan_b_arms)

    p_resume = sub.add_parser(
        "resume-check",
        help="Step A: replay trajectory prefixes into fresh containers; report desync rate",
    )
    p_resume.add_argument("--manifest", required=True, help="replay manifest.json")
    p_resume.add_argument("--tasks-dir", required=True)
    p_resume.add_argument("--model", default=None, help="only replay this model's traces")
    p_resume.add_argument("--limit", type=int, default=None)
    p_resume.add_argument("--platform", default="linux/amd64")
    p_resume.add_argument("--out", default="./vektori-out")
    p_resume.set_defaults(func=cmd_resume_check)

    p_bisect = sub.add_parser(
        "bisect",
        help="Step D: verifier-guided bisection to the forking step of failed trajectories",
    )
    p_bisect.add_argument("--manifest", required=True)
    p_bisect.add_argument("--tasks-dir", required=True)
    p_bisect.add_argument("--model", default=None, help="only bisect this model's losses")
    p_bisect.add_argument("--teacher-model", default=None)
    p_bisect.add_argument(
        "--continuation-cmd",
        default=None,
        help=(
            "REQUIRED. Shell command run per probe, with {task_dir} and "
            "{prefix_json} substituted; exit 0 iff the verifier passes. The "
            "teacher must continue from the replayed prefix, and no harbor "
            "entrypoint accepts a seeded container yet."
        ),
    )
    p_bisect.add_argument(
        "--replay-prefix",
        action="store_true",
        help="also replay the prefix into a container and assert consistency (Step A)",
    )
    p_bisect.add_argument("--samples-per-probe", type=_positive_int_arg, default=2)
    p_bisect.add_argument("--verify-probes", type=int, default=2)
    p_bisect.add_argument("--platform", default="linux/amd64")
    p_bisect.add_argument("--limit", type=int, default=None)
    p_bisect.add_argument("--out", default="./vektori-out")
    p_bisect.set_defaults(func=cmd_bisect)

    p_ground = sub.add_parser(
        "ground",
        help="Step E: compare diagnose labels against execution-located forking steps",
    )
    p_ground.add_argument("--bisection", required=True, help="bisection.json from `bisect`")
    p_ground.add_argument("--diagnosis", required=True, help="report.json from `diagnose`")
    p_ground.add_argument(
        "--judgments",
        default=None,
        help='JSON {run_id: true|false} of hand-inspected agreement (AC #4)',
    )
    p_ground.add_argument("--min-gap", type=_min_gap_arg, default=DEFAULT_MIN_GAP)
    p_ground.add_argument("--out", default="./vektori-out")
    p_ground.set_defaults(func=cmd_ground)

    p_probe = sub.add_parser(
        "probe-teacher",
        help="one request: can this hosted teacher score tokens we supply?",
        description=(
            "Sends a single scoring request and reports what came back. This is "
            "the empirical check that decides whether a hosted teacher can run OPD "
            "at all — a 400 here is a finding, not a bug. Nothing else in the "
            "pipeline should be run against a teacher that has not passed it."
        ),
    )
    p_probe.add_argument(
        "--backend", choices=("fireworks", "bedrock"), required=True
    )
    p_probe.add_argument(
        "--model",
        default=None,
        help="Fireworks resource path, or the Bedrock imported-model ARN",
    )
    p_probe.add_argument("--api-base", default=None, help="fireworks only")
    p_probe.add_argument("--region", default="us-east-1", help="bedrock only")
    p_probe.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="also probe score_ids_topk at this K (Fireworks caps at 5)",
    )
    p_probe.add_argument("--out", default=None, help="write the result as JSON here")
    p_probe.add_argument(
        "--echo",
        action="store_true",
        help=(
            "also run probe_echo_support() to verify the teacher can score "
            "supplied token ids (the Fireworks echo=True capability OPD depends on)"
        ),
    )
    p_probe.set_defaults(func=cmd_probe_teacher)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Nothing configured logging, so every logger.info/warning in the mining
    # path went nowhere. `validate_pr` logs the one line that distinguishes
    # "this PR genuinely has no failing test" from "validation never ran"
    # ("parsed pre=0 post=0 tests"), and it was invisible for the whole
    # prefect smoke run — the skip histogram looked like a measurement.
    # VEKTORI_LOG_LEVEL overrides; INFO is the useful default for long mines.
    logging.basicConfig(
        level=os.environ.get("VEKTORI_LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
