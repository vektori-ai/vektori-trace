"""`mine` and `mine-commits` — trace collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ...mining.inspect import audit_tasks, failure_histogram
from ...mining.miner import (
    HarborTraceRunner,
    collect_traces,
    mine_commits,
    mine_tasks,
)
from ...mining.spec import CommitRuntimeOptions, LLMSpec, PRRuntimeOptions


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

def register_mine(sub: argparse._SubParsersAction) -> None:
    """Register the `mine` subcommand on `sub`."""
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

def register_mine_commits(sub: argparse._SubParsersAction) -> None:
    """Register the `mine-commits` subcommand on `sub`."""
    p_mine_commits = sub.add_parser(
        "mine-commits",
        help=(
            "mine a repo's COMMIT history into sandbox-verified tasks. Sibling of `mine`: "
            # argparse runs help strings through %-formatting, so a literal
            # percent must be doubled or "% o" is read as an octal format spec
            # and `--help` dies with a TypeError.
            "reaches fixes that never became a PR with a linked issue (54%% of candidates in "
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
