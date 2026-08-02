"""The argparse tree. Imports every command module to wire `func=`."""

from __future__ import annotations

import argparse

from ..diagnose import (
    DEFAULT_MIN_GAP,
    DEFAULT_MIN_SUPPORT,
)
from ..passrate import DEFAULT_ROLLOUTS, PASSRATE_MAX, PASSRATE_MIN
from ..tokenizer_check import DEFAULT_STUDENT, DEFAULT_TEACHER
from ._args import _add_endpoint_args, _min_gap_arg, _min_support_arg, _positive_int_arg
from .commands.capture import cmd_capture_proxy
from .commands.diagnose import cmd_diagnose
from .commands.distill import cmd_distill
from .commands.env import cmd_checkenv, cmd_prove, cmd_selftest
from .commands.evaluate import cmd_import_gym, cmd_passk
from .commands.ground import cmd_ground
from .commands.mine import cmd_mine, cmd_mine_commits
from .commands.replay import cmd_replay
from .commands.resume import cmd_bisect, cmd_resume_check
from .commands.route import cmd_plan_b_arms, cmd_route
from .commands.select import cmd_select
from .commands.teacher import (
    cmd_align_report,
    cmd_build_bridge,
    cmd_check_tokenizers,
    cmd_probe_teacher,
)
from .commands.train import cmd_run_arms, cmd_train


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
    p_train.add_argument("--modal-gpu", default="L40S")
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
    p_arms.add_argument("--modal-gpu", default="L40S")
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
    p_arms.add_argument(
        "--capture-tokens",
        action="store_true",
        help=(
            "Phase 0.5: request vLLM return_token_ids during rollout collection and "
            "persist sampled ids next to each harbor job. Required before OPD/GRPO; "
            "optional for SFT (A2/A3 still re-tokenizeize when captures are absent)."
        ),
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
    p_passk.add_argument(
        "--no-escalate",
        action="store_true",
        help=(
            "run stage 1 only. Escalation fires on c == 0 regardless of how big "
            "stage 1 was, so a small --stage1-n silently becomes --stage2-n "
            "rollouts the moment a task fails"
        ),
    )
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

    p_cap = sub.add_parser(
        "capture-proxy",
        help=(
            "Phase 0.5: reverse-proxy a vLLM server, inject return_token_ids, and "
            "persist sampled prompt/completion ids as JSONL"
        ),
        description=(
            "Harbor agents we do not control still need sampled token ids for OPD. "
            "This proxy sits in front of your vLLM OpenAI-compatible server, forces "
            "`return_token_ids: true` on every chat/completions request, forwards "
            "the response unchanged, and appends each capture to "
            "<out>/token_captures.jsonl. Point harbor's api_base at the printed URL."
        ),
    )
    p_cap.add_argument(
        "--upstream",
        required=True,
        help="real vLLM api base, e.g. http://127.0.0.1:8000/v1",
    )
    p_cap.add_argument(
        "--out",
        default="./vektori-out/token-captures",
        help="directory for token_captures.jsonl",
    )
    p_cap.add_argument("--host", default="127.0.0.1")
    p_cap.add_argument(
        "--port",
        type=int,
        default=0,
        help="local listen port (0 = ephemeral; the printed api_base always wins)",
    )
    p_cap.add_argument(
        "--logprobs",
        action="store_true",
        help="also request per-token logprobs alongside token ids",
    )
    p_cap.set_defaults(func=cmd_capture_proxy)

    return parser
