"""`distill` — teacher-to-student distillation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .._args import _positive_int_arg


def _load_teacher_trajectories(source: Path) -> list[tuple[str, list[Any]]]:
    """Thin alias — the implementation lives in `vektori_trace.traces`.

    Kept as a name here because the CLI and its tests reference it, but the
    loader itself must be importable without dragging the CLI package (and its
    `openai` dependency) onto a GPU container.
    """
    from ...traces import load_teacher_trajectories

    return load_teacher_trajectories(source)


def _teacher_pool_for(args: argparse.Namespace) -> Any:
    """The teacher named by `--teacher-backend`, checked before any GPU time.

    Every backend fails fast here rather than on the first scoring call halfway
    into a run: whether the teacher can score supplied tokens at all is the
    precondition for the entire method (`docs/OPD.md`), and finding out late costs
    whatever the student instance has already billed.
    """
    backend = getattr(args, "teacher_backend", "vllm")
    if backend == "vllm":
        from ...providers.teacher.base import teacher_pool_from_endpoint

        if not args.teacher_api_base:
            raise ValueError("--teacher-api-base is required for --teacher-backend vllm")
        return teacher_pool_from_endpoint(
            args.teacher_api_base, model=args.teacher_served_name
        )
    if backend == "fireworks":
        from ...providers.teacher.fireworks import fireworks_pool_from_env

        return fireworks_pool_from_env(
            model=args.teacher_model_id, api_base=args.teacher_api_base
        )
    from ...providers.teacher.bedrock import BedrockTeacherPool

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
    from ...distill import OPDTrainConfig, run_opd_training, write_opd_report
    from ...providers.teacher.base import TeacherScoringError
    from ...reopd import iter_reopd_examples
    from ...runtime.endpoint import EndpointError
    from ...tokenizer_check import DEFAULT_STUDENT, DEFAULT_TEACHER, TokenizerMismatchError

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
        from ...vocab_bridge import CrossTokenizerBridge
        try:
            _bridge = CrossTokenizerBridge.load(bridge_path)
        except Exception as e:
            print(f"error: cannot load bridge {bridge_path}: {e}", file=sys.stderr)
            return 2

        if teacher_tok_id:
            from ...vocab_bridge import load_tokenizer
            _teacher_tok = load_tokenizer(teacher_tok_id)

        # Wrap pool in CrossTokenizerTeacherPool for provenance recording.
        from ...providers.teacher.cross import CrossTokenizerTeacherPool
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
    modal_gpu = getattr(args, "modal_gpu", None)
    try:
        if modal_gpu:
            # Everything above ran locally and is CPU/network only, so a bad
            # config still fails before a GPU exists. Only the training itself
            # goes remote. Modal is opt-in — never the default — because
            # allocating a GPU is a spend decision, not a fallback.
            if not cross_tokenizer:
                print(
                    "error: --modal-gpu currently covers the cross-tokenizer "
                    "path only (it ships the bridge to the container)",
                    file=sys.stderr,
                )
                return 2
            from ...distill import run_opd_training_modal

            result = run_opd_training_modal(
                cfg,
                prefixes_dir=Path(args.teacher_traces),
                bridge_path=Path(bridge_path),
                fireworks_model=args.teacher_model_id,
                teacher_tokenizer_id=teacher_tok_id or cfg.teacher_model,
                api_base=args.teacher_api_base,
                steps_per_trajectory=args.steps_per_trajectory,
                modal_gpu=modal_gpu,
                expected_examples=len(examples),
            )
        else:
            result = run_opd_training(
                examples, pool, cfg,
                bridge=_bridge,
                teacher_tokenizer=_teacher_tok,
            )
    except TokenizerMismatchError as e:
        print(f"error: teacher/student tokenizers differ: {e}", file=sys.stderr)
        return 2
    except (RuntimeError, TeacherScoringError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        # `align.AlignmentError` subclasses ValueError and is the likeliest
        # cross-tokenizer failure. Modal re-raises the original type client
        # side, so without this a multi-hour run ends as a bare traceback.
        print(f"error: alignment/config failure: {e}", file=sys.stderr)
        return 1

    md = write_opd_report(result, out_dir)
    print(f"adapter: {result.adapter_dir}")
    print(f"steps: {result.steps}  final loss: {result.final_loss}")
    print(f"mean log ratio: {result.mean_log_ratio_final}")
    if result.skipped_empty_samples:
        print(f"note: {result.skipped_empty_samples} example(s) sampled no tokens")
    print(f"report: {md}")
    return 0

def register_distill(sub: argparse._SubParsersAction) -> None:
    """Register the `distill` subcommand on `sub`."""
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
    p_distill.add_argument(
        "--modal-gpu",
        default=None,
        metavar="GPU",
        help=(
            "run the training on this Modal GPU (e.g. L40S) instead of in this "
            "process. Opt-in only: omitting it never allocates a GPU. Requires "
            "--cross-tokenizer and FIREWORKS_API_KEY in the environment"
        ),
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
