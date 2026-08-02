"""`distill` — teacher-to-student distillation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _load_teacher_trajectories(source: Path) -> list[tuple[str, list[Any]]]:
    """Teacher trajectories from harbor job dirs or ATIF JSON traces.

    Two shapes, because two things produce them: `replay` leaves harbor job
    directories, and mined/example traces are JSON. Anything unreadable is
    reported rather than skipped — a silently smaller corpus changes what the run
    measured.
    """
    from ...mining.atif import TrajectoryParseError, parse_job_trajectory
    from ...schema import Trace

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
