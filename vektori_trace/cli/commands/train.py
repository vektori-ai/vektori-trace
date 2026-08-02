"""`train` and `run-arms` — local training entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_train(args: argparse.Namespace) -> int:
    """One arm: serve → rejection-sample rollouts → LoRA SFT → adapter + report.

    Train extras are imported lazily so a base install never pays for torch.
    """
    # Lazy: torch/transformers/peft/modal must not be imported at cli.py top-level.
    from ...dataset import tokenize_sft_example
    from ...rollout import collect_rollouts
    from ...serve import serve_model, served_to_harbor_kwargs
    from ...train import TrainConfig, run_training, write_train_report

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
        from ...endpoint import endpoint_serve_cm

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
    from ...arms import DEFAULT_CANDIDATE_MODEL, ArmsConfig, run_arms

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
        capture_tokens=bool(getattr(args, "capture_tokens", False)),
    )
    try:
        report = run_arms(cfg)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    paths = report.get("_paths") or {}
    print(f"arms report: {paths.get('md', Path(args.out) / 'arms.md')}")
    return 0
