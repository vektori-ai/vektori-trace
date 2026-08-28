#!/usr/bin/env python3
"""Run one capture-only live Tau2 update.

The plan is a JSON list of ``{"episode_id", "task_id", "seed"}`` objects.
This command writes the existing ReOPD update layout and stops at SAMPLED; it
does not call the teacher or move weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vektori_trace.tau2.live_rollout import (
    EpisodePlan,
    RolloutSettings,
    Tau2EpisodeRunner,
    capture_live_update,
)
from vektori_trace.tau2.reopd_refresh import served_models
from vektori_trace.tau2.reopd_state import RunState


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--update", type=int, default=0)
    p.add_argument("--n-updates", type=int, required=True)
    p.add_argument("--plan", required=True, help="JSON episode-plan file")
    p.add_argument("--domain", default="retail")
    p.add_argument("--api-base", required=True)
    p.add_argument("--student-model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--policy-version", required=True)
    p.add_argument("--adapter-hash", required=True)
    p.add_argument("--gen-config-hash", required=True)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--max-input-tokens", type=int, required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--max-errors", type=int, default=10)
    p.add_argument("--user", default="user_simulator")
    p.add_argument("--user-model", required=True)
    p.add_argument("--user-model-args", default="{}")
    p.add_argument("--teacher-model", required=True)
    p.add_argument("--teacher-tokenizer", required=True)
    p.add_argument("--teacher-renderer", default="deepseek-v4-native")
    p.add_argument(
        "--allow-missing-reasoning",
        action="store_true",
        help="diagnostic only: admit actions without a non-empty <think> span",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_plans = json.loads(Path(args.plan).read_text())
    plans = [EpisodePlan(**row) for row in raw_plans]

    advertised = served_models(args.api_base, timeout=min(args.timeout, 30.0))
    if args.student_model not in advertised:
        raise RuntimeError(
            f"endpoint does not advertise {args.student_model!r}; it serves "
            f"{advertised}. Refusing a possible silent fallback to base weights."
        )

    settings = RolloutSettings(
        domain=args.domain,
        student_model=args.student_model,
        api_base=args.api_base,
        policy_version=args.policy_version,
        adapter_hash=args.adapter_hash,
        gen_config_hash=args.gen_config_hash,
        max_tokens=args.max_tokens,
        max_input_tokens=args.max_input_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        user=args.user,
        user_model=args.user_model,
        user_model_args=json.loads(args.user_model_args),
        require_reasoning=not args.allow_missing_reasoning,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    state = RunState(args.run_dir, n_updates=args.n_updates)
    state.freeze_manifest(
        {
            "kind": "tau2_live_opd",
            "domain": args.domain,
            "n_updates": args.n_updates,
            "student_model": args.student_model,
            "student_tokenizer": args.tokenizer,
            "teacher_model": args.teacher_model,
            "teacher_tokenizer": args.teacher_tokenizer,
            "teacher_renderer": args.teacher_renderer,
        }
    )
    report = capture_live_update(
        state.update(args.update),
        plans,
        settings=settings,
        teacher_context={
            "model": args.teacher_model,
            "tokenizer": args.teacher_tokenizer,
            "renderer": args.teacher_renderer,
        },
        runner=Tau2EpisodeRunner(settings, tokenizer),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
