#!/usr/bin/env python3
"""One-example GPU memory preflight (plan §14).

Answers exactly one question: **does the worst case that can actually be
sampled survive a forward + backward + optimizer step on this card?**

Only the maximum matters. The serving window is fixed at 40,960 and compaction
fires before anything exceeds it, so the largest fitting prefix dominates every
smaller one — if it passes, they all pass by construction. Smaller prefixes are
useful only to bisect a failure.

Deliberately not paid beyond the GPU:

- **Mock teacher scores.** The loss needs per-token advantages; where they came
  from is irrelevant to memory. A Fireworks call would spend money to learn
  nothing.
- **Synthetic action at the cap.** Real actions are shorter (median 534
  tokens), so a 9,216-token action is the worst case rather than a typical one.
- **No adapter save**, no sampling, no Harbor.

    python scripts/preflight_memory.py --adapter-path /adapters/... --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-14B")
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prefix-tokens", type=int, default=31591,
                    help="the largest fitting prefix in the reconstructed census")
    ap.add_argument("--action-tokens", type=int, default=9216,
                    help="the §8.3 cap; real actions are shorter")
    ap.add_argument("--no-checkpointing", action="store_true")
    ap.add_argument("--out", default="/data/preflight_memory.json")
    args = ap.parse_args()

    import torch

    from vektori_trace.replay_train import (
        ReplayTrainConfig,
        action_logprobs_under_prefix,
        build_optimizer,
        load_v0_for_training,
    )

    report: dict = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "prefix_tokens": args.prefix_tokens,
        "action_tokens": args.action_tokens,
        "total_tokens": args.prefix_tokens + args.action_tokens,
        "gradient_checkpointing": not args.no_checkpointing,
    }

    props = torch.cuda.get_device_properties(0)
    report["gpu"] = props.name
    report["gpu_total_gib"] = round(props.total_memory / 1024**3, 2)
    log(f"{props.name}, {report['gpu_total_gib']} GiB")

    cfg = ReplayTrainConfig(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        device=args.device,
        gradient_checkpointing=not args.no_checkpointing,
    )

    log("loading ck75 (bf16 + LoRA)...")
    t0 = time.time()
    model = load_v0_for_training(cfg)
    opt = build_optimizer(model, cfg)
    torch.cuda.synchronize()
    report["load_seconds"] = round(time.time() - t0, 1)
    report["after_load_gib"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
    log(f"loaded in {report['load_seconds']}s, {report['after_load_gib']} GiB allocated")

    torch.cuda.reset_peak_memory_stats()

    # Ids are arbitrary: memory depends on sequence length, not content. Kept
    # inside the vocab so nothing indexes out of range.
    vocab = int(model.config.vocab_size)
    prompt_ids = [(i * 7919) % vocab for i in range(args.prefix_tokens)]
    action_ids = [(i * 104729) % vocab for i in range(args.action_tokens)]

    log(f"forward over {report['total_tokens']:,} tokens...")
    t0 = time.time()
    cur = action_logprobs_under_prefix(
        model, prompt_ids, action_ids, device=args.device
    )
    torch.cuda.synchronize()
    report["forward_seconds"] = round(time.time() - t0, 1)
    report["after_forward_peak_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    log(f"forward {report['forward_seconds']}s, peak {report['after_forward_peak_gib']} GiB")

    # The real loss shape: detached advantages and behaviour logprobs, clipped
    # importance ratio, one global denominator.
    from vektori_trace.chunk_opd import DEFAULT_CLIP_EPS, clipped_is_policy_loss

    behavior = cur.detach().clone()
    advantages = torch.full_like(behavior, 0.1)
    log("backward...")
    t0 = time.time()
    # Already reduced to a scalar with its own denominator.
    loss = clipped_is_policy_loss(
        cur, behavior, advantages,
        clip_eps=DEFAULT_CLIP_EPS,
        denominator=float(len(action_ids)),
    )
    loss.backward()
    torch.cuda.synchronize()
    report["backward_seconds"] = round(time.time() - t0, 1)
    report["peak_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    report["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    report["loss"] = float(loss.detach())
    log(f"backward {report['backward_seconds']}s, PEAK {report['peak_gib']} GiB")

    grads = [p.grad for _n, p in model.named_parameters() if p.grad is not None]
    report["params_with_grad"] = len(grads)
    report["all_grads_finite"] = all(bool(torch.isfinite(g).all()) for g in grads)

    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    report["after_step_peak_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    report["headroom_gib"] = round(report["gpu_total_gib"] - report["peak_reserved_gib"], 2)
    report["passed"] = bool(
        report["all_grads_finite"] and report["params_with_grad"] > 0
    )

    log(f"RESULT: peak reserved {report['peak_reserved_gib']} GiB of "
        f"{report['gpu_total_gib']}, headroom {report['headroom_gib']} GiB")
    print(json.dumps(report, indent=2))
    Path(args.out).write_text(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
