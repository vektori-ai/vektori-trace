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
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class GPUSampler:
    """Poll nvidia-smi in the background and stream every sample.

    `torch.cuda.max_memory_allocated` reports what the *allocator* handed out.
    It does not see CUDA context, cuBLAS/cuDNN workspaces, kernel scratch, or
    fragmentation, and those are exactly the components that decide whether a
    48 GiB card would have held this. nvidia-smi sees all of it, so both are
    recorded and the gap between them is itself the answer.
    """

    def __init__(self, path: Path, interval: float = 2.0):
        self.path, self.interval = Path(path), interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._phase = "init"
        self._t = threading.Thread(target=self._run, daemon=True)

    def phase(self, name: str) -> None:
        self._phase = name

    def start(self):
        self._t.start()
        return self

    def _run(self) -> None:
        fh = self.path.open("w")
        while not self._stop.is_set():
            try:
                raw = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=memory.used,memory.total,utilization.gpu,"
                     "utilization.memory,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip().splitlines()[0]
                used, total, ugpu, umem, temp, power = [
                    x.strip() for x in raw.split(",")
                ]
                row = {
                    "t": round(time.time(), 2),
                    "phase": self._phase,
                    "mem_used_mib": int(float(used)),
                    "mem_total_mib": int(float(total)),
                    "util_gpu_pct": int(float(ugpu)),
                    "util_mem_pct": int(float(umem)),
                    "temp_c": int(float(temp)),
                    "power_w": float(power),
                }
                self.samples.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(
                    f"[gpu] {row['phase']:<10} {row['mem_used_mib']:>6}/"
                    f"{row['mem_total_mib']} MiB  util {row['util_gpu_pct']:>3}%  "
                    f"{row['power_w']:.0f}W  {row['temp_c']}C",
                    flush=True,
                )
            except Exception as e:  # a sampler must never kill the run
                print(f"[gpu] sample failed: {type(e).__name__}: {e}", flush=True)
            self._stop.wait(self.interval)
        fh.close()

    def stop(self) -> dict:
        self._stop.set()
        self._t.join(timeout=15)
        if not self.samples:
            return {"n_samples": 0}
        peak = max(s["mem_used_mib"] for s in self.samples)
        total = self.samples[0]["mem_total_mib"]
        by_phase: dict[str, int] = {}
        for s in self.samples:
            by_phase[s["phase"]] = max(by_phase.get(s["phase"], 0), s["mem_used_mib"])
        return {
            "n_samples": len(self.samples),
            "peak_mem_used_mib": peak,
            "peak_mem_used_gib": round(peak / 1024, 2),
            "mem_total_mib": total,
            "peak_by_phase_mib": by_phase,
            "max_util_gpu_pct": max(s["util_gpu_pct"] for s in self.samples),
            "mean_util_gpu_pct": round(
                sum(s["util_gpu_pct"] for s in self.samples) / len(self.samples), 1
            ),
            "max_power_w": max(s["power_w"] for s in self.samples),
            "max_temp_c": max(s["temp_c"] for s in self.samples),
        }


def mem_components(torch) -> dict:
    """Allocator view, broken out. `reserved - allocated` is fragmentation."""
    a = torch.cuda.memory_allocated()
    r = torch.cuda.memory_reserved()
    return {
        "allocated_gib": round(a / 1024**3, 2),
        "reserved_gib": round(r / 1024**3, 2),
        "fragmentation_gib": round((r - a) / 1024**3, 2),
    }


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
    ap.add_argument("--gpu-log", default="/data/preflight_gpu.jsonl",
                    help="one nvidia-smi sample per line, written live")
    ap.add_argument("--sample-interval", type=float, default=2.0)
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

    sampler = GPUSampler(Path(args.gpu_log), interval=args.sample_interval).start()
    sampler.phase("baseline")
    report["components"] = {}

    cfg = ReplayTrainConfig(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        device=args.device,
        gradient_checkpointing=not args.no_checkpointing,
    )

    log("loading ck75 (bf16 + LoRA)...")
    sampler.phase("load")
    t0 = time.time()
    model = load_v0_for_training(cfg)
    torch.cuda.synchronize()
    report["components"]["weights"] = mem_components(torch)
    log(f"  weights: {report['components']['weights']['allocated_gib']} GiB")

    opt = build_optimizer(model, cfg)
    torch.cuda.synchronize()
    report["components"]["after_optimizer_built"] = mem_components(torch)
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
    sampler.phase("forward")
    t0 = time.time()
    cur = action_logprobs_under_prefix(
        model, prompt_ids, action_ids, device=args.device
    )
    torch.cuda.synchronize()
    report["forward_seconds"] = round(time.time() - t0, 1)
    report["after_forward_peak_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    report["components"]["after_forward"] = mem_components(torch)
    log(f"forward {report['forward_seconds']}s, peak {report['after_forward_peak_gib']} GiB")

    # The real loss shape: detached advantages and behaviour logprobs, clipped
    # importance ratio, one global denominator.
    from vektori_trace.chunk_opd import DEFAULT_CLIP_EPS, clipped_is_policy_loss

    behavior = cur.detach().clone()
    advantages = torch.full_like(behavior, 0.1)
    log("backward...")
    sampler.phase("backward")
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

    report["components"]["after_backward"] = mem_components(torch)
    grads = [p.grad for _n, p in model.named_parameters() if p.grad is not None]
    report["params_with_grad"] = len(grads)
    report["all_grads_finite"] = all(bool(torch.isfinite(g).all()) for g in grads)
    report["grad_bytes_gib"] = round(
        sum(g.numel() * g.element_size() for g in grads) / 1024**3, 3
    )

    # AdamW allocates exp_avg + exp_avg_sq lazily on the first step, so this is
    # the first point at which optimizer state is real rather than projected.
    sampler.phase("opt_step")
    opt.step()
    torch.cuda.synchronize()
    report["components"]["after_opt_step"] = mem_components(torch)
    opt_state_gib = round(
        sum(
            v.numel() * v.element_size()
            for st in opt.state.values()
            for v in st.values()
            if hasattr(v, "numel")
        ) / 1024**3,
        3,
    )
    report["optimizer_state_gib"] = opt_state_gib
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    report["after_step_peak_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    report["headroom_gib"] = round(report["gpu_total_gib"] - report["peak_reserved_gib"], 2)
    report["passed"] = bool(
        report["all_grads_finite"] and report["params_with_grad"] > 0
    )

    sampler.phase("done")
    report["gpu_sampler"] = sampler.stop()

    # The question this run exists to answer for the *next* one.
    smi_peak = report["gpu_sampler"].get("peak_mem_used_gib")
    if smi_peak is not None:
        report["l40s_verdict"] = {
            "l40s_total_gib": 48.0,
            "observed_peak_gib": smi_peak,
            "fits_l40s": smi_peak < 45.0,
            "margin_gib": round(45.0 - smi_peak, 2),
            "note": "nvidia-smi peak includes CUDA context, cuBLAS workspaces and "
                    "fragmentation, which torch's allocator counters do not. "
                    "45 GiB rather than 48 leaves room for run-to-run variance.",
        }
        log(f"L40S verdict: peak {smi_peak} GiB — "
            f"{'WOULD FIT' if smi_peak < 45.0 else 'WOULD NOT FIT'}")

    log(f"RESULT: torch peak reserved {report['peak_reserved_gib']} GiB, "
        f"nvidia-smi peak {smi_peak} GiB of {report['gpu_total_gib']}")
    print(json.dumps(report, indent=2))
    Path(args.out).write_text(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
