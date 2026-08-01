#!/usr/bin/env python3
"""Hold a Modal-hosted vLLM student up until Ctrl-C, and print its api_base.

`serve.serve_model` is a context manager: the endpoint dies when the block
exits. That is right for `arms.py`, which owns the server's lifetime. It is
wrong for a pass@k sweep you want to run by hand, or for two people sharing one
endpoint from a shared box. This script keeps the block open.

Run it inside tmux. The endpoint lives exactly as long as this process.

    tmux new -s serve
    uv run python scripts/serve_student.py --gpu A10G --max-model-len 8192
    # Ctrl-B D to detach; the endpoint stays up

Then, in another window:

    export STUDENT_API_BASE=<printed url>
    uv run python scripts/vllm_monitor.py --api-base $STUDENT_API_BASE
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vektori_trace.serve import serve_model, served_to_harbor_kwargs

# Qwen3-8B, measured from its config.json:
#   36 layers, 8 KV heads (GQA), head_dim 128
#   KV/token = 2(K,V) * 36 * 8 * 128 * 2 bytes = 144 KiB
KV_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2
WEIGHTS_GIB = 15.3          # 8.19e9 params * 2 bytes (bf16)
OVERHEAD_GIB = 1.3          # CUDA context, activations, cuda graphs
GPU_VRAM_GIB = {"A10G": 24, "L40S": 48, "A100": 40, "A100-80GB": 80, "H100": 80}


def capacity(gpu: str, util: float) -> tuple[float, int]:
    """(KV GiB, total KV tokens) available on `gpu` at `util`."""
    vram = GPU_VRAM_GIB.get(gpu, 24)
    kv_gib = vram * util - WEIGHTS_GIB - OVERHEAD_GIB
    return kv_gib, int(kv_gib * (1024**3) / KV_BYTES_PER_TOKEN)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--gpu", default="A10G",
                    help="Modal GPU: A10G ($1.10/h, 24GB) | L40S ($1.95/h, 48GB)")
    ap.add_argument("--adapter-path", default=None,
                    help="Volume path (/adapters/...) to a LoRA adapter")
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="max tokens per sequence. MUST fit the KV budget or "
                         "vLLM refuses to start (default: 8192)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--write-env", default=None,
                    help="also append STUDENT_API_BASE=<url> to this file")
    args = ap.parse_args()

    kv_gib, kv_tokens = capacity(args.gpu, args.gpu_memory_utilization)
    concurrency = kv_tokens // args.max_model_len if args.max_model_len else 0

    print(f"gpu                 {args.gpu}")
    print(f"model               {args.base_model}  (~{WEIGHTS_GIB} GiB bf16)")
    print(f"kv budget           {kv_gib:.2f} GiB  =  {kv_tokens:,} tokens total")
    print(f"max-model-len       {args.max_model_len:,}")
    print(f"→ concurrent seqs   {concurrency}  at full length")

    if kv_gib <= 0:
        print("\nFATAL: no KV budget left after weights. Use a bigger GPU.",
              file=sys.stderr)
        return 2
    if args.max_model_len > kv_tokens:
        print(f"\nFATAL: --max-model-len {args.max_model_len:,} exceeds the "
              f"{kv_tokens:,}-token KV budget. vLLM will refuse to start.\n"
              f"       Lower it, raise --gpu-memory-utilization, or use --gpu L40S.",
              file=sys.stderr)
        return 2
    if concurrency < 2:
        print(f"\nWARNING: only {concurrency} sequence fits at full length. "
              "pass@k will run effectively serially.", file=sys.stderr)

    print("\nstarting (first run downloads ~16GB into the hf-model-cache "
          "Volume; later starts are fast)...\n", flush=True)

    stop = False

    def _sig(_s, _f):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    t0 = time.time()
    with serve_model(
        args.base_model,
        adapter_path=args.adapter_path,
        gpu=args.gpu,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    ) as served:
        print(f"UP in {time.time() - t0:.0f}s\n")
        print(f"  STUDENT_API_BASE={served.api_base}")
        print(f"  model_name       {served.model_name}")
        print(f"  harbor model     {served.harbor_model}\n")
        print("harbor kwargs:")
        print(json.dumps(served_to_harbor_kwargs(served), indent=2))
        if args.write_env:
            with open(args.write_env, "a") as fh:
                fh.write(f"\nSTUDENT_API_BASE={served.api_base}\n")
            print(f"\nappended STUDENT_API_BASE to {args.write_env}")
        print("\nendpoint is live. Ctrl-C to tear it down.\n", flush=True)
        while not stop:
            time.sleep(2)
    print("torn down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
