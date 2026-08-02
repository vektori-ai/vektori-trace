#!/usr/bin/env python3
"""Hold a Modal-hosted vLLM student up until Ctrl-C, and print its api_base.

`serve.serve_model` is a context manager: the endpoint dies when the block
exits. That is right for `arms.py`, which owns the server's lifetime. It is
wrong for a pass@k sweep you want to run by hand, or for two people sharing one
endpoint from a shared box. This script keeps the block open.

Run it inside tmux. The endpoint lives exactly as long as this process.

    tmux new -s serve
    uv run python scripts/serve_student.py --gpu L40S --max-model-len 40960
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
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vektori_trace.runtime.serve import serve_model, served_to_harbor_kwargs

# Qwen3-8B, measured from its config.json:
#   36 layers, 8 KV heads (GQA), head_dim 128
#   KV/token = 2(K,V) * 36 * 8 * 128 * 2 bytes = 144 KiB
KV_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2
WEIGHTS_GIB = 15.3          # 8.19e9 params * 2 bytes (bf16)
OVERHEAD_GIB = 1.3          # CUDA context, activations, cuda graphs
# Keys are Modal's own `gpu=` strings (modal.com/docs/guide/gpu). Note "A10",
# not "A10G" — Modal renamed it, and the old spelling is rejected before any
# container starts.
GPU_VRAM_GIB = {
    "T4": 16,
    "L4": 24,
    "A10": 24,
    "L40S": 48,
    "A100-40GB": 40,
    "A100": 40,
    "A100-80GB": 80,
    "H100": 80,
    "H200": 141,
    "B200": 180,
}


def capacity(gpu: str, util: float) -> tuple[float, int]:
    """(KV GiB, total KV tokens) available on `gpu` at `util`.

    Unknown GPU is an error, not a default. This previously fell back to 24 GiB,
    which silently computes an A10-sized budget for an H100 — the guard then
    blocks a `--max-model-len` that would have fit fine, or worse, the reverse.
    """
    if gpu not in GPU_VRAM_GIB:
        raise KeyError(
            f"unknown GPU {gpu!r}; known: {', '.join(sorted(GPU_VRAM_GIB))}. "
            "Use Modal's own gpu= strings (modal.com/docs/guide/gpu)."
        )
    kv_gib = GPU_VRAM_GIB[gpu] * util - WEIGHTS_GIB - OVERHEAD_GIB
    return kv_gib, int(kv_gib * (1024**3) / KV_BYTES_PER_TOKEN)


class _GpuSampler:
    """Poll the serving container's NVML on an interval into a JSONL on THIS box.

    Sampling from here rather than from inside the container is deliberate: the
    log has to survive the container dying, and a container dying is exactly the
    event you want telemetry for. The file lives on EBS and outlives everything.

    Runs in a daemon thread off the process that already holds the Modal session,
    because `ServedModel.sample_gpu` is a handle into that session and cannot be
    called from an unrelated process.
    """

    def __init__(self, served, path: Path, interval: float):
        self._served = served
        self._path = path
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.n_samples = 0
        self.n_errors = 0

    def start(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            record = {"logged_at": time.time()}
            try:
                sample = self._served.sample_gpu()
                record.update(sample if isinstance(sample, dict) else {"raw": sample})
            except Exception as e:  # telemetry must never kill the endpoint
                record["error"] = f"{type(e).__name__}: {e}"
                self.n_errors += 1
            try:
                with self._path.open("a") as fh:
                    fh.write(json.dumps(record) + "\n")
                    fh.flush()
                self.n_samples += 1
            except OSError:
                pass
            self._stop.wait(self._interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _smoke(api_base: str, model: str, timeout: float = 120.0) -> tuple[bool, str]:
    """One real completion. Returns (ok, human-readable detail)."""
    import json as _json
    import urllib.error
    import urllib.request

    body = _json.dumps({
        "model": model,
        "prompt": "1 + 1 =",
        "max_tokens": 1,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        api_base.rstrip("/") + "/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = _json.loads(r.read())
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read()[:200].decode('utf-8', 'replace')}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    choices = payload.get("choices") or []
    if not choices:
        return False, f"no choices in response: {str(payload)[:200]}"
    return True, f"{time.time() - t:.1f}s, got {choices[0].get('text','')!r}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--gpu", default="L40S",
                    help="Modal GPU: L40S ($1.95/h, 48GB) | A10 ($1.10/h, 24GB). "
                         "L40S is the default because Qwen3-8B bf16 leaves only "
                         "5 GiB of KV on an A10, below vLLM's own 40960 default")
    ap.add_argument("--adapter-path", default=None,
                    help="Volume path (/adapters/...) to a LoRA adapter")
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="max tokens per sequence. MUST fit the KV budget or "
                         "vLLM refuses to start (default: 8192)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-hours", type=float, default=12.0,
                    help="auto-shutdown after N hours so a forgotten endpoint "
                         "cannot bill indefinitely; 0 disables (default: 12)")
    ap.add_argument("--write-env", default=None,
                    help="also append STUDENT_API_BASE=<url> to this file")
    ap.add_argument("--gpu-log", default="gpu_log.jsonl",
                    help="NVML samples (util%%, memory, temp, power, SM clock) "
                         "appended here on this box; empty string disables")
    ap.add_argument("--gpu-log-interval", type=float, default=5.0,
                    help="seconds between NVML samples (default: 5)")
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

    # SIGHUP matters as much as the other two, and costs money when missed.
    # `tmux kill-session` sends SIGHUP; so does closing the terminal that owns
    # the process. Untrapped, Python dies without unwinding the `with` block, so
    # serve_model's __exit__ never runs, the Modal app stays "ephemeral" with a
    # live container, and the GPU keeps billing with nothing attached to it.
    # Observed: a killed session left an A10G running until it was stopped by
    # hand. Overnight that is real money for zero work.
    for _s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(_s, _sig)

    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None

    t0 = time.time()
    with serve_model(
        args.base_model,
        adapter_path=args.adapter_path,
        gpu=args.gpu,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    ) as served:
        print(f"UP in {time.time() - t0:.0f}s\n")

        # /health answering is not the same as being able to serve. A one-token
        # completion is: it exercises the tokenizer, the scheduler, the KV
        # allocator and the sampler. Without it "UP" only means a socket is
        # listening, and the first real failure would surface hundreds of
        # rollouts into a pass@k sweep instead of here.
        ok, detail = _smoke(served.api_base, served.model_name)
        print(f"  smoke completion  {'OK' if ok else 'FAILED'}  {detail}")
        if not ok:
            print("\nrefusing to report an endpoint that cannot complete. "
                  "Tearing down.", file=sys.stderr)
            return 3

        print(f"  STUDENT_API_BASE={served.api_base}")
        print(f"  model_name       {served.model_name}")
        print(f"  harbor model     {served.harbor_model}\n")
        print("harbor kwargs:")
        print(json.dumps(served_to_harbor_kwargs(served), indent=2))
        if args.write_env:
            with open(args.write_env, "a") as fh:
                fh.write(f"\nSTUDENT_API_BASE={served.api_base}\n")
            print(f"\nappended STUDENT_API_BASE to {args.write_env}")
        if args.max_hours:
            print(f"auto-shutdown in {args.max_hours}h "
                  f"(--max-hours 0 to disable)")

        sampler = None
        if args.gpu_log:
            sampler = _GpuSampler(served, Path(args.gpu_log), args.gpu_log_interval)
            sampler.start()
            print(f"gpu telemetry     {args.gpu_log} "
                  f"(every {args.gpu_log_interval}s)")

        print("\nendpoint is live. Ctrl-C to tear it down.\n", flush=True)
        while not stop:
            if deadline and time.time() > deadline:
                print(f"\nmax-hours ({args.max_hours}h) reached — shutting down.")
                break
            time.sleep(2)
        if sampler is not None:
            sampler.stop()
            print(f"gpu samples written: {sampler.n_samples}")
    # serve_model's __exit__ has now run. Belt and braces: an ephemeral Modal app
    # that outlives this process is a GPU nobody is using, so say plainly how to
    # check rather than assuming the teardown worked.
    print("torn down. verify with:  uv run modal app list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
