#!/usr/bin/env python3
"""Smoke-test Qwen3.8-27B on four tau2-bench retail tasks.

Capability probe, not a measurement: 1 rollout per task cannot produce a pass
rate. tau2's user simulator is an LLM, so the same task varies run to run --
retail #57 went 2/4 for claude-3.7 and 3/4 for gpt-4.1 in the published results.
What one rollout *can* tell you is whether the plumbing works: does the model
emit well-formed tool calls, does it orient in the domain, does it terminate.
If that looks clean, re-run with --num-trials 4 for a number worth citing.

Serving and running live in one process on purpose. `serve_model` is a context
manager -- the Modal endpoint exists only inside the `with` block, so tau2 has
to run from inside it. The upside is teardown: any exit path, including a
crash or Ctrl-C, releases the GPU. Started detached and forgotten, a separate
endpoint process would bill until SCALEDOWN_WINDOW_SECONDS (10 min) expired.

Memory (A100-80GB, bf16): 51.7 GiB weights + ~4 GiB KV + ~3 GiB CUDA/activations
= ~59 GiB. KV is cheap here because only 16 of 64 layers keep a cache -- the
other 48 are Gated DeltaNet with constant recurrent state -- so context length
costs almost nothing and the card is sized by weights alone.

    export FIREWORKS_API_KEY=...          # or /data/tau2/.env
    /data/vektori-trace/.venv/bin/python scripts/run_tau2_smoke.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vektori_trace.runtime.serve import serve_model  # noqa: E402

MODEL = "Qwen/Qwen3.8-27B"

# tau2 v0.2.0 retail, ranked by measured frontier pass rate (12 trials each:
# claude-3.7 / gpt-4.1 / o4-mini x 4). Retail only, so one policy document and
# one tool schema hold constant and difficulty is the only variable.
TASKS = [
    "73",   # 12/12 easy   -- return all but the coffee machine
    "75",   # 12/12 easy   -- exchange earbuds, all args stated
    "57",   #  9/12 medium -- 3-deep conditional that collapses; correct answer is do nothing
    "93",   #  9/12 medium -- exchange laptop; user gives zip, not email
]

# Without --enforce-eager, startup dies in CUDA graph capture with
# torch.OutOfMemoryError, and neither raising nor lowering
# --gpu-memory-utilization helps (vLLM recipe for this model).
# The tool-call parser is load-bearing: tau2 grades structured tool calls, and
# without it the model's calls arrive as prose and every task scores 0 for a
# reason that has nothing to do with capability.
VLLM_ARGS = [
    "--enforce-eager",
    "--reasoning-parser", "qwen3",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="A100-80GB")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=32768)
    # The date suffix is part of the slug: bare `deepseek-v4-flash` 404s with
    # "Model not found, inaccessible, and/or not deployed". Confirm any change
    # against `GET /v1/models` rather than the docs.
    ap.add_argument("--user-llm",
                    default="fireworks_ai/accounts/fireworks/models/deepseek-v4-flash-0731")
    ap.add_argument("--save-to", default="qwen38_27b_smoke")
    ap.add_argument("--tau2-dir", default="/data/tau2")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the tau2 command and exit without allocating a GPU")
    a = ap.parse_args()

    tau2_bin = shutil.which("tau2") or str(Path(sys.executable).parent / "tau2")
    if not Path(tau2_bin).exists():
        sys.exit(f"tau2 entry point not found at {tau2_bin}; "
                 f"uv pip install --python {sys.executable} -e {a.tau2_dir}")

    if not a.dry_run and not os.environ.get("FIREWORKS_API_KEY"):
        # tau2 also reads <tau2-dir>/.env, so a missing env var is not fatal --
        # but failing here beats discovering it after the model has loaded.
        if not (Path(a.tau2_dir) / ".env").exists():
            sys.exit("FIREWORKS_API_KEY unset and no .env; the user simulator cannot run")

    def tau2_cmd(api_base: str) -> list[str]:
        return [
            tau2_bin, "run",
            "--domain", a.domain,
            "--task-ids", *a.tasks,
            "--num-trials", str(a.num_trials),
            "--agent-llm", f"hosted_vllm/{MODEL}",
            "--agent-llm-args", f'{{"api_base": "{api_base}", "temperature": 0.0}}',
            "--user-llm", a.user_llm,
            "--user-llm-args", '{"temperature": 0.0}',
            "--max-concurrency", str(a.max_concurrency),
            "--save-to", a.save_to,
            "--log-level", "INFO",
        ]

    if a.dry_run:
        print("would serve:", MODEL, "on", a.gpu, "with", " ".join(VLLM_ARGS))
        print("would run:  ", " ".join(tau2_cmd("<api_base>")))
        return 0

    print(f"[serve] {MODEL} on {a.gpu} (first boot pulls ~52 GiB into the HF cache "
          f"volume; subsequent runs reuse it)", flush=True)
    t0 = time.time()
    with serve_model(
        MODEL,
        gpu=a.gpu,
        max_model_len=a.max_model_len,
        gpu_memory_utilization=0.90,
        extra_vllm_args=VLLM_ARGS,
    ) as served:
        print(f"[serve] up in {time.time()-t0:.0f}s at {served.api_base}", flush=True)
        cmd = tau2_cmd(served.api_base)
        print(f"[tau2 ] {' '.join(cmd)}", flush=True)
        # Inherit stdout/stderr so `tail -f` on the log shows tau2's own
        # progress lines as they happen rather than in one dump at the end.
        rc = subprocess.run(cmd, cwd=a.tau2_dir).returncode
        print(f"[tau2 ] exit {rc} after {time.time()-t0:.0f}s total", flush=True)

    print(f"[serve] torn down; results in {a.tau2_dir}/data/simulations/{a.save_to}.json")
    return rc


if __name__ == "__main__":
    sys.exit(main())
