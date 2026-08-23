#!/usr/bin/env python3
"""Smoke-test a served model on four tau2-bench retail tasks.

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
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vektori_trace.runtime.serve import serve_model  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from run_replay_opd import require_endpoint_model  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"

# Qwen3-14B is the model the rest of this repo is built around, so the
# interesting comparison is 14B vs 27B on identical tasks: frontier pass rates
# are a ceiling measure and do not predict where an open model fails. 14B is a
# plain dense Qwen3 -- no vision tower, so it skips the ~3.5 min multimodal
# warmup, and the qwen3_coder tool parser still applies.
VISION_MODELS = ("Qwen3.8-27B",)

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
]

# The tool-call parser must match what the model's chat template emits. Both
# Qwen3-14B and Qwen3.8-27B wrap calls in <tool_call> tags, but qwen3_coder --
# taken from the 27B vLLM recipe -- extracted nothing from 14B: 22 raw blocks
# left in message content, 0 parsed, so no tool ever executed and the model
# hallucinated order data to keep the conversation going. Per model, verified
# against the template, never carried across.
TOOL_PARSERS = {
    "Qwen3.8-27B": "qwen3_coder",
    "Qwen3-14B": "hermes",
}
DEFAULT_TOOL_PARSER = "hermes"


def _spend_so_far(results: Path) -> float | None:
    """Agent + user cost across every simulation written so far, or None.

    Partial reads are expected -- tau2 rewrites the file as it goes, so a poll
    can land mid-write and fail to parse. That is not an error; the next poll
    20s later sees a complete file.
    """
    try:
        d = json.loads(results.read_text())
    except Exception:
        return None
    return sum((s.get("agent_cost") or 0) + (s.get("user_cost") or 0)
               for s in d.get("simulations", []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gpu", default="A100-80GB")
    ap.add_argument("--tool-parser", default=None,
                    help="override; default is per-model in TOOL_PARSERS")
    ap.add_argument("--tasks", nargs="*", default=TASKS,
                    help="empty (--tasks) runs every task in the domain")
    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=32768)
    # The date suffix is part of the slug: bare `deepseek-v4-flash` 404s with
    # "Model not found, inaccessible, and/or not deployed". Confirm any change
    # against `GET /v1/models` rather than the docs.
    ap.add_argument("--user-llm",
                    default="fireworks_ai/accounts/fireworks/models/deepseek-v4-flash-0731")
    # Unique per run. A fixed name collides with the previous run's results
    # file, and tau2 then asks "resume the run? (y/n)" on stdin -- which,
    # detached, nobody answers, so the endpoint idles on a paid GPU until
    # something kills it. Keeping runs in separate files also means every run
    # stays on disk to compare against.
    ap.add_argument("--save-to", default=None,
                    help="default: <model>_smoke_<timestamp>")
    ap.add_argument("--tau2-dir", default="/data/tau2")
    ap.add_argument("--max-cost", type=float, default=10.0,
                    help="stop the sweep once agent+user spend exceeds this (USD)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the tau2 command and exit without allocating a GPU")
    a = ap.parse_args()

    short = a.model.split("/")[-1]
    if a.save_to is None:
        a.save_to = f"{short.replace('.', '_').replace('-', '_').lower()}" \
                    f"_smoke_{time.strftime('%Y%m%d_%H%M%S')}"

    vllm_args = list(VLLM_ARGS)
    vllm_args += ["--tool-call-parser",
                  a.tool_parser or TOOL_PARSERS.get(short, DEFAULT_TOOL_PARSER)]
    if short in VISION_MODELS:
        # Text-only benchmark: tell vLLM to expect no images so it skips the
        # multi-modal warmup, ~3.5 min of every 27B boot. Harmless to omit on a
        # text model, which has no vision tower to profile.
        vllm_args += ["--limit-mm-per-prompt", '{"image": 0}']

    tau2_bin = shutil.which("tau2") or str(Path(sys.executable).parent / "tau2")
    if not Path(tau2_bin).exists():
        sys.exit(f"tau2 entry point not found at {tau2_bin}; "
                 f"uv pip install --python {sys.executable} -e {a.tau2_dir}")

    if not a.dry_run and not os.environ.get("FIREWORKS_API_KEY"):
        # tau2 also reads <tau2-dir>/.env, so a missing env var is not fatal --
        # but failing here beats discovering it after the model has loaded.
        if not (Path(a.tau2_dir) / ".env").exists():
            sys.exit("FIREWORKS_API_KEY unset and no .env; the user simulator cannot run")

    # `agent_llm` is a parameter, never `f"hosted_vllm/{MODEL}"`. serve_model
    # registers vLLM under `_canonical_name(base_model)`, which strips the org
    # prefix: "Qwen/Qwen3.8-27B" is served as "Qwen3.8-27B", and asking for the
    # full HF path 404s every call. A LoRA registers a different name again, so
    # the only correct source is the live `ServedModel`.
    def tau2_cmd(api_base: str | None, agent_llm: str) -> list[str]:
        args = ('{"temperature": 0.0}' if api_base is None
                else f'{{"api_base": "{api_base}", "temperature": 0.0}}')
        return [
            tau2_bin, "run",
            "--domain", a.domain,
            *(["--task-ids", *a.tasks] if a.tasks else []),
            "--num-trials", str(a.num_trials),
            "--agent-llm", agent_llm,
            "--agent-llm-args", args,
            "--user-llm", a.user_llm,
            "--user-llm-args", '{"temperature": 0.0}',
            "--max-concurrency", str(a.max_concurrency),
            "--save-to", a.save_to,
            # tau2 v0.2.0 defaults to ERROR, which hides the request/response
            # detail that tells an empty tool_calls apart from a routing 404.
            # (`--verbose-logs` / `--auto-resume` are tau3 flags; not here.)
            "--log-level", "DEBUG",
        ]

    # A hosted agent (fireworks_ai/..., openai/...) is reached by slug through
    # litellm; there is nothing to serve, so skip serve_model entirely rather
    # than allocating a GPU that would sit idle for the whole sweep.
    hosted = "/" in a.model and a.model.split("/")[0] in {
        "fireworks_ai", "openai", "anthropic", "together_ai", "deepseek"}

    if hosted:
        cmd = tau2_cmd(None, a.model)
        print(f"[hosted] agent={a.model} (no GPU)", flush=True)
        print(f"[tau2  ] {' '.join(cmd)}", flush=True)
        if a.dry_run:
            return 0
        t0 = time.time()
        # Watchdog: tau2 writes every finished simulation into save_to as it
        # goes, and each carries its own agent_cost/user_cost. Poll that file
        # and kill the sweep the moment spend crosses --max-cost, so a pricing
        # surprise or a runaway retry loop cannot quietly bill all night.
        # Killing is safe: rerunning with the same --save-to resumes.
        proc = subprocess.Popen(cmd, cwd=a.tau2_dir, stdin=subprocess.DEVNULL)
        results = Path(a.tau2_dir) / "data" / "simulations" / f"{a.save_to}.json"
        while proc.poll() is None:
            time.sleep(20)
            spent = _spend_so_far(results)
            if spent is None:
                continue
            if spent > a.max_cost:
                print(f"[budget] ${spent:.2f} > --max-cost ${a.max_cost:.2f}; "
                      f"stopping. Rerun with --save-to {a.save_to} to resume.",
                      flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
        rc = proc.returncode if proc.returncode is not None else 1
        print(f"[budget] spent ~${_spend_so_far(results) or 0:.2f}", flush=True)
        print(f"[tau2  ] exit {rc} after {time.time()-t0:.0f}s", flush=True)
        print(f"[done  ] results in {a.tau2_dir}/data/simulations/{a.save_to}.json")
        return rc

    if a.dry_run:
        print("would serve:", a.model, "on", a.gpu, "with", " ".join(vllm_args))
        # Deliberately a placeholder: the real value is only known once the
        # endpoint is up. Printing a guess here is what let a wrong model name
        # survive a dry-run and 404 four minutes into a paid run.
        print("would run:  ", " ".join(
            tau2_cmd("<api_base>", "hosted_vllm/<served.model_name @ boot>")))
        return 0

    print(f"[serve] {a.model} on {a.gpu} (first boot pulls the weights into the HF cache "
          f"volume; subsequent runs reuse it)", flush=True)
    t0 = time.time()
    with serve_model(
        a.model,
        gpu=a.gpu,
        max_model_len=a.max_model_len,
        gpu_memory_utilization=0.90,
        extra_vllm_args=vllm_args,
    ) as served:
        print(f"[serve] up in {time.time()-t0:.0f}s at {served.api_base}", flush=True)
        # Ask the server what it advertises rather than trusting the name we
        # think we passed it. Wrong name = 404 on every rollout, discovered
        # minutes in with the GPU already billing; this fails in seconds.
        require_endpoint_model(served.api_base, served.model_name)
        print(f"[serve] endpoint advertises {served.model_name!r}", flush=True)
        cmd = tau2_cmd(served.api_base, served.harbor_model)
        print(f"[tau2 ] {' '.join(cmd)}", flush=True)
        # Inherit stdout/stderr so `tail -f` on the log shows tau2's own
        # progress lines as they happen rather than in one dump at the end.
        # stdin is /dev/null on purpose: any prompt tau2 decides to ask reads
        # EOF and the process exits, instead of blocking forever on input that
        # will never arrive while the GPU bills by the second.
        rc = subprocess.run(cmd, cwd=a.tau2_dir,
                            stdin=subprocess.DEVNULL).returncode
        print(f"[tau2 ] exit {rc} after {time.time()-t0:.0f}s total", flush=True)

    print(f"[serve] torn down; results in {a.tau2_dir}/data/simulations/{a.save_to}.json")
    return rc


if __name__ == "__main__":
    sys.exit(main())
