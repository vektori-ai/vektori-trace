#!/usr/bin/env python3
"""Run the ReOPD driver on a Modal GPU.

The driver's sampling and scoring stages are HTTP calls that could run anywhere,
but the optimizer step needs the model resident on a GPU. The EC2 box has none,
so the whole driver runs here and reaches back out to the separately-served
student endpoint.

Two GPUs are alive during a run, doing different jobs:

    L40S  serving ck35 under vLLM           (scripts/serve_student.py)
    A100  running this: the optimizer step  (this file)

They cannot be the same card. vLLM claims its GPU's memory entirely, so
training beside it is not possible.

    # 1. endpoint, in its own tmux window
    .venv/bin/python scripts/serve_student.py --base-model Qwen/Qwen3-4B \\
        --adapter ck35=/adapters/tau2/runs/a_warm_20260825_003343/checkpoint-35 \\
        --gpu L40S --max-model-len 20480 --write-env /data/tau2/ck35_env.sh

    # 2. the canary, then the full run
    .venv/bin/modal run scripts/tau2_reopd_modal.py --canary 4 \\
        --api-base "$STUDENT_API_BASE"
    .venv/bin/modal run scripts/tau2_reopd_modal.py --yes \\
        --api-base "$STUDENT_API_BASE"

Nothing allocates a GPU without `--canary` or `--yes`.
"""

from __future__ import annotations

import modal

VOLUME_NAME = "vektori-trace-adapters"
VOLUME_MOUNT = "/adapters"
HF_CACHE_VOLUME_NAME = "hf-model-cache"
HF_CACHE_MOUNT = "/root/.cache/huggingface"

#: The frozen corpus, staged on the volume. The box's /data/tau2 is not visible
#: from a Modal container, so the artifacts must live where the container can
#: read them.
CORPUS_IN_VOLUME = "corpora/b741bfceb1f3d027"
PARENT_IN_VOLUME = "tau2/runs/a_warm_20260825_003343/checkpoint-35"
RUNS_IN_VOLUME = "tau2/reopd"

app = modal.App("tau2-reopd")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.5.3",
        "peft==0.19.1",
        "accelerate==1.14.0",
        "safetensors==0.7.1",
        "numpy<3",
        "requests",
    )
    .env({
        "HF_HOME": HF_CACHE_MOUNT,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_dir("vektori_trace", remote_path="/root/vektori_trace",
                   ignore=["__pycache__"])
    .add_local_file("scripts/tau2_reopd_train.py",
                    remote_path="/root/scripts/tau2_reopd_train.py")
)


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={VOLUME_MOUNT: vol, HF_CACHE_MOUNT: hf_cache},
    timeout=60 * 60 * 6,
    max_containers=1,
    secrets=[modal.Secret.from_name("fireworks-api-key")],
)
def train(
    api_base: str,
    model: str,
    run_id: str,
    canary: int,
    n_updates: int,
    n_per_update: int,
    learning_rate: float,
    temperature: float,
    policy_file: str | None,
) -> dict:
    import importlib.util
    import json
    import os
    import sys

    sys.path.insert(0, "/root")
    artifacts = os.path.join(VOLUME_MOUNT, CORPUS_IN_VOLUME)
    parent = os.path.join(VOLUME_MOUNT, PARENT_IN_VOLUME)
    out = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME, run_id)

    if not os.path.isdir(artifacts):
        raise SystemExit(
            f"corpus not staged at {artifacts}. Copy the frozen artifacts to "
            "the volume first -- a Modal container cannot see /data/tau2."
        )
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not os.path.isfile(os.path.join(parent, name)):
            raise SystemExit(f"CK35 parent incomplete: missing {parent}/{name}")

    argv = [
        "tau2_reopd_train.py",
        "--artifacts", artifacts,
        "--simulations-dir", os.path.join(artifacts, "simulations"),
        "--parent", parent,
        "--base-model", "Qwen/Qwen3-4B",
        "--api-base", api_base,
        "--model", model,
        "--run-dir", out,
        "--learning-rate", str(learning_rate),
        "--temperature", str(temperature),
        "--n-updates", str(n_updates),
        "--n-per-update", str(n_per_update),
    ]
    if policy_file:
        argv += ["--policy-file", policy_file]
    if canary:
        argv += ["--canary", str(canary)]
    else:
        argv += ["--yes"]
    sys.argv = argv

    spec = importlib.util.spec_from_file_location(
        "tau2_reopd_train", "/root/scripts/tau2_reopd_train.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    try:
        rc = mod.main()
    finally:
        # Commit whatever landed, including a partial run: the durable markers
        # are what a resume reads, and losing them means re-buying every score.
        vol.commit()

    result = {"returncode": rc, "run_id": run_id,
              "out_dir": f"{RUNS_IN_VOLUME}/{run_id}"}
    for fn in ("manifest.json", "schedule.json"):
        p = os.path.join(out, fn)
        if os.path.exists(p):
            result[fn] = json.load(open(p))
    tel = os.path.join(out, "telemetry.jsonl")
    if os.path.exists(tel):
        rows = [json.loads(l) for l in open(tel) if l.strip()]
        result["telemetry_rows"] = len(rows)
        result["stages"] = [r for r in rows if r.get("event") == "stage"]
    return result


@app.local_entrypoint()
def main(
    api_base: str = "",
    model: str = "ck35",
    run_id: str = "",
    canary: int = 0,
    yes: bool = False,
    n_updates: int = 32,
    n_per_update: int = 16,
    learning_rate: float = 1e-5,
    temperature: float = 1.0,
    policy_file: str = "",
):
    import json
    import time

    if not canary and not yes:
        raise SystemExit(
            "GPU run refused. Use --canary 4 for one small update, or --yes "
            "for the full 32-update arm after reading the canary's numbers."
        )
    if not api_base:
        raise SystemExit(
            "--api-base is required: start the student endpoint first with "
            "scripts/serve_student.py and pass its URL."
        )

    rid = run_id or (
        f"{'canary' if canary else 'a_reopd'}_ck35_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    shape = f"{1 if canary else n_updates} updates x {canary or n_per_update} states"
    print(f"ReOPD: CK35 -> C30 ({shape})")
    print(f"  endpoint {api_base}")
    print(f"  output   {RUNS_IN_VOLUME}/{rid}")

    result = train.remote(api_base, model, rid, canary, n_updates,
                          n_per_update, learning_rate, temperature,
                          policy_file or None)
    print(json.dumps({k: v for k, v in result.items()
                      if k != "stages"}, indent=2))
    for s in result.get("stages", []):
        print(f"  u{s.get('update')} {s.get('stage'):8s} "
              f"{s.get('seconds')}s {s.get('loss', '')}")
    if result["returncode"] != 0:
        raise SystemExit("ReOPD failed; inspect the persisted run artifacts")
