#!/usr/bin/env python3
"""Continue ordinary SFT from CK35 on the frozen C30 corpus.

This is the next SFT stage, not replay OPD and not the diversity canary. It
loads CK35's LoRA weights, trains them for one C30 epoch (~37 optimizer steps),
and creates a new adapter with fresh optimizer and scheduler state.

Nothing allocates a GPU without ``--yes``.

    .venv/bin/modal run scripts/tau2_sft_continue_modal.py --yes
"""

from __future__ import annotations

import modal

VOLUME_NAME = "vektori-trace-adapters"
VOLUME_MOUNT = "/adapters"
HF_CACHE_VOLUME_NAME = "hf-model-cache"
HF_CACHE_MOUNT = "/root/.cache/huggingface"

CORPUS_IN_VOLUME = "corpora/b741bfceb1f3d027"
PARENT_IN_VOLUME = "tau2/runs/a_warm_20260825_003343/checkpoint-35"
RUNS_IN_VOLUME = "tau2/runs"
EXPECT_MANIFEST = "b741bfceb1f3d027"

app = modal.App("tau2-sft-c30-continue")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.5.3",
        "trl==1.10.0",
        "peft==0.19.1",
        "accelerate==1.14.0",
        "datasets==5.0.0",
        "bitsandbytes==0.50.1",
    )
    .env({
        "HF_HOME": HF_CACHE_MOUNT,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_dir("vektori_trace", remote_path="/root/vektori_trace",
                   ignore=["__pycache__"])
    .add_local_file("scripts/tau2_sft_train.py",
                    remote_path="/root/scripts/tau2_sft_train.py")
)


@app.function(
    gpu="L40S",
    image=image,
    volumes={VOLUME_MOUNT: vol, HF_CACHE_MOUNT: hf_cache},
    timeout=60 * 60,
    max_containers=1,
)
def train(run_id: str, epochs: float, lr: float) -> dict:
    import importlib.util
    import json
    import os
    import sys

    artifacts = os.path.join(VOLUME_MOUNT, CORPUS_IN_VOLUME)
    parent = os.path.join(VOLUME_MOUNT, PARENT_IN_VOLUME)
    out = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME, run_id)

    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not os.path.isfile(os.path.join(parent, name)):
            raise SystemExit(f"CK35 parent is incomplete: missing {parent}/{name}")
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"output {out} is not empty; refusing to mix runs")
    os.makedirs(out, exist_ok=True)

    sys.argv = [
        "tau2_sft_train.py",
        "--artifacts", artifacts,
        "--partition", "C30",
        "--manifest-hash", EXPECT_MANIFEST,
        "--model", "Qwen/Qwen3-4B",
        "--init-adapter", parent,
        "--out", out,
        "--epochs", str(epochs),
        "--lr", str(lr),
    ]
    sys.path.insert(0, "/root")
    spec = importlib.util.spec_from_file_location(
        "tau2_sft_train", "/root/scripts/tau2_sft_train.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    real_parse = __import__("argparse").ArgumentParser.parse_args

    def patched(parser, *args, **kwargs):
        ns = real_parse(parser, *args, **kwargs)
        ns.commit_fn = vol.commit
        return ns

    __import__("argparse").ArgumentParser.parse_args = patched
    try:
        rc = mod.main()
    finally:
        __import__("argparse").ArgumentParser.parse_args = real_parse
        vol.commit()

    result = {"returncode": rc, "run_id": run_id,
              "out_dir": f"{RUNS_IN_VOLUME}/{run_id}"}
    for filename in ("run_config.json", "run_summary.json", "failure.json"):
        path = os.path.join(out, filename)
        if os.path.exists(path):
            result[filename] = json.load(open(path))
    return result


@app.local_entrypoint()
def main(yes: bool = False, epochs: float = 1.0, lr: float = 1e-4,
         run_id: str = ""):
    import json
    import time

    if not yes:
        raise SystemExit(
            "GPU run refused: pass --yes after reviewing the C30 continued-SFT plan"
        )
    if epochs <= 0:
        raise SystemExit("--epochs must be positive")

    rid = run_id or f"a_sft_c30_ck35_{time.strftime('%Y%m%d_%H%M%S')}"
    print("continued SFT: CK35 -> C30")
    print(f"epochs: {epochs} | lr: {lr} | output: {RUNS_IN_VOLUME}/{rid}")
    result = train.remote(rid, epochs, lr)
    print(json.dumps(result, indent=2))
    if result["returncode"] != 0 or result.get("failure.json"):
        raise SystemExit("continued SFT failed; inspect the persisted run artifacts")
