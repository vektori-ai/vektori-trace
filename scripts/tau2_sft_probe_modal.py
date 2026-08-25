"""Three-step L40S probe of the Tau2 W30 trainer, plus adapter reload proof.

Mechanical validation only -- does the configuration fit, does it step, does the
adapter survive a round trip. It is not a quality signal and cannot compare
learning rates: three steps is not an experiment.

L40S (48 GB) rather than A10G (24 GB): the longest row is 13,344 tokens and the
logits tensor alone is 13,344 x 151,936, which is ~4 GB in bf16 before the loss
upcast. `chunked_nll` exists to avoid materialising it whole, but the margin on
a 24 GB card is thin enough that an OOM would cost more than the price gap. A100
is unnecessary.

    modal run scripts/tau2_sft_probe_modal.py

Everything the run produces is written to the adapters volume as it happens, so
an OOM at step 2 still leaves `run_config.json` and whatever steps completed.
"""
from __future__ import annotations

import modal

VOLUME_NAME = "vektori-trace-adapters"
VOLUME_MOUNT = "/adapters"
HF_CACHE_VOLUME_NAME = "hf-model-cache"
HF_CACHE_MOUNT = "/root/.cache/huggingface"

CORPUS_IN_VOLUME = "corpora/b741bfceb1f3d027"
OUT_IN_VOLUME = "tau2/a_warm_probe"

app = modal.App("tau2-sft-probe")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Exactly Stage A's proven set (`sft_stage_a_train_modal.py`). Guessing at
    # these pins cost two failed image builds: trl 0.29 has no `chunked_nll`,
    # and trl 1.10 requires datasets>=4.7.
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
        # A fragmentation mitigation, not extra memory: lets a segment grow
        # rather than stranding freed blocks at the wrong size.
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
    timeout=45 * 60,
    max_containers=1,
)
def probe() -> dict:
    import json
    import os
    import sys

    # The corpus is the immutable staged artifact on the volume; the trainer
    # verifies its hashes and derives W30 in memory.
    art = os.path.join(VOLUME_MOUNT, CORPUS_IN_VOLUME)
    print(f"corpus: {art} -> {sorted(os.listdir(art))}", flush=True)

    out_dir = os.path.join(VOLUME_MOUNT, OUT_IN_VOLUME)
    os.makedirs(out_dir, exist_ok=True)

    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB",
          flush=True)

    sys.argv = [
        "tau2_sft_train.py", "--probe",
        "--artifacts", art,
        "--partition", "W30",
        "--out", out_dir,
    ]
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tau2_sft_train", "/root/scripts/tau2_sft_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rc = 1
    try:
        rc = mod.main()
    finally:
        vol.commit()          # persist logs even if the step raised

    result = {"returncode": rc, "out_dir": OUT_IN_VOLUME}
    for fn in ("run_summary.json", "run_config.json", "failure.json"):
        p = os.path.join(out_dir, fn)
        if os.path.exists(p):
            result[fn] = json.load(open(p))
    steps_p = os.path.join(out_dir, "train_steps.jsonl")
    if os.path.exists(steps_p):
        result["steps"] = [json.loads(l) for l in open(steps_p)]
    vol.commit()
    return result


@app.local_entrypoint()
def main():
    import json

    res = probe.remote()
    print("\n" + "=" * 68)
    print(f"PROBE returncode {res['returncode']}   volume: {res['out_dir']}")
    for s in res.get("steps", []):
        print(f"  step {s['optimizer_step']}  loss {s['loss']:.4f}  "
              f"gnorm {s.get('grad_norm')}  vram {s.get('allocated_vram_gib')} GiB")
    summ = res.get("run_summary.json")
    if summ:
        print(f"  peak VRAM {summ.get('peak_vram_gib')} GiB | "
              f"{summ.get('elapsed_sec')}s | loss {summ.get('train_loss')}")
        rv = summ.get("reload_verification") or {}
        if rv:
            print(f"  RELOAD: differs from base by "
                  f"{rv.get('max_abs_logit_delta_vs_base')}, matches trained "
                  f"within {rv.get('max_abs_logit_delta_vs_trained')}")
            print(f"          {rv.get('verdict')}")
    if res.get("failure.json"):
        print(f"  FAILURE: {res['failure.json'].get('error')}: "
              f"{res['failure.json'].get('message', '')[:200]}")
    print("=" * 68)
    json.dump(res, open("/tmp/probe_result.json", "w"), indent=1)
    print("wrote /tmp/probe_result.json")
