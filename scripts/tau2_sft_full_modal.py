"""Full W30 warm-start SFT on one L40S: 273 rows, 3 epochs, ~102 steps -> A_warm.

Separate from the probe wrapper on purpose. The probe passes `--probe`, caps at
three steps and writes to `tau2/a_warm_probe`; reusing it with a flag flipped is
how a "full run" quietly trains for 53 seconds.

The output directory is timestamped and must not already exist: `RunLog` opens
its step log in append mode so a crash cannot truncate it, which makes reusing a
directory dangerous -- the gradient gate reads `train_steps.jsonl` back, and a
probe's three steps sitting in front of a full run's hundred would be judged as
one series.

Measured on the probe: 17.6 s/step, peak 24.3 GiB of 48. 102 steps is ~30 min.

    modal run scripts/tau2_sft_full_modal.py
    modal run scripts/tau2_sft_full_modal.py --epochs 1     # shorter arm
"""
from __future__ import annotations

import modal

VOLUME_NAME = "vektori-trace-adapters"
VOLUME_MOUNT = "/adapters"
HF_CACHE_VOLUME_NAME = "hf-model-cache"
HF_CACHE_MOUNT = "/root/.cache/huggingface"

CORPUS_IN_VOLUME = "corpora/b741bfceb1f3d027"
RUNS_IN_VOLUME = "tau2/runs"

#: Refuse to train against a corpus that is not the frozen one.
EXPECT_MANIFEST = "b741bfceb1f3d027"

app = modal.App("tau2-sft-full")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

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
    # 102 steps at the measured 17.6 s is ~30 min; 3x that is room for a slow
    # start without letting a genuinely hung run bill all night.
    timeout=100 * 60,
    max_containers=1,
)
def train(run_id: str, epochs: float = 3.0, lr: float = 1e-4) -> dict:
    import json
    import os
    import sys

    art = os.path.join(VOLUME_MOUNT, CORPUS_IN_VOLUME)
    out_dir = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME, run_id)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise SystemExit(f"output directory {out_dir} already exists and is "
                         "not empty; refusing to interleave two runs")
    os.makedirs(out_dir, exist_ok=True)

    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB",
          flush=True)
    print(f"corpus: {art}", flush=True)
    print(f"output: {out_dir}", flush=True)

    sys.argv = [
        "tau2_sft_train.py",
        "--artifacts", art,
        "--partition", "W30",
        "--manifest-hash", EXPECT_MANIFEST,
        "--out", out_dir,
        "--epochs", str(epochs),
        "--lr", str(lr),
    ]
    sys.path.insert(0, "/root")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tau2_sft_train", "/root/scripts/tau2_sft_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Persist each epoch checkpoint as it is written, not only at the end.
    _orig_parse = mod.main

    rc = 1
    try:
        import argparse
        real_parse = argparse.ArgumentParser.parse_args

        def patched(self, *a, **kw):
            ns = real_parse(self, *a, **kw)
            ns.commit_fn = vol.commit
            return ns

        argparse.ArgumentParser.parse_args = patched
        try:
            rc = mod.main()
        finally:
            argparse.ArgumentParser.parse_args = real_parse
    finally:
        vol.commit()

    result = {"returncode": rc, "run_id": run_id,
              "out_dir": os.path.join(RUNS_IN_VOLUME, run_id)}
    for fn in ("run_summary.json", "run_config.json", "failure.json"):
        p = os.path.join(out_dir, fn)
        if os.path.exists(p):
            result[fn] = json.load(open(p))
    for fn, key in (("train_steps.jsonl", "steps"),
                    ("checkpoints.jsonl", "checkpoints")):
        p = os.path.join(out_dir, fn)
        if os.path.exists(p):
            result[key] = [json.loads(l) for l in open(p)]
    vol.commit()
    return result


@app.local_entrypoint()
def main(epochs: float = 3.0, lr: float = 1e-4, run_id: str = ""):
    import json
    import time

    rid = run_id or f"a_warm_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"run_id: {rid}")
    print(f"output: volume:{RUNS_IN_VOLUME}/{rid}")
    print(f"watch : modal app logs tau2-sft-full -f")
    print()

    res = train.remote(rid, epochs=epochs, lr=lr)

    print("\n" + "=" * 70)
    print(f"FULL W30 RUN {res['run_id']}  returncode {res['returncode']}")
    steps = res.get("steps", [])
    if steps:
        print(f"  {len(steps)} steps | first loss {steps[0]['loss']:.4f} -> "
              f"last {steps[-1]['loss']:.4f}")
        norms = [s["grad_norm"] for s in steps if s.get("grad_norm") is not None]
        if norms:
            print(f"  grad_norm min {min(norms):.4f} max {max(norms):.4f}")
    for c in res.get("checkpoints", []):
        print(f"  checkpoint step {c['optimizer_step']}: "
              f"{len(c.get('files', {}))} files"
              f"{' [committed]' if c.get('committed') else ''}")
    summ = res.get("run_summary.json")
    if summ:
        print(f"  peak VRAM {summ.get('peak_vram_gib')} GiB | "
              f"{summ.get('elapsed_sec')}s | train_loss {summ.get('train_loss')}")
        rv = summ.get("reload_verification") or {}
        if rv:
            print(f"  adapter: {rv.get('lora_B_nonzero')} lora_B non-zero, "
                  f"effect {rv.get('adapter_effect_logit_delta')}")
    if res.get("failure.json"):
        f = res["failure.json"]
        print(f"  FAILURE {f.get('error')}: {f.get('message', '')[:200]}")
    print("=" * 70)
    json.dump(res, open(f"/tmp/{rid}_result.json", "w"), indent=1)
    print(f"wrote /tmp/{rid}_result.json")
