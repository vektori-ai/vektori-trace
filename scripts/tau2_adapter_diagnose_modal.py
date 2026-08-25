"""Diagnose the saved probe adapter. One model instance, short input, no training.

The probe's in-process reload check reported 98.4% argmax agreement with the
trained model and 98.1% with the *base* -- the reloaded adapter was barely more
similar to what it came from than to what it started as. That is a real signal,
not bf16 noise, and it does not need another training run to investigate: the
adapter is already on the volume.

What was wrong with the original check, and is fixed here:
  * it ran inside the training process, so it never tested a fresh load;
  * it held three 4B models on one GPU at once;
  * it materialised full-sequence x 152K-vocab logits three times;
  * it compared every position, most of which are masked prompt tokens that
    training never touched.

This loads ONE base model, inspects the adapter tensors directly, and compares
adapter-enabled against adapter-disabled on the same instance at a single
position.

    modal run scripts/tau2_adapter_diagnose_modal.py
"""
from __future__ import annotations

import modal

VOLUME_NAME = "vektori-trace-adapters"
VOLUME_MOUNT = "/adapters"
HF_CACHE_VOLUME_NAME = "hf-model-cache"
HF_CACHE_MOUNT = "/root/.cache/huggingface"
ADAPTER_IN_VOLUME = "tau2/a_warm_probe"
CORPUS_IN_VOLUME = "corpora/b741bfceb1f3d027"

app = modal.App("tau2-adapter-diagnose")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0", "transformers==5.5.3", "trl==1.10.0",
        "peft==0.19.1", "accelerate==1.14.0", "datasets==5.0.0",
        "bitsandbytes==0.50.1", "safetensors",
    )
    .env({"HF_HOME": HF_CACHE_MOUNT,
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)


@app.function(gpu="L40S", image=image,
              volumes={VOLUME_MOUNT: vol, HF_CACHE_MOUNT: hf_cache},
              timeout=25 * 60, max_containers=1)
def diagnose() -> dict:
    import json
    import os

    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adir = os.path.join(VOLUME_MOUNT, ADAPTER_IN_VOLUME)
    out: dict = {"adapter_dir": ADAPTER_IN_VOLUME,
                 "files": sorted(os.listdir(adir))}
    print(f"adapter files: {out['files']}", flush=True)

    # ---- 1. the tensors themselves -------------------------------------
    sd = load_file(os.path.join(adir, "adapter_model.safetensors"))
    cfg = json.load(open(os.path.join(adir, "adapter_config.json")))
    a_norms = {k: float(v.float().norm()) for k, v in sd.items() if "lora_A" in k}
    b_norms = {k: float(v.float().norm()) for k, v in sd.items() if "lora_B" in k}
    nonzero_b = sum(1 for v in b_norms.values() if v > 0)
    print(f"\nadapter_config: r={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
          f"targets={cfg.get('target_modules')}", flush=True)
    print(f"tensors: {len(sd)}  lora_A={len(a_norms)}  lora_B={len(b_norms)}",
          flush=True)
    print(f"lora_B with nonzero norm: {nonzero_b}/{len(b_norms)}  "
          f"(zero B means the adapter is a no-op)", flush=True)
    print(f"  |A| median {sorted(a_norms.values())[len(a_norms)//2]:.4f}  "
          f"|B| median {sorted(b_norms.values())[len(b_norms)//2]:.6f}", flush=True)
    out["adapter_config"] = {k: cfg.get(k) for k in
                             ("r", "lora_alpha", "lora_dropout", "target_modules")}
    out["n_tensors"] = len(sd)
    out["lora_B_nonzero"] = f"{nonzero_b}/{len(b_norms)}"
    out["lora_B_norm_median"] = sorted(b_norms.values())[len(b_norms) // 2]

    # ---- 2. one model, one short input ---------------------------------
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adir)
    model.eval()
    print(f"\nloaded one instance; VRAM "
          f"{torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

    ids = tok("The customer wants to cancel order #W123. First I should",
              return_tensors="pt").input_ids.to(0)

    def last_logits() -> torch.Tensor:
        with torch.no_grad():
            return model(input_ids=ids).logits[0, -1].float().cpu()

    # determinism of the *same* model, same input, twice
    l1, l2 = last_logits(), last_logits()
    self_delta = float((l1 - l2).abs().max())
    print(f"same model twice: max logit delta {self_delta:.6f} "
          f"({'deterministic' if self_delta == 0 else 'NONDETERMINISTIC'})",
          flush=True)
    out["self_consistency_delta"] = self_delta

    with_adapter = l1
    with model.disable_adapter():
        without = last_logits()

    d = float((with_adapter - without).abs().max())
    p_with = torch.softmax(with_adapter, -1)
    p_without = torch.softmax(without, -1)
    dp = float((p_with - p_without).abs().max())
    same_top = int(with_adapter.argmax()) == int(without.argmax())
    tw = torch.topk(p_with, 5)
    to = torch.topk(p_without, 5)
    print(f"\nadapter ON  top5: "
          f"{[(tok.decode([i]), round(float(p), 4)) for p, i in zip(tw.values, tw.indices)]}",
          flush=True)
    print(f"adapter OFF top5: "
          f"{[(tok.decode([i]), round(float(p), 4)) for p, i in zip(to.values, to.indices)]}",
          flush=True)
    print(f"\nenabling the adapter changes the last-token logits by {d:.4f} "
          f"(max prob delta {dp:.6f}); top-1 {'unchanged' if same_top else 'CHANGED'}",
          flush=True)

    out.update({
        "adapter_effect_logit_delta": round(d, 6),
        "adapter_effect_prob_delta": round(dp, 8),
        "top1_changed_by_adapter": not same_top,
        "top5_with": [[tok.decode([int(i)]), round(float(p), 5)]
                      for p, i in zip(tw.values, tw.indices)],
        "top5_without": [[tok.decode([int(i)]), round(float(p), 5)]
                         for p, i in zip(to.values, to.indices)],
    })

    if d == 0.0:
        out["verdict"] = "ADAPTER IS A NO-OP — enabling it changes nothing"
    elif nonzero_b == 0:
        out["verdict"] = "every lora_B is zero — the adapter cannot affect output"
    else:
        out["verdict"] = ("adapter loads, is applied, and measurably changes the "
                          "output")
    print(f"\nVERDICT: {out['verdict']}", flush=True)
    return out


@app.local_entrypoint()
def main():
    import json
    out = diagnose.remote()
    print("\n" + "=" * 68)
    print(f"adapter: {out['adapter_dir']}")
    print(f"config : {out['adapter_config']}")
    print(f"tensors: {out['n_tensors']}, lora_B nonzero {out['lora_B_nonzero']}, "
          f"median |B| {out['lora_B_norm_median']:.6f}")
    print(f"self-consistency (same model twice): {out['self_consistency_delta']}")
    print(f"adapter effect: logit delta {out['adapter_effect_logit_delta']}, "
          f"prob delta {out['adapter_effect_prob_delta']}")
    print(f"VERDICT: {out['verdict']}")
    print("=" * 68)
    json.dump(out, open("/tmp/adapter_diagnosis.json", "w"), indent=1)
