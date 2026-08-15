"""Run the TRL SFT on a single Modal GPU.

The dataset is staged on the shared adapter Volume rather than passed as a
function argument — 165 agent trajectories are ~16 MB of JSON, which is no
size to push through a gRPC call.

`--probe` loads the model, runs a few steps, reports real peak VRAM and
tokens/sec, and stops without saving. Do that before the full run: it turns the
ETA into a measurement and catches an OOM in ten minutes instead of four hours.

    modal run scripts/sft_train_modal.py --probe
    modal run scripts/sft_train_modal.py

Concurrency is pinned to one container. `docs/MODAL-CONCURRENCY.md` records the
2026-08-13 incident where an unbounded serve class quietly held three L40S at
once; a training run is a single invocation, but the cap makes that structural
rather than incidental.
"""

from __future__ import annotations

from pathlib import Path

import modal

# Mirrored from vektori_trace/runtime/modal_env.py rather than imported: Modal
# re-imports this module *inside* the container, where the local package is not
# installed, so a module-level import of it fails the run before any training
# starts. Four constants are cheaper than shipping the package.
VOLUME_NAME = "vektori-trace-adapters"
VOLUME_MOUNT = "/adapters"
HF_CACHE_VOLUME_NAME = "hf-model-cache"
HF_CACHE_MOUNT = "/root/.cache/huggingface"

GPU = "A100-80GB"
DATA_IN_VOLUME = "sft/sft_train.jsonl"
OUT_IN_VOLUME = "sft/qwen3-14b-dsv4-lora"

app = modal.App("vektori-trace-sft-trl")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

# Pinned to the versions the export and the mask check were verified against.
# Unpinned, the container resolved a different TRL whose SFTConfig rejects
# `warmup_ratio` — the run died on a kwarg after the image build and the model
# download were already paid for. Behaviour verified on one version and shipped
# on another is not verified at all.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.5.3",
        "trl==1.10.0",
        "peft==0.19.1",
        "accelerate==1.14.0",
        "datasets==5.0.0",
        "bitsandbytes",
    )
    .env({"HF_HOME": HF_CACHE_MOUNT})
)


@app.function(
    gpu=GPU,
    image=image,
    volumes={VOLUME_MOUNT: vol, HF_CACHE_MOUNT: hf_cache},
    timeout=8 * 60 * 60,
    # One GPU, always. See module docstring.
    max_containers=1,
)
def train(
    probe: bool = False,
    model: str = "Qwen/Qwen3-14B",
    max_length: int = 40960,
    epochs: float = 5.0,
    lr: float = 1e-4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    batch_size: int = 1,
    grad_accum: int = 8,
) -> dict:
    import json
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    data_path = Path(VOLUME_MOUNT) / DATA_IN_VOLUME
    rows = [json.loads(ln) for ln in data_path.read_text().splitlines() if ln.strip()]
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    print(f"loaded {len(ds)} segments from {data_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_dir = Path(VOLUME_MOUNT) / OUT_IN_VOLUME
    cfg = SFTConfig(
        output_dir=str(out_dir),
        model_init_kwargs={
            "dtype": torch.bfloat16,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
        },
        max_length=max_length,
        packing=False,
        assistant_only_loss=True,
        # Liger's fused CE would save the logit tensor but silently drops the
        # assistant mask (TRL #3781). Correctness over memory.
        use_liger_kernel=False,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        num_train_epochs=epochs,
        max_steps=3 if probe else -1,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=1,
        save_strategy="no" if probe else "epoch",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )

    # Prove the mask before spending the GPU. TRL #3927: assistant_only_loss
    # fails silently once a sequence passes max_length, and an all-IGNORE batch
    # trains on nothing while reporting a perfectly plausible loss.
    batch = next(iter(trainer.get_train_dataloader()))
    supervised = (batch["labels"] != -100).sum().item()
    total = batch["attention_mask"].sum().item()
    lengths = batch["attention_mask"].sum(dim=1).tolist()
    if supervised == 0:
        raise SystemExit("batch 0 has no supervised tokens — assistant mask is empty")
    if any(n >= max_length for n in lengths):
        raise SystemExit(f"batch 0 hit max_length {max_length} — TRL #3927 drops masks")
    kept = [t for t in batch["labels"][0].tolist() if t != -100][:80]
    print(f"MASK OK: {supervised}/{total} supervised ({100 * supervised / total:.1f}%), "
          f"lengths {lengths}", flush=True)
    print(f"first supervised text: {tokenizer.decode(kept)!r}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30

    seen = result.metrics.get("train_tokens", None)
    steps = result.metrics.get("train_steps_per_second", 0) * elapsed
    summary = {
        "probe": probe,
        "segments": len(ds),
        "elapsed_sec": round(elapsed, 1),
        "peak_vram_gib": round(peak, 1),
        "steps": round(steps, 1),
        "sec_per_optimizer_step": round(elapsed / max(steps, 1), 1),
        "train_loss": result.metrics.get("train_loss"),
        "tokens_seen": seen,
    }
    if probe:
        summary["note"] = (
            "probe only — multiply sec_per_optimizer_step by "
            f"(165/{batch_size * grad_accum} * {epochs}) for the full-run ETA"
        )
        print(json.dumps(summary, indent=2), flush=True)
        return summary

    trainer.save_model(str(out_dir))
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    vol.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(probe: bool = False, epochs: float = 5.0, max_length: int = 40960):
    out = train.remote(probe=probe, epochs=epochs, max_length=max_length)
    print(out)
