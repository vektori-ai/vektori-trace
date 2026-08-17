"""Continue the v1 LoRA on the protocol-corrected set — Modal, one A100-80GB.

`docs/SFT-REPAIR-PLAN.md` Phases 4-6. This is *not* `sft_train_modal.py` with
different flags. Three things differ, and each one is why that script cannot be
reused:

  1. **It continues an adapter instead of creating one.** `sft_train_modal.py`
     always builds a fresh `LoraConfig` from base (line 137). Here v1 is loaded
     through PEFT with `is_trainable=True` and no new config is constructed, so
     there is exactly one LoRA and it is the one being repaired.
  2. **It owns the loss mask.** TRL derives the mask from the chat template,
     which is per-*role*: it cannot supervise one assistant turn and skip the
     next. The ~48 post-compaction handoff turns are assistant prose that must
     stay in context and out of the loss, so the dataset arrives pre-tokenized
     with explicit labels and a label-preserving collator hands them through.
     That also retires TRL #3781 (Liger drops the mask) and TRL #3927 (the mask
     is silently lost past max_length).
  3. **It can run either precision.** v1 was fitted against an NF4 base
     (`sft_train_modal.py:104-110` is unconditional) but is *served* against
     BF16. `--probe` measures both arms before the real run picks one.

    modal run scripts/sft_repair_train_modal.py --probe          # 3 steps, BF16
    modal run scripts/sft_repair_train_modal.py --probe --nf4    # 3 steps, NF4
    modal run scripts/sft_repair_train_modal.py                  # the real run

Every invocation of this file costs GPU time and needs explicit per-run
approval (CLAUDE.md). Tear the app down the moment it returns.
"""

from __future__ import annotations

from pathlib import Path

import modal

# Mirrored from vektori_trace/runtime/modal_env.py rather than imported: Modal
# re-imports this module inside the container, where the local package is not
# installed, so a module-level import of it fails the run before training starts.
VOLUME_NAME = "vektori-trace-adapters"
VOLUME_MOUNT = "/adapters"
HF_CACHE_VOLUME_NAME = "hf-model-cache"
HF_CACHE_MOUNT = "/root/.cache/huggingface"

# BF16 at 40k context needs ~59-63 GiB; an L40S 48 GB cannot hold it, and NF4
# on one leaves no safe headroom above v1's measured 39.6 GiB. Serving hardware
# is not this run's hardware.
GPU = "A100-80GB"

DATA_IN_VOLUME = "sft-repaired/sft_repaired.jsonl"
V1_IN_VOLUME = "sft/qwen3-14b-dsv4-lora"       # immutable — never written to
OUT_IN_VOLUME = "sft/qwen3-14b-dsv4-lora-repaired"

MAX_LENGTH = 40960
TEMPLATE_KWARGS = {"enable_thinking": False}
IGNORE_INDEX = -100

app = modal.App("vektori-trace-sft-repair")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

# The versions v1 was trained and verified on. Unpinned, the container once
# resolved a TRL whose SFTConfig rejects `warmup_ratio` and the run died on a
# kwarg after the image build and model download were already paid for.
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
    max_containers=1,
)
def train(
    probe: bool = False,
    nf4: bool = False,
    model: str = "Qwen/Qwen3-14B",
    epochs: float = 3.0,
    lr: float = 1e-5,
    batch_size: int = 1,
    grad_accum: int = 8,
    save_steps: int = 10,
    seed: int = 0,
) -> dict:
    import json
    import time

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    # ---- data: pre-tokenized, labels already decided -----------------------
    data_path = Path(VOLUME_MOUNT) / DATA_IN_VOLUME
    rows = [json.loads(ln) for ln in data_path.read_text().splitlines() if ln.strip()]
    print(f"loaded {len(rows)} segments from {data_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = []
    for i, row in enumerate(rows):
        ex = tokenize_row(row, tokenizer)
        if ex is None:
            raise SystemExit(
                f"row {i} produced no supervised example — preflight should have "
                "caught this; refusing to train on a set it did not validate"
            )
        examples.append(ex)
    ds = Dataset.from_list(examples)

    supervised = sum(sum(1 for x in e["labels"] if x != IGNORE_INDEX) for e in examples)
    total = sum(len(e["input_ids"]) for e in examples)
    print(f"MASK: {supervised}/{total} tokens supervised "
          f"({100 * supervised / total:.1f}%) across ALL {len(examples)} rows", flush=True)
    if supervised == 0:
        raise SystemExit("no supervised tokens in the whole dataset")

    # ---- model: frozen base, v1 continued ----------------------------------
    model_kwargs: dict = {"dtype": torch.bfloat16}
    if nf4:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    base = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
    base.config.use_cache = False
    for p in base.parameters():
        p.requires_grad_(False)

    v1_path = Path(VOLUME_MOUNT) / V1_IN_VOLUME
    net = PeftModel.from_pretrained(base, str(v1_path), is_trainable=True)

    # No new LoraConfig was constructed, so there must be exactly one adapter and
    # every trainable parameter must belong to it. A second stacked LoRA, or a
    # base weight left unfrozen, would both show up here.
    adapters = list(net.peft_config)
    if adapters != ["default"]:
        raise SystemExit(f"expected exactly one adapter, found {adapters}")
    trainable = [n for n, p in net.named_parameters() if p.requires_grad]
    if not trainable:
        raise SystemExit("nothing is trainable — v1 loaded with is_trainable=False?")
    non_lora = [n for n in trainable if "lora_" not in n]
    if non_lora:
        raise SystemExit(f"{len(non_lora)} non-LoRA params are trainable: {non_lora[:5]}")
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"continuing v1 from {v1_path}: {len(trainable)} LoRA tensors, "
          f"{n_trainable:,} trainable params, rank {net.peft_config['default'].r}",
          flush=True)

    before = {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}

    steps_per_epoch = -(-len(ds) // (batch_size * grad_accum))
    max_steps = 3 if probe else int(steps_per_epoch * epochs)
    out_dir = Path(VOLUME_MOUNT) / OUT_IN_VOLUME

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        max_steps=max_steps,
        learning_rate=lr,
        optim="adamw_torch_fused",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        bf16=True,
        logging_steps=1,
        # Every ten optimizer steps, so the earliest checkpoint that passes the
        # format gates can be selected rather than the last one.
        save_strategy="no" if probe else "steps",
        save_steps=save_steps,
        save_total_limit=None,
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=net,
        args=args,
        train_dataset=ds,
        data_collator=label_preserving_collator(tokenizer.pad_token_id),
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30

    moved = sum(
        1
        for n, p in net.named_parameters()
        if p.requires_grad and not torch.equal(p.detach(), before[n])
    )
    summary = {
        "probe": probe,
        "precision": "nf4" if nf4 else "bf16",
        "segments": len(ds),
        "max_steps": max_steps,
        "steps_per_epoch": steps_per_epoch,
        "elapsed_sec": round(elapsed, 1),
        "peak_vram_gib": round(peak, 1),
        "sec_per_optimizer_step": round(elapsed / max(max_steps, 1), 1),
        "train_loss": result.metrics.get("train_loss"),
        "grad_norm_finite": bool(result.metrics.get("train_loss") == result.metrics.get("train_loss")),
        "lora_tensors_changed": moved,
        "lora_tensors_total": len(before),
        "supervised_tokens": supervised,
        "total_tokens": total,
    }

    if probe:
        # Nothing is saved: a three-step checkpoint is not a model candidate,
        # and writing one invites it being mistaken for one later.
        summary["note"] = (
            f"probe only — {summary['sec_per_optimizer_step']}s/step x "
            f"{steps_per_epoch * epochs:.0f} steps for the full run"
        )
        print(json.dumps(summary, indent=2), flush=True)
        return summary

    if moved == 0:
        raise SystemExit("no LoRA parameter changed — the run trained nothing")

    trainer.save_model(str(out_dir))
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    vol.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def tokenize_row(row: dict, tokenizer):
    """Render one segment to input_ids/labels with the explicit per-turn mask.

    Inlined rather than imported from `vektori_trace.dataset` because Modal
    re-imports this module inside a container where the local package is not
    installed. The logic is `dataset.tokenize_messages`; the preflight runs the
    real one over the same rows, and both must agree.
    """
    messages, supervise = row["messages"], row["supervise"]

    def encode(msgs):
        return tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False, **TEMPLATE_KWARGS
        )

    full = encode(messages)
    if hasattr(full, "get") and "input_ids" in full:
        full = full["input_ids"]
    labels = [IGNORE_INDEX] * len(full)
    prev = 0
    for i in range(len(messages)):
        prefix = encode(messages[: i + 1])
        if hasattr(prefix, "get") and "input_ids" in prefix:
            prefix = prefix["input_ids"]
        cur = len(prefix)
        if cur < prev:
            raise RuntimeError("chat template is not prefix-stable — refusing to mask")
        if supervise[i]:
            for j in range(prev, min(cur, len(full))):
                labels[j] = full[j]
        prev = cur
    if len(full) != prev:
        raise RuntimeError("prefix length != full length — refusing to mask")
    if len(full) > MAX_LENGTH:
        raise SystemExit(f"segment of {len(full)} tokens exceeds max_length {MAX_LENGTH}")
    if not any(lab != IGNORE_INDEX for lab in labels):
        return None
    return {"input_ids": full, "labels": labels, "attention_mask": [1] * len(full)}


def label_preserving_collator(pad_token_id: int):
    """Pad input_ids/attention_mask, pad labels with IGNORE_INDEX.

    The stock `DataCollatorForLanguageModeling` would rebuild labels from
    input_ids and throw away every masking decision made upstream, which is the
    whole point of pre-tokenizing. Mirrors `dataset.LabelPreservingCollator`,
    inlined for the same reason `tokenize_row` is.
    """
    import torch

    def collate(features: list[dict]) -> dict:
        width = max(len(f["input_ids"]) for f in features)

        def pad(key, value):
            return torch.tensor(
                [f[key] + [value] * (width - len(f[key])) for f in features],
                dtype=torch.long,
            )

        return {
            "input_ids": pad("input_ids", pad_token_id),
            "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", IGNORE_INDEX),
        }

    return collate


@app.local_entrypoint()
def main(probe: bool = False, nf4: bool = False, epochs: float = 3.0):
    print(train.remote(probe=probe, nf4=nf4, epochs=epochs))
