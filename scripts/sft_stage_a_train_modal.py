"""Stage A: a fresh LoRA on base Qwen3-14B, format only — Modal, one A100-80GB.

`docs/SFT-SCRATCH-PLAN.md` step 5. This is a fork of
`sft_repair_train_modal.py`, not a variant of `sft_train_modal.py`: the repair
trainer already owns the two mechanisms Stage A cannot do without — a
pre-tokenized dataset with an explicit per-row label mask, and `chunked_nll`.
`sft_train_modal.py` has neither (it derives the mask from the chat template via
`assistant_only_loss`, which is per-*role* and cannot skip a non-final assistant
turn), so starting from it would mean porting both across.

Four things differ from the repair trainer:

  1. **It creates an adapter instead of continuing one.** No v1, no repaired, no
     ck63. A fresh `LoraConfig` r=32 / alpha=64 / dropout 0.05 / `all-linear`,
     built here and asserted to be the only adapter on the model.
  2. **`max_length` is 8192, not 40960.** Stage A is 165 first-action rows,
     longest 6494 tokens (`/data/sft-stage-a/mix_report.json`). The 18
     parse-error recoveries that need 33k are Stage B's — amendment 2.
  3. **BF16 is the default arm.** The repair trainer defaults `--nf4` because a
     36,993-token row over a 151,936 vocab does not fit at either precision
     without it. That figure is NF4 @ 40960 and does not transfer: at 8192 the
     logit tensor is a fifth the size, and `chunked_nll` never materialises it
     whole anyway. So BF16 is tried first and measured. Peak > 60 GiB or an OOM
     switches to `--nf4`; the arm is recorded in `run_summary.json` and not
     re-litigated (plan step 5).
  4. **The probe checks what was trained, not just that something was.** The
     first supervised span must decode to bare JSON. If it decodes to
     `<think>\n\n</think>\n\n` the step-1 wrapper mask has failed and the run
     would teach empty thinking on all 165 rows — a CPU bug, caught before the
     epoch rather than at selection.

    modal run scripts/sft_stage_a_train_modal.py --probe          # 3 steps, bf16
    modal run scripts/sft_stage_a_train_modal.py --probe --nf4    # fallback arm
    modal run scripts/sft_stage_a_train_modal.py                  # the real run

`--probe` runs on the *longest* rows, not a random sample: peak memory is
dominated by sequence length. The sampler still shuffles within that selection.

Every invocation costs GPU time and needs explicit per-run approval (CLAUDE.md).
Tear the app down the moment it returns.
"""

from __future__ import annotations

from pathlib import Path

import modal

# Mirrored from vektori_trace/runtime/modal_env.py rather than imported: Modal
# re-imports this module inside a container where the local package is not
# installed, so a module-level import of it fails the run before training starts.
VOLUME_NAME = "vektori-trace-adapters"
VOLUME_MOUNT = "/adapters"
HF_CACHE_VOLUME_NAME = "hf-model-cache"
HF_CACHE_MOUNT = "/root/.cache/huggingface"

# Same card as the repair run. An L40S 48 GB is serving hardware; the BF16 arm
# alone is ~28 GiB of weights before an optimizer state exists.
GPU = "A100-80GB"

DATA_IN_VOLUME = "sft-stage-a/stage_a.jsonl"
OUT_IN_VOLUME = "sft/qwen3-14b-stage-a-lora"

# Stage A's longest row is 6,494 tokens. 8192 leaves headroom without paying for
# a tail that is not in this dataset. `tokenize_row` refuses anything longer
# rather than truncating, so a dataset change surfaces as a failure, not a
# silently clipped target.
MAX_LENGTH = 8192

# Fresh adapter — plan step 5. Not read from any existing checkpoint.
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"

# Above this, the BF16 arm is refused and the run is repeated with --nf4
# (plan step 5). Measured on the allocator's *reserved* pool, which is what
# actually has to fit in the card.
BF16_PEAK_CEILING_GIB = 60.0

# Qwen3 emits this before a *final* assistant turn's content. Mirrors
# `dataset.THINK_WRAPPER_TEXT`; it is masked, never supervised.
THINK_WRAPPER_TEXT = "<think>\n\n</think>\n\n"
# Thinking stays on: the corpus is tokenized the way the model is served.
# `docs/SFT-SCRATCH-PLAN.md` step 2.
TEMPLATE_KWARGS = {"enable_thinking": True}
IGNORE_INDEX = -100

app = modal.App("vektori-trace-sft-stage-a")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

# The versions the masking logic and the export were verified against. Unpinned,
# the container once resolved a TRL whose SFTConfig rejects `warmup_ratio` and
# the run died on a kwarg after the image build and model download were paid for.
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
    memory_history: bool = False,
    model: str = "Qwen/Qwen3-14B",
    epochs: float = 4.0,
    lr: float = 1e-4,
    batch_size: int = 1,
    grad_accum: int = 8,
    save_steps: int = 10,
    seed: int = 0,
) -> dict:
    import hashlib
    import json
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    marks: list[dict] = []

    def mark(stage: str) -> None:
        """Record CUDA memory at a boundary, so an OOM says *where* it grew."""
        if not torch.cuda.is_available():
            return
        row = {
            "stage": stage,
            "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
            "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
            "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        marks.append(row)
        print(f"MEM[{stage}]: allocated {row['allocated_gib']} GiB, "
              f"reserved {row['reserved_gib']} GiB", flush=True)

    # ---- data: pre-tokenized, labels already decided -----------------------
    data_path = Path(VOLUME_MOUNT) / DATA_IN_VOLUME
    rows = [json.loads(ln) for ln in data_path.read_text().splitlines() if ln.strip()]
    print(f"loaded {len(rows)} rows from {data_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = []
    for i, row in enumerate(rows):
        ex = tokenize_row(row, tokenizer)
        if ex is None:
            raise SystemExit(
                f"row {i} produced no supervised example — the Stage A builder "
                "should have caught this; refusing to train on a set it did not "
                "validate"
            )
        examples.append(ex)
    ds = Dataset.from_list(examples)

    supervised = sum(sum(1 for x in e["labels"] if x != IGNORE_INDEX) for e in examples)
    total = sum(len(e["input_ids"]) for e in examples)
    print(f"MASK: {supervised}/{total} tokens supervised "
          f"({100 * supervised / total:.1f}%) across ALL {len(examples)} rows", flush=True)
    if supervised == 0:
        raise SystemExit("no supervised tokens in the whole dataset")

    # `tokenize_row` is a copy of `dataset.tokenize_messages`, not a call to it —
    # Modal re-imports this module in a container without the local package. A
    # copy can drift, so the builder records what the real function produced per
    # row and this refuses to train on any disagreement.
    fp_path = data_path.parent / "tokenization_fingerprint.json"
    if not fp_path.exists():
        raise SystemExit(
            f"{fp_path} is missing — run scripts/sft_stage_a_dataset.py and stage "
            "its fingerprint before training"
        )
    expected = json.loads(fp_path.read_text())
    if expected.get("model") != model:
        raise SystemExit(
            f"fingerprint is for {expected.get('model')!r}, training {model!r}"
        )
    if expected.get("template_kwargs") != TEMPLATE_KWARGS:
        raise SystemExit(
            f"fingerprint rendered with {expected.get('template_kwargs')}, "
            f"training with {TEMPLATE_KWARGS}"
        )
    if expected.get("max_length") != MAX_LENGTH:
        raise SystemExit(
            f"fingerprint used max_length {expected.get('max_length')}, "
            f"training with {MAX_LENGTH}"
        )
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if expected.get("dataset_sha256") not in (None, data_sha):
        raise SystemExit(
            f"fingerprint is for dataset {expected['dataset_sha256'][:16]}, "
            f"this one is {data_sha[:16]}"
        )
    actual = [row_digest(e) for e in examples]
    if expected.get("per_row") != actual:
        bad = [
            i for i, (a, b) in enumerate(
                zip(expected.get("per_row", []), actual, strict=False)
            )
            if a != b
        ]
        raise SystemExit(
            f"tokenization disagrees with the builder on {len(bad)} row(s) "
            f"(first: row {bad[0] if bad else '?'}; "
            f"{len(expected.get('per_row', []))} vs {len(actual)} rows) — "
            "tokenize_row has drifted from dataset.tokenize_messages"
        )
    print(f"tokenization matches the builder fingerprint exactly on all "
          f"{len(actual)} rows (dataset {data_sha[:16]})", flush=True)

    # ---- model: frozen base, a brand new adapter ---------------------------
    model_kwargs: dict = {"dtype": torch.bfloat16}
    if nf4:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        # bitsandbytes quantizes weights as they are *placed*. Without a
        # device_map the model loads on CPU in full precision and Trainer moves
        # it to the GPU afterwards, unquantized — the config silently does
        # nothing.
        model_kwargs["device_map"] = {"": 0}
    print(f"precision: {'nf4' if nf4 else 'bf16'}, "
          f"model_kwargs keys {sorted(model_kwargs)}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
    base.config.use_cache = False

    # Report what actually loaded, before anything expensive. A flag that
    # silently does nothing is the failure this makes impossible.
    footprint = base.get_memory_footprint() / 2**30
    dtypes: dict[str, int] = {}
    for prm in base.parameters():
        key = str(prm.dtype)
        dtypes[key] = dtypes.get(key, 0) + prm.numel()
    print(f"base loaded: {footprint:.1f} GiB, param dtypes "
          f"{ {k: f'{v / 1e9:.2f}B' for k, v in sorted(dtypes.items())} }", flush=True)
    n_4bit = sum(
        1 for m in base.modules() if type(m).__name__ in {"Linear4bit", "Params4bit"}
    )
    print(f"bnb 4-bit modules: {n_4bit}", flush=True)
    mark("base_loaded")
    if nf4 and n_4bit == 0:
        raise SystemExit(
            "--nf4 was requested but the model contains zero bnb Linear4bit "
            "modules — the quantization_config was inert"
        )
    if nf4 and footprint > 15.0:
        raise SystemExit(
            f"--nf4 was requested but the base is {footprint:.1f} GiB — "
            "quantization did not apply, refusing to measure the wrong model"
        )
    if not nf4 and footprint < 20.0:
        raise SystemExit(
            f"bf16 was requested but the base is only {footprint:.1f} GiB — "
            "something quantized it unexpectedly"
        )
    if nf4:
        # A quantized base otherwise leaves layer norms in a dtype the backward
        # pass cannot use and the input embeddings produce no grad.
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=True
        )
    for p in base.parameters():
        p.requires_grad_(False)

    # The one line that makes this Stage A and not the repair run: the adapter is
    # constructed here from base, never loaded from disk.
    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    net = get_peft_model(base, lora)

    adapters = list(net.peft_config)
    if adapters != ["default"]:
        raise SystemExit(f"expected exactly one adapter, found {adapters}")
    trainable = [n for n, p in net.named_parameters() if p.requires_grad]
    if not trainable:
        raise SystemExit("nothing is trainable — get_peft_model produced no LoRA")
    non_lora = [n for n in trainable if "lora_" not in n]
    if non_lora:
        raise SystemExit(f"{len(non_lora)} non-LoRA params are trainable: {non_lora[:5]}")
    # chunked_nll refuses to run when lm_head is itself LoRA-adapted
    # (trl/trainer/sft_trainer.py:1310-1322), because it projects the hidden
    # states through lm_head itself. PEFT's "all-linear" excludes
    # get_output_embeddings() (peft/tuners/tuners_utils.py:1922-1930) — asserted
    # rather than trusted, since a mismatch surfaces as a ValueError deep in a run.
    adapted = net.peft_config["default"].target_modules or []
    if any("lm_head" in str(m) for m in adapted):
        raise SystemExit(
            f"lm_head is LoRA-adapted ({adapted}) — chunked_nll cannot be used"
        )
    # A fresh LoRA initialises B to zero, so step 0 is exactly the base model.
    # If any B were non-zero the adapter came from somewhere, which is the one
    # thing Stage A forbids.
    nonzero_b = [
        n for n, p in net.named_parameters()
        if "lora_B" in n and bool(p.detach().any())
    ]
    if nonzero_b:
        raise SystemExit(
            f"{len(nonzero_b)} lora_B tensors are non-zero at init "
            f"({nonzero_b[:3]}) — this adapter is not fresh"
        )

    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"fresh LoRA: {len(trainable)} tensors, {n_trainable:,} trainable params, "
          f"r={lora.r} alpha={lora.lora_alpha} dropout={lora.lora_dropout} "
          f"target={LORA_TARGET_MODULES}", flush=True)

    # LoRA freezes the base weights, so without this the checkpointed
    # activations have nothing requiring grad and the backward pass is empty —
    # a run that reports a plausible loss and learns nothing.
    net.enable_input_require_grads()

    mark("adapter_loaded")
    before = _snapshot_trainable(net)

    steps_per_epoch = -(-len(ds) // (batch_size * grad_accum))
    max_steps = 3 if probe else int(steps_per_epoch * epochs)
    out_dir = Path(VOLUME_MOUNT) / OUT_IN_VOLUME

    if probe:
        # Trainer's sampler is shuffled, so three steps may never touch the
        # longest row — and peak VRAM is dominated by sequence length. A probe
        # that measured only median rows would certify a footprint the real run
        # then exceeds.
        picked = _longest_first(examples, n=batch_size * grad_accum * max_steps)
        ds = ds.select(picked)
        lengths = [len(examples[i]["input_ids"]) for i in picked]
        print(f"probe: {len(picked)} longest rows, indices {picked[:8]}..., "
              f"lengths {lengths[:8]}... (max {lengths[0]}, min {lengths[-1]})",
              flush=True)

    # What the first supervised span actually is. Plan step 5 asks for this
    # explicitly: bare JSON means the wrapper mask held, `<think>` means step 1
    # failed and all 165 rows would teach an empty reasoning block.
    first = examples[picked[0]] if probe else examples[0]
    kept = [t for t in first["labels"] if t != IGNORE_INDEX]
    first_span = tokenizer.decode(kept[:64])
    print(f"first supervised span: {first_span!r}", flush=True)
    if first_span.lstrip().startswith("<think>"):
        raise SystemExit(
            "the first supervised span starts with <think> — the wrapper mask "
            "failed and this run would teach an empty reasoning block on every "
            "row. Fix it on CPU (docs/SFT-SCRATCH-PLAN.md step 1), do not train."
        )

    args = SFTConfig(
        output_dir=str(out_dir),
        # chunked_nll (trl/trainer/sft_trainer.py:117-234) drops every -100
        # position *before* the lm_head matmul and projects what remains in
        # 256-token chunks, each individually checkpointed. It is TRL's default
        # whenever Liger is off (sft_config.py:334-335); set explicitly so nobody
        # inherits it by accident, and so a port away from SFTConfig fails loudly.
        loss_type="chunked_nll",
        # Liger's fused CE would save the logit tensor but silently drops the
        # assistant mask (TRL #3781).
        use_liger_kernel=False,
        # Our labels are authoritative: built offline, verified per row against a
        # sha256 fingerprint. SFTTrainer detects a pre-tokenized dataset
        # (`is_processed = "input_ids" in column_names`, sft_trainer.py:1397) and
        # skips its own chat-template and masking pipeline entirely.
        assistant_only_loss=False,
        # Auto-detection preserves the labels but still runs truncation and
        # column handling over rows that are already final; a silent truncation
        # is exactly what the fingerprint exists to rule out.
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=MAX_LENGTH,
        packing=False,
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
        lr_scheduler_type="cosine",
        # A 3-step probe that spends all 3 in warmup measures the wrong step time.
        warmup_ratio=0.0 if probe else 0.03,
        bf16=True,
        logging_steps=1,
        # Every ten optimizer steps, so the *earliest* checkpoint clearing the 45
        # can be selected rather than the last one (plan step 6).
        save_strategy="no" if probe else "steps",
        save_steps=save_steps,
        save_total_limit=None,
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
    )

    # Trainer logs grad_norm per step but keeps no summary of it, and
    # `train_loss` being non-NaN says nothing about whether the backward pass
    # produced gradients at all.
    grad_norms: list[float] = []
    losses: list[float] = []

    class _Collect(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            if "grad_norm" in logs:
                grad_norms.append(float(logs["grad_norm"]))
            if "loss" in logs:
                losses.append(float(logs["loss"]))

    trainer = SFTTrainer(
        model=net,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        # Only replaced when None (sft_trainer.py:1190); refused only under
        # padding_free, which is off.
        data_collator=label_preserving_collator(tokenizer.pad_token_id),
        callbacks=[_Collect()],
    )
    print(f"loss_type={args.loss_type}, use_liger_kernel={args.use_liger_kernel}, "
          f"assistant_only_loss={args.assistant_only_loss}", flush=True)

    if memory_history:
        torch.cuda.memory._record_memory_history(max_entries=100_000)
    mark("pre_train")
    t0 = time.time()
    try:
        result = trainer.train()
    except torch.cuda.OutOfMemoryError as exc:
        # An OOM used to leave nothing but a traceback, because every summary was
        # built after train() returned. The failure is the measurement here.
        report = {
            "probe": probe,
            "precision": "nf4" if nf4 else "bf16",
            "outcome": "oom",
            "error": str(exc)[:2000],
            "elapsed_sec": round(time.time() - t0, 1),
            "memory_marks": marks,
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "base_footprint_gib": round(footprint, 2),
            "bnb_4bit_modules": n_4bit,
            "longest_row_tokens": max(len(e["input_ids"]) for e in examples),
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "next_arm": "nf4" if not nf4 else None,
        }
        if memory_history:
            snap = _artifact(f"memory_snapshot_{'nf4' if nf4 else 'bf16'}.pickle")
            torch.cuda.memory._dump_snapshot(str(snap))
            report["memory_snapshot"] = str(snap)
            print(f"memory snapshot written to {snap}", flush=True)
        out = _artifact(f"probe_failure_{'nf4' if nf4 else 'bf16'}.json")
        out.write_text(json.dumps(report, indent=2))
        vol.commit()
        print(json.dumps(report, indent=2), flush=True)
        raise SystemExit(
            f"PROBE OOM at {'nf4' if nf4 else 'bf16'} — report written to {out}."
            + ("" if nf4 else " Re-run with --nf4 (plan step 5).")
        ) from exc
    if memory_history:
        snap = _artifact(f"memory_snapshot_{'nf4' if nf4 else 'bf16'}.pickle")
        torch.cuda.memory._dump_snapshot(str(snap))
        print(f"memory snapshot written to {snap}", flush=True)
    elapsed = time.time() - t0
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    # Printed here, not only inside `summary`. The first bf16 probe measured the
    # peak and then threw in the moved-check below, which sits between this line
    # and the only place the number was reported — so ~6 GPU-minutes produced no
    # measurement at all. Whatever fails after this point, the number is out.
    print(f"PEAK: allocated {peak_alloc:.1f} GiB, reserved {peak_reserved:.1f} GiB "
          f"(ceiling {BF16_PEAK_CEILING_GIB:.0f} GiB, arm "
          f"{'nf4' if nf4 else 'bf16'})", flush=True)

    moved = _count_moved(net, before)
    train_loss = result.metrics.get("train_loss")

    def finite(x) -> bool:
        return x is not None and x == x and abs(x) != float("inf")

    summary = {
        "probe": probe,
        "stage": "A",
        "precision": "nf4" if nf4 else "bf16",
        "rows": len(ds),
        "max_steps": max_steps,
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "max_length": MAX_LENGTH,
        "lora": {
            "r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_MODULES, "fresh": True,
        },
        "elapsed_sec": round(elapsed, 1),
        # Reserved, not just allocated: the allocator's reserved pool is what
        # actually has to fit in the card, and it is what OOMs.
        "peak_vram_allocated_gib": round(peak_alloc, 1),
        "peak_vram_reserved_gib": round(peak_reserved, 1),
        "longest_row_tokens": max(len(e["input_ids"]) for e in examples),
        "sec_per_optimizer_step": round(elapsed / max(max_steps, 1), 1),
        "train_loss": train_loss,
        "losses": losses,
        "grad_norms": grad_norms,
        "loss_finite": finite(train_loss),
        "grad_norms_finite": bool(grad_norms) and all(finite(g) for g in grad_norms),
        "grad_norms_nonzero": bool(grad_norms) and all(g > 0 for g in grad_norms),
        "lora_tensors_changed": moved,
        "lora_tensors_total": len(before),
        "supervised_tokens": supervised,
        "total_tokens": total,
        "first_supervised_span": first_span,
        "base_footprint_gib": round(footprint, 2),
        "bnb_4bit_modules": n_4bit,
        "memory_marks": marks,
        "dataset_sha256": data_sha,
    }

    # These are the probe's actual questions, so they are asked before it can
    # report success. A probe that says "fine" on an empty backward pass is
    # worse than no probe.
    problems = []
    if not summary["loss_finite"]:
        problems.append(f"train_loss is not finite: {train_loss!r}")
    if not grad_norms:
        problems.append("no gradient norms were logged — nothing to verify")
    elif not summary["grad_norms_finite"]:
        problems.append(f"non-finite gradient norm in {grad_norms}")
    elif not summary["grad_norms_nonzero"]:
        problems.append(f"zero gradient norm in {grad_norms} — backward pass is empty")
    if moved == 0:
        problems.append("no LoRA parameter changed — trained nothing")
    if not nf4 and peak_reserved > BF16_PEAK_CEILING_GIB:
        # Not an OOM, but the plan's stated switch point: a bf16 arm this close
        # to the card is not the arm to run 84 steps on.
        problems.append(
            f"bf16 peak reserved {peak_reserved:.1f} GiB exceeds the "
            f"{BF16_PEAK_CEILING_GIB:.0f} GiB ceiling — re-run with --nf4 "
            "(plan step 5)"
        )
    if problems:
        summary["outcome"] = "failed"
        out = _artifact(f"probe_failure_{'nf4' if nf4 else 'bf16'}.json")
        out.write_text(json.dumps(summary, indent=2))
        vol.commit()
        print(json.dumps(summary, indent=2), flush=True)
        raise SystemExit("PROBE FAILED: " + "; ".join(problems))

    if probe:
        # Nothing is saved: a three-step checkpoint is not a model candidate,
        # and writing one invites it being mistaken for one later.
        summary["note"] = (
            f"probe only — {summary['sec_per_optimizer_step']}s/step x "
            f"{steps_per_epoch * epochs:.0f} steps for the full run"
        )
        print(json.dumps(summary, indent=2), flush=True)
        return summary

    trainer.save_model(str(out_dir))
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    vol.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _snapshot_trainable(net) -> dict:
    """CPU copies of every trainable parameter, for the did-it-move check.

    Deliberately on the CPU. The bf16 arm passes no `device_map`, so at snapshot
    time the model is still on the host and Trainer moves it to the GPU
    afterwards — a snapshot taken with a bare `.clone()` then holds CPU tensors
    that `torch.equal` refuses to compare against `cuda:0` ones. The first bf16
    probe died exactly there, after training had already succeeded. The repair
    trainer never hit it because its nf4 arm sets `device_map={"": 0}` and the
    params are on the GPU before the snapshot is taken.

    CPU is also the cheaper side: keeping ~128M bf16 params off the card is
    ~257 MiB of VRAM that the measurement does not have to account for.
    """
    return {
        n: p.detach().to("cpu", copy=True)
        for n, p in net.named_parameters()
        if p.requires_grad
    }


def _count_moved(net, before: dict) -> int:
    """How many trainable tensors differ from the snapshot.

    Both sides are forced to the CPU: a device-naive comparison is what broke
    the first probe, and `before` may legitimately have been captured on either
    device depending on the precision arm.
    """
    import torch

    return sum(
        1
        for n, p in net.named_parameters()
        if p.requires_grad and not torch.equal(p.detach().cpu(), before[n].cpu())
    )


def _artifact(name: str) -> Path:
    """A path under the Stage A directory on the volume, parents created.

    Failure reports go next to the dataset they were produced from, never into
    the repair run's directory — the two must not be read as one series.
    """
    out = Path(VOLUME_MOUNT) / "sft-stage-a" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _longest_first(examples: list[dict], n: int) -> list[int]:
    """Indices of the n longest rows, longest first.

    Selecting *only* long rows makes the probe deliberately conservative: every
    step is near worst case, so a peak measured here bounds the real run rather
    than sampling somewhere under it.
    """
    order = sorted(range(len(examples)), key=lambda i: -len(examples[i]["input_ids"]))
    return order[: min(n, len(order))]


def row_digest(example: dict) -> str:
    """Exact hash of one tokenized row.

    Counts are not identity: two different token sequences, or two different
    masks, can share a length and a supervised-token total. Since the trainer
    re-tokenizes with its own copy of the logic, the guard has to compare the
    tokens themselves or a drift that preserves counts slips through.
    """
    import hashlib

    h = hashlib.sha256()
    for key in ("input_ids", "labels", "attention_mask"):
        h.update(key.encode())
        h.update(b",".join(str(x).encode() for x in example[key]))
    return h.hexdigest()


def _ids(encoded):
    """Bare token ids from whatever apply_chat_template returned.

    Mirrors `dataset._unwrap_ids`. `list()` on a BatchEncoding yields dict keys
    and on a tokenizers.Encoding yields something opaque — either way a prefix
    comparison over those values succeeds without comparing token ids, which is
    how an assert becomes decoration.
    """
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif hasattr(encoded, "get") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if not isinstance(encoded, list):
        encoded = list(encoded)
    if len(encoded) == 1 and isinstance(encoded[0], list):
        encoded = encoded[0]
    if encoded and not all(isinstance(i, int) for i in encoded):
        raise RuntimeError(
            "apply_chat_template did not yield token ids "
            f"(first element is {type(encoded[0]).__name__})"
        )
    return encoded


def tokenize_row(row: dict, tokenizer):
    """Render one row to input_ids/labels: last message supervised, wrapper masked.

    Inlined rather than imported from `vektori_trace.dataset` because Modal
    re-imports this module inside a container where the local package is not
    installed. The logic is `dataset.tokenize_messages`; the builder runs the
    real one over the same rows and the per-row fingerprint refuses to train on
    any disagreement — so a drift here is caught, not trained.
    `tests/test_trainer_tokenizer_parity.py` checks the copy on CPU as well.

    Qwen3's template wraps an assistant turn in `<think>\n\n</think>\n\n` only
    when it is `loop.last`, so `render(messages[:i+1])` is four tokens longer
    than the matching span of `render(messages)` for any non-final assistant.
    Measuring lengths on the former and indexing them into the latter ran every
    supervised span past `<|im_end|>` into the following user turn. See
    `docs/SFT-SCRATCH-PLAN.md` step 1.
    """
    messages, supervise = row["messages"], row["supervise"]
    if len(supervise) != len(messages):
        raise SystemExit(
            f"supervise has {len(supervise)} entries for {len(messages)} messages"
        )
    if not messages or not any(supervise):
        return None
    bad = [i for i, sup in enumerate(supervise) if sup and i != len(messages) - 1]
    if bad:
        raise SystemExit(
            f"row supervises messages {bad}, which are not the last "
            f"(index {len(messages) - 1}). Only a final-message target can be "
            "located in the full render; split the row per action."
        )
    if messages[-1].get("role") != "assistant":
        raise SystemExit(
            f"the supervised final message has role {messages[-1].get('role')!r}"
        )

    full = _ids(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, **TEMPLATE_KWARGS
        )
    )
    prefix = _ids(
        tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, **TEMPLATE_KWARGS
        )
    )
    if full[: len(prefix)] != prefix:
        raise SystemExit(
            "render(messages[:-1], add_generation_prompt=True) is not a token "
            f"prefix of render(messages) ({len(prefix)} tokens) — refusing to mask"
        )

    start = len(prefix)
    wrapper = _ids(tokenizer.encode(THINK_WRAPPER_TEXT, add_special_tokens=False))
    if full[start : start + len(wrapper)] != wrapper:
        raise SystemExit(
            f"expected the reasoning wrapper {wrapper} at token {start}, found "
            f"{full[start : start + len(wrapper)]} — the template no longer emits "
            "<think></think> on a final assistant turn"
        )
    # Context, not target. Supervising these teaches an empty reasoning block on
    # every row: enable_thinking=True with the behaviour trained out.
    start += len(wrapper)

    if len(full) > MAX_LENGTH:
        raise SystemExit(f"row of {len(full)} tokens exceeds max_length {MAX_LENGTH}")
    labels = [IGNORE_INDEX] * start + full[start:]
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
def main(
    probe: bool = False,
    nf4: bool = False,
    memory_history: bool = False,
    epochs: float = 4.0,
):
    print(train.remote(
        probe=probe, nf4=nf4, memory_history=memory_history, epochs=epochs
    ))
