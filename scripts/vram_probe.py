"""Measure peak training VRAM for one real LoRA step on one GPU.

FINAL-PLAN.md assumes "an 8B LoRA fits one A10G". Nothing has ever run it, and a
24GB card is the permanent ceiling (no payment method → no L40S/A100/H100), so a
wrong assumption here is discovered hours into a billed run instead of now.

This drives the *real* `train_lora`, not a reimplementation, so what it measures
is what the pilot will spend. Cost is a fraction of a GPU-minute after the base
weights are in the shared HF cache Volume.

    uv run python scripts/vram_probe.py                     # 8192 ctx, checkpointing on
    uv run python scripts/vram_probe.py --no-grad-ckpt      # show what it buys
    uv run python scripts/vram_probe.py --seq-len 4096 --gpu L4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vektori_trace.modal_env import HF_CACHE_MOUNT, HF_CACHE_VOLUME_NAME


def _remote_stages(cfg: dict) -> dict:
    """Attribute VRAM to each setup stage, without training.

    Peak-only numbers cannot distinguish "quantization silently did nothing"
    from "quantization worked and something later ballooned" — and 4-bit
    measuring *worse* than bf16 means one of those is true. This reports the
    resident footprint after each step so the answer is read, not inferred.
    """
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    def mib() -> float:
        return round(torch.cuda.memory_allocated() / 2**20)

    stages: dict[str, float] = {"baseline": mib()}

    quant_kwargs = {}
    if cfg["load_in_4bit"]:
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        **quant_kwargs,
    )
    stages["after_from_pretrained"] = mib()

    # Which dtypes actually landed tells us directly whether bnb replaced the
    # Linear layers or quietly left them alone.
    dtype_bytes: dict[str, float] = {}
    for _, p in model.named_parameters():
        key = str(p.dtype)
        dtype_bytes[key] = dtype_bytes.get(key, 0) + p.numel() * p.element_size()
    stages_dtypes = {k: round(v / 2**20) for k, v in sorted(dtype_bytes.items())}

    # The single largest tensors, named. For a 151k-vocab model the embedding
    # and lm_head dominate and are exactly what an fp32 upcast would hit.
    biggest = sorted(
        ((n, p.numel() * p.element_size() / 2**20, str(p.dtype)) for n, p in model.named_parameters()),
        key=lambda t: -t[1],
    )[:6]

    if cfg["load_in_4bit"]:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg["gradient_checkpointing"]
        )
        stages["after_prepare_for_kbit"] = mib()

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    stages["after_get_peft_model"] = mib()

    props = torch.cuda.get_device_properties(0)
    return {
        "mode": "stages",
        "gpu": props.name,
        "total_mib": round(props.total_memory / 2**20),
        "stages_mib": stages,
        "param_mib_by_dtype": stages_dtypes,
        "largest_params": [{"name": n, "mib": round(m), "dtype": d} for n, m, d in biggest],
        **cfg,
    }


def _remote_probe(cfg: dict) -> dict:
    import torch

    from vektori_trace.dataset import TokenizedExample
    from vektori_trace.train import TrainConfig, train_lora

    seq_len = cfg["seq_len"]
    # Two examples so the Trainer has something to sample from across steps;
    # peak memory is set by a single batch, which is what we are measuring.
    examples = [
        TokenizedExample(
            input_ids=[1] * seq_len,
            # Mask the first half, as dataset.py does for the prompt: an
            # all-labels batch would understate nothing but overstate realism.
            labels=[-100] * (seq_len // 2) + [1] * (seq_len - seq_len // 2),
            attention_mask=[1] * seq_len,
        )
        for _ in range(2)
    ]

    props = torch.cuda.get_device_properties(0)
    torch.cuda.reset_peak_memory_stats()

    oom = None
    try:
        train_lora(
            examples,
            TrainConfig(
                base_model=cfg["base_model"],
                output_dir=Path("/tmp/vram-probe"),
                task_ids=["vram-probe"],
                max_steps=cfg["max_steps"],
                per_device_train_batch_size=cfg["batch_size"],
                gradient_accumulation_steps=cfg["grad_accum"],
                gradient_checkpointing=cfg["gradient_checkpointing"],
                bf16=True,
                load_in_4bit=cfg["load_in_4bit"],
                use_modal=False,
                stage_to_volume=False,
                # Writing the adapter is not part of the memory question and
                # costs GPU seconds; one step is enough to hit peak.
                save_total_limit=1,
            ),
        )
    except torch.cuda.OutOfMemoryError as e:
        oom = str(e).splitlines()[0]

    total_mib = props.total_memory / 2**20
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    reserved_mib = torch.cuda.max_memory_reserved() / 2**20
    return {
        "gpu": props.name,
        "total_mib": round(total_mib),
        "peak_allocated_mib": round(peak_mib),
        "peak_reserved_mib": round(reserved_mib),
        # Reserved is the number that decides whether the allocator can serve the
        # next request, so headroom is measured against it, not against allocated.
        "headroom_mib": round(total_mib - reserved_mib),
        "oom": oom,
        **cfg,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--gpu", default="L40S")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=2)
    p.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="QLoRA: load the frozen base in 4-bit NF4 (~16.4GB -> ~5.5GB for 8B)",
    )
    p.add_argument(
        "--no-grad-ckpt",
        dest="gradient_checkpointing",
        action="store_false",
        help="measure without gradient checkpointing (expected to be much larger)",
    )
    p.add_argument(
        "--stages",
        action="store_true",
        help="attribute memory to each setup stage instead of running a training step",
    )
    p.add_argument("--out", default="docs/vram-probe.json")
    args = p.parse_args(argv)

    import modal

    cfg = {
        "base_model": args.model,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_steps": args.max_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "load_in_4bit": args.load_in_4bit,
    }

    app = modal.App("vektori-trace-vram-probe")
    hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch", "transformers", "peft", "accelerate", "datasets", "bitsandbytes"
        )
        .env({"HF_HOME": HF_CACHE_MOUNT})
        .add_local_python_source("vektori_trace")
    )
    fn = app.function(
        gpu=args.gpu,
        image=image,
        volumes={HF_CACHE_MOUNT: hf_cache},
        timeout=60 * 60,
    )(_remote_stages if args.stages else _remote_probe)

    with app.run():
        result = fn.remote(cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out}")
    if result.get("oom"):
        print("\nOOM — this configuration does not fit. Reduce seq-len or enable checkpointing.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
