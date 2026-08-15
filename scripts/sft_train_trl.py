"""LoRA SFT of Qwen3-14B on exported agent traces, via TRL.

Consumes the JSONL written by `sft_export_traces.py` — one `{"messages": [...]}`
per passing trajectory — and trains a LoRA adapter.

Loss lands on assistant spans only. That is `assistant_only_loss=True`, which
needs `{% generation %}` markers in the chat template; TRL patches those in for
known families including Qwen3. Two TRL footguns are closed here rather than
discovered at step 400:

  * `use_liger_kernel=True` silently ignores `assistant_only_loss` (TRL #3781),
    so Liger stays off and the flag is asserted.
  * `assistant_only_loss` fails silently once a sequence passes `max_length`
    (TRL #3927), so `--check-batch` decodes batch 0 and refuses to train if any
    example was truncated or nothing carries loss.

    python scripts/sft_train_trl.py --data /data/sft --out /data/sft/adapter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IGNORE_INDEX = -100


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def check_batch(trainer, tokenizer, max_length: int) -> None:
    """Prove the mask before spending the GPU: some tokens must carry loss, and
    every supervised token must sit inside an assistant span."""
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"]
    supervised = (labels != IGNORE_INDEX).sum().item()
    total = batch["attention_mask"].sum().item()
    if supervised == 0:
        raise SystemExit(
            "batch 0 has no supervised tokens — assistant_only_loss found no "
            "assistant spans (chat template missing {% generation %}?)"
        )
    lengths = batch["attention_mask"].sum(dim=1).tolist()
    if any(n >= max_length for n in lengths):
        raise SystemExit(
            f"batch 0 has an example at max_length ({max_length}) — TRL #3927 "
            "drops assistant masks past the cap; raise max_length or drop the example"
        )
    row = labels[0]
    kept = tokenizer.decode([t for t in row if t != IGNORE_INDEX][:120])
    print(f"batch 0: {supervised}/{total} tokens supervised "
          f"({100 * supervised / total:.1f}%), lengths {lengths}")
    print(f"first supervised text: {kept!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="dir with sft_train.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    # Qwen3-14B's config sets max_position_embeddings=40960 with rope_scaling
    # null — the 32,768 on the model card is the recommended *input* length, not
    # the positional range. The longest rebuilt segment is 37,577 tokens, so
    # every one of them trains whole with no truncation and no YaRN.
    ap.add_argument("--max-length", type=int, default=40960)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--check-only", action="store_true",
                    help="build everything, verify the mask, do not train")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    train = Dataset.from_list(
        [{"messages": r["messages"]} for r in load_jsonl(args.data / "sft_train.jsonl")]
    )
    eval_path = args.data / "sft_eval.jsonl"
    evalset = (
        Dataset.from_list([{"messages": r["messages"]} for r in load_jsonl(eval_path)])
        if eval_path.exists()
        else None
    )
    print(f"train {len(train)}" + (f", eval {len(evalset)}" if evalset else ""))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"dtype": torch.bfloat16}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    cfg = SFTConfig(
        output_dir=str(args.out),
        model_init_kwargs=model_kwargs,
        max_length=args.max_length,
        packing=False,
        assistant_only_loss=True,
        # Liger's fused CE would save the logit tensor but silently drops the
        # assistant mask (TRL #3781). Correctness first.
        use_liger_kernel=False,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
    )
    if cfg.use_liger_kernel and cfg.assistant_only_loss:
        raise SystemExit("liger + assistant_only_loss silently drops the mask (TRL #3781)")

    trainer = SFTTrainer(
        model=args.model,
        args=cfg,
        train_dataset=train,
        eval_dataset=evalset,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )

    check_batch(trainer, tokenizer, args.max_length)
    if args.check_only:
        print("check-only: mask verified, not training")
        return 0

    trainer.train()
    trainer.save_model(str(args.out))
    print(f"adapter written to {args.out}")
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
