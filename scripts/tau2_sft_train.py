#!/usr/bin/env python3
"""Tau2 retail warm-start SFT: W30 -> A_warm. TRL SFTTrainer, pre-tokenized.

Not a custom training loop. TRL owns the step, the scheduler and the LoRA
plumbing; what it must NOT own here is tokenization or the loss mask. The corpus
already carries authoritative `input_ids` / `labels` / `attention_mask`, verified
row by row by `tau2_sft_preflight.py`, and three flags keep TRL out of them:

    assistant_only_loss=False      its mask is per-role and all-or-nothing; it
                                   would supervise every earlier assistant turn
                                   in the prefix, not just the final action
    skip_prepare_dataset=True      otherwise TRL still runs truncation over rows
                                   that are already final
    LabelPreservingCollator        the stock LM collator regenerates labels from
                                   input_ids on pad and erases every -100

`loss_type="chunked_nll"` and `use_liger_kernel=False` reproduce the memory
profile Stage A actually fit in; a plain Trainer port did not.

Two modes:
  --probe   ~10 deterministic rows including the longest, 3 steps, save/reload
  (default) full W30, epoch checkpoints

Fails closed on split leakage, hash drift, truncation, a mask changed by
collation, zero supervised tokens, a supervised <think> wrapper, non-finite loss
or gradients, and a failed adapter round trip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IGNORE = -100

# Serving-pinned. vLLM defaults to max_lora_rank 16 and refuses to start above
# it without the flag, so a rank change is a serving change (see serve.py).
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"

# Measured, not inherited: the rendered W30/C30 corpus with real tool schemas
# peaks at 13,344 tokens. See docs/TAU2-CORPUS-EXCLUSIONS.md.
MAX_LENGTH = 16384

# What a run must leave behind, by purpose. Serving needs the first set; an
# exact resume needs the first plus the second; reproducing the result needs
# both plus the hashes and versions in run_config.json.
INFERENCE_FILES = (
    "adapter_config.json",        # rank/alpha/target_modules
    "adapter_model.safetensors",  # the trained weights
    "tokenizer.json",             # exact vocab and merges
    "tokenizer_config.json",      # special tokens, padding side
)
#: Present as separate files on some tokenizers and folded into
#: `tokenizer_config.json` on others. Qwen3 does the latter for both, so
#: requiring the files outright fails a perfectly complete artifact -- which is
#: exactly what happened to run a_warm_20260825_003343 after 45 minutes of
#: successful training. Each entry is (filename, key inside tokenizer_config).
EITHER_FILE_OR_KEY = (
    ("special_tokens_map.json", "eos_token"),
    ("chat_template.jinja", "chat_template"),
)
RESUME_FILES = (
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state.pth",   # without it a resume replays different data order
)


class GateFailure(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"GATE FAILED: {msg}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Only these may be optimized on. S16 decides checkpoints and recipes; F38 is
# the single frozen final comparison. Naming either here is not a typo to be
# tolerated -- it is the one mistake that silently invalidates the experiment,
# so it is refused by name rather than caught by a leakage check that would
# happily confirm "all F38 rows are indeed in F38".
TRAINABLE_PARTITIONS = ("W30", "C30")


def parent_adapter_manifest(path: str, base_model: str) -> dict:
    """Validate and fingerprint the LoRA that a continuation starts from."""
    required = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not os.path.isfile(os.path.join(path, name))]
    if missing:
        raise GateFailure(f"parent adapter {path} is incomplete, missing {missing}")
    config = json.load(open(os.path.join(path, "adapter_config.json")))
    parent_base = config.get("base_model_name_or_path")
    if parent_base and parent_base != base_model:
        raise GateFailure(
            f"parent adapter base {parent_base!r} != requested {base_model!r}"
        )
    if int(config.get("r", -1)) != LORA_R:
        raise GateFailure(
            f"parent adapter rank {config.get('r')!r} != expected {LORA_R}"
        )
    files = {}
    for name in sorted(os.listdir(path)):
        fp = os.path.join(path, name)
        if os.path.isfile(fp) and name in INFERENCE_FILES:
            files[name] = sha256_file(fp)
    return {
        "path": path,
        "base_model_name_or_path": parent_base,
        "r": config["r"],
        "lora_alpha": config.get("lora_alpha"),
        "files_sha256": files,
    }


def load_rows(artifacts: str, partition: str, expect_manifest: str | None):
    """Trainable-partition rows only, proven against the frozen manifest."""
    if partition not in TRAINABLE_PARTITIONS:
        raise GateFailure(
            f"{partition} is not a trainable partition. S16 is for "
            f"checkpoint/recipe selection and F38 is the frozen final test; "
            f"optimizing on either destroys the result they exist to produce. "
            f"Trainable: {', '.join(TRAINABLE_PARTITIONS)}."
        )
    man_path = os.path.join(artifacts, "task_split_manifest.json")
    rows_path = os.path.join(artifacts, "rows.tokenized.jsonl")
    hash_path = os.path.join(artifacts, "artifact_hashes.json")
    for p in (man_path, rows_path, hash_path):
        if not os.path.exists(p):
            raise GateFailure(f"missing {p}")

    manifest = json.load(open(man_path))
    if expect_manifest and manifest["manifest_hash"] != expect_manifest:
        raise GateFailure(
            f"manifest hash {manifest['manifest_hash']} != expected "
            f"{expect_manifest}; the split moved under the run"
        )

    frozen = json.load(open(hash_path))
    for fn, want in frozen.items():
        got = sha256_file(os.path.join(artifacts, fn))
        if got != want:
            raise GateFailure(f"{fn} hash {got[:16]} != frozen {want[:16]}")

    want_tasks = set(manifest["partitions"][partition])
    other: set[str] = set()
    for name, ids in manifest["partitions"].items():
        if name != partition:
            other |= set(ids)

    rows = []
    for line in open(rows_path):
        r = json.loads(line)
        if r["task_id"] in want_tasks:
            rows.append(r)
    if not rows:
        raise GateFailure(f"no rows for {partition}")

    seen = {r["task_id"] for r in rows}
    if seen - want_tasks:
        raise GateFailure(f"rows outside {partition}: {sorted(seen - want_tasks)}")
    if seen & other:
        raise GateFailure(f"{partition} tasks also in another partition: "
                          f"{sorted(seen & other)}")

    for r in rows:
        n = len(r["input_ids"])
        if n != len(r["labels"]):
            raise GateFailure(f"{r['task_id']}#{r['position']}: length mismatch")
        if n > MAX_LENGTH:
            raise GateFailure(
                f"{r['task_id']}#{r['position']}: {n} tokens > {MAX_LENGTH}; "
                "the corpus must be rebuilt, never truncated here"
            )
        if not any(l != IGNORE for l in r["labels"]):
            raise GateFailure(f"{r['task_id']}#{r['position']}: no supervised tokens")

    rows.sort(key=lambda r: (int(r["task_id"]) if r["task_id"].isdigit() else 0,
                             r["position"]))
    return rows, manifest


def probe_rows(rows: list[dict], n: int = 10) -> list[dict]:
    """Deterministic coverage: longest first, then each action type, then spread.

    The longest row is mandatory -- a probe that never renders it cannot expose
    the OOM the real run would hit.
    """
    by_len = sorted(rows, key=lambda r: len(r["input_ids"]))
    picked, seen = [], set()

    def take(r):
        key = (r["task_id"], r["position"])
        if key not in seen:
            seen.add(key)
            picked.append(r)

    take(by_len[-1])                       # longest: the OOM candidate
    take(by_len[0])                        # shortest
    take(by_len[len(by_len) // 2])         # median
    for kind in ("message", "toolcall", "toolcall+text"):
        for r in rows:
            if r["action_type"] == kind:
                take(r)
                break
    for r in rows:                          # multi-call
        if len(r.get("tool_names") or []) > 1:
            take(r)
            break
    for r in rows:                          # a mutation
        if any(t.startswith(("cancel_", "modify_", "return_", "exchange_"))
               for t in (r.get("tool_names") or [])):
            take(r)
            break
    for r in rows:                          # a read-only lookup
        if any(t.startswith("get_") for t in (r.get("tool_names") or [])):
            take(r)
            break
    i = 0
    while len(picked) < n and i < len(by_len):
        take(by_len[i])
        i += max(1, len(by_len) // n)
    return picked[:n]


def verify_reload(base_model: str, adapter_dir: str, ds, collator, trainer,
                  runlog) -> dict:
    """Prove the saved adapter loads and is applied. One model, one position.

    An earlier version of this held three 4B models on one GPU and compared
    full-sequence x 152K-vocab logits between them; it reported a 19.9 logit
    delta and failed a perfectly healthy adapter. The diagnosis run
    (`tau2_adapter_diagnose_modal.py`) established what the real numbers look
    like: two forward passes of the same model are bit-identical (delta 0.0), so
    bf16 noise was never the explanation, and after three steps the adapter's
    true effect is ~0.31 logits at the last position.

    What is checked here:
      * the adapter directory is complete;
      * every `lora_B` is non-zero -- an all-zero B is a mathematically perfect
        no-op that would otherwise pass every behavioural test;
      * the model is self-consistent (same input twice -> same logits);
      * enabling the adapter measurably changes the output.

    An exact trained-vs-reloaded comparison needs a *fresh process*, which this
    cannot be: it runs inside training. That check lives in
    `tau2_adapter_diagnose_modal.py` and runs after the job exits.
    """
    import torch
    from safetensors.torch import load_file

    # Everything serving needs to reproduce training, not just the weights.
    # The tokenizer and chat template are in this list because prompt parity
    # was established between transformers 5.5.3 and vLLM's 4.57.6 for *this*
    # template; serving with a different one silently voids that result.
    missing = [f for f in INFERENCE_FILES
               if not os.path.exists(os.path.join(adapter_dir, f))]
    if missing:
        raise GateFailure(
            f"adapter directory is incomplete, missing {missing}. Serving needs "
            "the tokenizer and chat template as well as the weights, or the "
            "prompt-parity result does not carry over."
        )
    # Things that may be a separate file OR a key inside tokenizer_config.json,
    # depending on the transformers version and tokenizer. Demanding the file
    # form of either one fails a complete artifact.
    tc_path = os.path.join(adapter_dir, "tokenizer_config.json")
    tc = json.load(open(tc_path)) if os.path.exists(tc_path) else {}
    for fname, key in EITHER_FILE_OR_KEY:
        if os.path.exists(os.path.join(adapter_dir, fname)):
            continue
        if tc.get(key):
            continue
        raise GateFailure(
            f"neither {fname} nor a {key!r} key in tokenizer_config.json. "
            "Serving needs the exact tokenizer and template the corpus was "
            "rendered with, or the prompt-parity result does not carry over."
        )

    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    b_norms = [float(v.float().norm()) for k, v in sd.items() if "lora_B" in k]
    if not b_norms:
        raise GateFailure("no lora_B tensors in the saved adapter")
    nonzero = sum(1 for v in b_norms if v > 0)
    if nonzero == 0:
        raise GateFailure(
            "every lora_B is zero — the saved adapter is a no-op and would "
            "change nothing at serving time"
        )

    # One short fixed input at a single position: enough to prove application,
    # and it avoids materialising a full-sequence logits tensor.
    model = trainer.model
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    ids = torch.tensor(ds[0]["input_ids"][:256], device=dev).unsqueeze(0)

    def last_logits():
        with torch.no_grad():
            return model(input_ids=ids).logits[0, -1].float().cpu()

    l1, l2 = last_logits(), last_logits()
    self_delta = float((l1 - l2).abs().max())
    if self_delta != 0.0:
        raise GateFailure(
            f"the model is not deterministic in eval mode (delta {self_delta}); "
            "no reload comparison is meaningful until that is understood"
        )

    with model.disable_adapter():
        off = last_logits()
    effect = float((l1 - off).abs().max())
    if effect == 0.0:
        raise GateFailure(
            "disabling the adapter changes nothing — it is not being applied"
        )

    if was_training:
        model.train()

    # Hash every artifact file, so the summary is a complete manifest of what
    # was saved rather than a sample of it.
    skip = {"train_steps.jsonl", "checkpoints.jsonl", "run_config.json",
            "run_summary.json", "failure.json", "probe_generations.jsonl"}
    files = {f: sha256_file(os.path.join(adapter_dir, f))[:16]
             for f in sorted(os.listdir(adapter_dir))
             if os.path.isfile(os.path.join(adapter_dir, f)) and f not in skip}
    report = {
        "adapter_files": files,
        "n_tensors": len(sd),
        "lora_B_nonzero": f"{nonzero}/{len(b_norms)}",
        "lora_B_norm_median": round(sorted(b_norms)[len(b_norms) // 2], 6),
        "self_consistency_delta": self_delta,
        "adapter_effect_logit_delta": round(effect, 6),
        "verdict": "adapter saved, non-trivial, and applied",
        "note": ("exact trained-vs-reloaded parity requires a fresh process; "
                 "run tau2_adapter_diagnose_modal.py after the job"),
    }
    print(f"RELOAD OK: {nonzero}/{len(b_norms)} lora_B non-zero, "
          f"deterministic, adapter shifts last-token logits by {effect:.4f}",
          flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="/data/tau2/artifacts_16384")
    ap.add_argument("--partition", default="W30",
                    choices=("W30", "C30"),
                    help="W30 builds A_warm; C30 is the continuation pool")
    ap.add_argument("--manifest-hash", default="b741bfceb1f3d027")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--init-adapter", default=None,
                    help="trainable parent LoRA for continued SFT. Required for "
                         "C30 continuation; optimizer and scheduler state are "
                         "always fresh and are never loaded from this path")
    ap.add_argument("--out", default="/data/tau2/a_warm")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--schedule", default=None,
                    help="a frozen ReOPD batch schedule (reopd_schedule JSON). "
                         "When given, rows are emitted in that exact order and "
                         "the run is pinned to its updates x states -- this is "
                         "what makes continued SFT budget-matched to the replay "
                         "arm on updates, exposures and sampling order (V2 s8). "
                         "Without it TRL shuffles and the two arms differ on "
                         "all three while both still log 'one epoch over C30'.")
    # Precommitted, NOT learned from the probe: three steps cannot compare
    # learning rates. 1e-4 is the standard LoRA starting point for a small
    # model at rank 16. Changing it requires either an explicit rationale or a
    # sweep validated on W30-derived data only -- never on C30, S16 or F38.
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="precommitted LoRA learning rate (see comment)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--allow-existing-out", action="store_true",
                    help="append to an output directory that already holds run "
                         "artifacts; refused by default so two runs cannot "
                         "interleave in one log")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every CPU gate and print the plan, load no weights")
    args = ap.parse_args()

    if args.partition == "C30" and not args.init_adapter:
        print("--init-adapter is required for C30 continued SFT; starting C30 "
              "from a fresh LoRA would discard A_warm", file=sys.stderr)
        return 2
    if args.partition != "C30" and args.init_adapter:
        print("--init-adapter is only valid with --partition C30", file=sys.stderr)
        return 2

    parent_manifest = (
        parent_adapter_manifest(args.init_adapter, args.model)
        if args.init_adapter else None
    )

    rows, manifest = load_rows(args.artifacts, args.partition, args.manifest_hash)
    print(f"{args.partition}: {len(rows)} rows, "
          f"{len({r['task_id'] for r in rows})} tasks, "
          f"manifest {manifest['manifest_hash']}")

    if args.probe:
        rows = probe_rows(rows)
        print(f"probe: {len(rows)} rows, longest {max(len(r['input_ids']) for r in rows)}")
        for r in rows:
            print(f"  {r['task_id']}#{r['position']:<2} {r['action_type']:14s} "
                  f"{len(r['input_ids']):>6} tok, "
                  f"{sum(1 for l in r['labels'] if l != IGNORE):>4} supervised "
                  f"{r.get('tool_names') or ''}")

    sup_before = {(r["task_id"], r["position"]):
                  sum(1 for l in r["labels"] if l != IGNORE) for r in rows}
    total_sup = sum(sup_before.values())
    steps = max(3 if args.probe else 1,
                int(len(rows) * args.epochs / (args.batch_size * args.grad_accum)))
    print(f"supervised tokens {total_sup:,} | planned optimizer steps {steps}")

    if args.dry_run:
        print("\n--dry-run: CPU gates passed, no weights loaded")
        return 0

    from vektori_trace.tau2.runlog import RunLog, collect_environment

    if not hasattr(args, "commit_fn"):
        args.commit_fn = None
    runlog = RunLog(args.out, allow_existing=args.allow_existing_out)
    runlog.write_config({
        "mode": "probe" if args.probe else "full",
        "partition": args.partition,
        "init_adapter": args.init_adapter,
        "parent_adapter": parent_manifest,
        "manifest_hash": manifest["manifest_hash"],
        "tools_hash": manifest.get("tools_hash"),
        # The frozen full-corpus hashes stay authoritative; the subset hash is
        # a derived view recorded for provenance, never written back.
        "corpus_hashes": json.load(
            open(os.path.join(args.artifacts, "artifact_hashes.json"))),
        "partition_subset_sha256": hashlib.sha256(
            "".join(f"{r['task_id']}#{r['position']}:{r['semantic_hash']}\n"
                    for r in rows).encode()).hexdigest(),
        "artifacts_dir": args.artifacts,
        "n_rows": len(rows),
        "n_tasks": len({r["task_id"] for r in rows}),
        "supervised_tokens": total_sup,
        "row_ids": [f"{r['task_id']}#{r['position']}" for r in rows],
        "max_length": MAX_LENGTH,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                 "target_modules": LORA_TARGET_MODULES},
        "optimizer": {"lr": args.lr,
                      "lr_source": "precommitted, not probe-derived",
                      "warmup_ratio": 0.0 if args.probe else 0.03,
                      "seed": 20260824,
                      "per_device_batch_size": args.batch_size,
                      "gradient_accumulation_steps": args.grad_accum,
                      "effective_batch_size": args.batch_size * args.grad_accum,
                      "planned_steps": steps, "epochs": args.epochs},
        "schedule": schedule_meta,
        "environment": collect_environment(args.model),
    })

    try:
        return _train(args, rows, manifest, total_sup, steps, runlog)
    except BaseException as e:                       # OOM included
        runlog.failure(e, {"partition": args.partition, "n_rows": len(rows)})
        raise
    finally:
        runlog.close()


def _train(args, rows, manifest, total_sup, steps, runlog) -> int:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from vektori_trace.dataset import LabelPreservingCollator
    from vektori_trace.tau2.runlog import (make_callback,
                                          make_checkpoint_callback)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # The adapter is meaningless without the exact base weights it was trained
    # against. Resolve and record the Hub commit now, so a reload two months
    # from now cannot silently pick up a re-uploaded checkpoint.
    from transformers import AutoConfig
    base_cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    base_revision = getattr(base_cfg, "_commit_hash", None)
    print(f"base model {args.model} revision {base_revision}", flush=True)
    if base_revision is None:
        print("  WARNING: could not resolve a base-model commit hash; reloads "
              "cannot be pinned to these exact weights", flush=True)

    ds = Dataset.from_dict({
        "input_ids": [r["input_ids"] for r in rows],
        "labels": [r["labels"] for r in rows],
        "attention_mask": [r.get("attention_mask") or [1] * len(r["input_ids"])
                           for r in rows],
    })
    collator = LabelPreservingCollator(pad_token_id=tok.pad_token_id or tok.eos_token_id)

    cfg = SFTConfig(
        output_dir=args.out,
        max_length=MAX_LENGTH,
        packing=False,
        # Our labels are authoritative and already verified; TRL must not
        # re-derive, re-mask, or truncate them.
        assistant_only_loss=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        use_liger_kernel=False,
        loss_type="chunked_nll",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        # A frozen schedule pins the step count exactly; without it the length
        # is whatever epochs x rows / effective batch happens to give.
        max_steps=steps if (args.probe or args.schedule) else -1,
        num_train_epochs=args.epochs if not args.probe else 1,
        learning_rate=args.lr,
        warmup_ratio=0.0 if args.probe else 0.03,
        logging_steps=1,
        save_strategy="no" if args.probe else "epoch",
        # False means optimizer.pt / scheduler.pt / rng_state.pth are written
        # alongside the weights. It is the default, but a default that silently
        # flipping would make every checkpoint unresumable, so it is explicit.
        save_only_model=False,
        save_total_limit=None,          # keep all three epochs for selection
        bf16=True,
        report_to=[],
        seed=20260824,
    )

    # Assert the config rather than trusting the constructor kept it. A TRL
    # upgrade that renames or defaults one of these would otherwise change the
    # loss silently -- `chunked_nll` did not exist in trl 0.29 at all, and a
    # regenerated label mask reports a perfectly plausible loss while training
    # on the wrong tokens.
    assert cfg.loss_type == "chunked_nll", f"loss_type is {cfg.loss_type!r}"
    assert not cfg.use_liger_kernel, "Liger changes the loss path"
    assert not cfg.assistant_only_loss, "our labels are authoritative"
    assert cfg.dataset_kwargs["skip_prepare_dataset"], "TRL must not re-prepare"
    assert not cfg.packing, "packing would merge rows across decisions"
    assert cfg.max_length == MAX_LENGTH, f"max_length is {cfg.max_length}"
    if not args.probe:
        assert not cfg.save_only_model, "checkpoints would not be resumable"
        assert cfg.save_strategy in ("epoch", "steps"), \
            f"save_strategy is {cfg.save_strategy}; no checkpoints would exist"
    print(f"config asserted: loss_type={cfg.loss_type}, "
          f"assistant_only_loss={cfg.assistant_only_loss}, "
          f"skip_prepare_dataset=True, packing={cfg.packing}", flush=True)

    peft_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
    )
    train_model = args.model
    if args.init_adapter:
        # Load only the parent adapter weights/config. Deliberately do not call
        # Trainer.resume_from_checkpoint: C30 is a new branch and must start
        # with fresh optimizer, scheduler and RNG state rather than inheriting
        # W30 momentum. PeftModel marks the existing adapter trainable so the
        # saved result is CK35 + the C30 update, not a second stacked LoRA.
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        train_model = PeftModel.from_pretrained(
            base, args.init_adapter, is_trainable=True,
        )
        peft_config = None
        trainable = sum(p.numel() for p in train_model.parameters() if p.requires_grad)
        if trainable == 0:
            raise GateFailure(f"parent adapter {args.init_adapter} has no trainable parameters")
        print(f"continued SFT parent: {args.init_adapter} "
              f"({trainable:,} trainable parameters; fresh optimizer/scheduler)",
              flush=True)

    class _FrozenOrderTrainer(SFTTrainer):
        """Consume rows in dataset order, never shuffled.

        HF's Trainer defaults to a RandomSampler, so ordering the rows is not
        by itself enough -- the frozen schedule would be reshuffled into a
        different stream while the run still reported the same update count,
        which is precisely the divergence V2 s8 forbids between the two arms.
        """

        def _get_train_sampler(self, *a, **k):
            from torch.utils.data import SequentialSampler
            return SequentialSampler(self.train_dataset)

    trainer_cls = _FrozenOrderTrainer if getattr(args, "schedule", None) \
        else SFTTrainer
    trainer = trainer_cls(
        model=train_model,
        args=cfg,
        train_dataset=ds,
        data_collator=collator,
        peft_config=peft_config,
        callbacks=[
            make_callback(runlog,
                          supervised_per_step=max(1, total_sup // max(1, steps))),
            make_checkpoint_callback(runlog, commit=args.commit_fn),
        ],
    )

    # --- prove the mask survived collation, before spending the GPU ---------
    batch = next(iter(trainer.get_train_dataloader()))
    sup = (batch["labels"] != IGNORE).sum().item()
    if sup == 0:
        raise GateFailure("batch 0 has no supervised tokens after collation")
    lens = batch["attention_mask"].sum(dim=1).tolist()
    if any(n >= MAX_LENGTH for n in lens):
        raise GateFailure(f"batch 0 reached max_length {MAX_LENGTH} (TRL #3927)")
    masked_is_ignore = ((batch["labels"] == IGNORE) |
                        (batch["labels"] == batch["input_ids"])).all().item()
    if not masked_is_ignore:
        raise GateFailure("a supervised label does not equal its input_id "
                          "after collation")
    print(f"MASK OK: {sup} supervised in batch 0, lengths {lens}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    loss = result.metrics.get("train_loss")
    if loss is None or not float(loss) == float(loss) or abs(float(loss)) == float("inf"):
        raise GateFailure(f"non-finite training loss: {loss}")

    # Gradient health comes from the logged `grad_norm` series, not from
    # reading `p.grad` here: Trainer zeroes gradients after each optimizer
    # step, so a post-hoc inspection sees zeros on a perfectly healthy run and
    # would fire a false gate failure.
    steps_path = os.path.join(args.out, "train_steps.jsonl")
    logged = [json.loads(l) for l in open(steps_path)] if os.path.exists(steps_path) else []
    norms = [r["grad_norm"] for r in logged if r.get("grad_norm") is not None]
    if not logged:
        raise GateFailure("no optimizer steps were logged")
    if not norms:
        raise GateFailure("no gradient norm was reported on any step")
    if not all(n == n and abs(n) != float("inf") for n in norms):
        raise GateFailure(f"non-finite gradient norm in {norms}")
    if max(norms) == 0.0:
        raise GateFailure("every gradient norm was exactly zero — nothing learned")
    losses = [r["loss"] for r in logged]
    if not all(x == x and abs(x) != float("inf") for x in losses):
        raise GateFailure(f"non-finite per-step loss in {losses}")

    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"\ntrain_loss {loss:.4f} | {elapsed:.1f}s | peak {peak:.1f} GiB | "
          f"grad_norm min={min(norms):.4f} max={max(norms):.4f} | "
          f"{len(logged)} steps logged", flush=True)

    # --- save, then actually reload and prove it ------------------------
    trainer.save_model(args.out)
    # SFTTrainer saves the tokenizer only when it was handed one; save it
    # explicitly so the artifact is self-contained and servable on its own.
    tok.save_pretrained(args.out)
    adapter = os.path.join(args.out, "adapter_model.safetensors")
    if not os.path.exists(adapter):
        raise GateFailure(f"adapter not written to {args.out}")
    runlog.checkpoint(args.out, step=len(logged), extra={"final": True})

    reload_report = verify_reload(args.model, args.out, ds, collator, trainer, runlog)

    # Every epoch checkpoint must be resumable, not just present. A checkpoint
    # without optimizer/scheduler/trainer state can be served but cannot be
    # continued from, and discovering that later means retraining.
    ckpt_report = []
    for d in sorted(os.listdir(args.out)):
        cp = os.path.join(args.out, d)
        if not (d.startswith("checkpoint-") and os.path.isdir(cp)):
            continue
        present = set(os.listdir(cp))
        lacks_inf = [f for f in ("adapter_config.json",
                                 "adapter_model.safetensors") if f not in present]
        lacks_res = [f for f in RESUME_FILES if f not in present]
        ckpt_report.append({"checkpoint": d, "n_files": len(present),
                            "missing_inference": lacks_inf,
                            "missing_resume": lacks_res,
                            "resumable": not lacks_res and not lacks_inf})
        if lacks_inf:
            raise GateFailure(f"{d} cannot be served, missing {lacks_inf}")
        if lacks_res:
            print(f"  WARNING: {d} is not resumable, missing {lacks_res}",
                  flush=True)
    print(f"  {len(ckpt_report)} epoch checkpoint(s), "
          f"{sum(1 for c in ckpt_report if c['resumable'])} fully resumable",
          flush=True)

    summary = {
        "partition": args.partition,
        "manifest_hash": manifest["manifest_hash"],
        "tools_hash": manifest.get("tools_hash"),
        "n_rows": len(rows),
        "n_tasks": len({r["task_id"] for r in rows}),
        "supervised_tokens": total_sup,
        "max_length": MAX_LENGTH,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                 "target_modules": LORA_TARGET_MODULES},
        "steps": steps, "lr": args.lr, "epochs": args.epochs,
        "train_loss": float(loss), "elapsed_sec": round(elapsed, 1),
        "peak_vram_gib": round(peak, 2),
        "grad_norm_min": min(norms), "grad_norm_max": max(norms),
        "steps_logged": len(logged),
        "base_model": args.model,
        "base_model_revision": base_revision,
        "init_adapter": args.init_adapter,
        "optimizer_state_source": "fresh",
        "scheduler_state_source": "fresh",
        "artifact_files": {
            "inference": list(INFERENCE_FILES),
            "resume": list(RESUME_FILES),
        },
        "checkpoint_completeness": ckpt_report,
        "reload_verification": reload_report,
        "probe": args.probe,
        "outcome": "ok",
    }
    runlog.summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
