"""Stage B: continue the Stage A adapter on later actions — Modal, one A100-80GB.

`docs/SFT-SCRATCH-PLAN.md` step 8. A fork of `sft_stage_a_train_modal.py`, which
already owns everything this needs: a pre-tokenized dataset with an explicit
per-row label mask, `chunked_nll`, the fingerprint check, and the OOM report.

Five things differ from Stage A:

  1. **It continues an adapter instead of creating one.** Stage A's
     `checkpoint-84`, loaded `is_trainable=True`. Never v1, never `repaired`,
     never ck63 — those are a different protocol (see the run log). Stage A's
     "every lora_B is zero" assertion is *inverted* here: a continued adapter
     whose B tensors are all zero is not the trained one.
  2. **`max_length` is 40960, not 8192.** The 18 parse-error recoveries deferred
     by amendment 2 sit at turns 13-75; 14 of 18 exceed 8k, max ~33k.
  3. **NF4 is the default arm.** 40960 is the length at which the repair run
     needed it. `--bf16` tries the other arm and is expected to OOM.
  4. **The sampler reads `weight`.** This is the decision the cold-token floor
     was waiting on. The builder solves a per-row weight so cold-start replay
     carries >= 25% of supervised tokens (target 30%); no trainer in this repo
     read it, so uniformly the share was 14.5% and the build refused. A
     `WeightedRandomSampler` over those weights is what makes the number on
     disk the number that trains. The realised share is asserted here, before
     the first step, against the same floor the builder enforces — so the two
     cannot drift apart silently.
  5. **1 epoch at LR 5e-5**, not 4 at 1e-4. Stage A taught a format from
     scratch; Stage B must not overwrite it.

    modal run scripts/sft_stage_b_train_modal.py --probe        # 3 steps, nf4
    modal run scripts/sft_stage_b_train_modal.py --probe --bf16 # other arm
    modal run scripts/sft_stage_b_train_modal.py                # the real run

`--probe` runs on the *longest* rows, not a random sample: peak memory is
dominated by sequence length, and at 40960 that is the whole question.

Every invocation costs GPU time and needs explicit per-run approval (CLAUDE.md).
Tear the app down the moment it returns.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

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

DATA_IN_VOLUME = "sft-stage-b/stage_b.jsonl"
OUT_IN_VOLUME = "sft/qwen3-14b-stage-b-lora"
DEFAULT_SAVE_STEPS = 10
EXPECTED_STAGE_B_DATASET_SHA256 = (
    "c371033d1aa6bfbee9ac3041a0898e83580e0b6bf447bbff2dc9d5d601c805e4"
)
RESUME_REQUIRED_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
)


def validate_resume_checkpoint(
    resume_from: str,
    *,
    volume_mount: str | Path,
    max_steps: int,
    batch_size: int,
) -> tuple[Path, dict[str, Any], int]:
    """Validate a Stage B checkpoint before loading the base model.

    Only checkpoints written beneath this trainer's output directory are legal.
    The directory suffix, Trainer step, total schedule and batch size must all
    agree; accepting a merely checkpoint-shaped directory is how an unrelated
    optimizer state gets applied to the right-shaped LoRA without an error.
    """
    rel = Path(resume_from)
    expected_parent = Path(OUT_IN_VOLUME)
    match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", rel.name)
    if rel.is_absolute() or rel.parent != expected_parent or match is None:
        raise ValueError(
            "resume checkpoint must be "
            f"{OUT_IN_VOLUME}/checkpoint-<step>, got {resume_from!r}"
        )

    path = Path(volume_mount) / rel
    if not path.is_dir():
        raise ValueError(f"resume checkpoint not found: {path}")
    missing = [name for name in RESUME_REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(f"resume checkpoint {path} is incomplete, missing: {missing}")

    try:
        state = json.loads((path / "trainer_state.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Trainer state from {path}: {exc}") from exc

    done = int(state.get("global_step", 0))
    path_step = int(match.group(1))
    if done != path_step:
        raise ValueError(
            f"resume directory says checkpoint-{path_step} but Trainer state "
            f"reports global_step={done}"
        )
    if done >= max_steps:
        raise ValueError(
            f"resume checkpoint is at step {done} but max_steps is {max_steps} — "
            "nothing left to train"
        )
    prior_max = state.get("max_steps")
    if prior_max is None or int(prior_max) != max_steps:
        raise ValueError(
            f"resume checkpoint was built for max_steps={prior_max}, current run "
            f"computes {max_steps}"
        )
    prior_batch = state.get("train_batch_size")
    if prior_batch is None or int(prior_batch) != batch_size:
        raise ValueError(
            f"resume checkpoint used train_batch_size={prior_batch}, current run "
            f"uses {batch_size}"
        )
    return path, state, done


def resume_training_arg_mismatches(
    prior: Any, expected: dict[str, Any]
) -> list[str]:
    """Return training-dynamics changes that make a resume non-equivalent."""
    mismatches = []
    for name, wanted in expected.items():
        got = getattr(prior, name, None)
        if got != wanted:
            mismatches.append(f"{name}: checkpoint={got!r}, current={wanted!r}")
    return mismatches

#: The adapter Stage B continues. `checkpoint-84` is the Stage A selection —
#: 45/45 on the frozen manifest at 4096, step 7 rollout green. Pinned as a
#: constant rather than a flag default so pointing this run at the v1 adapter
#: takes an edit and a code review, not a typo.
BASE_ADAPTER_IN_VOLUME = "sft/qwen3-14b-stage-a-lora/checkpoint-84"

#: Row kind the builder gives replayed Stage A cold starts.
COLD_KIND = "cold_replay"

#: Same floor the builder enforces (`sft_stage_b_dataset.COLD_TOKEN_FLOOR`).
#: Duplicated deliberately: the builder cannot be imported in the container,
#: and a floor that lives in only one of the two can be satisfied on disk and
#: violated in training.
COLD_TOKEN_FLOOR = 0.25

# Plan step 8. The recoveries reach ~33k tokens, so this is not headroom, it is
# the requirement. `tokenize_row` still refuses anything longer rather than
# truncating.
MAX_LENGTH = 40960

# Stage B does not construct a LoraConfig — it continues one. These are what
# Stage A wrote, and what the loaded adapter is checked against: a shape
# mismatch here means the wrong adapter is on the volume path, which is the one
# way this run could silently become something else.
EXPECTED_LORA_R = 32
EXPECTED_LORA_ALPHA = 64

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

app = modal.App("vektori-trace-sft-stage-b")
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
    nf4: bool = True,
    memory_history: bool = False,
    model: str = "Qwen/Qwen3-14B",
    epochs: float = 1.0,
    lr: float = 5e-5,
    batch_size: int = 1,
    grad_accum: int = 8,
    save_steps: int = DEFAULT_SAVE_STEPS,
    seed: int = 0,
    resume_from: str = "",
) -> dict:
    import hashlib
    import json
    import time

    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
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
    # Kept outside the Dataset on purpose. `weight` is a property of the
    # *sampler*, not of a training example; as a column it would ride into the
    # collator and from there into the model's forward as an unexpected kwarg.
    weights: list[float] = []
    kinds: list[str] = []
    for i, row in enumerate(rows):
        ex = tokenize_row(row, tokenizer)
        if ex is None:
            raise SystemExit(
                f"row {i} produced no supervised example — the Stage B builder "
                "should have caught this; refusing to train on a set it did not "
                "validate"
            )
        examples.append(ex)
        w = row.get("weight")
        if w is None or not isinstance(w, (int, float)) or w <= 0:
            raise SystemExit(
                f"row {i} carries weight {w!r}; every row needs a positive "
                "weight or the sampler silently falls back to uniform, which is "
                "the 14.5% cold share the builder refuses"
            )
        weights.append(float(w))
        kinds.append(row.get("kind", "?"))
    ds = Dataset.from_list(examples)

    supervised = sum(sum(1 for x in e["labels"] if x != IGNORE_INDEX) for e in examples)
    total = sum(len(e["input_ids"]) for e in examples)
    print(f"MASK: {supervised}/{total} tokens supervised "
          f"({100 * supervised / total:.1f}%) across ALL {len(examples)} rows", flush=True)
    if supervised == 0:
        raise SystemExit("no supervised tokens in the whole dataset")

    # ---- the cold floor, as the sampler will actually realise it -----------
    # Expected supervised tokens per draw, split by kind. This is the number the
    # plan's fail-closed line is about: not what the file contains, but what the
    # gradient stream sees. Computed here rather than trusted from mix_report
    # because that report describes a dataset, and this describes this run.
    per_row_sup = [sum(1 for x in e["labels"] if x != IGNORE_INDEX) for e in examples]
    weighted_share, uniform_share = cold_shares(weights, per_row_sup, kinds)
    cold_rows = sum(1 for k in kinds if k == COLD_KIND)
    print(
        f"COLD: {cold_rows} rows, supervised-token share "
        f"{weighted_share:.1%} weighted / {uniform_share:.1%} uniform "
        f"(floor {COLD_TOKEN_FLOOR:.0%})", flush=True
    )
    if weighted_share < COLD_TOKEN_FLOOR:
        raise SystemExit(
            f"cold share is {weighted_share:.1%} under this run's own sampler, "
            f"floor is {COLD_TOKEN_FLOOR:.0%} — Stage B would drift off the "
            "format Stage A just taught. Rebuild the mix; do not train."
        )

    # `tokenize_row` is a copy of `dataset.tokenize_messages`, not a call to it —
    # Modal re-imports this module in a container without the local package. A
    # copy can drift, so the builder records what the real function produced per
    # row and this refuses to train on any disagreement.
    fp_path = data_path.parent / "tokenization_fingerprint.json"
    if not fp_path.exists():
        raise SystemExit(
            f"{fp_path} is missing — build with scripts/sft_stage_b_dataset.py "
            "(--sampler weighted), verify with scripts/sft_stage_b_verify.py, and "
            "stage the directory onto the volume before training"
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
    if data_sha != EXPECTED_STAGE_B_DATASET_SHA256:
        raise SystemExit(
            f"Stage B dataset is {data_sha}, expected the frozen plan dataset "
            f"{EXPECTED_STAGE_B_DATASET_SHA256} — a matching self-reported "
            "fingerprint is not enough to resume a different run"
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

    steps_per_epoch = -(-len(ds) // (batch_size * grad_accum))
    max_steps = 3 if probe else int(steps_per_epoch * epochs)
    out_dir = Path(VOLUME_MOUNT) / OUT_IN_VOLUME
    resume_path: Path | None = None
    resume_state: dict[str, Any] | None = None
    resume_step = 0
    resume_arg: str | None = None
    if resume_from:
        if probe:
            raise SystemExit("--resume-from is meaningless with --probe: a probe saves nothing")
        if not nf4:
            raise SystemExit(
                "the interrupted Stage B checkpoint was trained with NF4; "
                "refusing to resume it with --bf16"
            )
        try:
            resume_path, resume_state, resume_step = validate_resume_checkpoint(
                resume_from,
                volume_mount=VOLUME_MOUNT,
                max_steps=max_steps,
                batch_size=batch_size,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        # `training_args.bin` is our own trusted checkpoint artifact. Trainer
        # otherwise warns about only a small subset of mismatches and continues;
        # gradient accumulation, LR or warmup drift would change the resumed run.
        prior_args = torch.load(
            resume_path / "training_args.bin", map_location="cpu", weights_only=False
        )
        dynamics = {
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": lr,
            "seed": seed,
            "max_steps": max_steps,
            "warmup_ratio": 0.03,
            "max_grad_norm": 1.0,
        }
        mismatches = resume_training_arg_mismatches(prior_args, dynamics)
        if mismatches:
            raise SystemExit(
                "resume changes training dynamics: " + "; ".join(mismatches)
            )
        resume_arg = str(resume_path)
        print(
            f"RESUMING from {resume_path}: global_step {resume_step}/{max_steps}, "
            f"{max_steps - resume_step} steps remain; checkpoint saves every "
            f"{save_steps} steps",
            flush=True,
        )
        print(
            f"  prior epoch {resume_state.get('epoch')}, "
            f"logged history entries {len(resume_state.get('log_history', []))}",
            flush=True,
        )
    else:
        print(f"fresh run from {BASE_ADAPTER_IN_VOLUME} (no resume)", flush=True)

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

    # The one line that makes this Stage B: the adapter comes off the volume,
    # trainable, and is Stage A's selected checkpoint.
    adapter_path = Path(VOLUME_MOUNT) / BASE_ADAPTER_IN_VOLUME
    if not (adapter_path / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {adapter_path} — nothing to continue")
    cfg_on_disk = json.loads((adapter_path / "adapter_config.json").read_text())
    print(f"continuing {adapter_path} (r={cfg_on_disk.get('r')}, "
          f"alpha={cfg_on_disk.get('lora_alpha')}, "
          f"target={cfg_on_disk.get('target_modules')})", flush=True)
    # v1 and the repaired adapter are a different protocol entirely (they emit
    # the tool_call envelope). Continuing one of those by mistake would look
    # like a normal run and undo everything Stage A measured.
    for forbidden in ("dsv4", "repaired", "checkpoint-63"):
        if forbidden in str(adapter_path):
            raise SystemExit(
                f"{adapter_path} looks like the v1 lineage ({forbidden!r}) — "
                "Stage B continues Stage A, never v1"
            )
    if (cfg_on_disk.get("r"), cfg_on_disk.get("lora_alpha")) != (
        EXPECTED_LORA_R, EXPECTED_LORA_ALPHA
    ):
        raise SystemExit(
            f"adapter at {adapter_path} is r={cfg_on_disk.get('r')} "
            f"alpha={cfg_on_disk.get('lora_alpha')}, expected "
            f"r={EXPECTED_LORA_R} alpha={EXPECTED_LORA_ALPHA} — this is not the "
            "Stage A adapter"
        )
    net = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)

    adapters = list(net.peft_config)
    if adapters != ["default"]:
        raise SystemExit(f"expected exactly one adapter, found {adapters}")
    trainable = [n for n, p in net.named_parameters() if p.requires_grad]
    if not trainable:
        raise SystemExit(
            "nothing is trainable — the continued adapter loaded frozen "
            "(is_trainable=True is what makes a continuation trainable)"
        )
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
    # Stage A asserts every lora_B is zero, because a fresh adapter must be
    # exactly the base model at step 0. Stage B asserts the *opposite*: an
    # all-zero B means what loaded is untrained, so this would be Stage A again
    # at Stage B's learning rate rather than a continuation.
    b_tensors = [n for n, p in net.named_parameters() if "lora_B" in n]
    nonzero_b = [
        n for n, p in net.named_parameters()
        if "lora_B" in n and bool(p.detach().any())
    ]
    if not b_tensors:
        raise SystemExit("no lora_B tensors — nothing was loaded")
    if not nonzero_b:
        raise SystemExit(
            f"all {len(b_tensors)} lora_B tensors are zero — the adapter at "
            f"{adapter_path} is untrained, so this is not a continuation"
        )
    print(f"continued adapter: {len(nonzero_b)}/{len(b_tensors)} lora_B tensors "
          f"carry trained weight", flush=True)

    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"continued LoRA: {len(trainable)} tensors, {n_trainable:,} trainable "
          f"params, r={cfg_on_disk.get('r')} alpha={cfg_on_disk.get('lora_alpha')}",
          flush=True)

    # LoRA freezes the base weights, so without this the checkpointed
    # activations have nothing requiring grad and the backward pass is empty —
    # a run that reports a plausible loss and learns nothing.
    net.enable_input_require_grads()

    mark("adapter_loaded")
    # On resume, Trainer has not loaded checkpoint-50's adapter yet. Snapshot
    # in `on_train_begin`, after `_load_from_checkpoint` but before step 51, so
    # the moved check measures only work performed by this invocation.
    before = None if resume_arg else _snapshot_trainable(net)

    if probe:
        # Trainer's sampler is shuffled, so three steps may never touch the
        # longest row — and peak VRAM is dominated by sequence length. A probe
        # that measured only median rows would certify a footprint the real run
        # then exceeds.
        picked = _longest_first(examples, n=batch_size * grad_accum * max_steps)
        ds = ds.select(picked)
        # The sampler is indexed by dataset position, so the weight vector has
        # to follow the same selection or it addresses rows that are gone.
        weights = [weights[i] for i in picked]
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
        def on_train_begin(self, args, state, control, **kw):
            nonlocal before
            if before is None:
                before = _snapshot_trainable(net)
                print(
                    f"resume baseline captured after loading global step "
                    f"{state.global_step}",
                    flush=True,
                )

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            if "grad_norm" in logs:
                grad_norms.append(float(logs["grad_norm"]))
            if "loss" in logs:
                losses.append(float(logs["loss"]))

    # The sampler is the whole point of Stage B's mix. Trainer builds its
    # dataloader from `_get_train_sampler`, so overriding that is the seam —
    # and an upstream rename would silently restore uniform sampling, which is
    # the 14.5% cold share. Asserted below rather than assumed.
    if not hasattr(SFTTrainer, "_get_train_sampler"):
        raise SystemExit(
            "SFTTrainer has no _get_train_sampler — the sampler seam moved; "
            "find it before training, or the mix on disk is not the mix trained"
        )

    class _WeightedSFTTrainer(SFTTrainer):
        """Draw rows in proportion to `weight`, with replacement.

        The builder solves those weights so cold-start replay carries the plan's
        supervised-token share. Uniform shuffling ignores them and lands at
        14.5%, under the 25% floor — the reason the Stage B build refused
        without `--allow-unweighted`. `num_samples` is the row count, so one
        "epoch" is one dataset-sized set of draws, matching the plan's
        draws-per-epoch units.
        """

        def _get_train_sampler(self, *a, **kw):
            from torch.utils.data import WeightedRandomSampler

            gen = torch.Generator()
            gen.manual_seed(self.args.seed)
            return WeightedRandomSampler(
                weights=row_weights,
                num_samples=len(row_weights),
                replacement=True,
                generator=gen,
            )

    row_weights = list(weights)
    if len(row_weights) != len(ds):
        raise SystemExit(
            f"{len(row_weights)} weights for {len(ds)} rows — the sampler would "
            "address the wrong rows"
        )

    trainer = _WeightedSFTTrainer(
        model=net,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        # Only replaced when None (sft_trainer.py:1190); refused only under
        # padding_free, which is off.
        data_collator=label_preserving_collator(tokenizer.pad_token_id),
        callbacks=[_Collect()],
    )

    from torch.utils.data import WeightedRandomSampler as _WRS

    got_sampler = trainer._get_train_sampler(ds)
    if not isinstance(got_sampler, _WRS):
        raise SystemExit(
            f"the trainer's sampler is {type(got_sampler).__name__}, not "
            "WeightedRandomSampler — the mix on disk is not the mix that trains"
        )
    # `_get_train_sampler` returning the right object is not proof the loader
    # Accelerate prepares uses it: `prepare` can replace the sampler when it
    # shards. Ask the real dataloader, which is the thing that feeds the steps.
    #
    # Accelerate's `DataLoaderShard` always sets `.sampler` to a `SequentialSampler`
    # placeholder it uses for its own epoch/state bookkeeping — that attribute is
    # never None, so a `seen is None` fallback never fires and this used to check
    # the decoy. The sampler that actually draws batches is `.batch_sampler.sampler`
    # whenever a `batch_sampler` is present (verified off-GPU: draws show repeats
    # and non-sequential order, which only a WeightedRandomSampler produces).
    loader = trainer.get_train_dataloader()
    batch_sampler = getattr(loader, "batch_sampler", None)
    seen = getattr(batch_sampler, "sampler", None) if batch_sampler is not None \
        else getattr(loader, "sampler", None)
    if not isinstance(seen, _WRS):
        raise SystemExit(
            f"the prepared dataloader samples with {type(seen).__name__}, not "
            "WeightedRandomSampler — Accelerate replaced it, so the cold share "
            "measured above is not what will train"
        )
    print(f"dataloader sampler: {type(seen).__name__} (checked on the prepared "
          "loader, not just the factory)", flush=True)
    # What the sampler actually drew, not what it was asked for. A realised cold
    # count far from the expectation means the weight vector is not doing what
    # the mix report claims.
    drawn = list(iter(trainer._get_train_sampler(ds)))
    drawn_cold = sum(1 for i in drawn if kinds[i] == COLD_KIND) if not probe else None
    print(f"sampler: WeightedRandomSampler over {len(row_weights)} rows, "
          f"{len(drawn)} draws/epoch, cold draws {drawn_cold} "
          f"(plan floor 253, target 325)", flush=True)
    print(f"loss_type={args.loss_type}, use_liger_kernel={args.use_liger_kernel}, "
          f"assistant_only_loss={args.assistant_only_loss}", flush=True)

    if memory_history:
        torch.cuda.memory._record_memory_history(max_entries=100_000)
    mark("pre_train")
    t0 = time.time()
    try:
        result = trainer.train(resume_from_checkpoint=resume_arg)
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
    final_step = int(trainer.state.global_step)
    if final_step != max_steps:
        raise SystemExit(
            f"training returned at global_step={final_step}, expected {max_steps}; "
            "refusing to write a final adapter for an incomplete run"
        )
    executed_steps = final_step - resume_step
    if executed_steps <= 0:
        raise SystemExit(
            f"training executed {executed_steps} new steps "
            f"({resume_step} -> {final_step})"
        )
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    # Printed here, not only inside `summary`. The first bf16 probe measured the
    # peak and then threw in the moved-check below, which sits between this line
    # and the only place the number was reported — so ~6 GPU-minutes produced no
    # measurement at all. Whatever fails after this point, the number is out.
    print(f"PEAK: allocated {peak_alloc:.1f} GiB, reserved {peak_reserved:.1f} GiB "
          f"(ceiling {BF16_PEAK_CEILING_GIB:.0f} GiB, arm "
          f"{'nf4' if nf4 else 'bf16'})", flush=True)

    if before is None:
        raise SystemExit("resume baseline was never captured before training")
    moved = _count_moved(net, before)
    train_loss = result.metrics.get("train_loss")

    def finite(x) -> bool:
        return x is not None and x == x and abs(x) != float("inf")

    summary = {
        "probe": probe,
        "stage": "B",
        "continued_adapter": str(adapter_path),
        "resumed_from": str(resume_path) if resume_path else None,
        "initial_global_step": resume_step,
        "final_global_step": final_step,
        "steps_executed": executed_steps,
        "sampler": {
            "kind": "WeightedRandomSampler",
            "draws_per_epoch": len(row_weights),
            "cold_draws": drawn_cold,
            "cold_token_share_weighted": round(weighted_share, 4),
            "cold_token_share_uniform": round(uniform_share, 4),
            "cold_floor": COLD_TOKEN_FLOOR,
        },
        "precision": "nf4" if nf4 else "bf16",
        "rows": len(ds),
        "max_steps": max_steps,
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "max_length": MAX_LENGTH,
        "lora": {
            "r": cfg_on_disk.get("r"),
            "alpha": cfg_on_disk.get("lora_alpha"),
            "dropout": cfg_on_disk.get("lora_dropout"),
            "target_modules": cfg_on_disk.get("target_modules"),
            "fresh": False,
        },
        "elapsed_sec": round(elapsed, 1),
        # Reserved, not just allocated: the allocator's reserved pool is what
        # actually has to fit in the card, and it is what OOMs.
        "peak_vram_allocated_gib": round(peak_alloc, 1),
        "peak_vram_reserved_gib": round(peak_reserved, 1),
        "longest_row_tokens": max(len(e["input_ids"]) for e in examples),
        "sec_per_optimizer_step": round(elapsed / executed_steps, 1),
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
    """A path under the Stage B directory on the volume, parents created.

    Failure reports go next to the dataset they were produced from, never into
    the repair run's directory — the two must not be read as one series.
    """
    out = Path(VOLUME_MOUNT) / "sft-stage-b" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def cold_shares(
    weights: list[float],
    supervised: list[int],
    kinds: list[str],
    *,
    cold_kind: str = COLD_KIND,
) -> tuple[float, float]:
    """(weighted, uniform) supervised-token share carried by cold-start rows.

    The weighted figure is what a `WeightedRandomSampler` realises in
    expectation: E[cold tokens per draw] / E[tokens per draw]. The uniform one
    is the raw token ratio, which is what a plain shuffle gives. They differ by
    2x on the real Stage B mix, and the plan's floor is about the first only
    when a trainer actually samples by weight.

    Module level and pure so the number that gates a GPU run has a test.
    """
    if not weights:
        return 0.0, 0.0
    if not (len(weights) == len(supervised) == len(kinds)):
        raise ValueError(
            f"{len(weights)} weights, {len(supervised)} token counts, "
            f"{len(kinds)} kinds — these index the same rows and must align"
        )
    w_total = sum(weights)
    if w_total <= 0:
        raise ValueError("weights sum to zero — nothing would be sampled")
    exp_tok = sum(w * t for w, t in zip(weights, supervised, strict=True)) / w_total
    exp_cold = sum(
        w * t for w, t, k in zip(weights, supervised, kinds, strict=True)
        if k == cold_kind
    ) / w_total
    tok_total = sum(supervised)
    uniform_cold = sum(
        t for t, k in zip(supervised, kinds, strict=True) if k == cold_kind
    )
    return (
        (exp_cold / exp_tok) if exp_tok else 0.0,
        (uniform_cold / tok_total) if tok_total else 0.0,
    )


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
    bf16: bool = False,
    memory_history: bool = False,
    epochs: float = 1.0,
    resume_from: str = "",
):
    """`--bf16` selects the other arm; NF4 is the default at 40960.

    `--resume-from sft/qwen3-14b-stage-b-lora/checkpoint-50` continues an
    interrupted run: Trainer state (optimizer, LR schedule, step counter, RNG)
    is restored from that checkpoint, and the remaining steps run. Leave it
    empty for a fresh run from Stage A's checkpoint-84.
    """
    print(train.remote(
        probe=probe, nf4=not bf16, memory_history=memory_history, epochs=epochs,
        resume_from=resume_from,
    ))
