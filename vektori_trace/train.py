"""LoRA SFT loop for Step 6.

NOT THE ACTIVE TRAINING PATH. Agent-trace SFT runs through TRL —
`scripts/sft_train_modal.py` — which derives the assistant mask from the chat
template rather than from pre-tokenized `labels`. This module is kept as the
fallback and is still used by `arms.py`.

Masking is baked into `labels` (-100) by
`dataset.py`, so the stock HF Trainer causal-LM loss needs no custom loss.

Real runs execute on Modal GPU (`train_lora_modal`); offline unit tests call
`train_lora` locally on a tiny from-scratch model. Modal is not invoked in tests.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .dataset import LabelPreservingCollator, TokenizedExample, build_sft_dataset
from .runtime.modal_env import (
    HF_CACHE_MOUNT,
    HF_CACHE_VOLUME_NAME,
    VOLUME_MOUNT,
    VOLUME_NAME,
    volume_adapter_dir,
)


def _require_train():
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "training extras required: install with `uv sync --extra train` "
            "(or `pip install 'vektori-trace[train]'`)"
        ) from e


def _JsonlLogger(path: Path):
    """TrainerCallback appending every HF log dict to `path` as JSONL.

    Defined inside a function because `TrainerCallback` requires transformers,
    which cli.py must not import at module scope.
    """
    _require_train()
    from transformers import TrainerCallback

    class JsonlLogger(TrainerCallback):
        def __init__(self, path: Path) -> None:
            self.path = path
            self.path.parent.mkdir(parents=True, exist_ok=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            row = {"step": state.global_step, **logs}
            with self.path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")

    return JsonlLogger(path)


@dataclass
class LoraHyperparams:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class TrainConfig:
    base_model: str
    output_dir: Path
    task_ids: list[str]
    max_steps: int = 50
    per_device_train_batch_size: int = 1
    learning_rate: float = 2e-4
    seed: int = 0
    lora: LoraHyperparams = field(default_factory=LoraHyperparams)
    use_modal: bool = True
    modal_gpu: str = "L40S"
    # Distinguishes A2 vs A3 adapters on the shared Volume.
    arm: str | None = None
    # --- memory / throughput -------------------------------------------------
    # bf16 is the default because fp32 does not fit the pilot: Qwen3-8B is ~32GB
    # of weights in fp32 before optimizer state, which OOMs every 24GB card and
    # wastes an 80GB one. Ignored when CUDA is unavailable (offline tests train a
    # tiny model on CPU, where bf16 is slower and less numerically forgiving).
    bf16: bool = True
    # Load the frozen base in 4-bit NF4 (QLoRA). Measured on Modal: Qwen3-8B in
    # bf16 holds ~16.4GB of weights and peaks at ~20.5GB before the loss head is
    # even allocated, so it OOMs a 24GB A10G at 8192 *and* at 4096 context —
    # shortening sequences cannot fix a constant term (docs/vram-probe*.json).
    # 4-bit takes the base to ~5.5GB. LoRA adapters stay bf16 and are what the
    # optimizer actually updates, so the trained parameters are not quantized.
    load_in_4bit: bool = False
    gradient_accumulation_steps: int = 1
    # Trades ~30% step time for a large activation-memory reduction. Off by
    # default so short pilot runs stay fast; turn it on when sequences are long.
    gradient_checkpointing: bool = False
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    # HF keeps every checkpoint by default. Adapters land on a shared, billed
    # volume once per arm per seed, so retain only the newest.
    save_total_limit: int = 1
    # Set False for a self-managed GPU (EC2/local) where Modal is not in play at
    # all. The default preserves the Modal train → Modal serve handoff.
    stage_to_volume: bool = True
    # Optional held-out examples: without them a 29-task corpus cannot show
    # whether the adapter learned the capability or memorised the test output,
    # which is the one thing dataset.py's masking exists to prevent.
    eval_steps: int = 0


@dataclass
class TrainResult:
    adapter_dir: Path
    base_model: str
    task_ids: list[str]
    steps: int
    final_loss: float | None
    lora: dict[str, Any]
    seed: int
    # Path *inside* the Modal Volume (e.g. /adapters/...), for serve_model.
    # None when training locally without staging to the Volume.
    volume_adapter_path: str | None = None


def write_train_report(result: TrainResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "adapter_dir": str(result.adapter_dir),
        "volume_adapter_path": result.volume_adapter_path,
        "base_model": result.base_model,
        "task_ids": result.task_ids,
        "steps": result.steps,
        "final_loss": result.final_loss,
        "lora": result.lora,
        "seed": result.seed,
    }
    json_path = out_dir / "train.json"
    json_path.write_text(json.dumps(report, indent=2))
    lines = [
        "# Vektori-trace LoRA SFT\n",
        f"- base model: `{result.base_model}`\n",
        f"- adapter (local): `{result.adapter_dir}`\n",
        f"- adapter (Modal Volume): `{result.volume_adapter_path or 'n/a'}`\n",
        f"- tasks: {len(result.task_ids)}  ·  steps: {result.steps}  ·  "
        f"final loss: {result.final_loss if result.final_loss is not None else 'n/a'}\n",
        f"- seed: {result.seed}\n",
        f"- LoRA: r={result.lora.get('r')}, alpha={result.lora.get('alpha')}\n",
    ]
    md_path = out_dir / "train.md"
    md_path.write_text("".join(lines))
    return md_path


def train_lora(
    examples: list[TokenizedExample],
    config: TrainConfig,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
    eval_examples: list[TokenizedExample] | None = None,
) -> TrainResult:
    """Run LoRA SFT. Lazy-imports train extras. Writes adapter under output_dir."""
    _require_train()
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    if not examples:
        raise ValueError("no tokenized examples — rejection sampling kept nothing")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = config.output_dir / "adapter"

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bf16 only where it is both supported and a win. `is_bf16_supported()` is
    # False on pre-Ampere cards (e.g. T4/V100 on older EC2 families), where
    # requesting it would either error or silently emulate.
    use_bf16 = bool(
        config.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    # Quantization needs a CUDA device; on the CPU smoke tests it would either
    # error or silently fall back, so it is ignored there rather than honoured
    # halfway.
    use_4bit = bool(config.load_in_4bit and torch.cuda.is_available())
    if model is None:
        quant_kwargs: dict[str, Any] = {}
        if use_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                # Dequantize to bf16 for the matmul. fp32 compute would give
                # back most of the memory 4-bit just bought.
                bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float32,
                # Quantizes the quantization constants too. Small further win,
                # no measurable quality cost at r=16.
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
            **quant_kwargs,
        )
    if use_4bit:
        from peft import prepare_model_for_kbit_training

        # Casts layer norms and the lm_head to fp32 and enables input grads.
        # Without it a 4-bit base trains to NaN or produces an empty backward
        # pass, both of which look like a data bug rather than a setup bug.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
        # ...but it upcasts *every* non-quantized parameter to fp32, and on a
        # 151k-vocab model the two it catches are the embedding and lm_head at
        # ~1.2GB each. The weights are the small half: an fp32 lm_head emits
        # fp32 logits, so at 8192 context the logit tensor and its gradient go
        # 2.32GB -> 4.64GB each. Measured, that made 4-bit peak *higher* than
        # bf16 (docs/vram-stages-*.json). Cast the two big tensors back; the
        # layer norms this upcast exists to stabilise are tiny and stay fp32.
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float32
        for embed in (model.get_input_embeddings(), model.get_output_embeddings()):
            if embed is not None:
                embed.to(compute_dtype)

    target = list(config.lora.target_modules)
    named = {n.split(".")[-1] for n, _ in model.named_modules()}
    if not any(t in named for t in target):
        target = [n for n in ("c_attn", "c_proj", "q_proj", "v_proj") if n in named] or [
            "c_attn"
        ]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=target,
    )
    model = get_peft_model(model, peft_config)

    ds = build_sft_dataset(examples)
    eval_ds = build_sft_dataset(eval_examples) if eval_examples else None
    collator = LabelPreservingCollator(pad_token_id=tokenizer.pad_token_id)

    eval_kwargs: dict[str, Any] = {}
    if eval_ds is not None and config.eval_steps > 0:
        eval_kwargs = {"eval_strategy": "steps", "eval_steps": config.eval_steps}

    args = TrainingArguments(
        output_dir=str(config.output_dir / "hf_runs"),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        # Track the train batch size. HF defaults eval to 8, which on the
        # memory-constrained pilot GPU this config is tuned for (bf16 +
        # gradient checkpointing, train batch of 1) OOMs the first time
        # periodic evaluation fires — minutes into a run that was fine.
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=use_bf16,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        logging_steps=1,
        save_steps=max(config.max_steps, 1),
        save_total_limit=config.save_total_limit,
        report_to=[],
        seed=config.seed,
        remove_unused_columns=False,
        label_names=["labels"],
        **eval_kwargs,
    )
    # A loss curve is the only in-flight evidence that a billed GPU run is
    # working. `report_to=[]` keeps W&B optional; this writes the same series to
    # the run directory so a finished run can be audited without a SaaS account.
    history_path = config.output_dir / "train_log.jsonl"
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[_JsonlLogger(history_path)],
    )
    if config.gradient_checkpointing:
        # LoRA freezes the base weights, so without this the checkpointed
        # activations have nothing requiring grad and the backward pass is empty.
        model.enable_input_require_grads()
    train_out = trainer.train()
    final_loss = None
    if train_out and train_out.metrics:
        final_loss = train_out.metrics.get("train_loss")
        if final_loss is not None:
            final_loss = float(final_loss)
            if final_loss != final_loss:  # NaN
                raise RuntimeError("training loss is NaN — masking/collator bug likely")

    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    return TrainResult(
        adapter_dir=adapter_dir,
        base_model=config.base_model,
        task_ids=list(config.task_ids),
        steps=config.max_steps,
        final_loss=final_loss,
        lora=asdict(config.lora),
        seed=config.seed,
        volume_adapter_path=None,
    )


def train_lora_modal(
    examples: list[TokenizedExample],
    config: TrainConfig,
) -> TrainResult:
    """Execute `train_lora` on a Modal GPU; adapter lands on the shared Volume."""
    _require_train()
    try:
        import modal
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "modal is part of the train extra — `uv sync --extra train`"
        ) from e

    if not examples:
        raise ValueError("no tokenized examples — rejection sampling kept nothing")

    payload = [
        {
            "input_ids": e.input_ids,
            "labels": e.labels,
            "attention_mask": e.attention_mask,
        }
        for e in examples
    ]
    vol_path = volume_adapter_dir(config.base_model, config.seed, arm=config.arm)
    cfg = {
        "base_model": config.base_model,
        "task_ids": config.task_ids,
        "max_steps": config.max_steps,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "lora": asdict(config.lora),
        "modal_gpu": config.modal_gpu,
        "volume_adapter_path": vol_path,
        "arm": config.arm,
        "bf16": config.bf16,
        "load_in_4bit": config.load_in_4bit,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "max_grad_norm": config.max_grad_norm,
        "save_total_limit": config.save_total_limit,
    }

    app = modal.App("vektori-trace-train")
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch", "transformers", "peft", "accelerate", "datasets", "bitsandbytes"
        )
        .env({"HF_HOME": HF_CACHE_MOUNT})
        .add_local_python_source("vektori_trace")
    )

    @app.function(
        gpu=config.modal_gpu,
        image=image,
        # Same HF cache Volume `serve.py` mounts, so the base weights are
        # downloaded once for the whole project rather than once per arm.
        # Training runs one arm after another; without this each arm pays to
        # fetch the same base model on a GPU billed by the second.
        volumes={VOLUME_MOUNT: vol, HF_CACHE_MOUNT: hf_cache},
        timeout=4 * 60 * 60,
        # `_remote` is defined inside this function so it can close over `vol`.
        # Modal re-imports globals-scope functions by module path in the
        # container and rejects locally-defined ones unless they are pickled
        # instead. Without this the decorator raises InvalidError at call time,
        # before any GPU is allocated — the same fix `runtime/serve.py` already
        # carries for its locally-defined class.
        serialized=True,
    )
    def _remote(payload: list[dict], cfg: dict) -> dict:
        from pathlib import Path as P

        from vektori_trace.dataset import TokenizedExample as TE
        from vektori_trace.train import LoraHyperparams, TrainConfig, train_lora

        examples_r = [TE(**row) for row in payload]
        # Write directly into the Volume mount so serve_model can find it.
        out = P(cfg["volume_adapter_path"]).parent
        result = train_lora(
            examples_r,
            TrainConfig(
                base_model=cfg["base_model"],
                output_dir=out,
                task_ids=cfg["task_ids"],
                max_steps=cfg["max_steps"],
                per_device_train_batch_size=cfg["per_device_train_batch_size"],
                learning_rate=cfg["learning_rate"],
                seed=cfg["seed"],
                lora=LoraHyperparams(**cfg["lora"]),
                use_modal=False,
                arm=cfg.get("arm"),
                bf16=cfg.get("bf16", True),
                load_in_4bit=cfg.get("load_in_4bit", False),
                gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
                gradient_checkpointing=cfg.get("gradient_checkpointing", False),
                warmup_ratio=cfg.get("warmup_ratio", 0.03),
                lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
                max_grad_norm=cfg.get("max_grad_norm", 1.0),
                save_total_limit=cfg.get("save_total_limit", 1),
            ),
        )
        vol.commit()
        return {
            "adapter_dir": str(result.adapter_dir),
            "volume_adapter_path": cfg["volume_adapter_path"],
            "base_model": result.base_model,
            "task_ids": result.task_ids,
            "steps": result.steps,
            "final_loss": result.final_loss,
            "lora": result.lora,
            "seed": result.seed,
        }

    with app.run():
        remote = _remote.remote(payload, cfg)

    # Local stub records where the real weights live — do not pretend the
    # local dir contains the adapter files.
    adapter_dir = config.output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "modal_volume_path.txt").write_text(remote["volume_adapter_path"] + "\n")
    return TrainResult(
        adapter_dir=adapter_dir,
        base_model=remote["base_model"],
        task_ids=remote["task_ids"],
        steps=remote["steps"],
        final_loss=remote["final_loss"],
        lora=remote["lora"],
        seed=remote["seed"],
        volume_adapter_path=remote["volume_adapter_path"],
    )


def stage_local_adapter_to_volume(local_adapter_dir: Path, volume_path: str) -> str:
    """Upload a locally-trained adapter into the shared Volume; return volume_path."""
    _require_train()
    try:
        import modal
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "modal is part of the train extra — `uv sync --extra train`"
        ) from e

    if not local_adapter_dir.is_dir():
        raise FileNotFoundError(f"local adapter missing: {local_adapter_dir}")

    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    # Modal's put_directory lands contents at the remote path.
    with vol.batch_upload(force=True) as batch:
        batch.put_directory(str(local_adapter_dir), volume_path)
    return volume_path


def run_training(
    examples: list[TokenizedExample],
    config: TrainConfig,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
    eval_examples: list[TokenizedExample] | None = None,
) -> TrainResult:
    """Dispatch to Modal or local based on `config.use_modal`.

    Three destinations, not two:
      use_modal=True                        → train on a Modal GPU
      use_modal=False, stage_to_volume=True  → train here, upload for Modal serve
      use_modal=False, stage_to_volume=False → train here, touch Modal never (AWS)
    """
    if config.use_modal and model is None:
        if eval_examples:
            # `train_lora_modal` has no eval parameter and the Modal cfg payload
            # carries no eval data, so this would silently run without evaluation.
            raise NotImplementedError(
                "eval_examples is not wired through train_lora_modal; "
                "set use_modal=False to use step-based evaluation"
            )
        return train_lora_modal(examples, config)
    result = train_lora(
        examples, config, model=model, tokenizer=tokenizer, eval_examples=eval_examples
    )
    # An injected model means an offline unit test; never reach the network.
    if model is not None:
        return result
    # A self-managed GPU serves the adapter off local disk, so there is no Volume
    # to stage to. Previously this path attempted a Modal upload regardless and
    # relied on the exception handler, which turned "Modal isn't part of this run"
    # into a silent dependency on Modal being importable.
    if not config.stage_to_volume:
        return result
    vol_path = volume_adapter_dir(config.base_model, config.seed, arm=config.arm)
    try:
        stage_local_adapter_to_volume(result.adapter_dir, vol_path)
        result.volume_adapter_path = vol_path
    except RuntimeError:
        # Modal not available (offline unit test) — leave volume path unset.
        pass
    return result


__all__ = [
    "LoraHyperparams",
    "TrainConfig",
    "TrainResult",
    "run_training",
    "stage_local_adapter_to_volume",
    "train_lora",
    "train_lora_modal",
    "write_train_report",
]
