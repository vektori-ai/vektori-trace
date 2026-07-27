"""LoRA SFT loop for Step 6. Masking is baked into `labels` (-100) by
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
from .modal_env import VOLUME_MOUNT, VOLUME_NAME, volume_adapter_dir


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
    modal_gpu: str = "A10G"
    # Distinguishes A2 vs A3 adapters on the shared Volume.
    arm: str | None = None


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

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model, trust_remote_code=True, torch_dtype=torch.float32
        )

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
    collator = LabelPreservingCollator(pad_token_id=tokenizer.pad_token_id)

    args = TrainingArguments(
        output_dir=str(config.output_dir / "hf_runs"),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        learning_rate=config.learning_rate,
        logging_steps=1,
        save_steps=max(config.max_steps, 1),
        report_to=[],
        seed=config.seed,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
    )
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
    }

    app = modal.App("vektori-trace-train")
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch", "transformers", "peft", "accelerate", "datasets")
        .add_local_python_source("vektori_trace")
    )

    @app.function(
        gpu=config.modal_gpu,
        image=image,
        volumes={VOLUME_MOUNT: vol},
        timeout=4 * 60 * 60,
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
) -> TrainResult:
    """Dispatch to Modal or local based on `config.use_modal`."""
    if config.use_modal and model is None:
        return train_lora_modal(examples, config)
    result = train_lora(examples, config, model=model, tokenizer=tokenizer)
    # Local train + Modal serve still needs the weights on the Volume.
    if config.use_modal is False and model is not None:
        return result
    if not config.use_modal:
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
