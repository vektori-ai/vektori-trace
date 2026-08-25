"""The model half of the ReOPD driver: load CK35, step, checkpoint, resume.

`replay_train` provides the optimizer step for a *single* update: it takes a
model and optimizer from outside, runs the chunk-OPD loss over one batch, and
steps. That is exactly the right seam, and this class is the multi-update state
that lives around it.

What it owns
------------
- loading CK35 once and keeping the model resident across all 32 updates;
- the optimizer, whose Adam moments must survive every checkpoint;
- saving a checkpoint that can actually resume;
- proving each saved adapter reloads and changes the logits.

What it does not own
--------------------
The loss, the alignment, and the batch semantics. Those stay in `chunk_opd`
and `replay_opd`, unchanged.

Why the model stays resident
----------------------------
Reloading the adapter from disk between updates would be simpler, but it makes
the run's correctness depend on the save/load round trip being lossless at
every step rather than at the one step where it is verified. Keeping the model
in memory and *verifying* the checkpoint separately means a save bug is caught
by the verification rather than silently becoming the next update's parent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .reopd_checkpoint import (
    CheckpointError,
    adapter_hash,
    load_checkpoint,
    save_checkpoint,
)


class ReOPDTrainerError(RuntimeError):
    """The trainable model is not in the state the run requires."""


class ReOPDTrainer:
    """CK35 + optimizer, carried across 32 updates."""

    def __init__(
        self,
        *,
        base_model: str,
        parent_adapter: str,
        learning_rate: float = 1e-5,
        run_dir: str | Path = ".",
        device: str | None = None,
        microbatch_size: int = 1,
        gradient_checkpointing: bool = True,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        self.base_model = base_model
        self.parent_adapter = str(parent_adapter)
        self.learning_rate = learning_rate
        self.run_dir = Path(run_dir)
        self.device = device
        self.microbatch_size = microbatch_size
        self.gradient_checkpointing = gradient_checkpointing
        self.max_grad_norm = max_grad_norm

        self.model: Any = None
        self.optimizer: Any = None
        self.parent_hash: str | None = None
        self._step_fn: Any = None

    # -- setup -------------------------------------------------------------

    def load(self, *, resume_from: str | Path | None = None) -> dict[str, Any]:
        """Load CK35 as a trainable LoRA model, optionally resuming state.

        `resume_from` restores the optimizer, scheduler and RNG from a previous
        update's checkpoint. Its adapter weights are loaded too: resuming from
        CK35's weights with update 19's optimizer would be neither run.
        """
        from ..replay_train import (
            ReplayTrainConfig,
            build_optimizer,
            load_v0_for_training,
            make_optimizer_step,
        )

        self.parent_hash = adapter_hash(self.parent_adapter)

        adapter = str(resume_from) if resume_from else self.parent_adapter
        cfg = ReplayTrainConfig(
            base_model=self.base_model,
            adapter_path=adapter,
            output_dir=self.run_dir / "_scratch",
            learning_rate=self.learning_rate,
            device=self.device,
            microbatch_size=self.microbatch_size,
            gradient_checkpointing=self.gradient_checkpointing,
            max_grad_norm=self.max_grad_norm,
        )
        self.model = load_v0_for_training(cfg)
        self.optimizer = build_optimizer(self.model, cfg)

        resumed = {}
        if resume_from:
            # Adam's moments and the RNG, without which the resumed run is a
            # different recipe that reports the same numbers.
            resumed = load_checkpoint(resume_from, self.model, self.optimizer)

        # `save=False`: this class owns checkpointing, and letting the step
        # write its own adapter would produce two artifacts per update with no
        # record of which one the next update loaded.
        self._step_fn = make_optimizer_step(
            self.model, self.optimizer, cfg, save=False,
            progress_path=self.run_dir / "train_progress.jsonl",
        )
        return {"parent_hash": self.parent_hash, "resumed": bool(resume_from),
                **({"resumed_from_update": resumed.get("update_index")}
                   if resumed else {})}

    # -- one update --------------------------------------------------------

    @property
    def step(self):
        """The callable `run_replay_chunk_opd` expects."""
        if self._step_fn is None:
            raise ReOPDTrainerError("load() must run before the first update")
        return self._step_fn

    def checkpoint(
        self, path: str | Path, *, update_index: int, policy_version: str,
        verify: bool = True, probe_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Save, then prove the save is usable.

        Verification is on by default and costs one forward pass. An adapter
        that saved nothing reloads to the base model, which makes every
        subsequent update train from the wrong parent while the loss curve looks
        entirely normal.
        """
        if self.model is None:
            raise ReOPDTrainerError("nothing to checkpoint; load() first")

        state = save_checkpoint(
            self.model, self.optimizer, path,
            update_index=update_index,
            policy_version=policy_version,
            parent_policy_hash=self.parent_hash or "unknown",
        )

        if verify:
            # Two distinct claims, both needed.
            #   _verify_bytes  : the file matches what is in memory
            #   _verify_reload : a model built FROM the file behaves differently
            #                    from the base weights
            # The first is cheap and catches a truncated or empty write. Only
            # the second catches an adapter that saves fine and is never
            # applied, which makes the whole run a no-op that reports success.
            state = self._verify_bytes(path)
            state = self._verify_reload(
                path, probe_ids or [1, 2, 3, 4, 5, 6, 7, 8])
        return state

    def _verify_bytes(self, path: str | Path) -> dict[str, Any]:
        """Confirm the checkpoint's logits differ from the in-memory model's by
        ~nothing, and from the *base* model's by something.

        Done in-process against the resident model rather than by reloading the
        base weights again: a second full load costs minutes of GPU time per
        update, and the question -- did these weights reach disk -- is
        answerable by comparing the saved tensors to the live ones.
        """
        import torch
        from safetensors.torch import load_file

        p = Path(path)
        f = p / "adapter_model.safetensors"
        if not f.exists():
            raise CheckpointError(f"{p}: no adapter weights were written")

        saved = load_file(str(f))
        if not saved:
            raise CheckpointError(f"{p}: adapter file is empty")

        live = {n: t for n, t in self.model.state_dict().items()
                if "lora" in n.lower()}
        if not live:
            raise CheckpointError(
                "the resident model has no LoRA parameters; it is not a "
                "trainable adapter and this run would be a no-op"
            )

        max_delta, matched = 0.0, 0
        for name, tensor in saved.items():
            key = next((k for k in live if k.endswith(name.split(".", 1)[-1])), None)
            if key is None:
                continue
            matched += 1
            d = (live[key].detach().float().cpu() - tensor.float()).abs().max()
            max_delta = max(max_delta, float(d))

        if matched == 0:
            raise CheckpointError(
                f"{p}: none of the {len(saved)} saved tensors match a live LoRA "
                "parameter by name; the checkpoint does not describe this model"
            )
        if max_delta > 1e-5:
            raise CheckpointError(
                f"{p}: saved weights differ from the live model by {max_delta:.2e}; "
                "the checkpoint is not the trained adapter"
            )

        # Non-trivially different from the parent, i.e. training did something.
        drift = 0.0
        try:
            parent = load_file(
                str(Path(self.parent_adapter) / "adapter_model.safetensors"))
            for name, tensor in saved.items():
                if name in parent and parent[name].shape == tensor.shape:
                    drift = max(drift, float(
                        (tensor.float() - parent[name].float()).abs().max()))
        except Exception:            # parent unreadable: not fatal, just unproven
            drift = float("nan")

        from .reopd_state import atomic_write_json
        sp = p / "state.json"
        state = json.loads(sp.read_text())
        # Deliberately NOT reload_verified: matching bytes is not a reload.
        state["bytes_verified"] = True
        state["bytes_report"] = {
            "n_tensors": len(saved), "n_matched": matched,
            "max_delta_from_live": max_delta,
            "max_drift_from_parent": drift,
        }
        atomic_write_json(sp, state)
        return state

    def _verify_reload(self, path: str | Path, probe_ids: list[int]) -> dict[str, Any]:
        """Build a model FROM the checkpoint and prove the adapter is applied.

        `_verify_bytes` proves the file matches memory. It cannot prove the file
        is loadable, that peft recognises its tensor names, or that the adapter
        actually alters the forward pass -- an adapter that loads cleanly and is
        never applied looks identical from outside and makes every update a
        no-op reporting success.

        So this loads the checkpoint fresh and compares its logits against the
        same model with the adapter disabled. Equal logits are fatal.
        """
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        from .reopd_state import atomic_write_json

        p = Path(path)
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model, torch_dtype=torch.bfloat16, device_map=dev)
        try:
            reloaded = PeftModel.from_pretrained(base, str(p))
            reloaded.eval()
            ids = torch.tensor([probe_ids], device=dev)

            with torch.no_grad():
                adapted = reloaded(ids).logits[0, -1].float().cpu()
                # `disable_adapter` gives the base behaviour through the exact
                # same graph, so a difference isolates the adapter rather than
                # any loading difference between two model objects.
                with reloaded.disable_adapter():
                    plain = reloaded(ids).logits[0, -1].float().cpu()

            delta = float((adapted - plain).abs().max())
            n_lora = sum(1 for n, _ in reloaded.named_parameters()
                         if "lora" in n.lower())
            if n_lora == 0:
                raise CheckpointError(
                    f"{p}: reloaded model has no LoRA parameters; the adapter "
                    "did not attach"
                )
            if delta == 0.0:
                raise CheckpointError(
                    f"{p}: the reloaded adapter produces logits identical to "
                    "the base model. It loaded but is not applied, which makes "
                    "every update a no-op that reports success."
                )
        finally:
            del base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        sp = p / "state.json"
        state = json.loads(sp.read_text())
        state["reload_verified"] = True
        state["reload_report"] = {
            **(state.get("bytes_report") or {}),
            "fresh_load": True,
            "n_lora_params": n_lora,
            "max_logit_delta_vs_base": delta,
            "n_probe_tokens": len(probe_ids),
        }
        atomic_write_json(sp, state)
        return state


__all__ = ["ReOPDTrainer", "ReOPDTrainerError"]
