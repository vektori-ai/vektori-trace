"""Save and restore everything a 32-update run needs to resume correctly.

`replay_train` saves an adapter. That is enough for a one-update run and not
enough for thirty-two: resuming with a fresh optimizer discards Adam's first and
second moments, which silently changes the effective step size for every update
that follows. Nothing in the loss curve shows it -- the run continues, the
numbers stay finite, and the recipe is no longer the one the manifest claims.

So a checkpoint here is five things, not one:

    adapter weights      what the policy is
    optimizer state      Adam moments, or the LR is not what it says
    scheduler state      where in the schedule, if one exists
    RNG state            so a resumed run samples the stream it would have
    run state            update index, policy version, parent hash, reload proof

The reload proof
----------------
A saved adapter that does not change logits on reload makes the entire run a
no-op that reports success at every step. `verify_reload` compares logits from
the in-memory model against the reloaded one on a fixed probe, and refuses
equality with the *base* model. This is cheap and it is the only check that
catches a save which silently wrote nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .reopd_state import atomic_write_json


class CheckpointError(RuntimeError):
    """A checkpoint cannot be written, or cannot be trusted to resume."""


def rng_snapshot() -> dict[str, Any]:
    """Capture every RNG the sampling and training path draws from."""
    import base64
    import random

    import numpy as np
    import torch

    state: dict[str, Any] = {
        "python": base64.b64encode(
            json.dumps(random.getstate(), default=list).encode()
        ).decode(),
        "numpy": base64.b64encode(
            np.random.get_state()[1].tobytes()
        ).decode(),
        "torch": base64.b64encode(
            torch.get_rng_state().numpy().tobytes()
        ).decode(),
    }
    if torch.cuda.is_available():
        state["cuda"] = base64.b64encode(
            torch.cuda.get_rng_state().numpy().tobytes()
        ).decode()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    """Put the RNGs back where the checkpoint found them."""
    import base64
    import random

    import numpy as np
    import torch

    if state.get("python"):
        raw = json.loads(base64.b64decode(state["python"]).decode())
        random.setstate((raw[0], tuple(raw[1]), raw[2]))
    if state.get("numpy"):
        keys = np.frombuffer(base64.b64decode(state["numpy"]), dtype=np.uint32)
        np.random.set_state(("MT19937", keys, 624, 0, 0.0))
    if state.get("torch"):
        buf = np.frombuffer(base64.b64decode(state["torch"]), dtype=np.uint8)
        torch.set_rng_state(torch.tensor(buf.copy(), dtype=torch.uint8))
    if state.get("cuda") and torch.cuda.is_available():
        buf = np.frombuffer(base64.b64decode(state["cuda"]), dtype=np.uint8)
        torch.cuda.set_rng_state(torch.tensor(buf.copy(), dtype=torch.uint8))


def adapter_hash(path: str | Path) -> str:
    """Content hash of the adapter weights, for parent-lineage checks."""
    p = Path(path)
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        f = p / name
        if f.exists():
            h = hashlib.sha256()
            with f.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()[:16]
    raise CheckpointError(f"no adapter weights under {p}")


def save_checkpoint(
    model: Any,
    optimizer: Any,
    path: str | Path,
    *,
    update_index: int,
    policy_version: str,
    parent_policy_hash: str,
    scheduler: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a checkpoint that can actually resume the run.

    `state.json` is written **last**, after the weights and optimizer are on
    disk, because `reopd_state.validate_checkpoint` treats its presence as the
    claim that everything else is durable.
    """
    import torch

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(p))
    torch.save(optimizer.state_dict(), p / "optimizer.pt")
    sched_state = None
    if scheduler is not None:
        sched_state = scheduler.state_dict()
        torch.save(sched_state, p / "scheduler.pt")

    state = {
        "update_index": update_index,
        "policy_version": policy_version,
        "parent_policy_hash": parent_policy_hash,
        "adapter_hash": adapter_hash(p),
        "rng_state": rng_snapshot(),
        # Explicitly null rather than absent when there is no scheduler:
        # `replay_train.build_optimizer` returns a bare AdamW, and an absent key
        # reads as "forgot to save it" rather than "there is none".
        "scheduler_state": sched_state,
        "has_scheduler": scheduler is not None,
        "reload_verified": False,
        **(extra or {}),
    }
    atomic_write_json(p / "state.json", state)
    return state


def verify_reload(
    path: str | Path,
    *,
    base_model: str,
    probe_ids: list[int],
    reference_logits: Any = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Prove the saved adapter reloads and actually changes the model.

    Two distinct failures, both silent:

    - the adapter did not save (or saved empty), so reloading gives the base
      model and every subsequent update trains from the wrong parent;
    - the adapter loads but is not applied, which looks identical from the
      outside.

    Comparing against the base model's logits on a fixed probe catches both.
    Marks `reload_verified` in `state.json` on success -- `validate_checkpoint`
    refuses a checkpoint without it.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    p = Path(path)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map=dev
    )
    ids = torch.tensor([probe_ids], device=dev)
    with torch.no_grad():
        base_logits = base(ids).logits[0, -1].float().cpu()

    merged = PeftModel.from_pretrained(base, str(p))
    merged.eval()
    with torch.no_grad():
        adapted_logits = merged(ids).logits[0, -1].float().cpu()

    delta = float((adapted_logits - base_logits).abs().max())
    if delta == 0.0:
        raise CheckpointError(
            f"{p}: the reloaded adapter produces logits identical to the base "
            "model. Either nothing was saved or the adapter is not applied; "
            "every update after this one would train from the wrong parent."
        )

    report = {"max_logit_delta": delta, "n_probe_tokens": len(probe_ids)}
    if reference_logits is not None:
        drift = float((adapted_logits - reference_logits).abs().max())
        report["drift_from_in_memory"] = drift
        if drift > 1e-2:
            raise CheckpointError(
                f"{p}: reloaded logits differ from the in-memory model by "
                f"{drift:.4f}; the saved adapter is not the trained one."
            )

    state = json.loads((p / "state.json").read_text())
    state["reload_verified"] = True
    state["reload_report"] = report
    atomic_write_json(p / "state.json", state)
    return report


def load_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any,
    *,
    scheduler: Any = None,
    restore_rng_state: bool = True,
) -> dict[str, Any]:
    """Restore optimizer, scheduler and RNG from a checkpoint.

    The adapter itself is loaded separately (by `PeftModel.from_pretrained`),
    because loading weights and restoring training state are different concerns
    and the caller may want the first without the second.
    """
    import torch

    p = Path(path)
    state_path = p / "state.json"
    if not state_path.exists():
        raise CheckpointError(f"{p}: no state.json; this is not a resumable checkpoint")
    state = json.loads(state_path.read_text())

    opt_path = p / "optimizer.pt"
    if not opt_path.exists():
        raise CheckpointError(
            f"{p}: no optimizer.pt. Resuming with a fresh optimizer discards "
            "Adam's moments and silently changes the effective learning rate."
        )
    optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))

    if scheduler is not None:
        sp = p / "scheduler.pt"
        if not sp.exists():
            raise CheckpointError(f"{p}: scheduler expected but scheduler.pt is absent")
        scheduler.load_state_dict(torch.load(sp, map_location="cpu"))

    if restore_rng_state and state.get("rng_state"):
        restore_rng(state["rng_state"])
    return state


__all__ = [
    "CheckpointError",
    "adapter_hash",
    "load_checkpoint",
    "restore_rng",
    "rng_snapshot",
    "save_checkpoint",
    "verify_reload",
]
