"""Shared Modal Volume names for train ↔ serve adapter handoff."""

from __future__ import annotations

# Persistent across train_lora_modal and serve_model so A2/A3 can load the
# adapter that training just wrote. One Volume, one naming scheme.
VOLUME_NAME = "vektori-trace-adapters"
VOLUME_MOUNT = "/adapters"


def volume_adapter_dir(base_model: str, seed: int, arm: str | None = None) -> str:
    """Absolute path *inside* the Modal Volume for an adapter directory."""
    safe = base_model.replace("/", "_").replace(" ", "-")
    parts = [safe, f"seed{seed}"]
    if arm:
        parts.append(arm)
    return f"{VOLUME_MOUNT}/{'--'.join(parts)}/adapter"
