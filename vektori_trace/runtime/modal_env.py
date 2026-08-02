"""Shared Modal Volume names for train ↔ serve adapter handoff."""

from __future__ import annotations

# Persistent across train_lora_modal and serve_model so A2/A3 can load the
# adapter that training just wrote. One Volume, one naming scheme.
VOLUME_NAME = "vektori-trace-adapters"
VOLUME_MOUNT = "/adapters"

# Weights cache, mounted at HF_HOME. Without it every cold start re-downloads
# the whole model — ~16GB for an 8B student, ~60GB for the 30B teacher — and
# that download happens on a GPU we are paying for by the second. A pass@k
# sweep is hundreds of rollouts against one server, so this is the difference
# between paying for the model once and paying for it every time the container
# scales to zero.
HF_CACHE_VOLUME_NAME = "hf-model-cache"
HF_CACHE_MOUNT = "/root/.cache/huggingface"

# How long a served container stays up with no requests. Rollouts arrive in
# bursts with container/test time between them, so a short window means paying
# to reload weights mid-sweep. Ten minutes covers the gap between rollouts
# without holding a GPU overnight.
SCALEDOWN_WINDOW_SECONDS = 10 * 60


def volume_adapter_dir(base_model: str, seed: int, arm: str | None = None) -> str:
    """Absolute path *inside* the Modal Volume for an adapter directory."""
    safe = base_model.replace("/", "_").replace(" ", "-")
    parts = [safe, f"seed{seed}"]
    if arm:
        parts.append(arm)
    return f"{VOLUME_MOUNT}/{'--'.join(parts)}/adapter"
