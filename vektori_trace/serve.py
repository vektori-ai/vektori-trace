"""Modal-hosted vLLM lifecycle for rollout collection and A2/A3 eval.

Harbor reaches this endpoint via litellm's `hosted_vllm/<name>` provider:
`run_trial(..., model="hosted_vllm/<name>", api_base=<url>, model_info=...)`.
vLLM itself is installed only inside the Modal image — never in the local venv.

Adapter handoff: pass `adapter_path` as a Volume path (`/adapters/...`) from
`TrainResult.volume_adapter_path`. Local paths are staged onto the Volume first.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .modal_env import (
    HF_CACHE_MOUNT,
    HF_CACHE_VOLUME_NAME,
    SCALEDOWN_WINDOW_SECONDS,
    VOLUME_MOUNT,
    VOLUME_NAME,
)

# Default model_info harbor requires for hosted_vllm (token limits + costs).
DEFAULT_HOSTED_VLLM_MODEL_INFO: dict[str, Any] = {
    "max_input_tokens": 32768,
    "max_output_tokens": 8192,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}

_SERVE_APP_NAME = "vektori-trace-serve"

# Pinned together — see the comment in serve_model before changing either.
VLLM_VERSION = "0.21.0"
VLLM_CUDA_IMAGE = "nvidia/cuda:12.9.0-devel-ubuntu22.04"


def _require_modal():
    try:
        import modal  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "training extras required: install with `uv sync --extra train` "
            "(modal is in the train extra)"
        ) from e


@dataclass
class ServedModel:
    """A live Modal vLLM endpoint harbor/litellm can call."""

    api_base: str
    model_name: str
    model_info: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_HOSTED_VLLM_MODEL_INFO)
    )
    adapter_path: str | None = None
    gpu: str = "A10G"
    base_model: str = ""
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None

    @property
    def harbor_model(self) -> str:
        return f"hosted_vllm/{self.model_name}"


def _canonical_name(base_model: str) -> str:
    raw = base_model.split("/")[-1].replace(" ", "-")
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in raw)[:63]


def _resolve_volume_adapter(adapter_path: str | None) -> str | None:
    """Return a path inside the Volume mount, staging local dirs if needed."""
    if not adapter_path:
        return None
    if adapter_path.startswith(VOLUME_MOUNT):
        return adapter_path
    local = Path(adapter_path)
    # Local stub from modal train only has a pointer file — read it.
    pointer = local / "modal_volume_path.txt"
    if pointer.is_file():
        return pointer.read_text().strip()
    if local.is_dir() and any(local.iterdir()):
        from .modal_env import volume_adapter_dir
        from .train import stage_local_adapter_to_volume

        remote = volume_adapter_dir("local-staged", seed=0, arm=local.name)
        return stage_local_adapter_to_volume(local, remote)
    raise FileNotFoundError(
        f"adapter_path {adapter_path!r} is neither a Volume path nor a local adapter dir"
    )


@contextmanager
def serve_model(
    base_model: str,
    *,
    adapter_path: str | None = None,
    gpu: str = "A10G",
    model_info: dict[str, Any] | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    extra_vllm_args: list[str] | None = None,
) -> Iterator[ServedModel]:
    """Spin up Modal vLLM with optional LoRA; tear down on exit.

    Not unit-tested (needs GPU + real vLLM) — smoke-tested like measure_pass_rates.

    `max_model_len` is not optional in practice on a small card. vLLM defaults it
    to the model's own maximum — 40 960 for Qwen3-8B — and refuses to start when
    that exceeds what the KV cache can hold. On a 24 GB A10G the arithmetic is:

        24 GiB × 0.90 utilisation      = 21.6 GiB
        − 15.3 GiB bf16 weights
        − ~1.3 GiB CUDA context/workspace
        =  ~5.0 GiB for KV

    and Qwen3-8B costs 144 KiB per token of KV (36 layers × 8 GQA KV heads ×
    128 head_dim × 2 for K and V × 2 bytes), so ~36 700 tokens fit *in total,
    across all concurrent sequences*. Asking for 40 960 for a single sequence is
    already more than the card has, and vLLM fails at startup rather than
    silently degrading. Set this to the longest trajectory you actually need.
    """
    _require_modal()
    import modal

    info = dict(model_info or DEFAULT_HOSTED_VLLM_MODEL_INFO)
    name = _canonical_name(base_model)
    vol_adapter = _resolve_volume_adapter(adapter_path)
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

    # Base image and vLLM version are PINNED TOGETHER, and must move together.
    #
    # vLLM publishes wheels for several CUDA lines and `pip install vllm`
    # (unpinned) resolves to the newest, which currently drags in cu13 packages
    # (nvidia-nccl-cu13, nvidia-cudnn-cu13, ...). Dropping those on a CUDA 12.8
    # base is a version mismatch, and the earlier `debian_slim` image had no
    # CUDA toolchain at all, so vLLM's V1 engine died on:
    #   RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
    #                 doesn't exist
    # Both failures land *after* Modal has allocated a GPU and pulled ~16 GB of
    # weights, and both surface as "Engine core initialization failed" with the
    # real cause a hundred lines earlier.
    #
    # This is the pair Modal publishes in its own vLLM example, so it is a
    # tested combination rather than an inference from the changelog. `-devel`
    # (not `-runtime`) is required: it carries nvcc.
    image = (
        modal.Image.from_registry(
            VLLM_CUDA_IMAGE, add_python="3.12"
        )
        .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub")
        .env(
            {
                "HF_HOME": HF_CACHE_MOUNT,
                # Faster weight pulls from the Hub.
                "HF_XET_HIGH_PERFORMANCE": "1",
                # Emit engine stats every second so /metrics is actually live —
                # scripts/vllm_monitor.py reads exactly these.
                "VLLM_LOG_STATS_INTERVAL": "1",
            }
        )
    )

    app = modal.App(_SERVE_APP_NAME)

    @app.cls(
        gpu=gpu,
        image=image,
        # VllmServer is defined inside this function so it can close over
        # `base_model`, `name`, `vol_adapter` and the vLLM flags. Modal
        # normally requires globals-scope classes because it re-imports them by
        # module path in the container; `serialized=True` makes it pickle the
        # class instead, which is what a locally-defined one needs. Without it
        # every call raised LocalFunctionError before reaching the GPU.
        serialized=True,
        # HF_HOME is backed by a Volume so weights survive scale-to-zero. The
        # env var alone only relocated the cache inside an ephemeral container,
        # so every cold start paid to download the model again.
        volumes={VOLUME_MOUNT: vol, HF_CACHE_MOUNT: hf_cache},
        scaledown_window=SCALEDOWN_WINDOW_SECONDS,
        timeout=60 * 60,
    )
    class VllmServer:
        @modal.enter()
        def start(self) -> None:
            return None

        @modal.method()
        def health(self) -> str:
            return "ok"

        @modal.web_server(port=8000, startup_timeout=60 * 30)
        def openai_compat(self):
            import subprocess
            import time
            import urllib.request

            cmd = [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                base_model,
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--served-model-name",
                name,
            ]
            if max_model_len is not None:
                cmd += ["--max-model-len", str(max_model_len)]
            if gpu_memory_utilization is not None:
                cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
            if extra_vllm_args:
                cmd += list(extra_vllm_args)
            if vol_adapter:
                cmd += ["--enable-lora", "--lora-modules", f"{name}={vol_adapter}"]
            subprocess.Popen(cmd)
            deadline = time.time() + 60 * 25
            while time.time() < deadline:
                try:
                    urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
                    return
                except Exception:
                    time.sleep(2)
            raise RuntimeError("vLLM failed to become healthy in time")

    with app.run():
        server = VllmServer()
        _ = server.health.remote()
        web = server.openai_compat
        url = web.get_web_url() if hasattr(web, "get_web_url") else None
        if not url:
            url = getattr(web, "web_url", None) or (
                f"https://{_SERVE_APP_NAME}--openai-compat.modal.run"
            )
        served = ServedModel(
            api_base=url.rstrip("/") + "/v1",
            model_name=name,
            model_info=info,
            adapter_path=vol_adapter,
            gpu=gpu,
            base_model=base_model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        yield served


def served_to_harbor_kwargs(served: ServedModel) -> dict[str, Any]:
    """kwargs for `measure_pass_rates` / `collect_rollouts` / `run_trial`."""
    return {
        "model": served.harbor_model,
        "api_base": served.api_base,
        "model_info": served.model_info,
    }


def dump_serve_record(served: ServedModel) -> dict[str, Any]:
    """Environment record for arms.json (V0_PLAN.md: every run writes GPU/image)."""
    return {
        "api_base": served.api_base,
        "harbor_model": served.harbor_model,
        "base_model": served.base_model,
        "adapter_path": served.adapter_path,
        "gpu": served.gpu,
        "model_info": served.model_info,
        "volume": VOLUME_NAME,
    }


def litellm_generate(served: ServedModel, prompt: str, *, max_tokens: int = 512) -> str:
    """One completion against a served candidate — used by non-regression."""
    import litellm

    resp = litellm.completion(
        model=served.harbor_model,
        api_base=served.api_base,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        api_key="EMPTY",  # vLLM OpenAI-compat often ignores auth
    )
    return (resp.choices[0].message.content or "") if resp.choices else ""


__all__ = [
    "DEFAULT_HOSTED_VLLM_MODEL_INFO",
    "ServedModel",
    "dump_serve_record",
    "litellm_generate",
    "serve_model",
    "served_to_harbor_kwargs",
]
