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
) -> Iterator[ServedModel]:
    """Spin up Modal vLLM with optional LoRA; tear down on exit.

    Not unit-tested (needs GPU + real vLLM) — smoke-tested like measure_pass_rates.
    """
    _require_modal()
    import modal

    info = dict(model_info or DEFAULT_HOSTED_VLLM_MODEL_INFO)
    name = _canonical_name(base_model)
    vol_adapter = _resolve_volume_adapter(adapter_path)
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("vllm", "huggingface_hub")
        .env({"HF_HOME": HF_CACHE_MOUNT})
    )

    app = modal.App(_SERVE_APP_NAME)

    @app.cls(
        gpu=gpu,
        image=image,
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
