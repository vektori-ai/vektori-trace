"""Modal-hosted vLLM lifecycle for rollout collection and A2/A3 eval.

Harbor reaches this endpoint via litellm's `hosted_vllm/<name>` provider:
`run_trial(..., model="hosted_vllm/<name>", api_base=<url>, model_info=...)`.
vLLM itself is installed only inside the Modal image — never in the local venv.

Adapter handoff: pass `adapter_path` as a Volume path (`/adapters/...`) from
`TrainResult.volume_adapter_path`. Local paths are staged onto the Volume first.

Phase 0.5 (`docs/PILOT.md`): OPD/GRPO need the token ids vLLM actually sampled.
Pass `return_token_ids` through litellm (`litellm_generate_captured`,
`served_to_harbor_kwargs(capture_tokens=True)`), or put
`token_capture.capture_proxy` in front of `api_base` so harbor agents we do not
control still get the flag. Requires vLLM ≥ 0.10.2 in the serve image.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
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
    gpu: str = "L40S"
    base_model: str = ""
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    # Callable returning one NVML sample from the serving container, or None
    # outside a live Modal session. Lives on the object rather than being fetched
    # over HTTP because only the process holding the Modal app can call into the
    # container — which is the same process that holds the endpoint open.
    sample_gpu: Callable[[], dict[str, Any]] | None = field(
        default=None, repr=False, compare=False
    )

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
        from ..train import stage_local_adapter_to_volume
        from .modal_env import volume_adapter_dir

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
    gpu: str = "L40S",
    model_info: dict[str, Any] | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    extra_vllm_args: list[str] | None = None,
) -> Iterator[ServedModel]:
    """Spin up Modal vLLM with optional LoRA; tear down on exit.

    Not unit-tested (needs GPU + real vLLM) — smoke-tested like measure_pass_rates.

    `max_model_len` is not optional in practice on a small card. vLLM defaults it
    to the model's own maximum — 40 960 for Qwen3-8B and Qwen3-14B alike — and
    refuses to start when that exceeds what the KV cache can hold. This budget is
    per-model; below is Qwen3-14B (40 layers × 8 GQA KV heads × 128 head_dim × 2
    for K and V × 2 bytes = **160 KiB/token**). Qwen3-8B (36 layers) is 144
    KiB/token instead — see `scripts/serve_student.py`'s constants for whichever
    model is actually being served. The 8 KV heads are load-bearing either way:
    query heads share them, and under MHA this would be ~4x the KV/token and
    nothing usable would fit.

    What's left for KV, after weights, on the two cards this project uses
    (Qwen3-14B, 27.6 GiB bf16 weights):

        L40S   48 GiB × 0.90 = 43.2  − 27.6 − 1.3 = 14.3 GiB → ~93 900 tokens
        A10    24 GiB × 0.90 = 21.6  − 27.6 − 1.3 = negative → does not fit at all

    (`gpu=` takes Modal's own strings — "A10", not the card's AWS name "A10G",
    which Modal rejects before a container starts. See modal.com/docs/guide/gpu.)

    (27.6 GiB is 14.8e9 params × 2 bytes; ~1.3 GiB is CUDA context, cuda graphs
    and activation workspace — an *estimate*, and the term to distrust first if a
    start fails near the boundary.)

    Those budgets are totals across all concurrent sequences, not per sequence.
    On an L40S the vLLM default of 40 960 fits ~2 concurrent full-length
    sequences for Qwen3-14B (vs ~4 for Qwen3-8B). Set this to the longest
    trajectory you actually need — `scripts/serve_student.py` refuses impossible
    values before a GPU is allocated.
    """
    _require_modal()
    import modal

    info = dict(model_info or DEFAULT_HOSTED_VLLM_MODEL_INFO)
    if model_info is None and max_model_len is not None:
        # DEFAULT_HOSTED_VLLM_MODEL_INFO advertises 32768 input tokens. harbor
        # and litellm believe it and will send prompts that large. If vLLM was
        # started with a smaller --max-model-len those requests are rejected
        # mid-sweep, after the rollout has already done its work. Advertise the
        # window we actually launched with; reserve a little for the completion.
        info["max_input_tokens"] = max(1, max_model_len - 512)
        info["max_output_tokens"] = min(
            int(info.get("max_output_tokens", 8192)), max(1, max_model_len - 512)
        )
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
        # nvidia-ml-py is what `gpu_stats` reads. vLLM's /metrics reports engine
        # state (KV blocks, queue depth, throughput) but nothing about the card
        # itself, so "is the GPU busy or is it waiting on the verifier?" is
        # unanswerable without NVML.
        .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub", "nvidia-ml-py")
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

        @modal.method()
        def gpu_stats(self) -> dict[str, Any]:
            """One NVML sample of the card this container holds.

            A `@modal.method()`, deliberately, rather than a second web endpoint:
            `health` already proves this call shape works against a live
            container, whereas Modal's docs do not state that one Cls may expose
            two `@modal.web_server`s, and the serving path has too little
            live mileage to spend on an unverified feature.

            Never raises — a telemetry gap must not take down the endpoint that
            is doing the actual work, so a failed sample is reported as data.
            """
            import time as _t

            sample: dict[str, Any] = {"sampled_at": _t.time()}
            try:
                import pynvml

                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                sample.update(
                    {
                        "gpu_name": pynvml.nvmlDeviceGetName(h),
                        # The headline number: percent of the last sampling
                        # period during which a kernel was resident. Low util
                        # with queued requests means the bottleneck is not here.
                        "gpu_util_pct": util.gpu,
                        "mem_util_pct": util.memory,
                        "mem_used_mib": mem.used // (1024**2),
                        "mem_total_mib": mem.total // (1024**2),
                        "temperature_c": pynvml.nvmlDeviceGetTemperature(
                            h, pynvml.NVML_TEMPERATURE_GPU
                        ),
                        "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
                        "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(
                            h, pynvml.NVML_CLOCK_SM
                        ),
                    }
                )
            except Exception as exc:  # pragma: no cover - needs a real GPU
                sample["error"] = f"{type(exc).__name__}: {exc}"
            return sample

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
            # Inherit stdout/stderr so vLLM's own log reaches `modal app logs`.
            proc = subprocess.Popen(cmd)
            deadline = time.time() + 60 * 25
            while time.time() < deadline:
                # Fail the moment the process exits. Polling only /health meant a
                # vLLM that died at second 30 — a missing nvcc, a CUDA mismatch —
                # still burned the full 25-minute deadline on a GPU before
                # raising, and the error said "failed to become healthy in time"
                # rather than naming the cause.
                rc = proc.poll()
                if rc is not None:
                    raise RuntimeError(
                        f"vLLM exited with code {rc} before becoming healthy. "
                        "The real cause is in this container's log above — "
                        "engine-core tracebacks print it ~150 lines before the "
                        "final 'Engine core initialization failed'."
                    )
                try:
                    urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
                    return
                except Exception:
                    time.sleep(2)
            proc.terminate()
            raise RuntimeError("vLLM failed to become healthy within 25 minutes")

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
            sample_gpu=lambda: server.gpu_stats.remote(),
        )
        yield served


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


def litellm_generate(
    served: ServedModel,
    prompt: str,
    *,
    max_tokens: int = 512,
    return_token_ids: bool = False,
) -> str:
    """One completion against a served candidate — used by non-regression.

    When `return_token_ids` is True the request asks vLLM for sampled ids
    (Phase 0.5). The text return value is unchanged; use
    `litellm_generate_captured` when the ids themselves are needed.
    """
    import litellm

    from .token_capture import litellm_extra_body

    kwargs: dict[str, Any] = {
        "model": served.harbor_model,
        "api_base": served.api_base,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "api_key": "EMPTY",  # vLLM OpenAI-compat often ignores auth
    }
    if return_token_ids:
        kwargs["extra_body"] = litellm_extra_body(return_token_ids=True)
    resp = litellm.completion(**kwargs)
    return (resp.choices[0].message.content or "") if resp.choices else ""


def litellm_generate_captured(
    served: ServedModel,
    prompt: str,
    *,
    max_tokens: int = 512,
    logprobs: bool = False,
) -> Any:
    """Completion that *must* return sampled token ids (Phase 0.5).

    Raises `TokenCaptureError` if the endpoint ignored `return_token_ids` —
    silently falling back to re-tokenization is exactly the failure mode this
    path exists to prevent.
    """
    import litellm

    from .token_capture import extract_captured_completion, litellm_extra_body

    resp = litellm.completion(
        model=served.harbor_model,
        api_base=served.api_base,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        api_key="EMPTY",
        extra_body=litellm_extra_body(return_token_ids=True, logprobs=logprobs),
    )
    return extract_captured_completion(resp)


def served_to_harbor_kwargs(
    served: ServedModel,
    *,
    capture_tokens: bool = False,
    capture_logprobs: bool = False,
) -> dict[str, Any]:
    """kwargs for `measure_pass_rates` / `collect_rollouts` / `run_trial`.

    With `capture_tokens=True`, harbor agents receive litellm `extra_body` that
    requests vLLM `return_token_ids` (and optional logprobs). Prefer also
    wrapping `api_base` in `token_capture.capture_proxy` when the agent harness
    may drop unknown kwargs — the proxy injects the flag server-side.
    """
    out: dict[str, Any] = {
        "model": served.harbor_model,
        "api_base": served.api_base,
        "model_info": served.model_info,
    }
    if capture_tokens:
        from .token_capture import token_capture_agent_kwargs

        out["agent_kwargs"] = token_capture_agent_kwargs(
            return_token_ids=True, logprobs=capture_logprobs
        )
    return out


__all__ = [
    "DEFAULT_HOSTED_VLLM_MODEL_INFO",
    "ServedModel",
    "dump_serve_record",
    "litellm_generate",
    "litellm_generate_captured",
    "serve_model",
    "served_to_harbor_kwargs",
]
