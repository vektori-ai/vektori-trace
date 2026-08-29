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

import logging
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

_log = logging.getLogger(__name__)

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
    # Every LoRA registered on this endpoint, served name -> Volume path. One
    # base load can host many adapters; Phase 7 grades seven checkpoints and
    # paying the ~27.6 GiB base load seven times would cost more than the whole
    # evaluation. `model_name` is one of these keys.
    adapter_models: dict[str, str] = field(default_factory=dict)
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
    # Separate control-plane endpoint on the same serving container. ReOPD's
    # trainer commits checkpoints from another container; Modal Volume mounts
    # are snapshots and must be reloaded before vLLM can open the new path.
    reload_volume_url: str | None = None

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


def resolve_web_url(*candidates: Any, what: str = "endpoint") -> str:
    """The URL Modal actually assigned, or a hard failure.

    Extracted so it can be tested without Modal, a GPU or a network: the
    2026-08-28 incident was two endpoint starts that logged "UP in 183s" and
    then 404'd every request, because no URL could be obtained and the caller
    fabricated `https://{app_name}--openai-compat.modal.run` -- missing the
    workspace prefix, the class segment and an ephemeral app's `-dev` suffix.
    A fabricated address is strictly worse than an error: the endpoint looks
    healthy until the first real request, and the failure reads as the model's.

    Candidates are tried in order; for each, `get_web_url()` first, then a
    `web_url` attribute. A candidate that raises is skipped rather than
    aborting the search.
    """
    for candidate in candidates:
        if candidate is None:
            continue
        url = None
        getter = getattr(candidate, "get_web_url", None)
        if callable(getter):
            try:
                url = getter()
            except Exception:  # noqa: BLE001
                url = None
        if not url:
            url = getattr(candidate, "web_url", None)
        if url and isinstance(url, str) and url.startswith("https://"):
            return url
        if url:
            raise RuntimeError(
                f"Modal returned {url!r} for the {what} URL, which is not an "
                "https:// address; refusing to use it"
            )
    raise RuntimeError(
        f"Modal did not expose a URL for the {what}. Refusing to guess one: a "
        "fabricated URL 404s on every request while the server itself looks "
        "healthy, so the failure reads as a model failure rather than an "
        "addressing one."
    )


@contextmanager
def serve_model(
    base_model: str,
    *,
    adapter_path: str | None = None,
    adapter_paths: dict[str, str] | None = None,
    max_lora_rank: int | None = None,
    max_loras: int | None = None,
    gpu: str = "L40S",
    model_info: dict[str, Any] | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    extra_vllm_args: list[str] | None = None,
    on_app_started: Callable[[str], None] | None = None,
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
    if max_model_len is not None:
        # harbor and litellm believe whatever `model_info` advertises and will
        # send prompts that large. If vLLM was started with a smaller
        # --max-model-len those requests are rejected mid-sweep, after the
        # rollout has already done its work.
        #
        # This used to run only when the caller passed no `model_info` at all,
        # which is exactly backwards: `passk --model-info @file.json` is the
        # common path (the box keeps a `model_info_14b.json`), and a
        # hand-written file is far likelier to disagree with the served window
        # than the default is. Clamp both cases.
        #
        # input + output must fit the window *together* -- they share it. The
        # old arithmetic gave input `L-512` and output `min(8192, L-512)`, i.e.
        # 15 360 tokens of advertised budget for an 8 192-token context.
        out_cap = int(info.get("max_output_tokens", 16384))
        out_cap = max(1, min(out_cap, max_model_len // 2))
        in_cap = max(1, max_model_len - out_cap)
        if int(info.get("max_input_tokens", 0)) > in_cap:
            _log.warning(
                "model_info advertises max_input_tokens=%s but the server was "
                "started with --max-model-len=%s; clamping to %s so litellm "
                "cannot send prompts vLLM will reject mid-sweep",
                info.get("max_input_tokens"),
                max_model_len,
                in_cap,
            )
        info["max_input_tokens"] = min(int(info.get("max_input_tokens", in_cap)), in_cap)
        info["max_output_tokens"] = out_cap
        # Diagnostic: litellm forwards max_output_tokens as the request's
        # `max_tokens`, so this is the completion ceiling every rollout runs
        # under. A truncated completion arrives as finish_reason="length", and
        # with a reasoning parser an unterminated <think> leaves an empty
        # message that tau2 rejects outright -- so the number matters.
        _log.warning(
            "advertising max_input_tokens=%s max_output_tokens=%s "
            "(max_model_len=%s)",
            info["max_input_tokens"], out_cap, max_model_len,
        )
    name = _canonical_name(base_model)
    vol_adapter = _resolve_volume_adapter(adapter_path)
    # The adapter gets its OWN served name, and that is what callers are handed.
    # vLLM resolves a request's `model` against the served base name *before* it
    # consults the LoRA table, so registering the adapter under `name` — as this
    # did — makes every request fall through to the base weights. Nothing errors:
    # the sweep runs, the adapter is never applied, and the result reads as "OPD
    # changed nothing". Distinct names make that failure impossible, and make it
    # visible in /v1/models and in `harbor_model`.
    lora_name = f"{name}-lora" if vol_adapter else None
    # `adapter_paths` registers additional LoRAs under caller-chosen suffixes.
    # vLLM resolves a request's `model` against the base name before it consults
    # the LoRA table, so every adapter needs a name distinct from `name` or the
    # request silently falls through to base weights — the failure that made an
    # OPD sweep read as "changed nothing".
    lora_table: dict[str, str] = {}
    if vol_adapter:
        lora_table[lora_name] = vol_adapter
    for suffix, path in (adapter_paths or {}).items():
        served = f"{name}-{suffix}"
        if served == name:
            raise ValueError(
                f"adapter suffix {suffix!r} collides with the base served name "
                f"{name!r}; requests would resolve to base weights"
            )
        lora_table[served] = _resolve_volume_adapter(path)
    served_name = lora_name or (sorted(lora_table)[0] if lora_table else name)
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
                # Lets a client POST /v1/load_lora_adapter to swap an adapter
                # without restarting the server. Multi-update ReOPD needs the
                # endpoint to serve update N's policy before sampling update
                # N+1; restarting vLLM per update would add ~3 minutes x 32.
                #
                # vLLM documents this as unsafe for untrusted environments
                # because it loads weights from a caller-supplied path. Here the
                # only caller is the run's own driver and the path is a Modal
                # volume, so the exposure is the same as the launch flags.
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
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
        # Hard cap on a single container's lifetime — Modal kills it at this
        # point even while it is actively serving. One hour is shorter than the
        # thing this server exists for: run6's pass@k sweep ran 4.5 h against
        # one endpoint, and a 4-task n=4 eval is ~3 h. The old value was raised
        # by hand on the box mid-run and never committed, so every clean
        # checkout shipped a server that dies partway through its own sweep.
        # `scaledown_window` is what stops an *idle* container from billing;
        # this is only the ceiling for a busy one.
        timeout=12 * 60 * 60,
        # One engine, not N. Modal's unit of scaling is the in-flight *request*,
        # and its default is one per container — so on 2026-08-13 a sweep with
        # two rollout workers plus a /metrics poller had three requests open at
        # once and Modal answered by booting three containers, each holding its
        # own L40S and its own copy of the 27.6 GB model. 3x the bill, and vLLM
        # never batched anything because no engine ever received more than one
        # request. `max_containers=1` plus the concurrency below is what puts
        # the scheduling back where it belongs: vLLM's continuous batching,
        # which is the entire reason to run vLLM.
        max_containers=1,
    )
    # Applied to the class, per Modal's docs — all methods share the container,
    # so this covers `web_server` (the OpenAI API) and the `method()` calls
    # (`health`, `gpu_stats`) alike. `target_inputs` is autoscaler advice and is
    # inert while `max_containers=1`; it matters only if that cap is ever
    # raised. `max_inputs` is the hard ceiling on what enters one engine — set
    # well above the KV capacity so Modal never queues ahead of vLLM, whose
    # scheduler is the one that should decide what runs.
    @modal.concurrent(max_inputs=32, target_inputs=8)
    class VllmServer:
        @modal.enter()
        def start(self) -> None:
            return None

        @modal.method()
        def health(self) -> str:
            return "ok"

        @modal.fastapi_endpoint(method="POST")
        def reload_volume(self) -> dict[str, Any]:
            """Make checkpoints committed by another container visible here.

            This endpoint intentionally does only the filesystem synchronization;
            the caller still asks vLLM to load the adapter and verifies both the
            registered model name and its behavior afterwards.
            """
            import time as _t

            started = _t.time()
            vol.reload()
            return {"ok": True, "seconds": round(_t.time() - started, 3)}

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
            if lora_table:
                cmd += ["--enable-lora"]
                cmd += ["--lora-modules"] + [
                    f"{n}={p}" for n, p in sorted(lora_table.items())
                ]
                # vLLM's default max_lora_rank is 16 and every adapter this repo
                # trains is rank 32 (`adapter_config.json`, r: 32). Without this
                # the engine refuses to start — after Modal has allocated the
                # GPU and pulled the weights.
                if max_lora_rank is not None:
                    cmd += ["--max-lora-rank", str(max_lora_rank)]
                # How many distinct LoRAs may be live in one batch. Requests are
                # grouped by adapter, so the default of 1 is correct for a
                # sequential sweep and only costs throughput when it is not.
                if max_loras is not None:
                    cmd += ["--max-loras", str(max_loras)]
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
        # Expose the exact app created by this context before the first remote
        # call can allocate a GPU. Lifecycle wrappers use this to stop only the
        # app they own if startup or evaluation fails.
        if not app.app_id:
            raise RuntimeError("Modal app started without an app_id")
        if on_app_started is not None:
            on_app_started(app.app_id)
        server = VllmServer()
        _ = server.health.remote()
        web = server.openai_compat
        # Try every source that can name the URL, class Function first, and
        # raise if none does.
        #
        # An earlier note here asserted that `get_web_url()` exists only on the
        # class's Function and never on the instance's bound method on modal
        # 1.5.3. That is NOT established -- later probes on both the laptop and
        # the box returned a correct URL from the bound method -- so it is not
        # the root cause and should not be repeated as one. What *is* verified
        # is the failure it produced: no URL was obtained, and the old fallback
        # fabricated one (below). Trying both candidates costs nothing and does
        # not depend on which explanation is right.
        try:
            url = resolve_web_url(
                getattr(VllmServer, "openai_compat", None), web,
                what="OpenAI-compatible endpoint",
            )
        except RuntimeError:
            url = None
        if not url:
            # NEVER fabricate this. The old fallback built
            # `https://{app_name}--openai-compat.modal.run`, which is missing
            # the workspace prefix, the class segment and the `-dev` suffix an
            # ephemeral app carries -- so it resolves to nothing and every
            # request returns `404 modal-http: invalid function call`. vLLM
            # loads fine, the container is healthy, and the only broken thing
            # is the address, which makes it look like a model failure.
            # Observed 2026-08-28: two consecutive endpoint starts reported
            # "UP in 183s" and then failed their smoke completion.
            raise RuntimeError(
                "Modal did not expose a web URL for the OpenAI-compatible "
                "endpoint. Refusing to guess one: a fabricated URL 404s on "
                "every request while the server itself looks healthy."
            )
        try:
            reload_url = resolve_web_url(
                getattr(VllmServer, "reload_volume", None),
                server.reload_volume,
                what="serving-volume reload endpoint",
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Modal did not expose the serving-volume reload endpoint; "
                f"multi-update ReOPD cannot see newly committed checkpoints ({exc})"
            ) from exc
        served = ServedModel(
            api_base=url.rstrip("/") + "/v1",
            model_name=served_name,
            model_info=info,
            adapter_path=vol_adapter,
            adapter_models=dict(lora_table),
            gpu=gpu,
            base_model=base_model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            sample_gpu=lambda: server.gpu_stats.remote(),
            reload_volume_url=reload_url,
        )
        yield served


def dump_serve_record(served: ServedModel) -> dict[str, Any]:
    """Environment record for arms.json (V0_PLAN.md: every run writes GPU/image)."""
    return {
        "api_base": served.api_base,
        "harbor_model": served.harbor_model,
        "base_model": served.base_model,
        "adapter_path": served.adapter_path,
        "reload_volume_url": served.reload_volume_url,
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


#: Rendering options every request must carry, matching how the SFT corpus was
#: tokenized (`docs/SFT-SCRATCH-PLAN.md` step 2). Qwen3's template default is
#: already thinking-on, so this pins a decision rather than changing behaviour —
#: but an unpinned default is exactly how the prompt-seed probe ended up serving
#: a configuration nobody had chosen. `enable_thinking=False` would additionally
#: write `<think>\n\n</think>\n\n` into the *prompt*, so the model would be
#: asked to continue from a position its targets never started at.
CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": True}

#: The agent-constructor parameter Terminus2 actually forwards to the LLM call.
#: Named here because getting it wrong fails silently: unknown kwargs reach
#: `BaseAgent.__init__(**kwargs)` and are dropped without an error.
LLM_CALL_KWARGS_KEY = "llm_call_kwargs"


def served_to_harbor_kwargs(
    served: ServedModel,
    *,
    capture_tokens: bool = False,
    capture_logprobs: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """kwargs for `measure_pass_rates` / `collect_rollouts` / `run_trial`.

    Always carries `extra_body.chat_template_kwargs` so the served template
    renders the way the corpus was tokenized. terminus-2 forwards unknown kwargs
    into litellm and `extra_body` is the documented seam for vLLM-only fields;
    without this the harness silently used the template default, which is how a
    rollout and a Phase 7 sweep of "the same" checkpoint measured different
    prompts.

    With `capture_tokens=True`, harbor agents additionally receive litellm
    `extra_body` requesting vLLM `return_token_ids` (and optional logprobs).
    Prefer also wrapping `api_base` in `token_capture.capture_proxy` when the
    agent harness may drop unknown kwargs — the proxy injects the flag
    server-side.
    """
    out: dict[str, Any] = {
        "model": served.harbor_model,
        "api_base": served.api_base,
        "model_info": served.model_info,
    }
    extra_body: dict[str, Any] = {
        "chat_template_kwargs": dict(
            CHAT_TEMPLATE_KWARGS if chat_template_kwargs is None else chat_template_kwargs
        )
    }
    if capture_tokens:
        from .token_capture import token_capture_agent_kwargs

        extra_body.update(
            token_capture_agent_kwargs(
                return_token_ids=True, logprobs=capture_logprobs
            ).get("extra_body", {})
        )
    # `llm_call_kwargs`, not `extra_body` directly. Terminus2 declares
    # `llm_call_kwargs` as a named parameter and splats it into
    # `self._llm.call(...)` (terminus_2.py:713); litellm then deep-merges its
    # `extra_body` into the request. Anything else falls into Terminus2's
    # `**kwargs`, is forwarded to `BaseAgent.__init__`, which accepts `**kwargs`
    # and never reads them — so a top-level `extra_body` is discarded in
    # silence, with no error and no warning. That is the prompt-seed finding
    # ("Harbor's terminus path dropped them"), and `LLM_CALL_KWARGS_KEY` exists
    # so a rename in harbor breaks a test instead of a run.
    out["agent_kwargs"] = {LLM_CALL_KWARGS_KEY: {"extra_body": extra_body}}
    return out


__all__ = [
    "CHAT_TEMPLATE_KWARGS",
    "DEFAULT_HOSTED_VLLM_MODEL_INFO",
    "LLM_CALL_KWARGS_KEY",
    "ServedModel",
    "dump_serve_record",
    "litellm_generate",
    "litellm_generate_captured",
    "serve_model",
    "served_to_harbor_kwargs",
]
