"""Attach to a vLLM server *we* started — EC2, SageMaker, bare metal, or local.

`serve.py` spawns a Modal container and owns its lifecycle. That is one way to
get an OpenAI-compatible endpoint, not the only one, and nothing downstream
depends on it: harbor reaches a candidate through litellm's `hosted_vllm/<name>`
provider given `api_base` + `model_info`, and `arms.py` takes the serve context
manager as an injectable (`ArmsConfig.serve_cm`). So a self-managed server drops
in behind the same seam.

This module is the AWS path. It *does not start or stop anything* — the server's
lifetime belongs to whoever launched it (an EC2 instance, a systemd unit, a
`vllm serve` in tmux). It waits for health, discovers the served model name, and
yields the same `ServedModel` the Modal path yields. Teardown is a no-op, which
is the correct semantics: exiting a Python context manager must not terminate a
GPU box someone else is paying for and may still be using.

Deliberately stdlib-only (`urllib`) — probing an endpoint must not require the
`train` extra, so `route`/`passk` on a laptop can point at a remote GPU.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .serve import DEFAULT_HOSTED_VLLM_MODEL_INFO, ServedModel

# Long enough to cover weight load for a 30B fp8 teacher off a cold page cache,
# short enough that a typo in --api-base fails the same minute you made it.
DEFAULT_WAIT_SECONDS = 20 * 60


class EndpointError(RuntimeError):
    """The endpoint is unreachable, unhealthy, or not OpenAI-compatible."""


def _normalise_base(api_base: str) -> str:
    """Return the `/v1` base. Accepts with or without the suffix."""
    base = api_base.strip().rstrip("/")
    if not base:
        raise EndpointError("api_base is empty")
    if not base.startswith(("http://", "https://")):
        raise EndpointError(
            f"api_base must include a scheme (http:// or https://), got {api_base!r}"
        )
    return base if base.endswith("/v1") else base + "/v1"


def _root_of(v1_base: str) -> str:
    return v1_base[: -len("/v1")]


def _get_json(url: str, *, timeout: float = 10.0) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_health(api_base: str, *, wait_seconds: float = DEFAULT_WAIT_SECONDS) -> None:
    """Block until the server answers, or raise EndpointError.

    Tries `/health` (vLLM's cheap liveness probe) and falls back to `/v1/models`,
    since some proxies expose only the OpenAI surface.
    """
    v1 = _normalise_base(api_base)
    health = _root_of(v1) + "/health"
    deadline = time.monotonic() + max(wait_seconds, 0.0)
    last: Exception | None = None
    while True:
        for url in (health, v1 + "/models"):
            try:
                _get_json(url, timeout=5.0)
                return
            except json.JSONDecodeError:
                # /health returns an empty 200 body on vLLM — reaching it is the
                # signal; a body was never the point.
                return
            except (urllib.error.URLError, OSError, urllib.error.HTTPError) as e:
                last = e
        if time.monotonic() >= deadline:
            raise EndpointError(
                f"no healthy OpenAI-compatible server at {v1} after "
                f"{wait_seconds:.0f}s (last error: {last}). Start one with "
                "`vllm serve <model> --host 0.0.0.0 --port 8000`, and check the "
                "security group allows this host on that port."
            )
        time.sleep(3.0)


def discover_model_name(api_base: str) -> str:
    """Read the served model name from `/v1/models`.

    vLLM serves under the full HF path unless `--served-model-name` says
    otherwise. Guessing it wrong produces a 404 on every rollout, several
    minutes into a sweep, so ask the server instead.
    """
    v1 = _normalise_base(api_base)
    try:
        payload = _get_json(v1 + "/models")
    except Exception as e:
        raise EndpointError(f"could not list models at {v1}/models: {e}") from e
    data = payload.get("data") if isinstance(payload, dict) else None
    if not data:
        raise EndpointError(f"{v1}/models returned no models: {payload!r}")
    name = data[0].get("id")
    if not name:
        raise EndpointError(f"{v1}/models entry has no id: {data[0]!r}")
    return str(name)


def _post(url: str, payload: dict[str, Any], *, timeout: float = 300.0) -> None:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout):
        return None


def _adapter_name(adapter_path: str) -> str:
    """Stable LoRA name for an adapter dir: `<arm-dir>-lora`.

    Arms are served one at a time under distinct names so a stale registration
    from A2 cannot answer A3's requests.
    """
    parts = [p for p in adapter_path.rstrip("/").split("/") if p and p != "adapter"]
    stem = parts[-1] if parts else "adapter"
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in stem)
    return f"{safe[:48]}-lora"


def load_lora_adapter(api_base: str, adapter_path: str, *, name: str | None = None) -> str:
    """Register a LoRA with a running vLLM at runtime; return the served name.

    Needs the server started with `--enable-lora` and
    `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`, and `adapter_path` must be readable *by
    the server process* — the same box, or a shared mount. This is what makes the
    A2/A3 arms work without restarting the server between arms, which on a
    self-managed GPU would otherwise mean reloading base weights per arm.
    """
    v1 = _normalise_base(api_base)
    lora_name = name or _adapter_name(adapter_path)
    try:
        _post(
            v1 + "/load_lora_adapter",
            {"lora_name": lora_name, "lora_path": adapter_path},
        )
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:400]
        except Exception:
            pass
        # Already-registered is success; anything else is not.
        if "already" in detail.lower():
            return lora_name
        raise EndpointError(
            f"runtime LoRA load failed (HTTP {e.code}): {detail}. Start vLLM with "
            "`--enable-lora` and `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`, and make sure "
            f"{adapter_path!r} is readable by the server process."
        ) from e
    except (urllib.error.URLError, OSError) as e:
        raise EndpointError(f"runtime LoRA load failed: {e}") from e
    return lora_name


def unload_lora_adapter(api_base: str, name: str) -> None:
    """Best-effort deregistration. Never raises — teardown must not mask results."""
    try:
        _post(_normalise_base(api_base) + "/unload_lora_adapter", {"lora_name": name})
    except Exception:
        pass


@contextmanager
def attach_endpoint(
    base_model: str,
    *,
    api_base: str,
    adapter_path: str | None = None,
    model_name: str | None = None,
    model_info: dict[str, Any] | None = None,
    gpu: str = "external",
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
) -> Iterator[ServedModel]:
    """Yield a `ServedModel` for an already-running server.

    With `adapter_path`, the adapter is registered over vLLM's runtime-LoRA API
    and served under its own name, then deregistered on exit. The *server* is
    never stopped: its lifetime belongs to whoever launched the instance, and
    exiting a context manager must not terminate a GPU box that may still be in
    use — the difference from `serve.serve_model`, which owns what it spawns.
    """
    wait_for_health(api_base, wait_seconds=wait_seconds)
    lora_name: str | None = None
    if adapter_path:
        lora_name = load_lora_adapter(api_base, adapter_path)
        name = lora_name
    else:
        name = model_name or discover_model_name(api_base)
    try:
        yield ServedModel(
            api_base=_normalise_base(api_base),
            model_name=name,
            model_info=dict(model_info or DEFAULT_HOSTED_VLLM_MODEL_INFO),
            adapter_path=adapter_path,
            gpu=gpu,
            base_model=base_model,
        )
    finally:
        if lora_name:
            unload_lora_adapter(api_base, lora_name)


def endpoint_serve_cm(
    api_base: str,
    *,
    model_name: str | None = None,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
):
    """Build a drop-in replacement for `serve.serve_model`.

    Matches the call signature `arms.run_arms` uses (`serve_cm(model, gpu=...)`,
    plus `adapter_path`/`model_info`), so `ArmsConfig.serve_cm=endpoint_serve_cm(url)`
    runs the entire A0–A4 sweep against your own GPU with nothing else changed.

    `gpu` is accepted and recorded but ignored — the hardware was chosen when the
    instance was launched, and silently re-labelling it in `arms.json` would put a
    false GPU in the provenance record.
    """

    @contextmanager
    def _cm(
        base_model: str,
        adapter_path: str | None = None,
        gpu: str = "external",
        model_info: dict[str, Any] | None = None,
    ) -> Iterator[ServedModel]:
        with attach_endpoint(
            base_model,
            api_base=api_base,
            adapter_path=adapter_path,
            model_name=model_name,
            model_info=model_info,
            gpu=gpu,
            wait_seconds=wait_seconds,
        ) as served:
            yield served

    return _cm


__all__ = [
    "DEFAULT_WAIT_SECONDS",
    "EndpointError",
    "attach_endpoint",
    "discover_model_name",
    "endpoint_serve_cm",
    "load_lora_adapter",
    "unload_lora_adapter",
    "wait_for_health",
]
