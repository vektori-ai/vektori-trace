"""Swap the served adapter between updates, and prove the swap happened.

ReOPD is on-policy in the action: `log pi_old` in the importance ratio must come
from the policy that actually sampled it. After update N the trainer holds a new
adapter, so before sampling update N+1 the *endpoint* has to serve that adapter
too. If it keeps serving CK35, every ratio compares two different distributions
-- finite loss, plausible metrics, wrong gradient, and nothing in any log to
show for it.

Why not restart vLLM
--------------------
Restarting the server per update costs ~3 minutes of model load, 32 times, on a
billing GPU. vLLM instead exposes `/v1/load_lora_adapter` when
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`, which swaps an adapter in seconds.

Why the verification is not optional
------------------------------------
The load endpoint returning 200 means the server accepted the request, not that
subsequent completions use the new weights. A silent no-op here produces exactly
the stale-policy failure this module exists to prevent, so the swap is confirmed
two ways: the new name appears in `/v1/models`, and a fixed probe prompt gives
different logprobs than it did under the previous adapter.

A probe that returns *identical* logprobs is treated as a failure. Two adapters
one optimizer step apart should differ somewhere on a several-token probe; if
they do not, either the update was a no-op or the swap did not take.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class RefreshError(RuntimeError):
    """The endpoint is not serving the policy the run believes it is."""


def _post(url: str, payload: dict, timeout: float) -> tuple[int, str]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read().decode()
        except Exception:
            return e.code, str(e)


def _get(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def served_models(api_base: str, *, timeout: float = 30.0) -> list[str]:
    body = _get(api_base.rstrip("/") + "/models", timeout)
    return [m.get("id") for m in (body.get("data") or [])]


def probe_logprobs(
    api_base: str,
    model: str,
    prompt_token_ids: list[int],
    *,
    n_tokens: int = 8,
    timeout: float = 120.0,
) -> list[float]:
    """Greedy logprobs for a fixed prompt, as a fingerprint of the policy.

    Temperature 0 so the result is a property of the weights rather than of the
    sampler: two calls to the same adapter must agree exactly, or the comparison
    against the next adapter means nothing.
    """
    status, raw = _post(
        api_base.rstrip("/") + "/completions",
        {"model": model, "prompt": prompt_token_ids, "max_tokens": n_tokens,
         "temperature": 0.0, "logprobs": 0},
        timeout,
    )
    if status != 200:
        raise RefreshError(f"probe failed HTTP {status}: {raw[:300]}")
    choice = (json.loads(raw).get("choices") or [{}])[0]
    lps = (choice.get("logprobs") or {}).get("token_logprobs") or []
    if not lps:
        raise RefreshError(
            f"probe returned no logprobs for {model!r}; the swap cannot be "
            "verified without them"
        )
    return [float(x) for x in lps]


def load_adapter(
    api_base: str,
    name: str,
    path: str,
    *,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """POST /v1/load_lora_adapter. Requires VLLM_ALLOW_RUNTIME_LORA_UPDATING."""
    status, raw = _post(
        api_base.rstrip("/") + "/load_lora_adapter",
        {"lora_name": name, "lora_path": path},
        timeout,
    )
    if status not in (200, 201):
        hint = ""
        if status == 404:
            hint = (" The endpoint has no /v1/load_lora_adapter route: it was "
                    "started without VLLM_ALLOW_RUNTIME_LORA_UPDATING=1.")
        raise RefreshError(f"load_lora_adapter HTTP {status}: {raw[:300]}.{hint}")
    return {"name": name, "path": path, "status": status}


def unload_adapter(api_base: str, name: str, *, timeout: float = 120.0) -> None:
    """Best effort. A stale adapter left loaded costs memory, not correctness."""
    _post(api_base.rstrip("/") + "/unload_lora_adapter",
          {"lora_name": name}, timeout)


def refresh_policy(
    api_base: str,
    *,
    new_name: str,
    new_path: str,
    probe_prompt_ids: list[int],
    previous_logprobs: list[float] | None = None,
    previous_name: str | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Serve `new_path` as `new_name`, and prove the endpoint really swapped.

    Returns a report carrying the new probe logprobs, which the caller passes
    back as `previous_logprobs` on the next refresh -- so each update is checked
    against the one before it rather than against a fixed baseline.
    """
    load_adapter(api_base, new_name, new_path, timeout=timeout)

    # 1. the server admits to serving it
    names = served_models(api_base)
    if new_name not in names:
        raise RefreshError(
            f"{new_name!r} is not in /v1/models after loading it ({names}). "
            "vLLM resolves an unknown name against the base weights, so "
            "sampling would silently use the wrong policy."
        )

    # 2. it behaves differently from the adapter it replaced
    lps = probe_logprobs(api_base, new_name, probe_prompt_ids, timeout=timeout)
    report: dict[str, Any] = {
        "name": new_name, "path": new_path,
        "probe_logprobs": lps, "n_probe_tokens": len(lps),
    }
    if previous_logprobs:
        n = min(len(lps), len(previous_logprobs))
        delta = max(abs(a - b) for a, b in zip(lps[:n], previous_logprobs[:n]))
        report["max_logprob_delta"] = delta
        if delta == 0.0:
            raise RefreshError(
                f"{new_name!r} gives logprobs identical to the previous policy "
                "on a greedy probe. Either the optimizer step was a no-op or "
                "the swap did not take; both mean the next update would sample "
                "from the wrong distribution."
            )

    if previous_name and previous_name != new_name:
        unload_adapter(api_base, previous_name)
        report["unloaded"] = previous_name
    return report


__all__ = [
    "RefreshError",
    "load_adapter",
    "probe_logprobs",
    "refresh_policy",
    "served_models",
    "unload_adapter",
]
