"""Sample the student at a frozen C30 prefix, byte-exactly.

Extracted from `tau2_swr_canary.py` rather than reimplemented. The canary's
capture path has already run against a live vLLM endpoint and carries fixes for
failures that are invisible without them -- a server silently truncating a
prompt from the front, ids that do not reconstruct the returned text, a cap hit
recorded as a completed action. Rewriting that for the training path would mean
rediscovering each one with paid teacher calls attached.

The request sends `prompt` as an **integer array**, not a string. C30 rows
carry the exact Qwen ids the corpus was built with, so re-rendering the messages
at training time would retokenize under whatever tokenizer happens to be
installed and quietly train on a different prefix than the one that was frozen.

What a capture must carry, and why
----------------------------------
- `behavior_logprobs`: `log pi_old` in the importance ratio. It cannot be
  recovered after sampling -- a later forward pass gives `log pi_current`,
  which is a different quantity -- so a capture without it is unusable and
  raising here is cheaper than discovering it at loss time.
- `action_token_bytes`: cross-tokenizer alignment operates on bytes, not text.
  Derived from the pinned tokenizer, never from `text.encode()`, which would
  assert nothing about the returned ids.
- `finish_reason`: a cap hit means the action is a fragment. Scoring a fragment
  as a completed decision is the failure that produced the 0/13 OPD run.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable

from ..replay_sample import token_bytes_from_ids

#: Only trailing template/whitespace tokens may separate the reconstructed
#: bytes from the text the server reported. Anything else is a real
#: id/text disagreement.
_ONLY_SPECIALS = re.compile(r"(<\|[^|>]+\|>|\s)+")


class ReOPDSampleError(RuntimeError):
    """A capture cannot be trusted as a scorable action."""


def post_json(url: str, payload: dict, timeout: float) -> tuple[int, dict]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"error": str(e)}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        # A timeout or a refused connection is not an HTTPError, so without this
        # it propagates raw. Callers that branch on the status code -- notably
        # verify_endpoint's long-prompt probe -- would crash on a slow-but-
        # healthy server instead of reporting a transport failure they can act
        # on. 0 is not a real HTTP status, which is the point: it cannot be
        # mistaken for a server-issued rejection.
        return 0, {"error": f"{type(e).__name__}: {e}"}


def capture_fingerprint(
    prefix_id: str,
    sample_index: int,
    *,
    model: str,
    policy_version: str,
    temperature: float,
    max_tokens: int,
    prompt_ids: list[int],
) -> str:
    """What an existing capture on disk is valid for.

    `max_tokens` is included because a capture taken under a smaller cap could
    be truncated, and `policy_version` because the adapter behind a model name
    changes on every update -- a resume that reused a capture from update 3
    inside update 7 would be training on an action the current policy never
    sampled, with `log pi_old` from the wrong distribution.
    """
    h = hashlib.sha256()
    for part in (prefix_id, str(sample_index), model, policy_version,
                 f"{temperature:.6f}", str(max_tokens)):
        h.update(part.encode())
        h.update(b"\x00")
    h.update(json.dumps(prompt_ids).encode())
    return h.hexdigest()[:32]


def build_capture(
    prefix_id: str,
    sample_index: int,
    body: dict,
    prompt_ids: list[int],
    fingerprint: str,
    tokenizer: Any,
    *,
    policy_version: str,
) -> dict:
    """One `/completions` response -> one durable, verified capture row."""
    choice = (body.get("choices") or [{}])[0]
    token_ids = choice.get("token_ids") or body.get("token_ids") or []
    got_prompt = choice.get("prompt_token_ids") or body.get("prompt_token_ids") or []
    logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
    text = choice.get("text")
    tag = f"{prefix_id}#{sample_index}"

    if not isinstance(text, str) or not text:
        raise ReOPDSampleError(f"{tag}: empty sampled action")
    if not token_ids or len(token_ids) != len(logprobs):
        raise ReOPDSampleError(
            f"{tag}: need one behavior logprob per action token "
            f"(got {len(token_ids)} ids, {len(logprobs)} logprobs). These are "
            "log pi_old and cannot be recovered after sampling."
        )
    if not got_prompt:
        raise ReOPDSampleError(
            f"{tag}: server omitted prompt ids, so the prompt it actually "
            "consumed cannot be verified"
        )
    if [int(x) for x in got_prompt] != prompt_ids:
        raise ReOPDSampleError(
            f"{tag}: server consumed different prompt ids than the frozen "
            "manifest holds. An over-long prompt is dropped from the front, "
            "which samples a state this run cannot describe."
        )
    if choice.get("finish_reason") == "length":
        raise ReOPDSampleError(
            f"{tag}: action hit max_tokens. A truncated action is a fragment, "
            "not a decision; scoring one is the failure that produced the 0/13 "
            "OPD run. Raise the cap rather than scoring it."
        )

    token_bytes = token_bytes_from_ids(tokenizer, [int(x) for x in token_ids])
    action_bytes = b"".join(token_bytes)

    # The ids must reconstruct the text as a PREFIX, not exactly: vLLM returns
    # the stop token among the sampled ids but renders `text` without it.
    recon = action_bytes.decode("utf-8", "replace")
    if not recon.startswith(text):
        raise ReOPDSampleError(
            f"{tag}: returned token ids do not reconstruct the returned text; "
            f"a capture that cannot be reproduced from its own ids is not "
            f"scorable.\n  text  {text[:80]!r}\n  recon {recon[:80]!r}"
        )
    trailer = recon[len(text):]
    if trailer and not _ONLY_SPECIALS.fullmatch(trailer):
        raise ReOPDSampleError(
            f"{tag}: ids reconstruct {trailer!r} beyond the returned text, "
            "which is not a trailing special token."
        )

    return {
        "key": tag,
        "prefix_id": prefix_id,
        "sample_index": sample_index,
        "fingerprint": fingerprint,
        "policy_version": policy_version,
        "served_model": body.get("model"),
        "request_id": body.get("id"),
        "prompt_token_ids": prompt_ids,
        "action_token_ids": [int(x) for x in token_ids],
        "behavior_logprobs": [float(x) for x in logprobs],
        "action_text": text,
        "action_bytes_b64": base64.b64encode(action_bytes).decode(),
        "action_token_bytes_b64": [base64.b64encode(b).decode()
                                   for b in token_bytes],
        "action_sha256": hashlib.sha256(action_bytes).hexdigest(),
        "n_action_bytes": len(action_bytes),
        "finish_reason": choice.get("finish_reason"),
    }


def sample_batch(
    prefixes: list[Any],
    *,
    api_base: str,
    model: str,
    tokenizer: Any,
    policy_version: str,
    max_tokens: int,
    temperature: float,
    n_samples: int = 1,
    timeout: float = 600.0,
    already: dict[str, dict] | None = None,
    on_capture: Callable[[dict], None] | None = None,
    poster: Callable[..., tuple[int, dict]] | None = None,
) -> list[dict]:
    """Sample every prefix, reusing any capture already on disk.

    `on_capture` is called with each capture as it lands so the caller can
    persist it immediately. A crash at sample 15 of 16 must not discard the
    GPU-generated actions before it -- their behaviour logprobs cannot be
    recreated.

    `poster` is injected so this is testable without an endpoint.
    """
    post = poster or post_json
    have = dict(already or {})
    out: list[dict] = []

    for prefix in prefixes:
        prompt_ids = list(prefix.prompt_token_ids)
        for i in range(n_samples):
            key = f"{prefix.prefix_id}#{i}"
            fp = capture_fingerprint(
                prefix.prefix_id, i, model=model, policy_version=policy_version,
                temperature=temperature, max_tokens=max_tokens,
                prompt_ids=prompt_ids,
            )
            prior = have.get(key)
            if prior is not None:
                if prior.get("fingerprint") != fp:
                    raise ReOPDSampleError(
                        f"{key}: capture on disk was taken under different "
                        "settings (model, policy version, temperature, cap or "
                        "prompt). Resuming into it would mix two runs."
                    )
                out.append(prior)
                continue

            status, body = post(
                api_base.rstrip("/") + "/completions",
                {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens,
                 "temperature": temperature, "logprobs": 0,
                 "return_token_ids": True},
                timeout,
            )
            if status != 200:
                raise ReOPDSampleError(f"{key}: HTTP {status}: {str(body)[:300]}")

            cap = build_capture(
                prefix.prefix_id, i, body, prompt_ids, fp, tokenizer,
                policy_version=policy_version,
            )
            out.append(cap)
            if on_capture is not None:
                on_capture(cap)
    return out


def capture_to_sampled_action(capture: dict, *, tokenizer: Any = None) -> Any:
    """A stored capture row -> the `SampledAction` the loss path consumes.

    Reconstructs from the persisted bytes rather than re-decoding the ids: the
    capture already proved those agree, and re-deriving them here would make a
    resumed run depend on the tokenizer still being byte-identical.
    """
    from ..replay_opd import SampledAction

    token_bytes = [base64.b64decode(b) for b in capture["action_token_bytes_b64"]]
    return SampledAction(
        prefix_id=capture["prefix_id"],
        sample_index=int(capture["sample_index"]),
        action_bytes=base64.b64decode(capture["action_bytes_b64"]),
        action_token_ids=[int(t) for t in capture["action_token_ids"]],
        action_token_bytes=token_bytes,
        behavior_logprobs=[float(x) for x in capture["behavior_logprobs"]],
        policy_version=capture["policy_version"],
        # Required by the optimizer step: `log pi_current` must be recomputed
        # under the same conditioning `log pi_old` was captured in, or the
        # importance ratio compares two different distributions while every
        # metric stays finite.
        prompt_token_ids=[int(t) for t in capture["prompt_token_ids"]],
        termination_reason=capture.get("finish_reason"),
    )


__all__ = [
    "ReOPDSampleError",
    "build_capture",
    "capture_fingerprint",
    "capture_to_sampled_action",
    "post_json",
    "sample_batch",
]
