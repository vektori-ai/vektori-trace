"""Bedrock Custom Model Import `TeacherPool` — same protocol as `teacher.py`.

Bedrock CMI serves your own weights (the pilot's `Qwen3-Coder-30B-A3B-Instruct`)
with no capacity reservation, which is what makes it viable when H100 capacity
never materialises on launch. What makes it viable *for OPD* is narrower and
newer: models imported after 2025-11-11 expose log probabilities, and AWS's own
matrix says the `OpenAICompletion` schema supports **prompt and output** token
logprobs, not just output.

    BedrockCompletion      output tokens only        (`max_gen_len` routes here)
    OpenAICompletion       prompt and output tokens  (`max_tokens` routes here)
    OpenAIChatCompletion   prompt and output tokens

`max_tokens` versus `max_gen_len` is the router, and it is load-bearing: the
BedrockCompletion schema returns `logprobs` for generated tokens only, which is
the wrong quantity — OPD needs log π_t(a_t) for tokens the *student* sampled.

The one thing this module cannot know from documentation
--------------------------------------------------------
AWS documents `prompt_logprobs=N` on `OpenAIChatCompletion` with a `messages`
payload. It documents `OpenAICompletion` as supporting prompt logprobs but does
not show that request, and it never states whether `prompt` accepts an **integer
array**. Both are load-bearing here and neither is assumable, so
`cli.py probe-teacher --backend bedrock` sends exactly one request and reports
what came back. Until that has run against a real imported model, treat this
module as unverified — `docs/HOSTED_TEACHERS.md` tracks the open question.

Response shape is handled defensively for the same reason. AWS's own vision
example shows `prompt_logprobs` and `prompt_token_ids` as **top-level** response
fields (vLLM's own shape leaking through), while vLLM's OpenAI server nests
`prompt_logprobs` under `choices[0]`. This module accepts either and says which
one it found, rather than picking one and failing obscurely against the other.

`boto3` is imported lazily so the package keeps installing without AWS extras.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .teacher import TeacherScoringError, _extract_logprob

#: Only the four CMI regions; `ap-south-1` (the repo's usual default) is not one.
CMI_REGIONS = ("us-east-1", "us-west-2", "eu-central-1", "ap-northeast-1")


@dataclass
class BedrockTeacherPool:
    """`TeacherPool` over a Bedrock Custom Model Import deployment.

    `model_id` is the imported model's ARN (or its short id). `region` must be a
    CMI region; passing one that is not gives a 404 on `invoke_model` that reads
    like a missing model rather than a missing feature, so it is checked here.
    """

    model_id: str
    region: str = "us-east-1"
    timeout: float = 120.0
    client: Any | None = None
    # Recorded in provenance so a run states which teacher produced its scores.
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.region not in CMI_REGIONS:
            raise TeacherScoringError(
                f"region {self.region!r} does not support Bedrock Custom Model "
                f"Import — use one of {', '.join(CMI_REGIONS)}"
            )
        if self.client is None:
            try:
                import boto3
            except ImportError as e:  # pragma: no cover - depends on extras
                raise TeacherScoringError(
                    "boto3 is required for the Bedrock teacher: "
                    "`uv sync --extra aws`"
                ) from e
            from botocore.config import Config

            self.client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=Config(read_timeout=self.timeout, retries={"max_attempts": 3}),
            )

    # -- text ----------------------------------------------------------------

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        """Greedy continuation — the bisection/intervention path, not OPD scoring."""
        out = self._invoke(
            {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
        )
        choices = out.get("choices") or []
        if not choices:
            raise TeacherScoringError(f"no choices in completion response: {out!r}")
        return choices[0].get("text") or ""

    # -- scoring -------------------------------------------------------------

    def prompt_logprobs(self, prompt: str, tokens: list[int]) -> list[float]:
        """Part of the `reopd.TeacherPool` protocol — unavailable here, by design.

        Same reason as `teacher_fireworks.FireworksTeacherPool.prompt_logprobs`:
        Bedrock exposes no tokenizer endpoint, and tokenising locally is the
        boundary-shift risk `teacher.py` exists to rule out. The OPD loop holds
        the prefix as ids and calls `score_ids`.
        """
        _ = prompt, tokens
        raise TeacherScoringError(
            "BedrockTeacherPool cannot score a *text* prefix: Bedrock has no "
            "/tokenize endpoint. Use score_ids(prompt_ids, tokens)."
        )

    def score_ids(self, prompt_ids: list[int], tokens: list[int]) -> list[float]:
        """log π_t(t_i | prompt_ids, t_<i) for each id in `tokens`."""
        if not tokens:
            return []
        if not prompt_ids:
            raise TeacherScoringError(
                "prompt_ids is empty — the first token would have nothing "
                "conditioning it and the first prompt_logprobs entry is null"
            )
        tail = self._scored_tail(prompt_ids, tokens, top_k=1)
        return [
            _extract_logprob(entry, token_id, position=i)
            for i, (entry, token_id) in enumerate(zip(tail, tokens, strict=True))
        ]

    def score_ids_topk(
        self, prompt_ids: list[int], tokens: list[int], top_k: int
    ) -> list[dict[int, float]]:
        """Teacher's top-`top_k` at each scored position, plus the sampled token.

        The response shape is vLLM's — `{token_id: {"logprob": ...}}` per position
        — so this reuses `teacher.py`'s parsing rather than a second copy of it.
        Unlike the Fireworks path there is no documented cap on `prompt_logprobs`,
        so thunlp/OPD's K=16 is reachable in principle; the probe is what settles
        whether this deployment honours it.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not tokens:
            return []
        if not prompt_ids:
            raise TeacherScoringError("prompt_ids is empty — nothing conditions the first token")
        tail = self._scored_tail(prompt_ids, tokens, top_k=top_k)
        out: list[dict[int, float]] = []
        for i, (entry, token_id) in enumerate(zip(tail, tokens, strict=True)):
            if not isinstance(entry, dict):
                raise TeacherScoringError(f"unexpected prompt_logprobs entry: {entry!r}")
            row: dict[int, float] = {}
            for key, value in entry.items():
                lp = value.get("logprob") if isinstance(value, dict) else value
                if lp is None:
                    continue
                row[int(key)] = float(lp)
            if token_id not in row:
                raise TeacherScoringError(
                    f"sampled token {token_id} missing from the top-{top_k} entry at "
                    f"position {i} — vLLM always includes the prompt's own token, so "
                    "this response did not come from the tokens we sent"
                )
            out.append(row)
        return out

    # -- transport -----------------------------------------------------------

    def _invoke(self, payload: dict[str, Any]) -> Any:
        """One `invoke_model` call. `max_tokens` (never `max_gen_len`) is the router
        that selects the OpenAICompletion schema, the only one with prompt logprobs."""
        try:
            resp = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload),
                accept="application/json",
                contentType="application/json",
            )
        except Exception as e:  # botocore raises a family of ClientErrors
            raise TeacherScoringError(
                f"invoke_model on {self.model_id} ({self.region}) failed: {e}"
            ) from e
        return json.loads(resp["body"].read())

    def _scored_tail(
        self, prompt_ids: list[int], tokens: list[int], *, top_k: int
    ) -> list[Any]:
        """POST the scoring request and return only the continuation's entries."""
        full = [int(t) for t in prompt_ids] + [int(t) for t in tokens]
        out = self._invoke(
            {
                # The integer-array form. Whether CMI's OpenAICompletion accepts
                # it is the open question the probe answers; a 400 here is a
                # finding, not a bug to work around.
                "prompt": full,
                "max_tokens": 1,
                "temperature": 0.0,
                "prompt_logprobs": top_k,
            }
        )
        raw, where = _find_prompt_logprobs(out)
        if raw is None:
            raise TeacherScoringError(
                "response carries no `prompt_logprobs` at the top level or under "
                "choices[0]. Either this model was imported before 2025-11-11 (no "
                "logprob support) or the request routed to BedrockCompletion, which "
                f"scores generated tokens only. Response keys: {sorted(out)}"
            )
        if len(raw) != len(full):
            raise TeacherScoringError(
                f"prompt_logprobs length {len(raw)} != prompt length {len(full)} "
                f"(found at {where}) — refusing to guess the alignment"
            )
        return list(raw[len(prompt_ids) :])

    def provenance(self) -> dict[str, Any]:
        """Record for arms.json / train reports — which teacher scored this run."""
        return {
            "teacher_model": self.model_id,
            "teacher_api_base": f"bedrock:{self.region}",
            "teacher_host": "bedrock-cmi",
            **self.meta,
        }


def _find_prompt_logprobs(out: dict[str, Any]) -> tuple[list[Any] | None, str]:
    """`prompt_logprobs` from either shape, and which shape it was.

    AWS's documented CMI response puts it top-level; vLLM's OpenAI server nests it
    under `choices[0]`. Both are plausible for an imported model and the doc does
    not settle it, so accept either and report which — the probe records this so
    the next reader does not have to rediscover it.
    """
    top = out.get("prompt_logprobs")
    if top is not None:
        return list(top), "top level"
    choices = out.get("choices") or []
    if choices and isinstance(choices[0], dict):
        nested = choices[0].get("prompt_logprobs")
        if nested is not None:
            return list(nested), "choices[0]"
    return None, "nowhere"


__all__ = ["CMI_REGIONS", "BedrockTeacherPool"]
