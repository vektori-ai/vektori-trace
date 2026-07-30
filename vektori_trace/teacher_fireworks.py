"""Fireworks-hosted `TeacherPool` — same protocol as `teacher.VllmTeacherPool`.

`teacher.py` and `docs/OPD.md` used to say no hosted API can score tokens you
supply. That was true of the generic OpenAI-compatible surface and is **not**
true of Fireworks' `/completions`, which accepts three things together:

- `prompt` as an **array of integers** — an already-tokenised prompt, so the ids
  the student sampled reach the teacher unchanged (no re-tokenisation boundary).
- `echo_last: N` — documented as "echo back the last N tokens of the prompt …
  useful for obtaining logprobs of the prompt suffix". That is `prompt_logprobs`
  restricted to a suffix, which is exactly the window OPD needs.
- `logprobs: true` + `top_logprobs: K` — the OpenAI-compatible response format,
  whose entries carry `token_id` on both the returned token and every
  alternative. Ids, not strings: `align_topk_rows` can consume them directly.

Fireworks built their own on-policy distillation recipe on this same mechanism
(`training.recipes.distillation_loop`, mode `sampled_reverse_kl`), which is the
strongest evidence the path is supported rather than incidental.

Two facts about the numbers, both of which belong in a run's provenance:

- `logprob` is the model logprob *before* temperature and sampling-filter
  renormalisation. That is the quantity OPD wants — log π_t(a_t) under the
  teacher's own distribution. The sibling field `sampling_logprob` is the
  post-filter one and would silently make the objective a KL against a
  temperature-warped teacher.
- Public inference caps `top_logprobs` at 5 (`--max-logprobs` default). The
  sampled-token path (`score_ids`) is unaffected; `score_ids_topk` cannot reach
  thunlp/OPD's K=16 here. `TOP_LOGPROBS_CAP` is enforced rather than clamped —
  silently running K=5 when the config said 16 would change the objective.

Alignment discipline
--------------------
`teacher.py` guarantees alignment by tokenising server-side and checking every
returned entry is keyed by the id it asked about. There is no `/tokenize` here,
so the equivalent guarantee comes from `token_id`: this module locates the
returned run of ids that equals `tokens` exactly and refuses to guess when it is
absent or ambiguous. Quantisation is the remaining caveat — Fireworks serves FP8,
so these logprobs carry quantisation noise inside a term the student
differentiates through (`docs/OPD.md`).

Stdlib-only, like `teacher.py` and `endpoint.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .teacher import TeacherScoringError, _post_json

#: Public Fireworks inference rejects `top_logprobs` above this.
TOP_LOGPROBS_CAP = 5

DEFAULT_FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1"

#: The pilot teacher (`docs/PILOT.md`: Qwen3-Coder-30B-A3B → Qwen3-8B), served.
DEFAULT_FIREWORKS_TEACHER = "accounts/fireworks/models/qwen3-coder-30b-a3b-instruct"


@dataclass
class FireworksTeacherPool:
    """`TeacherPool` over Fireworks serverless or a dedicated Fireworks deployment.

    `model` is a Fireworks resource path — `accounts/fireworks/models/<id>` for a
    served model, or `accounts/<account>/deployments/<id>` for your own. The
    tokenizer constraint from `docs/OPD.md` is unchanged and independent of
    hosting: `check_tokenizers` must pass for the ids sent here to mean the same
    strings to teacher and student.
    """

    model: str = DEFAULT_FIREWORKS_TEACHER
    api_key: str | None = None
    api_base: str = DEFAULT_FIREWORKS_BASE
    timeout: float = 120.0
    # Recorded in provenance so a run states which teacher produced its scores.
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.api_base = self.api_base.rstrip("/")
        if self.api_key is None:
            self.api_key = os.environ.get("FIREWORKS_API_KEY")
        if not self.api_key:
            raise TeacherScoringError(
                "no Fireworks API key — pass api_key= or set FIREWORKS_API_KEY"
            )

    # -- text ----------------------------------------------------------------

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        """Greedy continuation — the bisection/intervention path, not OPD scoring."""
        out = self._post(
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }
        )
        choices = out.get("choices") or []
        if not choices:
            raise TeacherScoringError(f"no choices in completion response: {out!r}")
        return choices[0].get("text") or ""

    # -- scoring -------------------------------------------------------------

    def prompt_logprobs(self, prompt: str, tokens: list[int]) -> list[float]:
        """Part of the `reopd.TeacherPool` protocol — unavailable here, by design.

        `teacher.VllmTeacherPool` implements this by tokenising the text prefix
        server-side (`POST /tokenize`) and handing the ids to `score_ids`.
        Fireworks exposes no tokenizer endpoint, and a *local* tokenizer is
        exactly the thing `teacher.py`'s alignment discipline forbids: a different
        revision than the loaded weights shifts a boundary and silently corrupts
        the objective rather than crashing.

        The OPD loop never calls this — `distill.run_opd_training` holds the
        prefix as ids already and calls `score_ids`.
        """
        _ = prompt, tokens
        raise TeacherScoringError(
            "FireworksTeacherPool cannot score a *text* prefix: Fireworks has no "
            "/tokenize endpoint, and tokenising locally risks the boundary shift "
            "this module exists to prevent. Use score_ids(prompt_ids, tokens)."
        )

    def score_ids(self, prompt_ids: list[int], tokens: list[int]) -> list[float]:
        """log π_t(t_i | prompt_ids, t_<i) for each id in `tokens`.

        Same contract as `teacher.VllmTeacherPool.score_ids`: the teacher never
        samples, it scores tokens the *student* sampled.
        """
        if not tokens:
            return []
        if not prompt_ids:
            raise TeacherScoringError(
                "prompt_ids is empty — the first token would have nothing "
                "conditioning it"
            )
        rows = self._scored_entries([int(t) for t in prompt_ids], tokens, top_k=0)
        return [_entry_logprob(entry, tid, position=i)
                for i, (entry, tid) in enumerate(zip(rows, tokens, strict=True))]

    def score_ids_topk(
        self, prompt_ids: list[int], tokens: list[int], top_k: int
    ) -> list[dict[int, float]]:
        """Teacher's top-`top_k` at each scored position, plus the sampled token.

        Capped at `TOP_LOGPROBS_CAP` by the public inference API. Unlike vLLM,
        Fireworks does not promise the prompt's own token appears in the returned
        alternatives, so the sampled token's own `logprob` (always present on the
        entry) is merged in explicitly — `align_topk_rows` requires it.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if top_k > TOP_LOGPROBS_CAP:
            raise TeacherScoringError(
                f"top_k={top_k} exceeds Fireworks' public `top_logprobs` cap of "
                f"{TOP_LOGPROBS_CAP}. Lower --top-k, or run the top-K objective "
                "against a self-hosted teacher (`teacher.VllmTeacherPool`), which "
                "reaches thunlp/OPD's K=16. Refusing to clamp: K is part of the "
                "objective, not a tuning knob."
            )
        if not tokens:
            return []
        if not prompt_ids:
            raise TeacherScoringError("prompt_ids is empty — nothing conditions the first token")
        rows = self._scored_entries([int(t) for t in prompt_ids], tokens, top_k=top_k)
        out: list[dict[int, float]] = []
        for i, (entry, token_id) in enumerate(zip(rows, tokens, strict=True)):
            row: dict[int, float] = {}
            for alt in entry.get("top_logprobs") or []:
                if not isinstance(alt, dict) or "token_id" not in alt:
                    raise TeacherScoringError(
                        f"top_logprobs entry at position {i} carries no token_id "
                        f"({alt!r}) — the legacy logprobs format keys alternatives by "
                        "token *string*, which cannot be mapped back to ids"
                    )
                row[int(alt["token_id"])] = float(alt["logprob"])
            # The sampled token always, even when outside the top-K.
            row[int(token_id)] = _entry_logprob(entry, token_id, position=i)
            out.append(row)
        return out

    def provenance(self) -> dict[str, Any]:
        """Record for arms.json / train reports — which teacher scored this run.

        `teacher_quantisation` is not cosmetic: Fireworks serves FP8, so a run's
        teacher logprobs carry quantisation noise that a self-hosted bf16 teacher
        does not. Two runs whose only difference is this field are not comparable.
        """
        return {
            "teacher_model": self.model,
            "teacher_api_base": self.api_base,
            "teacher_host": "fireworks",
            "teacher_quantisation": "fp8 (Fireworks serving default)",
            **self.meta,
        }

    # -- transport -----------------------------------------------------------

    def _post(self, payload: dict[str, Any]) -> Any:
        return _post_json(
            self.api_base + "/completions",
            payload,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def _scored_entries(
        self, prefix_ids: list[int], tokens: list[int], *, top_k: int
    ) -> list[dict[str, Any]]:
        """POST the scoring request and return one entry per token in `tokens`."""
        ids = [int(t) for t in tokens]
        payload: dict[str, Any] = {
            "model": self.model,
            # The integer-array `prompt` form, which is why no re-tokenisation
            # happens anywhere in this path.
            "prompt": [int(t) for t in prefix_ids] + ids,
            # `echo_last` is the whole point: score the suffix without paying to
            # ship logprobs for a 3.5k-token prefix we are not training on.
            "echo_last": len(ids),
            # Not 0: some deployments reject a zero-token generation. One token is
            # sampled and discarded; its entry is dropped by the id-run match below.
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": top_k,
        }
        out = self._post(payload)
        choices = out.get("choices") or []
        if not choices:
            raise TeacherScoringError(f"no choices in scoring response: {out!r}")
        logprobs = choices[0].get("logprobs")
        if not logprobs:
            raise TeacherScoringError(
                "response carries no `logprobs` — this deployment does not return "
                "them, or `echo_last` was dropped. OPD cannot run against it."
            )
        content = logprobs.get("content")
        if content is None:
            raise TeacherScoringError(
                "response used the legacy logprobs format (no `content`), whose "
                "alternatives are keyed by token string rather than token_id. "
                f"Cannot align. Got keys: {sorted(logprobs)}"
            )
        entries = [e for e in content if isinstance(e, dict)]
        window = _locate_id_run(entries, ids)
        return window


def _locate_id_run(entries: list[dict[str, Any]], ids: list[int]) -> list[dict[str, Any]]:
    """Return the sub-run of `entries` whose `token_id`s equal `ids` exactly.

    `echo_last=N` should put the scored tokens first, so position 0 is tried
    before scanning. A scan that finds the run more than once is an error, not a
    coin flip: `docs/OPD.md`'s whole failure mode is a one-token offset that
    corrupts the objective silently instead of crashing.
    """
    n = len(ids)
    got = []
    for i, e in enumerate(entries):
        if "token_id" not in e:
            raise TeacherScoringError(
                f"logprobs entry {i} carries no token_id ({e!r}) — cannot verify "
                "that the teacher scored the tokens we sent"
            )
        got.append(int(e["token_id"]))
    if got[:n] == ids:
        return entries[:n]
    hits = [s for s in range(len(got) - n + 1) if got[s : s + n] == ids]
    if len(hits) == 1:
        return entries[hits[0] : hits[0] + n]
    if not hits:
        raise TeacherScoringError(
            f"the {n} tokens we asked about do not appear in the returned "
            f"logprobs (got {len(got)} entries, first ids {got[:8]}, wanted "
            f"{ids[:8]}) — `echo_last` did not echo what we sent"
        )
    raise TeacherScoringError(
        f"the scored token run appears {len(hits)} times in the response at "
        f"offsets {hits[:5]} — refusing to guess which one is ours"
    )


def _entry_logprob(entry: dict[str, Any], token_id: int, *, position: int) -> float:
    """The teacher's own logprob for `token_id`, checked against the entry's id.

    Deliberately `logprob` and never `sampling_logprob`: the latter is
    renormalised by temperature and sampling filters, so training against it
    would be a KL to a warped teacher rather than to the teacher.
    """
    if int(entry.get("token_id", -1)) != int(token_id):
        raise TeacherScoringError(
            f"position {position}: teacher scored token {entry.get('token_id')} "
            f"but we sent {token_id} — the ids we supplied are not the ids it received"
        )
    lp = entry.get("logprob")
    if lp is None:
        raise TeacherScoringError(f"no logprob for token {token_id} at position {position}")
    return float(lp)


def fireworks_pool_from_env(
    *,
    model: str | None = None,
    api_base: str | None = None,
    require_scoring: bool = True,
) -> FireworksTeacherPool:
    """Build a pool and fail fast if this deployment cannot score supplied tokens.

    Same trade as `teacher.teacher_pool_from_endpoint`: one tiny request turns
    "OPD silently trains on garbage" into "OPD refuses to start". The probe uses
    literal ids because there is no server-side tokenizer to ask.
    """
    pool = FireworksTeacherPool(
        model=model or DEFAULT_FIREWORKS_TEACHER,
        api_base=api_base or DEFAULT_FIREWORKS_BASE,
    )
    if require_scoring:
        # Ids are arbitrary-but-valid for any Qwen3 vocab; the check is on the
        # shape of the response, not on what the tokens mean.
        probe_prefix = [9707, 11]
        probe_tokens = [1879, 0]
        scored = pool.score_ids(probe_prefix, probe_tokens)
        if len(scored) != len(probe_tokens):
            raise TeacherScoringError(
                f"probe returned {len(scored)} logprobs for {len(probe_tokens)} tokens"
            )
    return pool


__all__ = [
    "DEFAULT_FIREWORKS_BASE",
    "DEFAULT_FIREWORKS_TEACHER",
    "TOP_LOGPROBS_CAP",
    "FireworksTeacherPool",
    "fireworks_pool_from_env",
]
