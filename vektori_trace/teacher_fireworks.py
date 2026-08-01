"""Fireworks-hosted `TeacherPool` — same protocol as `teacher.VllmTeacherPool`.

`teacher.py` and `docs/OPD.md` used to say no hosted API can score tokens you
supply. That was true of the generic OpenAI-compatible surface and is **not**
true of Fireworks' `/completions`, which accepts three things together:

- `prompt` as an **array of integers** — an already-tokenised prompt, so the ids
  the student sampled reach the teacher unchanged (no re-tokenisation boundary).
- `echo: true` (or `echo_last: N`, the cheaper documented variant) — logprobs
  over prompt positions, which is the window OPD needs.
- `logprobs: true` + `top_logprobs: K` — the OpenAI-compatible response format,
  whose entries carry `token_id` on both the returned token and every
  alternative. Ids, not strings: `align_topk_rows` can consume them directly.

This is not inferred from the API reference. Fireworks' own distillation recipe
does exactly it — `training/utils/distillation/sampling.py` in `fw-ai/cookbook`,
`_request_teacher_echo`, which posts::

    prompt=token_ids, max_tokens=1, temperature=0.0,
    logprobs=True, echo=True, raw_output=True, top_logprobs=K

against a frozen teacher deployment, then reads `choices[0].logprobs.content` for
one teacher logprob per sampled response token. Their `sampled_reverse_kl` mode
trains on `teacher_logprob - sampling_logprob` — the same objective as
`opd.reverse_kl_surrogate`. So the request shape here is the vendor's own, not a
reconstruction from prose.

Two details taken from that source rather than the docs: the teacher runs on a
**dedicated inference deployment** (the recipe auto-creates one when handed a
base model), and `top_logprobs` is omitted entirely rather than sent as 0.

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

#: Default Fireworks teacher id for hosted scoring (cross-tokenizer path uses
#: DeepSeek-V4-Flash; see FINAL-PLAN.md). Legacy same-vocab pilot used Qwen3.
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
    #: `"full"` sends `echo: true` — the shape Fireworks' own distillation recipe
    #: uses in production, and therefore the one known to work. `"last"` sends
    #: `echo_last: N`, which is documented and ships far fewer logprobs for a long
    #: prefix, but which the vendor's code does not use. Default to the proven one
    #: and treat the cheap one as an optimisation to verify with the probe.
    echo_mode: str = "full"
    # Recorded in provenance so a run states which teacher produced its scores.
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.api_base = self.api_base.rstrip("/")
        if self.echo_mode not in ("full", "last"):
            raise ValueError(f"echo_mode must be 'full' or 'last', got {self.echo_mode!r}")
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
            "teacher_echo_mode": self.echo_mode,
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
        prefix = [int(t) for t in prefix_ids]
        payload: dict[str, Any] = {
            "model": self.model,
            # The integer-array `prompt` form, which is why no re-tokenisation
            # happens anywhere in this path.
            "prompt": prefix + ids,
            # Not 0: some deployments reject a zero-token generation. One token is
            # sampled and discarded; its entry is dropped by the alignment below.
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            # Fireworks' own recipe sets this on every teacher scoring request.
            "raw_output": True,
        }
        if self.echo_mode == "last":
            # Cheaper: score the suffix without shipping logprobs for a
            # 3.5k-token prefix we are not training on. Documented, but not the
            # shape Fireworks' own code uses — see `echo_mode`.
            payload["echo_last"] = len(ids)
        else:
            payload["echo"] = True
        if top_k > 0:
            # Omitted rather than sent as 0, matching the vendor recipe: some
            # deployments treat a present-but-zero `top_logprobs` as a request for
            # alternatives they then refuse.
            payload["top_logprobs"] = top_k
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
        return _align_scored_entries(entries, len(prefix), ids, echo_mode=self.echo_mode)


def _align_scored_entries(
    entries: list[dict[str, Any]],
    prefix_len: int,
    ids: list[int],
    *,
    echo_mode: str = "full",
) -> list[dict[str, Any]]:
    """The entries that scored `ids`, verified by token id where possible.

    Two strategies, in order of how much they can prove:

    1. **By id.** The response schema marks `token_id` required, so normally every
       entry says which token it scored and the run equal to `ids` can be located
       outright. This survives any echo offset, which matters because `echo` and
       `echo_last` do not put the scored window in the same place.
    2. **By position**, only when the entries carry no ids at all. Fireworks' own
       recipe aligns this way — `content[1:]` predicts `tokens[1:]`, response
       tokens at `prompt_len - 1` — and falls back to tokenizer round-tripping,
       which is exactly the local-tokenizer risk `teacher.py` rules out. So this
       branch dispatches on the returned length instead, and refuses when the
       length matches neither layout.
    """
    n = len(ids)
    total = prefix_len + n
    got = [int(e["token_id"]) for e in entries if isinstance(e.get("token_id"), int)]

    if len(got) == len(entries) and entries:
        # Every entry is identified. `echo_last` puts the run first, so a match at
        # 0 is the answer rather than one candidate among many.
        if got[:n] == ids:
            return entries[:n]
        hits = [s for s in range(len(got) - n + 1) if got[s : s + n] == ids]
        if len(hits) == 1:
            return entries[hits[0] : hits[0] + n]
        if not hits:
            raise TeacherScoringError(
                f"the {n} tokens we asked about do not appear in the returned "
                f"logprobs (got {len(got)} entries, first ids {got[:8]}, wanted "
                f"{ids[:8]}) — the teacher did not echo what we sent"
            )
        raise TeacherScoringError(
            f"the scored token run appears {len(hits)} times in the response at "
            f"offsets {hits[:5]} — refusing to guess which one is ours"
        )

    # No ids anywhere: fall back to the layout the vendor's own code assumes.
    # Which layouts are admissible depends on what was *asked for* — accepting a
    # suffix-shaped response to an `echo: true` request would be a guess.
    if echo_mode == "last":
        if len(entries) in (n, n + 1):
            # Only the suffix was echoed, plus possibly the generated token.
            return entries[:n]
    else:
        if len(entries) == total:
            # content[i] scores token[i]; token 0 is unconditioned and meaningless.
            return entries[prefix_len:total]
        if len(entries) == total - 1:
            # content[i] scores token[i+1] — the P+C-1 "training-aligned" shape.
            return entries[prefix_len - 1 : prefix_len - 1 + n]
    raise TeacherScoringError(
        f"logprobs entries carry no token_id and their count ({len(entries)}) "
        f"matches no echo_mode={echo_mode!r} layout for a {total}-token prompt "
        f"scoring {n} tokens — refusing to guess the alignment"
    )


def _entry_logprob(entry: dict[str, Any], token_id: int, *, position: int) -> float:
    """The teacher's own logprob for `token_id`, checked against the entry's id.

    Deliberately `logprob` and never `sampling_logprob`: the latter is
    renormalised by temperature and sampling filters, so training against it
    would be a KL to a warped teacher rather than to the teacher.
    """
    got = entry.get("token_id")
    if isinstance(got, int) and got != int(token_id):
        raise TeacherScoringError(
            f"position {position}: teacher scored token {got} but we sent "
            f"{token_id} — the ids we supplied are not the ids it received"
        )
    # `got is None` means this deployment returns no ids and the entry was located
    # positionally instead. `_align_scored_entries` has already refused every
    # layout it cannot account for, so there is nothing further to check here.
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
