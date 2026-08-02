"""vLLM-backed `TeacherPool` — the one external call OPD depends on.

`reopd.TeacherPool` is a Protocol; until now the only implementation was
`InMemoryTeacherPool`, a fake returning a constant logprob. This is the real one:
it scores *supplied* tokens against the teacher via vLLM's `prompt_logprobs`.

The quantity OPD needs is log π_t(a_t) for the *student's* action. Plain
OpenAI-compatible `logprobs` is the wrong one — it covers only tokens the server
itself sampled. `prompt_logprobs` is a vLLM extension, not an OpenAI-API feature,
which is why PLAN.md C1 reaches for a self-hosted teacher.

Two hosted endpoints do nonetheless expose the right quantity through their own
extensions, and have their own pools rather than being forced through this one:
`providers.teacher.fireworks.FireworksTeacherPool` (`echo_last` over an integer-array
prompt) and `providers.teacher.bedrock.BedrockTeacherPool` (Custom Model Import's
`prompt_logprobs`). `docs/HOSTED_TEACHERS.md` covers the trade — quantisation and
a top-K cap in exchange for no GPU to hold. This module remains the reference
implementation and the only one with an unquantised teacher.

Backend-agnostic by construction — an EC2 instance running `vllm serve`, a Modal
container, or localhost all look identical here. Stdlib-only for the same reason
as `endpoint.py`.

Alignment discipline
--------------------
`opd.reverse_kl_surrogate` compares student and teacher logprobs positionally, so
a one-token offset silently corrupts the objective rather than crashing. Two
guards, both cheap:

- Tokenisation happens **on the server** (`POST /tokenize`), never with a local
  tokenizer that could be a different revision than the loaded weights.
- The returned count must equal `len(tokens)` exactly, and each returned entry
  must be keyed by the token id we asked about. Mismatch raises.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ...endpoint import EndpointError, _normalise_base, _root_of
from ...tokenizer_check import DEFAULT_TEACHER


class TeacherScoringError(RuntimeError):
    """The teacher endpoint cannot produce usable `prompt_logprobs`."""


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> Any:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Hosted teachers (`providers/teacher/fireworks.py`) need an Authorization
            # header; a self-hosted vLLM server needs none.
            **(headers or {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:500]
        except Exception:
            pass
        raise TeacherScoringError(f"POST {url} failed: HTTP {e.code} {detail}") from e
    except (urllib.error.URLError, OSError) as e:
        raise TeacherScoringError(f"POST {url} failed: {e}") from e


@dataclass
class VllmTeacherPool:
    """`TeacherPool` over a self-hosted vLLM OpenAI-compatible server.

    `model` must be the name the server *serves* (see `endpoint.discover_model_name`),
    not necessarily the HF path.
    """

    api_base: str
    model: str = DEFAULT_TEACHER
    timeout: float = 120.0
    # Recorded in provenance so a run states which teacher produced its scores.
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.api_base = _normalise_base(self.api_base)

    # -- text ----------------------------------------------------------------

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        """Greedy continuation — the bisection/intervention path, not OPD scoring."""
        out = _post_json(
            self.api_base + "/completions",
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=self.timeout,
        )
        choices = out.get("choices") or []
        if not choices:
            raise TeacherScoringError(f"no choices in completion response: {out!r}")
        return choices[0].get("text") or ""

    # -- scoring -------------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        """Server-side tokenisation, so ids match the loaded weights exactly."""
        url = _root_of(self.api_base) + "/tokenize"
        out = _post_json(
            url, {"model": self.model, "prompt": text}, timeout=self.timeout
        )
        tokens = out.get("tokens")
        if not isinstance(tokens, list):
            raise TeacherScoringError(
                f"{url} did not return a token list (got {out!r}). vLLM exposes "
                "/tokenize; a proxy in front of the server may be hiding it."
            )
        return [int(t) for t in tokens]

    def prompt_logprobs(self, prompt: str, tokens: list[int]) -> list[float]:
        """log π_t(t_i | prompt, t_<i) for each id in `tokens`.

        Text-prompt form, satisfying the `reopd.TeacherPool` protocol. The OPD
        loop uses `score_ids` instead: it already holds the prefix as ids, and
        round-tripping them through text is a re-tokenisation that could shift a
        boundary.
        """
        if not tokens:
            return []
        return self.score_ids(self.tokenize(prompt), tokens)

    def score_ids(self, prompt_ids: list[int], tokens: list[int]) -> list[float]:
        """`prompt_logprobs` over ids the caller already has — no re-tokenisation.

        `max_tokens=0` means "score, do not generate" — the teacher never samples
        here, which is the whole point: we are asking what the teacher thinks of
        the tokens the *student* sampled.

        Valid only because `check_tokenizers` establishes that teacher and student
        share a vocabulary byte-for-byte; otherwise these ids would mean different
        strings to the two models. The OPD loop enforces that check before it runs.
        """
        if not tokens:
            return []
        if not prompt_ids:
            raise TeacherScoringError(
                "prompt_ids is empty — the first token would have nothing "
                "conditioning it and vLLM returns None for that position"
            )
        tail = self._scored_tail(prompt_ids, tokens, top_k=0)
        return [
            _extract_logprob(entry, token_id, position=i)
            for i, (entry, token_id) in enumerate(zip(tail, tokens, strict=True))
        ]

    def score_ids_topk(
        self, prompt_ids: list[int], tokens: list[int], top_k: int
    ) -> list[dict[int, float]]:
        """Teacher's top-`top_k` logprobs at each scored position, plus the sampled token.

        Same request as `score_ids` with `prompt_logprobs=top_k` instead of 0, so it
        costs the teacher nothing extra — one round-trip either way.

        This is what thunlp/OPD's `LOG_PROB_TOP_K` retains (their default is 16),
        and their reported reason is the one that matters here: at student-visited
        states a small shared token set carries 97–99% of the probability mass, so a
        top-K set is nearly the whole distribution. Having it turns the objective
        from a single-sample estimate into an analytic KL over that set
        (`opd.topk_reverse_kl`) — same teacher cost, much lower gradient variance.

        Each returned dict maps token id → teacher logprob and always contains the
        token the student actually sampled, because vLLM includes the prompt's own
        token even when it falls outside the top-K.
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
                    f"position {i} — vLLM should always include the prompt's own token"
                )
            out.append(row)
        return out

    def _scored_tail(
        self, prompt_ids: list[int], tokens: list[int], *, top_k: int
    ) -> list[Any]:
        """POST the scoring request and return only the continuation's entries."""
        full = [int(t) for t in prompt_ids] + [int(t) for t in tokens]
        out = _post_json(
            self.api_base + "/completions",
            {
                "model": self.model,
                # vLLM accepts token ids in `prompt`, which avoids a second
                # tokenisation of the concatenation (whose boundary could differ
                # from tokenising the two pieces separately).
                "prompt": full,
                "max_tokens": 0,
                "temperature": 0.0,
                "prompt_logprobs": top_k,
            },
            timeout=self.timeout,
        )
        choices = out.get("choices") or []
        if not choices:
            raise TeacherScoringError(f"no choices in scoring response: {out!r}")
        raw = choices[0].get("prompt_logprobs")
        if raw is None:
            raise TeacherScoringError(
                "response carries no `prompt_logprobs` — this endpoint is not a "
                "vLLM server, or is fronted by a proxy that drops the field. OPD "
                "cannot run against it (PLAN.md C1)."
            )
        if len(raw) != len(full):
            raise TeacherScoringError(
                f"prompt_logprobs length {len(raw)} != prompt length {len(full)} — "
                "refusing to guess the alignment"
            )
        return list(raw[len(prompt_ids) :])

    def provenance(self) -> dict[str, Any]:
        """Record for arms.json / train reports — which teacher scored this run."""
        return {"teacher_model": self.model, "teacher_api_base": self.api_base, **self.meta}


def _extract_logprob(entry: Any, token_id: int, *, position: int) -> float:
    """Pull the logprob for `token_id` out of one vLLM prompt_logprobs entry.

    Entries are `{token_id: {"logprob": float, ...}}` with string-or-int keys
    depending on the JSON round-trip. The first prompt position is `None` (nothing
    conditions it) — reaching that here means the prompt/continuation split was
    wrong, so it is an error rather than a zero.
    """
    if entry is None:
        raise TeacherScoringError(
            f"prompt_logprobs is None at scored position {position} — the "
            "continuation was misaligned with the prompt"
        )
    if not isinstance(entry, dict):
        raise TeacherScoringError(f"unexpected prompt_logprobs entry: {entry!r}")
    for key in (str(token_id), token_id):
        if key in entry:
            value = entry[key]
            lp = value.get("logprob") if isinstance(value, dict) else value
            if lp is None:
                raise TeacherScoringError(
                    f"no logprob for token {token_id} at position {position}"
                )
            return float(lp)
    raise TeacherScoringError(
        f"token {token_id} absent from prompt_logprobs at position {position} "
        f"(keys: {list(entry)[:8]}) — teacher scored a different token than we sent, "
        "which means the ids we supplied are not the ids it received"
    )


def teacher_pool_from_endpoint(
    api_base: str,
    *,
    model: str | None = None,
    require_prompt_logprobs: bool = True,
) -> VllmTeacherPool:
    """Build a pool, resolving the served name and failing fast if scoring is absent.

    The check costs one two-token request and turns "OPD silently trains on
    garbage" into "OPD refuses to start", which is the trade V0_PLAN.md asks for
    everywhere else.
    """
    from ...endpoint import discover_model_name, wait_for_health

    wait_for_health(api_base)
    name = model or discover_model_name(api_base)
    pool = VllmTeacherPool(api_base=api_base, model=name)
    if require_prompt_logprobs:
        try:
            probe_ids = pool.tokenize(" ok")
            scored = pool.prompt_logprobs("probe:", probe_ids)
        except (TeacherScoringError, EndpointError) as e:
            raise TeacherScoringError(
                f"teacher at {api_base} cannot score supplied tokens: {e}"
            ) from e
        if len(scored) != len(probe_ids):
            raise TeacherScoringError(
                f"probe returned {len(scored)} logprobs for {len(probe_ids)} tokens"
            )
    return pool


__all__ = [
    "TeacherScoringError",
    "VllmTeacherPool",
    "teacher_pool_from_endpoint",
]


class InMemoryIdScoringPool:
    """Offline `score_ids` double for the OPD loop — deterministic, no network.

    Distinct from `reopd.InMemoryTeacherPool`, which predates `score_ids`: this
    one returns a *per-token* value derived from the id, so a test can tell
    whether the loop preserved token order rather than just token count.
    """

    def __init__(self, *, base: float = -0.5, per_token: float = -0.01):
        self.base = base
        self.per_token = per_token
        self.score_calls = 0
        self.last_prompt_ids: list[int] = []
        self.last_tokens: list[int] = []

    def score_ids(self, prompt_ids: list[int], tokens: list[int]) -> list[float]:
        self.score_calls += 1
        self.last_prompt_ids = list(prompt_ids)
        self.last_tokens = list(tokens)
        return [self.base + self.per_token * i for i, _ in enumerate(tokens)]

    def score_ids_topk(
        self, prompt_ids: list[int], tokens: list[int], top_k: int
    ) -> list[dict[int, float]]:
        """Top-K rows that always contain the sampled token, like vLLM's."""
        self.score_calls += 1
        self.last_prompt_ids = list(prompt_ids)
        self.last_tokens = list(tokens)
        rows = []
        for i, tid in enumerate(tokens):
            row = {int(tid): self.base + self.per_token * i}
            # Filler alternatives with lower logprobs, ids that cannot collide
            # with the sampled token.
            n = 0
            while len(row) < top_k:
                candidate = 10_000 + n
                if candidate != int(tid):
                    row[candidate] = self.base - 1.0 - n
                n += 1
            rows.append(row)
        return rows

    def prompt_logprobs(self, prompt: str, tokens: list[int]) -> list[float]:
        return self.score_ids([0], tokens)

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        _ = prompt, max_tokens
        return "ok"

    def provenance(self) -> dict[str, Any]:
        return {"teacher_model": "in-memory", "teacher_api_base": "none"}


__all__.append("InMemoryIdScoringPool")
