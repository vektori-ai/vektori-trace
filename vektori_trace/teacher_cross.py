"""Cross-tokenizer teacher wrapper for DeepSeek-V4-Flash (or any hosted teacher).

The core problem: OPD assumes teacher and student share a vocabulary, so the
student's ids can be sent directly to the teacher for scoring. When the teacher
is DeepSeek-V4-Flash and the student is Qwen3-8B they do not — different
merges, different added tokens, different special tokens.

This module solves it by working entirely in *teacher ids*:

1. The frozen prefix (from a ReOPD trajectory) is rendered into the
   DeepSeek-V4 chat format using ``encoding_dsv4.encode_messages``.
2. That rendered string is tokenised with the teacher's tokenizer to get
   teacher-side ids.
3. The student's sampled action text is also rendered and tokenised on the
   teacher side.
4. All of those ids are passed directly to the underlying ``IdScoringPool``
   (today ``FireworksTeacherPool``), which speaks teacher ids natively.

The vocabulary bridge (span alignment, cross-KL loss) lives elsewhere
(``vocab_bridge.py``, ``cross_kl.py``). This module is **only** the prefix
encoder + pool wrapper that produces teacher-side logprobs.

Offline-safe: no network, no torch. All teacher-tokenizer calls are pure CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .encoding_dsv4 import (
    ENCODING_DSV4_SHA256,
    encode_messages,
)
from .schema import Turn

# ---------------------------------------------------------------------------
# Protocols (duck-typed, no torch/transformers import needed in this file)
# ---------------------------------------------------------------------------


@runtime_checkable
class IdScoringPool(Protocol):
    """Minimal teacher API required by CrossTokenizerTeacherPool."""

    def score_ids(self, prompt_ids: list[int], tokens: list[int]) -> list[float]: ...

    def provenance(self) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Turn → OpenAI-style message helpers
# ---------------------------------------------------------------------------


def turns_to_openai_messages(turns: list[Turn]) -> list[dict[str, Any]]:
    """Convert ReOPD ``Turn`` objects to OpenAI-style message dicts.

    Rules mirroring ``dataset.turns_to_messages`` but producing the OpenAI
    format expected by ``encoding_dsv4.encode_messages``:

    - Subagent turns (``depth > 0``) are dropped.
    - Subagent-marker system turns are dropped.
    - ``Turn.thinking`` is kept as ``reasoning_content`` for assistant turns
      so the DeepSeek encoder can render it in thinking mode.
    - Tool calls are rendered in OpenAI ``tool_calls`` format.
    - Tool-result turns (``role == "tool"``) are passed through — the encoder
      handles merging them into the preceding user message.
    """
    messages: list[dict[str, Any]] = []
    for t in turns:
        if t.subagent_depth > 0:
            continue
        if t.role == "system" and (t.content or "").startswith("[subagent"):
            continue
        msg: dict[str, Any] = {"role": t.role}
        if t.content is not None:
            msg["content"] = t.content
        elif t.tool_calls:
            msg["content"] = ""
        if t.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": (
                            tc.args
                            if isinstance(tc.args, str)
                            else __import__("json").dumps(tc.args)
                        ),
                    },
                }
                for tc in t.tool_calls
            ]
        if t.role == "tool":
            msg["tool_call_id"] = t.tool_call_id or ""
        if t.thinking is not None and t.role == "assistant":
            msg["reasoning_content"] = t.thinking
        messages.append(msg)
    return messages


# ---------------------------------------------------------------------------
# Rendering and tokenisation
# ---------------------------------------------------------------------------


def render_teacher_prefix(
    messages: list[dict[str, Any]],
    *,
    thinking_mode: str = "chat",
) -> str:
    """Render OpenAI-style messages into the DeepSeek-V4 prompt string.

    Uses ``encoding_dsv4.encode_messages`` verbatim — this is the only render
    path for the teacher side. Using any other path (e.g. the student's Jinja
    chat template) would produce a different string, the teacher would score
    different conditioning, and the loss would be quietly wrong.

    ``thinking_mode`` must be ``"chat"`` or ``"thinking"``, matching the
    deployment's inference-time setting.
    """
    return encode_messages(messages, thinking_mode=thinking_mode)


def encode_teacher_ids(text: str, teacher_tokenizer: Any) -> list[int]:
    """Tokenise ``text`` with the teacher tokenizer; return a flat id list.

    Deliberately avoids ``add_special_tokens=True`` at the call site — the
    rendered string from ``render_teacher_prefix`` already contains the BOS
    and all turn boundaries, so doubling them would shift every token boundary.
    """
    ids = teacher_tokenizer.encode(text, add_special_tokens=False)
    # HF tokenizers may return a list or a tensor; normalise to list[int].
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(i) for i in ids]


# ---------------------------------------------------------------------------
# Prefix cache
# ---------------------------------------------------------------------------


@dataclass
class TeacherPrefixCache:
    """Keyed cache of teacher-side prefix ids.

    Key: ``(task, step_index, encoding_dsv4_hash, thinking_mode)``.

    The cache exists for **determinism**, not speed: re-rendering a frozen
    prefix at step 200 must produce the same ids as at step 5. Any render-path
    non-determinism (e.g. a ``<｜latest_reminder｜>`` date) would otherwise
    cause the teacher to condition on different text across steps. The cache
    surfaces that failure immediately, before it silently corrupts the loss.

    **A cache miss on an already-seen key that was computed differently is a
    hard error.** The only legal outcomes are hit (return cached ids) and
    first-time miss (store and return new ids).
    """

    _store: dict[tuple[str, int, str, str], list[int]] = field(
        default_factory=dict, init=False, repr=False
    )

    def get(
        self,
        task: str,
        step_index: int,
        *,
        thinking_mode: str = "chat",
    ) -> list[int] | None:
        key = (task, step_index, ENCODING_DSV4_SHA256, thinking_mode)
        return self._store.get(key)

    def put(
        self,
        task: str,
        step_index: int,
        ids: list[int],
        *,
        thinking_mode: str = "chat",
    ) -> None:
        """Store ``ids`` for ``(task, step_index)``.

        Raises ``ValueError`` if a different id list is already cached for
        this key — that means the prefix changed between calls, which is the
        bug the cache exists to detect.
        """
        key = (task, step_index, ENCODING_DSV4_SHA256, thinking_mode)
        existing = self._store.get(key)
        if existing is not None and existing != ids:
            raise ValueError(
                f"TeacherPrefixCache conflict for key {key!r}: "
                f"stored {len(existing)} ids, got {len(ids)} ids — "
                "the prefix changed between calls; this is a hard error"
            )
        self._store[key] = ids


# ---------------------------------------------------------------------------
# Cross-tokenizer teacher pool
# ---------------------------------------------------------------------------


@dataclass
class CrossTokenizerTeacherPool:
    """Wraps any ``IdScoringPool`` to work with teacher-side tokenisation.

    Caller contract
    ---------------
    ``prompt_ids`` and ``tokens`` passed to ``score_ids`` / ``score_ids_topk``
    **must already be teacher-side ids**. The bridge layer (``vocab_bridge``,
    ``distill.encode_prefix_pair``) is responsible for converting student ids
    to teacher ids before calling here. This class does not convert — it
    delegates directly to the underlying pool.

    Why the wrapper exists
    ----------------------
    ``FireworksTeacherPool.score_ids`` assumes the ids it receives are
    consistent with the model's own vocabulary — which they are once the caller
    has converted them. The wrapper exists to:

    - Carry ``teacher_tokenizer`` so that ``render_teacher_prefix`` /
      ``encode_teacher_ids`` are accessible from the same object that holds
      the pool.
    - Produce ``provenance()`` that records the ``encoding_dsv4`` hash and
      ``thinking_mode`` alongside the underlying pool's provenance.
    - Expose ``probe_echo_support()`` as a structured result rather than as a
      bare bool, so the probe can be checked programmatically.
    """

    pool: Any  # IdScoringPool — Any to avoid a circular import on Protocol
    teacher_tokenizer: Any
    thinking_mode: str = "chat"
    bridge: Any | None = None  # CrossTokenizerBridge, optional
    meta: dict[str, Any] = field(default_factory=dict)

    def score_ids(self, prompt_ids: list[int], tokens: list[int]) -> list[float]:
        """Delegate to the underlying pool.

        ``prompt_ids`` and ``tokens`` are already teacher-side ids.
        """
        return self.pool.score_ids(prompt_ids, tokens)

    def score_ids_topk(
        self, prompt_ids: list[int], tokens: list[int], top_k: int
    ) -> list[dict[int, float]]:
        """Delegate to the underlying pool's ``score_ids_topk`` if present."""
        if not hasattr(self.pool, "score_ids_topk"):
            raise AttributeError(
                f"{type(self.pool).__name__} does not implement score_ids_topk"
            )
        return self.pool.score_ids_topk(prompt_ids, tokens, top_k)

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        """Delegate to the underlying pool's ``generate`` if present."""
        if not hasattr(self.pool, "generate"):
            raise AttributeError(
                f"{type(self.pool).__name__} does not implement generate"
            )
        return self.pool.generate(prompt, max_tokens=max_tokens)

    def provenance(self) -> dict[str, Any]:
        """Provenance record for arms.json / train reports.

        Includes:
        - ``encoding_dsv4_sha256``: the hash of the vendored encoder; a change
          here means the rendered prefix changed and old/new runs are not
          comparable.
        - ``thinking_mode``: affects which special tokens the encoder inserts.
        - ``fp8_note``: reminder that Fireworks serves FP8 (if the pool records
          this, it is already in the underlying provenance).
        - The underlying pool's own provenance, merged in.
        """
        underlying = {}
        if hasattr(self.pool, "provenance"):
            underlying = self.pool.provenance()
        prov: dict[str, Any] = {
            "cross_tokenizer": True,
            "encoding_dsv4_sha256": ENCODING_DSV4_SHA256,
            "thinking_mode": self.thinking_mode,
            "fp8_note": (
                "Teacher logprobs may carry FP8 quantisation noise "
                "(Fireworks serving default) — see underlying provenance."
            ),
            **underlying,
            **self.meta,
        }
        return prov

    def probe_echo_support(
        self,
        *,
        probe_prefix: list[int] | None = None,
        probe_tokens: list[int] | None = None,
    ) -> dict[str, Any]:
        """P0 probe: verify the underlying pool can score supplied ids.

        Returns a structured dict so the result can be checked programmatically
        (``result["ok"]``). Does **not** raise on failure — the caller (CLI or
        test) decides what to do with a failing probe.

        In unit tests with a fake pool injected the probe runs entirely
        offline; no real API key is required.
        """
        if probe_prefix is None:
            # Arbitrary valid-looking ids — the probe checks response *shape*,
            # not that the tokens mean anything specific.
            probe_prefix = [9707, 11]
        if probe_tokens is None:
            probe_tokens = [1879, 0]

        try:
            scored = self.pool.score_ids(probe_prefix, probe_tokens)
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "probe_prefix": probe_prefix,
                "probe_tokens": probe_tokens,
            }

        if len(scored) != len(probe_tokens):
            return {
                "ok": False,
                "error": (
                    f"expected {len(probe_tokens)} logprobs, "
                    f"got {len(scored)}"
                ),
                "probe_prefix": probe_prefix,
                "probe_tokens": probe_tokens,
                "scored": scored,
            }

        # Shape check: every returned score must be a finite real number.
        # A wrong echo shape that still has the right *count* (e.g. nulls,
        # strings, NaN) would otherwise look like a passing P0 probe.
        bad: list[str] = []
        for i, v in enumerate(scored):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                bad.append(f"index {i}: type {type(v).__name__}")
                continue
            if v != v or v in (float("inf"), float("-inf")):  # NaN / Inf
                bad.append(f"index {i}: non-finite {v!r}")
        if bad:
            return {
                "ok": False,
                "error": "echo scores failed shape check: " + "; ".join(bad),
                "probe_prefix": probe_prefix,
                "probe_tokens": probe_tokens,
                "scored": scored,
            }

        return {
            "ok": True,
            "probe_prefix": probe_prefix,
            "probe_tokens": probe_tokens,
            "scored": scored,
            "n_returned": len(scored),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


__all__ = [
    "CrossTokenizerTeacherPool",
    "TeacherPrefixCache",
    "encode_teacher_ids",
    "render_teacher_prefix",
    "turns_to_openai_messages",
]
