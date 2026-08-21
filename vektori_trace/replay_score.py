"""Scoring replay actions with DeepSeek (plan §8.2, §4).

One sampled ck75 action becomes `(teacher_token_bytes, teacher_logprobs)` — the
pair `replay_opd.build_replay_batch` needs. The §6.3 probe already proved the
transport; this is the adapter that feeds it replay data.

The rendering contract (§4) is the whole difficulty
---------------------------------------------------
An autoregressive log probability is only meaningful under the context it was
computed in, so the teacher must score the action *appended to its own render of
the same semantic history*:

    canonical messages -> pinned DeepSeek renderer -> teacher prefix
    teacher prefix + exact action bytes             -> tokenize jointly
                                                    -> score the action span

Three things follow, and each is a way to get a finite, wrong loss:

1. **Joint tokenisation, never concatenation.** `tokenize(prefix + action)` is
   not `tokenize(prefix) + tokenize(action)`; a token can straddle the boundary.
   The action span is located by taking the joint encoding's tail after the
   prefix's own ids, and verified by decoding it back to the action bytes.
2. **The renderer's turn terminator is not part of the action.** Closing an
   assistant turn appends EOS, which ck75 never sampled. Decoding with special
   tokens hidden makes such a span look byte-exact while carrying it, so the
   decode here keeps them visible — the same bug the Fireworks probe hit.
3. **The prefix is the student's semantic history, re-rendered.** Not the
   student's *string*: the two chat templates differ, and §4 requires the
   semantic messages to agree while the serialisations may not.

Cost note: every action carries its full prefix, so a 30k-token replay prefix is
re-sent per action. §8.3's 8 prefixes x 4 samples means each prefix's tokens are
paid four times. `score_replay_batch` reports that repeated-prefix cost because
§8.4 asks for a ledger, and it is the dominant term.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .replay_opd import ReplayOPDError, SampledAction


class ScoringError(ReplayOPDError):
    """An action could not be scored under the teacher's rendering."""


@dataclass
class ScoredAction:
    """One action's teacher-side tokenisation and per-token log probabilities."""

    key: str
    teacher_token_bytes: list[bytes]
    teacher_logprobs: list[float]
    n_prefix_tokens: int
    n_trailing_dropped: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_teacher_tokens(self) -> int:
        return len(self.teacher_token_bytes)

    def as_pair(self) -> tuple[list[bytes], list[float]]:
        """The shape `build_replay_batch` consumes."""
        return self.teacher_token_bytes, self.teacher_logprobs


def _decode(tok: Any, ids: list[int]) -> str:
    """Decode with special tokens visible — see point 2 in the module docstring."""
    for kwargs in ({"skip_special_tokens": False}, {}):
        try:
            return tok.decode(ids, **kwargs)
        except TypeError:
            continue
        except Exception:  # pragma: no cover - tokenizer flavour differences
            return ""
    return ""


def _token_bytes(tok: Any, ids: list[int]) -> list[bytes]:
    """Per-token byte strings, which is what `align_by_bytes` consumes."""
    return [_decode(tok, [int(i)]).encode() for i in ids]


def locate_action_span(
    prefix_messages: list[dict[str, Any]],
    action_text: str,
    teacher_tokenizer: Any,
    *,
    thinking_mode: str = "chat",
) -> tuple[list[int], list[int], int]:
    """Teacher ids for the prefix and for the action, by joint tokenisation.

    Returns `(prefix_ids, action_ids, n_trailing_dropped)`.

    Raises `ScoringError` rather than guessing when the joint encoding does not
    extend the prefix's, which is the boundary-straddle case §4 forbids
    resolving by offset arithmetic.
    """
    from .providers.teacher.cross import encode_teacher_ids, render_teacher_prefix

    prefix_text = render_teacher_prefix(prefix_messages, thinking_mode=thinking_mode)
    joint_text = render_teacher_prefix(
        [*prefix_messages, {"role": "assistant", "content": action_text}],
        thinking_mode=thinking_mode,
    )
    if action_text not in joint_text:
        raise ScoringError(
            "the DeepSeek renderer altered the action bytes; no byte-exact span "
            "exists to score"
        )

    prefix_ids = encode_teacher_ids(prefix_text, teacher_tokenizer)
    joint_ids = encode_teacher_ids(joint_text, teacher_tokenizer)
    if joint_ids[: len(prefix_ids)] != prefix_ids:
        raise ScoringError(
            "the joint encoding does not extend the prefix encoding — a token "
            "straddles the prefix/action boundary. §4 forbids resolving this "
            "with guessed offsets."
        )

    action_ids = joint_ids[len(prefix_ids):]
    dropped = 0
    while len(action_ids) > 1 and _decode(teacher_tokenizer, action_ids) != action_text:
        if not _decode(teacher_tokenizer, action_ids).startswith(action_text):
            break
        action_ids = action_ids[:-1]
        dropped += 1

    decoded = _decode(teacher_tokenizer, action_ids)
    if decoded != action_text:
        raise ScoringError(
            f"scored span decodes to {decoded[:80]!r}, expected {action_text[:80]!r}"
        )
    if not action_ids:
        raise ScoringError("action tokenised to zero teacher tokens")
    return prefix_ids, action_ids, dropped


def score_action(
    action: SampledAction,
    prefix_messages: list[dict[str, Any]],
    teacher_tokenizer: Any,
    pool: Any,
    *,
    thinking_mode: str = "chat",
) -> ScoredAction:
    """Score one ck75 action under DeepSeek's own render of the same history.

    `pool` is a `FireworksTeacherPool` (or anything with the same `score_ids`
    contract). Its refusals are not caught: `_align_scored_entries` raises when
    the echoed run is absent or ambiguous, and `_entry_logprob` raises on a
    token-id mismatch — those are §11 stop conditions and must reach the caller.
    """
    import math

    action_text = action.action_bytes.decode("utf-8")
    prefix_ids, action_ids, dropped = locate_action_span(
        prefix_messages, action_text, teacher_tokenizer, thinking_mode=thinking_mode
    )

    logprobs = pool.score_ids(prefix_ids, action_ids)
    if len(logprobs) != len(action_ids):
        raise ScoringError(
            f"{action.key}: teacher returned {len(logprobs)} logprobs for "
            f"{len(action_ids)} action tokens"
        )
    for i, v in enumerate(logprobs):
        if not math.isfinite(v):
            raise ScoringError(
                f"{action.key}: non-finite teacher logprob at position {i} "
                "(§11 stop condition)"
            )

    return ScoredAction(
        key=action.key,
        teacher_token_bytes=_token_bytes(teacher_tokenizer, action_ids),
        teacher_logprobs=[float(v) for v in logprobs],
        n_prefix_tokens=len(prefix_ids),
        n_trailing_dropped=dropped,
        meta={
            "n_student_tokens": len(action.action_token_ids),
            "n_teacher_tokens": len(action_ids),
        },
    )


def score_replay_batch(
    actions: list[SampledAction],
    prefix_messages_by_id: dict[str, list[dict[str, Any]]],
    teacher_tokenizer: Any,
    pool: Any,
    *,
    thinking_mode: str = "chat",
) -> tuple[dict[str, tuple[list[bytes], list[float]]], dict[str, Any]]:
    """Score every action; return the mapping `build_replay_batch` wants plus a ledger.

    Fails on the first unscoreable action rather than returning a partial map:
    `build_replay_batch` refuses a batch with a missing score anyway, and
    stopping here means the error names the action instead of the absence.
    """
    scored: dict[str, tuple[list[bytes], list[float]]] = {}
    rows: list[ScoredAction] = []

    for action in actions:
        messages = prefix_messages_by_id.get(action.prefix_id)
        if messages is None:
            raise ScoringError(
                f"{action.key}: no rendered prefix for {action.prefix_id}"
            )
        s = score_action(
            action, messages, teacher_tokenizer, pool, thinking_mode=thinking_mode
        )
        scored[action.key] = s.as_pair()
        rows.append(s)

    # §8.4's ledger. Prefix tokens dominate: each is re-sent once per sample, so
    # the repeated cost is what a batch actually pays the teacher for.
    prefix_tokens = {}
    for a, s in zip(actions, rows, strict=True):
        prefix_tokens.setdefault(a.prefix_id, s.n_prefix_tokens)
    total_scored = sum(s.n_teacher_tokens for s in rows)
    total_prefix_sent = sum(s.n_prefix_tokens for s in rows)

    ledger = {
        "n_actions": len(rows),
        "n_teacher_requests": len(rows),
        "teacher_scored_tokens": total_scored,
        "teacher_prefix_tokens_sent": total_prefix_sent,
        "unique_prefix_tokens": sum(prefix_tokens.values()),
        "repeated_prefix_tokens": total_prefix_sent - sum(prefix_tokens.values()),
        "teacher_input_tokens": total_prefix_sent + total_scored,
        "n_trailing_tokens_dropped": sum(s.n_trailing_dropped for s in rows),
        "student_vs_teacher_tokens": {
            s.key: (s.meta["n_student_tokens"], s.meta["n_teacher_tokens"])
            for s in rows
        },
    }
    return scored, ledger


__all__ = [
    "ScoredAction",
    "ScoringError",
    "locate_action_span",
    "score_action",
    "score_replay_batch",
]
