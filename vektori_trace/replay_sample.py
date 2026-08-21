"""Turning ck75 captures into replay `SampledAction`s (plan §8.3).

`runtime/token_capture.CapturedCompletion` already carries what §5's importance
ratio needs — the sampled `token_ids` and the per-token `logprobs` vLLM returns
when `logprobs` is requested. This module converts one capture into one
`SampledAction`, and it exists because three things have to be checked at that
boundary and none of them is checkable later:

1. **Behaviour log probabilities must actually be present.** A capture taken
   without `capture_logprobs=True` has `logprobs=None`, and `log pi_old` cannot
   be recovered afterwards — the sampling run would have to be repeated. Failing
   here costs one error message; failing at training time costs the rollout.
2. **The ids must be the bytes.** `action_token_bytes` is what
   `align.align_by_bytes` aligns against, so it is derived from the ids with the
   pinned student tokenizer rather than from `text`. For a reasoning model the
   two disagree by construction: `CapturedCompletion.text` excludes the chain of
   thought while `token_ids` spans it.
3. **The cap must have bound the sample, not truncated it.** A capture whose
   `finish_reason` is "length" is a *truncated* action: its final tokens were
   cut mid-thought, and the teacher will score a fragment as though it were a
   completed action. §7.1's cap rule exists to make this rare; this check makes
   it visible when it happens anyway.

What this module does not do is sample. Driving Harbor-free generation at a
stored prefix belongs to whatever serves ck75; here we only adapt its output.
"""

from __future__ import annotations

from typing import Any

from .replay_opd import ReplayOPDError, SampledAction


class CaptureAdaptError(ReplayOPDError):
    """A capture cannot become a supervised replay action."""


def token_bytes_from_ids(tokenizer: Any, token_ids: list[int]) -> list[bytes]:
    """Per-token byte strings for `token_ids`, under the student tokenizer.

    `align.align_by_bytes` consumes exactly this: one `bytes` per token, whose
    concatenation is the action's bytes. Derived by decoding each id on its own
    and encoding the result, which is the only form that survives a tokenizer
    whose vocabulary is not byte-addressable through a public API.

    Special tokens are kept visible during decode. Dropping them would make a
    span carrying a template token decode to exactly the action bytes while
    still containing a token the model did not sample as content — the same
    class of bug the Fireworks probe hit with a trailing EOS.
    """
    out: list[bytes] = []
    for tid in token_ids:
        piece = None
        for kwargs in ({"skip_special_tokens": False}, {}):
            try:
                piece = tokenizer.decode([int(tid)], **kwargs)
                break
            except TypeError:
                continue
        if piece is None:
            raise CaptureAdaptError(f"tokenizer could not decode id {tid}")
        out.append(piece.encode())
    return out


def sampled_action_from_capture(
    capture: Any,
    tokenizer: Any,
    *,
    prefix_id: str,
    sample_index: int,
    policy_version: str,
    allow_truncated: bool = False,
) -> SampledAction:
    """One `CapturedCompletion` -> one `SampledAction`.

    `capture` is duck-typed rather than imported so this module stays usable
    against a capture loaded from JSONL as a plain object.

    Raises rather than returning None on a capture that cannot be supervised:
    silently skipping samples changes the batch size, and §8.3 fixes that at
    eight prefixes x four actions precisely so the denominator is known.
    """
    token_ids = [int(t) for t in (getattr(capture, "token_ids", None) or [])]
    if not token_ids:
        raise CaptureAdaptError(
            f"{prefix_id}#{sample_index}: capture has no sampled token ids"
        )

    logprobs = getattr(capture, "logprobs", None)
    if logprobs is None:
        raise CaptureAdaptError(
            f"{prefix_id}#{sample_index}: capture carries no per-token logprobs. "
            "These are log pi_old in the importance ratio and cannot be "
            "recovered after sampling — re-run generation with "
            "capture_logprobs=True rather than substituting anything here."
        )
    logprobs = [float(v) for v in logprobs]
    if len(logprobs) != len(token_ids):
        raise CaptureAdaptError(
            f"{prefix_id}#{sample_index}: {len(logprobs)} logprobs for "
            f"{len(token_ids)} sampled tokens — the capture is internally "
            "inconsistent and cannot be aligned"
        )

    finish = getattr(capture, "finish_reason", None)
    if finish == "length" and not allow_truncated:
        raise CaptureAdaptError(
            f"{prefix_id}#{sample_index}: finish_reason='length' — the cap cut "
            "this action mid-sequence. Scoring it would have the teacher grade a "
            "fragment as a completed action. Raise the task-derived cap (§7.1) or "
            "pass allow_truncated=True and record the cap-hit rate (§10)."
        )

    token_bytes = token_bytes_from_ids(tokenizer, token_ids)
    action_bytes = b"".join(token_bytes)

    return SampledAction(
        prefix_id=prefix_id,
        sample_index=sample_index,
        action_bytes=action_bytes,
        action_token_ids=token_ids,
        action_token_bytes=token_bytes,
        behavior_logprobs=logprobs,
        policy_version=policy_version,
        termination_reason=finish,
        meta={
            "request_id": getattr(capture, "request_id", None),
            "model": getattr(capture, "model", None),
            "n_prompt_tokens": len(getattr(capture, "prompt_token_ids", None) or []),
            "truncated": finish == "length",
        },
    )


def summarize_cap_hits(actions: list[SampledAction]) -> dict[str, Any]:
    """Cap-hit rate across a batch — §10's "cap rate".

    Worth reporting even when zero: a rising cap rate is one of §11's stop
    conditions ("sharp growth in length, cap hits, repetition"), and a rate that
    was never measured cannot be seen to rise.
    """
    n = len(actions)
    hits = sum(1 for a in actions if a.meta.get("truncated"))
    lengths = [len(a.action_token_ids) for a in actions]
    return {
        "n_actions": n,
        "n_cap_hits": hits,
        "cap_hit_rate": hits / n if n else 0.0,
        "mean_action_tokens": sum(lengths) / n if n else 0.0,
        "max_action_tokens": max(lengths, default=0),
        "min_action_tokens": min(lengths, default=0),
    }


__all__ = [
    "CaptureAdaptError",
    "sampled_action_from_capture",
    "summarize_cap_hits",
    "token_bytes_from_ids",
]
