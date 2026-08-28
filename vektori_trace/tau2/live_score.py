"""Score a live Qwen action through its DeepSeek-native semantic equivalent.

`replay_score.score_action` renders the teacher prefix and asks it to score the
**raw student action bytes**. For the frozen action-only replay corpus that was
right. For reasoning-inclusive live actions it is not: the bytes carry Qwen
control markup (`<think>`, `</think>`, `<tool_call>`, `<|im_end|>`) that
DeepSeek never emits, and the pinned reference forbids exactly this
(`opd_reference/reward_manager_opd.py:337` -- "response content only, no chat
template special tokens on either side").

Measured on the 2026-08-28 batch, `<tool_call>` drew -55.3 to -50.8 on *every*
occurrence while semantic classes sat near -0.6. That is a verdict on notation.

So this module scores the *semantics*:

    Qwen raw action
      -> split_generation()                 reasoning / content / tool_calls
      -> DeepSeek structured assistant msg  reasoning_content=, content=,
                                            tool_calls= (native DSML render)
      -> score each payload separately      teacher logprobs over payload bytes
      -> align_by_bytes() per payload       payload bytes are IDENTICAL on both
                                            sides, so the existing dual pointer
                                            applies without reinterpretation
      -> map back to Qwen token indices     markup, tool serialization,
                                            boundary-straddlers: zero weight

Why payload-wise rather than one span: the two renderings agree on the
reasoning and content *text* byte-for-byte, and disagree on everything around
it. Aligning each agreeing region separately is the only mapping that does not
require inventing a correspondence, and `align_by_bytes` already fails closed
when its two streams disagree in length -- which is the property that makes the
per-payload call safe.

Tool calls are scored by DeepSeek natively (so the teacher *conditions* on the
same decision the student made) but carry **no** transferred credit, because
the DSML and JSON serializations share no bytes to map through. Version one
supervises reasoning and visible content only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "LiveScoreError",
    "ProjectedScore",
    "score_live_action",
]


class LiveScoreError(RuntimeError):
    """A live action whose semantics could not be scored defensibly."""


@dataclass
class ProjectedScore:
    """Teacher credit for one live action, mapped onto student token indices.

    `advantage_weights` is the mask the training path must honour: a student
    token absent from `teacher_logprob_by_index` carries zero OPD weight, by
    construction rather than by a downstream filter that could be forgotten.
    """

    key: str
    #: student token index -> teacher logprob for the chunk covering it
    teacher_logprob_by_index: dict[int, float] = field(default_factory=dict)
    #: student token index -> why it carries no weight
    excluded: dict[int, str] = field(default_factory=dict)
    n_prefix_tokens: int = 0
    n_teacher_tokens: int = 0
    payload_report: dict[str, Any] = field(default_factory=dict)

    @property
    def n_supervised(self) -> int:
        return len(self.teacher_logprob_by_index)

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "n_supervised": self.n_supervised,
            "n_excluded": len(self.excluded),
            "n_prefix_tokens": self.n_prefix_tokens,
            "n_teacher_tokens": self.n_teacher_tokens,
            "payloads": self.payload_report,
        }


def _deepseek_assistant_message(
    reasoning: str | None,
    content: str | None,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """The DeepSeek-native form of one Qwen action.

    Mirrors `providers/teacher/cross.py`'s conversion so the two cannot drift:
    reasoning goes to `reasoning_content`, tool calls to OpenAI-shaped
    `tool_calls` which `encoding_dsv4` renders as a DSML block.
    """
    import json as _json

    msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.get("id", f"call_{i}"),
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": (
                        tc["arguments"]
                        if isinstance(tc.get("arguments"), str)
                        else _json.dumps(tc.get("arguments", {}))
                    ),
                },
            }
            for i, tc in enumerate(tool_calls)
        ]
    return msg


def _locate(haystack: str, needle: str, what: str) -> tuple[int, int]:
    idx = haystack.find(needle)
    if idx < 0:
        raise LiveScoreError(
            f"{what} payload is not a literal substring of the DeepSeek render; "
            "there is no byte correspondence to transfer a score across"
        )
    return idx, idx + len(needle)


def score_live_action(
    *,
    key: str,
    raw_text: str,
    student_token_bytes: list[bytes],
    semantic_history: list[dict[str, Any]],
    teacher_tokenizer: Any,
    pool: Any,
    thinking_mode: str = "thinking",
) -> ProjectedScore:
    """Teacher credit for the semantic payloads of one live Qwen action.

    Reuses, without reimplementing: `split_generation` (parse),
    `render_teacher_prefix`/`encode_teacher_ids` (DeepSeek render + ids),
    `align_by_bytes` (dual pointer). Everything here is the decision about
    *which* bytes are eligible and the arithmetic that maps a teacher chunk's
    logprob onto the student tokens covering the same bytes.
    """
    from vektori_trace.align import align_by_bytes
    from vektori_trace.providers.teacher.cross import (
        encode_teacher_ids,
        render_teacher_prefix,
    )
    from vektori_trace.tau2.live_agent import split_generation
    from vektori_trace.tau2.live_projection import project_action

    projected = project_action(raw_text, student_token_bytes)
    out = ProjectedScore(key=key, excluded=dict(projected.excluded))
    if not projected.payloads:
        # Nothing transferable -- e.g. the unterminated-`<think>` regression.
        out.payload_report = {"n_payloads": 0, "reason": "no eligible payload"}
        return out

    reasoning, content, tool_calls = split_generation(raw_text)
    assistant = _deepseek_assistant_message(reasoning, content, tool_calls)

    prefix_text = render_teacher_prefix(
        semantic_history, thinking_mode=thinking_mode
    )
    joint_text = render_teacher_prefix(
        [*semantic_history, assistant], thinking_mode=thinking_mode
    )
    if not joint_text.startswith(prefix_text):
        raise LiveScoreError(
            f"{key}: the DeepSeek joint render does not extend the prefix "
            "render; no scored span can be defined"
        )
    prefix_ids = encode_teacher_ids(prefix_text, teacher_tokenizer)
    joint_ids = encode_teacher_ids(joint_text, teacher_tokenizer)
    if joint_ids[: len(prefix_ids)] != prefix_ids:
        raise LiveScoreError(
            f"{key}: joint ids do not extend prefix ids -- a teacher token "
            "straddles the prefix/action boundary"
        )
    action_ids = joint_ids[len(prefix_ids):]
    if not action_ids:
        raise LiveScoreError(f"{key}: the native action rendered to zero tokens")

    logprobs = pool.score_ids(prefix_ids, action_ids)
    if len(logprobs) != len(action_ids):
        raise LiveScoreError(
            f"{key}: teacher returned {len(logprobs)} logprobs for "
            f"{len(action_ids)} action tokens"
        )
    for i, v in enumerate(logprobs):
        if not math.isfinite(v):
            raise LiveScoreError(
                f"{key}: non-finite teacher logprob at position {i}"
            )

    out.n_prefix_tokens = len(prefix_ids)
    out.n_teacher_tokens = len(action_ids)

    # Teacher-side byte layout of the rendered action, so a payload's bytes can
    # be located among the teacher tokens that cover them.
    # Reuse the scorer's own byte extractor rather than a second one: a
    # divergence here would move every payload boundary silently.
    from vektori_trace.replay_score import _token_bytes

    teacher_bytes = _token_bytes(teacher_tokenizer, action_ids)
    action_text_ds = joint_text[len(prefix_text):]

    report: dict[str, Any] = {"n_payloads": len(projected.payloads)}
    for span in projected.payloads:
        payload = raw_text.encode("utf-8")[span.byte_start : span.byte_end]
        payload_text = payload.decode("utf-8")

        # Same text on both sides -- that is the premise of the projection.
        d_start, d_end = _locate(action_text_ds, payload_text, span.kind)
        d_start_b = len(action_text_ds[:d_start].encode("utf-8"))
        d_end_b = len(action_text_ds[:d_end].encode("utf-8"))

        t_pieces: list[bytes] = []
        t_indices: list[int] = []
        pos = 0
        straddle = False
        for j, piece in enumerate(teacher_bytes):
            nxt = pos + len(piece)
            if pos < d_end_b and nxt > d_start_b:
                if pos < d_start_b or nxt > d_end_b:
                    straddle = True
                t_pieces.append(piece)
                t_indices.append(j)
            pos = nxt
        if straddle or not t_pieces:
            # A teacher token crossing the payload boundary cannot be split, so
            # the payload is skipped rather than scored on a guessed fraction.
            report[span.kind] = {"skipped": "teacher token straddles boundary"}
            for i, kind in list(projected.supervised.items()):
                if kind == span.kind:
                    out.excluded[i] = "teacher_boundary_straddle"
            continue

        s_indices = [i for i, k in projected.supervised.items() if k == span.kind]
        s_pieces = [student_token_bytes[i] for i in sorted(s_indices)]
        if b"".join(s_pieces) != b"".join(t_pieces):
            report[span.kind] = {"skipped": "student/teacher payload bytes differ"}
            for i in s_indices:
                out.excluded[i] = "payload_bytes_disagree"
            continue

        alignment = align_by_bytes(s_pieces, t_pieces)
        payload_lp = [float(logprobs[j]) for j in t_indices]
        ordered = sorted(s_indices)
        for chunk in alignment.spans:
            lp = sum(payload_lp[j] for j in chunk.teacher_idx)
            for si in chunk.student_idx:
                out.teacher_logprob_by_index[ordered[si]] = lp / len(chunk.student_idx)
        report[span.kind] = {
            "n_student_tokens": len(s_pieces),
            "n_teacher_tokens": len(t_pieces),
            "n_chunks": len(alignment.spans),
        }

    out.payload_report = report
    return out
