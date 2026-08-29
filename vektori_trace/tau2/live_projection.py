"""Score a Qwen action's *semantics*, not its serialization.

The 2026-08-28 two-update proof handed the aligner a complete raw Qwen action --
`<think>`, `</think>`, `<tool_call>`, the JSON body, `<|im_end|>` and all -- and
asked DeepSeek for its likelihood. The pinned reference forbids exactly that
(`opd_reference/reward_manager_opd.py:337`):

    Both student_ids and teacher_ids should be the *response content only*
    (no chat template special tokens on either side).

Upstream goes further and *translates* chat-template markers between families
rather than passing one model's markup to the other
(`_build_chat_template_mapping`). Its "already clean" note on the Qwen->DeepSeek
branch means the replacement leaves no trailing newline -- NOT that the Qwen
marker is safe to score unchanged.

**Upstream stops at chat-template markers.** Verified against the vendored
revision: it contains no occurrence of `tool_call`, `DSML` or `invoke name`,
so it offers no Qwen-JSON -> DeepSeek-DSML conversion; the paper's experiments
did not exercise a multi-turn tool protocol. The two defects therefore have
different owners:

    <|im_end|>                      -> upstream's template translation covers it
    <tool_call>{JSON}</tool_call>   -> ours to project or exclude; nothing
                                       upstream to reuse

The measured cost of skipping that step, over the archived batch:

| class | n | mean advantage |
| --- | ---: | ---: |
| markup (`<tool_call>`, `<|im_end|>`) | 80 | **-24.47** |
| reasoning | 11,707 | -0.60 |
| content | 1,760 | -0.70 |
| tool json | 631 | -0.28 |

`<tool_call>` alone scored -55.3 to -50.8 on every occurrence: DeepSeek serves
tool calls as a DSML block (``<｜DSML｜invoke name=...>``) and has never emitted
Qwen's `<tool_call>` wrapper, so the score is a verdict on *notation*, not on
the decision. Masking those tokens is not enough on its own -- the reasoning
that follows is still *conditioned* on markup the teacher would not produce --
so the fix is to score a natively-rendered equivalent action and transfer the
result only where both sides represent the same bytes.

**This module is glue.** It adds no tokenizer, no parser and no aligner:

    split_generation()      live_agent      -- parse the Qwen action
    encode_messages()       encoding_dsv4   -- render DeepSeek-native
    align_by_bytes()        align           -- the existing dual pointer

What is new is only the decision about *which byte ranges are eligible* and the
bookkeeping that proves every student token was either supervised or excluded
with a stated reason.

Version one supervises **reasoning and visible content**, which are
byte-identical across the two renderings, and excludes tool-call serialization
entirely: the *semantics* of a tool call correspond (same name, same
arguments), but the bytes do not, and a heuristic span mapping is exactly the
kind of plausible-looking guess that produces a finite loss and a wrong number.
A token straddling a payload boundary is dropped whole and counted.
"""

from __future__ import annotations

#: Bumped when the semantic-projection scope changes -- which bytes are
#: eligible to carry teacher credit. Bound into score fingerprints: a cached
#: score is only valid for the projection that produced its spans.
#: - "v1": reasoning + visible content; Qwen markup, tool-call wrappers, tool
#:   JSON serialization, `<|im_end|>` and boundary-straddling tokens all carry
#:   zero weight. Tool calls are conditioned on but never credited, because
#:   Hermes JSON and DeepSeek DSML share no bytes to map through.
PROJECTION_VERSION = "v1"

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ProjectionError",
    "PayloadSpan",
    "ProjectedAction",
    "project_action",
]


class ProjectionError(RuntimeError):
    """An action whose semantics cannot be projected onto the teacher."""


#: Why a student token carries no OPD weight.
EXCLUDE_MARKUP = "control_markup"
EXCLUDE_TOOL = "tool_serialization"
EXCLUDE_STRADDLE = "straddles_payload_boundary"
EXCLUDE_OUTSIDE = "outside_any_payload"


@dataclass(frozen=True)
class PayloadSpan:
    """A byte range of the raw action that carries transferable semantics."""

    kind: str          # "reasoning" | "content"
    byte_start: int
    byte_end: int

    @property
    def n_bytes(self) -> int:
        return self.byte_end - self.byte_start


@dataclass
class ProjectedAction:
    """One action's eligibility map, with a reason for every exclusion."""

    payloads: list[PayloadSpan]
    #: Student token index -> payload kind, for tokens that may be supervised.
    supervised: dict[int, str] = field(default_factory=dict)
    #: Student token index -> reason, for tokens that may not.
    excluded: dict[int, str] = field(default_factory=dict)
    reasoning: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_supervised(self) -> int:
        return len(self.supervised)

    @property
    def n_excluded(self) -> int:
        return len(self.excluded)

    def report(self) -> dict[str, Any]:
        by_reason: dict[str, int] = {}
        for reason in self.excluded.values():
            by_reason[reason] = by_reason.get(reason, 0) + 1
        by_kind: dict[str, int] = {}
        for kind in self.supervised.values():
            by_kind[kind] = by_kind.get(kind, 0) + 1
        total = self.n_supervised + self.n_excluded
        return {
            "n_tokens": total,
            "n_supervised": self.n_supervised,
            "n_excluded": self.n_excluded,
            "retained_fraction": (
                round(self.n_supervised / total, 4) if total else 0.0
            ),
            "supervised_by_kind": by_kind,
            "excluded_by_reason": by_reason,
            "has_reasoning": bool((self.reasoning or "").strip()),
            "n_tool_calls": len(self.tool_calls),
        }


def _find_payload_spans(raw: str, reasoning: str | None, content: str | None) -> list[PayloadSpan]:
    """Byte ranges of the reasoning and content payloads inside `raw`.

    Located by search rather than reconstruction: the payload must be a literal
    substring of what the student actually emitted, or there is no byte
    correspondence to transfer a score across.
    """
    spans: list[PayloadSpan] = []
    raw_b = raw.encode("utf-8")

    # Search *after* the reasoning block for content, so a content string that
    # also appears inside the reasoning cannot capture the wrong bytes. A model
    # that reasons "I will say: Here you go." and then says "Here you go."
    # would otherwise map the content payload onto the reasoning occurrence --
    # silently, with every downstream assertion still passing.
    search_from = 0
    for kind, text in (("reasoning", reasoning), ("content", content)):
        if not text or not text.strip():
            continue
        payload = text.encode("utf-8")
        idx = raw_b.find(payload, search_from)
        if idx < 0:
            raise ProjectionError(
                f"{kind} payload is not a literal substring of the raw action "
                f"at or after byte {search_from}; there is no byte "
                "correspondence to transfer a teacher score across, and a "
                "fuzzy match would be a guess"
            )
        # Ambiguity is refused rather than resolved by position: two identical
        # candidate spans mean the mapping is not determined by the bytes.
        if raw_b.find(payload, idx + 1) >= 0 and kind == "content":
            raise ProjectionError(
                f"{kind} payload occurs more than once at or after byte "
                f"{search_from}; the byte range that carries the teacher score "
                "is ambiguous, and picking one would be a guess"
            )
        spans.append(PayloadSpan(kind, idx, idx + len(payload)))
        search_from = idx + len(payload)
    return spans


def project_action(
    raw_text: str,
    token_bytes: list[bytes],
) -> ProjectedAction:
    """Decide which student tokens are eligible for teacher-transferred credit.

    Reuses `live_agent.split_generation` for parsing -- the same function the
    capture path uses, so the projection can never disagree with what was
    archived.

    Every token lands in exactly one of `supervised` or `excluded`; the caller
    can assert `n_supervised + n_excluded == len(token_bytes)` and know nothing
    was silently dropped.
    """
    from vektori_trace.tau2.live_agent import split_generation

    reasoning, content, tool_calls = split_generation(raw_text)

    joined = b"".join(token_bytes)
    if joined != raw_text.encode("utf-8"):
        # The capture path allows a trailing special (`<|im_end|>`); anything
        # else means the token stream is not this action.
        if not joined.startswith(raw_text.encode("utf-8")):
            raise ProjectionError(
                "token bytes do not reconstruct the raw action; the projection "
                "would map spans onto a different byte string"
            )

    # A malformed action -- `<think>` opened and never closed, which is exactly
    # the update-1 regression -- makes `split_generation` return the whole
    # prefix as *content*, markup included. Supervising that would transfer the
    # teacher's opinion of `<think>` onto the student under the label
    # "content", which is the precise failure this module exists to prevent.
    # No closed reasoning block means no transferable reasoning payload.
    if reasoning is None and "<think>" in raw_text:
        content = None

    spans = _find_payload_spans(raw_text, reasoning, content)
    out = ProjectedAction(
        payloads=spans,
        reasoning=reasoning,
        content=content,
        tool_calls=list(tool_calls),
    )

    # Tool-call serialization is excluded wholesale in version one. Named
    # separately from generic markup so the report says *why* those tokens
    # carry no weight, and so a later version that projects tool semantics can
    # find them.
    tool_ranges: list[tuple[int, int]] = []
    raw_b = raw_text.encode("utf-8")
    start = 0
    while True:
        open_i = raw_b.find(b"<tool_call>", start)
        if open_i < 0:
            break
        close_i = raw_b.find(b"</tool_call>", open_i)
        end = (close_i + len(b"</tool_call>")) if close_i >= 0 else len(raw_b)
        tool_ranges.append((open_i, end))
        start = end

    pos = 0
    for i, piece in enumerate(token_bytes):
        t_start, t_end = pos, pos + len(piece)
        pos = t_end

        if any(t_start < e and t_end > s for s, e in tool_ranges):
            out.excluded[i] = EXCLUDE_TOOL
            continue

        covering = [
            sp for sp in spans if t_start < sp.byte_end and t_end > sp.byte_start
        ]
        if not covering:
            out.excluded[i] = EXCLUDE_OUTSIDE
            continue
        if len(covering) > 1:
            out.excluded[i] = EXCLUDE_STRADDLE
            continue
        sp = covering[0]
        # Fully inside, or it straddles the payload's own boundary -- which is
        # where the `<think>`/`</think>` markers live. Dropping the whole token
        # is deliberate: half a token cannot carry half an advantage.
        if t_start >= sp.byte_start and t_end <= sp.byte_end:
            out.supervised[i] = sp.kind
        else:
            out.excluded[i] = EXCLUDE_STRADDLE

    return out
