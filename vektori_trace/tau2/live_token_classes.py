"""Which tokens an OPD update actually pushes on, by class.

The two-update proof (2026-08-28) produced a valid reverse-KL signal -- 3,388
positive and 10,713 negative token advantages, not the uniform suppression a
reading of the 31 *action-level means* suggested, since a mean over ~457 tokens
says nothing about the signs beneath it. But it also produced a single
`-55.289` outlier under `advantage_clamp=None`, and `</think>` came back
negative in 27 of 31 turns. A global histogram cannot tell those apart.

This module answers the narrower question: **for each class of token, what did
the teacher actually reward or punish?**

The classes exist because a cross-model advantage conflates two different
preferences:

- **semantic** -- was this reasoning, this tool, this argument a good choice?
- **syntactic** -- does DeepSeek like *Qwen's serialization* of it?

Only the first should ever reach the student. `<think>`, `</think>`,
`<tool_call>` and `</tool_call>` are Qwen control markup with no counterpart in
DeepSeek's native rendering, so a teacher score transferred onto them is a
preference about a representation the teacher does not use. That is the
mechanism that makes `MARKUP` the class to mask, and the reason this file
classifies before anything is masked: masking a class we have not measured is
how a real signal gets thrown away alongside a spurious one.

Pure analysis -- no GPU, no teacher call, no training. It reads an archived
update and reports; `classify_action` is also what a masking implementation
should consume, so the mask and the report can never disagree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "TokenClass",
    "ClassStats",
    "classify_action",
    "class_report",
    "MARKUP_PATTERNS",
]


class TokenClass:
    """What a supervised student token is part of."""

    #: Qwen control markup. No DeepSeek counterpart; a transferred score here
    #: is a preference about serialization, not about the decision.
    MARKUP = "markup"
    #: Text inside `<think>...</think>`.
    REASONING = "reasoning"
    #: The JSON body of a tool call -- name and arguments.
    TOOL_JSON = "tool_json"
    #: Visible assistant text outside both.
    CONTENT = "content"


#: Byte patterns that are pure control markup. Ordered longest-first so
#: `</think>` cannot be partially matched by `<think>`.
MARKUP_PATTERNS: tuple[bytes, ...] = (
    b"</tool_call>",
    b"<tool_call>",
    b"</think>",
    b"<think>",
)

_TOOL_BLOCK = re.compile(rb"<tool_call>(.*?)</tool_call>", re.DOTALL)
_THINK_BLOCK = re.compile(rb"<think>(.*?)</think>", re.DOTALL)
#: An unterminated think block -- exactly what update 1 produced. Classified so
#: a malformed action can still be analysed rather than silently skipped.
_THINK_OPEN = re.compile(rb"<think>(.*)", re.DOTALL)


def _byte_classes(raw: bytes) -> list[str]:
    """Class of every byte in a raw Qwen action.

    Markup is stamped last so it wins over the block interiors it delimits: a
    `</think>` sits inside no region but must never be counted as reasoning.
    """
    classes = [TokenClass.CONTENT] * len(raw)

    for m in _THINK_BLOCK.finditer(raw):
        for i in range(m.start(1), m.end(1)):
            classes[i] = TokenClass.REASONING
    if not _THINK_BLOCK.search(raw):
        # Unterminated `<think>` (the update-1 regression). Everything after
        # the opener up to a tool call is still reasoning payload.
        m = _THINK_OPEN.search(raw)
        if m is not None:
            end = m.end(1)
            tool = _TOOL_BLOCK.search(raw)
            if tool is not None and tool.start() > m.start(1):
                end = tool.start()
            for i in range(m.start(1), end):
                classes[i] = TokenClass.REASONING

    for m in _TOOL_BLOCK.finditer(raw):
        for i in range(m.start(1), m.end(1)):
            classes[i] = TokenClass.TOOL_JSON

    for pat in MARKUP_PATTERNS:
        start = 0
        while True:
            idx = raw.find(pat, start)
            if idx < 0:
                break
            for i in range(idx, idx + len(pat)):
                classes[i] = TokenClass.MARKUP
            start = idx + len(pat)
    return classes


def classify_action(token_bytes: list[bytes]) -> list[str]:
    """Class per student token, from its byte span in the raw action.

    A token is markup only if **every** byte it covers is markup: a token
    straddling `</think>` and following text carries real content and must not
    be masked wholesale. That conservative rule is deliberate -- over-masking
    silently deletes supervision, and the resulting run still looks healthy.
    """
    raw = b"".join(token_bytes)
    byte_cls = _byte_classes(raw)
    out: list[str] = []
    pos = 0
    for piece in token_bytes:
        span = byte_cls[pos : pos + len(piece)]
        pos += len(piece)
        if not span:
            out.append(TokenClass.CONTENT)
        elif all(c == TokenClass.MARKUP for c in span):
            out.append(TokenClass.MARKUP)
        else:
            non_markup = [c for c in span if c != TokenClass.MARKUP]
            out.append(max(set(non_markup), key=non_markup.count))
    return out


@dataclass
class ClassStats:
    """Advantage distribution for one token class."""

    n: int = 0
    n_positive: int = 0
    n_negative: int = 0
    n_zero: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    examples_extreme: list[dict[str, Any]] = field(default_factory=list)

    def add(self, adv: float, token: str) -> None:
        self.n += 1
        self.total += adv
        if adv > 0:
            self.n_positive += 1
        elif adv < 0:
            self.n_negative += 1
        else:
            self.n_zero += 1
        if self.minimum is None or adv < self.minimum:
            self.minimum = adv
        if self.maximum is None or adv > self.maximum:
            self.maximum = adv
        self.examples_extreme.append({"advantage": adv, "token": token})
        self.examples_extreme.sort(key=lambda r: r["advantage"])
        if len(self.examples_extreme) > 12:
            # Keep the tails, drop the middle: the outliers are the finding.
            self.examples_extreme = (
                self.examples_extreme[:6] + self.examples_extreme[-6:]
            )

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_zero": self.n_zero,
            "mean": round(self.mean, 6),
            "min": None if self.minimum is None else round(self.minimum, 6),
            "max": None if self.maximum is None else round(self.maximum, 6),
            "extremes": [
                {"advantage": round(r["advantage"], 4), "token": r["token"]}
                for r in self.examples_extreme
            ],
        }


def class_report(
    per_token: Iterable[tuple[str, float, str]],
) -> dict[str, Any]:
    """Aggregate `(class, advantage, token_text)` triples into a report."""
    stats: dict[str, ClassStats] = {}
    for cls, adv, tok in per_token:
        stats.setdefault(cls, ClassStats()).add(float(adv), tok)

    total = sum(s.n for s in stats.values())
    return {
        "n_supervised_tokens": total,
        "by_class": {k: v.to_json() for k, v in sorted(stats.items())},
        "markup_share": (
            round(stats[TokenClass.MARKUP].n / total, 4)
            if total and TokenClass.MARKUP in stats
            else 0.0
        ),
    }
