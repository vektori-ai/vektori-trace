"""Per-step byte-offset alignment between student and teacher token streams.

Implements the two-pointer byte-offset merge described in FINAL-PLAN.md §7.
Pure Python — no network, no torch required.

Algorithm (verbatim from the plan):

    while s < n_s and t < n_t:
        s_end, t_end = s_off[s], t_off[t]    # cumulative byte ends
        if   s_end < t_end: s += 1
        elif s_end > t_end: t += 1
        else:
            s += 1; t += 1
            spans.append(Span(student=range(s_start, s), teacher=range(t_start, t), ...))
            s_start, t_start = s, t

Four deliberate departures from TRL (FINAL-PLAN.md §7):
1. Both sides' offsets come from local tokenizers, never API strings.
2. No trailing catch-all — oversize tail is a desync, not a merge.
3. Bayesian indexing: lp[k] scores token_ids[k] directly (echo=True).
4. No sequence packing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


class AlignmentError(ValueError):
    """Hard failure in the byte-alignment — never silently best-effort.

    Every mode produces a finite, plausible-looking loss if it passes
    silently, which is why the design is assert-heavy (FINAL-PLAN.md §10).
    """


class SpanKind(StrEnum):
    ESTIMATOR_A = "ESTIMATOR_A"  # 1 student token ↔ 1 teacher token
    ESTIMATOR_B = "ESTIMATOR_B"  # multi-token span, policy-gradient surrogate
    DROP = "DROP"                # empty, or contains only special tokens


@dataclass(frozen=True)
class Span:
    """A maximal run of student and teacher tokens covering identical bytes.

    student_idx and teacher_idx are half-open ranges [start, end) indexing
    into the respective token sequences passed to align_by_bytes.
    """

    student_idx: range  # [start, end) into the student token sequence
    teacher_idx: range  # [start, end) into the teacher token sequence
    byte_start: int     # first byte of this span (inclusive)
    byte_end: int       # first byte past this span (exclusive)

    @property
    def n_student(self) -> int:
        """Number of student tokens in this span."""
        return len(self.student_idx)

    @property
    def n_teacher(self) -> int:
        """Number of teacher tokens in this span."""
        return len(self.teacher_idx)

    @property
    def byte_len(self) -> int:
        return self.byte_end - self.byte_start


@dataclass
class Alignment:
    """Complete alignment of a student and teacher token stream."""

    spans: tuple[Span, ...]
    n_student_tokens: int
    n_teacher_tokens: int
    granularity: float  # len(spans) / n_student_tokens; 0.0 when n_student == 0
    dropped: int        # spans removed before returning (zero-byte etc.)


def align_by_bytes(
    student_token_bytes: list[bytes],
    teacher_token_bytes: list[bytes],
    *,
    max_span_student_tokens: int = 8,
) -> Alignment:
    """Exact two-pointer byte-offset alignment of two token streams.

    Both lists must have EOS tokens (and any zero-byte specials) already
    stripped before calling.  Raises AlignmentError on:

    1. Total byte-length disagreement between streams (before merging).
    2. Leftover tokens after the merge (no trailing catch-all).
    3. Any span with more than ``max_span_student_tokens`` student tokens.

    ``granularity`` = len(spans) / n_student_tokens, or 0.0 when empty.
    """
    n_s = len(student_token_bytes)
    n_t = len(teacher_token_bytes)

    # Hard fail 1 — total byte lengths must match before we attempt the merge.
    total_s = sum(len(b) for b in student_token_bytes)
    total_t = sum(len(b) for b in teacher_token_bytes)
    if total_s != total_t:
        raise AlignmentError(
            f"Byte length mismatch before alignment: student={total_s} bytes, "
            f"teacher={total_t} bytes.  "
            "Check for NFC normalisation differences (Qwen normalises to NFC, "
            "DeepSeek does not) — a verified hazard (FINAL-PLAN.md §4 finding 2)."
        )

    # Cumulative byte end-offsets: s_off[i] is the last byte (exclusive) of token i.
    s_off: list[int] = []
    acc = 0
    for b in student_token_bytes:
        acc += len(b)
        s_off.append(acc)

    t_off: list[int] = []
    acc = 0
    for b in teacher_token_bytes:
        acc += len(b)
        t_off.append(acc)

    spans: list[Span] = []
    s = 0
    t = 0
    s_start = 0
    t_start = 0
    byte_cursor = 0  # tracks byte_start for the next span

    while s < n_s and t < n_t:
        s_end = s_off[s]
        t_end = t_off[t]

        if s_end < t_end:
            s += 1
        elif s_end > t_end:
            t += 1
        else:
            # Byte boundaries agree — check size before incrementing so we can
            # report the span text in the error.
            n_span_s = (s + 1) - s_start
            if n_span_s > max_span_student_tokens:
                span_text = b"".join(student_token_bytes[s_start : s + 1])
                raise AlignmentError(
                    f"Span exceeds max_span_student_tokens={max_span_student_tokens}: "
                    f"{n_span_s} student tokens covering {span_text!r}.  "
                    "This is a desync — no trailing catch-all is applied "
                    "(FINAL-PLAN.md §7)."
                )
            # Advance both pointers then record range(s_start, s) — the
            # algorithm from the plan increments first so the range is correct.
            s += 1
            t += 1
            spans.append(
                Span(
                    student_idx=range(s_start, s),
                    teacher_idx=range(t_start, t),
                    byte_start=byte_cursor,
                    byte_end=s_end,
                )
            )
            byte_cursor = s_end
            s_start = s
            t_start = t

    # Hard fail 2 — leftover tokens indicate zero-byte tokens remaining after
    # all byte content is consumed (e.g. a zero-byte special not stripped).
    if s < n_s or t < n_t:
        raise AlignmentError(
            f"Leftover tokens after alignment: {n_s - s} student, {n_t - t} teacher.  "
            "No trailing catch-all is applied — this is a desync.  "
            "Ensure EOS and zero-byte special tokens are stripped before calling "
            "align_by_bytes (FINAL-PLAN.md §7)."
        )

    granularity = len(spans) / n_s if n_s > 0 else 0.0
    return Alignment(
        spans=tuple(spans),
        n_student_tokens=n_s,
        n_teacher_tokens=n_t,
        granularity=granularity,
        dropped=0,
    )


def classify_spans(
    alignment: Alignment,
    *,
    student_special_mask: Sequence[bool] | None = None,
    teacher_special_mask: Sequence[bool] | None = None,
) -> list[tuple[SpanKind, str]]:
    """Classify each span as ESTIMATOR_A, ESTIMATOR_B, or DROP.

    ESTIMATOR_A  — exactly 1 student token ↔ 1 teacher token.  The analytic
                   coarse-grained reverse KL (§6 estimator A) is available.
    ESTIMATOR_B  — multi-token span.  Use the policy-gradient surrogate (§6
                   estimator B): ``Σ log π_s`` and ``Σ log π_t`` over the span.
    DROP         — empty byte span or span consisting entirely of special tokens
                   on either side.

    ``student_special_mask[i]`` is True when student token *i* is a special
    token (BOS, EOS, padding, etc.).  If not supplied, no DROP-for-specials
    classification is performed.
    """
    result: list[tuple[SpanKind, str]] = []

    for span in alignment.spans:
        if span.byte_len == 0:
            result.append((SpanKind.DROP, "empty byte span"))
            continue

        if student_special_mask is not None and all(
            student_special_mask[i] for i in span.student_idx
        ):
            result.append((SpanKind.DROP, "student span contains only special tokens"))
            continue

        if teacher_special_mask is not None and all(
            teacher_special_mask[i] for i in span.teacher_idx
        ):
            result.append((SpanKind.DROP, "teacher span contains only special tokens"))
            continue

        if span.n_student == 1 and span.n_teacher == 1:
            result.append((SpanKind.ESTIMATOR_A, "1:1 span"))
        else:
            result.append(
                (
                    SpanKind.ESTIMATOR_B,
                    f"{span.n_student}:{span.n_teacher} multi-token span",
                )
            )

    return result


def span_logprob_sums(
    alignment: Alignment,
    student_logprobs: Sequence[float],
    teacher_logprobs: Sequence[float],
) -> list[tuple[float, float]]:
    """Sum log-probabilities within each span.

    Returns a list of ``(sum_student, sum_teacher)`` for each span:

    * ``sum_student`` = ``Σ log π_s(token)`` over student tokens in the span.
    * ``sum_teacher`` = ``Σ log π_t(token)`` over teacher tokens in the span.

    For 1:1 spans these are just the single-token log-probabilities.  For
    multi-token spans they are the span-level joint log-probabilities
    (log-probability of the span's byte sequence under each tokenisation).

    Feeding these into ``opd.reverse_kl_surrogate`` (treating each span as one
    "token") gives estimator B.  When the tokenisation is identical and all
    spans are 1:1, the result equals plain per-token reverse KL exactly
    (FINAL-PLAN.md §7 equivalence oracle, test #12).
    """
    if len(student_logprobs) != alignment.n_student_tokens:
        raise ValueError(
            f"student_logprobs has {len(student_logprobs)} entries but "
            f"alignment.n_student_tokens={alignment.n_student_tokens}"
        )
    if len(teacher_logprobs) != alignment.n_teacher_tokens:
        raise ValueError(
            f"teacher_logprobs has {len(teacher_logprobs)} entries but "
            f"alignment.n_teacher_tokens={alignment.n_teacher_tokens}"
        )

    result: list[tuple[float, float]] = []
    for span in alignment.spans:
        sum_s = sum(student_logprobs[i] for i in span.student_idx)
        sum_t = sum(teacher_logprobs[i] for i in span.teacher_idx)
        result.append((sum_s, sum_t))
    return result


__all__ = [
    "Alignment",
    "AlignmentError",
    "Span",
    "SpanKind",
    "align_by_bytes",
    "classify_spans",
    "span_logprob_sums",
]
