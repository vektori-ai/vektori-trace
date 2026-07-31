"""Tests for vektori_trace.align — byte-offset alignment between student and teacher.

Test #12 (the equivalence oracle) is deliberately first: it is the only test
that catches off-by-one span reindexing, which otherwise produces finite,
plausible-looking losses indefinitely.
"""

from __future__ import annotations

import pytest

from vektori_trace.align import (
    AlignmentError,
    SpanKind,
    align_by_bytes,
    classify_spans,
    span_logprob_sums,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _span_reverse_kl_python(sums: list[tuple[float, float]]) -> float:
    """Policy-gradient reverse KL from (sum_s, sum_t) span tuples.

    For 1:1 spans this equals token-level reverse_kl_surrogate exactly.
    Denominator is the number of spans (= n_supervised_tokens for 1:1).
    """
    n = len(sums)
    if n == 0:
        return 0.0
    return sum((s - t) * s for s, t in sums) / n


def _plain_reverse_kl_python(student_lps: list[float], teacher_lps: list[float]) -> float:
    """Per-token reverse KL surrogate in pure Python."""
    n = len(student_lps)
    if n == 0:
        return 0.0
    return sum((s - t) * s for s, t in zip(student_lps, teacher_lps, strict=True)) / n


# ---------------------------------------------------------------------------
# Test #12 — Equivalence oracle (NON-NEGOTIABLE; written first)
#
# When student and teacher have identical byte-for-byte tokenisation, every
# span must be 1:1, classify_spans must label them all ESTIMATOR_A, and
# span_logprob_sums fed into the reverse-KL surrogate must equal plain
# per-token reverse KL.
# ---------------------------------------------------------------------------


def test_equivalence_oracle_identical_tokenization_spans_are_all_1_to_1():
    token_bytes = [b"hello", b" world", b"!", b"\n", b"foo"]
    alignment = align_by_bytes(token_bytes, token_bytes)

    assert alignment.n_student_tokens == 5
    assert alignment.n_teacher_tokens == 5
    assert len(alignment.spans) == 5
    assert alignment.granularity == pytest.approx(1.0)

    for span in alignment.spans:
        assert span.n_student == 1, f"expected 1:1, got n_student={span.n_student}"
        assert span.n_teacher == 1, f"expected 1:1, got n_teacher={span.n_teacher}"


def test_equivalence_oracle_classify_all_estimator_a():
    token_bytes = [b"hello", b" world", b"!", b"\n", b"foo"]
    alignment = align_by_bytes(token_bytes, token_bytes)
    classified = classify_spans(alignment)

    assert len(classified) == 5
    for kind, reason in classified:
        assert kind == SpanKind.ESTIMATOR_A, f"expected ESTIMATOR_A, got {kind!r}: {reason}"


def test_equivalence_oracle_span_logprob_sums_match_token_level():
    """Core oracle: span reverse KL == plain reverse KL when tokenisation is identical."""
    token_bytes = [b"hello", b" world", b"!", b"\n", b"foo"]
    student_logprobs = [-0.5, -1.2, -0.3, -2.1, -0.8]
    teacher_logprobs = [-0.8, -0.9, -0.4, -1.7, -1.0]

    alignment = align_by_bytes(token_bytes, token_bytes)
    sums = span_logprob_sums(alignment, student_logprobs, teacher_logprobs)

    # Each span sum must equal the single-token logprob.
    for i, (sum_s, sum_t) in enumerate(sums):
        assert sum_s == pytest.approx(student_logprobs[i], abs=1e-12)
        assert sum_t == pytest.approx(teacher_logprobs[i], abs=1e-12)

    # span-level reverse KL must equal plain reverse KL.
    span_kl = _span_reverse_kl_python(sums)
    plain_kl = _plain_reverse_kl_python(student_logprobs, teacher_logprobs)
    assert span_kl == pytest.approx(plain_kl, rel=1e-9, abs=1e-12)


def test_equivalence_oracle_byte_start_end_coverage():
    """byte_start and byte_end tile the full byte range without overlap."""
    token_bytes = [b"ab", b"cde", b"f"]
    alignment = align_by_bytes(token_bytes, token_bytes)

    spans = list(alignment.spans)
    assert spans[0].byte_start == 0
    for i in range(len(spans) - 1):
        assert spans[i].byte_end == spans[i + 1].byte_start
    assert spans[-1].byte_end == 6  # len("abcdef")


# ---------------------------------------------------------------------------
# Numeric example from FINAL-PLAN.md §4
#
# 'x = 1234 + 56789'
#   student: x | Ġ= | Ġ | 1 | 2 | 3 | 4 | Ġ+ | Ġ | 5 | 6 | 7 | 8 | 9
#   teacher: x | Ġ= | Ġ | 123 | 4 | Ġ+ | Ġ | 567 | 89
#
# "Byte offsets agree at every boundary, so 1234 closes as a 3↔1 span then a
# 1↔1 span. Nothing dropped, nothing desynced."
# ---------------------------------------------------------------------------

_STUDENT_NUMERIC = [
    b"x", b" =", b" ",
    b"1", b"2", b"3", b"4",
    b" +", b" ",
    b"5", b"6", b"7", b"8", b"9",
]  # 14 tokens, 16 bytes total

_TEACHER_NUMERIC = [
    b"x", b" =", b" ",
    b"123", b"4",
    b" +", b" ",
    b"567", b"89",
]  # 9 tokens, 16 bytes total


def test_numeric_example_no_exception():
    align_by_bytes(_STUDENT_NUMERIC, _TEACHER_NUMERIC)  # must not raise


def test_numeric_example_span_count():
    alignment = align_by_bytes(_STUDENT_NUMERIC, _TEACHER_NUMERIC)
    assert len(alignment.spans) == 9


def test_numeric_example_granularity():
    alignment = align_by_bytes(_STUDENT_NUMERIC, _TEACHER_NUMERIC)
    assert alignment.granularity == pytest.approx(9 / 14, rel=1e-9)


def test_numeric_example_digit_grouping_spans():
    """'123' must be a 3:1 span; '89' must be a 2:1 span."""
    alignment = align_by_bytes(_STUDENT_NUMERIC, _TEACHER_NUMERIC)
    spans = list(alignment.spans)

    # span index 3 covers bytes 4–7 ("123")
    assert spans[3].n_student == 3
    assert spans[3].n_teacher == 1
    assert spans[3].byte_start == 4
    assert spans[3].byte_end == 7

    # span index 8 covers bytes 14–16 ("89")
    assert spans[8].n_student == 2
    assert spans[8].n_teacher == 1
    assert spans[8].byte_start == 14
    assert spans[8].byte_end == 16


def test_numeric_example_classification():
    alignment = align_by_bytes(_STUDENT_NUMERIC, _TEACHER_NUMERIC)
    classified = classify_spans(alignment)

    # 1:1 spans: indices 0,1,2 (x, Ġ=, Ġ), 4 (4), 5 (Ġ+), 6 (Ġ) — six total
    estimator_a = [kind for kind, _ in classified if kind == SpanKind.ESTIMATOR_A]
    estimator_b = [kind for kind, _ in classified if kind == SpanKind.ESTIMATOR_B]
    assert len(estimator_a) == 6
    assert len(estimator_b) == 3  # "123", "567", "89"


def test_numeric_example_no_desync_and_no_leftover():
    alignment = align_by_bytes(_STUDENT_NUMERIC, _TEACHER_NUMERIC)
    assert alignment.n_student_tokens == 14
    assert alignment.n_teacher_tokens == 9
    # Every student and teacher token is accounted for in exactly one span.
    covered_s: set[int] = set()
    covered_t: set[int] = set()
    for span in alignment.spans:
        for i in span.student_idx:
            assert i not in covered_s, f"student token {i} in two spans"
            covered_s.add(i)
        for i in span.teacher_idx:
            assert i not in covered_t, f"teacher token {i} in two spans"
            covered_t.add(i)
    assert covered_s == set(range(14))
    assert covered_t == set(range(9))


# ---------------------------------------------------------------------------
# Hard fail 1 — byte length mismatch
# ---------------------------------------------------------------------------


def test_byte_length_mismatch_raises():
    with pytest.raises(AlignmentError, match=r"[Bb]yte length mismatch"):
        align_by_bytes([b"hello"], [b"world!"])  # 5 vs 6 bytes


def test_byte_length_mismatch_message_mentions_both_counts():
    with pytest.raises(AlignmentError) as exc:
        align_by_bytes([b"ab", b"c"], [b"abcd"])  # 3 vs 4 bytes
    msg = str(exc.value)
    assert "3" in msg and "4" in msg


def test_byte_length_mismatch_nfc_note_present():
    """The error message should mention the NFC hazard."""
    with pytest.raises(AlignmentError) as exc:
        align_by_bytes([b"a"], [b"ab"])
    assert "NFC" in str(exc.value) or "normaliz" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Hard fail 2 — leftover tokens after merge
# ---------------------------------------------------------------------------


def test_leftover_zero_byte_student_token_raises():
    """A zero-byte trailing token can survive the byte-length check but cause a leftover."""
    with pytest.raises(AlignmentError, match=r"[Ll]eftover"):
        align_by_bytes([b"hello", b""], [b"hello"])


def test_leftover_zero_byte_teacher_token_raises():
    with pytest.raises(AlignmentError, match=r"[Ll]eftover"):
        align_by_bytes([b"hello"], [b"hello", b""])


# ---------------------------------------------------------------------------
# Hard fail 3 — oversize span
# ---------------------------------------------------------------------------


def test_oversize_span_raises():
    """9 single-byte student tokens vs 1 nine-byte teacher token exceeds default 8."""
    student = [b"a"] * 9
    teacher = [b"aaaaaaaaa"]
    with pytest.raises(AlignmentError, match="max_span_student_tokens"):
        align_by_bytes(student, teacher)


def test_oversize_span_custom_threshold():
    """max_span_student_tokens=3 rejects any 4-student-token span."""
    student = [b"a"] * 4
    teacher = [b"aaaa"]
    with pytest.raises(AlignmentError, match="max_span_student_tokens"):
        align_by_bytes(student, teacher, max_span_student_tokens=3)


def test_exactly_at_max_span_does_not_raise():
    """A span of exactly max_span_student_tokens tokens must succeed."""
    student = [b"a"] * 8
    teacher = [b"aaaaaaaa"]
    alignment = align_by_bytes(student, teacher, max_span_student_tokens=8)
    assert len(alignment.spans) == 1
    assert alignment.spans[0].n_student == 8


def test_oversize_span_error_contains_span_text():
    student = [b"xyz"] * 9
    teacher = [b"xyz" * 9]  # 27 bytes, 1 token
    with pytest.raises(AlignmentError) as exc:
        align_by_bytes(student, teacher)
    assert b"xyz".decode() in str(exc.value) or "xyz" in str(exc.value)


# ---------------------------------------------------------------------------
# classify_spans — special-token masking
# ---------------------------------------------------------------------------


def test_classify_spans_all_estimator_a_without_specials():
    token_bytes = [b"a", b"b", b"c"]
    alignment = align_by_bytes(token_bytes, token_bytes)
    classified = classify_spans(alignment)
    assert all(k == SpanKind.ESTIMATOR_A for k, _ in classified)


def test_classify_spans_drop_if_student_span_is_special():
    student_bytes = [b"<s>", b"hello", b"</s>"]  # 3 student tokens
    teacher_bytes = [b"<s>", b"hello", b"</s>"]  # 3 teacher tokens
    alignment = align_by_bytes(student_bytes, teacher_bytes)

    # Mark first and last student tokens as special.
    special_s = [True, False, True]
    classified = classify_spans(alignment, student_special_mask=special_s)

    kinds = [k for k, _ in classified]
    assert kinds[0] == SpanKind.DROP
    assert kinds[1] == SpanKind.ESTIMATOR_A
    assert kinds[2] == SpanKind.DROP


def test_classify_spans_estimator_b_for_multi_token_span():
    student_bytes = [b"a", b"b", b"c"]  # 3 tokens, 3 bytes
    teacher_bytes = [b"abc"]  # 1 token, 3 bytes
    alignment = align_by_bytes(student_bytes, teacher_bytes)
    classified = classify_spans(alignment)
    assert len(classified) == 1
    assert classified[0][0] == SpanKind.ESTIMATOR_B


def test_classify_spans_reason_string_nonempty():
    token_bytes = [b"hello"]
    alignment = align_by_bytes(token_bytes, token_bytes)
    classified = classify_spans(alignment)
    _, reason = classified[0]
    assert isinstance(reason, str) and len(reason) > 0


# ---------------------------------------------------------------------------
# span_logprob_sums — basic correctness
# ---------------------------------------------------------------------------


def test_span_logprob_sums_1_to_1_passthrough():
    token_bytes = [b"a", b"b", b"c"]
    student_lps = [-0.1, -0.2, -0.3]
    teacher_lps = [-0.4, -0.5, -0.6]
    alignment = align_by_bytes(token_bytes, token_bytes)
    sums = span_logprob_sums(alignment, student_lps, teacher_lps)
    assert len(sums) == 3
    for i, (s, t) in enumerate(sums):
        assert s == pytest.approx(student_lps[i])
        assert t == pytest.approx(teacher_lps[i])


def test_span_logprob_sums_multi_token_span():
    """Sum across multi-student-token span must accumulate all student logprobs."""
    student_bytes = [b"a", b"b", b"c"]  # 3 tokens → 1 span vs teacher's 1 token
    teacher_bytes = [b"abc"]
    student_lps = [-0.1, -0.2, -0.3]
    teacher_lps = [-0.5]
    alignment = align_by_bytes(student_bytes, teacher_bytes)
    sums = span_logprob_sums(alignment, student_lps, teacher_lps)
    assert len(sums) == 1
    sum_s, sum_t = sums[0]
    assert sum_s == pytest.approx(-0.6, abs=1e-12)  # -0.1 + -0.2 + -0.3
    assert sum_t == pytest.approx(-0.5, abs=1e-12)


def test_span_logprob_sums_wrong_student_length_raises():
    token_bytes = [b"a", b"b"]
    alignment = align_by_bytes(token_bytes, token_bytes)
    with pytest.raises(ValueError, match="student"):
        span_logprob_sums(alignment, [-0.1], [-0.2, -0.3])


def test_span_logprob_sums_wrong_teacher_length_raises():
    token_bytes = [b"a", b"b"]
    alignment = align_by_bytes(token_bytes, token_bytes)
    with pytest.raises(ValueError, match="teacher"):
        span_logprob_sums(alignment, [-0.1, -0.2], [-0.3])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_inputs_returns_empty_alignment():
    alignment = align_by_bytes([], [])
    assert len(alignment.spans) == 0
    assert alignment.n_student_tokens == 0
    assert alignment.n_teacher_tokens == 0
    assert alignment.granularity == 0.0


def test_single_token_identical():
    alignment = align_by_bytes([b"hello"], [b"hello"])
    assert len(alignment.spans) == 1
    span = alignment.spans[0]
    assert span.student_idx == range(0, 1)
    assert span.teacher_idx == range(0, 1)
    assert span.byte_start == 0
    assert span.byte_end == 5
    assert alignment.granularity == pytest.approx(1.0)


def test_asymmetric_split_student_finer():
    """Student has 4 tokens, teacher has 2 tokens covering same bytes."""
    student = [b"a", b"b", b"c", b"d"]  # 4×1 byte
    teacher = [b"ab", b"cd"]            # 2×2 bytes
    alignment = align_by_bytes(student, teacher)
    assert len(alignment.spans) == 2
    assert alignment.spans[0].n_student == 2
    assert alignment.spans[0].n_teacher == 1
    assert alignment.spans[1].n_student == 2
    assert alignment.spans[1].n_teacher == 1


def test_asymmetric_split_teacher_finer():
    """Teacher has 4 tokens, student has 2 tokens covering same bytes."""
    student = [b"ab", b"cd"]            # 2×2 bytes
    teacher = [b"a", b"b", b"c", b"d"]  # 4×1 byte
    alignment = align_by_bytes(student, teacher)
    assert len(alignment.spans) == 2
    assert alignment.spans[0].n_student == 1
    assert alignment.spans[0].n_teacher == 2


def test_granularity_formula():
    """granularity = n_spans / n_student_tokens."""
    # 3 student tokens, 1 teacher token → 1 span
    alignment = align_by_bytes([b"a", b"b", b"c"], [b"abc"])
    assert alignment.granularity == pytest.approx(1 / 3, rel=1e-9)


def test_span_idx_ranges_are_contiguous_and_cover_all_tokens():
    """Every student and teacher token appears in exactly one span."""
    student = [b"x", b" =", b" ", b"1", b"2", b"3", b"4", b" +"]
    teacher = [b"x", b" = ", b"123", b"4", b" +"]
    alignment = align_by_bytes(student, teacher)
    covered_s: set[int] = set()
    covered_t: set[int] = set()
    for span in alignment.spans:
        for i in span.student_idx:
            assert i not in covered_s
            covered_s.add(i)
        for i in span.teacher_idx:
            assert i not in covered_t
            covered_t.add(i)
    assert covered_s == set(range(alignment.n_student_tokens))
    assert covered_t == set(range(alignment.n_teacher_tokens))


def test_span_byte_ranges_tile_total():
    """Span byte ranges form a partition of [0, total_bytes)."""
    student = [b"hello", b" ", b"world"]
    teacher = [b"hello ", b"world"]
    alignment = align_by_bytes(student, teacher)
    spans = list(alignment.spans)
    assert spans[0].byte_start == 0
    for i in range(len(spans) - 1):
        assert spans[i].byte_end == spans[i + 1].byte_start
    assert spans[-1].byte_end == 11  # len("hello world")


def test_multibyte_utf8_tokens():
    """Tokens containing multi-byte UTF-8 characters align by raw bytes."""
    # "café" is 5 bytes (é = 0xC3 0xA9)
    student = [b"caf\xc3\xa9"]   # 5 bytes
    teacher = [b"ca", b"f\xc3\xa9"]  # 2 + 3 bytes
    alignment = align_by_bytes(student, teacher)
    assert len(alignment.spans) == 1
    assert alignment.spans[0].n_student == 1
    assert alignment.spans[0].n_teacher == 2
    assert alignment.spans[0].byte_end == 5


def test_span_logprob_sums_multi_teacher_tokens():
    """Sum across multi-teacher-token span must accumulate all teacher logprobs."""
    student = [b"hello "]   # 1 token, 6 bytes
    teacher = [b"he", b"ll", b"o "]  # 3 tokens, 6 bytes
    student_lps = [-1.0]
    teacher_lps = [-0.3, -0.4, -0.2]
    alignment = align_by_bytes(student, teacher)
    sums = span_logprob_sums(alignment, student_lps, teacher_lps)
    assert len(sums) == 1
    sum_s, sum_t = sums[0]
    assert sum_s == pytest.approx(-1.0)
    assert sum_t == pytest.approx(-0.9, abs=1e-12)  # -0.3 + -0.4 + -0.2
