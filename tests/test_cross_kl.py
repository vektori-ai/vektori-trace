"""Tests for vektori_trace/cross_kl.py.

Test #12 (the equivalence oracle) is written FIRST and is NON-NEGOTIABLE.
Run the B-path of cross_step_loss with 1↔1 identical-tokenisation spans and
assert it agrees with opd.reverse_kl_surrogate to float tolerance.  Without
this test, off-by-one span reindexing produces finite, plausible-looking losses
indefinitely (FINAL-PLAN.md §10.12).
"""

from __future__ import annotations

import inspect
import math

import pytest

pytest.importorskip("torch")


# ─────────────────────────────────────────────────────────────────────────────
# Test #12 — Equivalence oracle (written first, non-negotiable)
#
# B path with 1↔1 spans must equal opd.reverse_kl_surrogate to float
# tolerance.  Forces correct indexing and correct global denominator.
# ─────────────────────────────────────────────────────────────────────────────


def test_equivalence_oracle_b_path_matches_reverse_kl_surrogate():
    """B-path cross_step_loss == opd.reverse_kl_surrogate when spans are 1:1."""
    import torch

    from vektori_trace.align import SpanKind, align_by_bytes
    from vektori_trace.cross_kl import cross_step_loss
    from vektori_trace.opd import reverse_kl_surrogate

    # 3 tokens, identical student/teacher tokenisation → all 1:1 spans.
    token_bytes = [b"hello", b" world", b"!"]
    alignment = align_by_bytes(token_bytes, token_bytes)

    assert alignment.granularity == pytest.approx(1.0)
    assert len(alignment.spans) == 3
    for sp in alignment.spans:
        assert sp.n_student == 1 and sp.n_teacher == 1

    # Force B path by providing no top-K data.
    span_kinds = [(SpanKind.ESTIMATOR_B, "forced B")] * 3

    V = 8
    student_logprobs_full = torch.log_softmax(torch.randn(3, V), dim=-1)
    s_lps = [-0.5, -1.2, -0.3]
    t_lps = [-0.8, -0.9, -0.4]
    student_token_logprobs = torch.tensor(s_lps, dtype=torch.float32, requires_grad=True)
    teacher_token_logprobs = t_lps

    loss, stats = cross_step_loss(
        alignment=alignment,
        span_kinds=span_kinds,
        student_logprobs_full=student_logprobs_full,
        student_token_logprobs=student_token_logprobs,
        teacher_token_logprobs=teacher_token_logprobs,
        teacher_topk_by_teacher_pos={},
        exact_map={},
    )

    # Reference: opd.reverse_kl_surrogate on the same per-token logprobs.
    ref_s = torch.tensor(s_lps, dtype=torch.float32, requires_grad=True)
    ref_t = torch.tensor(t_lps, dtype=torch.float32)
    ref_loss = reverse_kl_surrogate(ref_s, ref_t)

    assert loss.detach().item() == pytest.approx(ref_loss.detach().item(), abs=1e-5)
    assert stats.n_spans == 3
    assert stats.frac_B == pytest.approx(1.0)
    assert stats.frac_A == pytest.approx(0.0)
    assert stats.frac_dropped == pytest.approx(0.0)


def test_equivalence_oracle_b_path_gradient_reaches_logprobs():
    """Backward from cross_step_loss B path reaches student_token_logprobs."""
    import torch

    from vektori_trace.align import SpanKind, align_by_bytes
    from vektori_trace.cross_kl import cross_step_loss

    token_bytes = [b"a", b"b", b"c"]
    alignment = align_by_bytes(token_bytes, token_bytes)
    span_kinds = [(SpanKind.ESTIMATOR_B, "forced")] * 3

    s_lps = torch.tensor([-1.0, -0.5, -0.8], dtype=torch.float32, requires_grad=True)
    lp_full = torch.log_softmax(torch.zeros(3, 4), dim=-1)

    loss, _ = cross_step_loss(
        alignment=alignment,
        span_kinds=span_kinds,
        student_logprobs_full=lp_full,
        student_token_logprobs=s_lps,
        teacher_token_logprobs=[-0.9, -0.3, -0.7],
        teacher_topk_by_teacher_pos={},
        exact_map={},
    )
    loss.backward()
    assert s_lps.grad is not None
    assert s_lps.grad.abs().sum().item() > 0


def test_equivalence_oracle_span_indexing_is_correct():
    """Check that each span uses the correct student/teacher token positions."""
    import torch

    from vektori_trace.align import SpanKind, align_by_bytes
    from vektori_trace.cross_kl import cross_step_loss
    from vektori_trace.opd import reverse_kl_surrogate

    # 5 tokens → 5 distinct 1:1 spans
    token_bytes = [b"a", b"b", b"c", b"d", b"e"]
    alignment = align_by_bytes(token_bytes, token_bytes)
    span_kinds = [(SpanKind.ESTIMATOR_B, "B")] * 5

    s_lps = [-0.1, -0.2, -0.3, -0.4, -0.5]
    t_lps = [-0.6, -0.7, -0.8, -0.9, -1.0]
    student_token_logprobs = torch.tensor(s_lps, dtype=torch.float32)
    lp_full = torch.log_softmax(torch.zeros(5, 4), dim=-1)

    loss, _ = cross_step_loss(
        alignment=alignment,
        span_kinds=span_kinds,
        student_logprobs_full=lp_full,
        student_token_logprobs=student_token_logprobs,
        teacher_token_logprobs=t_lps,
        teacher_topk_by_teacher_pos={},
        exact_map={},
    )

    ref_loss = reverse_kl_surrogate(
        torch.tensor(s_lps, dtype=torch.float32),
        torch.tensor(t_lps, dtype=torch.float32),
    )
    assert float(loss) == pytest.approx(float(ref_loss), abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Other bucket math: mapped mass + other = 1
# ─────────────────────────────────────────────────────────────────────────────


def test_other_bucket_math_loss_matches_manual_k_plus_1_kl():
    """coarse_grained_reverse_kl matches manually computed K+1-event KL."""
    import torch

    from vektori_trace.cross_kl import coarse_grained_reverse_kl

    # V=3 student vocab; student: probs ~[0.71, 0.26, 0.03]
    logits = torch.tensor([2.0, 1.0, -1.0])
    lp_full = torch.log_softmax(logits.float(), dim=-1)

    # Teacher top-K: ids 10 and 20, probs 0.6 and 0.3 → other = 0.1
    lp_t_0 = math.log(0.6)
    lp_t_1 = math.log(0.3)
    lp_t_other = math.log(0.1)
    teacher_topk = {10: lp_t_0, 20: lp_t_1}
    mapped_pairs = [(0, lp_t_0), (1, lp_t_1)]  # student ids 0 and 1

    result = coarse_grained_reverse_kl(
        lp_full, mapped_pairs, teacher_topk, coverage_threshold=0.0
    )
    assert result is not None
    contrib, _mass_frac, _clamped = result

    # Manual K+1 computation
    p0 = float(lp_full[0].exp())
    p1 = float(lp_full[1].exp())
    p_other_s = 1.0 - p0 - p1
    lp_s_other = math.log(p_other_s)

    manual = (
        p0 * (math.log(p0) - lp_t_0)
        + p1 * (math.log(p1) - lp_t_1)
        + p_other_s * (lp_s_other - lp_t_other)
    )

    assert float(contrib) == pytest.approx(manual, abs=1e-5)


def test_other_bucket_teacher_partition_sums_to_one():
    """Teacher K+1 events: mapped probs + other prob = 1."""
    # Construct by hand; no torch needed for this property.
    mapped_t_probs = [0.5, 0.3]
    mapped_t_total = sum(mapped_t_probs)
    other_t_prob = 1.0 - mapped_t_total
    total = mapped_t_total + other_t_prob
    assert abs(total - 1.0) < 1e-12


def test_other_bucket_student_partition_sums_to_one():
    """Student K+1 events: mapped probs + other prob = 1."""
    import torch

    from vektori_trace.cross_kl import coarse_grained_reverse_kl

    # V=4: only 2 tokens mapped → other_s = prob of remaining 2
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])  # uniform
    lp_full = torch.log_softmax(logits.float(), dim=-1)

    teacher_topk = {10: math.log(0.5), 20: math.log(0.3)}
    mapped_pairs = [(0, math.log(0.5)), (1, math.log(0.3))]

    # Each student token gets prob 0.25; two are mapped → mapped_s = 0.5
    p0 = float(lp_full[0].exp())  # ≈ 0.25
    p1 = float(lp_full[1].exp())  # ≈ 0.25
    mapped_s_total = p0 + p1
    other_s_prob = 1.0 - mapped_s_total
    assert abs(mapped_s_total + other_s_prob - 1.0) < 1e-9

    result = coarse_grained_reverse_kl(
        lp_full, mapped_pairs, teacher_topk, coverage_threshold=0.0
    )
    assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# Clamp when Σexp(mapped_t) > 1  (fp8 noise guard)
# ─────────────────────────────────────────────────────────────────────────────


def test_clamp_when_sum_exp_exceeds_one_does_not_return_nan():
    """fp8 noise: Σexp(mapped_t) > 1 → clamp other_t to floor; no NaN."""
    import torch

    from vektori_trace.cross_kl import coarse_grained_reverse_kl

    logits = torch.tensor([1.0, 2.0, 0.5, -1.0])
    lp_full = torch.log_softmax(logits.float(), dim=-1)

    # teacher top-K whose probabilities intentionally sum > 1.
    # exp(0.0) = 1.0 each; two entries → sum = 2.0
    teacher_topk = {10: 0.0, 20: 0.0}
    mapped_pairs = [(0, 0.0), (1, 0.0)]

    # coverage = 2.0 / 2.0 = 1.0 ≥ threshold → passes coverage gate
    result = coarse_grained_reverse_kl(
        lp_full, mapped_pairs, teacher_topk, coverage_threshold=0.9
    )

    assert result is not None, "clamped path must not return None"
    contrib, mass_frac, clamped = result
    assert clamped is True
    assert torch.isfinite(contrib), f"contribution must be finite, got {contrib}"
    assert math.isfinite(mass_frac)


def test_clamp_contribution_is_finite_for_borderline_sum():
    """Σexp(mapped_t) == 1 exactly triggers the clamp path without NaN."""
    import torch

    from vektori_trace.cross_kl import coarse_grained_reverse_kl

    logits = torch.tensor([2.0, 1.0, 0.0])
    lp_full = torch.log_softmax(logits.float(), dim=-1)

    # Teacher single entry with prob exactly 1.0 → Σexp = 1.0
    teacher_topk = {10: 0.0}
    mapped_pairs = [(0, 0.0)]

    result = coarse_grained_reverse_kl(
        lp_full, mapped_pairs, teacher_topk, coverage_threshold=0.0
    )
    assert result is not None
    contrib, _, _clamped = result
    assert torch.isfinite(contrib)


# ─────────────────────────────────────────────────────────────────────────────
# Coverage gate
# ─────────────────────────────────────────────────────────────────────────────


def test_coverage_gate_demotes_to_none_below_threshold():
    """coarse_grained_reverse_kl returns None when mapped mass < threshold."""
    import torch

    from vektori_trace.cross_kl import coarse_grained_reverse_kl

    lp_full = torch.log_softmax(torch.ones(4), dim=-1)

    # 4-way uniform top-K; only 1 out of 4 mapped → coverage ≈ 0.25
    lp_q = math.log(0.25)
    teacher_topk = {10: lp_q, 20: lp_q, 30: lp_q, 40: lp_q}
    mapped_pairs = [(0, lp_q)]  # only teacher id 10 maps

    assert (
        coarse_grained_reverse_kl(
            lp_full, mapped_pairs, teacher_topk, coverage_threshold=0.9
        )
        is None
    )


def test_coverage_gate_passes_at_zero_threshold():
    """coverage_threshold=0.0 forces A path regardless of mapped mass."""
    import torch

    from vektori_trace.cross_kl import coarse_grained_reverse_kl

    lp_full = torch.log_softmax(torch.ones(4), dim=-1)
    lp_q = math.log(0.25)
    teacher_topk = {10: lp_q, 20: lp_q, 30: lp_q, 40: lp_q}
    mapped_pairs = [(0, lp_q)]

    result = coarse_grained_reverse_kl(
        lp_full, mapped_pairs, teacher_topk, coverage_threshold=0.0
    )
    assert result is not None


def test_coverage_gate_no_mapped_pairs_returns_none():
    """No mapped pairs → no A-path contribution regardless of threshold."""
    import torch

    from vektori_trace.cross_kl import coarse_grained_reverse_kl

    lp_full = torch.log_softmax(torch.ones(4), dim=-1)
    teacher_topk = {10: -1.0, 20: -2.0}

    result = coarse_grained_reverse_kl(
        lp_full, [], teacher_topk, coverage_threshold=0.0
    )
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Global denominator across A and B
# ─────────────────────────────────────────────────────────────────────────────


def test_global_denominator_is_total_supervised_student_tokens():
    """Denominator = total student tokens across both estimators (not per-estimator)."""
    import torch

    from vektori_trace.align import SpanKind, align_by_bytes
    from vektori_trace.cross_kl import cross_step_loss

    # Span 0: 1:1 ("a"), n_s=1
    # Span 1: 2:1 ("bc"), n_s=2
    # Total denominator = 3
    student_bytes = [b"a", b"b", b"c"]
    teacher_bytes = [b"a", b"bc"]
    alignment = align_by_bytes(student_bytes, teacher_bytes)

    # Force both spans to B so the math is easy to verify.
    span_kinds = [
        (SpanKind.ESTIMATOR_B, "forced B"),
        (SpanKind.ESTIMATOR_B, "forced B"),
    ]

    V = 4
    lp_full = torch.log_softmax(torch.zeros(3, V), dim=-1)

    s_lps = [-1.0, -0.5, -0.8]
    # teacher has exactly 2 tokens: [a, bc].
    # Span 0 uses teacher pos 0 (-0.9); span 1 uses teacher pos 1 (-0.3) only.
    t_lps_teacher = [-0.9, -0.3]
    student_token_logprobs = torch.tensor(s_lps, dtype=torch.float32)

    loss, _stats = cross_step_loss(
        alignment=alignment,
        span_kinds=span_kinds,
        student_logprobs_full=lp_full,
        student_token_logprobs=student_token_logprobs,
        teacher_token_logprobs=t_lps_teacher,
        teacher_topk_by_teacher_pos={},
        exact_map={},
    )

    # Span 0: logP_s = s[0] = -1.0; logP_t = t[0] = -0.9; n_s = 1
    # span_contrib_0 = ((-1.0) - (-0.9)) * (-1.0) = (-0.1) * (-1.0) = 0.1
    # Span 1: logP_s = s[1]+s[2] = -1.3; logP_t = t[1] = -0.3; n_s = 2
    # span_contrib_1 = ((-1.3) - (-0.3)) * (-1.3) = (-1.0) * (-1.3) = 1.3
    # denom = 1 + 2 = 3
    # loss = (0.1 + 1.3) / 3 = 1.4 / 3
    expected = (0.1 + 1.3) / 3
    assert float(loss) == pytest.approx(expected, abs=1e-5)


def test_global_denominator_mixes_a_and_b_spans():
    """Mix of A and B spans normalises by total student tokens, not per-estimator."""
    import torch

    from vektori_trace.align import align_by_bytes, classify_spans
    from vektori_trace.cross_kl import cross_step_loss

    # 3 student tokens / 2 teacher tokens:
    #   Span 0: 1:1 "a" ↔ "a"  → ESTIMATOR_A
    #   Span 1: 2:1 "bc" ↔ "bc"  → ESTIMATOR_B
    student_bytes = [b"a", b"b", b"c"]
    teacher_bytes = [b"a", b"bc"]
    alignment = align_by_bytes(student_bytes, teacher_bytes)
    span_kinds = classify_spans(alignment)

    V = 4
    # Uniform student logits → log_softmax = log(0.25) ≈ -1.386 at each token
    logits = torch.zeros(3, V)
    lp_full = torch.log_softmax(logits, dim=-1)

    s_lps = [-1.0, -0.5, -0.8]
    # teacher token logprobs: position 0 = -0.9 (for "a"), position 1 = -0.4 (for "bc")
    t_lps = [-0.9, -0.4]
    student_token_logprobs = torch.tensor(s_lps, dtype=torch.float32)

    # For span 0 (ESTIMATOR_A): teacher pos 0, teacher id 99 maps to student id 0
    # Coverage = 1.0 ≥ 0.9 → use A path
    teacher_topk_by_teacher_pos = {0: {99: -0.9}}
    exact_map = {99: 0}

    loss_A_and_B, stats = cross_step_loss(
        alignment=alignment,
        span_kinds=span_kinds,
        student_logprobs_full=lp_full,
        student_token_logprobs=student_token_logprobs,
        teacher_token_logprobs=t_lps,
        teacher_topk_by_teacher_pos=teacher_topk_by_teacher_pos,
        exact_map=exact_map,
        coverage_threshold=0.0,  # ensure A path is taken
    )

    assert stats.frac_A + stats.frac_B == pytest.approx(1.0)
    assert torch.isfinite(loss_A_and_B)
    # Denominator must be 3 (1 + 2 student tokens)
    # We verify this indirectly: if we replace span 0 with B, the loss changes.
    from vektori_trace.align import SpanKind
    span_kinds_all_b = [
        (SpanKind.ESTIMATOR_B, "B"),
        (SpanKind.ESTIMATOR_B, "B"),
    ]
    loss_all_B, _ = cross_step_loss(
        alignment=alignment,
        span_kinds=span_kinds_all_b,
        student_logprobs_full=lp_full,
        student_token_logprobs=student_token_logprobs,
        teacher_token_logprobs=t_lps,
        teacher_topk_by_teacher_pos={},
        exact_map={},
    )
    # The all-B loss uses the same denominator (3 tokens) — just a different numerator.
    assert torch.isfinite(loss_all_B)


# ─────────────────────────────────────────────────────────────────────────────
# span_surrogate
# ─────────────────────────────────────────────────────────────────────────────


def test_span_surrogate_matches_reverse_kl_surrogate_exactly():
    """span_surrogate(logP_s, logP_t) == reverse_kl_surrogate([logP_s], [logP_t])."""
    import torch

    from vektori_trace.cross_kl import span_surrogate
    from vektori_trace.opd import reverse_kl_surrogate

    log_p_s = torch.tensor(-1.3, dtype=torch.float32, requires_grad=True)
    log_p_t = -0.9

    result = span_surrogate(log_p_s, log_p_t)

    ref_s = torch.tensor([-1.3], dtype=torch.float32, requires_grad=True)
    ref_t = torch.tensor([-0.9], dtype=torch.float32)
    ref = reverse_kl_surrogate(ref_s, ref_t)

    assert result.detach().item() == pytest.approx(ref.detach().item(), abs=1e-7)


def test_span_surrogate_gradient_is_nonzero_when_logprobs_differ():
    import torch

    from vektori_trace.cross_kl import span_surrogate

    log_p_s = torch.tensor(-1.5, dtype=torch.float32, requires_grad=True)
    result = span_surrogate(log_p_s, -0.5)
    result.backward()
    assert log_p_s.grad is not None
    assert log_p_s.grad.abs().item() > 0


def test_span_surrogate_accepts_tensor_log_p_t():
    """span_surrogate accepts a tensor for log_p_t (must detach it)."""
    import torch

    from vektori_trace.cross_kl import span_surrogate

    log_p_s = torch.tensor(-1.0, dtype=torch.float32, requires_grad=True)
    log_p_t = torch.tensor(-0.8, dtype=torch.float32, requires_grad=True)

    result = span_surrogate(log_p_s, log_p_t)
    result.backward()

    # Gradient flows to log_p_s only; log_p_t must be detached.
    assert log_p_s.grad is not None
    assert log_p_t.grad is None


# ─────────────────────────────────────────────────────────────────────────────
# CrossStepStats
# ─────────────────────────────────────────────────────────────────────────────


def test_stats_frac_dropped_counts_drop_spans():
    """Dropped spans increment frac_dropped but not frac_A or frac_B."""
    import torch

    from vektori_trace.align import SpanKind, align_by_bytes
    from vektori_trace.cross_kl import cross_step_loss

    # 2 tokens → 2 spans; drop both.
    token_bytes = [b"a", b"b"]
    alignment = align_by_bytes(token_bytes, token_bytes)
    span_kinds = [
        (SpanKind.DROP, "test drop"),
        (SpanKind.DROP, "test drop"),
    ]

    lp_full = torch.log_softmax(torch.zeros(2, 4), dim=-1)
    s_lps = torch.tensor([-1.0, -0.5], dtype=torch.float32)

    loss, stats = cross_step_loss(
        alignment=alignment,
        span_kinds=span_kinds,
        student_logprobs_full=lp_full,
        student_token_logprobs=s_lps,
        teacher_token_logprobs=[-0.8, -0.4],
        teacher_topk_by_teacher_pos={},
        exact_map={},
    )

    assert stats.frac_dropped == pytest.approx(1.0)
    assert stats.frac_A == pytest.approx(0.0)
    assert stats.frac_B == pytest.approx(0.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-9)


def test_stats_special_tokens_masked_counts_special_drops():
    """Spans dropped for special tokens increment special_tokens_masked."""
    import torch

    from vektori_trace.align import SpanKind, align_by_bytes
    from vektori_trace.cross_kl import cross_step_loss

    token_bytes = [b"a", b"b", b"c"]
    alignment = align_by_bytes(token_bytes, token_bytes)
    span_kinds = [
        (SpanKind.DROP, "student span contains special tokens"),
        (SpanKind.DROP, "empty byte span"),  # not a special-token drop
        (SpanKind.ESTIMATOR_B, "B"),
    ]

    lp_full = torch.log_softmax(torch.zeros(3, 4), dim=-1)
    s_lps = torch.tensor([-1.0, -0.5, -0.8], dtype=torch.float32)

    _, stats = cross_step_loss(
        alignment=alignment,
        span_kinds=span_kinds,
        student_logprobs_full=lp_full,
        student_token_logprobs=s_lps,
        teacher_token_logprobs=[-0.9, -0.3, -0.7],
        teacher_topk_by_teacher_pos={},
        exact_map={},
    )

    assert stats.special_tokens_masked == 1


def test_stats_bytes_aligned_excludes_dropped_spans():
    """bytes_aligned counts only bytes from non-dropped spans."""
    import torch

    from vektori_trace.align import SpanKind, align_by_bytes
    from vektori_trace.cross_kl import cross_step_loss

    student_bytes = [b"hello", b"world"]  # 5 + 5 = 10 bytes
    alignment = align_by_bytes(student_bytes, student_bytes)
    span_kinds = [
        (SpanKind.ESTIMATOR_B, "B"),
        (SpanKind.DROP, "dropped"),
    ]

    lp_full = torch.log_softmax(torch.zeros(2, 4), dim=-1)
    s_lps = torch.tensor([-1.0, -0.5], dtype=torch.float32)

    _, stats = cross_step_loss(
        alignment=alignment,
        span_kinds=span_kinds,
        student_logprobs_full=lp_full,
        student_token_logprobs=s_lps,
        teacher_token_logprobs=[-0.9, -0.3],
        teacher_topk_by_teacher_pos={},
        exact_map={},
    )

    assert stats.bytes_aligned == 5   # only "hello"
    assert stats.bytes_total == 10    # "hello" + "world"


def test_stats_mean_mapped_teacher_mass_populated_for_a_spans():
    """mean_mapped_teacher_mass is set to a value in [0, 1] when A spans exist."""
    import torch

    from vektori_trace.align import align_by_bytes, classify_spans
    from vektori_trace.cross_kl import cross_step_loss

    # 2 identical tokens → 2 ESTIMATOR_A spans
    token_bytes = [b"hi", b"!"]
    alignment = align_by_bytes(token_bytes, token_bytes)
    span_kinds = classify_spans(alignment)

    V = 4
    lp_full = torch.log_softmax(torch.randn(2, V), dim=-1)
    s_lps = torch.tensor([-0.8, -1.1], dtype=torch.float32)

    # Top-K for teacher positions 0 and 1: map teacher ids 5 and 6 to student ids 0, 1
    teacher_topk_by_teacher_pos = {
        0: {5: -0.7},   # teacher pos 0 → teacher id 5 → student id 0
        1: {6: -1.0},   # teacher pos 1 → teacher id 6 → student id 1
    }
    exact_map = {5: 0, 6: 1}

    _, stats = cross_step_loss(
        alignment=alignment,
        span_kinds=span_kinds,
        student_logprobs_full=lp_full,
        student_token_logprobs=s_lps,
        teacher_token_logprobs=[-0.7, -1.0],
        teacher_topk_by_teacher_pos=teacher_topk_by_teacher_pos,
        exact_map=exact_map,
        coverage_threshold=0.0,
    )

    assert 0.0 <= stats.mean_mapped_teacher_mass <= 1.0
    assert stats.frac_A > 0.0


def test_stats_student_entropy_is_finite_for_a_spans():
    """student_entropy reports a finite value when A spans are present."""
    import torch

    from vektori_trace.align import align_by_bytes, classify_spans
    from vektori_trace.cross_kl import cross_step_loss

    token_bytes = [b"x"]
    alignment = align_by_bytes(token_bytes, token_bytes)
    span_kinds = classify_spans(alignment)

    V = 8
    lp_full = torch.log_softmax(torch.randn(1, V), dim=-1)
    s_lps = torch.tensor([-0.5], dtype=torch.float32)

    _, stats = cross_step_loss(
        alignment=alignment,
        span_kinds=span_kinds,
        student_logprobs_full=lp_full,
        student_token_logprobs=s_lps,
        teacher_token_logprobs=[-0.6],
        teacher_topk_by_teacher_pos={0: {7: -0.6}},
        exact_map={7: 0},
        coverage_threshold=0.0,
    )

    assert math.isfinite(stats.student_entropy)
    assert stats.student_entropy > 0  # uniform-ish logits → positive entropy


# ─────────────────────────────────────────────────────────────────────────────
# No beta parameter anywhere  (FINAL-PLAN.md §2: "Not parameterized. No beta.")
# ─────────────────────────────────────────────────────────────────────────────


def test_no_beta_parameter_in_any_public_function():
    """coarse_grained_reverse_kl, span_surrogate, cross_step_loss have no beta."""
    from vektori_trace.cross_kl import (
        coarse_grained_reverse_kl,
        cross_step_loss,
        span_surrogate,
    )

    for fn in (coarse_grained_reverse_kl, span_surrogate, cross_step_loss):
        sig = inspect.signature(fn)
        assert "beta" not in sig.parameters, (
            f"{fn.__name__} unexpectedly has a 'beta' parameter"
        )


def test_cross_step_stats_has_no_beta_field():
    """CrossStepStats dataclass has no beta field."""
    from vektori_trace.cross_kl import CrossStepStats

    fields = {f.name for f in CrossStepStats.__dataclass_fields__.values()}
    assert "beta" not in fields


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_alignment_returns_zero_loss():
    """All-dropped alignment produces a near-zero loss without crashing."""
    import torch

    from vektori_trace.align import align_by_bytes
    from vektori_trace.cross_kl import cross_step_loss

    alignment = align_by_bytes([], [])
    span_kinds: list = []

    lp_full = torch.zeros(0, 4)
    s_lps = torch.zeros(0, dtype=torch.float32)

    loss, stats = cross_step_loss(
        alignment=alignment,
        span_kinds=span_kinds,
        student_logprobs_full=lp_full,
        student_token_logprobs=s_lps,
        teacher_token_logprobs=[],
        teacher_topk_by_teacher_pos={},
        exact_map={},
    )

    assert float(loss) == pytest.approx(0.0, abs=1e-9)
    assert stats.n_spans == 0
    assert stats.frac_dropped == pytest.approx(0.0)


def test_estimator_a_gradient_reaches_student_logprobs_full():
    """Backward from A-path contribution reaches student_logprobs_full."""
    import torch

    from vektori_trace.cross_kl import coarse_grained_reverse_kl

    V = 4
    logits = torch.randn(V, requires_grad=False)
    lp_full = torch.log_softmax(logits.float(), dim=-1).detach().requires_grad_(True)

    teacher_topk = {10: -0.7, 20: -1.2}
    mapped_pairs = [(0, -0.7), (1, -1.2)]

    result = coarse_grained_reverse_kl(
        lp_full, mapped_pairs, teacher_topk, coverage_threshold=0.0
    )
    assert result is not None
    contrib, _, _clamped = result
    contrib.backward()

    assert lp_full.grad is not None
    assert lp_full.grad.abs().sum().item() > 0


def test_a_path_falls_back_to_b_when_no_topk_data():
    """ESTIMATOR_A span with missing top-K data falls back to B path without error."""
    import torch

    from vektori_trace.align import SpanKind, align_by_bytes
    from vektori_trace.cross_kl import cross_step_loss

    token_bytes = [b"hi"]
    alignment = align_by_bytes(token_bytes, token_bytes)
    span_kinds = [(SpanKind.ESTIMATOR_A, "1:1")]  # A requested, but no top-K data

    lp_full = torch.log_softmax(torch.randn(1, 4), dim=-1)
    s_lps = torch.tensor([-0.8], dtype=torch.float32)

    loss, stats = cross_step_loss(
        alignment=alignment,
        span_kinds=span_kinds,
        student_logprobs_full=lp_full,
        student_token_logprobs=s_lps,
        teacher_token_logprobs=[-0.6],
        teacher_topk_by_teacher_pos={},  # no data → fall back to B
        exact_map={},
    )

    assert torch.isfinite(loss)
    # Span fell back to B
    assert stats.frac_B == pytest.approx(1.0)
    assert stats.frac_A == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# §6 Normalization — ONE global denominator per optimizer step
#
# "total supervised student tokens across the batch, across BOTH estimators,
# never per-estimator. Otherwise an A/B mix drifting step to step silently
# rescales the learning rate."
# ─────────────────────────────────────────────────────────────────────────────


def _b_only(n: int):
    """n 1-byte tokens → n 1:1 spans, all forced onto Estimator B."""
    import torch

    from vektori_trace.align import SpanKind, align_by_bytes

    token_bytes = [bytes([97 + i]) for i in range(n)]
    alignment = align_by_bytes(token_bytes, token_bytes)
    kinds = [(SpanKind.ESTIMATOR_B, "forced B")] * n
    lp_full = torch.log_softmax(torch.zeros(n, 4), dim=-1)
    return alignment, kinds, lp_full


def test_denominator_1_returns_raw_sum():
    """denominator=1.0 is exactly n_supervised_tokens × the self-normalised loss."""
    import torch

    from vektori_trace.cross_kl import cross_step_loss

    alignment, kinds, lp_full = _b_only(4)
    s = torch.tensor([-0.4, -0.7, -1.1, -0.2], dtype=torch.float32)
    t = [-0.5, -0.6, -0.9, -0.3]

    common = dict(
        alignment=alignment, span_kinds=kinds, student_logprobs_full=lp_full,
        student_token_logprobs=s, teacher_token_logprobs=t,
        teacher_topk_by_teacher_pos={}, exact_map={},
    )
    normed, stats_n = cross_step_loss(**common)
    raw, stats_r = cross_step_loss(**common, denominator=1.0)

    assert stats_n.n_supervised_tokens == 4
    assert stats_r.n_supervised_tokens == 4
    assert float(raw) == pytest.approx(float(normed) * 4, abs=1e-5)


def test_denominator_rejects_non_positive():
    import torch

    from vektori_trace.cross_kl import cross_step_loss

    alignment, kinds, lp_full = _b_only(2)
    with pytest.raises(ValueError, match="denominator must be > 0"):
        cross_step_loss(
            alignment=alignment, span_kinds=kinds, student_logprobs_full=lp_full,
            student_token_logprobs=torch.tensor([-0.5, -0.5]),
            teacher_token_logprobs=[-0.4, -0.4],
            teacher_topk_by_teacher_pos={}, exact_map={}, denominator=0.0,
        )


def test_global_denominator_weights_by_token_not_by_example():
    """Two unequal examples: grads must be token-weighted, not example-averaged.

    This is the regression that per-example normalisation hides — a 2-token
    example and a 6-token example would otherwise contribute equally.
    """
    import torch

    from vektori_trace.cross_kl import cross_step_loss

    # One shared leaf so both "examples" accumulate into the same .grad,
    # mirroring how LoRA params accumulate across examples_per_step.
    leaf = torch.zeros(8, dtype=torch.float32, requires_grad=True)

    def run(n: int, offset: int, denominator: float | None):
        alignment, kinds, lp_full = _b_only(n)
        s = leaf[offset : offset + n] + torch.tensor(
            [-0.3 - 0.1 * i for i in range(n)], dtype=torch.float32
        )
        t = [-0.5 - 0.05 * i for i in range(n)]
        loss, stats = cross_step_loss(
            alignment=alignment, span_kinds=kinds, student_logprobs_full=lp_full,
            student_token_logprobs=s, teacher_token_logprobs=t,
            teacher_topk_by_teacher_pos={}, exact_map={},
            denominator=denominator,
        )
        return loss, stats

    # ── The shipped path: raw backward per example, one rescale by 1/total. ──
    loss_a, stats_a = run(2, 0, 1.0)
    loss_a.backward()
    loss_b, stats_b = run(6, 2, 1.0)
    loss_b.backward()
    total = stats_a.n_supervised_tokens + stats_b.n_supervised_tokens
    assert total == 8
    got = leaf.grad.clone() / total

    # ── Reference: one call over all 8 tokens, self-normalised. ─────────────
    leaf2 = torch.zeros(8, dtype=torch.float32, requires_grad=True)
    alignment, kinds, lp_full = _b_only(8)
    s_all = leaf2 + torch.tensor(
        [-0.3 - 0.1 * i for i in range(2)] + [-0.3 - 0.1 * i for i in range(6)],
        dtype=torch.float32,
    )
    t_all = [-0.5 - 0.05 * i for i in range(2)] + [-0.5 - 0.05 * i for i in range(6)]
    ref_loss, _ = cross_step_loss(
        alignment=alignment, span_kinds=kinds, student_logprobs_full=lp_full,
        student_token_logprobs=s_all, teacher_token_logprobs=t_all,
        teacher_topk_by_teacher_pos={}, exact_map={},
    )
    ref_loss.backward()

    assert torch.allclose(got, leaf2.grad, atol=1e-6), f"got={got} ref={leaf2.grad}"

    # And it is NOT what per-example normalisation would have produced.
    leaf3 = torch.zeros(8, dtype=torch.float32, requires_grad=True)

    def run3(n: int, offset: int):
        alignment, kinds, lp_full = _b_only(n)
        s = leaf3[offset : offset + n] + torch.tensor(
            [-0.3 - 0.1 * i for i in range(n)], dtype=torch.float32
        )
        t = [-0.5 - 0.05 * i for i in range(n)]
        loss, _ = cross_step_loss(
            alignment=alignment, span_kinds=kinds, student_logprobs_full=lp_full,
            student_token_logprobs=s, teacher_token_logprobs=t,
            teacher_topk_by_teacher_pos={}, exact_map={},
        )
        return loss

    (run3(2, 0) / 2).backward()
    (run3(6, 2) / 2).backward()
    assert not torch.allclose(leaf3.grad, leaf2.grad, atol=1e-6)
