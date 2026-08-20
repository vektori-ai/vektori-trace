"""Tests for vektori_trace.chunk_opd — the published cross-tokenizer OPD loss.

Covers the Phase-0 §6.2 assertion list from docs/OPD-MULTITURN-PLAN.md.

The equivalence oracle is first and non-negotiable, for the same reason
test_align.py puts its own oracle first: every other failure mode here still
produces a finite, plausible-looking loss. A 1:1 chunk must reduce to ordinary
sampled reverse-KL exactly, or the credit assignment is wrong in a way no
downstream metric will reveal.
"""

from __future__ import annotations

import math

import pytest

from vektori_trace.align import align_by_bytes
from vektori_trace.chunk_opd import (
    CHUNK_LOSS_ID,
    DEFAULT_LARGE_CHUNK_THRESHOLD,
    LEGACY_LOSS_IDS,
    ChunkOPDError,
    assert_chunk_loss_selected,
    assign_chunk_advantages,
)

torch = pytest.importorskip("torch", reason="chunk loss needs the train extra")

from vektori_trace.chunk_opd import (  # noqa: E402
    clip_fraction,
    clipped_is_policy_loss,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bytes_for(text: str) -> list[bytes]:
    """One token per character — a trivial tokenisation for 1:1 oracles."""
    return [c.encode() for c in text]


# ---------------------------------------------------------------------------
# Equivalence oracle (NON-NEGOTIABLE; written first)
#
# Plan §5: "For a 1:1 chunk this reduces to traditional sampled reverse-KL:
#           A_i = log pi_D(t_i) - log pi_old(s_i)"
# ---------------------------------------------------------------------------


def test_equivalence_oracle_1to1_advantage_is_plain_logprob_difference():
    token_bytes = [b"hello", b" world", b"!", b"\n", b"foo"]
    alignment = align_by_bytes(token_bytes, token_bytes)

    behavior = [-0.5, -1.25, -0.1, -2.0, -0.75]
    teacher = [-0.25, -1.5, -0.4, -1.0, -0.9]

    advantages, supervised, stats = assign_chunk_advantages(
        alignment, behavior, teacher
    )

    assert all(supervised)
    assert stats.n_one_to_one == 5
    assert stats.n_chunks == 5
    assert stats.exact_1to1_token_fraction == 1.0

    for i, (b, t) in enumerate(zip(behavior, teacher, strict=True)):
        assert advantages[i] == pytest.approx(t - b, abs=1e-12), (
            f"token {i}: 1:1 chunk must reduce to log pi_D - log pi_old"
        )


def test_equivalence_oracle_holds_for_identical_logprobs_zero_advantage():
    """Teacher agreeing exactly with the student means no gradient signal."""
    token_bytes = [b"ab", b"cd", b"ef"]
    alignment = align_by_bytes(token_bytes, token_bytes)
    lps = [-0.3, -1.1, -0.7]

    advantages, supervised, _ = assign_chunk_advantages(alignment, lps, list(lps))

    assert all(supervised)
    assert all(a == pytest.approx(0.0, abs=1e-12) for a in advantages)


# ---------------------------------------------------------------------------
# Semantic-prior assignment: the paper's constraint
# ---------------------------------------------------------------------------


def test_multi_token_chunk_targets_sum_back_to_teacher_chunk_likelihood():
    """Plan §6.2: "assigned Qwen target log probabilities sum back to L_T"."""
    # One chunk: 2 student tokens <-> 3 teacher tokens, same 6 bytes.
    student = [b"abc", b"def"]
    teacher_bytes = [b"ab", b"cd", b"ef"]
    alignment = align_by_bytes(student, teacher_bytes)
    assert len(alignment.spans) == 1

    behavior = [-1.0, -3.0]
    teacher = [-0.5, -0.75, -0.25]  # L_T = -1.5

    advantages, supervised, stats = assign_chunk_advantages(
        alignment, behavior, teacher
    )
    assert all(supervised)
    assert stats.n_many_to_many == 1

    # log q_i = A_i + log p_i, and the constraint is sum_i log q_i == L_T.
    targets = [a + b for a, b in zip(advantages, behavior, strict=True)]
    assert sum(targets) == pytest.approx(sum(teacher), abs=1e-12)


def test_semantic_prior_preserves_student_internal_proportions():
    """log q_i proportional to log p_i — the structure-preserving property."""
    student = [b"abc", b"def"]
    teacher_bytes = [b"abcdef"]
    alignment = align_by_bytes(student, teacher_bytes)

    behavior = [-1.0, -3.0]  # ratio 1:3
    teacher = [-2.0]

    advantages, _, _ = assign_chunk_advantages(alignment, behavior, teacher)
    targets = [a + b for a, b in zip(advantages, behavior, strict=True)]

    assert sum(targets) == pytest.approx(-2.0, abs=1e-12)
    # The 1:3 proportion of the student's own logprobs must survive.
    assert targets[1] / targets[0] == pytest.approx(3.0, abs=1e-12)


def test_advantage_matches_closed_form_ratio_expression():
    """A_i = (L_T/L_S - 1) * log p_i, verbatim from core_algos.py:1128."""
    student = [b"xy", b"z"]
    teacher_bytes = [b"xyz"]
    alignment = align_by_bytes(student, teacher_bytes)

    behavior = [-0.8, -1.6]
    teacher = [-3.0]
    L_S, L_T = sum(behavior), sum(teacher)

    advantages, _, _ = assign_chunk_advantages(alignment, behavior, teacher)

    for i, b in enumerate(behavior):
        assert advantages[i] == pytest.approx((L_T / L_S - 1.0) * b, abs=1e-12)


def test_degenerate_near_zero_L_S_spreads_teacher_budget_uniformly():
    """Upstream's |L_S| < 1e-8 branch (core_algos.py:1120), not a divide by ~0."""
    student = [b"ab", b"cd"]
    teacher_bytes = [b"abcd"]
    alignment = align_by_bytes(student, teacher_bytes)

    behavior = [0.0, 0.0]  # L_S == 0 exactly
    teacher = [-4.0]

    advantages, supervised, stats = assign_chunk_advantages(
        alignment, behavior, teacher
    )

    assert stats.n_degenerate_chunks == 1
    assert all(supervised)
    assert all(math.isfinite(a) for a in advantages)
    # Uniform: L_T / n - log p_i, with log p_i == 0.
    assert advantages == [pytest.approx(-2.0), pytest.approx(-2.0)]


# ---------------------------------------------------------------------------
# Sentinels and fail-closed behaviour (plan §6.2, §11)
# ---------------------------------------------------------------------------


def test_oversize_chunk_becomes_unsupervised_sentinel_not_an_error():
    n = DEFAULT_LARGE_CHUNK_THRESHOLD + 1
    student = [b"a"] * n
    teacher_bytes = [b"a" * n]
    alignment = align_by_bytes(student, teacher_bytes, max_span_student_tokens=n)

    behavior = [-0.5] * n
    teacher = [-1.0]

    advantages, supervised, stats = assign_chunk_advantages(
        alignment, behavior, teacher
    )

    assert not any(supervised)
    assert all(a == 0.0 for a in advantages)
    assert stats.n_sentinel_tokens == n
    assert stats.n_supervised_tokens == 0
    assert stats.aligned_fraction == 0.0


def test_behavior_logprob_length_mismatch_is_a_hard_error():
    """Plan §6.5: no silent min_len truncation."""
    token_bytes = [b"a", b"b", b"c"]
    alignment = align_by_bytes(token_bytes, token_bytes)

    with pytest.raises(ChunkOPDError, match="refusing to truncate"):
        assign_chunk_advantages(alignment, [-0.5, -0.5], [-1.0, -1.0, -1.0])


def test_teacher_logprob_length_mismatch_is_a_hard_error():
    token_bytes = [b"a", b"b", b"c"]
    alignment = align_by_bytes(token_bytes, token_bytes)

    with pytest.raises(ChunkOPDError, match="refusing to truncate"):
        assign_chunk_advantages(alignment, [-0.5] * 3, [-1.0, -1.0])


@pytest.mark.parametrize("bad", [float("-inf"), float("inf"), float("nan")])
def test_non_finite_teacher_logprob_fails_closed(bad):
    token_bytes = [b"a", b"b"]
    alignment = align_by_bytes(token_bytes, token_bytes)

    with pytest.raises(ChunkOPDError, match="not finite"):
        assign_chunk_advantages(alignment, [-0.5, -0.5], [-1.0, bad])


@pytest.mark.parametrize("bad", [float("-inf"), float("nan")])
def test_non_finite_behavior_logprob_fails_closed(bad):
    token_bytes = [b"a", b"b"]
    alignment = align_by_bytes(token_bytes, token_bytes)

    with pytest.raises(ChunkOPDError, match="not finite"):
        assign_chunk_advantages(alignment, [-0.5, bad], [-1.0, -1.0])


def test_all_advantages_finite_on_a_mixed_chunk_action():
    """Plan §7.4: every chunk/advantage value must be finite."""
    student = [b"def ", b"foo", b"(", b"x", b"):"]
    teacher_bytes = [b"de", b"f ", b"fo", b"o(", b"x)", b":"]
    alignment = align_by_bytes(student, teacher_bytes)

    behavior = [-0.1, -0.4, -1.3, -0.9, -0.2]
    teacher = [-0.2, -0.3, -0.5, -0.6, -0.7, -0.8]

    advantages, _, stats = assign_chunk_advantages(alignment, behavior, teacher)

    assert all(math.isfinite(a) for a in advantages)
    assert stats.n_chunks == len(alignment.spans)
    assert stats.n_supervised_tokens + stats.n_sentinel_tokens == len(student)


def test_clamp_bounds_advantages_and_counts_clamped_tokens():
    token_bytes = [b"a", b"b"]
    alignment = align_by_bytes(token_bytes, token_bytes)

    advantages, _, stats = assign_chunk_advantages(
        alignment, [-0.5, -0.5], [-20.0, -0.6], clamp=5.0
    )

    assert stats.n_clamped_tokens == 1
    assert advantages[0] == pytest.approx(-5.0)
    assert advantages[1] == pytest.approx(-0.1)


def test_stats_classify_chunk_shapes():
    # 1:1, then 2:1, then 1:2
    student = [b"a", b"bc", b"d", b"ef"]
    teacher_bytes = [b"a", b"b", b"c", b"de", b"f"]
    alignment = align_by_bytes(student, teacher_bytes)

    _, _, stats = assign_chunk_advantages(
        alignment, [-0.5] * 4, [-0.5] * 5
    )

    assert stats.n_chunks == len(alignment.spans)
    assert stats.n_one_to_one >= 1
    assert sum(
        [
            stats.n_one_to_one,
            stats.n_one_to_many,
            stats.n_many_to_one,
            stats.n_many_to_many,
        ]
    ) == stats.n_chunks
    assert 0.0 <= stats.exact_1to1_token_fraction <= 1.0


# ---------------------------------------------------------------------------
# Clipped importance-sampling policy loss
# ---------------------------------------------------------------------------


def test_ratio_is_one_when_current_equals_behavior():
    """A one-step smoke samples and trains at the same weights: rho == 1."""
    behavior = torch.tensor([-0.5, -1.0, -0.25])
    current = behavior.clone().requires_grad_(True)
    adv = torch.tensor([0.5, -0.5, 1.0])

    loss = clipped_is_policy_loss(current, behavior, adv)

    # rho == 1 so loss == -mean(A_i)
    assert float(loss.detach()) == pytest.approx(-float(adv.mean()), abs=1e-6)
    assert clip_fraction(current, behavior) == pytest.approx(0.0)


def test_gradient_flows_only_through_current_logprobs():
    behavior = torch.tensor([-0.5, -1.0], requires_grad=True)
    adv = torch.tensor([0.5, -0.5], requires_grad=True)
    current = torch.tensor([-0.4, -1.1], requires_grad=True)

    loss = clipped_is_policy_loss(current, behavior, adv)
    loss.backward()

    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert bool((current.grad != 0).any())
    # Detached inside the loss: teacher-derived values never receive gradient.
    assert behavior.grad is None
    assert adv.grad is None


def test_loss_mask_excludes_unsupervised_positions_from_loss_and_denominator():
    behavior = torch.tensor([-0.5, -1.0, -0.25])
    current = behavior.clone().requires_grad_(True)
    adv = torch.tensor([1.0, 0.0, 3.0])
    mask = torch.tensor([1.0, 0.0, 1.0])

    loss = clipped_is_policy_loss(current, behavior, adv, mask)

    # Only positions 0 and 2 count: -(1.0 + 3.0)/2
    assert float(loss.detach()) == pytest.approx(-2.0, abs=1e-6)


def test_fully_masked_batch_returns_zero_that_still_carries_a_graph():
    behavior = torch.tensor([-0.5, -1.0])
    current = behavior.clone().requires_grad_(True)
    adv = torch.tensor([1.0, 2.0])
    mask = torch.zeros(2)

    loss = clipped_is_policy_loss(current, behavior, adv, mask)
    loss.backward()

    assert float(loss.detach()) == 0.0
    assert current.grad is not None
    assert float(current.grad.abs().sum()) == 0.0


def test_clipping_engages_on_a_large_ratio_and_is_reported():
    behavior = torch.tensor([-5.0])
    current = torch.tensor([-0.1], requires_grad=True)  # rho = exp(4.9), huge
    adv = torch.tensor([1.0])

    loss = clipped_is_policy_loss(current, behavior, adv, clip_eps=0.2)

    # Positive advantage + huge ratio: min() picks the clipped branch, 1.2.
    assert float(loss.detach()) == pytest.approx(-1.2, abs=1e-6)
    assert clip_fraction(current, behavior, clip_eps=0.2) == pytest.approx(1.0)


def test_negative_advantage_uses_the_unclipped_branch_when_ratio_is_large():
    behavior = torch.tensor([-5.0])
    current = torch.tensor([-0.1], requires_grad=True)
    adv = torch.tensor([-1.0])

    loss = clipped_is_policy_loss(current, behavior, adv, clip_eps=0.2)

    rho = math.exp(-0.1 - (-5.0))
    # min(rho * -1, 1.2 * -1) = rho * -1, so loss = +rho.
    assert float(loss.detach()) == pytest.approx(rho, abs=1e-4)


def test_explicit_denominator_returns_raw_sum_for_global_normalization():
    """Plan §7.3: one global denominator across the batch, not per-example means."""
    behavior = torch.tensor([-0.5, -1.0])
    current = behavior.clone().requires_grad_(True)
    adv = torch.tensor([2.0, 4.0])

    raw = clipped_is_policy_loss(current, behavior, adv, denominator=1.0)
    selfnorm = clipped_is_policy_loss(current, behavior, adv)

    assert float(raw.detach()) == pytest.approx(-6.0, abs=1e-6)
    assert float(selfnorm.detach()) == pytest.approx(float(raw.detach()) / 2.0, abs=1e-6)


def test_shape_mismatch_is_a_hard_error():
    behavior = torch.tensor([-0.5, -1.0])
    current = torch.tensor([-0.5])
    adv = torch.tensor([1.0, 1.0])

    with pytest.raises(ChunkOPDError, match="shape mismatch"):
        clipped_is_policy_loss(current, behavior, adv)


def test_bad_denominator_rejected():
    behavior = torch.tensor([-0.5])
    current = torch.tensor([-0.5], requires_grad=True)
    adv = torch.tensor([1.0])

    with pytest.raises(ChunkOPDError, match="denominator must be > 0"):
        clipped_is_policy_loss(current, behavior, adv, denominator=0.0)


# ---------------------------------------------------------------------------
# End-to-end: alignment -> advantages -> loss on one realistic action
# ---------------------------------------------------------------------------


def test_end_to_end_json_action_produces_finite_gradient():
    """A Terminus-shaped JSON action tokenised differently on each side."""
    action = '{"cmd": "ls -la /workspace"}'
    student = [
        b'{"cmd"', b': "', b"ls", b" -la", b" /work", b"space", b'"}',
    ]
    teacher_bytes = [
        b"{", b'"cmd', b'":', b' "ls', b" -", b"la /", b"workspace", b'"', b"}",
    ]
    assert b"".join(student) == action.encode()
    assert b"".join(teacher_bytes) == action.encode()

    alignment = align_by_bytes(student, teacher_bytes)

    behavior = [-0.2, -0.4, -0.9, -1.1, -0.3, -0.15, -0.05]
    teacher = [-0.1, -0.3, -0.2, -0.8, -1.0, -0.5, -0.25, -0.1, -0.05]

    advantages, supervised, stats = assign_chunk_advantages(
        alignment, behavior, teacher
    )

    assert all(math.isfinite(a) for a in advantages)
    assert stats.n_supervised_tokens > 0

    behavior_t = torch.tensor(behavior)
    current_t = (behavior_t + 0.01).detach().requires_grad_(True)
    adv_t = torch.tensor(advantages)
    mask_t = torch.tensor([1.0 if s else 0.0 for s in supervised])

    loss = clipped_is_policy_loss(current_t, behavior_t, adv_t, mask_t)
    loss.backward()

    assert torch.isfinite(loss)
    assert current_t.grad is not None
    assert torch.isfinite(current_t.grad).all()
    # Masked-out positions must receive exactly zero gradient.
    for i, s in enumerate(supervised):
        if not s:
            assert float(current_t.grad[i]) == 0.0


def test_every_action_byte_belongs_to_exactly_one_chunk():
    """Plan §6.2: chunks are ordered, non-overlapping, complete, and minimal."""
    student = [b"ab", b"cde", b"f"]
    teacher_bytes = [b"a", b"bcd", b"ef"]
    alignment = align_by_bytes(student, teacher_bytes)

    total = sum(len(b) for b in student)
    covered = bytearray(total)
    for span in alignment.spans:
        for pos in range(span.byte_start, span.byte_end):
            covered[pos] += 1

    assert all(c == 1 for c in covered), "every byte covered exactly once"
    # Ordered and contiguous.
    cursor = 0
    for span in alignment.spans:
        assert span.byte_start == cursor
        cursor = span.byte_end
    assert cursor == total


# ---------------------------------------------------------------------------
# Loss-selection fence: cross_kl stays diagnostic, never a training path
# ---------------------------------------------------------------------------


def test_chunk_loss_is_accepted_for_a_cross_tokenizer_run():
    assert_chunk_loss_selected(CHUNK_LOSS_ID)


@pytest.mark.parametrize("legacy", sorted(LEGACY_LOSS_IDS))
def test_legacy_estimator_b_is_refused_on_a_cross_tokenizer_run(legacy):
    """cross_kl produces a finite plausible number from the same inputs, so a
    misconfiguration is invisible in the logs. It must fail loudly instead."""
    with pytest.raises(ChunkOPDError, match="legacy/diagnostic only"):
        assert_chunk_loss_selected(legacy)


def test_unknown_loss_is_refused():
    with pytest.raises(ChunkOPDError, match="unknown loss"):
        assert_chunk_loss_selected("forward_kl")


def test_same_tokenizer_runs_are_unaffected_by_the_fence():
    """reverse_kl_surrogate stays correct when the tokenizers actually match."""
    for loss in [*LEGACY_LOSS_IDS, "anything"]:
        assert_chunk_loss_selected(loss, cross_tokenizer=False)
