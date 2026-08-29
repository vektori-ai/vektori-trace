"""Live projected advantages must equal `chunk_opd` exactly.

The live path and the replay path must be the *same function* of the same
alignment. Until 2026-08-29 they were not: live divided each chunk's `L_T`
across its student tokens and took a fresh ratio per token, which agrees with
the chunk rule only when the student logprobs inside a chunk are equal. Every
test here therefore uses unequal student logprobs somewhere.
"""

from __future__ import annotations

import math

import pytest

from vektori_trace.align import Alignment, Span
from vektori_trace.chunk_opd import assign_chunk_advantages
from vektori_trace.tau2.live_batch import (
    LiveBatchError,
    chunks_to_alignment,
    projected_turn_advantages,
)
from vektori_trace.tau2.live_score import ProjectedChunk


def _chunk(cid, idx, tlps, kind="reasoning"):
    return ProjectedChunk(
        chunk_id=cid, kind=kind, student_idx=tuple(idx),
        teacher_logprobs=tuple(tlps),
    )


def _reference(shape, behavior, teacher):
    """`assign_chunk_advantages` over a dense alignment of the same shape."""
    spans, s, t = [], 0, 0
    for n_s, n_t in shape:
        spans.append(Span(student_idx=range(s, s + n_s),
                          teacher_idx=range(t, t + n_t),
                          byte_start=s, byte_end=s + n_s))
        s += n_s
        t += n_t
    al = Alignment(spans=tuple(spans), n_student_tokens=s, n_teacher_tokens=t,
                   granularity=len(spans) / s if s else 0.0, dropped=0)
    return assign_chunk_advantages(al, behavior, teacher)


class TestTheRegression:
    """The exact case that distinguishes the two rules."""

    def test_n_to_1_unequal_logprobs_gives_zero_when_teacher_agrees(self):
        # student [-0.5, -1.0, -1.5] sums to -3.0; teacher chunk total -3.0.
        # Chunk rule: ratio = 1 -> every advantage is 0.
        ta = projected_turn_advantages(
            turn_index=0,
            action_token_ids=[1, 2, 3],
            behavior_logprobs=[-0.5, -1.0, -1.5],
            chunks=[_chunk("reasoning:0", (0, 1, 2), (-3.0,))],
        )
        assert ta.advantages == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
        assert ta.supervised_mask == [True, True, True]

    def test_the_broken_per_token_rule_would_have_produced_mixed_credit(self):
        """Pin the defect, so a regression cannot pass silently."""
        behavior = [-0.5, -1.0, -1.5]
        share = -3.0 / 3          # what the flat dict used to store
        broken = [(share / b - 1.0) * b for b in behavior]
        assert broken == pytest.approx([-0.5, 0.0, 0.5])
        ta = projected_turn_advantages(
            turn_index=0, action_token_ids=[1, 2, 3],
            behavior_logprobs=behavior,
            chunks=[_chunk("reasoning:0", (0, 1, 2), (-3.0,))],
        )
        assert ta.advantages != pytest.approx(broken)

    def test_equal_logprobs_agree_and_therefore_prove_nothing(self):
        """Why one-byte-per-token fixtures never caught this."""
        behavior = [-1.0, -1.0, -1.0]
        share = -3.0 / 3
        broken = [(share / b - 1.0) * b for b in behavior]
        ta = projected_turn_advantages(
            turn_index=0, action_token_ids=[1, 2, 3],
            behavior_logprobs=behavior,
            chunks=[_chunk("reasoning:0", (0, 1, 2), (-3.0,))],
        )
        assert ta.advantages == pytest.approx(broken, abs=1e-12)


class TestEquivalence:
    """Live output == chunk_opd output, for every alignment shape."""

    @pytest.mark.parametrize(
        "shape,behavior,teacher",
        [
            # 1:1
            ([(1, 1), (1, 1), (1, 1)], [-0.5, -1.0, -1.5], [-0.4, -1.2, -0.9]),
            # N:1 with unequal student logprobs, teacher disagreeing
            ([(3, 1)], [-0.5, -1.0, -1.5], [-2.0]),
            # 1:N
            ([(1, 3)], [-0.75], [-0.2, -0.3, -0.4]),
            # M:N
            ([(2, 3)], [-0.4, -1.6], [-0.5, -0.25, -0.75]),
            # mixed, the realistic shape
            ([(1, 1), (3, 2), (1, 1), (2, 4)],
             [-0.3, -0.5, -1.0, -1.5, -0.8, -0.2, -2.0],
             [-0.25, -0.6, -0.7, -0.9, -1.1, -0.15, -0.35, -0.45]),
        ],
        ids=["1:1", "N:1", "1:N", "M:N", "mixed"],
    )
    def test_matches_reference(self, shape, behavior, teacher):
        ref_adv, ref_sup, ref_stats = _reference(shape, behavior, teacher)

        chunks, s, t = [], 0, 0
        for k, (n_s, n_t) in enumerate(shape):
            chunks.append(_chunk(f"reasoning:{k}", range(s, s + n_s),
                                 teacher[t:t + n_t]))
            s += n_s
            t += n_t
        ta = projected_turn_advantages(
            turn_index=0, action_token_ids=list(range(len(behavior))),
            behavior_logprobs=behavior, chunks=chunks,
        )
        assert ta.advantages == pytest.approx(ref_adv, abs=1e-12)
        assert ta.supervised_mask == ref_sup
        assert ta.stats.n_supervised_tokens == ref_stats.n_supervised_tokens
        assert ta.stats.n_chunks == ref_stats.n_chunks

    def test_sparse_indices_map_back_correctly(self):
        """Eligible tokens are scattered; markup sits between them."""
        # action has 7 tokens; only 0,1,2 and 5,6 are eligible.
        behavior = [-0.5, -1.0, -1.5, -9.0, -9.0, -0.4, -1.6]
        chunks = [
            _chunk("reasoning:0", (0, 1, 2), (-3.0,)),
            _chunk("content:0", (5, 6), (-0.5, -0.25), kind="content"),
        ]
        ta = projected_turn_advantages(
            turn_index=0, action_token_ids=list(range(7)),
            behavior_logprobs=behavior, chunks=chunks,
        )
        # unsupervised positions untouched
        assert ta.advantages[3] == 0.0 and ta.advantages[4] == 0.0
        assert ta.supervised_mask == [True, True, True, False, False, True, True]
        # the N:1 chunk still lands on zero
        assert ta.advantages[:3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
        # and the M:N chunk matches the reference computed on its own
        ref, _, _ = _reference([(2, 2)], [-0.4, -1.6], [-0.5, -0.25])
        assert ta.advantages[5:] == pytest.approx(ref, abs=1e-12)

    def test_out_of_order_chunks_still_match(self):
        behavior = [-0.5, -1.0, -1.5, -0.4, -1.6]
        ordered = [_chunk("a", (0, 1, 2), (-3.0,)),
                   _chunk("b", (3, 4), (-0.5, -0.25))]
        ta1 = projected_turn_advantages(
            turn_index=0, action_token_ids=list(range(5)),
            behavior_logprobs=behavior, chunks=ordered)
        ta2 = projected_turn_advantages(
            turn_index=0, action_token_ids=list(range(5)),
            behavior_logprobs=behavior, chunks=list(reversed(ordered)))
        assert ta1.advantages == pytest.approx(ta2.advantages, abs=1e-12)


class TestFailsClosed:
    def test_empty_chunk_refused(self):
        with pytest.raises(LiveBatchError, match="no student token"):
            chunks_to_alignment([_chunk("a", (), (-1.0,))], 3)

    def test_chunk_with_no_teacher_logprob_refused(self):
        with pytest.raises(LiveBatchError, match="no teacher logprob"):
            chunks_to_alignment([_chunk("a", (0,), ())], 3)

    def test_overlapping_chunks_refused(self):
        with pytest.raises(LiveBatchError, match="reuses student token"):
            chunks_to_alignment(
                [_chunk("a", (0, 1), (-1.0,)), _chunk("b", (1, 2), (-1.0,))], 3)

    def test_out_of_range_index_refused(self):
        with pytest.raises(LiveBatchError, match="outside"):
            chunks_to_alignment([_chunk("a", (0, 9), (-1.0,))], 3)

    def test_unordered_student_idx_refused(self):
        with pytest.raises(LiveBatchError, match="not ascending"):
            chunks_to_alignment([_chunk("a", (2, 0), (-1.0,))], 3)

    def test_non_finite_teacher_logprob_refused(self):
        with pytest.raises(LiveBatchError):
            projected_turn_advantages(
                turn_index=0, action_token_ids=[1, 2],
                behavior_logprobs=[-0.5, -1.0],
                chunks=[_chunk("a", (0, 1), (-math.inf,))])

    def test_non_finite_behavior_logprob_refused(self):
        with pytest.raises(LiveBatchError, match="non-finite behavior"):
            projected_turn_advantages(
                turn_index=0, action_token_ids=[1, 2],
                behavior_logprobs=[-0.5, math.nan],
                chunks=[_chunk("a", (0, 1), (-1.0,))])

    def test_no_chunks_supervises_nothing(self):
        ta = projected_turn_advantages(
            turn_index=0, action_token_ids=[1, 2, 3],
            behavior_logprobs=[-0.5, -1.0, -1.5], chunks=[])
        assert ta.supervised_mask == [False, False, False]
        assert ta.advantages == [0.0, 0.0, 0.0]
        assert ta.stats.n_sentinel_tokens == 3


class TestRoundTrip:
    def test_chunk_survives_json(self):
        c = _chunk("reasoning:2", (4, 5, 6), (-1.0, -2.0))
        back = ProjectedChunk.from_json(c.to_json())
        assert back == c
        assert back.teacher_logprob_sum == pytest.approx(-3.0)
