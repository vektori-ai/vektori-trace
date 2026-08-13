"""The `other` bucket is only resolvable to the teacher's quantisation noise.

`other = 1 − Σ(top-K)` subtracts FP8-served probabilities carrying ~2e-3
relative error each, so the result means nothing below ~1e-3. Measured on the
first real GPU run against deepseek-v4-flash-0731: max observed Σ − 1 was
9.07e-4, and 28.6% of Estimator-A positions produced an `other` mass below that
floor — used, until this change, as though resolved.
"""

from __future__ import annotations

import math

import pytest

from vektori_trace.cross_kl import (
    OTHER_MASS_FAULT,
    OTHER_MASS_FLOOR,
    coarse_grained_reverse_kl,
)

torch = pytest.importorskip("torch")


def _student(vocab: int = 8):
    """Full log_softmax over a tiny vocab, with grad."""
    logits = torch.zeros(vocab, requires_grad=True)
    return torch.log_softmax(logits, dim=-1)


def _call(mapped_probs: list[float]):
    """Run estimator A with teacher masses given as probabilities."""
    pairs = [(i, math.log(p)) for i, p in enumerate(mapped_probs)]
    topk = {100 + i: math.log(p) for i, p in enumerate(mapped_probs)}
    return coarse_grained_reverse_kl(_student(), pairs, topk)


def test_floor_is_the_measured_noise_level():
    """Not a round number chosen for comfort — the provider's resolution."""
    assert OTHER_MASS_FLOOR == 1e-3
    assert OTHER_MASS_FAULT > OTHER_MASS_FLOOR


def test_other_below_the_floor_is_not_treated_as_resolved():
    """A position whose top-K sums to 0.9999 has an `other` we cannot measure.

    Before this change it produced other_t = log(1e-4) = −9.2 and was used as
    though real. It must now be floored.
    """
    out = _call([0.9999])
    assert out is not None
    contribution, _coverage, clamped = out
    assert not clamped, "0.9999 is inside the floor; that is not a fault"
    assert math.isfinite(float(contribution))


def test_no_cliff_across_sum_equals_one():
    """The old conditional floor swung the teacher term ~7 nats on a 2e-6 input
    wobble: Σ=0.999999 → −13.8, Σ=1.000001 → log(1e-9) = −20.7. The loss is
    π_s(other)·(log π_s(other) − other_t), so that fabricated a large repulsion
    from everything outside the teacher's top-K."""
    just_under = _call([0.999999])
    just_over = _call([1.000001])
    assert just_under is not None and just_over is not None
    a = float(just_under[0])
    b = float(just_over[0])
    assert math.isclose(a, b, rel_tol=1e-6), (
        f"discontinuity across Σ=1: {a} vs {b} — the floor must be unconditional"
    )


def test_fp8_scale_overflow_is_not_reported_as_a_fault():
    """Σ − 1 = 9.07e-4 was the worst real observation. It is rounding, not a
    protocol error, and must not abort a run."""
    out = _call([1.000907])
    assert out is not None
    _, _, clamped = out
    assert not clamped, "the measured worst-case FP8 overflow must not trip §10.11"


def test_overflow_beyond_the_noise_floor_is_still_reported():
    """The tripwire must survive. Mass exceeding 1 by more than the floor is
    not explainable by rounding — that is alignment or request-shape trouble,
    and raising the threshold would be the wrong response."""
    out = _call([1.05])
    assert out is not None
    _, _, clamped = out
    assert clamped


def test_teacher_other_never_produces_nan_or_inf():
    """Whatever the provider returns, the teacher term stays finite."""
    for probs in ([0.5], [0.999], [1.0], [1.000001], [1.2], [0.25, 0.25, 0.25]):
        out = _call(probs)
        if out is None:
            continue
        assert math.isfinite(float(out[0])), f"non-finite contribution for {probs}"
