"""The Fireworks student loop's alignment and objective, without Fireworks.

Two things in `providers/student/fireworks.py` can be wrong in a way no run would reveal:

1. `build_opd_datum` lines up three arrays — tokens, loss weights, teacher
   logprobs — by index. A prefix/action off-by-one produces a finite loss that
   trains the student against the teacher's scores for the *wrong* positions.
2. `opd_loss_fn` has to be differentiable through the student logprobs, because
   `forward_backward_custom` computes `d_loss/d_logprob` by calling `.backward()`
   on whatever it returns. A loss that accidentally detaches yields zero
   gradients and a training run that silently does nothing.

Both are checkable locally: the datum is plain tensors, and the loss function's
contract is "given datums and logprob tensors, return a differentiable scalar".
`tinker` is stubbed — only its three data containers are used, and the real
package would add nothing to what is under test.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vektori_trace.providers.student.fireworks import (  # noqa: E402
    FireworksOPDConfig,
    build_opd_datum,
    opd_loss_fn,
    run_fireworks_opd,
)


class FakeTensorData:
    def __init__(self, data, dtype, shape):
        self.data = data
        self.dtype = dtype
        self.shape = shape


class FakeModelInput:
    def __init__(self, tokens):
        self.tokens = tokens

    @classmethod
    def from_ints(cls, tokens):
        return cls(tokens)


class FakeDatum:
    def __init__(self, model_input, loss_fn_inputs):
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs


class FakeTinker:
    Datum = FakeDatum
    ModelInput = FakeModelInput
    TensorData = FakeTensorData


@pytest.fixture
def tinker():
    return FakeTinker()


def test_datum_weights_are_zero_on_the_prefix_and_one_on_the_action(tinker):
    """The API's documented convention: 0 = prompt/no-loss, 1 = response/learned.

    Inverting it trains the student to imitate the frozen teacher prefix and
    ignore its own sampled action — the exact opposite of on-policy distillation.
    """
    datum = build_opd_datum(tinker, [1, 2, 3, 4, 5], prompt_len=3, teacher_logprobs=[-0.5, -1.5])

    assert datum.loss_fn_inputs["weights"].data == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_teacher_logprobs_land_on_the_action_positions(tinker):
    """The teacher scored only the sampled tokens, so its scores are zero-padded
    across the prefix. Those entries are masked out and must never shift."""
    datum = build_opd_datum(tinker, [1, 2, 3, 4, 5], prompt_len=3, teacher_logprobs=[-0.5, -1.5])

    assert datum.loss_fn_inputs["teacher_logprobs"].data == [0.0, 0.0, 0.0, -0.5, -1.5]


def test_datum_carries_the_full_sequence(tinker):
    datum = build_opd_datum(tinker, [1, 2, 3, 4, 5], prompt_len=3, teacher_logprobs=[-0.5, -1.5])

    assert datum.model_input.tokens == [1, 2, 3, 4, 5]


def test_wrong_number_of_teacher_logprobs_refuses_to_build(tinker):
    """Caught here rather than in the loss, where it would be silently truncated
    to min_len and train on a misaligned objective."""
    with pytest.raises(ValueError, match="refusing to build a misaligned datum"):
        build_opd_datum(tinker, [1, 2, 3, 4, 5], prompt_len=3, teacher_logprobs=[-0.5])


def test_no_sampled_tokens_is_an_error(tinker):
    with pytest.raises(ValueError, match="no sampled tokens"):
        build_opd_datum(tinker, [1, 2, 3], prompt_len=3, teacher_logprobs=[])


def test_loss_is_differentiable_through_the_student_logprobs(tinker):
    """`forward_backward_custom` calls .backward() on this scalar and ships the
    resulting d_loss/d_logprob to the trainer. No grad here, no training there."""
    datum = build_opd_datum(tinker, [1, 2, 3, 4, 5], prompt_len=3, teacher_logprobs=[-0.5, -1.5])
    student_lp = torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.9], requires_grad=True)

    loss, _ = opd_loss_fn([datum], [student_lp])
    loss.backward()

    assert student_lp.grad is not None
    assert torch.any(student_lp.grad != 0)


def test_only_the_action_positions_carry_gradient(tinker):
    """The prefix is frozen teacher context. A gradient there means the student is
    being trained to reproduce the prefix, which is off-policy SFT, not OPD."""
    datum = build_opd_datum(tinker, [1, 2, 3, 4, 5], prompt_len=3, teacher_logprobs=[-0.5, -1.5])
    student_lp = torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.9], requires_grad=True)

    loss, _ = opd_loss_fn([datum], [student_lp])
    loss.backward()

    assert torch.all(student_lp.grad[:3] == 0)
    assert torch.all(student_lp.grad[3:] != 0)


def test_gradient_sign_follows_the_opd_coefficient(tinker):
    """`(log π_s − log π_t) ∇log π_s`: where the student is *more* confident than
    the teacher the coefficient is positive, and minimising pushes that logprob
    down. Getting the sign backwards trains away from the teacher."""
    datum = build_opd_datum(tinker, [1, 2], prompt_len=1, teacher_logprobs=[-2.0])
    # Student at -0.5 is more confident than the teacher at -2.0.
    student_lp = torch.tensor([0.0, -0.5], requires_grad=True)

    loss, _ = opd_loss_fn([datum], [student_lp])
    loss.backward()

    # Positive gradient: a descent step reduces this logprob toward the teacher.
    assert student_lp.grad[1] > 0


def test_metrics_report_the_monitoring_ratio(tinker):
    """`mean_log_ratio` is the scalar that should trend to 0 as the student
    approaches the teacher; it carries no gradient and only exists to be read."""
    datum = build_opd_datum(tinker, [1, 2, 3], prompt_len=1, teacher_logprobs=[-1.0, -1.0])
    student_lp = torch.tensor([0.0, -1.0, -1.0], requires_grad=True)

    _, metrics = opd_loss_fn([datum], [student_lp])

    assert metrics["mean_log_ratio"] == pytest.approx(0.0)
    assert metrics["action_tokens"] == 2.0


def test_shorter_logprobs_than_the_datum_are_truncated_not_misread(tinker):
    """The documented pattern: the forward pass can return fewer logprobs than the
    datum has tokens, and zipping mismatched lengths shifts the objective."""
    datum = build_opd_datum(tinker, [1, 2, 3, 4, 5], prompt_len=3, teacher_logprobs=[-0.5, -1.5])
    student_lp = torch.tensor([-0.1, -0.2, -0.3, -0.4], requires_grad=True)

    loss, metrics = opd_loss_fn([datum], [student_lp])

    assert torch.isfinite(loss)
    assert metrics["action_tokens"] == 1.0


def test_topk_is_refused_rather_than_silently_downgraded():
    """`topk_reverse_kl` needs student logits over the teacher's top-K set, which
    forward_backward_custom does not return. Running the sampled-token objective
    instead would optimise something the config did not ask for."""
    cfg = FireworksOPDConfig(top_k=16)

    with pytest.raises(ValueError, match="cannot run on the Training API"):
        run_fireworks_opd([object()], object(), cfg)  # type: ignore[arg-type]


def test_no_examples_is_an_error():
    with pytest.raises(ValueError, match="nothing to distil from"):
        run_fireworks_opd([], object(), FireworksOPDConfig())  # type: ignore[arg-type]
