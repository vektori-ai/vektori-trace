"""The cross-tokenizer datum/loss on the Fireworks student path (plan §6.5).

`test_student_fireworks.py` covers the same-tokenizer path, where the teacher
returns one logprob per student token. That contract is invalid against
DeepSeek-V4, so this file covers the replacement:

- `build_cross_opd_datum` carries per-*student*-token advantages and behaviour
  logprobs (post-alignment), not teacher logprobs;
- `cross_opd_loss_fn` hard-errors on a length mismatch instead of truncating to
  `min_len`, and normalises once by the global supervised-token count.

`tinker` is stubbed exactly as in the sibling file — only its three data
containers are touched.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vektori_trace.providers.student.fireworks import (  # noqa: E402
    build_cross_opd_datum,
    cross_opd_loss_fn,
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


# ---------------------------------------------------------------------------
# Datum construction
# ---------------------------------------------------------------------------


def test_advantages_and_behavior_are_zero_padded_across_the_prefix(tinker):
    datum = build_cross_opd_datum(
        tinker,
        full_tokens=[1, 2, 3, 4, 5],
        prompt_len=3,
        advantages=[0.5, -0.25],
        behavior_logprobs=[-1.0, -2.0],
    )

    assert datum.loss_fn_inputs["weights"].data == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert datum.loss_fn_inputs["advantages"].data == [0.0, 0.0, 0.0, 0.5, -0.25]
    assert datum.loss_fn_inputs["behavior_logprobs"].data == [
        0.0, 0.0, 0.0, -1.0, -2.0,
    ]


def test_teacher_token_count_never_appears_in_the_datum(tinker):
    """The whole point of the cross path: 3 teacher tokens, 2 student tokens.

    Alignment already consumed the teacher side, so nothing here has to know
    the teacher emitted a different number of tokens.
    """
    datum = build_cross_opd_datum(
        tinker,
        full_tokens=[1, 2, 3, 4, 5],
        prompt_len=3,
        advantages=[0.1, 0.2],  # 2 student tokens
        behavior_logprobs=[-0.5, -0.5],
    )
    assert "teacher_logprobs" not in datum.loss_fn_inputs
    assert len(datum.loss_fn_inputs["advantages"].data) == 5


def test_supervised_mask_zeroes_sentinel_positions_in_the_weights(tinker):
    datum = build_cross_opd_datum(
        tinker,
        full_tokens=[1, 2, 3, 4, 5],
        prompt_len=2,
        advantages=[0.5, 0.0, 0.25],
        behavior_logprobs=[-1.0, -1.0, -1.0],
        supervised_mask=[True, False, True],
    )
    assert datum.loss_fn_inputs["weights"].data == [0.0, 0.0, 1.0, 0.0, 1.0]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            dict(advantages=[0.5], behavior_logprobs=[-1.0, -2.0]),
            "advantages are per-student-token",
        ),
        (
            dict(advantages=[0.5, 0.5], behavior_logprobs=[-1.0]),
            "refusing to build a misaligned datum",
        ),
    ],
)
def test_length_mismatch_is_rejected(tinker, kwargs, match):
    with pytest.raises(ValueError, match=match):
        build_cross_opd_datum(
            tinker, full_tokens=[1, 2, 3, 4, 5], prompt_len=3, **kwargs
        )


def test_mask_length_mismatch_is_rejected(tinker):
    with pytest.raises(ValueError, match="mask entries"):
        build_cross_opd_datum(
            tinker,
            full_tokens=[1, 2, 3, 4, 5],
            prompt_len=3,
            advantages=[0.5, 0.5],
            behavior_logprobs=[-1.0, -1.0],
            supervised_mask=[True],
        )


def test_empty_action_is_rejected(tinker):
    with pytest.raises(ValueError, match="no sampled tokens"):
        build_cross_opd_datum(
            tinker,
            full_tokens=[1, 2, 3],
            prompt_len=3,
            advantages=[],
            behavior_logprobs=[],
        )


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def _datum(tinker, **kw):
    return build_cross_opd_datum(tinker, **kw)


def test_loss_is_differentiable_through_student_logprobs(tinker):
    datum = _datum(
        tinker,
        full_tokens=[1, 2, 3, 4, 5],
        prompt_len=3,
        advantages=[0.5, -0.5],
        behavior_logprobs=[-1.0, -2.0],
    )
    student_lp = torch.tensor(
        [-0.1, -0.2, -0.3, -1.0, -2.0], requires_grad=True
    )

    loss, metrics = cross_opd_loss_fn([datum], [student_lp])
    loss.backward()

    assert torch.isfinite(loss)
    assert student_lp.grad is not None
    assert torch.isfinite(student_lp.grad).all()
    assert bool((student_lp.grad[3:] != 0).any())
    # Prefix positions carry no loss weight, so no gradient.
    assert float(student_lp.grad[:3].abs().sum()) == 0.0
    assert metrics["action_tokens"] == 2.0


def test_length_mismatch_raises_instead_of_truncating_to_min_len(tinker):
    """Plan §6.5: remove the silent `min_len` truncation."""
    datum = _datum(
        tinker,
        full_tokens=[1, 2, 3, 4, 5],
        prompt_len=3,
        advantages=[0.5, -0.5],
        behavior_logprobs=[-1.0, -2.0],
    )
    short = torch.tensor([-0.1, -0.2, -0.3, -1.0], requires_grad=True)

    with pytest.raises(RuntimeError, match="Not truncating"):
        cross_opd_loss_fn([datum], [short])


def test_global_denominator_not_per_example_mean(tinker):
    """Two examples of different length share one supervised-token denominator.

    A per-example mean would weight the 1-token action as heavily as the
    3-token one; the plan requires a single global division.
    """
    short = _datum(
        tinker,
        full_tokens=[1, 2],
        prompt_len=1,
        advantages=[1.0],
        behavior_logprobs=[-1.0],
    )
    long = _datum(
        tinker,
        full_tokens=[1, 2, 3, 4],
        prompt_len=1,
        advantages=[1.0, 1.0, 1.0],
        behavior_logprobs=[-1.0, -1.0, -1.0],
    )
    # current == behavior everywhere, so every ratio is 1 and each supervised
    # token contributes exactly -A_i = -1.
    lp_short = torch.tensor([-0.5, -1.0], requires_grad=True)
    lp_long = torch.tensor([-0.5, -1.0, -1.0, -1.0], requires_grad=True)

    loss, metrics = cross_opd_loss_fn([short, long], [lp_short, lp_long])

    assert metrics["action_tokens"] == 4.0
    # -4 / 4 == -1, not the -1 a per-example mean would also give by luck;
    # the check that matters is the denominator being the token count.
    assert float(loss.detach()) == pytest.approx(-1.0, abs=1e-6)
    assert metrics["mean_advantage"] == pytest.approx(1.0, abs=1e-6)


def test_sentinel_positions_are_excluded_from_the_denominator(tinker):
    datum = _datum(
        tinker,
        full_tokens=[1, 2, 3, 4],
        prompt_len=1,
        advantages=[2.0, 0.0, 2.0],
        behavior_logprobs=[-1.0, -1.0, -1.0],
        supervised_mask=[True, False, True],
    )
    lp = torch.tensor([-0.5, -1.0, -1.0, -1.0], requires_grad=True)

    loss, metrics = cross_opd_loss_fn([datum], [lp])
    loss.backward()

    assert metrics["action_tokens"] == 2.0
    # -(2 + 2) / 2
    assert float(loss.detach()) == pytest.approx(-2.0, abs=1e-6)
    # The masked position gets no gradient.
    assert float(lp.grad[2]) == 0.0


def test_clip_fraction_is_zero_when_current_equals_behavior(tinker):
    datum = _datum(
        tinker,
        full_tokens=[1, 2, 3],
        prompt_len=1,
        advantages=[0.5, 0.5],
        behavior_logprobs=[-1.0, -2.0],
    )
    lp = torch.tensor([-0.5, -1.0, -2.0], requires_grad=True)

    _, metrics = cross_opd_loss_fn([datum], [lp])

    assert metrics["clip_fraction"] == pytest.approx(0.0)


def test_zero_advantage_everywhere_yields_zero_loss_and_zero_gradient(tinker):
    """A teacher in perfect agreement must not move the weights."""
    datum = _datum(
        tinker,
        full_tokens=[1, 2, 3],
        prompt_len=1,
        advantages=[0.0, 0.0],
        behavior_logprobs=[-1.0, -2.0],
    )
    lp = torch.tensor([-0.5, -1.0, -2.0], requires_grad=True)

    loss, _ = cross_opd_loss_fn([datum], [lp])
    loss.backward()

    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-9)
    assert float(lp.grad.abs().sum()) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Config gate (plan §7.1 cap, §6.5 loss selection)
# ---------------------------------------------------------------------------


def test_the_shipped_default_config_is_refused():
    """FireworksOPDConfig still defaults max_new_tokens=256 — the previous cap.

    This is the whole reason the gate exists: a driver that just constructs the
    config and runs would silently reproduce the old truncation.
    """
    from vektori_trace.chunk_opd import ChunkOPDError
    from vektori_trace.providers.student.fireworks import (
        FireworksOPDConfig,
        validate_cross_opd_config,
    )

    assert FireworksOPDConfig().max_new_tokens == 256
    with pytest.raises(ChunkOPDError, match="previous"):
        validate_cross_opd_config(FireworksOPDConfig())


def test_task_derived_cap_passes_the_gate():
    from vektori_trace.providers.student.fireworks import (
        FireworksOPDConfig,
        validate_cross_opd_config,
    )

    validate_cross_opd_config(FireworksOPDConfig(max_new_tokens=2048))


def test_gate_refuses_the_legacy_span_surrogate():
    from vektori_trace.chunk_opd import ChunkOPDError
    from vektori_trace.providers.student.fireworks import (
        FireworksOPDConfig,
        validate_cross_opd_config,
    )

    with pytest.raises(ChunkOPDError, match="legacy/diagnostic only"):
        validate_cross_opd_config(
            FireworksOPDConfig(max_new_tokens=2048), loss_id="cross_step_loss"
        )


def test_gate_refuses_top_k_on_the_chunk_objective():
    from vektori_trace.providers.student.fireworks import (
        FireworksOPDConfig,
        validate_cross_opd_config,
    )

    with pytest.raises(ValueError, match="no meaning for the chunk objective"):
        validate_cross_opd_config(FireworksOPDConfig(max_new_tokens=2048, top_k=5))
