"""The two OPD data sources are different objects with different losses.

Stage 3 (bisection) produces student-prefix / teacher-continuation SFT data;
Stage 6 (ReOPD replay) produces teacher-prefix / student-action examples scored
per token. These tests pin the boundary of each, because a mask that is one
message off is silently wrong rather than loud.
"""

from __future__ import annotations

import pytest

from vektori_trace.intervene import bisect_forking_step
from vektori_trace.reopd import (
    bisection_training_example,
    build_reopd_example,
    build_teacher_continuation_example,
    prefix_turns_through_step,
    reopd_loss_mask,
)
from vektori_trace.schema import ToolCall, Turn


def _student_turns() -> list[Turn]:
    """3 action steps, each an assistant tool call followed by its result."""
    out: list[Turn] = [Turn(index=0, role="user", content="fix the bug")]
    for step in range(3):
        i = 1 + 2 * step
        out.append(
            Turn(
                index=i,
                role="assistant",
                content=f"step {step}",
                tool_calls=[ToolCall(id=f"c{step}", name="bash", args={"cmd": "pytest"})],
            )
        )
        out.append(Turn(index=i + 1, role="tool", content="FAILED test_x.py", tool_call_id=f"c{step}"))
    return out


def _bisect(turns: list[Turn], last_recoverable: int):
    return bisect_forking_step(
        turns,
        continue_with_teacher=lambda _t, T: last_recoverable >= T,
        samples_per_probe=1,
    )


def test_prefix_ends_after_the_observation_of_step_T() -> None:
    """The teacher continues from the state *after* step T's tool result, and the
    continuation is the next assistant action. Including the next assistant turn
    would put a student action into the supervised span; excluding the tool
    result would hide the observation the teacher is reacting to."""
    turns = _student_turns()

    prefix = prefix_turns_through_step(turns, 1)
    assert [t.index for t in prefix] == [0, 1, 2, 3, 4]
    assert prefix[-1].role == "tool"

    assert prefix_turns_through_step(turns, -1) == []
    assert [t.index for t in prefix_turns_through_step(turns, 0)] == [0, 1, 2]
    with pytest.raises(IndexError):
        prefix_turns_through_step(turns, 3)


def test_training_example_is_cut_at_the_deepest_passing_prefix() -> None:
    turns = _student_turns()
    result = _bisect(turns, last_recoverable=1)
    assert result.largest_recoverable_T == 1
    assert result.forking_step == 2

    cont = [Turn(index=99, role="assistant", content="retry with prefix")]
    ex = bisection_training_example(turns, result, cont, task="t")
    assert ex is not None
    # Cut at the bisection's answer, not at the end of the trajectory and not at
    # the forking step itself: step 2 is the action that broke it.
    assert ex.prefix_steps == 1
    assert [t.index for t in ex.student_prefix_turns] == [0, 1, 2, 3, 4]
    assert ex.teacher_continuation_turns == cont
    assert ex.forking_step == 2


def test_no_training_data_when_the_bisection_proved_nothing() -> None:
    turns = _student_turns()
    cont = [Turn(index=99, role="assistant", content="retry with prefix")]

    # Nothing recoverable — every continuation failed, so there is no passing
    # teacher continuation to train on.
    none_ok = _bisect(turns, last_recoverable=-2)
    assert none_ok.largest_recoverable_T is None
    assert bisection_training_example(turns, none_ok, cont, task="t") is None

    # Dropped for desync: the prefix state was wrong, so the continuation was
    # generated from a workspace that never existed.
    result = _bisect(turns, last_recoverable=1)
    result.dropped = True
    assert bisection_training_example(turns, result, cont, task="t") is None

    # Unverified prefix replay: not a desync, but not execution-grounded either.
    result2 = _bisect(turns, last_recoverable=1)
    result2.resume_unverified = True
    assert bisection_training_example(turns, result2, cont, task="t") is None
    kept = bisection_training_example(
        turns, result2, cont, task="t", require_verified_prefix=False
    )
    assert kept is not None
    assert kept.verified_prefix is False

    # An empty continuation would tokenize to an example with no supervised
    # tokens at all.
    assert bisection_training_example(turns, _bisect(turns, 1), [], task="t") is None


def test_continuation_example_tokenizes_with_the_student_prefix_masked() -> None:
    """End to end: bisection output → tokenized SFT example. Loss lands on the
    teacher continuation only; the whole student prefix is IGNORE_INDEX."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from test_dataset import _tiny_tokenizer

    from vektori_trace.dataset import IGNORE_INDEX, _encode_messages, turns_to_messages

    turns = _student_turns()
    result = _bisect(turns, last_recoverable=1)
    cont = [Turn(index=99, role="assistant", content="retry with prefix")]
    ex = bisection_training_example(turns, result, cont, task="t")
    assert ex is not None

    tok = _tiny_tokenizer()
    tokenized = ex.tokenize(tok)
    assert tokenized is not None

    n_prefix = len(_encode_messages(tok, turns_to_messages(ex.student_prefix_turns)))
    assert 0 < n_prefix < len(tokenized.labels)
    assert all(lab == IGNORE_INDEX for lab in tokenized.labels[:n_prefix]), (
        "a student-prefix token carries loss"
    )
    assert all(lab != IGNORE_INDEX for lab in tokenized.labels[n_prefix:]), (
        "a teacher-continuation token was masked"
    )


def test_reopd_prefix_message_count_skips_subagent_turns() -> None:
    """`n_prefix_messages` counts chat messages, not turns — subagent turns are
    dropped before templating, so a turn count would mask one message too many."""
    turns = [
        Turn(index=0, role="user", content="q"),
        Turn(index=1, role="assistant", content="sub", subagent_depth=1),
        Turn(index=2, role="assistant", content="a1"),
        Turn(index=3, role="assistant", content="a2"),
    ]
    ex = build_reopd_example(turns, task="t", action_index=1)
    assert ex.teacher_action_turn.content == "a2"
    assert len(ex.prefix_turns) == 3
    assert ex.n_prefix_messages == 2


def test_reopd_mask_supervises_only_the_student_action() -> None:
    ids, mask = reopd_loss_mask([1, 2, 3], [4, 5])
    assert ids == [1, 2, 3, 4, 5]
    assert mask == [0, 0, 0, 1, 1]


def test_reopd_mask_zeroes_the_gradient_on_the_teacher_prefix() -> None:
    """The mask is the `loss_mask` of `opd.reverse_kl_surrogate`: the frozen
    teacher prefix must contribute no gradient, or ReOPD trains the student to
    imitate teacher tokens it never sampled."""
    torch = pytest.importorskip("torch")
    from vektori_trace.opd import reverse_kl_surrogate

    _ids, mask = reopd_loss_mask([1, 2, 3], [4, 5])
    student = torch.randn(1, 5, requires_grad=True)
    teacher = torch.randn(1, 5)
    reverse_kl_surrogate(student, teacher, torch.tensor([mask])).backward()
    assert torch.count_nonzero(student.grad[0, :3]) == 0
    assert torch.count_nonzero(student.grad[0, 3:]) > 0


def test_teacher_continuation_example_rejects_an_empty_continuation() -> None:
    with pytest.raises(ValueError, match="nothing would carry loss"):
        build_teacher_continuation_example(
            _student_turns(), task="t", T=0, continuation_turns=[]
        )
