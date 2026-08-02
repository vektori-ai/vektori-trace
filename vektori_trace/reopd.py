"""Step G — training-data construction for the two OPD data sources (PLAN.md).

PLAN.md describes two things that are easy to conflate, and this module keeps
them apart because they feed **different objectives**:

1. **ReOPD prefix replay** (Stage 6, primary training mode). The prefix comes
   from a *pre-collected teacher* trajectory; the **student** acts at one step;
   that step is supervised per-token against the teacher's `prompt_logprobs`.
   The supervised tokens are student-sampled, so there is a `∇log π_s` to weight
   and the objective is `opd.reverse_kl_surrogate`. Loss is 0 on the frozen
   prefix and 1 on the student's action tokens — see `reopd_loss_mask`.
   `docs/OPD.md` axis 1: "Teacher prefix → student acts at one step".

2. **Bisection-derived teacher continuations** (Stage 3, second output). The
   prefix comes from the *student's failed* trajectory; the **teacher**
   continues to termination and the verifier passes. Every supervised token is
   teacher-generated, so there is no student sample to weight — this is
   SFT-style next-token loss with the student prefix masked out
   (`dataset.tokenize_teacher_continuation`). `docs/OPD.md` axis 1 lists it as a
   separate row: "Student prefix → teacher continuation | intervention /
   DAgger".

Same masking machinery, opposite prefix ownership, different losses. Neither is
a description of the other.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .dataset import tokenize_teacher_continuation, turns_to_messages
from .evaluate.resume import assistant_tool_steps
from .schema import Turn
from .tokenizer_check import DEFAULT_TEACHER

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .evaluate.intervene import BisectionResult


class TeacherPool(Protocol):
    """Text + prompt_logprobs scoring for OPD (self-hosted vLLM/SGLang)."""

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str: ...

    def prompt_logprobs(self, prompt: str, tokens: list[int]) -> list[float]: ...


@dataclass
class ContainerPoolConfig:
    image: str
    warm_size: int = 4
    shared_per_repo: bool = True  # SWE-smith: one image per repo, not per task


@dataclass
class TeacherPoolConfig:
    model: str = DEFAULT_TEACHER
    endpoint: str | None = None  # vLLM/SGLang base URL
    require_prompt_logprobs: bool = True


# --------------------------------------------------------------------------
# 1. ReOPD prefix replay — teacher prefix, student acts, per-token OPD loss
# --------------------------------------------------------------------------


@dataclass
class ReOPDStepExample:
    """One ReOPD example: frozen *teacher* prefix + the step the student acts at.

    `teacher_action_turn` is the teacher's own action at that step. It is **not**
    the training target: at training time the student samples the action, the
    teacher scores those sampled tokens, and `opd.reverse_kl_surrogate` supplies
    the gradient. The teacher's action is kept for reference/dry-runs and as the
    prompt boundary — everything before it is the frozen prefix.
    """

    task: str
    step_index: int
    prefix_turns: list[Turn]
    teacher_action_turn: Turn
    later_teacher_turns: list[Turn] = field(default_factory=list)

    @property
    def n_prefix_messages(self) -> int:
        """Chat messages in the frozen prefix — the span that carries no loss."""
        return len(turns_to_messages(self.prefix_turns))


def build_reopd_example(
    teacher_turns: list[Turn],
    *,
    task: str,
    action_index: int,
) -> ReOPDStepExample:
    """Split a teacher trajectory into frozen prefix + the step the student acts at.

    `action_index` indexes parent assistant turns (depth 0).
    """
    parent_assistant_idxs = [
        i
        for i, t in enumerate(teacher_turns)
        if t.role == "assistant" and t.subagent_depth == 0
    ]
    if action_index < 0 or action_index >= len(parent_assistant_idxs):
        raise IndexError(
            f"action_index={action_index} out of range for "
            f"{len(parent_assistant_idxs)} parent assistant turns"
        )
    turn_i = parent_assistant_idxs[action_index]
    return ReOPDStepExample(
        task=task,
        step_index=action_index,
        prefix_turns=teacher_turns[:turn_i],
        teacher_action_turn=teacher_turns[turn_i],
        later_teacher_turns=teacher_turns[turn_i + 1 :],
    )


def iter_reopd_examples(
    trajectories: list[tuple[str, list[Turn]]],
    *,
    steps_per_traj: int | None = None,
) -> Iterator[ReOPDStepExample]:
    """Yield ReOPD examples from pre-collected teacher trajectories."""
    for task, turns in trajectories:
        parent_n = sum(
            1 for t in turns if t.role == "assistant" and t.subagent_depth == 0
        )
        limit = parent_n if steps_per_traj is None else min(parent_n, steps_per_traj)
        for i in range(limit):
            yield build_reopd_example(turns, task=task, action_index=i)


def reopd_loss_mask(
    prefix_ids: list[int],
    action_ids: list[int],
) -> tuple[list[int], list[int]]:
    """Concatenated ids and the loss mask for one ReOPD step.

    The mask is 0 across the frozen teacher prefix and 1 across the student's
    sampled action tokens — the `loss_mask` argument of
    `opd.reverse_kl_surrogate`. This is the whole of ReOPD's masking semantics;
    there is no per-token label tensor because the target is not a token id but
    the teacher's logprob of the token the student sampled.
    """
    return (
        list(prefix_ids) + list(action_ids),
        [0] * len(prefix_ids) + [1] * len(action_ids),
    )


# --------------------------------------------------------------------------
# 2. Bisection output — student prefix, teacher continuation, masked SFT
# --------------------------------------------------------------------------


@dataclass
class TeacherContinuationExample:
    """Stage-3 training artefact: student prefix + a *passing* teacher continuation.

    Supervised tokens are all teacher-generated, so this is next-token SFT with
    the student prefix masked — not the OPD surrogate.
    """

    task: str
    prefix_steps: int  # T: index into assistant_tool_steps; -1 = empty prefix
    student_prefix_turns: list[Turn]
    teacher_continuation_turns: list[Turn]
    forking_step: int | None = None
    verified_prefix: bool = True

    def tokenize(self, tokenizer: Any, *, max_length: int = 4096) -> Any:
        """Tokenize with the student prefix at IGNORE_INDEX (dataset.py)."""
        return tokenize_teacher_continuation(
            self.student_prefix_turns,
            self.teacher_continuation_turns,
            tokenizer,
            max_length=max_length,
        )


def prefix_turns_through_step(turns: list[Turn], T: int) -> list[Turn]:
    """Turns forming the student prefix replayed to action step `T`.

    `T` indexes `resume.assistant_tool_steps`, matching `resume.replay_prefix`
    and the bisection probe points. The prefix ends after the tool results of
    step `T`, because that observation is the state the teacher continues from —
    the continuation begins at the next parent assistant turn.
    """
    if T < -1:
        raise ValueError(f"T must be >= -1, got {T}")
    if T < 0:
        return []
    steps = assistant_tool_steps(turns)
    if len(steps) <= T:
        raise IndexError(f"T={T} out of range for {len(steps)} action steps")
    action_turn_index = steps[T][0]
    pos = next(
        i
        for i, t in enumerate(turns)
        if t.index == action_turn_index and t.role == "assistant"
    )
    end = pos + 1
    while end < len(turns) and not (
        turns[end].role == "assistant" and turns[end].subagent_depth == 0
    ):
        end += 1
    return turns[:end]


def build_teacher_continuation_example(
    turns: list[Turn],
    *,
    task: str,
    T: int,
    continuation_turns: list[Turn],
    forking_step: int | None = None,
    verified_prefix: bool = True,
) -> TeacherContinuationExample:
    """Assemble the Stage-3 artefact from a student trajectory and a continuation."""
    if not continuation_turns:
        raise ValueError("continuation_turns is empty — nothing would carry loss")
    return TeacherContinuationExample(
        task=task,
        prefix_steps=T,
        student_prefix_turns=prefix_turns_through_step(turns, T),
        teacher_continuation_turns=list(continuation_turns),
        forking_step=forking_step,
        verified_prefix=verified_prefix,
    )


def bisection_training_example(
    turns: list[Turn],
    result: BisectionResult,
    continuation_turns: list[Turn],
    *,
    task: str,
    require_verified_prefix: bool = True,
) -> TeacherContinuationExample | None:
    """Stage-3 training data from one bisection, or None if it yields none.

    Only the largest recoverable prefix is used: PLAN.md says "teacher
    continuations that **pass**", and that is the deepest probe whose
    continuation the verifier passed. Returns None when the task was dropped for
    desync, when no prefix was recoverable, or (by default) when the prefix
    replay was never verified — a continuation from an unchecked workspace is
    not the execution-grounded artefact Stage 3 claims.

    `continuation_turns` must be supplied by the caller: `intervene.ContinueFn`
    returns only a pass/fail bool, so the passing continuation's turns are not
    recoverable from `BisectionResult` alone. A driver that wants training data
    has to retain them.
    """
    if result.dropped or result.largest_recoverable_T is None:
        return None
    if require_verified_prefix and result.resume_unverified:
        return None
    if not continuation_turns:
        return None
    return build_teacher_continuation_example(
        turns,
        task=task,
        T=result.largest_recoverable_T,
        continuation_turns=continuation_turns,
        forking_step=result.forking_step,
        verified_prefix=not result.resume_unverified,
    )


class InMemoryTeacherPool:
    """Test double / dry-run teacher pool — no network."""

    def __init__(self, *, logprob: float = -0.1, text: str = "ok"):
        self.logprob = logprob
        self.text = text
        self.generate_calls = 0
        self.score_calls = 0

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        self.generate_calls += 1
        _ = prompt, max_tokens
        return self.text

    def prompt_logprobs(self, prompt: str, tokens: list[int]) -> list[float]:
        self.score_calls += 1
        _ = prompt
        return [self.logprob] * len(tokens)


__all__ = [
    "ContainerPoolConfig",
    "InMemoryTeacherPool",
    "ReOPDStepExample",
    "TeacherContinuationExample",
    "TeacherPool",
    "TeacherPoolConfig",
    "bisection_training_example",
    "build_reopd_example",
    "build_teacher_continuation_example",
    "iter_reopd_examples",
    "prefix_turns_through_step",
    "reopd_loss_mask",
]
