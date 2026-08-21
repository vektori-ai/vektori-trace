"""Multi-turn rollout records and the seam from a rollout to one OPD update.

`docs/OPD-MULTITURN-PLAN.md` §6.4 asks for a *mocked two-turn* proof that the
pieces connect: rollout captures ids and behaviour log probabilities, the later
prefix contains the earlier Harbor observation, template/environment tokens
carry zero loss, the student recomputes current log probabilities with autograd,
gradient reaches trainable parameters, and exactly one optimizer step changes
the adapter.

Until this module existed, `align.align_by_bytes`, `chunk_opd`, and the Fireworks
datum builders were each tested against hand-written inputs with **nothing
joining them**. This is that join, and it is deliberately small: no Harbor, no
network, no GPU. What it owns is the bookkeeping that is easy to get silently
wrong.

Three invariants it exists to enforce
-------------------------------------
1. **Behaviour log probabilities belong to a named policy version.** `A_i` is
   built from `log pi_old`; using the training-time value instead collapses the
   importance ratio to 1 and silently turns the update into something else. A
   turn therefore carries `policy_version`, and a batch mixing versions is
   refused (§11: "behavior log probabilities are absent or from the wrong ck75
   policy version").
2. **Only sampled bytes are supervised.** A turn stores the action's token ids
   separately from its prefix, so no renderer header, role token, empty-think
   wrapper, or environment observation can reach the loss by construction rather
   than by a mask someone remembered to apply (§4, §7.4).
3. **The observation actually got fed back.** `assert_observation_carried`
   checks that turn *n*'s Harbor output appears in turn *n+1*'s prefix. Without
   it a "multi-turn" rollout can be n independent single-turn rollouts, which
   would still produce a finite loss and a plausible report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .align import align_by_bytes
from .chunk_opd import ChunkStats, assign_chunk_advantages


class RolloutError(ValueError):
    """A rollout record cannot support a correct OPD update."""


@dataclass
class TurnRecord:
    """One assistant turn: what was sampled, under which policy, and what came back.

    `action_token_ids` / `behavior_logprobs` cover **only** the sampled action.
    The prefix is kept as text (`student_prefix_text`) rather than ids because
    nothing downstream supervises it — it exists so the observation-carried check
    has something to search, and so a report can show what conditioned the turn.
    """

    turn_index: int
    #: Rendered student prefix for this turn. Not supervised.
    student_prefix_text: str
    #: Exact bytes ck75 sampled. These are what the teacher scores.
    action_bytes: bytes
    #: Qwen ids for `action_bytes`, in order.
    action_token_ids: list[int]
    #: Per-token byte strings for those ids — what `align_by_bytes` consumes.
    action_token_bytes: list[bytes]
    #: log pi_old(s_i) captured at sampling time, one per action token.
    behavior_logprobs: list[float]
    #: Harbor's bounded output after executing `action_bytes`.
    observation: str
    #: The frozen policy that produced this turn.
    policy_version: str
    termination_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.action_token_ids)
        if n == 0:
            raise RolloutError(
                f"turn {self.turn_index}: no action tokens — an empty action has "
                "no gradient and must not become a zero-loss datum"
            )
        if len(self.behavior_logprobs) != n:
            raise RolloutError(
                f"turn {self.turn_index}: {len(self.behavior_logprobs)} behaviour "
                f"logprobs for {n} action tokens"
            )
        if len(self.action_token_bytes) != n:
            raise RolloutError(
                f"turn {self.turn_index}: {len(self.action_token_bytes)} token byte "
                f"strings for {n} action tokens"
            )
        joined = b"".join(self.action_token_bytes)
        if joined != self.action_bytes:
            raise RolloutError(
                f"turn {self.turn_index}: token bytes reconstruct {joined!r} but the "
                f"recorded action is {self.action_bytes!r} — the ids are not the "
                "bytes that were executed (§11: executed action differs from record)"
            )

    @property
    def n_action_tokens(self) -> int:
        return len(self.action_token_ids)


@dataclass
class TrajectoryRecord:
    """One Harbor episode: ordered turns under a single frozen policy version."""

    task: str
    turns: list[TurnRecord]
    policy_version: str
    sandbox_id: str | None = None
    outcome: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.turns:
            raise RolloutError(f"{self.task}: trajectory has no turns")
        for t in self.turns:
            if t.policy_version != self.policy_version:
                raise RolloutError(
                    f"{self.task} turn {t.turn_index}: policy_version "
                    f"{t.policy_version!r} != trajectory's {self.policy_version!r}. "
                    "§7.1 freezes one version for the whole rollout batch."
                )
        idx = [t.turn_index for t in self.turns]
        if idx != sorted(idx) or len(set(idx)) != len(idx):
            raise RolloutError(f"{self.task}: turn indices not strictly ordered: {idx}")

    @property
    def n_supervised_tokens(self) -> int:
        return sum(t.n_action_tokens for t in self.turns)


def assert_observation_carried(traj: TrajectoryRecord) -> None:
    """Each turn's observation must appear in the next turn's prefix (§7.4).

    Without this a "multi-turn" rollout can be several independent single-turn
    rollouts wearing a trenchcoat: the loss stays finite, the report looks
    normal, and the state distribution is not the student's own.

    Only non-empty observations are checked — a turn that produced no output has
    nothing to carry forward.
    """
    for prev, nxt in zip(traj.turns, traj.turns[1:], strict=False):
        obs = prev.observation.strip()
        if not obs:
            continue
        if obs not in nxt.student_prefix_text:
            raise RolloutError(
                f"{traj.task}: turn {prev.turn_index}'s observation does not appear "
                f"in turn {nxt.turn_index}'s prefix — the turns are not conditioned "
                "on each other, so this is not a multi-turn trajectory (§7.4)"
            )


@dataclass
class TurnAdvantages:
    """Detached per-token advantages for one turn, ready for the policy loss."""

    turn_index: int
    action_token_ids: list[int]
    behavior_logprobs: list[float]
    advantages: list[float]
    supervised_mask: list[bool]
    stats: ChunkStats

    @property
    def n_supervised(self) -> int:
        return sum(1 for s in self.supervised_mask if s)


def advantages_for_turn(
    turn: TurnRecord,
    teacher_token_bytes: list[bytes],
    teacher_logprobs: list[float],
    *,
    large_chunk_threshold: int | None = None,
    clamp: float | None = None,
) -> TurnAdvantages:
    """align -> semantic-prior credit assignment, for one turn.

    This is the seam §6.5 describes: byte-align the student's and teacher's
    tokenisations of the *same action bytes*, then assign the teacher's chunk
    likelihood back to student tokens. Alignment failures propagate as
    `AlignmentError` and credit-assignment failures as `ChunkOPDError`; neither
    is swallowed, because a finite-but-wrong loss is the failure mode the plan
    cares about most.
    """
    teacher_bytes = b"".join(teacher_token_bytes)
    if teacher_bytes != turn.action_bytes:
        raise RolloutError(
            f"turn {turn.turn_index}: teacher tokenisation covers {teacher_bytes!r} "
            f"but the sampled action was {turn.action_bytes!r} — the teacher scored "
            "different bytes than were executed"
        )

    alignment = align_by_bytes(turn.action_token_bytes, teacher_token_bytes)
    kwargs: dict[str, Any] = {"clamp": clamp}
    if large_chunk_threshold is not None:
        kwargs["large_chunk_threshold"] = large_chunk_threshold
    advantages, supervised, stats = assign_chunk_advantages(
        alignment, turn.behavior_logprobs, teacher_logprobs, **kwargs
    )
    return TurnAdvantages(
        turn_index=turn.turn_index,
        action_token_ids=list(turn.action_token_ids),
        behavior_logprobs=list(turn.behavior_logprobs),
        advantages=advantages,
        supervised_mask=supervised,
        stats=stats,
    )


def global_supervised_token_count(batches: list[TurnAdvantages]) -> int:
    """§7.3 step 4: the single denominator for the whole optimizer batch.

    Summed across every turn of every trajectory, counting only positions that
    carry a real advantage. Per-example means would let a short action weigh as
    much as a long one.
    """
    return sum(b.n_supervised for b in batches)


def assert_single_policy_version(trajs: list[TrajectoryRecord]) -> str:
    """Every trajectory in one update must come from the same frozen policy.

    §7.4: "both trajectories were generated entirely by frozen v0". Mixing
    versions makes `log pi_old` mean two different things inside one ratio.
    """
    if not trajs:
        raise RolloutError("no trajectories in this batch")
    versions = {t.policy_version for t in trajs}
    if len(versions) != 1:
        raise RolloutError(
            f"batch mixes policy versions {sorted(versions)} — one optimizer step "
            "must be built from a single frozen policy (§7.4)"
        )
    return versions.pop()


__all__ = [
    "RolloutError",
    "TrajectoryRecord",
    "TurnAdvantages",
    "TurnRecord",
    "advantages_for_turn",
    "assert_observation_carried",
    "assert_single_policy_version",
    "global_supervised_token_count",
]
