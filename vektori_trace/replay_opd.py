"""`run_replay_chunk_opd` — the driver that joins parts this repo already has.

Plan §8 (`docs/OPD-MULTITURN-PLAN.md`), one replay-prefix update from frozen
`v0`. Nothing here reimplements ReOPD or the chunk loss; it is the adapter
between four existing pieces:

    replay_select   which stored states to act at
    reopd           prefix splitting, teacher-action exclusion, masking
    teacher/cross   pinned DeepSeek render + Fireworks realized-token scoring
    chunk_opd       the cross-tokenizer semantic-prior loss

Why a driver rather than importing the ReOPD stack
--------------------------------------------------
The official ReOPD code (arXiv:2607.04763) assumes SLIME/Megatron/SGLang, a
local HF teacher, same-tokenizer model pairs, and a multi-GPU reproduction run.
Our setup is ck75 + Fireworks-hosted DeepSeek + *different* tokenizers + the
chunk loss. Adopting their stack would mean rebuilding the project around it and
would still not solve the cross-tokenizer problem, which is the whole point of
`chunk_opd`. What we take from the paper is its semantics — student acts once at
a stored prefix, no environment call during training — and its early-step
schedule (`replay_select.reopd_step_weights`), not its infrastructure.

The invariant that makes this replay rather than SFT
----------------------------------------------------
The stored DeepSeek action at the replay step is **never** a target. It is
excluded from the prefix by `reopd.prefix_turns_through_step` and does not enter
the loss in any form. Every supervised token is one ck75 sampled, scored by
DeepSeek under its own tokenizer. If that stops being true the run is replay SFT
wearing an OPD label, so `assert_action_is_student_sampled` checks it explicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .chunk_opd import assert_token_cap_is_task_derived
from .opd_rollout import TurnAdvantages, advantages_for_turn
from .replay_select import ReplayPrefix, ReplaySelectionError, assert_no_source_dominates


class ReplayOPDError(RuntimeError):
    """The replay update cannot proceed as specified."""


@dataclass
class SampledAction:
    """One ck75 action sampled at a replay prefix, with its behaviour scores.

    `sample_index` distinguishes the four independent draws §8.3 asks for at
    each prefix. `policy_version` is recorded per action rather than per batch
    so a mixed-version batch is detectable at the point of use.
    """

    prefix_id: str
    sample_index: int
    action_bytes: bytes
    action_token_ids: list[int]
    action_token_bytes: list[bytes]
    behavior_logprobs: list[float]
    policy_version: str
    termination_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.action_token_ids)
        if n == 0:
            raise ReplayOPDError(
                f"{self.prefix_id}#{self.sample_index}: empty action — a zero-token "
                "sample has no gradient and must be dropped, not scored"
            )
        if len(self.behavior_logprobs) != n:
            raise ReplayOPDError(
                f"{self.prefix_id}#{self.sample_index}: "
                f"{len(self.behavior_logprobs)} behaviour logprobs for {n} tokens"
            )
        if len(self.action_token_bytes) != n:
            raise ReplayOPDError(
                f"{self.prefix_id}#{self.sample_index}: "
                f"{len(self.action_token_bytes)} token byte strings for {n} tokens"
            )
        if b"".join(self.action_token_bytes) != self.action_bytes:
            raise ReplayOPDError(
                f"{self.prefix_id}#{self.sample_index}: token bytes do not "
                "reconstruct the sampled action"
            )

    @property
    def key(self) -> str:
        return f"{self.prefix_id}#{self.sample_index}"


@dataclass
class ReplayBatch:
    """Everything one optimizer step consumes, after scoring and alignment."""

    prefixes: list[ReplayPrefix]
    advantages: list[TurnAdvantages]
    keys: list[str]
    policy_version: str
    supervised_tokens_by_prefix: dict[str, int]
    spread_report: dict[str, Any]

    @property
    def global_supervised_tokens(self) -> int:
        return sum(a.n_supervised for a in self.advantages)


def assert_action_is_student_sampled(
    action: SampledAction, stored_teacher_action_bytes: bytes | None
) -> None:
    """The supervised action must be ck75's, not the stored DeepSeek one (§8.4).

    A byte-identical match is not proof of a bug on its own — a short, highly
    constrained action can legitimately coincide — but it is the signature of
    accidentally supervising the trace's own continuation, which would make this
    replay SFT. Treated as a hard failure: the alternative is a run whose label
    is wrong and whose numbers look fine.
    """
    if stored_teacher_action_bytes is None:
        return
    if action.action_bytes == stored_teacher_action_bytes:
        raise ReplayOPDError(
            f"{action.key}: the sampled action is byte-identical to the stored "
            "DeepSeek action at this step. The stored continuation must never be "
            "the supervised target (§8.1). If this is a genuine coincidence, drop "
            "the sample rather than relaxing the check."
        )


def build_replay_batch(
    prefixes: Sequence[ReplayPrefix],
    actions: Sequence[SampledAction],
    teacher_tokenizations: dict[str, tuple[list[bytes], list[float]]],
    *,
    stored_teacher_actions: dict[str, bytes] | None = None,
    max_task_share: float = 0.5,
    max_trace_share: float = 0.35,
    large_chunk_threshold: int | None = None,
    advantage_clamp: float | None = None,
) -> ReplayBatch:
    """Align and credit-assign every sampled action; enforce §8.4's spread.

    `teacher_tokenizations` maps an action key to `(teacher_token_bytes,
    teacher_logprobs)` — the output of scoring that exact action under the
    pinned DeepSeek renderer. Producing it is the caller's job because it is the
    only step that costs money, and a driver that silently re-scored would make
    the cost unpredictable.

    Fails closed on a missing score rather than dropping the action: a batch
    quietly short of a few samples still normalises and still trains, which is
    exactly the invisible-error class §11 targets.
    """
    by_id = {p.prefix_id: p for p in prefixes}
    if not actions:
        raise ReplayOPDError("no sampled actions in this batch")

    versions = {a.policy_version for a in actions}
    if len(versions) != 1:
        raise ReplayOPDError(
            f"batch mixes policy versions {sorted(versions)} — §8.3 freezes one "
            "ck75 version for all samples"
        )
    policy_version = versions.pop()

    advantages: list[TurnAdvantages] = []
    keys: list[str] = []
    tokens_by_prefix: dict[str, int] = {}

    for i, action in enumerate(actions):
        if action.prefix_id not in by_id:
            raise ReplayOPDError(
                f"{action.key}: no such prefix in this batch ({action.prefix_id})"
            )
        if action.key not in teacher_tokenizations:
            raise ReplayOPDError(
                f"{action.key}: no teacher score. Refusing to train on a partial "
                "batch — a missing action changes the denominator silently."
            )
        if stored_teacher_actions is not None:
            assert_action_is_student_sampled(
                action, stored_teacher_actions.get(action.prefix_id)
            )

        teacher_bytes, teacher_lps = teacher_tokenizations[action.key]
        # `advantages_for_turn` wants a TurnRecord-shaped object; a replay action
        # is one turn with no observation, so build it inline rather than forcing
        # callers to construct a fake trajectory.
        turn = _as_turn(action, turn_index=i)
        adv = advantages_for_turn(
            turn,
            teacher_bytes,
            teacher_lps,
            large_chunk_threshold=large_chunk_threshold,
            clamp=advantage_clamp,
        )
        advantages.append(adv)
        keys.append(action.key)
        tokens_by_prefix[action.prefix_id] = (
            tokens_by_prefix.get(action.prefix_id, 0) + adv.n_supervised
        )

    used = [by_id[pid] for pid in tokens_by_prefix]
    spread = assert_no_source_dominates(
        tokens_by_prefix,
        used,
        max_task_share=max_task_share,
        max_trace_share=max_trace_share,
    )

    return ReplayBatch(
        prefixes=list(used),
        advantages=advantages,
        keys=keys,
        policy_version=policy_version,
        supervised_tokens_by_prefix=tokens_by_prefix,
        spread_report=spread,
    )


def _as_turn(action: SampledAction, *, turn_index: int) -> Any:
    """Adapt a replay action to the `TurnRecord` shape `advantages_for_turn` takes.

    A replay sample is a single turn whose prefix is the stored trace and whose
    observation never happens — ReOPD's defining property is that no environment
    is queried during training. `student_prefix_text` is left empty because
    nothing downstream of here reads it: only the action tokens are supervised.
    """
    from .opd_rollout import TurnRecord

    return TurnRecord(
        turn_index=turn_index,
        student_prefix_text="",
        action_bytes=action.action_bytes,
        action_token_ids=action.action_token_ids,
        action_token_bytes=action.action_token_bytes,
        behavior_logprobs=action.behavior_logprobs,
        observation="",
        policy_version=action.policy_version,
        termination_reason=action.termination_reason,
    )


def run_replay_chunk_opd(
    prefixes: Sequence[ReplayPrefix],
    actions: Sequence[SampledAction],
    teacher_tokenizations: dict[str, tuple[list[bytes], list[float]]],
    optimizer_step: Callable[[ReplayBatch], dict[str, Any]],
    *,
    max_new_tokens: int,
    n_samples_per_prefix: int = 4,
    stored_teacher_actions: dict[str, bytes] | None = None,
    selection_policy: str = "stratified-diagnostic",
    advantage_clamp: float | None = None,
    large_chunk_threshold: int | None = None,
) -> dict[str, Any]:
    """One replay-prefix OPD update, end to end, minus the GPU.

    `optimizer_step` receives the assembled `ReplayBatch` and owns the
    differentiable half: recompute `log pi_current`, apply
    `chunk_opd.clipped_is_policy_loss` with a single global denominator, one
    `optimizer.step()`, save `v_replay`. It is injected rather than implemented
    here so this function is testable without a GPU and so the training backend
    can change without touching the batch semantics.

    Returns a report, not a model. §10 wants the alignment and advantage
    statistics archived regardless of what the step did with them.

    `selection_policy` is recorded, not enforced: §8.3's stratified sample and
    ReOPD's `kappa^t` schedule are different experiments and a report that does
    not say which one it ran cannot be compared with anything.
    """
    # §7.1: the previous 256-token cap must not come back. Checked here rather
    # than at sampling time too, because a batch sampled under a bad cap is
    # still unusable and this is the last place to catch it.
    assert_token_cap_is_task_derived(max_new_tokens)

    if n_samples_per_prefix <= 0:
        raise ReplayOPDError(
            f"n_samples_per_prefix must be > 0, got {n_samples_per_prefix}"
        )

    expected = len(prefixes) * n_samples_per_prefix
    if len(actions) != expected:
        raise ReplayOPDError(
            f"{len(actions)} actions for {len(prefixes)} prefixes x "
            f"{n_samples_per_prefix} samples (expected {expected}). A short batch "
            "silently changes the global denominator."
        )

    batch = build_replay_batch(
        prefixes,
        actions,
        teacher_tokenizations,
        stored_teacher_actions=stored_teacher_actions,
        large_chunk_threshold=large_chunk_threshold,
        advantage_clamp=advantage_clamp,
    )
    if batch.global_supervised_tokens <= 0:
        raise ReplayOPDError(
            "no supervised tokens survived alignment — every action was sentinel; "
            "an optimizer step here would be a no-op reported as a success"
        )

    step_report = optimizer_step(batch)

    chunk_kinds: dict[str, int] = {}
    for adv in batch.advantages:
        for kind, n in adv.stats.tokens_by_kind.items():
            chunk_kinds[kind] = chunk_kinds.get(kind, 0) + n

    return {
        "policy_version": batch.policy_version,
        "selection_policy": selection_policy,
        "n_prefixes": len(batch.prefixes),
        "n_actions": len(actions),
        "n_samples_per_prefix": n_samples_per_prefix,
        "max_new_tokens": max_new_tokens,
        "global_supervised_tokens": batch.global_supervised_tokens,
        "supervised_tokens_by_prefix": batch.supervised_tokens_by_prefix,
        "supervised_tokens_by_chunk_kind": chunk_kinds,
        "spread": batch.spread_report,
        "prefix_ids": [p.prefix_id for p in batch.prefixes],
        "action_keys": batch.keys,
        "per_action_stats": [a.stats.to_dict() for a in batch.advantages],
        "optimizer": step_report,
    }


__all__ = [
    "ReplayBatch",
    "ReplayOPDError",
    "ReplaySelectionError",
    "SampledAction",
    "assert_action_is_student_sampled",
    "build_replay_batch",
    "run_replay_chunk_opd",
]
