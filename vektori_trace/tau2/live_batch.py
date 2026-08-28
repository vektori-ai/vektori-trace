"""Turn projected per-token teacher credit into a `ReplayBatch`.

`build_replay_batch` takes `(teacher_token_bytes, teacher_logprobs)` per action
and re-aligns the **raw** student bytes against them. That is correct for the
replay arm, where the scored span *is* the student's action. It is exactly
wrong for a projected live score: the teacher scored a DeepSeek-native
rendering whose bytes are not the Qwen action's, so re-aligning would
reintroduce the contamination `live_score` exists to remove.

So the live path stops before that seam. `score_live_action` has already done
the alignment -- per payload, on byte-identical text -- and produced
`teacher_logprob_by_index`: a chunk logprob for each *eligible* student token.
This module converts that into the `TurnAdvantages` the optimizer consumes,
applying upstream's credit rule (`chunk_opd`) per token rather than per
re-derived chunk:

    A_i = (L_T / L_S - 1) * log p_i        for an eligible token
    A_i = 0, supervised_mask[i] = False    for everything else

The mask is the whole point. A token absent from `teacher_logprob_by_index` --
`<tool_call>`, `<|im_end|>`, tool JSON, a boundary straddler -- contributes
nothing to the loss *and* nothing to the global denominator, so it can neither
push the policy nor dilute the tokens that should.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

__all__ = [
    "LiveBatchError",
    "estimate_batch_shares",
    "projected_turn_advantages",
    "build_projected_batch",
]

#: Below this the proportional rule divides by ~0; upstream spreads the teacher
#: budget uniformly instead (`core_algos.py:1120`).
DEGENERATE_LS_EPS = 1e-8


class LiveBatchError(RuntimeError):
    """A projected batch that must not reach the optimizer."""


def estimate_batch_shares(
    prefixes: Sequence[Any],
    actions: Sequence[Any],
    *,
    max_task_share: float,
    max_trace_share: float,
) -> dict[str, Any]:
    """Would this batch's balance check pass? Answerable before scoring.

    `build_projected_batch` weights each turn by its *supervised token* count,
    which is not known until the teacher has been paid. But the dominant term
    is how many turns each task contributed and how long they are, and raw
    action-token counts are on disk the moment sampling ends. So this
    approximates the same ratio from `action_token_ids` and reports whether the
    real check is likely to refuse the batch.

    The point is placement, not precision. On 2026-08-28 update 2 paid $0.05
    and 410 s of DeepSeek scoring before `build_projected_batch` refused a
    batch whose imbalance -- task 93 at 15 of 34 turns -- was determined the
    instant the rollout ended. A gate that only fires after the expensive stage
    is a gate in the wrong place.

    This is deliberately advisory: it NEVER replaces the real check, because it
    estimates a quantity it cannot know exactly. It exists so a doomed batch is
    refused in one second instead of after a paid scoring pass.
    """
    by_task: dict[str, int] = {}
    by_trace: dict[str, int] = {}
    total = 0
    for prefix, action in zip(prefixes, actions):
        n = len(getattr(action, "token_ids", None)
                or getattr(action, "action_token_ids", None) or ())
        if n <= 0:
            n = 1
        total += n
        by_task[prefix.task] = by_task.get(prefix.task, 0) + n
        by_trace[prefix.trace_id] = by_trace.get(prefix.trace_id, 0) + n

    if total == 0:
        return {"ok": False, "reason": "no sampled tokens", "total": 0}

    task_share = {k: v / total for k, v in by_task.items()}
    trace_share = {k: v / total for k, v in by_trace.items()}
    worst_task = max(task_share.values()) if task_share else 0.0
    worst_trace = max(trace_share.values()) if trace_share else 0.0

    reasons = []
    if worst_task > max_task_share:
        top = max(task_share, key=task_share.get)
        reasons.append(
            f"task {top} would take ~{worst_task:.3f} of the batch, over "
            f"{max_task_share:.3f}"
        )
    if worst_trace > max_trace_share:
        top = max(trace_share, key=trace_share.get)
        reasons.append(
            f"episode {top} would take ~{worst_trace:.3f} of the batch, over "
            f"{max_trace_share:.3f}"
        )

    return {
        "ok": not reasons,
        "reason": "; ".join(reasons),
        "estimated_task_share": {k: round(v, 4) for k, v in task_share.items()},
        "estimated_trace_share": {k: round(v, 4)
                                  for k, v in trace_share.items()},
        "worst_task_share": round(worst_task, 4),
        "worst_trace_share": round(worst_trace, 4),
        "total_action_tokens": total,
        "basis": "raw action tokens (pre-scoring estimate, not the real check)",
    }


def projected_turn_advantages(
    *,
    turn_index: int,
    action_token_ids: list[int],
    behavior_logprobs: list[float],
    teacher_logprob_by_index: dict[int, float],
    prompt_token_ids: list[int] | None = None,
    clamp: float | None = None,
) -> Any:
    """Per-token advantages from already-aligned teacher credit.

    `teacher_logprob_by_index[i]` is the teacher's log-likelihood for the chunk
    covering student token `i`, as established by `score_live_action` against
    byte-identical payload text. Tokens absent from it are unsupervised.
    """
    from vektori_trace.chunk_opd import ChunkStats
    from vektori_trace.opd_rollout import TurnAdvantages

    n = len(action_token_ids)
    if len(behavior_logprobs) != n:
        raise LiveBatchError(
            f"turn {turn_index}: {len(behavior_logprobs)} behavior logprobs for "
            f"{n} action tokens"
        )

    advantages = [0.0] * n
    supervised = [False] * n
    stats = ChunkStats()
    n_clamped = 0

    for i in range(n):
        if i not in teacher_logprob_by_index:
            continue
        L_T = float(teacher_logprob_by_index[i])
        L_S = float(behavior_logprobs[i])
        if not math.isfinite(L_T) or not math.isfinite(L_S):
            raise LiveBatchError(
                f"turn {turn_index}: non-finite logprob at token {i} "
                f"(teacher={L_T}, behavior={L_S})"
            )
        if abs(L_S) < DEGENERATE_LS_EPS:
            # Upstream's degenerate branch: the student assigns ~no mass here,
            # so the proportional rule is undefined. Hand the token the
            # teacher's budget directly rather than dividing by ~0.
            adv = L_T - L_S
            stats.n_degenerate_chunks += 1
        else:
            adv = (L_T / L_S - 1.0) * L_S
        if clamp is not None and abs(adv) > clamp:
            adv = math.copysign(clamp, adv)
            n_clamped += 1
        advantages[i] = adv
        supervised[i] = True
        stats.n_supervised_tokens += 1
        stats.advantage_sum += adv
        stats.advantage_abs_sum += abs(adv)

    stats.n_clamped_tokens = n_clamped
    stats.n_sentinel_tokens = n - stats.n_supervised_tokens

    return TurnAdvantages(
        turn_index=turn_index,
        action_token_ids=list(action_token_ids),
        behavior_logprobs=list(behavior_logprobs),
        advantages=advantages,
        supervised_mask=supervised,
        stats=stats,
        prompt_token_ids=list(prompt_token_ids) if prompt_token_ids else None,
    )


def build_projected_batch(
    prefixes: Sequence[Any],
    actions: Sequence[Any],
    projected_scores: dict[str, Any],
    *,
    policy_version: str,
    max_task_share: float = 0.5,
    max_trace_share: float = 0.35,
    clamp: float | None = None,
    enforce_shares: bool = True,
) -> Any:
    """`ReplayBatch` from projected scores, without re-aligning raw bytes.

    Mirrors `build_replay_batch`'s contract -- fails closed on a missing score
    rather than dropping the action, because a silently smaller batch changes
    the global denominator -- but consumes `ProjectedScore` instead of
    `(teacher_bytes, teacher_logprobs)`.
    """
    from vektori_trace.replay_opd import ReplayBatch

    if len(prefixes) != len(actions):
        raise LiveBatchError(
            f"{len(prefixes)} prefixes for {len(actions)} actions"
        )

    advantages = []
    keys = []
    by_prefix: dict[str, int] = {}
    by_task: dict[str, int] = {}
    by_trace: dict[str, int] = {}

    for i, (prefix, action) in enumerate(zip(prefixes, actions)):
        score = projected_scores.get(action.key)
        if score is None:
            raise LiveBatchError(
                f"{action.key}: no projected score. Refusing to train on a "
                "partial batch -- a missing action changes the denominator "
                "silently."
            )
        if action.policy_version != policy_version:
            raise LiveBatchError(
                f"{action.key}: sampled under {action.policy_version!r}, but "
                f"this batch is {policy_version!r}"
            )
        ta = projected_turn_advantages(
            turn_index=i,
            action_token_ids=action.action_token_ids,
            behavior_logprobs=action.behavior_logprobs,
            teacher_logprob_by_index=score.teacher_logprob_by_index,
            prompt_token_ids=action.prompt_token_ids,
            clamp=clamp,
        )
        advantages.append(ta)
        keys.append(action.key)
        by_prefix[prefix.prefix_id] = by_prefix.get(prefix.prefix_id, 0) + ta.n_supervised
        by_task[prefix.task] = by_task.get(prefix.task, 0) + ta.n_supervised
        by_trace[prefix.trace_id] = by_trace.get(prefix.trace_id, 0) + ta.n_supervised

    total = sum(a.n_supervised for a in advantages)
    if total == 0:
        raise LiveBatchError(
            "the projected batch supervises zero tokens; an optimizer step "
            "over an empty denominator is not a step"
        )

    task_share = {k: v / total for k, v in by_task.items()}
    trace_share = {k: v / total for k, v in by_trace.items()}
    worst_task = max(task_share.values()) if task_share else 0.0
    worst_trace = max(trace_share.values()) if trace_share else 0.0

    # `enforce_shares=False` is the LIVE contract, and it is a deliberate
    # methodological choice rather than a relaxation.
    #
    # Replay selected prefixes from a frozen corpus of thousands: a lopsided
    # batch was reshuffled for free, so the share limits were a *sampling*
    # constraint (`replay_select.assert_no_source_dominates`, plan §8.4).
    # Live episodes are generated one at a time at real cost, and their length
    # is an OUTCOME, not a knob -- task 93 runs 15 turns because the student
    # struggles there.
    #
    # Refusing a batch after seeing its realized length is therefore
    # outcome-dependent selection: hard tasks produce more states, so a
    # length-triggered gate systematically excludes the very trajectories
    # on-policy distillation exists to learn from. Upstream bounds pathology
    # with `max_turns` and token caps, not with share rejection.
    #
    # Balance belongs BEFORE the rollout -- equal preregistered episode counts
    # per task -- not after it. The shares are still computed and reported, so
    # concentration is visible in every run report.
    if enforce_shares:
        if worst_task > max_task_share:
            raise LiveBatchError(
                f"task share {worst_task:.3f} exceeds {max_task_share}; one task "
                "dominates the update"
            )
        if worst_trace > max_trace_share:
            raise LiveBatchError(
                f"trace share {worst_trace:.3f} exceeds {max_trace_share}; one "
                "episode dominates the update"
            )

    return ReplayBatch(
        prefixes=list(prefixes),
        advantages=advantages,
        keys=keys,
        policy_version=policy_version,
        supervised_tokens_by_prefix=by_prefix,
        spread_report={
            "total_supervised_tokens": total,
            "task_share": task_share,
            "trace_share": trace_share,
            "max_task_share": worst_task,
            "max_trace_share": worst_trace,
            "projection": "semantic (reasoning+content); markup, tool "
                          "serialization and boundary tokens carry zero weight",
        },
    )
