"""Cross-tokenizer OPD: semantic-prior credit assignment + clipped IS policy loss.

This is the loss of record for `docs/OPD-MULTITURN-PLAN.md` — a port of Niu et
al., *Breaking the Tokenizer Barrier: On-Policy Distillation across Model
Families* (arXiv:2606.09456), pinned at revision `927a8264`. The upstream
source sits in `vektori_trace/opd_reference/` so the port can be diffed against it.

Why this module exists rather than a tweak to `cross_kl.py`
-----------------------------------------------------------
`cross_kl.span_surrogate` ("estimator B") treats a whole aligned span as **one
sampled unit**: it forms `(Σlog π_s − Σlog π_t) · Σlog π_s` and lets autograd
spread that over the span's tokens. That is a different estimator from the
published one, and the plan (§6.5) says to replace it on the training path.
The difference is not cosmetic — for an m-token chunk, estimator B applies one
scalar coefficient to the *sum*, whereas the paper solves

    min_q  ½ Σ (log q_i − k log p_i)²    s.t.  Σ log q_i = L_T

whose closed form redistributes the teacher's chunk budget across student tokens
*in proportion to the student's own log-probabilities*, preserving the student's
internal semantic structure within the chunk:

    log q_i = (L_T / L_S) · log p_i
    A_i     = log q_i − log p_i = (L_T / L_S − 1) · log p_i

Boundary with `cross_kl.py`
---------------------------
`cross_kl.py` stays as a legacy regression oracle and diagnostic — `distill.py`
still calls it, and deleting it would lose the estimator-A comparison. But it is
**not permitted on a cross-tokenizer training path**:

- `chunk_opd` is the only allowed loss for DeepSeek→Qwen runs;
- `cross_kl.cross_step_loss` / `span_surrogate` are diagnostic-only;
- `assert_chunk_loss_selected()` below makes a config that asks for estimator B
  on a `--cross-tokenizer` production run fail loudly rather than silently
  optimising the unpublished objective.

Nothing here reaches back into `cross_kl`.

The 1:1 reduction that makes this checkable
-------------------------------------------
For a 1:1 chunk, `L_S = log p_i` and `L_T = log π_t`, so

    A_i = (L_T / L_S − 1) · log p_i = L_T − L_S = log π_t(t_i) − log π_old(s_i)

which is exactly traditional sampled reverse-KL. `test_chunk_opd.py` asserts
this against `align.py`'s equivalence oracle, and it is the single test most
likely to catch an indexing slip, so it is written first.

Gradient ownership
------------------
Everything the teacher touched is detached. `A_i` is detached in full — it is
built from `log π_old` (the frozen behaviour policy captured at rollout) and the
teacher's scores, never from `log π_current`. Gradient flows *only* through
`log π_current` inside the importance ratio, which is what makes this a policy
loss rather than a regression onto teacher logprobs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .align import Alignment, AlignmentError

#: Chunks larger than this on either side become an `inf` sentinel rather than a
#: hard failure. Upstream default (`_align_chunks(large_chunk_threshold=6)`):
#: runs this long are typically garbled/replacement-character regions where the
#: redistributed teacher likelihood is not trustworthy. Sentinel positions get
#: advantage 0, so they contribute no gradient — they are excluded from the
#: numerator but still counted, so a run that silently stops aligning shows up
#: as a falling `aligned_fraction` rather than a quietly rescaled loss.
DEFAULT_LARGE_CHUNK_THRESHOLD = 6

#: Upstream's degenerate guard: when |L_S| is under this, `L_T / L_S` explodes,
#: so the teacher budget is spread uniformly instead (`core_algos.py:1120`).
DEGENERATE_LS_EPS = 1e-8

#: PPO-style clip width for the importance ratio. Upstream takes this from
#: verl's `clip_ratio`; 0.2 is verl's default.
DEFAULT_CLIP_EPS = 0.2


class ChunkOPDError(ValueError):
    """Hard failure in chunk credit assignment.

    Distinct from `AlignmentError`: that one means the byte streams did not
    line up, this one means they did but the numbers attached to them are
    unusable (wrong length, non-finite teacher score, absent behaviour policy).
    Both fail closed — the plan's §11 stop conditions treat a silently finite
    loss as the worst outcome.
    """


@dataclass
class ChunkStats:
    """Per-action alignment and advantage diagnostics (plan §10 report fields)."""

    n_chunks: int = 0
    n_one_to_one: int = 0
    n_one_to_many: int = 0
    n_many_to_one: int = 0
    n_many_to_many: int = 0
    #: Student tokens carrying a real (non-sentinel) advantage.
    n_supervised_tokens: int = 0
    #: Student tokens zeroed by the large-chunk / unaligned-tail sentinel.
    n_sentinel_tokens: int = 0
    n_degenerate_chunks: int = 0
    n_clamped_tokens: int = 0
    max_chunk_student_tokens: int = 0
    max_chunk_teacher_tokens: int = 0
    max_chunk_bytes: int = 0
    #: SimpleOPD's exact-span coverage, logged as a diagnostic only (plan §2).
    exact_1to1_token_fraction: float = 0.0
    advantage_sum: float = 0.0
    advantage_abs_sum: float = 0.0
    advantage_min: float = 0.0
    advantage_max: float = 0.0
    #: §10 "advantage sign": counts, not just the mean, because a mean near zero
    #: is produced both by a teacher that agrees and by large opposing pressures
    #: that cancel. Those are different situations and must be distinguishable.
    n_advantage_positive: int = 0
    n_advantage_negative: int = 0
    n_advantage_zero: int = 0
    #: §10 "contribution by chunk type" — summed |A_i| and token count keyed by
    #: "1:1"/"1:N"/"N:1"/"M:N", so a report can say whether the signal is coming
    #: from cleanly aligned positions or from the ragged ones.
    advantage_abs_by_kind: dict[str, float] = field(default_factory=dict)
    tokens_by_kind: dict[str, int] = field(default_factory=dict)
    #: §10 "chunk byte/token length distributions" — full histograms rather than
    #: just the maxima, keyed by length.
    student_len_hist: dict[int, int] = field(default_factory=dict)
    teacher_len_hist: dict[int, int] = field(default_factory=dict)
    byte_len_hist: dict[int, int] = field(default_factory=dict)
    chunk_kinds: list[str] = field(default_factory=list)

    @property
    def aligned_fraction(self) -> float:
        total = self.n_supervised_tokens + self.n_sentinel_tokens
        return self.n_supervised_tokens / total if total else 0.0

    @property
    def mean_advantage(self) -> float:
        return self.advantage_sum / self.n_supervised_tokens if self.n_supervised_tokens else 0.0

    @property
    def mean_abs_advantage(self) -> float:
        """Magnitude irrespective of sign — §10 "advantage magnitude"."""
        return (
            self.advantage_abs_sum / self.n_supervised_tokens
            if self.n_supervised_tokens
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        d = {
            k: v for k, v in self.__dict__.items() if k != "chunk_kinds"
        }
        # Histogram keys are ints; JSON object keys must be strings.
        for k in ("student_len_hist", "teacher_len_hist", "byte_len_hist"):
            d[k] = {str(n): c for n, c in sorted(d[k].items())}
        d["aligned_fraction"] = self.aligned_fraction
        d["mean_advantage"] = self.mean_advantage
        d["mean_abs_advantage"] = self.mean_abs_advantage
        return d


def _chunk_kind(n_s: int, n_t: int) -> str:
    if n_s == 1 and n_t == 1:
        return "1:1"
    if n_s == 1:
        return "1:N"
    if n_t == 1:
        return "N:1"
    return "M:N"


def assign_chunk_advantages(
    alignment: Alignment,
    behavior_logprobs: list[float],
    teacher_logprobs: list[float],
    *,
    large_chunk_threshold: int = DEFAULT_LARGE_CHUNK_THRESHOLD,
    clamp: float | None = None,
) -> tuple[list[float], list[bool], ChunkStats]:
    """Semantic-prior credit assignment — the paper's Eq. for `log q_i` and `A_i`.

    Consumes byte-aligned spans from `align.align_by_bytes` (our aligner; see the
    opd_reference README for why we align on bytes instead of upstream's decode-and-
    compare) and applies upstream's per-chunk redistribution unchanged.

    Args:
        alignment: spans covering **exactly** the sampled action on both sides.
            Prefix, template, and environment tokens must already be excluded —
            this function has no way to tell them apart and will happily
            supervise them if handed them (plan §7.4).
        behavior_logprobs: `log π_old(s_i)` per student action token, captured
            during rollout under the frozen policy version. Not `log π_current`:
            using the training-time value here would put the student's own
            gradient inside the detached advantage.
        teacher_logprobs: `log π_D(t_j)` per teacher action token.
        large_chunk_threshold: sentinel cutoff, upstream default 6.
        clamp: optional symmetric per-token advantage clamp
            (`opd_loss_max_clamp`). `None` = no clamp, upstream's default.

    Returns:
        `(advantages, supervised_mask, stats)`. `advantages[i]` is `A_i` for
        student token *i*; `supervised_mask[i]` is False on sentinel positions,
        whose advantage is 0.0 and which must be excluded from the loss
        denominator by the caller.

    Raises:
        ChunkOPDError: on length disagreement or a non-finite input score. A
            teacher `-inf` is a real event (the teacher assigned the realized
            token zero probability) but it makes `L_T` non-finite and the whole
            chunk meaningless, so it fails closed rather than propagating NaN.
    """
    if len(behavior_logprobs) != alignment.n_student_tokens:
        raise ChunkOPDError(
            f"behavior_logprobs has {len(behavior_logprobs)} entries but alignment "
            f"covers {alignment.n_student_tokens} student tokens — refusing to "
            "truncate to the shorter (plan §6.5: a length mismatch is a hard error)"
        )
    if len(teacher_logprobs) != alignment.n_teacher_tokens:
        raise ChunkOPDError(
            f"teacher_logprobs has {len(teacher_logprobs)} entries but alignment "
            f"covers {alignment.n_teacher_tokens} teacher tokens — refusing to "
            "truncate to the shorter (plan §6.5)"
        )
    for name, seq in (("behavior", behavior_logprobs), ("teacher", teacher_logprobs)):
        for i, v in enumerate(seq):
            if not math.isfinite(v):
                raise ChunkOPDError(
                    f"{name}_logprobs[{i}] = {v!r} is not finite. A non-finite "
                    "score makes the chunk's L_T/L_S ratio meaningless; this is a "
                    "plan §11 stop condition, not something to clamp away."
                )

    n_s = alignment.n_student_tokens
    advantages = [0.0] * n_s
    supervised = [False] * n_s
    stats = ChunkStats()

    for span in alignment.spans:
        n_span_s = span.n_student
        n_span_t = span.n_teacher
        stats.n_chunks += 1
        kind = _chunk_kind(n_span_s, n_span_t)
        stats.chunk_kinds.append(kind)
        if kind == "1:1":
            stats.n_one_to_one += 1
        elif kind == "1:N":
            stats.n_one_to_many += 1
        elif kind == "N:1":
            stats.n_many_to_one += 1
        else:
            stats.n_many_to_many += 1
        stats.max_chunk_student_tokens = max(stats.max_chunk_student_tokens, n_span_s)
        stats.max_chunk_teacher_tokens = max(stats.max_chunk_teacher_tokens, n_span_t)
        stats.max_chunk_bytes = max(stats.max_chunk_bytes, span.byte_len)
        stats.student_len_hist[n_span_s] = stats.student_len_hist.get(n_span_s, 0) + 1
        stats.teacher_len_hist[n_span_t] = stats.teacher_len_hist.get(n_span_t, 0) + 1
        stats.byte_len_hist[span.byte_len] = stats.byte_len_hist.get(span.byte_len, 0) + 1

        # Sentinel: upstream marks over-long chunks inf and lets core_algos turn
        # that into advantage 0. We carry the mask explicitly instead of encoding
        # it as inf, so a sentinel can never be mistaken for a real score.
        if n_span_s > large_chunk_threshold or n_span_t > large_chunk_threshold:
            stats.n_sentinel_tokens += n_span_s
            continue

        L_S = sum(behavior_logprobs[i] for i in span.student_idx)
        L_T = sum(teacher_logprobs[j] for j in span.teacher_idx)

        if abs(L_S) < DEGENERATE_LS_EPS:
            # Upstream `core_algos.py:1120`: student assigns ~0 logprob mass to
            # the chunk, so the proportional rule divides by ~0. Spread the
            # teacher budget uniformly instead.
            stats.n_degenerate_chunks += 1
            target = L_T / n_span_s
            for i in span.student_idx:
                a = target - behavior_logprobs[i]
                advantages[i] = a
                supervised[i] = True
        else:
            ratio = L_T / L_S
            for i in span.student_idx:
                # log q_i = (L_T / L_S) · log p_i ; A_i = log q_i − log p_i
                advantages[i] = (ratio - 1.0) * behavior_logprobs[i]
                supervised[i] = True

        stats.n_supervised_tokens += n_span_s
        stats.tokens_by_kind[kind] = stats.tokens_by_kind.get(kind, 0) + n_span_s
        stats.advantage_abs_by_kind[kind] = stats.advantage_abs_by_kind.get(
            kind, 0.0
        ) + sum(abs(advantages[i]) for i in span.student_idx)

    if clamp is not None:
        if clamp <= 0:
            raise ChunkOPDError(f"clamp must be > 0, got {clamp!r}")
        any_clamped = False
        for i, a in enumerate(advantages):
            if supervised[i] and abs(a) > clamp:
                advantages[i] = math.copysign(clamp, a)
                stats.n_clamped_tokens += 1
                any_clamped = True
        if any_clamped:
            # The per-kind magnitudes were accumulated pre-clamp; recompute them
            # so the reported contribution matches the advantages actually used.
            stats.advantage_abs_by_kind = {}
            for span, kind in zip(alignment.spans, stats.chunk_kinds, strict=True):
                if not all(supervised[i] for i in span.student_idx):
                    continue
                stats.advantage_abs_by_kind[kind] = stats.advantage_abs_by_kind.get(
                    kind, 0.0
                ) + sum(abs(advantages[i]) for i in span.student_idx)

    live = [advantages[i] for i in range(n_s) if supervised[i]]
    if live:
        stats.advantage_sum = sum(live)
        stats.advantage_abs_sum = sum(abs(a) for a in live)
        stats.advantage_min = min(live)
        stats.advantage_max = max(live)
        stats.n_advantage_positive = sum(1 for a in live if a > 0.0)
        stats.n_advantage_negative = sum(1 for a in live if a < 0.0)
        stats.n_advantage_zero = sum(1 for a in live if a == 0.0)
    one_to_one_tokens = sum(
        sp.n_student
        for sp in alignment.spans
        if sp.n_student == 1 and sp.n_teacher == 1
    )
    stats.exact_1to1_token_fraction = one_to_one_tokens / n_s if n_s else 0.0

    return advantages, supervised, stats


def _require_train() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover - env guard
        raise RuntimeError(
            "training extras required: install with `uv sync --extra train`"
        ) from e


def clipped_is_policy_loss(
    current_logprobs: Any,
    behavior_logprobs: Any,
    advantages: Any,
    loss_mask: Any | None = None,
    *,
    clip_eps: float = DEFAULT_CLIP_EPS,
    denominator: float | None = None,
) -> Any:
    """The paper's clipped importance-sampling objective, per student token.

        rho_i = exp(log π_current(s_i) − log π_old(s_i))
        loss  = −mean_i min(rho_i · A_i, clip(rho_i, 1−ε, 1+ε) · A_i)

    Negated because callers minimise: maximising the clipped objective is
    minimising its negation, and every trainer in this repo calls `.backward()`
    on a loss it descends.

    Only `current_logprobs` carries gradient. `behavior_logprobs` and
    `advantages` are detached here regardless of what the caller passes, so a
    caller that forgets cannot leak the student's gradient into the advantage.

    `denominator` mirrors `cross_kl.cross_step_loss`'s contract, because the
    plan (§7.3) requires one *global* normalisation across the whole batch, not
    per-example means:

    * `None` — self-normalise by this call's supervised-token count. Correct
      when the step has one microbatch; used by the tests.
    * a float — return the raw sum divided by that. A batch loop passes
      `1.0`, accumulates, and rescales once by the global supervised-token
      count, which is exactly the global denominator because backward is linear.
    """
    _require_train()
    import torch

    if current_logprobs.shape != behavior_logprobs.shape:
        raise ChunkOPDError(
            f"shape mismatch: current={tuple(current_logprobs.shape)} "
            f"behavior={tuple(behavior_logprobs.shape)}"
        )
    if current_logprobs.shape != advantages.shape:
        raise ChunkOPDError(
            f"shape mismatch: current={tuple(current_logprobs.shape)} "
            f"advantages={tuple(advantages.shape)}"
        )

    old = behavior_logprobs.detach().float()
    adv = advantages.detach().float()
    cur = current_logprobs.float()

    log_rho = cur - old
    rho = torch.exp(log_rho)
    unclipped = rho * adv
    clipped = torch.clamp(rho, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_token = -torch.min(unclipped, clipped)

    if loss_mask is None:
        total = per_token.sum()
        count = torch.tensor(
            float(per_token.numel()), dtype=per_token.dtype, device=per_token.device
        )
    else:
        mask = loss_mask.to(per_token.dtype)
        total = (per_token * mask).sum()
        count = mask.sum()

    if denominator is not None:
        if denominator <= 0.0:
            raise ChunkOPDError(f"denominator must be > 0, got {denominator!r}")
        return total / float(denominator)
    if float(count) == 0.0:
        # Nothing supervised — a zero that still carries a graph, so the
        # optimiser step is a no-op rather than a crash. Same convention as
        # `opd.reverse_kl_surrogate` and `cross_kl.cross_step_loss`.
        return per_token.sum() * 0.0
    return total / count


def clip_fraction(
    current_logprobs: Any,
    behavior_logprobs: Any,
    loss_mask: Any | None = None,
    *,
    clip_eps: float = DEFAULT_CLIP_EPS,
) -> float:
    """Fraction of supervised tokens whose ratio left the trust region.

    Monitoring only, no gradient. Plan §10 asks for clipping to be reported;
    a rising value means the update is fighting the behaviour policy, which on
    a one-step smoke should be ≈0 (current == behaviour, so rho == 1).
    """
    _require_train()
    import torch

    with torch.no_grad():
        rho = torch.exp(current_logprobs.float() - behavior_logprobs.float())
        out = (rho < 1.0 - clip_eps) | (rho > 1.0 + clip_eps)
        if loss_mask is None:
            return float(out.float().mean().item()) if out.numel() else 0.0
        mask = loss_mask.to(out.dtype if out.dtype.is_floating_point else torch.float32)
        denom = float(mask.sum().item())
        return float((out.float() * mask).sum().item()) / denom if denom else 0.0


#: The per-turn generation cap the previous OPD run used. Plan §7.1: "Use the
#: task-derived per-turn token cap already validated for ck75; do not return to
#: the previous 256-token cap." It is still `distill.FireworksOPDConfig`'s
#: default, so a driver that forgets to override it silently reproduces the old
#: truncation — hence an explicit guard rather than a comment.
PREVIOUS_TOKEN_CAP = 256

#: Floor for a task-derived cap. Not a tuned value: it is simply above
#: `PREVIOUS_TOKEN_CAP`, so "did you override the old default" is answerable.
MIN_TASK_DERIVED_TOKEN_CAP = 512


def assert_token_cap_is_task_derived(max_new_tokens: int) -> None:
    """Refuse the previous 256-token per-turn cap (plan §7.1).

    A truncated action is not a wrong action the teacher can grade — it is a
    *different* action whose tail never existed. Alignment still succeeds on the
    truncated bytes, so the loss stays finite and the failure is invisible in
    every downstream metric, which is why this is a hard gate rather than a
    warning.
    """
    if max_new_tokens <= 0:
        raise ChunkOPDError(f"max_new_tokens must be > 0, got {max_new_tokens!r}")
    if max_new_tokens <= PREVIOUS_TOKEN_CAP:
        raise ChunkOPDError(
            f"max_new_tokens={max_new_tokens} is at or below the previous "
            f"{PREVIOUS_TOKEN_CAP}-token cap, which plan §7.1 forbids returning to. "
            "Pass the task-derived cap validated for ck75 "
            f"(>= {MIN_TASK_DERIVED_TOKEN_CAP})."
        )
    if max_new_tokens < MIN_TASK_DERIVED_TOKEN_CAP:
        raise ChunkOPDError(
            f"max_new_tokens={max_new_tokens} is below the "
            f"{MIN_TASK_DERIVED_TOKEN_CAP} floor for a task-derived cap (§7.1)."
        )


#: Loss identifiers a cross-tokenizer production run may select. `cross_kl`'s
#: estimator B stays reachable for diagnostics and legacy regression, but it is
#: not the published objective and must never be what a paid run optimises.
CHUNK_LOSS_ID = "chunk_opd"
LEGACY_LOSS_IDS = frozenset({"cross_step_loss", "span_surrogate", "estimator_b"})


def assert_chunk_loss_selected(loss_id: str, *, cross_tokenizer: bool = True) -> None:
    """Fail unless a cross-tokenizer run is configured for the published loss.

    Call this from any `--cross-tokenizer` production entry point before
    spending. `cross_kl`'s span surrogate produces a finite, plausible number
    from the same inputs, so a misconfiguration is invisible in the logs — which
    is exactly the class of failure the plan's §11 stop conditions target.

    Same-tokenizer runs (`cross_tokenizer=False`) are unaffected:
    `opd.reverse_kl_surrogate` remains correct there.
    """
    if not cross_tokenizer:
        return
    if loss_id in LEGACY_LOSS_IDS:
        raise ChunkOPDError(
            f"loss {loss_id!r} is legacy/diagnostic only and cannot be used for a "
            "cross-tokenizer training run: it treats a whole aligned span as one "
            f"sampled unit, which is not the published objective. Use {CHUNK_LOSS_ID!r} "
            "(docs/OPD-MULTITURN-PLAN.md §6.5)."
        )
    if loss_id != CHUNK_LOSS_ID:
        raise ChunkOPDError(
            f"unknown loss {loss_id!r} for a cross-tokenizer run; expected "
            f"{CHUNK_LOSS_ID!r}"
        )


__all__ = [
    "CHUNK_LOSS_ID",
    "DEFAULT_CLIP_EPS",
    "DEFAULT_LARGE_CHUNK_THRESHOLD",
    "DEGENERATE_LS_EPS",
    "LEGACY_LOSS_IDS",
    "MIN_TASK_DERIVED_TOKEN_CAP",
    "PREVIOUS_TOKEN_CAP",
    "AlignmentError",
    "ChunkOPDError",
    "ChunkStats",
    "assert_chunk_loss_selected",
    "assert_token_cap_is_task_derived",
    "assign_chunk_advantages",
    "clip_fraction",
    "clipped_is_policy_loss",
]
