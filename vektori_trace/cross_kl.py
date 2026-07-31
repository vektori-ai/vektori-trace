"""Cross-tokenizer OPD loss: coarse-grained reverse KL and policy-gradient surrogate.

FINAL-PLAN.md §6 — reverse KL, two estimators, one shared global denominator.

Public API:
    CrossStepStats              — per-step monitoring statistics
    coarse_grained_reverse_kl   — Estimator A: analytic K+1-event coarse-grained reverse KL
    span_surrogate              — Estimator B: policy-gradient surrogate wrapping
                                  opd.reverse_kl_surrogate
    cross_step_loss             — assemble contributions with ONE global denominator

All arithmetic is float32 (log1p near 1.0 loses too much in bf16, §10 of the plan).
No beta parameter anywhere — the objective is not parameterised (§2, §6).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vektori_trace.align import Alignment, SpanKind


def _require_train() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "training extras required: install with `uv sync --extra train`"
        ) from e


# ── CrossStepStats ─────────────────────────────────────────────────────────────


@dataclass
class CrossStepStats:
    """Per-step monitoring statistics from cross_step_loss.

    Fields match FINAL-PLAN.md §Deliverable-2 exactly; no beta field.
    ``dropped_by_reason`` covers §10.10 content-type / cause breakdown for
    dropped spans (keyed by classify_spans reason string).
    """

    n_spans: int
    granularity: float
    frac_A: float               # fraction of supervised spans using Estimator A
    frac_B: float               # fraction of supervised spans using Estimator B
    frac_dropped: float         # fraction of all spans that are dropped
    bytes_aligned: int          # bytes covered by non-dropped spans
    bytes_total: int            # bytes in all spans
    special_tokens_masked: int  # spans dropped because they contained only specials
    mean_mapped_teacher_mass: float  # mean mapped-teacher-mass fraction for A spans
    student_entropy: float      # mean Shannon entropy of π_s at A-span positions
    dropped_by_reason: dict[str, int] | None = None


# ── Estimator A: coarse-grained reverse KL ─────────────────────────────────────


def coarse_grained_reverse_kl(
    student_logprobs_full: Any,
    mapped_pairs: list[tuple[int, float]],
    teacher_topk: dict[int, float],
    coverage_threshold: float = 0.9,
    other_floor: float = 1e-9,
) -> tuple[Any, float] | None:
    """Analytic coarse-grained reverse KL for a 1↔1 span (Estimator A).

    Implements the K+1-event partition (FINAL-PLAN.md §6 (A))::

        mapped_t = [lp_i for mapped teacher entries]
        other_t  = log1p(−Σ exp(mapped_t))               # fp32
        mapped_s = [student logprob at mapped student id] # from full log_softmax
        other_s  = log1p(−Σ exp(mapped_s))               # differentiable
        loss_A   = Σ_e π_s(e) · (log π_s(e) − log π_t(e).detach())

    Args:
        student_logprobs_full: [V] float32 tensor — full log_softmax at this student
            position; must carry gradient for the backward pass.
        mapped_pairs: pre-filtered ``[(student_id, teacher_logprob), ...]`` for the
            teacher top-K entries that have a byte-identical student token.
        teacher_topk: raw ``{teacher_id: logprob}`` dict — used *only* for the
            coverage gate; not differentiable.
        coverage_threshold: minimum ratio (mapped teacher mass / total top-K mass)
            required to proceed with Estimator A.  Below threshold, returns None and
            the caller should fall back to Estimator B.
        other_floor: minimum probability for the "other" bucket.  When fp8
            quantisation noise pushes Σexp(mapped_t) ≥ 1, clamping prevents NaN.

    Returns:
        ``(contribution_tensor, mapped_teacher_mass_fraction)`` on success, or
        ``None`` when the coverage gate fires.  The contribution is the raw
        per-span term (no global denominator); the caller divides by the total
        student-token count.
    """
    _require_train()
    import torch

    if not mapped_pairs or not teacher_topk:
        return None

    # ── Coverage gate ────────────────────────────────────────────────────────
    total_topk_mass = sum(math.exp(lp) for lp in teacher_topk.values())
    mapped_t_masses = [math.exp(lp) for _, lp in mapped_pairs]
    mapped_t_total = sum(mapped_t_masses)
    mapped_coverage = mapped_t_total / total_topk_mass if total_topk_mass > 0.0 else 0.0

    if mapped_coverage < coverage_threshold:
        return None

    # ── Teacher side: K+1 log-probs (fp32, detached) ─────────────────────────
    log_floor = math.log(other_floor)
    # fp8 noise: Σexp ≥ 1.0 → clamp "other" to floor; count silently (§10.11).
    other_t_val = log_floor if mapped_t_total >= 1.0 else math.log1p(-mapped_t_total)

    mapped_t_lps = [lp for _, lp in mapped_pairs]
    teacher_lps = torch.tensor(
        [*mapped_t_lps, other_t_val],
        dtype=torch.float32,
        device=student_logprobs_full.device,
    )  # [K+1], no grad — teacher side always detached

    # ── Student side: K+1 log-probs (fp32, differentiable) ───────────────────
    lp_full = student_logprobs_full.float()  # [V], may carry grad
    mapped_s_ids = torch.tensor(
        [sid for sid, _ in mapped_pairs],
        dtype=torch.long,
        device=lp_full.device,
    )
    mapped_s_lps = lp_full[mapped_s_ids]  # [K], inherits grad from lp_full

    # Differentiable other_s = log(1 − Σexp(mapped_s_lps))
    sum_exp_s = mapped_s_lps.exp().sum()
    sum_exp_s_val = sum_exp_s.detach().item()
    if sum_exp_s_val >= 1.0:
        # Numerical edge case (or fp32 rounding when almost all mass is mapped).
        other_s_tensor = torch.tensor(log_floor, dtype=torch.float32, device=lp_full.device)
    else:
        other_s_tensor = torch.log1p(-sum_exp_s)  # differentiable via autograd

    # [K+1]: mapped student log-probs followed by the "other" log-prob
    student_lps_cat = torch.cat([mapped_s_lps, other_s_tensor.unsqueeze(0)])

    # ── loss_A = Σ_e π_s(e) · (log π_s(e) − log π_t(e).detach()) ────────────
    pi_s = student_lps_cat.exp()  # [K+1], differentiable
    contribution = (pi_s * (student_lps_cat - teacher_lps.detach())).sum()  # scalar

    return contribution, mapped_coverage


# ── Estimator B: policy-gradient surrogate ─────────────────────────────────────


def span_surrogate(
    log_p_s: Any,
    log_p_t: float | Any,
) -> Any:
    """Per-span policy-gradient surrogate for Estimator B.

    Wraps ``opd.reverse_kl_surrogate`` applied to single-element tensors,
    returning the raw per-span contribution (no global denominator).
    ``cross_step_loss`` sums these contributions and divides by the global
    student-token count::

        coef   = log_p_s.detach() − log_p_t     # detached; no grad
        return   coef × log_p_s                  # gradient: coef · ∇log_p_s

    Args:
        log_p_s: scalar tensor — Σ log π_s(student tokens in span), differentiable.
        log_p_t: float or scalar tensor — Σ log π_t(teacher tokens in span); will
            be detached if a tensor is supplied.

    Returns:
        Scalar tensor: the per-span surrogate contribution.  For a 1:1 span this
        is identical to one term of ``opd.reverse_kl_surrogate``, which is the
        basis of the B-path equivalence oracle (test #12).
    """
    from vektori_trace import opd

    _require_train()
    import torch

    lp_s = log_p_s.float().reshape(1)
    if isinstance(log_p_t, torch.Tensor):
        lp_t = log_p_t.float().detach().reshape(1)
    else:
        lp_t = torch.tensor([float(log_p_t)], dtype=torch.float32, device=lp_s.device)

    # reverse_kl_surrogate normalises by numel()=1, so the result is the raw
    # per-span term (s − t) × s, exactly what the global denominator in
    # cross_step_loss needs.
    return opd.reverse_kl_surrogate(lp_s, lp_t)


# ── cross_step_loss ────────────────────────────────────────────────────────────


def cross_step_loss(
    alignment: Alignment,
    span_kinds: list[tuple[SpanKind, str]],
    student_logprobs_full: Any,
    student_token_logprobs: Any,
    teacher_token_logprobs: Sequence[float],
    teacher_topk_by_teacher_pos: dict[int, dict[int, float]],
    exact_map: dict[int, int],
    coverage_threshold: float = 0.9,
    other_floor: float = 1e-9,
) -> tuple[Any, CrossStepStats]:
    """Cross-tokenizer OPD loss with ONE shared global denominator.

    Iterates over spans, applies Estimator A or B, accumulates raw contributions,
    then divides by the total number of supervised student tokens.  This single
    denominator ensures that step-to-step drift in the A/B mix does not silently
    rescale the effective learning rate (FINAL-PLAN.md §6 Normalization).

    For each span:
    * **DROP** — skip; count ``special_tokens_masked`` when appropriate.
    * **ESTIMATOR_A** — try ``coarse_grained_reverse_kl``; fall back to B if the
      coverage gate fires or if no top-K data is available for this teacher position.
    * **ESTIMATOR_B** — ``span_surrogate(Σlog π_s, Σlog π_t)``.

    Args:
        alignment: from ``align_by_bytes``.
        span_kinds: output of ``classify_spans(alignment, ...)``.  Same length as
            ``alignment.spans``.
        student_logprobs_full: [n_student, V] float32 tensor — full log_softmax at
            every student position (required by Estimator A for the K+1 partition).
        student_token_logprobs: [n_student] tensor — per-token log π_s, used by
            Estimator B for span Σlog π_s sums.
        teacher_token_logprobs: length n_teacher — per-token log π_t scalars, used
            by Estimator B for span Σlog π_t sums.
        teacher_topk_by_teacher_pos: ``{teacher_position: {teacher_id: logprob}}``.
            Missing positions cause ESTIMATOR_A spans to fall back to B.
        exact_map: ``{teacher_id: student_id}`` byte-identical map from
            ``CrossTokenizerBridge``.
        coverage_threshold: passed to ``coarse_grained_reverse_kl``.
        other_floor: passed to ``coarse_grained_reverse_kl``.

    Returns:
        ``(loss, CrossStepStats)``.  When no spans are supervised the loss is a
        zero tensor that still carries a computation graph (optimiser step is a
        no-op, not a crash).
    """
    _require_train()
    import torch

    n_all_spans = len(alignment.spans)
    assert len(span_kinds) == n_all_spans, (
        f"span_kinds has {len(span_kinds)} entries but alignment has {n_all_spans} spans"
    )

    total_bytes = sum(sp.byte_len for sp in alignment.spans)

    contributions: list[Any] = []
    total_student_tokens = 0
    n_A = 0
    n_B = 0
    n_dropped = 0
    special_tokens_masked = 0
    bytes_aligned = 0
    mapped_teacher_masses: list[float] = []
    entropy_sum = 0.0
    n_entropy = 0
    dropped_by_reason: dict[str, int] = {}

    for span, (kind, reason) in zip(alignment.spans, span_kinds, strict=True):
        # ── DROP ─────────────────────────────────────────────────────────────
        if kind == SpanKind.DROP:
            n_dropped += 1
            dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + 1
            if "special" in reason.lower():
                special_tokens_masked += 1
            continue

        n_s = span.n_student

        # ── ESTIMATOR_A (1:1 spans) — try analytic path first ────────────────
        if kind == SpanKind.ESTIMATOR_A:
            t_idx = span.teacher_idx[0]
            topk = teacher_topk_by_teacher_pos.get(t_idx)

            if topk:
                mapped_pairs = [
                    (exact_map[tid], tlp)
                    for tid, tlp in topk.items()
                    if tid in exact_map
                ]
                s_idx = span.student_idx[0]
                lp_full_pos = student_logprobs_full[s_idx]  # [V]

                result_A = coarse_grained_reverse_kl(
                    lp_full_pos,
                    mapped_pairs,
                    topk,
                    coverage_threshold=coverage_threshold,
                    other_floor=other_floor,
                )

                if result_A is not None:
                    contrib, mass_frac = result_A
                    contributions.append(contrib)
                    total_student_tokens += n_s  # always 1 for 1:1 spans
                    bytes_aligned += span.byte_len
                    n_A += 1
                    mapped_teacher_masses.append(mass_frac)

                    # Shannon entropy H = −Σ_v π_s(v) log π_s(v), all fp32
                    lp_v = lp_full_pos.float()
                    p_v = lp_v.exp()
                    entropy = float(-(p_v * lp_v).sum().item())
                    entropy_sum += entropy
                    n_entropy += 1
                    continue
                # Coverage gate fired → fall through to Estimator B.

        # ── ESTIMATOR_B (multi-token spans, or A demoted by coverage gate) ──
        s_indices = list(span.student_idx)
        s_lps = student_token_logprobs[s_indices].float()
        log_p_s = s_lps.sum()  # Σ log π_s over student tokens in span

        log_p_t = sum(float(teacher_token_logprobs[j]) for j in span.teacher_idx)
        # Σ log π_t over teacher tokens in span — scalar, never differentiable

        contrib_b = span_surrogate(log_p_s, log_p_t)
        contributions.append(contrib_b)
        total_student_tokens += n_s
        bytes_aligned += span.byte_len
        n_B += 1

    # ── ONE global denominator: total supervised student tokens ───────────────
    if contributions:
        raw = torch.stack([c.float().squeeze() for c in contributions]).sum()
    else:
        # Zero that still carries a computation graph (no-op optimiser step).
        raw = student_token_logprobs.float().sum() * 0.0

    loss = raw / float(total_student_tokens) if total_student_tokens > 0 else raw * 0.0

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_supervised = n_A + n_B
    n_total = n_supervised + n_dropped

    frac_A = n_A / n_supervised if n_supervised > 0 else 0.0
    frac_B = n_B / n_supervised if n_supervised > 0 else 0.0
    frac_dropped = n_dropped / n_total if n_total > 0 else 0.0

    stats = CrossStepStats(
        n_spans=n_total,
        granularity=alignment.granularity,
        frac_A=frac_A,
        frac_B=frac_B,
        frac_dropped=frac_dropped,
        bytes_aligned=bytes_aligned,
        bytes_total=total_bytes,
        special_tokens_masked=special_tokens_masked,
        mean_mapped_teacher_mass=(
            sum(mapped_teacher_masses) / len(mapped_teacher_masses)
            if mapped_teacher_masses
            else 0.0
        ),
        student_entropy=entropy_sum / n_entropy if n_entropy > 0 else float("nan"),
        dropped_by_reason=dropped_by_reason or None,
    )

    return loss, stats


__all__ = [
    "CrossStepStats",
    "coarse_grained_reverse_kl",
    "cross_step_loss",
    "span_surrogate",
]
