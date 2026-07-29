"""Step H — OPD + RL training branches (PLAN.md).

Configuration:
  teacher: Qwen3-Coder-Next 80B (prompt_logprobs)
  student: Qwen3-8B + LoRA
  loss: reverse KL (mode-seeking)
  trainer target: verl distillation_ppo_loss + GRPO

Defines losses, branch configs, and a local reverse-KL helper that mirrors
verl's distillation path for tests / dry-runs. Full verl launch via VerlLaunchSpec.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .tokenizer_check import (
    DEFAULT_STUDENT,
    DEFAULT_TEACHER,
    check_tokenizers,
)

Branch = Literal["RL", "OPD"]


@dataclass
class OPDConfig:
    teacher_model: str = DEFAULT_TEACHER
    student_model: str = DEFAULT_STUDENT
    distillation_loss_coef: float = 1.0
    reverse_kl: bool = True  # mode-seeking; 8B cannot cover 80B
    teacher_endpoint: str | None = None
    lora_rank: int = 16
    learning_rate: float = 1e-5
    max_steps: int = 200
    # C1: OPD scores *supplied* tokens, so the teacher must be self-hosted
    # (vLLM/SGLang). A spec without an endpoint cannot run the loss it declares.
    require_prompt_logprobs: bool = True


@dataclass
class GRPOConfig:
    student_model: str = DEFAULT_STUDENT
    group_size: int = 8
    learning_rate: float = 1e-5
    max_steps: int = 200
    lora_rank: int = 16
    # ReOPD: no containers in the loop — prefixes pre-collected
    use_reopd_prefixes: bool = True


@dataclass
class VerlLaunchSpec:
    """Serializable launch spec for verl (GRPO and/or distillation_ppo_loss)."""

    branch: Branch
    student_model: str
    teacher_model: str | None = None
    teacher_endpoint: str | None = None
    distillation_loss_coef: float = 0.0
    group_size: int = 8
    lora_rank: int = 16
    learning_rate: float = 1e-5
    max_steps: int = 200
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mean_log_ratio(
    student_logprobs: list[float],
    teacher_logprobs: list[float],
) -> float:
    """mean(log π_s − log π_t) over supplied tokens — a monitoring scalar.

    Deliberately not named for the loss: it carries no gradient and is not the
    training objective. It is the sample estimate of the reverse-KL *coefficient*
    that `reverse_kl_surrogate` (the actual objective) detaches and weights
    `∇log π_s` by, logged during dry runs to watch the divergence move.
    """
    if len(student_logprobs) != len(teacher_logprobs):
        raise ValueError(
            f"logprob length mismatch: student={len(student_logprobs)} "
            f"teacher={len(teacher_logprobs)} — tokenizer check should have caught this"
        )
    if not student_logprobs:
        return 0.0
    return sum(s - t for s, t in zip(student_logprobs, teacher_logprobs, strict=True)) / len(
        student_logprobs
    )


def _require_train() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover - env guard
        raise RuntimeError(
            "training extras required: install with `uv sync --extra train`"
        ) from e


def token_logprobs(logits: Any, tokens: Any) -> Any:
    """log π(token_t | prefix) for each supplied token — the OPD scoring path.

    `logits` is [B, T, V] over positions predicting `tokens` [B, T]; the caller
    owns the HF next-token shift (see `dataset.py`), so no shifting happens here.
    """
    _require_train()
    import torch

    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return logprobs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def reverse_kl_surrogate(
    student_logprobs: Any,
    teacher_logprobs: Any,
    loss_mask: Any | None = None,
) -> Any:
    """Trainable reverse-KL objective whose gradient is `Σ (log π_s − log π_t) ∇log π_s`.

    That is the gradient PLAN.md and `docs/OPD.md` state for OPD: nonzero
    wherever student and teacher disagree, regardless of outcome, so it can move
    mass into regions the student assigns ≈0 probability.

    The coefficient is detached; only the `∇log π_s` path carries gradient, which
    is what makes this a policy-gradient surrogate rather than a plain difference
    of logprobs. `teacher_logprobs` come from the teacher's `prompt_logprobs` and
    never require grad.

    `loss_mask` is 1 on supervised tokens and 0 elsewhere — the student prefix in
    teacher-continuation training, and every non-assistant span, are masked out
    exactly as in `dataset.py`. Normalised by the number of supervised tokens so
    the scale does not drift with sequence length.
    """
    _require_train()
    import torch

    if student_logprobs.shape != teacher_logprobs.shape:
        raise ValueError(
            f"logprob shape mismatch: student={tuple(student_logprobs.shape)} "
            f"teacher={tuple(teacher_logprobs.shape)} — tokenizer check should "
            "have caught this"
        )
    teacher = teacher_logprobs.detach()
    coef = (student_logprobs.detach() - teacher)
    per_token = coef * student_logprobs
    if loss_mask is None:
        denom = torch.tensor(
            float(per_token.numel()), dtype=per_token.dtype, device=per_token.device
        )
        total = per_token.sum()
    else:
        mask = loss_mask.to(per_token.dtype)
        total = (per_token * mask).sum()
        denom = mask.sum()
    if float(denom) == 0.0:
        # Nothing supervised in this batch — a zero that still carries a graph,
        # so the optimiser step is a no-op rather than a crash.
        return per_token.sum() * 0.0
    return total / denom


def mask_from_labels(labels: Any, ignore_index: int = -100) -> Any:
    """1 where `labels` carries loss, 0 at IGNORE_INDEX — bridges `dataset.py`."""
    _require_train()
    return (labels != ignore_index)


def grpo_advantage(rewards: list[float]) -> list[float]:
    """Group-relative advantages: (r - mean) / std within the group."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = var**0.5
    if std < 1e-8:
        return [0.0] * len(rewards)
    return [(r - mean) / std for r in rewards]


def build_branch_spec(
    branch: Branch,
    *,
    opd: OPDConfig | None = None,
    grpo: GRPOConfig | None = None,
    verify_tokenizers: bool = True,
) -> VerlLaunchSpec:
    """Build a verl launch spec for RL (GRPO) or OPD (reverse KL distillation)."""
    if branch == "OPD":
        cfg = opd or OPDConfig()
        if verify_tokenizers:
            check_tokenizers(
                cfg.teacher_model,
                cfg.student_model,
            )
        if cfg.require_prompt_logprobs and not cfg.teacher_endpoint:
            raise RuntimeError(
                "OPD requires prompt_logprobs (PLAN.md C1) — set "
                "OPDConfig.teacher_endpoint to a self-hosted vLLM/SGLang base "
                "URL, or set require_prompt_logprobs=False for a dry-run spec"
            )
        return VerlLaunchSpec(
            branch="OPD",
            student_model=cfg.student_model,
            teacher_model=cfg.teacher_model,
            teacher_endpoint=cfg.teacher_endpoint,
            distillation_loss_coef=cfg.distillation_loss_coef,
            lora_rank=cfg.lora_rank,
            learning_rate=cfg.learning_rate,
            max_steps=cfg.max_steps,
            extra={"loss": "reverse_kl", "verl": "distillation_ppo_loss"},
        )

    cfg_g = grpo or GRPOConfig()
    return VerlLaunchSpec(
        branch="RL",
        student_model=cfg_g.student_model,
        group_size=cfg_g.group_size,
        lora_rank=cfg_g.lora_rank,
        learning_rate=cfg_g.learning_rate,
        max_steps=cfg_g.max_steps,
        extra={
            "loss": "grpo",
            "verl": "grpo",
            "use_reopd_prefixes": cfg_g.use_reopd_prefixes,
        },
    )


def blended_loss(
    policy_loss: float,
    distill_loss: float,
    *,
    distillation_loss_coef: float,
) -> float:
    """verl-style blend: policy + coef * distillation."""
    return policy_loss + distillation_loss_coef * distill_loss


__all__ = [
    "Branch",
    "GRPOConfig",
    "OPDConfig",
    "VerlLaunchSpec",
    "blended_loss",
    "build_branch_spec",
    "grpo_advantage",
    "mask_from_labels",
    "mean_log_ratio",
    "reverse_kl_surrogate",
    "token_logprobs",
]
