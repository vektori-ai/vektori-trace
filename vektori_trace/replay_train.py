"""The optimizer step for one replay update (plan §8.3, §7.3).

`run_replay_chunk_opd` takes `optimizer_step` as an injected callable so the
batch semantics stay testable without a GPU. This is the concrete one: recompute
`log pi_current` under autograd, apply `chunk_opd.clipped_is_policy_loss` with a
single global denominator, take exactly one step, save `v_replay`.

Four properties it exists to guarantee
--------------------------------------
1. **One optimizer step, not one per microbatch.** §8.3 says "exactly one
   optimizer step". Microbatches accumulate raw sums and the division happens
   once, at the end, by the batch-wide supervised-token count. `backward` is
   linear, so accumulating raw and rescaling once is exactly the global
   denominator without holding every graph at once.
2. **`v0` is never overwritten.** The adapter is loaded from `v0` and saved to a
   *new* directory. §8.3: "Archive `v_replay` separately".
3. **The step is proven to have moved something.** A run that silently produced
   no gradient — every position a sentinel, a masking bug — would otherwise
   report success and save a byte-identical adapter. The step compares LoRA
   tensors before and after and fails if nothing changed.
4. **Nothing but ck75's sampled tokens is supervised.** The advantages arrive
   per-student-token from `chunk_opd`, and the mask comes from the sentinel
   flags. No prefix token is ever in the tensor.

Everything torch/peft is imported lazily so the module (and the semantics tests
around it) import without the train extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .replay_opd import ReplayBatch, ReplayOPDError


class ReplayTrainError(ReplayOPDError):
    """The optimizer step cannot run, or ran without effect."""


@dataclass
class ReplayTrainConfig:
    """Knobs for the single update. Everything here belongs in the manifest."""

    base_model: str = "Qwen/Qwen3-14B"
    #: The frozen v0 adapter. Loaded, never written.
    adapter_path: str | None = None
    #: Where v_replay goes. Must differ from `adapter_path`.
    output_dir: Path = Path("./vektori-out/opd-replay/v_replay")
    learning_rate: float = 1e-5
    #: PPO-style clip width; pinned at the reference default (see chunk_opd).
    clip_eps: float | None = None
    #: Sequences per forward pass. Replay actions are short (ck75 samples ~260
    #: tokens) but prefixes are not, so this stays small by default.
    microbatch_size: int = 1
    max_grad_norm: float | None = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    device: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.adapter_path and Path(self.adapter_path) == self.output_dir:
            raise ReplayTrainError(
                "output_dir equals adapter_path — v0 must not be overwritten "
                "(§8.3: archive v_replay separately)"
            )


def _lora_params(model: Any) -> list[tuple[str, Any]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def current_logprobs(model: Any, token_ids: list[int], *, device: Any = None):
    """`log pi_current` for each supplied id, autograd live.

    Next-token convention: logits at position t score token t+1, so the first
    supplied id is context and receives no score. The caller aligns advantages
    to the returned length rather than assuming a 1:1 with `token_ids`.
    """
    import torch

    x = torch.tensor([token_ids], device=device)
    out = model(input_ids=x)
    logits = out.logits[:, :-1, :].float()
    targets = x[:, 1:]
    lp = torch.log_softmax(logits, dim=-1)
    return lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).squeeze(0)


def make_optimizer_step(
    model: Any,
    optimizer: Any,
    cfg: ReplayTrainConfig,
    *,
    save: bool = True,
):
    """Build the `optimizer_step` callable `run_replay_chunk_opd` expects.

    Returned closure takes a `ReplayBatch` and returns a report dict. Taking the
    model and optimizer from outside keeps this testable on a tiny CPU model:
    the semantics under test are the accumulation and the single step, not which
    weights they touch.
    """
    import torch

    from .chunk_opd import DEFAULT_CLIP_EPS, clip_fraction, clipped_is_policy_loss

    clip_eps = cfg.clip_eps if cfg.clip_eps is not None else DEFAULT_CLIP_EPS

    def step(batch: ReplayBatch) -> dict[str, Any]:
        denom = batch.global_supervised_tokens
        if denom <= 0:
            raise ReplayTrainError(
                "no supervised tokens — a step here would be a no-op reported "
                "as a success"
            )

        params = [p for _n, p in _lora_params(model)]
        if not params:
            raise ReplayTrainError("no trainable parameters — is the adapter attached?")
        before = [p.detach().clone() for p in params]

        model.train()
        optimizer.zero_grad(set_to_none=True)

        n_supervised = 0
        loss_total = 0.0
        clip_fracs: list[float] = []

        for adv in batch.advantages:
            ids = adv.action_token_ids
            if len(ids) < 2:
                # Nothing to score: the first id is context under the
                # next-token convention.
                continue
            cur = current_logprobs(model, ids, device=cfg.device)
            n = cur.shape[0]

            beh = torch.tensor(
                adv.behavior_logprobs[1 : n + 1], dtype=torch.float32,
                device=cur.device,
            )
            a = torch.tensor(
                adv.advantages[1 : n + 1], dtype=torch.float32, device=cur.device
            )
            mask = torch.tensor(
                [1.0 if s else 0.0 for s in adv.supervised_mask[1 : n + 1]],
                dtype=torch.float32,
                device=cur.device,
            )

            # Raw sum per example; the single global division happens below.
            loss = clipped_is_policy_loss(
                cur, beh, a, mask, clip_eps=clip_eps, denominator=1.0
            )
            # Scale before backward so accumulated gradients already carry the
            # global denominator — equivalent to summing then dividing, without
            # keeping every graph alive.
            (loss / denom).backward()

            loss_total += float(loss.detach().item())
            n_supervised += int(mask.sum().item())
            if bool((mask > 0).any()):
                clip_fracs.append(clip_fraction(cur.detach(), beh, mask, clip_eps=clip_eps))

        if n_supervised == 0:
            raise ReplayTrainError(
                "every supervised position was masked out — nothing to learn from"
            )

        grad_norm = None
        if cfg.max_grad_norm is not None:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            )
        finite = all(
            bool(torch.isfinite(p.grad).all()) for p in params if p.grad is not None
        )
        if not finite:
            raise ReplayTrainError("non-finite gradient (§11 stop condition)")

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # §8.4: v_replay must differ from v0. A step that changed nothing would
        # otherwise be reported as a success and saved as a duplicate.
        moved = sum(
            1 for b, p in zip(before, params, strict=True)
            if not torch.equal(b, p.detach())
        )
        if moved == 0:
            raise ReplayTrainError(
                "the optimizer step changed no LoRA parameter — v_replay would "
                "be byte-identical to v0"
            )

        max_delta = max(
            float((p.detach() - b).abs().max())
            for b, p in zip(before, params, strict=True)
        )

        saved: str | None = None
        if save:
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(cfg.output_dir))
            saved = str(cfg.output_dir)

        return {
            "loss": loss_total / denom,
            "global_supervised_tokens": denom,
            "supervised_positions_scored": n_supervised,
            "n_examples": len(batch.advantages),
            "grad_norm": grad_norm,
            "clip_fraction": sum(clip_fracs) / len(clip_fracs) if clip_fracs else 0.0,
            "clip_eps": clip_eps,
            "learning_rate": cfg.learning_rate,
            "lora_tensors_moved": moved,
            "lora_tensors_total": len(params),
            "max_param_delta": max_delta,
            "adapter_saved_to": saved,
            "optimizer_steps": 1,
        }

    return step


def load_v0_for_training(cfg: ReplayTrainConfig):
    """Load the frozen v0 adapter as a trainable LoRA model.

    Kept separate from `make_optimizer_step` so the step can be tested against a
    tiny model while this stays the one place that touches ck75's real weights.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not cfg.adapter_path:
        raise ReplayTrainError("adapter_path is required to load v0")

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, cfg.adapter_path, is_trainable=True)
    if not _lora_params(model):
        raise ReplayTrainError(
            f"{cfg.adapter_path} loaded but exposes no trainable LoRA parameters"
        )
    return model


def build_optimizer(model: Any, cfg: ReplayTrainConfig):
    import torch

    params = [p for _n, p in _lora_params(model)]
    return torch.optim.AdamW(
        params,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )


__all__ = [
    "ReplayTrainConfig",
    "ReplayTrainError",
    "build_optimizer",
    "current_logprobs",
    "load_v0_for_training",
    "make_optimizer_step",
]
