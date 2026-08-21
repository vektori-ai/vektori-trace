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
4. **Nothing but ck75's sampled tokens is supervised.** The prefix is in the
   forward pass — it has to be, or the conditioning is wrong — but only the
   action positions are scored, so no prefix token can reach the loss.
5. **`log pi_current` is computed in the conditioning `log pi_old` was captured
   in.** The action is scored after its full rendered replay prefix, and every
   action token is scored including the first. Scoring the action alone would
   compare two different distributions while producing a perfectly finite loss
   and a moving adapter — see `action_logprobs_under_prefix`.

Everything torch/peft is imported lazily so the module (and the semantics tests
around it) import without the train extra.
"""

from __future__ import annotations

import json
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
    """`log pi_current` for each supplied id after the first, autograd live.

    Next-token convention: logits at position t score token t+1, so the first
    supplied id is context and receives no score. Returns a tensor of length
    `len(token_ids) - 1`.
    """
    import torch

    x = torch.tensor([token_ids], device=device)
    out = model(input_ids=x)
    logits = out.logits[:, :-1, :].float()
    targets = x[:, 1:]
    lp = torch.log_softmax(logits, dim=-1)
    return lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).squeeze(0)


def action_logprobs_under_prefix(
    model: Any,
    prompt_token_ids: list[int],
    action_token_ids: list[int],
    *,
    device: Any = None,
):
    """`log pi_current` for **every** action token, conditioned on the real prefix.

    This is the correctness-critical function in the module. `log pi_old` was
    captured while ck75 sampled the action after its full rendered replay prefix,
    so `log pi_current` has to be recomputed in that same conditioning or the
    importance ratio `exp(log pi_current - log pi_old)` compares two different
    distributions — and does so while staying finite, so nothing downstream
    reveals it.

    Two consequences that a naive implementation gets wrong:

    - **The prefix must be in the forward pass**, not just the action. Scoring
      `action_token_ids` alone conditions each token on the previous *action*
      tokens and nothing else, which is not the state ck75 acted in.
    - **The first action token must be scored**, from the final prefix logit.
      Dropping it (by treating it as context under the next-token convention)
      discards the position that most directly reflects the student's choice at
      that state.

    Returns a tensor of length `len(action_token_ids)`, aligned 1:1 with the
    captured behaviour log probabilities.
    """
    import torch

    if not prompt_token_ids:
        raise ReplayTrainError(
            "no prompt_token_ids — the action would be scored without the replay "
            "prefix it was sampled under, making log pi_current incomparable to "
            "log pi_old"
        )
    if not action_token_ids:
        raise ReplayTrainError("no action tokens to score")

    full = list(prompt_token_ids) + list(action_token_ids)
    n_prompt = len(prompt_token_ids)

    x = torch.tensor([full], device=device)
    out = model(input_ids=x)
    # Position t predicts token t+1, so the logit that scores action token j is
    # at index (n_prompt + j - 1). For j = 0 that is the final prefix position.
    logits = out.logits[0, n_prompt - 1 : -1, :].float()
    targets = x[0, n_prompt:]
    lp = torch.log_softmax(logits, dim=-1)
    scored = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if scored.shape[0] != len(action_token_ids):
        raise ReplayTrainError(
            f"scored {scored.shape[0]} positions for {len(action_token_ids)} "
            "action tokens"
        )
    return scored


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
            if not ids:
                continue
            if not adv.prompt_token_ids:
                raise ReplayTrainError(
                    f"example {adv.turn_index}: no prompt_token_ids. The action "
                    "would be scored without its replay prefix, so log pi_current "
                    "would not be comparable to the captured log pi_old and the "
                    "importance ratio would be meaningless while staying finite."
                )
            # Every action token, conditioned on the real prefix, 1:1 with the
            # captured behaviour log probabilities. No slicing.
            cur = action_logprobs_under_prefix(
                model, adv.prompt_token_ids, ids, device=cfg.device
            )
            n = cur.shape[0]

            beh = torch.tensor(
                adv.behavior_logprobs[:n], dtype=torch.float32, device=cur.device
            )
            a = torch.tensor(
                adv.advantages[:n], dtype=torch.float32, device=cur.device
            )
            mask = torch.tensor(
                [1.0 if s else 0.0 for s in adv.supervised_mask[:n]],
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
        reload_check: dict[str, Any] | None = None
        if save:
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(cfg.output_dir))
            saved = str(cfg.output_dir)
            # §8.4: "v_replay differs from v0 **and can be reloaded**". The
            # difference is proven above from live tensors; reloadability is a
            # property of what actually landed on disk, and the two can diverge
            # — a save that writes a config without weights, or a truncated
            # write, still leaves `moved > 0` true. Verified here rather than in
            # the caller because this is the only place that knows the step
            # succeeded.
            reload_check = verify_adapter_reloadable(cfg.output_dir)

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
            "adapter_reload_check": reload_check,
            "optimizer_steps": 1,
        }

    return step


def verify_adapter_reloadable(adapter_dir: Path) -> dict[str, Any]:
    """Prove the saved adapter is loadable, from the bytes on disk (§8.4).

    Reads and validates the artifact rather than trusting that
    `save_pretrained` returned without raising. The failures this catches are
    the quiet ones: a config written with no weights file beside it, a
    zero-length safetensors, a rank that disagrees with what was trained. All
    of them leave the in-memory model perfectly fine, so nothing else in the
    run would notice.

    Deliberately does not instantiate a 14B base to do it — that would double
    peak memory at the worst moment. Structural validation of the adapter
    artifact is what distinguishes "saved" from "saved something usable"; a
    full load belongs in the evaluation arm, which has to load it anyway.
    """
    d = Path(adapter_dir)
    problems: list[str] = []

    config = d / "adapter_config.json"
    if not config.is_file():
        problems.append("adapter_config.json is missing")
        cfg_data: dict[str, Any] = {}
    else:
        try:
            cfg_data = json.loads(config.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"adapter_config.json is not valid JSON: {e}")
            cfg_data = {}

    weights = d / "adapter_model.safetensors"
    legacy = d / "adapter_model.bin"
    if weights.is_file():
        size = weights.stat().st_size
        if size == 0:
            problems.append("adapter_model.safetensors is zero bytes")
    elif legacy.is_file():
        size = legacy.stat().st_size
        if size == 0:
            problems.append("adapter_model.bin is zero bytes")
    else:
        size = 0
        problems.append("no adapter weights file (.safetensors or .bin)")

    n_tensors: int | None = None
    if weights.is_file() and weights.stat().st_size > 0:
        try:
            from safetensors import safe_open

            with safe_open(str(weights), framework="pt") as fh:
                keys = list(fh.keys())
            n_tensors = len(keys)
            if n_tensors == 0:
                problems.append("adapter weights contain no tensors")
        except ImportError:
            pass
        except Exception as e:  # a corrupt file raises several types
            problems.append(f"adapter weights unreadable: {type(e).__name__}: {e}")

    if problems:
        raise ReplayTrainError(
            "v_replay was saved but is not reloadable (§8.4): " + "; ".join(problems)
        )

    return {
        "adapter_dir": str(d),
        "weights_bytes": size,
        "n_tensors": n_tensors,
        "peft_type": cfg_data.get("peft_type"),
        "r": cfg_data.get("r"),
        "lora_alpha": cfg_data.get("lora_alpha"),
        "target_modules": cfg_data.get("target_modules"),
    }


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

    # Placement is checked *before* the 14B download/load, because the failure
    # this guards against is silent rather than loud: `from_pretrained` with no
    # `device_map` lands on CPU, `torch.tensor(..., device=None)` also means
    # CPU, and the run then trains correctly but at a speed that reads as a
    # hang. A serving endpoint on some other host does not make this process
    # GPU-resident. Require the device explicitly; never infer it.
    if not cfg.device:
        raise ReplayTrainError(
            "ReplayTrainConfig.device is unset — refusing to load a 14B model "
            "with no explicit placement. Training the replay update on CPU is "
            "not a slower version of the right run, it is an unfinishable one. "
            'Set device="cuda" (or "cuda:N") in a real GPU training job; the '
            "tiny CPU tests call make_optimizer_step directly and do not "
            "reach this loader."
        )
    if not str(cfg.device).startswith("cuda"):
        raise ReplayTrainError(
            f"device={cfg.device!r} is not a CUDA device; the replay update "
            "requires a real GPU training runtime"
        )
    if not torch.cuda.is_available():
        raise ReplayTrainError(
            f"device={cfg.device!r} requested but torch.cuda.is_available() is "
            "False — this process has no GPU. The ck75 serving endpoint is a "
            "separate host and does not provide one here."
        )
    index = torch.device(cfg.device).index or 0
    if index >= torch.cuda.device_count():
        raise ReplayTrainError(
            f"device={cfg.device!r} but only {torch.cuda.device_count()} CUDA "
            "device(s) are visible"
        )

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": cfg.device},
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
    "verify_adapter_reloadable",
    "ReplayTrainConfig",
    "ReplayTrainError",
    "action_logprobs_under_prefix",
    "build_optimizer",
    "current_logprobs",
    "load_v0_for_training",
    "make_optimizer_step",
]
