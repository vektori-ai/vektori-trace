"""OPD with the student on Fireworks' Training API and the teacher anywhere.

`distill.run_opd_training` is the same loop with the student in-process: sample
from the current policy, ask the teacher what it thinks of the sampled tokens,
take a reverse-KL step. This module keeps the loop and the objective identical
and swaps only where the student's forward/backward runs — Fireworks GPUs
instead of the local one — because that is the part the pilot has no hardware
for. The loss is `opd.reverse_kl_surrogate`, imported, not reimplemented.

How the Training API inverts the usual arrangement
--------------------------------------------------
`forward_backward_custom(datums, loss_fn)` runs the forward pass on remote GPUs,
ships **per-token logprobs** back with `requires_grad=True`, calls `loss_fn`
locally, then ships `d_loss/d_logprob` back for the model backward. So the
student logprobs OPD needs are exactly what the API hands you, and the objective
stays ordinary Python.

Two consequences worth stating plainly:

- **`top_k > 0` is not available here.** `topk_reverse_kl` needs student *logits*
  over the teacher's top-K token set; `forward_backward_custom` returns logprobs
  for the datum's own tokens and nothing else. The sampled-token objective
  (`reverse_kl_surrogate`) is the one this module runs, and it raises rather than
  silently substituting it when a config asks for top-K. Fireworks' own recipe
  makes the same split: their `topk_forward_kl` mode is forward KL via multi-target
  `cross_entropy`, a different objective, not reverse KL over K.
- **On-policy costs a weight sync.** The student's weights live on the trainer;
  sampling happens on a deployment. Between them sits a checkpoint
  (`save_weights_for_sampler` → `create_deployment_sampler(model_path=...)`),
  so "on-policy" is only true up to `sync_every` steps. `distill.py` has no such
  gap — the in-process model samples from the weights it just updated. Setting
  `sync_every=1` closes it at the cost of a checkpoint per step; the config
  records whatever was used, because a run that sampled from 8-step-stale weights
  is off-policy by 8 steps and its report should say so.

Everything Fireworks-specific is imported lazily, so the module imports (and its
alignment logic tests) without the training extras installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...dataset import turns_to_messages
from ...distill import IdScoringPool, OPDTrainResult
from ...opd import mean_log_ratio, reverse_kl_surrogate
from ...reopd import ReOPDStepExample
from ...tokenizer_check import DEFAULT_STUDENT, DEFAULT_TEACHER

#: Student base model and its trainer shape. `qwen3-8b` is the pilot's student
#: (`docs/PILOT.md`) and Fireworks' own quickstart base, so the shape is known good.
DEFAULT_STUDENT_MODEL = "accounts/fireworks/models/qwen3-8b"
DEFAULT_TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3-8b-128k-h200"


@dataclass
class FireworksOPDConfig:
    """Knobs for the remote-student OPD loop.

    Deliberately mirrors `distill.OPDTrainConfig` field-for-field where the two
    loops share a meaning, so a run's report can be compared across backends
    without a translation table.
    """

    # -- what to train -------------------------------------------------------
    base_model: str = DEFAULT_STUDENT_MODEL
    training_shape_id: str = DEFAULT_TRAINING_SHAPE
    #: HF path for the tokenizer. Must be the tokenizer of `base_model`, and must
    #: pass `check_tokenizers` against the teacher — the constraint is unchanged
    #: by hosting (`docs/OPD.md`).
    tokenizer_model: str = DEFAULT_STUDENT
    teacher_model: str = DEFAULT_TEACHER
    #: 0 = full-parameter. >0 attaches LoRA of that rank, the cheap arm.
    lora_rank: int = 16

    # -- the loop ------------------------------------------------------------
    output_dir: Path = Path("./vektori-out/opd-fireworks")
    max_steps: int = 200
    learning_rate: float = 1e-5
    examples_per_step: int = 4
    seed: int = 0
    #: 1.0 keeps the sample on-policy; any other value makes the sampled tokens
    #: come from a distribution that is not π_s and the surrogate's gradient is
    #: then weighted by logprobs of the wrong policy.
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 256
    max_seq_len: int = 8192
    #: Steps between trainer→deployment weight syncs. See the module docstring:
    #: this is the size of the off-policy gap, not a performance knob.
    sync_every: int = 1
    #: Present so a config written for `distill.py` fails loudly here instead of
    #: quietly optimising a different objective. See the module docstring.
    top_k: int = 0
    verify_tokenizers: bool = True

    # -- optimizer -----------------------------------------------------------
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.01

    # -- teardown ------------------------------------------------------------
    #: Promote the final checkpoint to a deployable model id, or None to leave it
    #: as a checkpoint. Lowercase a-z, 0-9, hyphen; a rejected id orphans the blob.
    output_model_id: str | None = None
    cleanup_trainer_on_close: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


def build_opd_datum(
    tinker: Any,
    full_tokens: list[int],
    prompt_len: int,
    teacher_logprobs: list[float],
) -> Any:
    """One `tinker.Datum` carrying the sampled sequence and the teacher's scores.

    Three arrays have to agree index-for-index or the objective is silently wrong,
    which is the failure mode `docs/OPD.md` cares about most:

    - `weights` is 0 on prefix tokens and 1 on sampled tokens, matching the API's
      documented convention (0 = prompt/no-loss, 1 = response/learned).
    - `teacher_logprobs` covers only the sampled tokens, so it is zero-padded
      across the prefix. Those entries are masked out and never reach the loss;
      the padding is there so one index means one thing across all three arrays.
    - The teacher scored exactly `full_tokens[prompt_len:]`, which is checked by
      length here rather than assumed by the caller.

    The teacher's scores ride in `loss_fn_inputs` rather than a closure because
    `forward_backward_custom` hands the loss function its datums back — this way
    the numbers travel with the sequence they belong to and a reordered batch
    cannot mismatch them.
    """
    import torch

    n = len(full_tokens)
    action_len = n - prompt_len
    if action_len <= 0:
        raise ValueError(f"no sampled tokens: prompt_len={prompt_len}, total={n}")
    if len(teacher_logprobs) != action_len:
        raise ValueError(
            f"teacher returned {len(teacher_logprobs)} logprobs for {action_len} "
            "sampled tokens — refusing to build a misaligned datum"
        )

    weights = torch.zeros(n, dtype=torch.float32)
    weights[prompt_len:] = 1.0
    teacher = torch.zeros(n, dtype=torch.float32)
    teacher[prompt_len:] = torch.tensor(teacher_logprobs, dtype=torch.float32)

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(full_tokens),
        loss_fn_inputs={
            "weights": tinker.TensorData(
                data=weights.tolist(), dtype="float32", shape=[n]
            ),
            "teacher_logprobs": tinker.TensorData(
                data=teacher.tolist(), dtype="float32", shape=[n]
            ),
        },
    )


def opd_loss_fn(data: list[Any], logprobs_list: list[Any]) -> tuple[Any, dict[str, float]]:
    """`opd.reverse_kl_surrogate` in the shape `forward_backward_custom` expects.

    Signature is fixed by the API: datums in, per-token student logprobs in (each
    with `requires_grad=True`), scalar loss and a metrics dict out. The gradient
    that leaves here is `(log π_s − log π_t) ∇log π_s` summed over sampled tokens,
    which is the OPD gradient `docs/OPD.md` states — unchanged by the fact that
    `∇log π_s` is completed on a remote GPU.

    Truncation to `min_len` follows the documented pattern: the forward pass can
    return fewer logprobs than the datum has tokens, and mismatched lengths break
    the objective silently rather than loudly.
    """
    import torch

    total = torch.tensor(0.0)
    n_tokens = 0.0
    ratios: list[float] = []

    for datum, student_lp in zip(data, logprobs_list, strict=True):
        weights = torch.tensor(
            datum.loss_fn_inputs["weights"].data, dtype=torch.float32
        )
        teacher = torch.tensor(
            datum.loss_fn_inputs["teacher_logprobs"].data, dtype=torch.float32
        )
        min_len = min(len(student_lp), len(weights), len(teacher))
        s = student_lp[:min_len].float()
        t = teacher[:min_len]
        mask = weights[:min_len]

        # Sum, not mean: `reverse_kl_surrogate` already normalises by the number
        # of supervised tokens, so weighting each sequence by its own token count
        # here would double-normalise and let long samples dominate.
        per_seq = reverse_kl_surrogate(s, t, mask)
        total = total + per_seq
        n_tokens += float(mask.sum())

        supervised = mask > 0
        if bool(supervised.any()):
            ratios.append(
                mean_log_ratio(
                    s.detach()[supervised].tolist(), t[supervised].tolist()
                )
            )

    loss = total / max(len(data), 1)
    metrics = {
        "opd_loss": float(loss.item()),
        "action_tokens": n_tokens,
        # Monitoring only, no gradient: should trend toward 0 as the student's
        # distribution approaches the teacher's.
        "mean_log_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
    }
    return loss, metrics


def run_fireworks_opd(
    examples: list[ReOPDStepExample],
    pool: IdScoringPool,
    cfg: FireworksOPDConfig,
) -> OPDTrainResult:
    """The loop: sample on the deployment, score with the teacher, step on the trainer.

    Returns the same `OPDTrainResult` as `distill.run_opd_training` so
    `distill.write_opd_report` works unchanged and the two backends produce
    comparable reports. `adapter_dir` holds the run log and the checkpoint
    identity rather than local weights — the weights live on Fireworks.
    """
    import asyncio
    import random

    if not examples:
        raise ValueError("no ReOPD examples — nothing to distil from")
    if cfg.top_k > 0:
        raise ValueError(
            "top_k>0 selects `topk_reverse_kl`, which needs student logits over the "
            "teacher's top-K set. forward_backward_custom returns logprobs for the "
            "datum's own tokens only, so that objective cannot run on the Training "
            "API. Use top_k=0 here, or run the top-K arm against a local student "
            "(`distill.run_opd_training`)."
        )

    import tinker
    import transformers
    from fireworks.training.sdk import (
        AdaptiveConcurrencyController,
        FiretitanServiceClient,
    )

    if cfg.verify_tokenizers:
        # Sending student-sampled ids to the teacher is only meaningful if both
        # models read those ids as the same strings. Unchanged by hosting.
        from ...tokenizer_check import check_tokenizers

        check_tokenizers(cfg.teacher_model, cfg.tokenizer_model)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.output_dir / "opd_log.jsonl"
    rng = random.Random(cfg.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.tokenizer_model, trust_remote_code=True
    )

    service = FiretitanServiceClient.from_firetitan_config(
        base_model=cfg.base_model,
        tokenizer_model=cfg.tokenizer_model,
        training_shape_id=cfg.training_shape_id,
        lora_rank=cfg.lora_rank,
        learning_rate=cfg.learning_rate,
        # The deployment is what the student samples from, so unlike the SFT
        # quickstart this loop cannot run without one.
        create_deployment=True,
        cleanup_trainer_on_close=cfg.cleanup_trainer_on_close,
    )

    order = list(range(len(examples)))
    rng.shuffle(order)
    cursor = 0

    def next_example() -> ReOPDStepExample:
        nonlocal cursor
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0
        ex = examples[order[cursor]]
        cursor += 1
        return ex

    final_loss: float | None = None
    final_ratio: float | None = None
    tokens_scored = 0
    skipped = 0
    completed_steps = 0
    checkpoint_path: str | None = None

    try:
        training_client = service.create_training_client(
            base_model=cfg.base_model, lora_rank=cfg.lora_rank
        )
        sampler = service.create_deployment_sampler(
            tokenizer=tokenizer,
            concurrency_controller=AdaptiveConcurrencyController(initial_window=16),
        )

        with log_path.open("w") as log:
            for step in range(cfg.max_steps):
                batch: list[Any] = []
                step_ratios: list[float] = []
                step_tokens = 0

                for _ in range(cfg.examples_per_step):
                    ex = next_example()
                    messages = turns_to_messages(ex.prefix_turns)
                    if not messages:
                        raise ValueError(
                            f"{ex.task} step {ex.step_index}: empty prefix — the "
                            "student would sample with nothing conditioning it"
                        )

                    # Sampling and tokenisation happen in one place, so the ids the
                    # teacher scores are the ids the student actually emitted —
                    # the same guarantee `teacher.py` gets from server-side
                    # /tokenize, obtained differently.
                    completions = asyncio.run(
                        sampler.sample_with_tokens(
                            messages=messages,
                            n=1,
                            max_tokens=cfg.max_new_tokens,
                            temperature=cfg.temperature,
                            top_p=cfg.top_p,
                            max_seq_len=cfg.max_seq_len,
                        )
                    )
                    if not completions:
                        # Dropped by the length filter, prompt or completion.
                        skipped += 1
                        continue
                    c = completions[0]
                    full_tokens = [int(t) for t in c.full_tokens]
                    prompt_len = int(c.prompt_len)
                    action_ids = full_tokens[prompt_len:]
                    if not action_ids:
                        # Immediate EOS: no sampled tokens, so no gradient exists.
                        # A zero loss here would be a real datum claiming teacher
                        # and student agree, which is not what happened.
                        skipped += 1
                        continue

                    teacher_lp = pool.score_ids(full_tokens[:prompt_len], action_ids)
                    if len(teacher_lp) != len(action_ids):
                        raise RuntimeError(
                            f"teacher returned {len(teacher_lp)} logprobs for "
                            f"{len(action_ids)} sampled tokens"
                        )
                    batch.append(
                        build_opd_datum(tinker, full_tokens, prompt_len, teacher_lp)
                    )
                    step_tokens += len(action_ids)

                if not batch:
                    # Every example in this step sampled nothing. Record it; a run
                    # where this is common has a broken sampling config, not a
                    # converged model.
                    log.write(json.dumps({"step": step, "skipped_step": True}) + "\n")
                    continue

                result = training_client.forward_backward_custom(
                    batch, opd_loss_fn
                ).result()
                training_client.optim_step(
                    tinker.AdamParams(
                        learning_rate=cfg.learning_rate,
                        beta1=cfg.adam_beta1,
                        beta2=cfg.adam_beta2,
                        eps=cfg.adam_eps,
                        weight_decay=cfg.weight_decay,
                    )
                ).result()
                completed_steps += 1
                tokens_scored += step_tokens

                metrics = dict(result.metrics or {})
                final_loss = _as_float(metrics.get("opd_loss"))
                final_ratio = _as_float(metrics.get("mean_log_ratio"))
                if final_loss is not None and final_loss != final_loss:  # NaN
                    raise RuntimeError(
                        "OPD loss is NaN — check teacher/student tokenizer identity "
                        "and that the sampled ids are the ids the teacher scored"
                    )
                step_ratios.append(final_ratio or 0.0)

                # Push the weights just trained onto the deployment, so the next
                # step samples from the current policy rather than a stale one.
                if cfg.sync_every > 0 and (step + 1) % cfg.sync_every == 0:
                    saved = training_client.save_weights_for_sampler(
                        f"opd-step-{step + 1:05d}",
                        # LoRA snapshots are always the full adapter; on
                        # full-parameter runs only "base" is promotable, and the
                        # final checkpoint has to be, so never "delta" here.
                        checkpoint_type="base",
                    ).result()
                    checkpoint_path = saved.path
                    sampler = service.create_deployment_sampler(
                        model_path=saved.path,
                        tokenizer=tokenizer,
                        concurrency_controller=AdaptiveConcurrencyController(
                            initial_window=16
                        ),
                    )

                log.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": final_loss,
                            "mean_log_ratio": final_ratio,
                            "action_tokens": step_tokens,
                            "n_examples": len(batch),
                            "synced": checkpoint_path,
                        }
                    )
                    + "\n"
                )
                log.flush()

        if checkpoint_path is None and completed_steps:
            saved = training_client.save_weights_for_sampler(
                "opd-final", checkpoint_type="base"
            ).result()
            checkpoint_path = saved.path
        if cfg.output_model_id and checkpoint_path:
            _promote_final(service, cfg)
    finally:
        # The trainer bills while it exists. Teardown belongs here, not in the
        # happy path — the same reasoning `docs/AWS.md` gives for stop-instances.
        service.close()

    provenance = {
        "branch": "OPD",
        "loss": "reverse_kl_surrogate",
        "top_k": 0,
        "temperature": cfg.temperature,
        "student_host": "fireworks-training-api",
        "training_shape_id": cfg.training_shape_id,
        "lora_rank": cfg.lora_rank,
        # The size of the off-policy gap, in steps. 1 = each step sampled from the
        # weights the previous step produced.
        "sync_every": cfg.sync_every,
        "checkpoint": checkpoint_path,
        **cfg.meta,
    }
    if hasattr(pool, "provenance"):
        provenance.update(pool.provenance())

    return OPDTrainResult(
        adapter_dir=cfg.output_dir,
        student_model=cfg.base_model,
        teacher_model=cfg.teacher_model,
        steps=completed_steps,
        final_loss=final_loss,
        mean_log_ratio_final=final_ratio,
        n_examples=len(examples),
        action_tokens_scored=tokens_scored,
        skipped_empty_samples=skipped,
        lora={"rank": cfg.lora_rank},
        seed=cfg.seed,
        log_path=log_path,
        provenance=provenance,
    )


def _promote_final(service: Any, cfg: FireworksOPDConfig) -> None:
    """Turn the newest promotable checkpoint into a deployable model.

    Selects by parsed `createTime` rather than list order, and passes the
    4-segment `name` straight through — the positional `(job_id, checkpoint_id)`
    form is deprecated, and `saved.path` is a snapshot identity that can differ
    from the checkpoint resource id.
    """
    from datetime import datetime

    rows = [r for r in service.list_checkpoints(service.trainer_job_id) if r.get("promotable")]
    if not rows:
        raise RuntimeError(
            "no promotable checkpoint — full-parameter 'delta' saves cannot be "
            "promoted; the run should have saved 'base'"
        )
    target = max(
        rows,
        key=lambda row: datetime.fromisoformat(row["createTime"].replace("Z", "+00:00")),
    )
    service.promote_checkpoint(
        name=target["name"],
        output_model_id=cfg.output_model_id,
        base_model=cfg.base_model,
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "DEFAULT_STUDENT_MODEL",
    "DEFAULT_TRAINING_SHAPE",
    "FireworksOPDConfig",
    "build_opd_datum",
    "opd_loss_fn",
    "run_fireworks_opd",
]


# ── Cross-tokenizer path (docs/OPD-MULTITURN-PLAN.md §6.5) ────────────────────
#
# `build_opd_datum` / `opd_loss_fn` above require one teacher logprob per student
# token. That holds only when teacher and student share a tokenizer, which
# `check_tokenizers` enforces on that path. DeepSeek-V4 does not share Qwen3's
# tokenizer, so the datum below carries what survives alignment instead:
#
#   - `advantages`  — detached per-*student*-token A_i from
#                     `chunk_opd.assign_chunk_advantages`, already computed
#                     against the teacher's own tokenization. The teacher's token
#                     count never appears here; it was consumed during alignment.
#   - `behavior_logprobs` — log pi_old(s_i), for the importance ratio. The
#                     same-tokenizer path has no equivalent because
#                     `reverse_kl_surrogate` is not an importance-sampled
#                     objective.
#
# Both are per-student-token, so the three-array index correspondence that
# `build_opd_datum` protects is preserved — only the *source* of the numbers
# changes.


def build_cross_opd_datum(
    tinker: Any,
    full_tokens: list[int],
    prompt_len: int,
    advantages: list[float],
    behavior_logprobs: list[float],
    supervised_mask: list[bool] | None = None,
) -> Any:
    """One `tinker.Datum` for the cross-tokenizer chunk objective.

    `advantages` and `behavior_logprobs` cover only the sampled action and are
    zero-padded across the prefix, exactly as `build_opd_datum` pads the teacher
    scores — one index means one thing across every array.

    `supervised_mask` lets sentinel positions (unalignable tails, over-long
    chunks) be dropped from both the numerator and the denominator. Without it
    every action token is supervised. A sentinel's advantage is already 0.0, so
    it contributes no gradient either way; the mask is what keeps it out of the
    *denominator*, so a partly-unaligned action is not silently rescaled.
    """
    import torch

    n = len(full_tokens)
    action_len = n - prompt_len
    if action_len <= 0:
        raise ValueError(f"no sampled tokens: prompt_len={prompt_len}, total={n}")
    if len(advantages) != action_len:
        raise ValueError(
            f"{len(advantages)} advantages for {action_len} sampled tokens — "
            "advantages are per-student-token after chunk alignment; a mismatch "
            "means the alignment covered a different span than was sampled"
        )
    if len(behavior_logprobs) != action_len:
        raise ValueError(
            f"{len(behavior_logprobs)} behavior logprobs for {action_len} sampled "
            "tokens — refusing to build a misaligned datum"
        )
    if supervised_mask is not None and len(supervised_mask) != action_len:
        raise ValueError(
            f"{len(supervised_mask)} mask entries for {action_len} sampled tokens"
        )

    weights = torch.zeros(n, dtype=torch.float32)
    if supervised_mask is None:
        weights[prompt_len:] = 1.0
    else:
        weights[prompt_len:] = torch.tensor(
            [1.0 if s else 0.0 for s in supervised_mask], dtype=torch.float32
        )
    adv = torch.zeros(n, dtype=torch.float32)
    adv[prompt_len:] = torch.tensor(advantages, dtype=torch.float32)
    behavior = torch.zeros(n, dtype=torch.float32)
    behavior[prompt_len:] = torch.tensor(behavior_logprobs, dtype=torch.float32)

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(full_tokens),
        loss_fn_inputs={
            "weights": tinker.TensorData(
                data=weights.tolist(), dtype="float32", shape=[n]
            ),
            "advantages": tinker.TensorData(
                data=adv.tolist(), dtype="float32", shape=[n]
            ),
            "behavior_logprobs": tinker.TensorData(
                data=behavior.tolist(), dtype="float32", shape=[n]
            ),
        },
    )


def cross_opd_loss_fn(
    data: list[Any], logprobs_list: list[Any]
) -> tuple[Any, dict[str, float]]:
    """`chunk_opd.clipped_is_policy_loss` in `forward_backward_custom`'s shape.

    Differs from `opd_loss_fn` in two ways that matter:

    - **No `min_len` truncation.** The plan (§6.5) makes a length mismatch a hard
      error: a forward pass returning fewer logprobs than the datum has tokens
      means the remote tokenisation disagrees with the one alignment was computed
      against, and silently dropping the suffix would train on advantages
      belonging to other positions.
    - **One global denominator.** Each example contributes its raw sum
      (`denominator=1.0`); the total is divided once by the batch-wide supervised
      token count. `opd_loss_fn` divides by `len(data)`, which weights a short
      action the same as a long one — acceptable for a mean-of-means objective,
      wrong for the plan's "summed and divided once by the global count of
      supervised ck75 tokens".
    """
    import torch

    from ...chunk_opd import clip_fraction, clipped_is_policy_loss

    total = torch.tensor(0.0)
    n_tokens = 0.0
    clip_fracs: list[float] = []
    adv_sum = 0.0

    for datum, student_lp in zip(data, logprobs_list, strict=True):
        weights = torch.tensor(
            datum.loss_fn_inputs["weights"].data, dtype=torch.float32
        )
        adv = torch.tensor(
            datum.loss_fn_inputs["advantages"].data, dtype=torch.float32
        )
        behavior = torch.tensor(
            datum.loss_fn_inputs["behavior_logprobs"].data, dtype=torch.float32
        )

        if len(student_lp) != len(weights):
            raise RuntimeError(
                f"forward returned {len(student_lp)} logprobs for a datum with "
                f"{len(weights)} tokens. Not truncating: the advantages were "
                "aligned against the full sequence and would land on the wrong "
                "positions (docs/OPD-MULTITURN-PLAN.md §6.5)"
            )

        s = student_lp.float()
        # Raw sum here; the global denominator is applied once below.
        total = total + clipped_is_policy_loss(
            s, behavior, adv, weights, denominator=1.0
        )
        n_tokens += float(weights.sum())

        supervised = weights > 0
        if bool(supervised.any()):
            clip_fracs.append(clip_fraction(s, behavior, weights))
            adv_sum += float(adv[supervised].sum())

    loss = total / max(n_tokens, 1.0)
    metrics = {
        "opd_loss": float(loss.detach().item()),
        "action_tokens": n_tokens,
        "clip_fraction": sum(clip_fracs) / len(clip_fracs) if clip_fracs else 0.0,
        # Monitoring only: mean advantage should sit near 0 when the student
        # already matches the teacher, and its sign says which way the update
        # pushes.
        "mean_advantage": adv_sum / n_tokens if n_tokens else 0.0,
    }
    return loss, metrics


__all__ += ["build_cross_opd_datum", "cross_opd_loss_fn"]


def validate_cross_opd_config(
    cfg: FireworksOPDConfig,
    *,
    loss_id: str = "chunk_opd",
) -> None:
    """Gate a cross-tokenizer run's config before anything is spent.

    Three plan requirements that are each invisible at runtime if violated:

    - **§7.1 token cap.** `FireworksOPDConfig.max_new_tokens` still defaults to
      256, the cap the previous run used. A truncated action still aligns and
      still produces a finite loss, so nothing downstream reveals it.
    - **§6.5 loss selection.** `cross_kl`'s span surrogate yields a plausible
      number from the same inputs; only an explicit check separates it from the
      published objective.
    - **`top_k`.** Unused on this path — the chunk objective needs the realized
      path's log probabilities, not a top-K set. A config carrying it was
      written for a different objective.

    Call this from the driver, not from the loss: by the time a loss function
    runs, the rollout that used the wrong cap has already happened.
    """
    from ...chunk_opd import assert_chunk_loss_selected, assert_token_cap_is_task_derived

    assert_chunk_loss_selected(loss_id, cross_tokenizer=True)
    assert_token_cap_is_task_derived(cfg.max_new_tokens)
    if cfg.top_k:
        raise ValueError(
            f"top_k={cfg.top_k} has no meaning for the chunk objective, which "
            "scores the realized path rather than a top-K set. A config setting "
            "it was written for a different loss (docs/OPD-MULTITURN-PLAN.md §2)."
        )


__all__ += ["validate_cross_opd_config"]
