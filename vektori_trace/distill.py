"""On-policy distillation (OPD) training loop — PLAN.md Step H, OPD branch.

The pieces existed separately and nothing drove them: `opd.reverse_kl_surrogate`
is the objective, `reopd.ReOPDStepExample` is the data, `teacher.VllmTeacherPool`
supplies the scores. This is the optimizer loop that closes them, and the reason
it is not `train.py`: HF's `Trainer` optimises a loss over a *fixed* dataset, and
OPD has no fixed dataset. Every step samples fresh tokens from the current policy
and asks the teacher what it thinks of them. That sample-score-update cycle is
the method, so the loop is written out rather than hidden behind a callback.

One step
--------
1. Encode the frozen teacher prefix with `add_generation_prompt=True` — the
   student must be positioned to *start* an assistant turn.
2. **Student samples** the action (`do_sample=True`). Not greedy: the objective's
   gradient is `Σ (log π_s − log π_t) ∇log π_s` over sampled tokens, and greedy
   decoding would collapse that expectation onto one path.
3. **Teacher scores those exact ids** via `prompt_logprobs` — never re-encoding
   text, which could shift a token boundary and misalign the two logprob vectors.
4. Forward the concatenation *with* grad, take the student's logprob for each
   sampled token, and step on `reverse_kl_surrogate`.

Why the student samples from the training model rather than from vLLM
--------------------------------------------------------------------
vLLM would sample far faster, but the sample must come from the *current* policy,
so every step would need a weight sync into the server. That is the hard part of
GRPO infrastructure and the reason `docs/PILOT.md` defers the RL branch to verl.
OPD does not need it: one forward pass per step is already required for the
gradient, so sampling from the same in-process model costs one extra pass and
removes the sync entirely. The teacher stays on vLLM because it is frozen.

Cost shape: the teacher is queried every step, so teacher latency — not student
FLOPs — sets the step time. That is why the pilot pair uses an MoE teacher.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .dataset import turns_to_messages
from .opd import mean_log_ratio, reverse_kl_surrogate, token_logprobs, topk_reverse_kl
from .reopd import ReOPDStepExample, reopd_loss_mask
from .tokenizer_check import DEFAULT_STUDENT, DEFAULT_TEACHER
from .train import LoraHyperparams, _require_train


class IdScoringPool(Protocol):
    """A teacher that can score ids the caller supplies (`teacher.VllmTeacherPool`)."""

    def score_ids(self, prompt_ids: list[int], tokens: list[int]) -> list[float]: ...


class TopKScoringPool(Protocol):
    """A teacher that also returns its top-K at each scored position."""

    def score_ids_topk(
        self, prompt_ids: list[int], tokens: list[int], top_k: int
    ) -> list[dict[int, float]]: ...


@dataclass
class OPDTrainConfig:
    student_model: str = DEFAULT_STUDENT
    teacher_model: str = DEFAULT_TEACHER
    output_dir: Path = Path("./vektori-out/opd")
    max_steps: int = 200
    learning_rate: float = 1e-5
    # Examples per optimizer step. Each one costs a student sample, a teacher
    # round-trip, and a student forward — so this is the real batch knob.
    examples_per_step: int = 4
    lora: LoraHyperparams = field(default_factory=LoraHyperparams)
    seed: int = 0
    # Sampling. temperature=1.0 keeps the sample on-policy: any other value makes
    # the sampled tokens come from a distribution that is not π_s, and the
    # surrogate's gradient is then weighted by logprobs of the wrong policy.
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 256
    # Left-truncated when longer. The teacher scores the *same* truncated ids, so
    # the two sides stay aligned; what is lost is context, not correctness.
    max_prefix_tokens: int = 3584
    # 0 = the declared objective: `reverse_kl_surrogate` over sampled tokens.
    # >0 = `topk_reverse_kl`, an analytic KL over the teacher's top-K set
    # (thunlp/OPD uses 16). Lower variance, same teacher cost — but a *different*
    # objective, so which one ran belongs in the provenance record, and switching
    # the default is a pre-registration decision, not a tuning knob.
    top_k: int = 0
    bf16: bool = True
    gradient_checkpointing: bool = False
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    # Skip the teacher/student tokenizer identity check. Only for offline tests
    # with a fake pool: sending student ids to a real teacher with a different
    # vocabulary trains on scores for different strings than the student sampled.
    verify_tokenizers: bool = True

    def __post_init__(self) -> None:
        # `do_sample=True` with temperature=0.0 raises inside HF's sampler — after
        # the student weights are loaded and the tokenizer check has run, which is
        # minutes and a GPU allocation into the run. Reject it at construction.
        if self.temperature <= 0.0:
            raise ValueError(
                f"--temperature must be > 0 (got {self.temperature}); the OPD "
                "sample is drawn, not greedy. Use 1.0 to stay on-policy."
            )


@dataclass
class OPDTrainResult:
    adapter_dir: Path
    student_model: str
    teacher_model: str
    steps: int
    final_loss: float | None
    mean_log_ratio_final: float | None
    n_examples: int
    action_tokens_scored: int
    skipped_empty_samples: int
    lora: dict[str, Any]
    seed: int
    log_path: Path | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


def encode_prefix(example: ReOPDStepExample, tokenizer: Any, *, max_prefix_tokens: int) -> list[int]:
    """Frozen prefix as ids, positioned for the student to open an assistant turn.

    `add_generation_prompt=True` is load-bearing: without it the student continues
    the last message instead of starting its own turn, and every sampled token
    would be scored in the wrong role.
    """
    messages = turns_to_messages(example.prefix_turns)
    if not messages:
        raise ValueError(
            f"{example.task} step {example.step_index}: empty prefix — the student "
            "would sample with nothing conditioning it"
        )
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "get") and "input_ids" in ids:
        ids = ids["input_ids"]
    ids = list(ids)
    if len(ids) > max_prefix_tokens:
        # Keep the most recent context; the earliest turns are the least relevant
        # to the action about to be taken.
        ids = ids[-max_prefix_tokens:]
    return ids


def _sample_action(model: Any, tokenizer: Any, prefix_ids: list[int], cfg: OPDTrainConfig) -> list[int]:
    """Sample one action from the *current* policy. No grad — only the ids matter."""
    import torch

    was_training = model.training
    # eval() for the sample so LoRA dropout does not make the sampling policy
    # differ from the policy whose logprobs the gradient uses.
    model.eval()
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=torch.tensor([prefix_ids], device=model.device),
                attention_mask=torch.ones(1, len(prefix_ids), dtype=torch.long, device=model.device),
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_new_tokens=cfg.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        if was_training:
            model.train()
    return [int(t) for t in out[0][len(prefix_ids) :].tolist()]


def _student_action_logits(model: Any, prefix_ids: list[int], action_ids: list[int]) -> Any:
    """Student logits at the action positions, with grad. Shape [1, len(action), V].

    Position arithmetic: `logits[:, i]` predicts token `i+1`, so the distribution
    over action token `j` (absolute position `p+j`) is `logits[:, p+j-1]`. Hence
    the slice `[p-1 : p+a-1]`.
    """
    import torch

    p, a = len(prefix_ids), len(action_ids)
    full = torch.tensor([prefix_ids + action_ids], device=model.device)
    logits = model(input_ids=full, attention_mask=torch.ones_like(full)).logits

    # Cross-check the slice against reopd's documented masking semantics rather
    # than trusting the arithmetic above. `mask[1:]` aligns with `logits[:, :-1]`.
    _, mask = reopd_loss_mask(prefix_ids, action_ids)
    if mask[1:][p - 1 : p + a - 1] != [1] * a:
        raise RuntimeError(
            "OPD slice does not line up with reopd_loss_mask — refusing to train "
            "on a misaligned objective"
        )
    return logits[:, p - 1 : p + a - 1, :]


def align_topk_rows(
    rows: list[dict[int, float]], sampled: list[int], k: int
) -> tuple[list[list[int]], list[list[float]]]:
    """Ragged teacher top-K rows → two rectangular [T][k] lists.

    vLLM returns `k` entries per position, or `k+1` when the sampled token falls
    outside the top-K (it always includes the prompt's own token). A rectangular
    tensor needs one width, and the two ways to get there are not equally safe:

    - Padding a short row with a placeholder id breaks the objective. A padded
      entry with teacher logprob −inf but nonzero student mass sends
      `p_s (log p_s − log p_t)` to +inf.
    - Dropping the *least probable* entry cannot: it removes a token carrying
      almost none of the mass (97–99% sits in the top few, per thunlp/OPD).

    So: keep the sampled token always, then the highest-logprob remainder, to
    exactly `k` per position. Raises if a row is shorter than `k` — that means the
    teacher returned fewer alternatives than asked for and the caller should lower
    `top_k` rather than have this silently reshape the objective.
    """
    ids_out: list[list[int]] = []
    lps_out: list[list[float]] = []
    for i, (row, token_id) in enumerate(zip(rows, sampled, strict=True)):
        if len(row) < k:
            raise RuntimeError(
                f"teacher returned {len(row)} logprobs at position {i}, fewer than "
                f"top_k={k} — lower --top-k rather than reshape the objective"
            )
        ordered = sorted(row.items(), key=lambda kv: kv[1], reverse=True)
        kept = [(token_id, row[token_id])]
        for tid, lp in ordered:
            if len(kept) >= k:
                break
            if tid != token_id:
                kept.append((tid, lp))
        ids_out.append([tid for tid, _ in kept])
        lps_out.append([lp for _, lp in kept])
    return ids_out, lps_out


def run_opd_training(
    examples: list[ReOPDStepExample],
    pool: IdScoringPool,
    cfg: OPDTrainConfig,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> OPDTrainResult:
    """Train the student against teacher scores. Writes an adapter under output_dir.

    `model`/`tokenizer` are injectable so offline tests can run the real loop on a
    tiny from-scratch model with a fake pool — the loop under test is the same one
    that runs on the GPU.
    """
    _require_train()
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    if not examples:
        raise ValueError("no ReOPD examples — nothing to distil from")

    if cfg.verify_tokenizers:
        # Sending student-sampled ids to the teacher is only meaningful if both
        # models read those ids as the same strings. This is the gate for that.
        from .tokenizer_check import check_tokenizers

        check_tokenizers(cfg.teacher_model, cfg.student_model)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = cfg.output_dir / "adapter"
    log_path = cfg.output_dir / "opd_log.jsonl"

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(cfg.student_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = bool(cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.student_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        )
        if torch.cuda.is_available():
            model = model.cuda()
        target = list(cfg.lora.target_modules)
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                target_modules=target,
            ),
        )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # LoRA freezes the base weights, so without this the checkpointed
        # activations have nothing requiring grad and backward is empty.
        model.enable_input_require_grads()

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters — LoRA did not attach to the model")
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(cfg.max_steps * cfg.warmup_ratio), 0),
        num_training_steps=cfg.max_steps,
    )

    order = list(range(len(examples)))
    rng.shuffle(order)
    cursor = 0

    def next_example() -> ReOPDStepExample:
        nonlocal cursor, order
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0
        ex = examples[order[cursor]]
        cursor += 1
        return ex

    model.train()
    final_loss: float | None = None
    final_ratio: float | None = None
    tokens_scored = 0
    skipped = 0
    completed_steps = 0

    with log_path.open("w") as log:
        for step in range(cfg.max_steps):
            optimizer.zero_grad(set_to_none=True)
            step_losses: list[Any] = []
            step_ratios: list[float] = []
            step_tokens = 0

            for _ in range(cfg.examples_per_step):
                ex = next_example()
                prefix_ids = encode_prefix(ex, tokenizer, max_prefix_tokens=cfg.max_prefix_tokens)
                action_ids = _sample_action(model, tokenizer, prefix_ids, cfg)
                if not action_ids:
                    # Immediate EOS: no sampled tokens, so no gradient exists. A
                    # zero loss here would be a real datum saying "teacher and
                    # student agree", which is not what happened.
                    skipped += 1
                    continue

                logits = _student_action_logits(model, prefix_ids, action_ids)
                action_t = torch.tensor([action_ids], device=logits.device)
                # Sampled-token logprobs either way: they are the objective in the
                # top_k=0 case and the monitoring scalar in both.
                student_lp = token_logprobs(logits, action_t)

                if cfg.top_k > 0:
                    rows = pool.score_ids_topk(prefix_ids, action_ids, cfg.top_k)
                    if len(rows) != len(action_ids):
                        raise RuntimeError(
                            f"teacher returned {len(rows)} top-K rows for "
                            f"{len(action_ids)} sampled tokens"
                        )
                    ids, lps = align_topk_rows(rows, action_ids, cfg.top_k)
                    loss = topk_reverse_kl(
                        logits,
                        torch.tensor([ids], device=logits.device),
                        torch.tensor([lps], dtype=torch.float32, device=logits.device),
                    )
                    # The monitoring ratio still uses the sampled token, so it
                    # stays comparable across both objectives.
                    teacher_lp = [row[tid] for row, tid in zip(rows, action_ids, strict=True)]
                else:
                    teacher_lp = pool.score_ids(prefix_ids, action_ids)
                    if len(teacher_lp) != len(action_ids):
                        raise RuntimeError(
                            f"teacher returned {len(teacher_lp)} logprobs for "
                            f"{len(action_ids)} sampled tokens"
                        )
                    teacher_t = torch.tensor(
                        [teacher_lp], dtype=student_lp.dtype, device=student_lp.device
                    )
                    loss = reverse_kl_surrogate(student_lp, teacher_t)

                # Scale before backward so accumulated grads average over the
                # examples that actually produced one (skips excluded).
                (loss / cfg.examples_per_step).backward()

                step_losses.append(float(loss.detach()))
                step_ratios.append(
                    mean_log_ratio(
                        [float(x) for x in student_lp.detach()[0].tolist()], teacher_lp
                    )
                )
                step_tokens += len(action_ids)

            if not step_losses:
                # Every example in this step sampled nothing — no gradient to
                # apply. Record it; a run where this is common is a broken
                # sampling config, not a converged model.
                log.write(json.dumps({"step": step, "skipped_step": True}) + "\n")
                continue

            # Before the step, not after: a NaN loss caught downstream has already
            # flowed through the clip and into the LoRA weights. Nothing corrupt is
            # persisted either way (the run aborts before `save_pretrained`), but
            # failing here keeps the weights in memory consistent with the log.
            mean_loss = sum(step_losses) / len(step_losses)
            if mean_loss != mean_loss:  # NaN
                raise RuntimeError(
                    "OPD loss is NaN — check teacher/student tokenizer identity and "
                    "that the sampled ids are the ids the teacher scored"
                )
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            # Read the LR *before* advancing: after `scheduler.step()` this
            # reports the next step's rate, so a log line would attribute the
            # wrong LR to the update that just happened (and the last line of a
            # cosine schedule would always read 0).
            step_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            completed_steps += 1

            final_loss = mean_loss
            final_ratio = sum(step_ratios) / len(step_ratios)
            tokens_scored += step_tokens
            log.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": final_loss,
                        # Monitoring only, carries no gradient: the sample estimate
                        # of the reverse-KL coefficient. It should trend toward 0
                        # as the student's distribution approaches the teacher's.
                        "mean_log_ratio": final_ratio,
                        "action_tokens": step_tokens,
                        "lr": step_lr,
                        "n_examples": len(step_losses),
                    }
                )
                + "\n"
            )
            log.flush()

    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # State the objective that actually ran, not the branch name: the two paths
    # optimise different quantities and a report must not conflate them.
    provenance = {
        "branch": "OPD",
        "loss": "topk_reverse_kl" if cfg.top_k > 0 else "reverse_kl_surrogate",
        "top_k": cfg.top_k,
        "temperature": cfg.temperature,
    }
    if hasattr(pool, "provenance"):
        provenance.update(pool.provenance())

    return OPDTrainResult(
        adapter_dir=adapter_dir,
        student_model=cfg.student_model,
        teacher_model=cfg.teacher_model,
        steps=completed_steps,
        final_loss=final_loss,
        mean_log_ratio_final=final_ratio,
        n_examples=len(examples),
        action_tokens_scored=tokens_scored,
        skipped_empty_samples=skipped,
        lora=asdict(cfg.lora),
        seed=cfg.seed,
        log_path=log_path,
        provenance=provenance,
    )


def write_opd_report(result: OPDTrainResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "adapter_dir": str(result.adapter_dir),
        "student_model": result.student_model,
        "teacher_model": result.teacher_model,
        "steps": result.steps,
        "final_loss": result.final_loss,
        "mean_log_ratio_final": result.mean_log_ratio_final,
        "n_examples": result.n_examples,
        "action_tokens_scored": result.action_tokens_scored,
        "skipped_empty_samples": result.skipped_empty_samples,
        "lora": result.lora,
        "seed": result.seed,
        "log": str(result.log_path) if result.log_path else None,
        "provenance": result.provenance,
    }
    (out_dir / "opd.json").write_text(json.dumps(payload, indent=2) + "\n")
    ratio = result.mean_log_ratio_final
    lines = [
        "# Vektori-trace OPD (on-policy distillation)\n",
        f"- student: `{result.student_model}`\n",
        f"- teacher: `{result.teacher_model}`\n",
        f"- adapter: `{result.adapter_dir}`\n",
        f"- steps: {result.steps}  ·  final loss: "
        f"{result.final_loss if result.final_loss is not None else 'n/a'}\n",
        f"- mean log ratio (monitoring): {ratio if ratio is not None else 'n/a'}\n",
        f"- action tokens scored: {result.action_tokens_scored}"
        f"  ·  empty samples skipped: {result.skipped_empty_samples}\n",
        f"- seed: {result.seed}  ·  LoRA r={result.lora.get('r')}\n",
    ]
    md_path = out_dir / "opd.md"
    md_path.write_text("".join(lines))
    return md_path


__all__ = [
    "IdScoringPool",
    "OPDTrainConfig",
    "OPDTrainResult",
    "TopKScoringPool",
    "align_topk_rows",
    "encode_prefix",
    "run_opd_training",
    "write_opd_report",
]
