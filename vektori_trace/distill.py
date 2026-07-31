"""On-policy distillation (OPD) training loop.

Same-vocab path: student and teacher share a tokenizer. Student samples an
action, teacher scores those exact ids via `prompt_logprobs`. Gradient from
`reverse_kl_surrogate`.

Cross-tokenizer path (FINAL-PLAN.md): student is Qwen3-8B, teacher is
DeepSeek-V4-Flash. Tokenisations are aligned by bytes (§7 two-pointer merge),
the aligned token streams are supervised with `cross_step_loss` (§6 estimators
A and B). Same reverse-KL objective, different score routing.

One step (same-vocab)
---------------------
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
GRPO infrastructure and the reason the RL branch defers to verl.
OPD does not need it: one forward pass per step is already required for the
gradient, so sampling from the same in-process model costs one extra pass and
removes the sync entirely. The teacher stays on vLLM because it is frozen.

Cost shape: the teacher is queried every step, so teacher latency — not student
FLOPs — sets the step time.
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
    # ── Cross-tokenizer OPD (FINAL-PLAN.md) ──────────────────────────────────
    # When True, student and teacher have different vocabularies; alignment is
    # done by bytes (align_by_bytes) and the loss is cross_step_loss.
    cross_tokenizer: bool = False
    # Path to a CrossTokenizerBridge JSON artifact. May be None when bridge= is
    # injected directly into run_opd_training (e.g. offline tests).
    bridge_path: Path | str | None = None
    # "chat" or "thinking" — must match the teacher deployment's inference setting.
    thinking_mode: str = "chat"
    # Hard-fail if alignment granularity (spans / student tokens) falls below this.
    min_alignment_granularity: float = 0.5
    # Hard-fail if any single span covers more than this many student tokens.
    max_span_student_tokens: int = 8
    # 0 = no top-K; >0 = also call score_ids_topk at this K for Estimator A.
    # Fireworks caps at 5 (FINAL-PLAN.md §2).
    cross_top_k: int = 5
    # Optional teacher tokenizer — injectable for offline tests so no HF download
    # is needed. When None and cross_tokenizer=True, loaded from teacher_model.
    teacher_tokenizer: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # `do_sample=True` with temperature=0.0 raises inside HF's sampler — after
        # the student weights are loaded and the tokenizer check has run, which is
        # minutes and a GPU allocation into the run. Reject it at construction.
        if self.temperature <= 0.0:
            raise ValueError(
                f"--temperature must be > 0 (got {self.temperature}); the OPD "
                "sample is drawn, not greedy. Use 1.0 to stay on-policy."
            )
        if self.cross_tokenizer and self.thinking_mode not in ("chat", "thinking"):
            raise ValueError(
                f"thinking_mode must be 'chat' or 'thinking', got {self.thinking_mode!r}"
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


def encode_prefix_pair(
    example: ReOPDStepExample,
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    *,
    max_prefix_tokens: int,
    thinking_mode: str = "chat",
    prefix_cache: Any | None = None,
) -> tuple[list[int], list[int], str]:
    """Encode the prefix for both student and teacher tokenizers.

    Student encoding: same as `encode_prefix` (apply_chat_template +
    add_generation_prompt=True).

    Teacher encoding: turns_to_openai_messages → render_teacher_prefix →
    encode_teacher_ids, which produces teacher-side ids suitable for
    `pool.score_ids`.

    Truncation is at a **shared message boundary**: earliest turns are dropped
    one at a time until BOTH encoded sequences fit within `max_prefix_tokens`.
    This keeps both sides conditioning on the same conversational context.

    When `prefix_cache` is provided it is used to skip re-rendering the teacher
    prefix on repeated encounters of the same (task, step_index).

    Returns ``(student_ids, teacher_ids, teacher_prefix_text)``.  The rendered
    teacher prefix text is required for the §10.3 junction assert.
    """
    from .teacher_cross import (
        TeacherPrefixCache,
        encode_teacher_ids,
        render_teacher_prefix,
        turns_to_openai_messages,
    )

    prefix_turns = list(example.prefix_turns)

    for _ in range(len(prefix_turns) + 1):
        if not prefix_turns:
            break

        # ── Student encoding ──────────────────────────────────────────────
        messages = turns_to_messages(prefix_turns)
        if not messages:
            break
        student_ids = student_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(student_ids, "get") and "input_ids" in student_ids:
            student_ids = student_ids["input_ids"]
        student_ids = list(student_ids)

        # ── Teacher encoding (with optional cache) ────────────────────────
        teacher_messages = turns_to_openai_messages(prefix_turns)
        teacher_text = render_teacher_prefix(teacher_messages, thinking_mode=thinking_mode)

        cached_teacher: list[int] | None = None
        if isinstance(prefix_cache, TeacherPrefixCache):
            cached_teacher = prefix_cache.get(
                example.task or "", example.step_index, thinking_mode=thinking_mode
            )

        if cached_teacher is not None:
            teacher_ids = cached_teacher
        else:
            teacher_ids = encode_teacher_ids(teacher_text, teacher_tokenizer)

        if len(student_ids) <= max_prefix_tokens and len(teacher_ids) <= max_prefix_tokens:
            # Cache only the fitted prefix — putting before the fit check would
            # store a too-long encoding and then conflict after truncation.
            if (
                cached_teacher is None
                and isinstance(prefix_cache, TeacherPrefixCache)
            ):
                prefix_cache.put(
                    example.task or "", example.step_index, teacher_ids,
                    thinking_mode=thinking_mode,
                )
            return student_ids, teacher_ids, teacher_text

        # Drop the earliest turn and retry.
        prefix_turns = prefix_turns[1:]

    raise ValueError(
        f"{example.task} step {example.step_index}: empty prefix after cross-tokenizer "
        "truncation — no turns fit within max_prefix_tokens on both sides"
    )


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
    bridge: Any | None = None,
    teacher_tokenizer: Any | None = None,
    prefix_cache: Any | None = None,
) -> OPDTrainResult:
    """Train the student against teacher scores. Writes an adapter under output_dir.

    `model`/`tokenizer` are injectable so offline tests can run the real loop on a
    tiny from-scratch model with a fake pool — the loop under test is the same one
    that runs on the GPU.

    Cross-tokenizer kwargs:
      bridge: CrossTokenizerBridge — overrides cfg.bridge_path when provided.
      teacher_tokenizer: teacher-side HF tokenizer for encoding teacher prefixes
          and retokenising the student's action text.  When None, loaded from
          cfg.teacher_model (or taken from cfg.teacher_tokenizer).
      prefix_cache: TeacherPrefixCache — caches rendered teacher prefix ids so
          that the same (task, step_index) renders identically across all steps.
    """
    _require_train()
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    if not examples:
        raise ValueError("no ReOPD examples — nothing to distil from")

    if cfg.verify_tokenizers and not cfg.cross_tokenizer:
        from .tokenizer_check import check_tokenizers
        check_tokenizers(cfg.teacher_model, cfg.student_model)

    # ── Cross-tokenizer setup ─────────────────────────────────────────────────
    _bridge = bridge
    _teacher_tok: Any = teacher_tokenizer if teacher_tokenizer is not None else cfg.teacher_tokenizer
    _prefix_cache = prefix_cache
    if cfg.cross_tokenizer and _prefix_cache is None:
        from .teacher_cross import TeacherPrefixCache
        _prefix_cache = TeacherPrefixCache()

    if cfg.cross_tokenizer:
        from .encoding_dsv4 import ENCODING_DSV4_SHA256, verify_encoding_dsv4_pin
        from .vocab_bridge import CrossTokenizerBridge

        # §10.7 — refuse to train against a drifted vendored encoder.
        verify_encoding_dsv4_pin()

        if _bridge is None and cfg.bridge_path is not None:
            _bridge = CrossTokenizerBridge.load(cfg.bridge_path)
        if _bridge is None:
            raise ValueError(
                "cross_tokenizer=True requires a bridge — pass bridge= kwarg or "
                "set cfg.bridge_path"
            )
        if _bridge.encoding_dsv4_hash != ENCODING_DSV4_SHA256:
            raise RuntimeError(
                f"bridge encoding_dsv4 hash mismatch: "
                f"bridge has {_bridge.encoding_dsv4_hash!r}, "
                f"current encoder is {ENCODING_DSV4_SHA256!r}. "
                "Rebuild the bridge with `vektori-trace build-bridge`."
            )
        if _teacher_tok is None:
            _teacher_tok = AutoTokenizer.from_pretrained(
                cfg.teacher_model, trust_remote_code=True
            )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = cfg.output_dir / "adapter"
    log_path = cfg.output_dir / "opd_log.jsonl"

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(cfg.student_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Cross-tokenizer gate runs after both tokenizers exist (FINAL-PLAN.md:
    # verify_tokenizers calls check_cross_tokenizer — not check_tokenizers,
    # and not nothing). Offline tests set verify_tokenizers=False.
    if cfg.verify_tokenizers and cfg.cross_tokenizer:
        from .vocab_bridge import check_cross_tokenizer

        check_cross_tokenizer(
            cfg.teacher_model,
            cfg.student_model,
            teacher_tokenizer=_teacher_tok,
            student_tokenizer=tokenizer,
            thinking_mode=cfg.thinking_mode,
        )

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
    _last_cross_stats: Any = None  # most recent CrossStepStats (cross path only)

    with log_path.open("w") as log:
        for step in range(cfg.max_steps):
            optimizer.zero_grad(set_to_none=True)
            step_losses: list[Any] = []
            step_ratios: list[float] = []
            step_tokens = 0

            for _ in range(cfg.examples_per_step):
                ex = next_example()

                if cfg.cross_tokenizer:
                    # ── Cross-tokenizer path (FINAL-PLAN.md) ─────────────────
                    from .align import AlignmentError, align_by_bytes, classify_spans
                    from .cross_kl import cross_step_loss
                    from .teacher_cross import encode_teacher_ids

                    try:
                        student_prefix, teacher_prefix, teacher_prefix_text = encode_prefix_pair(
                            ex, tokenizer, _teacher_tok,
                            max_prefix_tokens=cfg.max_prefix_tokens,
                            thinking_mode=cfg.thinking_mode,
                            prefix_cache=_prefix_cache,
                        )
                    except ValueError:
                        skipped += 1
                        continue

                    action_ids = _sample_action(model, tokenizer, student_prefix, cfg)
                    if not action_ids:
                        skipped += 1
                        continue

                    # Build student byte list — strip tokens with empty bytes
                    # (EOS, unknown specials) so align_by_bytes sees only
                    # content-carrying tokens.
                    student_bytes_list: list[bytes] = []
                    student_aligned_positions: list[int] = []
                    for pos, tid in enumerate(action_ids):
                        b = _bridge.student_table.table.get(tid, b"")
                        if b:
                            student_bytes_list.append(b)
                            student_aligned_positions.append(pos)

                    if not student_bytes_list:
                        skipped += 1
                        continue

                    # Reconstruct the sampled text and re-tokenise for the teacher.
                    # Strict UTF-8: replacement would silently change byte length
                    # and produce a plausible-looking AlignmentError or worse, a
                    # finite wrong loss (FINAL-PLAN.md §10).
                    try:
                        action_text = b"".join(student_bytes_list).decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise AlignmentError(
                            f"student action bytes are not valid UTF-8: {exc}"
                        ) from exc
                    teacher_action_ids = encode_teacher_ids(action_text, _teacher_tok)

                    # §10.3 — prefix/action junction must land on a teacher token
                    # boundary. encode(prefix)+encode(action) == encode(prefix+action).
                    teacher_joined = encode_teacher_ids(
                        teacher_prefix_text + action_text, _teacher_tok
                    )
                    if list(teacher_prefix) + list(teacher_action_ids) != list(teacher_joined):
                        raise AlignmentError(
                            "teacher prefix/action byte junction is not on a token "
                            "boundary: encode(prefix)+encode(action) != "
                            "encode(prefix+action)"
                        )

                    # Build teacher byte list — same EOS-stripping convention.
                    teacher_bytes_list: list[bytes] = []
                    teacher_aligned_positions: list[int] = []
                    for pos, tid in enumerate(teacher_action_ids):
                        b = _bridge.teacher_table.table.get(tid, b"")
                        if b:
                            teacher_bytes_list.append(b)
                            teacher_aligned_positions.append(pos)

                    if not teacher_bytes_list:
                        skipped += 1
                        continue

                    # Byte alignment — AlignmentError is a hard fail (FINAL-PLAN
                    # §10 / "never best-effort"), not a skippable example.
                    alignment = align_by_bytes(
                        student_bytes_list,
                        teacher_bytes_list,
                        max_span_student_tokens=cfg.max_span_student_tokens,
                    )

                    # Granularity gate (hard fail #6 per FINAL-PLAN.md §10).
                    if alignment.granularity < cfg.min_alignment_granularity:
                        raise RuntimeError(
                            f"alignment granularity {alignment.granularity:.3f} < "
                            f"min_alignment_granularity={cfg.min_alignment_granularity:.3f} "
                            "— the student action and teacher re-tokenisation have "
                            "desynced beyond the pre-registered floor"
                        )

                    # Special-token masks over the *aligned* streams (§10.4).
                    from .vocab_bridge import _special_ids

                    student_specials = _special_ids(tokenizer)
                    teacher_specials = _special_ids(_teacher_tok)
                    student_special_mask = [
                        action_ids[p] in student_specials for p in student_aligned_positions
                    ]
                    teacher_special_mask = [
                        teacher_action_ids[p] in teacher_specials
                        for p in teacher_aligned_positions
                    ]
                    span_kinds = classify_spans(
                        alignment,
                        student_special_mask=student_special_mask,
                        teacher_special_mask=teacher_special_mask,
                    )

                    # ── Teacher scoring (teacher-side ids) ───────────────────
                    teacher_lp_all = pool.score_ids(teacher_prefix, teacher_action_ids)
                    teacher_lp_aligned = [teacher_lp_all[p] for p in teacher_aligned_positions]

                    teacher_topk_by_teacher_pos: dict[int, dict[int, float]] = {}
                    if cfg.cross_top_k > 0 and hasattr(pool, "score_ids_topk"):
                        teacher_topk_all = pool.score_ids_topk(
                            teacher_prefix, teacher_action_ids, cfg.cross_top_k
                        )
                        for aligned_pos, orig_pos in enumerate(teacher_aligned_positions):
                            teacher_topk_by_teacher_pos[aligned_pos] = teacher_topk_all[orig_pos]

                    # ── Student logits (student-side ids, with grad) ──────────
                    logits = _student_action_logits(model, student_prefix, action_ids)
                    action_t = torch.tensor([action_ids], device=logits.device)

                    # Full log_softmax over vocab at every action position.
                    # Shape [A, V] where A = len(action_ids).
                    lp_all = torch.log_softmax(logits[0].float(), dim=-1)
                    # Per-token log π_s for the sampled ids, shape [A].
                    token_lp_all = token_logprobs(logits, action_t)[0]

                    # Slice to aligned (non-EOS) positions.
                    pos_tensor = torch.tensor(
                        student_aligned_positions, dtype=torch.long, device=logits.device
                    )
                    student_logprobs_full = lp_all[pos_tensor]       # [n_aligned, V]
                    student_token_logprobs_vec = token_lp_all[pos_tensor]  # [n_aligned]

                    loss, cross_stats = cross_step_loss(
                        alignment=alignment,
                        span_kinds=span_kinds,
                        student_logprobs_full=student_logprobs_full,
                        student_token_logprobs=student_token_logprobs_vec,
                        teacher_token_logprobs=teacher_lp_aligned,
                        teacher_topk_by_teacher_pos=teacher_topk_by_teacher_pos,
                        exact_map=_bridge.exact_map,
                        student_token_bytes=student_bytes_list,
                    )

                    (loss / cfg.examples_per_step).backward()

                    step_losses.append(float(loss.detach()))
                    n_aligned = len(student_aligned_positions)
                    step_tokens += n_aligned
                    # Monitoring ratio — use span-level sums so student and
                    # teacher have the same count regardless of span width.
                    if n_aligned > 0 and alignment.spans:
                        from .align import span_logprob_sums
                        span_pairs = span_logprob_sums(
                            alignment,
                            [float(x) for x in student_token_logprobs_vec.detach().tolist()],
                            teacher_lp_aligned,
                        )
                        if span_pairs:
                            step_ratios.append(
                                sum(s - t for s, t in span_pairs) / len(span_pairs)
                            )
                    # Carry cross_stats for log line below.
                    _last_cross_stats = cross_stats

                else:
                    # ── Same-vocab path (byte-identical to before) ────────────
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
            final_ratio = sum(step_ratios) / len(step_ratios) if step_ratios else None
            tokens_scored += step_tokens
            log_entry: dict[str, Any] = {
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
            if cfg.cross_tokenizer and _last_cross_stats is not None:
                log_entry.update({
                    "granularity": _last_cross_stats.granularity,
                    "frac_A": _last_cross_stats.frac_A,
                    "frac_B": _last_cross_stats.frac_B,
                    "frac_dropped": _last_cross_stats.frac_dropped,
                    "bytes_aligned": _last_cross_stats.bytes_aligned,
                    "bytes_total": _last_cross_stats.bytes_total,
                    "special_tokens_masked": _last_cross_stats.special_tokens_masked,
                    "n_other_clamped": _last_cross_stats.n_other_clamped,
                    "n_A": _last_cross_stats.n_A,
                    "n_B": _last_cross_stats.n_B,
                    "dropped_by_reason": _last_cross_stats.dropped_by_reason,
                    "dropped_by_content_type": _last_cross_stats.dropped_by_content_type,
                })
                # §10.11 — above 1% of A positions clamped → logprobs too noisy.
                n_A = _last_cross_stats.n_A
                if n_A > 0 and _last_cross_stats.n_other_clamped / n_A > 0.01:
                    raise RuntimeError(
                        f"§10.11: {_last_cross_stats.n_other_clamped}/{n_A} "
                        "Estimator-A positions needed other_t clamp "
                        "(>1%) — teacher logprobs too noisy for coarse-grained A"
                    )
            log.write(json.dumps(log_entry) + "\n")
            log.flush()

    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # State the objective that actually ran, not the branch name: the two paths
    # optimise different quantities and a report must not conflate them.
    if cfg.cross_tokenizer:
        from .encoding_dsv4 import ENCODING_DSV4_SHA256
        provenance: dict[str, Any] = {
            "branch": "OPD",
            "loss": "cross_tokenizer_reverse_kl",
            "thinking_mode": cfg.thinking_mode,
            "cross_top_k": cfg.cross_top_k,
            "temperature": cfg.temperature,
            "encoding_dsv4": ENCODING_DSV4_SHA256,
        }
        if _bridge is not None:
            provenance["bridge_teacher_fingerprint"] = _bridge.teacher_fingerprint.vocab_sha256
            provenance["bridge_student_fingerprint"] = _bridge.student_fingerprint.vocab_sha256
    else:
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
    "encode_prefix_pair",
    "run_opd_training",
    "write_opd_report",
]
