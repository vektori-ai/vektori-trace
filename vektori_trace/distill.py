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
import math
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .dataset import turns_to_messages
from .opd import mean_log_ratio, reverse_kl_surrogate, token_logprobs, topk_reverse_kl
from .providers.teacher.protocol import IdScoringPool, TopKScoringPool
from .reopd import ReOPDStepExample, reopd_loss_mask
from .tokenizer_check import CROSS_TEACHER, DEFAULT_STUDENT, DEFAULT_TEACHER
from .train import LoraHyperparams, _require_train


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
        # `teacher_model` defaults to the same-vocab Qwen pilot teacher. In the
        # cross path that default is not merely unhelpful, it is wrong: it is
        # the *student's* tokenizer family, so the byte tables would agree, the
        # alignment would be trivially 1↔1, and the run would look healthy while
        # scoring against a teacher nobody chose. Redirect the untouched default;
        # an explicit teacher_model is always respected.
        if self.cross_tokenizer and self.teacher_model == DEFAULT_TEACHER:
            self.teacher_model = CROSS_TEACHER


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
    from .providers.teacher.cross import (
        TeacherPrefixCache,
        encode_teacher_ids,
        render_teacher_prefix,
        turns_to_openai_messages,
    )

    prefix_turns = list(example.prefix_turns)

    # `n_dropped_turns` is the loop counter *and* part of the cache key, so it
    # is the truncation depth this iteration is testing.
    for n_dropped_turns in range(len(prefix_turns) + 1):
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

        # ── Teacher encoding ──────────────────────────────────────────────
        # Always re-render and re-encode, then `put`. The cache is a drift
        # detector, not a shortcut: the plan is explicit that caching saves
        # ~1 ms against a ~1 s round-trip, so "speed decides nothing", and its
        # whole value is catching a prefix that renders differently at step 200
        # than at step 5. Serving a cached value without re-deriving it would
        # silence exactly the bug it exists to surface.
        teacher_messages = turns_to_openai_messages(prefix_turns)
        teacher_text = render_teacher_prefix(teacher_messages, thinking_mode=thinking_mode)
        teacher_ids = encode_teacher_ids(teacher_text, teacher_tokenizer)

        if len(student_ids) <= max_prefix_tokens and len(teacher_ids) <= max_prefix_tokens:
            # Keyed on the truncation depth too: both sides must describe the
            # same turn set, so a prefix cached at one depth must never be
            # served at another (§7 shared message boundary).
            if isinstance(prefix_cache, TeacherPrefixCache):
                prefix_cache.put(
                    example.task or "", example.step_index, teacher_ids,
                    n_dropped_turns=n_dropped_turns,
                    thinking_mode=thinking_mode,
                )
            return student_ids, teacher_ids, teacher_text

        # Drop the earliest turn and retry — on both sides together.
        prefix_turns = prefix_turns[1:]

    raise ValueError(
        f"{example.task} step {example.step_index}: empty prefix after cross-tokenizer "
        "truncation — no turns fit within max_prefix_tokens on both sides"
    )


def _decode_utf8_dropping_partial_tail(raw: bytes) -> tuple[str, int]:
    """Decode `raw` as UTF-8, dropping at most 3 bytes of a truncated tail.

    Returns ``(text, n_bytes_dropped)``. A decode error that is *not* an
    incomplete sequence at the very end re-raises: mid-stream invalid UTF-8
    means the byte table or the alignment is wrong, which must hard-fail.
    """
    try:
        return raw.decode("utf-8"), 0
    except UnicodeDecodeError as exc:
        # A truncated tail reports `unexpected end of data` (or an invalid
        # continuation) with `start` inside the last 3 bytes of the buffer —
        # the longest UTF-8 sequence is 4 bytes.
        if exc.end != len(raw) or len(raw) - exc.start > 3:
            from .align import AlignmentError

            raise AlignmentError(
                f"student action bytes are not valid UTF-8: {exc}"
            ) from exc
        dropped = len(raw) - exc.start
        return raw[: exc.start].decode("utf-8"), dropped


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


def _setup_cross_tokenizer(
    cfg: OPDTrainConfig,
    *,
    bridge: Any | None,
    teacher_tokenizer: Any | None,
    prefix_cache: Any | None,
) -> tuple[Any | None, Any, Any | None]:
    """Resolve and validate the cross-tokenizer inputs.

    Returns `(bridge, teacher_tokenizer, prefix_cache)`. Precedence for each is
    the caller's kwarg, then `cfg`, then a load from `cfg.teacher_model`. On the
    same-vocab path everything passes through untouched.

    Split out of `run_opd_training` because it imports no torch: the two hard
    failures here — no bridge, and a bridge built against a drifted
    `encoding_dsv4` — are reachable offline, and both are worth failing on
    before a GPU is allocated.
    """
    _bridge = bridge
    _teacher_tok: Any = teacher_tokenizer if teacher_tokenizer is not None else cfg.teacher_tokenizer
    _prefix_cache = prefix_cache
    if cfg.cross_tokenizer and _prefix_cache is None:
        from .providers.teacher.cross import TeacherPrefixCache

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
            # `load_tokenizer`, not AutoTokenizer: the teacher side only ever
            # needs encode/get_vocab, so a repo whose model config this
            # `transformers` cannot parse must not block training. The *student*
            # stays on AutoTokenizer — it needs `apply_chat_template`, which a
            # bare `tokenizers.Tokenizer` does not have.
            from .vocab_bridge import load_tokenizer

            _teacher_tok = load_tokenizer(cfg.teacher_model)

    return _bridge, _teacher_tok, _prefix_cache


def _resolve_student_model(cfg: OPDTrainConfig, model: Any | None) -> Any:
    """Load the student and attach LoRA, unless the caller injected a model.

    Offline tests pass a tiny from-scratch model, which is why the injected
    path skips loading but still gets the gradient-checkpointing treatment —
    that setting is about how the model is *run*, not where it came from.
    """
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

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
    return model


def _build_optimizer(cfg: OPDTrainConfig, model: Any) -> tuple[list[Any], Any, Any]:
    """AdamW over the trainable (LoRA) parameters, plus a cosine schedule.

    `params` is returned rather than kept local because the training loop needs
    the same list for gradient clipping — recomputing it there would silently
    diverge if anything ever froze or unfroze a parameter mid-run.
    """
    import torch
    from transformers import get_cosine_schedule_with_warmup

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters — LoRA did not attach to the model")
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(cfg.max_steps * cfg.warmup_ratio), 0),
        num_training_steps=cfg.max_steps,
    )
    return params, optimizer, scheduler


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
    from transformers import AutoTokenizer

    if not examples:
        raise ValueError("no ReOPD examples — nothing to distil from")

    if cfg.verify_tokenizers and not cfg.cross_tokenizer:
        from .tokenizer_check import check_tokenizers
        check_tokenizers(cfg.teacher_model, cfg.student_model)

    _bridge, _teacher_tok, _prefix_cache = _setup_cross_tokenizer(
        cfg, bridge=bridge, teacher_tokenizer=teacher_tokenizer, prefix_cache=prefix_cache
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

    # Special-token id sets, resolved once: they are consulted for every sampled
    # action, and `_special_ids` walks the whole added vocabulary each call.
    _student_specials: frozenset[int] = frozenset()
    _teacher_specials: frozenset[int] = frozenset()
    if cfg.cross_tokenizer:
        from .vocab_bridge import _special_ids

        _student_specials = _special_ids(tokenizer)
        _teacher_specials = _special_ids(_teacher_tok)

    model = _resolve_student_model(cfg, model)
    params, optimizer, scheduler = _build_optimizer(cfg, model)

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
    # Run-level aggregates for provenance (§10.10 / §10.11).
    _run_bytes_aligned = 0
    _run_bytes_total = 0
    _run_n_A = 0
    _run_n_B = 0
    _run_n_other_clamped = 0
    _run_special_tokens_masked = 0
    _run_dropped_by_content_type: dict[str, int] = {}
    _run_dropped_spans = 0
    _run_total_spans = 0
    _run_eos_stripped = 0
    _run_utf8_tail_dropped = 0
    _run_low_granularity_skipped = 0
    _run_granularity_sum = 0.0
    _run_granularity_n = 0
    _run_student_entropy_sum = 0.0
    _run_student_entropy_n = 0

    # Per-example detail goes to its own file. `opd_log.jsonl` is one line per
    # optimiser step and other tools index it positionally, so interleaving a
    # second record type there would silently change an artifact that already
    # has readers. Same run, three granularities, no schema change.
    examples_log_path = cfg.output_dir / "opd_examples.jsonl"
    #: Per-token: what the student sampled, its logprob, the teacher's logprob
    #: for the same token, and the teacher's top-K alternatives. Cross-tokenizer
    #: only — the same-vocab path has no alignment to explain.
    tokens_log_path = cfg.output_dir / "opd_tokens.jsonl"

    def _teacher_tok_decode(tid: int) -> str | None:
        """Decode one teacher id for the token log; never raise."""
        try:
            return _teacher_tok.decode([int(tid)])
        except Exception:
            return None

    with (
        log_path.open("w") as log,
        examples_log_path.open("w") as ex_log,
        tokens_log_path.open("w") as tok_log_f,
    ):
        tok_log = tok_log_f if cfg.cross_tokenizer else None
        for step in range(cfg.max_steps):
            optimizer.zero_grad(set_to_none=True)
            step_losses: list[Any] = []
            step_ratios: list[float] = []
            step_tokens = 0
            # §6 global denominator: supervised student tokens across the whole
            # batch, across both estimators.  Applied to the accumulated grads
            # once, after the example loop.
            step_supervised_tokens = 0
            # Step-aggregated cross stats (not last-example-only).
            step_n_A = 0
            step_n_B = 0
            step_n_dropped = 0
            step_n_spans = 0
            step_bytes_aligned = 0
            step_bytes_total = 0
            step_special_masked = 0
            step_n_other_clamped = 0
            step_granularity_sum = 0.0
            step_granularity_n = 0
            step_entropy_sum = 0.0
            step_entropy_n = 0
            step_dropped_by_reason: dict[str, int] = {}
            step_dropped_by_content_type: dict[str, int] = {}

            for _ in range(cfg.examples_per_step):
                ex = next_example()
                # Per-example log state. Both are reset here because the loop
                # body reuses names across iterations: without this an example
                # that aborts early would be logged with the *previous*
                # example's token count and span stats, which is worse than no
                # record at all.
                _ex_action_tokens: int | None = None
                _ex_action_text: str | None = None
                _ex_cross_stats: Any = None

                if cfg.cross_tokenizer:
                    # ── Cross-tokenizer path (FINAL-PLAN.md) ─────────────────
                    from .align import AlignmentError, align_by_bytes, classify_spans
                    from .cross_kl import cross_step_loss
                    from .providers.teacher.cross import encode_teacher_ids

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

                    # §7 — "EOS is stripped on both sides before alignment, and
                    # counted: it has no byte extent and is the likeliest source
                    # of a phantom zero-width span."
                    #
                    # Byte-emptiness alone does NOT identify EOS here. A special
                    # token's literal form is ASCII, so `_token_str_to_bytes`
                    # round-trips `<|im_end|>` through the ByteLevel table to ten
                    # real bytes. Stripping on emptiness therefore stripped
                    # nothing: the EOS literal entered the alignment as content,
                    # was re-tokenised and sent to the teacher as text, and could
                    # form a span wide enough to trip the max_span_student_tokens
                    # desync check on a perfectly ordinary action.
                    #
                    # Strip trailing special-or-empty tokens by id. Mid-sequence
                    # specials stay in the byte stream — removing them would
                    # splice unrelated bytes together — and are masked out of the
                    # loss by classify_spans instead (§10.4 "partition around,
                    # mask, count").
                    trimmed_ids = list(action_ids)
                    while trimmed_ids and (
                        trimmed_ids[-1] in _student_specials
                        or not _bridge.student_table.table.get(trimmed_ids[-1], b"")
                    ):
                        trimmed_ids.pop()
                        _run_eos_stripped += 1

                    student_bytes_list: list[bytes] = []
                    student_aligned_positions: list[int] = []
                    for pos, tid in enumerate(trimmed_ids):
                        b = _bridge.student_table.table.get(tid, b"")
                        if b:
                            student_bytes_list.append(b)
                            student_aligned_positions.append(pos)
                        elif tid in _student_specials:
                            _run_eos_stripped += 1
                        else:
                            # A sampled id with no byte-table entry that is not a
                            # known special. Qwen3-8B's config declares
                            # vocab_size=151936 while its tokenizer defines
                            # 151669 — so 267 ids are samplable from the softmax
                            # and have no bytes. Dropping one mid-sequence
                            # splices the byte stream: the teacher is then sent
                            # text the student never produced, every later
                            # position is conditioned differently on the two
                            # sides, granularity still reads 1.000, and the loss
                            # is finite. Exactly the §10 silent-corruption shape,
                            # so hard-fail instead.
                            raise AlignmentError(
                                f"sampled student id {tid} at action position {pos} "
                                "has no entry in the bridge byte table and is not a "
                                "known special token. The model's softmax is wider "
                                "than its tokenizer (config vocab_size vs "
                                "len(get_vocab())); dropping it would splice the "
                                "byte stream and silently mis-condition the teacher."
                            )

                    if not student_bytes_list:
                        # The sample was EOS (or specials) and nothing else. No
                        # supervised token exists; counting it as agreement would
                        # be a fabricated datum (same rule as the same-vocab
                        # empty-sample case).
                        skipped += 1
                        continue

                    # Reconstruct the sampled text and re-tokenise for the teacher.
                    # Strict UTF-8: replacement would silently change byte length
                    # and produce a plausible-looking AlignmentError or worse, a
                    # finite wrong loss (FINAL-PLAN.md §10).
                    #
                    # One exception: `max_new_tokens` can cut the sample partway
                    # through a multi-byte character (Qwen3 emits ZWJ and rare
                    # CJK as byte-fallback tokens, so an emoji sequence spans
                    # several). That truncation is a sampling-window artifact at
                    # the *tail*, not a desync — hard-failing it would abort a
                    # 40-step run because one action ended on an emoji. Drop the
                    # incomplete trailing sequence and count it. A decode error
                    # anywhere else is still a hard fail.
                    raw_action_bytes = b"".join(student_bytes_list)
                    action_text, n_trunc = _decode_utf8_dropping_partial_tail(raw_action_bytes)
                    if n_trunc:
                        _run_utf8_tail_dropped += 1
                        keep = len(raw_action_bytes) - n_trunc
                        acc = 0
                        cut = len(student_bytes_list)
                        for i, b in enumerate(student_bytes_list):
                            if acc >= keep:
                                cut = i
                                break
                            acc += len(b)
                        student_bytes_list = student_bytes_list[:cut]
                        student_aligned_positions = student_aligned_positions[:cut]
                        if not student_bytes_list:
                            skipped += 1
                            continue
                        action_text = b"".join(student_bytes_list).decode("utf-8")
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

                    # Granularity gate (§10.6). The failure it exists to catch is
                    # *sequence-level* distillation being reported as token-level
                    # OPD — a property of the run, not of one sample. Applying it
                    # per example aborted the whole run on legitimate content:
                    # measured against the real tokenizers, a single line like
                    # `size=1073741824 free=536870912 used=536870912` gives 0.471,
                    # three unix timestamps 0.526, a 40-digit run 0.362 — every
                    # one byte-exact and correctly aligned. §4 measured 0.667 on
                    # *aggregated* numeric content; per-sample variance is much
                    # wider, so one unlucky action out of thousands killed a
                    # 40-step run with no checkpoint recovery.
                    #
                    # Skip and count the sample; gate the run-level aggregate
                    # below, which is what the section actually asserts.
                    if alignment.granularity < cfg.min_alignment_granularity:
                        _run_low_granularity_skipped += 1
                        skipped += 1
                        continue

                    # Special-token masks over the *aligned* streams (§10.4).
                    # Trailing specials are already gone; this catches the
                    # mid-sequence ones, whose bytes must stay in the stream to
                    # keep the two byte sequences equal but which carry no
                    # supervisable signal.
                    student_special_mask = [
                        trimmed_ids[p] in _student_specials
                        for p in student_aligned_positions
                    ]
                    teacher_special_mask = [
                        teacher_action_ids[p] in _teacher_specials
                        for p in teacher_aligned_positions
                    ]
                    span_kinds = classify_spans(
                        alignment,
                        student_special_mask=student_special_mask,
                        teacher_special_mask=teacher_special_mask,
                    )

                    # ── Teacher scoring (teacher-side ids) ───────────────────
                    # ONE round-trip. `score_ids_topk` already merges each
                    # scored token's own logprob into its row (the sampled
                    # token is guaranteed present even when outside the top-K),
                    # so calling `score_ids` as well would be a second echo call
                    # returning information we already hold. The plan names
                    # teacher latency — not student FLOPs — as what sets step
                    # time (§2 Cost shape), so the duplicate doubled it.
                    teacher_topk_by_teacher_pos: dict[int, dict[int, float]] = {}
                    if cfg.cross_top_k > 0 and hasattr(pool, "score_ids_topk"):
                        teacher_topk_all = pool.score_ids_topk(
                            teacher_prefix, teacher_action_ids, cfg.cross_top_k
                        )
                        if len(teacher_topk_all) != len(teacher_action_ids):
                            raise RuntimeError(
                                f"teacher returned {len(teacher_topk_all)} top-K rows "
                                f"for {len(teacher_action_ids)} scored tokens"
                            )
                        teacher_lp_all = []
                        for i, tid in enumerate(teacher_action_ids):
                            row = teacher_topk_all[i]
                            if int(tid) not in row:
                                raise RuntimeError(
                                    f"top-K row {i} carries no logprob for the scored "
                                    f"token id {tid} — score_ids_topk must merge the "
                                    "scored token in even when it is outside the top-K"
                                )
                            teacher_lp_all.append(row[int(tid)])
                        for aligned_pos, orig_pos in enumerate(teacher_aligned_positions):
                            teacher_topk_by_teacher_pos[aligned_pos] = teacher_topk_all[orig_pos]
                    else:
                        # cross_top_k=0 disables Estimator A; B needs only the
                        # per-token scalars, so one plain echo call suffices.
                        teacher_lp_all = pool.score_ids(teacher_prefix, teacher_action_ids)
                        if len(teacher_lp_all) != len(teacher_action_ids):
                            raise RuntimeError(
                                f"teacher returned {len(teacher_lp_all)} logprobs for "
                                f"{len(teacher_action_ids)} scored tokens"
                            )

                    teacher_lp_aligned = [teacher_lp_all[p] for p in teacher_aligned_positions]

                    # ── Student logits (student-side ids, with grad) ──────────
                    # `trimmed_ids`, not `action_ids`: the stripped trailing EOS
                    # has no aligned position, so forwarding it would only widen
                    # the tensor and make `student_aligned_positions` index into
                    # a differently-sized axis than the one it was built from.
                    logits = _student_action_logits(model, student_prefix, trimmed_ids)
                    action_t = torch.tensor([trimmed_ids], device=logits.device)

                    # Full log_softmax over vocab at every action position.
                    # Shape [A, V] where A = len(trimmed_ids).
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
                        # Raw sum: the §6 denominator is the *batch* supervised
                        # token count, which is not known until every example in
                        # this optimiser step has run.  Gradients are rescaled
                        # once by 1/total below — backward is linear, so this is
                        # exactly one global denominator, without holding every
                        # example's graph alive simultaneously.
                        denominator=1.0,
                    )

                    loss.backward()
                    step_supervised_tokens += cross_stats.n_supervised_tokens

                    # Report the normalised per-example value so `loss` in the
                    # log stays on the same scale as the same-vocab path.
                    step_losses.append(
                        float(loss.detach()) / cross_stats.n_supervised_tokens
                        if cross_stats.n_supervised_tokens
                        else 0.0
                    )
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
                    # Carry cross_stats for step aggregation below.
                    _last_cross_stats = cross_stats
                    _ex_cross_stats = cross_stats

                    # Per-example log state. Set here as well as in the
                    # same-vocab branch below: the two paths do their own token
                    # accounting, and a field assigned in only one of them is
                    # null for every real cross-tokenizer run — which is exactly
                    # what the offline fixture caught.
                    _ex_action_tokens = len(trimmed_ids)
                    try:
                        _ex_action_text = tokenizer.decode(trimmed_ids)
                    except Exception as _e:
                        _ex_action_text = f"<decode failed: {_e!r}>"

                    # ── Per-token forensics ──────────────────────────────────
                    # The step and example logs say a step went badly; only this
                    # says which tokens, what the student wrote, and what the
                    # teacher wanted instead. That is the difference between
                    # "loss was high" and a diagnosis. Written to its own file
                    # so the two aggregate logs keep their schemas.
                    if tok_log is not None:
                        try:
                            _s_lp = [float(x) for x in student_token_logprobs_vec.detach().tolist()]
                            for _i, _tid in enumerate(
                                [trimmed_ids[p] for p in student_aligned_positions]
                            ):
                                _rec: dict[str, Any] = {
                                    "step": step,
                                    "task": ex.task,
                                    "example_step_index": ex.step_index,
                                    "pos": _i,
                                    "student_id": int(_tid),
                                    "student_token": tokenizer.decode([int(_tid)]),
                                    "student_logprob": _s_lp[_i] if _i < len(_s_lp) else None,
                                    "teacher_logprob": (
                                        teacher_lp_aligned[_i]
                                        if _i < len(teacher_lp_aligned)
                                        else None
                                    ),
                                }
                                _row = teacher_topk_by_teacher_pos.get(_i)
                                if _row:
                                    # Coverage at this position: how much of the
                                    # teacher's reported top-K mass maps to a
                                    # byte-identical student token. Unmapped mass
                                    # is exact but undirected — it lands in the
                                    # `other` bucket and cannot say which token
                                    # to prefer. Below ~0.9 the span is demoted
                                    # from estimator A to B, so this is the
                                    # number that decides which estimator ran.
                                    _tot = sum(math.exp(v) for v in _row.values())
                                    _mapped = sum(
                                        math.exp(v)
                                        for k, v in _row.items()
                                        if _bridge is not None and int(k) in _bridge.exact_map
                                    )
                                    _rec["mapped_mass"] = (
                                        round(_mapped / _tot, 4) if _tot > 0 else None
                                    )
                                    _rec["mapped_count"] = sum(
                                        1
                                        for k in _row
                                        if _bridge is not None and int(k) in _bridge.exact_map
                                    )
                                    _rec["topk_width"] = len(_row)
                                    # Teacher's K+1 partition at this position:
                                    # what it would rather have said, and how
                                    # strongly. Decoded so it is readable
                                    # without the teacher tokenizer to hand.
                                    _rec["teacher_topk"] = [
                                        {
                                            "id": int(k),
                                            "logprob": round(float(v), 4),
                                            "token": (
                                                _teacher_tok_decode(k)
                                                if _teacher_tok is not None
                                                else None
                                            ),
                                        }
                                        for k, v in sorted(
                                            _row.items(), key=lambda kv: -kv[1]
                                        )[:6]
                                    ]
                                tok_log.write(json.dumps(_rec) + "\n")
                            tok_log.flush()
                        except Exception as _e:  # forensics must never kill a run
                            tok_log.write(
                                json.dumps({"step": step, "task": ex.task,
                                            "token_log_error": repr(_e)}) + "\n"
                            )
                            tok_log.flush()
                    step_n_A += cross_stats.n_A
                    step_n_B += cross_stats.n_B
                    step_n_spans += cross_stats.n_spans
                    n_drop = cross_stats.n_spans - cross_stats.n_A - cross_stats.n_B
                    step_n_dropped += max(0, n_drop)
                    step_bytes_aligned += cross_stats.bytes_aligned
                    step_bytes_total += cross_stats.bytes_total
                    step_special_masked += cross_stats.special_tokens_masked
                    step_n_other_clamped += cross_stats.n_other_clamped
                    step_granularity_sum += cross_stats.granularity
                    step_granularity_n += 1
                    if cross_stats.dropped_by_reason:
                        for k, v in cross_stats.dropped_by_reason.items():
                            step_dropped_by_reason[k] = step_dropped_by_reason.get(k, 0) + v
                    if cross_stats.dropped_by_content_type:
                        for k, v in cross_stats.dropped_by_content_type.items():
                            step_dropped_by_content_type[k] = (
                                step_dropped_by_content_type.get(k, 0) + v
                            )
                    if not math.isnan(cross_stats.student_entropy):
                        step_entropy_sum += cross_stats.student_entropy
                        step_entropy_n += 1

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
                    _ex_action_tokens = len(action_ids)
                    try:
                        _ex_action_text = tokenizer.decode(action_ids)
                    except Exception as _e:
                        _ex_action_text = f"<decode failed: {_e!r}>"

                # One record per *example*, not just per optimiser step. The
                # step line aggregates over `examples_per_step`, which is enough
                # to watch a run but not enough to explain one: with a corpus
                # drawn from several tasks, "granularity fell" is only
                # actionable if you can say which task's examples it fell on.
                # Written for the cross path and the same-vocab path alike.
                _ex_rec: dict[str, Any] = {
                    "step": step,
                    "task": ex.task,
                    "example_step_index": ex.step_index,
                    "action_tokens": _ex_action_tokens,
                    "loss": step_losses[-1] if step_losses else None,
                    "mean_log_ratio": step_ratios[-1] if step_ratios else None,
                    # What the student actually wrote. Without it the numeric
                    # columns describe a decision whose content is unrecoverable
                    # — and "it edited the workflow file instead of the resolver"
                    # is the kind of finding that only reads out of the text.
                    "action_text": _ex_action_text,
                }
                if cfg.cross_tokenizer and _ex_cross_stats is not None:
                    _cs = _ex_cross_stats
                    _ex_rec.update({
                        "n_A": _cs.n_A,
                        "n_B": _cs.n_B,
                        "n_spans": _cs.n_spans,
                        "granularity": _cs.granularity,
                        "bytes_aligned": _cs.bytes_aligned,
                        "bytes_total": _cs.bytes_total,
                        "n_other_clamped": _cs.n_other_clamped,
                        "special_tokens_masked": _cs.special_tokens_masked,
                        "dropped_by_reason": _cs.dropped_by_reason or None,
                        "dropped_by_content_type": _cs.dropped_by_content_type or None,
                        "student_entropy": (
                            None
                            if math.isnan(_cs.student_entropy)
                            else _cs.student_entropy
                        ),
                    })
                ex_log.write(json.dumps(_ex_rec) + "\n")
                ex_log.flush()

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

            if cfg.cross_tokenizer:
                # §6 Normalization — "one shared global denominator per optimizer
                # step: total supervised student tokens across the batch, across
                # BOTH estimators, never per-estimator". Each example backwarded
                # its raw sum, so one rescale here is that denominator. Doing it
                # per example instead would weight a 12-token example the same as
                # a 200-token one and let an A/B mix drift silently rescale the
                # learning rate — the exact failure the section names.
                if step_supervised_tokens <= 0:
                    log.write(
                        json.dumps({"step": step, "skipped_step": True,
                                    "reason": "no supervised tokens"}) + "\n"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scale = 1.0 / float(step_supervised_tokens)
                with torch.no_grad():
                    for p in params:
                        if p.grad is not None:
                            p.grad.mul_(scale)

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
            if cfg.cross_tokenizer and step_granularity_n > 0:
                supervised = step_n_A + step_n_B
                log_entry.update({
                    # The §6 global denominator actually applied to this step.
                    "supervised_tokens": step_supervised_tokens,
                    "granularity": step_granularity_sum / step_granularity_n,
                    "frac_A": step_n_A / supervised if supervised else 0.0,
                    "frac_B": step_n_B / supervised if supervised else 0.0,
                    "frac_dropped": (
                        step_n_dropped / step_n_spans if step_n_spans else 0.0
                    ),
                    "bytes_aligned": step_bytes_aligned,
                    "bytes_total": step_bytes_total,
                    "special_tokens_masked": step_special_masked,
                    "n_other_clamped": step_n_other_clamped,
                    "n_A": step_n_A,
                    "n_B": step_n_B,
                    "dropped_by_reason": step_dropped_by_reason or None,
                    "dropped_by_content_type": step_dropped_by_content_type or None,
                    "frac_dropped_by_content_type": (
                        {
                            k: v / step_n_spans
                            for k, v in step_dropped_by_content_type.items()
                        }
                        if step_n_spans and step_dropped_by_content_type
                        else None
                    ),
                    "student_entropy": (
                        step_entropy_sum / step_entropy_n if step_entropy_n else None
                    ),
                })
                _run_granularity_sum += step_granularity_sum
                _run_granularity_n += step_granularity_n
                _run_bytes_aligned += step_bytes_aligned
                _run_bytes_total += step_bytes_total
                _run_n_A += step_n_A
                _run_n_B += step_n_B
                _run_n_other_clamped += step_n_other_clamped
                _run_special_tokens_masked += step_special_masked
                _run_dropped_spans += step_n_dropped
                _run_total_spans += step_n_spans
                if step_entropy_n:
                    _run_student_entropy_sum += step_entropy_sum
                    _run_student_entropy_n += step_entropy_n
                for k, v in step_dropped_by_content_type.items():
                    _run_dropped_by_content_type[k] = (
                        _run_dropped_by_content_type.get(k, 0) + v
                    )
                # §10.11 — above 1% of A positions clamped → logprobs too noisy.
                # Aggregated across all examples in this optimiser step.
                if step_n_A > 0 and step_n_other_clamped / step_n_A > 0.01:
                    raise RuntimeError(
                        f"§10.11: {step_n_other_clamped}/{step_n_A} "
                        "Estimator-A positions needed other_t clamp "
                        "(>1%) — teacher logprobs too noisy for coarse-grained A"
                    )
            elif cfg.cross_tokenizer and _last_cross_stats is not None:
                # Defensive: an example produced stats but granularity_n stayed 0.
                log_entry["cross_stats_note"] = "no aggregated cross examples this step"
            log.write(json.dumps(log_entry) + "\n")
            log.flush()

            # Progress on stdout as well as in the log. On Modal the JSONL files
            # live on container-local disk until the run ends, so `modal app
            # logs` is the only live window into a run that costs money by the
            # second — silence for two hours is not an acceptable view of it.
            _gran = (
                _run_granularity_sum / _run_granularity_n
                if _run_granularity_n
                else float("nan")
            )
            _ratio = log_entry.get("mean_log_ratio")
            _ratio_s = f"{_ratio:.3f}" if _ratio is not None else "n/a"
            _cs = log_entry.get("cross_stats")
            _ent = _cs.get("student_entropy") if isinstance(_cs, dict) else None
            _ent_s = f"{_ent:.3f}" if _ent is not None else "n/a"
            print(
                f"step {step + 1:3d}/{cfg.max_steps} loss={mean_loss:.4f} "
                f"ratio={_ratio_s} tok={step_tokens} "
                f"A={step_n_A} B={step_n_B} gran={_gran:.3f} ent={_ent_s}",
                flush=True,
            )

    # §10.6 at run level — "granularity below min_alignment_granularity → abort.
    # Near-zero granularity is *sequence-level* distillation being reported as
    # token-level OPD: valid arithmetic, false claim." That is a claim about the
    # run, so it is asserted over the run's mean rather than per sample. Raise
    # before the adapter is written: an adapter on disk implies a valid run.
    if cfg.cross_tokenizer and _run_granularity_n > 0:
        _mean_gran = _run_granularity_sum / _run_granularity_n
        if _mean_gran < cfg.min_alignment_granularity:
            raise RuntimeError(
                f"mean alignment granularity {_mean_gran:.3f} < "
                f"min_alignment_granularity={cfg.min_alignment_granularity:.3f} "
                f"over {_run_granularity_n} supervised examples "
                f"({_run_low_granularity_skipped} further examples were skipped "
                "for falling below the floor individually). This run is "
                "sequence-level distillation reported as token-level OPD."
            )

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
            # §10.10 / §10.11 run-level coverage (every run).
            "bytes_aligned": _run_bytes_aligned,
            "bytes_total": _run_bytes_total,
            "frac_bytes_aligned": (
                _run_bytes_aligned / _run_bytes_total if _run_bytes_total else 0.0
            ),
            "n_A": _run_n_A,
            "n_B": _run_n_B,
            "n_other_clamped": _run_n_other_clamped,
            "special_tokens_masked": _run_special_tokens_masked,
            "eos_stripped": _run_eos_stripped,
            "utf8_tail_dropped": _run_utf8_tail_dropped,
            "low_granularity_skipped": _run_low_granularity_skipped,
            "mean_granularity": (
                _run_granularity_sum / _run_granularity_n if _run_granularity_n else None
            ),
            "dropped_by_content_type": _run_dropped_by_content_type or None,
            "frac_dropped_by_content_type": (
                {
                    k: v / _run_total_spans
                    for k, v in _run_dropped_by_content_type.items()
                }
                if _run_total_spans and _run_dropped_by_content_type
                else None
            ),
            "frac_dropped": (
                _run_dropped_spans / _run_total_spans if _run_total_spans else 0.0
            ),
            "student_entropy": (
                _run_student_entropy_sum / _run_student_entropy_n
                if _run_student_entropy_n
                else None
            ),
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


#: Seconds of wall clock to budget per supervised example, used to size the
#: Modal timeout. One example is a student `generate` (~15-25 s for a 14B via HF
#: eager), one teacher echo round-trip (~1-3 s), and a forward+backward over the
#: full prefix (~2-4 s). 45 s is the pessimistic end of that, because the cost of
#: overestimating is an idle container and the cost of underestimating is a
#: killed run with no adapter.
_MODAL_SECONDS_PER_EXAMPLE = 45

#: Students at or above this many billions of parameters cannot train without
#: gradient checkpointing on a 48 GB card: activations for a 3 840-token
#: sequence are ~31 GB on top of ~29.6 GB of bf16 weights.
_CHECKPOINTING_REQUIRED_FROM_B = 13.0


def _student_size_b(student_model: str) -> float | None:
    """Parameter count in billions parsed from an HF id, or None if absent.

    Only used to *force safety on*, never to relax it, so a name we cannot parse
    is not an error — it just means the caller's own setting stands.
    """
    import re

    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", student_model)
    return float(m.group(1)) if m else None


def modal_timeout_for(cfg: OPDTrainConfig) -> int:
    """Wall-clock budget for one OPD run, derived from the work requested.

    A fixed 4 h timeout silently kills a default-length run (`--max-steps 200`
    × `examples_per_step 4` ≈ 7 h) after the GPU has already been paid for, and
    Modal's kill takes the container's unflushed log with it. Size the timeout
    from the work instead, with a floor for cold start and weight download.
    """
    work = cfg.max_steps * cfg.examples_per_step * _MODAL_SECONDS_PER_EXAMPLE
    return int(min(max(work + 30 * 60, 60 * 60), 24 * 60 * 60))


def force_gpu_safe_config(cfg: OPDTrainConfig) -> OPDTrainConfig:
    """Turn on settings a large student cannot run without.

    A 14B student is ~29.6 GB of bf16 weights; un-checkpointed activations for a
    3 840-token sequence are ~31 GB on top of that, so a 48 GB card OOMs. The
    CLI flag defaults to off, which means the safe configuration is the one you
    have to remember. Fix that here rather than at minute forty of a paid run.

    Only ever tightens: a student whose size cannot be parsed keeps the caller's
    setting, and an already-enabled flag is left alone.
    """
    size_b = _student_size_b(cfg.student_model)
    if (
        size_b is not None
        and size_b >= _CHECKPOINTING_REQUIRED_FROM_B
        and not cfg.gradient_checkpointing
    ):
        from dataclasses import replace

        print(
            f"note: forcing gradient_checkpointing for a {size_b:g}B student — "
            "without it activations alone exceed the card"
        )
        return replace(cfg, gradient_checkpointing=True)
    return cfg


def run_opd_training_modal(
    cfg: OPDTrainConfig,
    *,
    prefixes_dir: Path,
    bridge_path: Path,
    fireworks_model: str,
    teacher_tokenizer_id: str,
    api_base: str | None = None,
    steps_per_trajectory: int | None = None,
    modal_gpu: str = "L40S",
    expected_examples: int | None = None,
) -> OPDTrainResult:
    """Execute `run_opd_training` on a Modal GPU. Adapter lands on the Volume.

    The machine that drives this project has no GPU, and `run_opd_training`
    runs in whatever process invokes it, so without this there is nowhere for
    cross-tokenizer OPD to run.

    **Inputs travel as files, not as arguments.** `ReOPDStepExample` holds
    `Turn` dataclasses that have a `from_dict` but no `to_dict`, and the bridge
    is ~10 MB; hand-rolling either serialisation is a silent-corruption risk for
    no gain. Both are uploaded to the shared Volume and the remote side rebuilds
    them with the same helpers the CLI uses, so there is one code path, not two.

    `fireworks_model` and `teacher_tokenizer_id` are separate on purpose: the
    first is a Fireworks resource path (`accounts/fireworks/models/…`), the
    second an HF repo id (`deepseek-ai/…`). They are different namespaces and
    passing one where the other belongs fails on the GPU, after the image pull.
    """
    import os
    import time

    # Validate everything cheap *before* importing modal, so a missing key or a
    # missing corpus reports itself as that, on any machine, rather than as an
    # import error — and long before a container could exist.
    prefixes_dir = Path(prefixes_dir)
    bridge_path = Path(bridge_path)
    if not prefixes_dir.is_dir():
        raise FileNotFoundError(f"prefix corpus not found: {prefixes_dir}")
    if not bridge_path.is_file():
        raise FileNotFoundError(f"bridge not found: {bridge_path}")

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        # The container costs money from the image pull onward, and this error
        # would otherwise arrive after the weights had downloaded.
        raise RuntimeError(
            "FIREWORKS_API_KEY is not set — the Modal container needs it to "
            "reach the teacher. Export it before launching."
        )

    cfg = force_gpu_safe_config(cfg)

    try:
        import modal
    except ImportError as e:  # pragma: no cover - env guard
        raise RuntimeError(
            "modal is part of the train extra — `uv sync --extra train`"
        ) from e

    from .runtime.modal_env import (
        HF_CACHE_MOUNT,
        HF_CACHE_VOLUME_NAME,
        VOLUME_MOUNT,
        VOLUME_NAME,
    )

    run_id = f"{int(time.time())}-seed{cfg.seed}"
    # Volume upload paths are relative to the volume ROOT, while reads inside a
    # container are under VOLUME_MOUNT. Do not "fix" these to match
    # `train.stage_local_adapter_to_volume`, which passes a mount-prefixed path
    # and therefore lands its files one directory deeper than it reads them.
    inputs_root = f"opd-inputs/{run_id}"
    remote_prefixes = f"{VOLUME_MOUNT}/{inputs_root}/prefixes"
    remote_bridge = f"{VOLUME_MOUNT}/{inputs_root}/bridge.json"
    remote_out = f"{VOLUME_MOUNT}/opd-out/{run_id}"

    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
    with vol.batch_upload(force=True) as batch:
        batch.put_directory(str(prefixes_dir), f"{inputs_root}/prefixes")
        batch.put_file(str(bridge_path), f"{inputs_root}/bridge.json")

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch", "transformers", "tokenizers", "peft", "accelerate")
        .env({"HF_HOME": HF_CACHE_MOUNT})
        .add_local_python_source("vektori_trace")
    )

    remote_cfg = {
        "student_model": cfg.student_model,
        "teacher_model": cfg.teacher_model,
        "max_steps": cfg.max_steps,
        "learning_rate": cfg.learning_rate,
        "examples_per_step": cfg.examples_per_step,
        "seed": cfg.seed,
        "temperature": cfg.temperature,
        "max_new_tokens": cfg.max_new_tokens,
        "max_prefix_tokens": cfg.max_prefix_tokens,
        "top_k": cfg.top_k,
        "bf16": cfg.bf16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "cross_tokenizer": cfg.cross_tokenizer,
        "thinking_mode": cfg.thinking_mode,
        "min_alignment_granularity": cfg.min_alignment_granularity,
        "max_span_student_tokens": cfg.max_span_student_tokens,
        "cross_top_k": cfg.cross_top_k,
        "lora": asdict(cfg.lora),
    }
    timeout_s = modal_timeout_for(cfg)
    app = modal.App("vektori-trace-opd")

    @app.function(
        gpu=modal_gpu,
        image=image,
        secrets=[modal.Secret.from_dict({"FIREWORKS_API_KEY": api_key})],
        volumes={VOLUME_MOUNT: vol, HF_CACHE_MOUNT: hf_cache},
        timeout=timeout_s,
        # `_remote` is defined here so it can close over the volume handles and
        # the config. Modal re-imports globals-scope functions by module path in
        # the container; `serialized=True` makes it pickle this one instead.
        # Without it the decorator raises before any GPU is allocated.
        serialized=True,
    )
    def _remote(job: dict) -> dict:
        import shutil as _shutil
        from dataclasses import asdict as _asdict
        from pathlib import Path as P

        from vektori_trace.distill import (
            LoraHyperparams as _Lora,
        )
        from vektori_trace.distill import (
            OPDTrainConfig as _Cfg,
        )
        from vektori_trace.distill import (
            run_opd_training as _run,
        )
        from vektori_trace.providers.teacher.cross import CrossTokenizerTeacherPool
        from vektori_trace.providers.teacher.fireworks import fireworks_pool_from_env
        from vektori_trace.reopd import iter_reopd_examples
        from vektori_trace.traces import load_teacher_trajectories
        from vektori_trace.vocab_bridge import CrossTokenizerBridge, load_tokenizer

        c = dict(job["cfg"])
        lora = _Lora(**c.pop("lora"))
        # Log to container-local disk: `run_opd_training` flushes it every step,
        # and per-step flushes onto a FUSE mount are slow. It is copied into the
        # Volume in the `finally` below, so a crash still leaves the evidence.
        local_out = P("/tmp/opd-out")
        local_out.mkdir(parents=True, exist_ok=True)
        cfg_r = _Cfg(output_dir=local_out, lora=lora, **c)

        bridge = CrossTokenizerBridge.load(job["bridge"])
        ttok = load_tokenizer(job["teacher_tokenizer_id"])
        pool = CrossTokenizerTeacherPool(
            pool=fireworks_pool_from_env(
                model=job["fireworks_model"], api_base=job["api_base"]
            ),
            teacher_tokenizer=ttok,
            thinking_mode=cfg_r.thinking_mode,
            bridge=bridge,
        )

        trajectories = load_teacher_trajectories(P(job["prefixes"]))
        examples = list(
            iter_reopd_examples(trajectories, steps_per_traj=job["steps_per_traj"])
        )

        dest = P(job["out"])
        try:
            result = _run(
                examples, pool, cfg_r, bridge=bridge, teacher_tokenizer=ttok
            )
            payload = _asdict(result)
            payload["adapter_dir"] = str(result.adapter_dir)
            payload["log_path"] = str(result.log_path) if result.log_path else None
            payload["n_trajectories"] = len(trajectories)
            payload["error"] = None
            return payload
        except Exception as e:
            return {
                "error": f"{type(e).__name__}: {e}",
                "n_examples": len(examples),
                "n_trajectories": len(trajectories),
            }
        finally:
            # Commit whatever exists. Every late abort in this loop — alignment,
            # the clamp gate, the run-level granularity gate — happens after
            # hours of GPU and teacher time, and `opd_log.jsonl` is the only
            # artifact that says why. Losing it is losing the run twice.
            dest.mkdir(parents=True, exist_ok=True)
            for f in local_out.rglob("*"):
                rel = f.relative_to(local_out)
                target = dest / rel
                if f.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    _shutil.copy2(f, target)
            vol.commit()

    job = {
        "cfg": remote_cfg,
        "bridge": remote_bridge,
        "prefixes": remote_prefixes,
        "out": remote_out,
        "fireworks_model": fireworks_model,
        "teacher_tokenizer_id": teacher_tokenizer_id,
        "api_base": api_base,
        "steps_per_traj": steps_per_trajectory,
    }
    print(
        f"OPD on Modal: gpu={modal_gpu} timeout={timeout_s // 60}min "
        f"run_id={run_id} → {remote_out}"
    )
    with app.run():
        remote = _remote.remote(job)

    if remote.get("error"):
        raise RuntimeError(
            f"remote OPD run failed: {remote['error']} — partial artifacts "
            f"committed to {remote_out} inside Volume {VOLUME_NAME}"
        )

    # The remote rebuilds the corpus from uploaded files. If the upload was
    # partial, the run silently trained on less than we measured; assert rather
    # than discover it in the report. `find_trajectory` breaks ties on mtime,
    # which Volume uploads do not preserve.
    if expected_examples is not None and remote["n_examples"] != expected_examples:
        raise RuntimeError(
            f"remote built {remote['n_examples']} examples but the local "
            f"pre-flight built {expected_examples} — the uploaded corpus does "
            "not match the one that was checked"
        )

    provenance = dict(remote.get("provenance") or {})
    provenance.update({"modal_gpu": modal_gpu, "run_id": run_id, "ran_on": "modal"})

    # The weights live on the Volume; the local dir records where, and does not
    # pretend to hold them.
    adapter_dir = cfg.output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "modal_volume_path.txt").write_text(f"{remote_out}/adapter\n")
    return OPDTrainResult(
        adapter_dir=adapter_dir,
        student_model=remote["student_model"],
        teacher_model=remote["teacher_model"],
        steps=remote["steps"],
        final_loss=remote["final_loss"],
        mean_log_ratio_final=remote["mean_log_ratio_final"],
        n_examples=remote["n_examples"],
        action_tokens_scored=remote["action_tokens_scored"],
        skipped_empty_samples=remote["skipped_empty_samples"],
        lora=remote["lora"],
        seed=remote["seed"],
        log_path=Path(f"{remote_out}/opd_log.jsonl"),
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
