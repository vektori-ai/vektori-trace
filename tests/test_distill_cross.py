"""P5 offline fixture — cross-tokenizer OPD path.

Tests the cross-tokenizer branch of run_opd_training with:
- A tiny GPT-2 student (same as test_distill.py)
- A synthetic CrossTokenizerBridge where student_table == teacher_table (every
  token id maps to the same bytes on both sides) so alignment is 1↔1 perfect.
- A fake teacher tokenizer that encodes like the student, so the action text
  re-tokenises to the same ids.
- A fake pool that scores teacher-side ids.

The equivalence oracle (§7): with identical byte tables and a 1↔1 bridge,
the cross-tokenizer loss must be close to the same-vocab reverse_kl_surrogate
over the aligned tokens. The test verifies finite loss and adapter creation.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from vektori_trace.dataset import TEST_CHAT_TEMPLATE
from vektori_trace.distill import (
    OPDTrainConfig,
    encode_prefix_pair,
    run_opd_training,
)
from vektori_trace.encoding_dsv4 import ENCODING_DSV4_SHA256
from vektori_trace.reopd import build_reopd_example
from vektori_trace.schema import ToolCall, Turn
from vektori_trace.tokenizer_check import TokenizerFingerprint
from vektori_trace.train import LoraHyperparams
from vektori_trace.vocab_bridge import CrossTokenizerBridge

pytest.importorskip("torch")
pytest.importorskip("peft")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _turns() -> list[Turn]:
    return [
        Turn(index=0, role="user", content="fix the failing test"),
        Turn(
            index=1,
            role="assistant",
            content="reading the file",
            tool_calls=[ToolCall(id="t1", name="read", args={"path": "a.py"})],
        ),
        Turn(index=2, role="tool", tool_call_id="t1", content="def f(): return 1"),
        Turn(index=3, role="assistant", content="patching now"),
    ]


@pytest.fixture()
def tiny():
    """Tiny GPT-2 student — same fixture as test_distill.py."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tok.chat_template = TEST_CHAT_TEMPLATE
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel(
        GPT2Config(
            n_layer=2, n_head=2, n_embd=32,
            vocab_size=max(tok.vocab_size, 64),
            n_positions=512,
        )
    )
    model = get_peft_model(
        model,
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, target_modules=["c_attn"]),
    )
    return model, tok


def _make_bridge(tok) -> CrossTokenizerBridge:
    """Synthetic bridge where student_table == teacher_table.

    Uses build_byte_table for proper ByteLevel decoding, so the action
    text round-trip (sample ids → bytes → text → re-encode → teacher ids)
    is correct. Every token maps to the same bytes on both sides, so
    align_by_bytes produces all 1↔1 spans (granularity=1.0).
    """
    from vektori_trace.vocab_bridge import build_byte_table, build_exact_token_map

    bt = build_byte_table(tok)
    exact_map = build_exact_token_map(bt, bt, tok_teacher=tok, tok_student=tok)

    dummy_fp = TokenizerFingerprint(
        name="test-tokenizer",
        vocab_size=bt.vocab_size,
        merges_sha256="0" * 64,
        vocab_sha256="0" * 64,
    )
    return CrossTokenizerBridge(
        teacher_table=bt,
        student_table=bt,
        exact_map=exact_map,
        teacher_fingerprint=dummy_fp,
        student_fingerprint=dummy_fp,
        encoding_dsv4_hash=ENCODING_DSV4_SHA256,
        thinking_mode="chat",
    )


def _fake_teacher_tokenizer(student_tok):
    """A teacher tokenizer that encodes text exactly like the student."""
    # Wrap the student tokenizer so that encode() produces the same ids.
    # encode_teacher_ids calls tokenizer.encode(text, add_special_tokens=False).
    class _FakeTeacherTok:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return student_tok.encode(text, add_special_tokens=add_special_tokens)

        def apply_chat_template(self, *args, **kwargs):
            return student_tok.apply_chat_template(*args, **kwargs)

        def get_vocab(self):
            return student_tok.get_vocab()

        def convert_ids_to_tokens(self, ids):
            return student_tok.convert_ids_to_tokens(ids)

        def get_added_tokens_decoder(self):
            # The real API `_special_ids` reads. A fake that omits it would let
            # the §10.4 masking regression back in unnoticed.
            return student_tok.backend_tokenizer.get_added_tokens_decoder()

        @property
        def vocab_size(self):
            return student_tok.vocab_size

        @property
        def pad_token_id(self):
            return student_tok.pad_token_id

        @property
        def eos_token_id(self):
            return student_tok.eos_token_id

    return _FakeTeacherTok()


class _FakeCrossPool:
    """Fake pool for cross-tokenizer tests — accepts teacher-side ids."""

    def __init__(self, *, base: float = -0.5, per_token: float = -0.01):
        self.base = base
        self.per_token = per_token
        self.score_calls = 0
        self.last_prompt_ids: list[int] = []
        self.last_tokens: list[int] = []

    def score_ids(self, prompt_ids: list[int], tokens: list[int]) -> list[float]:
        self.score_calls += 1
        self.last_prompt_ids = list(prompt_ids)
        self.last_tokens = list(tokens)
        return [self.base + self.per_token * i for i, _ in enumerate(tokens)]

    def score_ids_topk(
        self, prompt_ids: list[int], tokens: list[int], top_k: int
    ) -> list[dict[int, float]]:
        self.score_calls += 1
        self.last_prompt_ids = list(prompt_ids)
        self.last_tokens = list(tokens)
        rows = []
        for i, tid in enumerate(tokens):
            row = {int(tid): self.base + self.per_token * i}
            n = 0
            while len(row) < top_k:
                candidate = 10_000 + n
                if candidate != int(tid):
                    row[candidate] = self.base - 1.0 - n
                n += 1
            rows.append(row)
        return rows

    def provenance(self) -> dict[str, Any]:
        return {"teacher_model": "fake-cross", "teacher_api_base": "none"}


def _cfg(tmp_path, **kw):
    base = {
        "output_dir": tmp_path / "opd",
        "max_steps": 2,
        "examples_per_step": 1,
        "max_new_tokens": 6,
        "verify_tokenizers": False,
        "lora": LoraHyperparams(r=4, alpha=8, dropout=0.0, target_modules=["c_attn"]),
        "cross_tokenizer": True,
        "cross_top_k": 3,
    }
    base.update(kw)
    return OPDTrainConfig(**base)


# ─────────────────────────────────────────────────────────────────────────────
# Unit test: encode_prefix_pair returns two non-empty id lists
# ─────────────────────────────────────────────────────────────────────────────


def test_encode_prefix_pair_returns_two_lists(tiny):
    """encode_prefix_pair returns (student_ids, teacher_ids, teacher_text)."""
    from vektori_trace.reopd import build_reopd_example

    _, tok = tiny
    fake_teacher_tok = _fake_teacher_tokenizer(tok)
    ex = build_reopd_example(_turns(), task="t", action_index=1)

    s_ids, t_ids, t_text = encode_prefix_pair(
        ex, tok, fake_teacher_tok, max_prefix_tokens=512, thinking_mode="chat"
    )

    assert len(s_ids) > 0, "student prefix must not be empty"
    assert len(t_ids) > 0, "teacher prefix must not be empty"
    assert isinstance(t_text, str) and t_text, "teacher prefix text required for §10.3"


def test_encode_prefix_pair_truncates_at_message_boundary(tiny):
    """When max_prefix_tokens forces truncation, both ids fit within the limit.

    With max_prefix_tokens=50:
    - Full encoding (3 turns): teacher=182 tokens > 50  → truncates
    - After dropping turns until 1 turn remains: teacher=39 ≤ 50  → fits
    """
    from vektori_trace.reopd import build_reopd_example

    _, tok = tiny
    fake_teacher_tok = _fake_teacher_tokenizer(tok)
    ex = build_reopd_example(_turns(), task="t", action_index=1)

    # max_prefix_tokens=50 forces truncation: full teacher encoding is ~182 tokens,
    # but with 1 turn the teacher encoding is ~39 tokens which fits.
    s_ids, t_ids, _t_text = encode_prefix_pair(
        ex, tok, fake_teacher_tok, max_prefix_tokens=50, thinking_mode="chat"
    )

    assert len(s_ids) <= 50, f"student prefix {len(s_ids)} exceeds max_prefix_tokens=50"
    assert len(t_ids) <= 50, f"teacher prefix {len(t_ids)} exceeds max_prefix_tokens=50"


# ─────────────────────────────────────────────────────────────────────────────
# P5: cross-tokenizer loop trains and writes an adapter
# ─────────────────────────────────────────────────────────────────────────────


def _ascii_action_ids(tok, text: str = " hello") -> list[int]:
    """Deterministic valid-UTF-8 action ids for offline cross-tokenizer tests.

    Random tiny-GPT-2 samples can emit byte sequences that are not valid UTF-8;
    with the §10 strict decode those hard-fail the run. Tests pin a known-good
    ASCII action so alignment/junction stay deterministic.
    """
    ids = tok.encode(text, add_special_tokens=False)
    assert ids, "action text must encode to at least one token"
    return list(ids)


@pytest.fixture()
def ascii_action(monkeypatch, tiny):
    """Patch `_sample_action` to always return a fixed ASCII token sequence."""
    _, tok = tiny
    fixed = _ascii_action_ids(tok)

    def _fixed_sample(model, tokenizer, prefix_ids, cfg):
        return list(fixed)

    monkeypatch.setattr("vektori_trace.distill._sample_action", _fixed_sample)
    return fixed


def test_cross_tokenizer_loop_trains_and_writes_adapter(tmp_path, tiny, ascii_action):
    """Full cross-tokenizer loop: finite loss, adapter written, 2 steps complete."""
    model, tok = tiny
    bridge = _make_bridge(tok)
    fake_teacher_tok = _fake_teacher_tokenizer(tok)
    pool = _FakeCrossPool()

    ex = build_reopd_example(_turns(), task="t", action_index=1)

    result = run_opd_training(
        [ex],
        pool,
        _cfg(tmp_path),
        model=model,
        tokenizer=tok,
        bridge=bridge,
        teacher_tokenizer=fake_teacher_tok,
    )

    assert result.steps == 2
    assert result.final_loss is not None
    assert result.final_loss == result.final_loss  # not NaN
    assert result.adapter_dir.is_dir()
    assert (result.adapter_dir / "adapter_config.json").is_file()
    assert result.provenance["loss"] == "cross_tokenizer_reverse_kl"
    assert result.provenance["thinking_mode"] == "chat"


def test_cross_tokenizer_loop_loss_is_finite(tmp_path, tiny, ascii_action):
    """Cross-tokenizer loss must be finite after multiple steps."""
    import math

    model, tok = tiny
    bridge = _make_bridge(tok)
    fake_teacher_tok = _fake_teacher_tokenizer(tok)
    pool = _FakeCrossPool()

    ex = build_reopd_example(_turns(), task="t", action_index=1)

    result = run_opd_training(
        [ex],
        pool,
        _cfg(tmp_path, max_steps=3, learning_rate=1e-2),
        model=model,
        tokenizer=tok,
        bridge=bridge,
        teacher_tokenizer=fake_teacher_tok,
    )

    assert result.final_loss is not None
    assert math.isfinite(result.final_loss)


def test_cross_tokenizer_gradient_reaches_lora(tmp_path, tiny, ascii_action):
    """Backward from cross-tokenizer loss reaches LoRA parameters."""
    import torch

    model, tok = tiny
    bridge = _make_bridge(tok)
    fake_teacher_tok = _fake_teacher_tokenizer(tok)

    before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad and "lora" in n
    }
    assert before, "no LoRA parameters to check"

    ex = build_reopd_example(_turns(), task="t", action_index=1)

    run_opd_training(
        [ex],
        _FakeCrossPool(),
        _cfg(tmp_path, max_steps=3, learning_rate=1e-2),
        model=model,
        tokenizer=tok,
        bridge=bridge,
        teacher_tokenizer=fake_teacher_tok,
    )

    moved = [
        n
        for n, p in model.named_parameters()
        if n in before and not torch.allclose(before[n], p.detach())
    ]
    assert moved, "LoRA weights did not move — gradient is not connected"


def test_cross_tokenizer_log_records_granularity(tmp_path, tiny, ascii_action):
    """Log lines for the cross path include granularity and frac_A/frac_B."""
    model, tok = tiny
    bridge = _make_bridge(tok)
    fake_teacher_tok = _fake_teacher_tokenizer(tok)

    ex = build_reopd_example(_turns(), task="t", action_index=1)

    result = run_opd_training(
        [ex],
        _FakeCrossPool(),
        _cfg(tmp_path, max_steps=2),
        model=model,
        tokenizer=tok,
        bridge=bridge,
        teacher_tokenizer=fake_teacher_tok,
    )

    rows = [json.loads(line) for line in result.log_path.read_text().splitlines()
            if not json.loads(line).get("skipped_step")]
    assert rows, "no log rows written"
    for row in rows:
        # Cross-path log entries carry alignment stats.
        assert "granularity" in row, f"missing 'granularity' in row: {row}"
        assert "frac_A" in row or "frac_B" in row, "missing estimator fractions"
        assert "n_other_clamped" in row


def test_cross_tokenizer_provenance_records_bridge_fingerprints(tmp_path, tiny, ascii_action):
    """Provenance carries bridge fingerprints and encoding_dsv4 hash."""
    model, tok = tiny
    bridge = _make_bridge(tok)
    fake_teacher_tok = _fake_teacher_tokenizer(tok)

    ex = build_reopd_example(_turns(), task="t", action_index=1)

    result = run_opd_training(
        [ex],
        _FakeCrossPool(),
        _cfg(tmp_path),
        model=model,
        tokenizer=tok,
        bridge=bridge,
        teacher_tokenizer=fake_teacher_tok,
    )

    prov = result.provenance
    assert prov.get("encoding_dsv4") == ENCODING_DSV4_SHA256
    assert "bridge_teacher_fingerprint" in prov
    assert "bridge_student_fingerprint" in prov
    assert "bytes_aligned" in prov
    assert "bytes_total" in prov
    assert "dropped_by_content_type" in prov or prov.get("dropped_by_content_type") is None
    assert "frac_dropped" in prov
    assert "n_other_clamped" in prov
    assert "eos_stripped" in prov
    assert "student_entropy" in prov


def test_cross_tokenizer_equivalence_oracle_matches_reverse_kl(tiny, ascii_action):
    """Identical-tokenizer cross path ≈ reverse_kl_surrogate on 1↔1 spans.

    With student_table == teacher_table and a 1↔1 bridge, Estimator B on each
    span must match ``opd.reverse_kl_surrogate`` over the same token logprobs
    (FINAL-PLAN §7 equivalence oracle, distill-level).
    """
    import torch

    from vektori_trace import opd
    from vektori_trace.align import align_by_bytes, classify_spans
    from vektori_trace.cross_kl import cross_step_loss
    from vektori_trace.vocab_bridge import build_byte_table

    _, tok = tiny
    bridge = _make_bridge(tok)
    action_ids = _ascii_action_ids(tok)
    bt = build_byte_table(tok)
    s_bytes = [bt.table[i] for i in action_ids if bt.table.get(i)]
    assert s_bytes
    alignment = align_by_bytes(s_bytes, s_bytes)
    kinds = classify_spans(alignment)
    assert all(k == kinds[0][0] for k in [x[0] for x in kinds])  # all same kind

    # Fake per-token logprobs — student differentiable, teacher detached scalars.
    n = len(s_bytes)
    student_token_lp = torch.tensor(
        [-0.4 - 0.05 * i for i in range(n)], dtype=torch.float32, requires_grad=True
    )
    teacher_token_lp = [-0.5 - 0.03 * i for i in range(n)]
    # Full vocab logits unused when every span is B or when A has no topk.
    V = max(tok.vocab_size, 64)
    student_full = torch.randn(n, V, dtype=torch.float32)
    student_full = torch.log_softmax(student_full, dim=-1)

    loss_cross, stats = cross_step_loss(
        alignment=alignment,
        span_kinds=kinds,
        student_logprobs_full=student_full,
        student_token_logprobs=student_token_lp,
        teacher_token_logprobs=teacher_token_lp,
        teacher_topk_by_teacher_pos={},  # force B path
        exact_map=bridge.exact_map,
        student_token_bytes=s_bytes,
    )

    loss_ref = opd.reverse_kl_surrogate(
        student_token_lp.unsqueeze(0),
        torch.tensor([teacher_token_lp], dtype=torch.float32),
    )
    assert torch.allclose(loss_cross, loss_ref, rtol=1e-5, atol=1e-6), (
        f"cross={float(loss_cross)} ref={float(loss_ref)} stats={stats}"
    )


def test_cross_tokenizer_thinking_mode_validation():
    """OPDTrainConfig rejects unknown thinking_mode when cross_tokenizer=True."""
    with pytest.raises(ValueError, match="thinking_mode"):
        OPDTrainConfig(cross_tokenizer=True, thinking_mode="unknown")

    # Valid modes must not raise.
    OPDTrainConfig(cross_tokenizer=True, thinking_mode="chat")
    OPDTrainConfig(cross_tokenizer=True, thinking_mode="thinking")


def test_cross_tokenizer_requires_bridge_at_training_time(tmp_path, tiny):
    """run_opd_training raises if cross_tokenizer=True but no bridge is provided."""
    model, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)

    with pytest.raises(ValueError, match="bridge"):
        run_opd_training(
            [ex],
            _FakeCrossPool(),
            _cfg(tmp_path),  # cross_tokenizer=True, no bridge_path
            model=model,
            tokenizer=tok,
            # bridge= not passed → should raise
        )


def test_same_vocab_path_unchanged(tmp_path, tiny):
    """Same-vocab path (cross_tokenizer=False) still works after the refactor."""
    from vektori_trace.providers.teacher.base import InMemoryIdScoringPool

    model, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)
    pool = InMemoryIdScoringPool()

    result = run_opd_training(
        [ex],
        pool,
        OPDTrainConfig(
            output_dir=tmp_path / "opd",
            max_steps=2,
            examples_per_step=1,
            max_new_tokens=6,
            verify_tokenizers=False,
            lora=LoraHyperparams(r=4, alpha=8, dropout=0.0, target_modules=["c_attn"]),
        ),
        model=model,
        tokenizer=tok,
    )

    assert result.steps == 2
    assert result.final_loss is not None
    assert result.final_loss == result.final_loss  # not NaN
    assert result.adapter_dir.is_dir()
    assert result.provenance["loss"] == "reverse_kl_surrogate"


# ─────────────────────────────────────────────────────────────────────────────
# P6: teacher continuation vs corrupted — catches scrambled alignment
# ─────────────────────────────────────────────────────────────────────────────


def test_p6_span_log_ratio_favors_teacher_own_text(tiny):
    """FINAL-PLAN.md P6 (offline fixture).

    Score a correct continuation vs a deliberately corrupted one. Span-level
    mean log-ratio (log π_s − log π_t) must clearly favor the teacher's own
    text — the only offline check that catches a scrambled alignment producing
    a finite but wrong loss.
    """
    from vektori_trace.align import align_by_bytes, span_logprob_sums
    from vektori_trace.vocab_bridge import build_byte_table

    _, tok = tiny
    bt = build_byte_table(tok)
    correct = "return result"
    corrupt = "return Xesult"  # one-byte corruption

    def mean_log_ratio_for(text: str, *, teacher_bonus: float) -> float:
        ids = tok.encode(text, add_special_tokens=False)
        byte_list = [bt.table[i] for i in ids if bt.table.get(i)]
        assert byte_list, text
        alignment = align_by_bytes(byte_list, byte_list)
        # Fixed student logprobs; teacher LP higher (less negative) when bonus>0.
        s_lp = [-0.5] * len(byte_list)
        t_lp = [-0.1 - teacher_bonus] * len(byte_list)
        pairs = span_logprob_sums(alignment, s_lp, t_lp)
        assert pairs
        return sum(s - t for s, t in pairs) / len(pairs)

    # Teacher assigns higher probability to its own text than to the corruption.
    ratio_correct = mean_log_ratio_for(correct, teacher_bonus=0.0)
    ratio_corrupt = mean_log_ratio_for(corrupt, teacher_bonus=2.0)
    assert ratio_correct < ratio_corrupt, (
        f"correct continuation must win on span log-ratio: "
        f"correct={ratio_correct}, corrupt={ratio_corrupt}"
    )


def test_p6_scrambled_teacher_logprobs_are_detectable(tiny):
    """A permuted teacher logprob vector changes per-span (s,t) pairs.

    Global means can be invariant under permutation; the scrambled-alignment
    bug shows up as wrong *pairing*. This fixture asserts the per-span pairs
    differ — the signal P6 is meant to catch.
    """
    from vektori_trace.align import align_by_bytes, span_logprob_sums
    from vektori_trace.vocab_bridge import build_byte_table

    _, tok = tiny
    bt = build_byte_table(tok)
    text = "alpha beta gamma"
    ids = tok.encode(text, add_special_tokens=False)
    byte_list = [bt.table[i] for i in ids if bt.table.get(i)]
    alignment = align_by_bytes(byte_list, byte_list)
    n = len(byte_list)
    assert n >= 2

    s_lp = [-0.4 - 0.01 * i for i in range(n)]
    t_lp = [-0.01 * (2**i) for i in range(n)]
    aligned = span_logprob_sums(alignment, s_lp, t_lp)
    rotated = span_logprob_sums(alignment, s_lp, t_lp[1:] + t_lp[:1])
    assert aligned != rotated


def test_cross_tokenizer_log_records_global_denominator(tmp_path, tiny, ascii_action):
    """§6: the step log carries the batch-level supervised-token denominator."""
    model, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)
    pool = _FakeCrossPool()

    result = run_opd_training(
        [ex],
        pool,
        _cfg(tmp_path, examples_per_step=2),
        model=model,
        tokenizer=tok,
        bridge=_make_bridge(tok),
        teacher_tokenizer=_fake_teacher_tokenizer(tok),
    )

    lines = [
        json.loads(ln)
        for ln in result.log_path.read_text().splitlines()
        if ln.strip()
    ]
    stepped = [ln for ln in lines if not ln.get("skipped_step")]
    assert stepped, "no completed steps"
    for ln in stepped:
        # Present, positive, and at least as large as any single example could
        # supply on its own — i.e. it is a batch total, not a per-example count.
        assert ln["supervised_tokens"] > 0
        assert ln["supervised_tokens"] >= ln["n_A"] + ln["n_B"]


def test_cross_tokenizer_uses_one_teacher_round_trip_per_example(tmp_path, tiny, ascii_action):
    """§2 Cost shape: teacher latency sets step time, so one echo call per example.

    score_ids_topk already carries each scored token's own logprob, so a
    separate score_ids call would be a second round-trip for data we hold.
    """
    model, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)
    pool = _FakeCrossPool()

    run_opd_training(
        [ex],
        pool,
        _cfg(tmp_path, max_steps=3, examples_per_step=2, cross_top_k=3),
        model=model,
        tokenizer=tok,
        bridge=_make_bridge(tok),
        teacher_tokenizer=_fake_teacher_tokenizer(tok),
    )

    # 3 steps × 2 examples = 6 scoring calls, not 12.
    assert pool.score_calls == 6, f"expected 6 teacher calls, got {pool.score_calls}"


def test_cross_top_k_zero_falls_back_to_plain_score_ids(tmp_path, tiny, ascii_action):
    """cross_top_k=0 disables Estimator A; B needs only the per-token scalars."""
    model, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)

    class _NoTopK(_FakeCrossPool):
        def score_ids_topk(self, *a, **kw):  # pragma: no cover - must not run
            raise AssertionError("score_ids_topk called with cross_top_k=0")

    pool = _NoTopK()
    result = run_opd_training(
        [ex],
        pool,
        _cfg(tmp_path, max_steps=1, cross_top_k=0),
        model=model,
        tokenizer=tok,
        bridge=_make_bridge(tok),
        teacher_tokenizer=_fake_teacher_tokenizer(tok),
    )
    assert pool.score_calls == 1
    assert result.provenance["n_A"] == 0


def test_topk_row_missing_scored_token_is_a_hard_error(tmp_path, tiny, ascii_action):
    """A top-K row without the scored token's own logprob must not be guessed at."""
    model, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)

    class _DropsScoredToken(_FakeCrossPool):
        def score_ids_topk(self, prompt_ids, tokens, top_k):
            rows = super().score_ids_topk(prompt_ids, tokens, top_k)
            rows[0].pop(int(tokens[0]), None)
            return rows

    with pytest.raises(RuntimeError, match="carries no logprob for the scored"):
        run_opd_training(
            [ex],
            _DropsScoredToken(),
            _cfg(tmp_path, max_steps=1),
            model=model,
            tokenizer=tok,
            bridge=_make_bridge(tok),
            teacher_tokenizer=_fake_teacher_tokenizer(tok),
        )


# ─────────────────────────────────────────────────────────────────────────────
# §7 — "EOS is stripped on both sides before alignment, and counted."
# ─────────────────────────────────────────────────────────────────────────────


def test_special_token_literal_has_real_bytes_not_empty(tiny):
    """Why emptiness is the wrong strip test: EOS round-trips to real bytes.

    `<|endoftext|>` is ASCII, so every character is in the ByteLevel alphabet
    and `_token_str_to_bytes` returns the literal ten-plus bytes — not b"".
    Any strip keyed on byte-emptiness therefore strips nothing.
    """
    from vektori_trace.vocab_bridge import build_byte_table

    _, tok = tiny
    bt = build_byte_table(tok)
    eos = tok.eos_token_id
    assert eos is not None
    assert bt.table[eos] != b"", "EOS has no bytes — this test's premise is stale"


def test_trailing_eos_is_stripped_and_counted(tmp_path, tiny, monkeypatch):
    """A sampled action ending in EOS must not carry the EOS literal into alignment."""
    model, tok = tiny
    content = _ascii_action_ids(tok)
    with_eos = [*content, tok.eos_token_id]
    monkeypatch.setattr(
        "vektori_trace.distill._sample_action",
        lambda model, tokenizer, prefix_ids, cfg: list(with_eos),
    )

    ex = build_reopd_example(_turns(), task="t", action_index=1)
    pool = _FakeCrossPool()
    result = run_opd_training(
        [ex],
        pool,
        _cfg(tmp_path, max_steps=1),
        model=model,
        tokenizer=tok,
        bridge=_make_bridge(tok),
        teacher_tokenizer=_fake_teacher_tokenizer(tok),
    )

    # Counted, not silently swallowed.
    assert result.provenance["eos_stripped"] >= 1
    # The teacher was asked to score the action text only — no EOS literal.
    assert "endoftext" not in tok.decode(pool.last_tokens)
    # And the EOS token itself never reached the teacher.
    assert tok.eos_token_id not in pool.last_tokens


def test_action_that_is_only_eos_is_skipped_not_scored(tmp_path, tiny, monkeypatch):
    """EOS-only sample has no supervised token; it is a skip, not agreement."""
    model, tok = tiny
    monkeypatch.setattr(
        "vektori_trace.distill._sample_action",
        lambda model, tokenizer, prefix_ids, cfg: [tok.eos_token_id],
    )

    ex = build_reopd_example(_turns(), task="t", action_index=1)
    pool = _FakeCrossPool()
    result = run_opd_training(
        [ex],
        pool,
        _cfg(tmp_path, max_steps=1),
        model=model,
        tokenizer=tok,
        bridge=_make_bridge(tok),
        teacher_tokenizer=_fake_teacher_tokenizer(tok),
    )

    assert result.skipped_empty_samples >= 1
    assert pool.score_calls == 0


def test_cached_prefix_never_crosses_truncation_depths(tiny):
    """A prefix cached untruncated must not be served after truncation.

    Regression: the cache key omitted truncation depth, so a teacher prefix
    that fitted at depth 0 kept being returned while the student side dropped
    turns — a truncated student context paired with an untruncated teacher one,
    the asymmetry §7 forbids.
    """
    from vektori_trace.providers.teacher.cross import TeacherPrefixCache, encode_teacher_ids
    from vektori_trace.reopd import build_reopd_example

    _, tok = tiny
    fake_teacher_tok = _fake_teacher_tokenizer(tok)
    ex = build_reopd_example(_turns(), task="t", action_index=1)
    cache = TeacherPrefixCache()

    # Depth 0: generous budget, nothing is dropped.
    _s0, t0, text0 = encode_prefix_pair(
        ex, tok, fake_teacher_tok, max_prefix_tokens=10_000,
        thinking_mode="chat", prefix_cache=cache,
    )
    # Depth >0: tight budget forces turns to be dropped on both sides.
    s1, t1, text1 = encode_prefix_pair(
        ex, tok, fake_teacher_tok, max_prefix_tokens=50,
        thinking_mode="chat", prefix_cache=cache,
    )

    assert len(t1) < len(t0), "tight budget did not actually truncate"
    assert len(s1) <= 50 and len(t1) <= 50
    # The returned ids describe the text returned alongside them — not a stale
    # deeper prefix pulled from the cache.
    assert t1 == encode_teacher_ids(text1, fake_teacher_tok)
    assert t0 == encode_teacher_ids(text0, fake_teacher_tok)
    # Both depths are retained under distinct keys.
    assert cache.get("t", ex.step_index, n_dropped_turns=0) == t0


def test_prefix_render_drift_is_caught_by_the_cache(tiny, monkeypatch):
    """A non-deterministic render for the same key must hard-fail."""
    from vektori_trace.providers.teacher.cross import PrefixCacheConflict, TeacherPrefixCache
    from vektori_trace.reopd import build_reopd_example

    _, tok = tiny
    fake_teacher_tok = _fake_teacher_tokenizer(tok)
    ex = build_reopd_example(_turns(), task="t", action_index=1)
    cache = TeacherPrefixCache()

    encode_prefix_pair(
        ex, tok, fake_teacher_tok, max_prefix_tokens=10_000,
        thinking_mode="chat", prefix_cache=cache,
    )

    # Simulate a date/locale creeping into the render path between steps.
    monkeypatch.setattr(
        "vektori_trace.providers.teacher.cross.render_teacher_prefix",
        lambda messages, *, thinking_mode="chat": "drifted prefix text",
    )
    with pytest.raises(PrefixCacheConflict, match="re-rendered differently"):
        encode_prefix_pair(
            ex, tok, fake_teacher_tok, max_prefix_tokens=10_000,
            thinking_mode="chat", prefix_cache=cache,
        )


def test_estimator_A_actually_runs_in_the_training_loop(tmp_path, tiny, ascii_action):
    """The analytic branch must be reachable end to end, not just in unit tests.

    Every equivalence oracle at the cross_step_loss level forces the B path by
    passing no top-K, so without this the real loop could have been running
    100% Estimator B — the silent demotion §9 warns about for the hosted
    student — and every test would still be green.
    """
    model, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)

    result = run_opd_training(
        [ex],
        _FakeCrossPool(),
        _cfg(tmp_path, max_steps=1, cross_top_k=3),
        model=model,
        tokenizer=tok,
        bridge=_make_bridge(tok),
        teacher_tokenizer=_fake_teacher_tokenizer(tok),
    )

    assert result.provenance["n_A"] > 0, (
        f"Estimator A never ran: {result.provenance}"
    )
    lines = [json.loads(x) for x in result.log_path.read_text().splitlines() if x.strip()]
    stepped = [ln for ln in lines if not ln.get("skipped_step")]
    assert stepped and stepped[0]["frac_A"] > 0.0
    # Entropy must be recorded from step 0 — §6's mode-collapse tripwire.
    assert stepped[0]["student_entropy"] is not None


def test_cross_tokenizer_defaults_to_the_deepseek_teacher():
    """The Qwen pilot default is the student's own family — wrong for cross."""
    from vektori_trace.tokenizer_check import CROSS_TEACHER, DEFAULT_TEACHER

    same = OPDTrainConfig(output_dir="/tmp/x")
    assert same.teacher_model == DEFAULT_TEACHER

    cross = OPDTrainConfig(output_dir="/tmp/x", cross_tokenizer=True)
    assert cross.teacher_model == CROSS_TEACHER == "deepseek-ai/DeepSeek-V4-Flash-0731"


def test_explicit_teacher_model_is_respected_in_cross_mode():
    """The redirect only replaces the untouched default, never a real choice."""
    cfg = OPDTrainConfig(
        output_dir="/tmp/x", cross_tokenizer=True, teacher_model="some/other-teacher"
    )
    assert cfg.teacher_model == "some/other-teacher"


def test_token_log_is_actually_written_with_both_sides(tmp_path, tiny, ascii_action):
    """The per-token log must survive a real loop, not just exist in source.

    Source-level assertions prove the code mentions `student_logprob`; they do
    not prove a run emits a parseable record with both sides of the comparison
    in it. This runs the loop and reads the file back.
    """
    model, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)

    result = run_opd_training(
        [ex],
        _FakeCrossPool(),
        _cfg(tmp_path, max_steps=1, cross_top_k=3),
        model=model,
        tokenizer=tok,
        bridge=_make_bridge(tok),
        teacher_tokenizer=_fake_teacher_tokenizer(tok),
    )

    tokens_path = result.adapter_dir.parent / "opd_tokens.jsonl"
    assert tokens_path.is_file(), "opd_tokens.jsonl was never written"
    rows = [json.loads(x) for x in tokens_path.read_text().splitlines() if x.strip()]
    assert rows, "token log is empty after a step that trained"
    assert not any("token_log_error" in r for r in rows), (
        f"token logging raised: {[r for r in rows if 'token_log_error' in r][:1]}"
    )

    r = rows[0]
    # Both sides of the comparison the objective is built from.
    assert isinstance(r["student_logprob"], float)
    assert isinstance(r["teacher_logprob"], float)
    assert r["student_logprob"] <= 0.0 and r["teacher_logprob"] <= 0.0
    assert isinstance(r["student_token"], str)
    assert r["task"] == "t"
    assert r["pos"] == 0

    # Teacher alternatives and the coverage that decides A vs B. cross_top_k=3
    # here, so at least one row must carry them.
    with_topk = [x for x in rows if "teacher_topk" in x]
    assert with_topk, "no row carried the teacher's top-K"
    t = with_topk[0]
    assert t["teacher_topk"] and "logprob" in t["teacher_topk"][0]
    assert 0.0 <= t["mapped_mass"] <= 1.0
    assert t["mapped_count"] >= 0
    assert t["topk_width"] >= 1


def test_example_log_records_what_the_student_wrote(tmp_path, tiny, ascii_action):
    """`action_text` must round-trip to a real string, not None."""
    model, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)

    result = run_opd_training(
        [ex],
        _FakeCrossPool(),
        _cfg(tmp_path, max_steps=1, cross_top_k=3),
        model=model,
        tokenizer=tok,
        bridge=_make_bridge(tok),
        teacher_tokenizer=_fake_teacher_tokenizer(tok),
    )

    ex_path = result.adapter_dir.parent / "opd_examples.jsonl"
    assert ex_path.is_file(), "opd_examples.jsonl was never written"
    rows = [json.loads(x) for x in ex_path.read_text().splitlines() if x.strip()]
    assert rows
    assert rows[0]["task"] == "t"
    assert isinstance(rows[0]["action_text"], str) and rows[0]["action_text"]
    assert rows[0]["action_tokens"] and rows[0]["action_tokens"] > 0
