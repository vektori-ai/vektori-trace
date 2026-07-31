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
    assert "n_other_clamped" in prov
    assert "eos_stripped" in prov


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
    from vektori_trace.teacher import InMemoryIdScoringPool

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
