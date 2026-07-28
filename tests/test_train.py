"""Tiny-model mechanical smoke test for train.py — wiring, not real weights.
Fully offline: from-scratch GPT2 + injected chat template (no Hub download).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from vektori_trace.dataset import (
    TEST_CHAT_TEMPLATE,
    TokenizedExample,
    tokenize_sft_example,
)
from vektori_trace.schema import Turn
from vektori_trace.train import LoraHyperparams, TrainConfig, train_lora


def _offline_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<|user|>": 2,
        "<|assistant|>": 3,
        "<|end|>": 4,
        "say": 5,
        "hi": 6,
        "hello": 7,
        "there": 8,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|end|>",
    )
    tok.chat_template = TEST_CHAT_TEMPLATE
    return tok


def test_train_lora_one_step_finite_loss(tmp_path: Path) -> None:
    tok = _offline_tokenizer()
    turns = [
        Turn(index=0, role="user", content="say hi"),
        Turn(index=1, role="assistant", content="hello there"),
    ]
    ex = tokenize_sft_example(turns, tok)
    assert ex is not None
    examples = [ex, TokenizedExample(ex.input_ids[:], ex.labels[:], ex.attention_mask[:])]

    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=max(tok.vocab_size, 64),
        n_positions=128,
    )
    model = GPT2LMHeadModel(config)

    result = train_lora(
        examples,
        TrainConfig(
            base_model="tiny-local",
            output_dir=tmp_path / "out",
            task_ids=["t1"],
            max_steps=1,
            seed=0,
            use_modal=False,
            lora=LoraHyperparams(r=4, alpha=8, dropout=0.0, target_modules=["c_attn"]),
        ),
        model=model,
        tokenizer=tok,
    )
    assert result.adapter_dir.is_dir()
    assert result.final_loss is not None
    assert result.final_loss == result.final_loss  # not NaN
    assert (tmp_path / "out" / "adapter").exists()
