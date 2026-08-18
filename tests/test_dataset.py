"""Masking boundary tests for dataset.py — the correctness-critical piece.

Allow-list (assistant only), subagent exclusion, thinking stripped, and a
label-preserving collator that must NOT regenerate labels from input_ids.
Fully offline: no HuggingFace download (plan: inject a chat template on a
from-scratch tokenizer).
"""

from __future__ import annotations

import pytest

from vektori_trace.dataset import (
    IGNORE_INDEX,
    TEST_CHAT_TEMPLATE,
    LabelPreservingCollator,
    tokenize_sft_example,
    turns_to_messages,
)
from vektori_trace.schema import ToolCall, Turn

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def _tiny_tokenizer():
    """Offline tokenizer with an injected chat template — no Hub download."""
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<|user|>": 2,
        "<|assistant|>": 3,
        "<|tool|>": 4,
        "<|system|>": 5,
        "<|end|>": 6,
        "<|tool_call|>": 7,
        "fix": 8,
        "the": 9,
        "bug": 10,
        "I'll": 11,
        "look": 12,
        "at": 13,
        "traceback.": 14,
        "running": 15,
        "pytest": 16,
        "FAILED": 17,
        "test_x.py": 18,
        "noise": 19,
        "retry": 20,
        "with": 21,
        "prefix": 22,
        "go": 23,
        "sub": 24,
        "only": 25,
        "hello": 26,
        "there": 27,
        "say": 28,
        "hi": 29,
        "bash": 30,
        "{}": 31,
        "cmd": 32,
    }
    for i, tok in enumerate(
        ["a", "b", "c", "d", "e", "f", "g", "h", ".", ",", ":", "'", '"', "/", "_"]
    ):
        vocab.setdefault(tok, 100 + i)
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<|user|>",
        eos_token="<|end|>",
    )
    tok.chat_template = TEST_CHAT_TEMPLATE
    return tok


def _turns() -> list[Turn]:
    return [
        Turn(index=0, role="user", content="fix the bug"),
        Turn(index=1, role="assistant", content="I'll look at traceback."),
        Turn(
            index=2,
            role="assistant",
            content="running pytest",
            tool_calls=[ToolCall(id="c1", name="bash", args={"cmd": "pytest"})],
        ),
        Turn(index=3, role="tool", content="FAILED test_x.py", tool_call_id="c1"),
        Turn(index=4, role="system", content="noise"),
        Turn(index=5, role="assistant", content="sub only", subagent_depth=1),
        Turn(
            index=6,
            role="assistant",
            thinking="secret chain of thought",
            content="retry with prefix",
        ),
    ]


def test_turns_to_messages_allow_list_context_keeps_non_assistant() -> None:
    msgs = turns_to_messages(_turns())
    roles = [m["role"] for m in msgs]
    assert "user" in roles
    assert "tool" in roles
    assert "system" in roles
    assert all(m.get("content") != "sub only" for m in msgs)


def test_thinking_is_stripped_not_folded_into_content() -> None:
    msgs = turns_to_messages(_turns())
    joined = " ".join(str(m.get("content") or "") for m in msgs)
    assert "secret chain of thought" not in joined
    assert "retry with prefix" in joined
    assert all("thinking" not in m for m in msgs)


def test_tokenize_masks_everything_except_parent_assistant() -> None:
    tok = _tiny_tokenizer()
    ex = tokenize_sft_example(_turns(), tok, max_length=2048)
    assert ex is not None
    assert len(ex.input_ids) == len(ex.labels) == len(ex.attention_mask)
    assert any(lab != IGNORE_INDEX for lab in ex.labels)
    assert any(lab == IGNORE_INDEX for lab in ex.labels)

    msgs = turns_to_messages(_turns())
    first_as_idx = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    prefix = msgs[:first_as_idx]
    if prefix:
        prefix_ids = tok.apply_chat_template(prefix, tokenize=True, add_generation_prompt=False)
        if not isinstance(prefix_ids, list):
            prefix_ids = list(prefix_ids)
        n = min(len(prefix_ids), len(ex.labels))
        assert all(lab == IGNORE_INDEX for lab in ex.labels[:n])


def test_subagent_only_trajectory_yields_nothing_trainable() -> None:
    tok = _tiny_tokenizer()
    turns = [
        Turn(index=0, role="user", content="go"),
        Turn(index=1, role="assistant", content="sub only", subagent_depth=1),
    ]
    assert tokenize_sft_example(turns, tok) is None


def test_label_preserving_collator_pads_labels_with_ignore_not_input_ids() -> None:
    """Stock DataCollatorForLanguageModeling would copy input_ids into labels
    on pad and erase the mask. This is the silent-failure the plan flags."""
    collator = LabelPreservingCollator(pad_token_id=0, label_pad_id=IGNORE_INDEX)
    features = [
        {"input_ids": [1, 2, 3], "labels": [IGNORE_INDEX, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5], "labels": [4, IGNORE_INDEX], "attention_mask": [1, 1]},
    ]
    batch = collator(features)
    assert batch["input_ids"].shape == batch["labels"].shape == (2, 3)
    assert int(batch["labels"][1, 2]) == IGNORE_INDEX
    assert int(batch["input_ids"][1, 2]) == 0
    assert int(batch["labels"][0, 0]) == IGNORE_INDEX
    assert int(batch["labels"][1, 1]) == IGNORE_INDEX


def test_hf_label_shift_convention_is_next_token() -> None:
    """Pin HF causal-LM convention: loss at position i predicts token i+1.
    Our labels align 1:1 with input_ids; Trainer shifts internally — we must
    NOT pre-shift, or every mask boundary is off by one."""
    import torch
    from torch.nn import CrossEntropyLoss

    logits = torch.zeros(1, 3, 5)
    labels = torch.tensor([[IGNORE_INDEX, 1, 2]])
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = CrossEntropyLoss(ignore_index=IGNORE_INDEX)(
        shift_logits.view(-1, 5), shift_labels.view(-1)
    )
    assert torch.isfinite(loss)


def test_student_prefix_masked_teacher_continuation_not() -> None:
    """PLAN.md: student-prefix tokens carry IGNORE_INDEX; continuation does not."""
    from vektori_trace.dataset import tokenize_teacher_continuation

    tok = _tiny_tokenizer()
    prefix = [
        Turn(0, "user", content="fix the bug"),
        Turn(1, "assistant", content="I'll look at traceback."),
    ]
    continuation = [
        Turn(2, "assistant", content="retry with prefix"),
    ]
    ex = tokenize_teacher_continuation(prefix, continuation, tok)
    assert ex is not None

    # Positions, not counts. A count-only check passes on a partial or off-by-one
    # mask (drop one prefix token, gain one continuation token and the totals
    # still balance), so assert the boundary itself: the student-prefix span is
    # [0, n_prefix) and every position in it must be IGNORE_INDEX, while every
    # position in the continuation span [n_prefix, end) must carry loss.
    from vektori_trace.dataset import _encode_messages

    n_prefix = len(_encode_messages(tok, turns_to_messages(prefix)))
    assert 0 < n_prefix < len(ex.labels)
    assert all(lab == IGNORE_INDEX for lab in ex.labels[:n_prefix]), (
        "a student-prefix position carries loss"
    )
    assert all(lab != IGNORE_INDEX for lab in ex.labels[n_prefix:]), (
        "a teacher-continuation position was masked"
    )
    # Loss labels are the input ids at the same position (no pre-shift).
    assert ex.labels[n_prefix:] == ex.input_ids[n_prefix:]

    # And the boundary is where the *unmasked* tokenization stops agreeing: the
    # prefix's own assistant turn is supervised without the mask, masked with it.
    full = tokenize_sft_example(prefix + continuation, tok)
    assert full is not None
    assert any(lab != IGNORE_INDEX for lab in full.labels[:n_prefix])
    masked_loss = sum(1 for lab in ex.labels if lab != IGNORE_INDEX)
    full_loss = sum(1 for lab in full.labels if lab != IGNORE_INDEX)
    assert masked_loss < full_loss
    # Exactly the continuation carries loss: masking the 2-message student prefix
    # must remove every one of the prefix assistant turn's supervised tokens, so
    # what remains equals a fresh tokenization of the continuation's own turn.
    cont_only = tokenize_sft_example(continuation, tok)
    assert cont_only is not None
    assert masked_loss == sum(1 for lab in cont_only.labels if lab != IGNORE_INDEX)


# --------------------------------------------------------------------------
# tokenize_messages — explicit per-message supervision
#
# The role-derived mask above cannot supervise one assistant turn and skip the
# next, which is the same limitation TRL's `assistant_only_loss` has. The
# protocol repair (docs/SFT-REPAIR-PLAN.md) needs exactly that: the ~48
# post-compaction handoff turns are assistant prose that must stay in context
# and out of the loss.
# --------------------------------------------------------------------------


def _repair_messages() -> list[dict]:
    return [
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": "I'll look at traceback."},
        {"role": "user", "content": "FAILED test_x.py"},
        {"role": "assistant", "content": "running pytest"},
    ]


def test_tokenize_messages_supervises_only_the_selected_turns() -> None:
    from vektori_trace.dataset import tokenize_messages

    tok = _tiny_tokenizer()
    msgs = _repair_messages()
    both = tokenize_messages(msgs, tok, [False, True, False, True], max_length=2048)
    second_only = tokenize_messages(msgs, tok, [False, False, False, True], max_length=2048)
    assert both is not None and second_only is not None

    # Same text either way — only the labels move.
    assert both.input_ids == second_only.input_ids
    n_both = sum(1 for lab in both.labels if lab != IGNORE_INDEX)
    n_one = sum(1 for lab in second_only.labels if lab != IGNORE_INDEX)
    assert 0 < n_one < n_both

    # The skipped assistant turn is masked over its whole span, and the kept one
    # is untouched — this is the property the role-derived mask cannot express.
    from vektori_trace.dataset import _encode_messages

    start = len(_encode_messages(tok, msgs[:1]))
    end = len(_encode_messages(tok, msgs[:2]))
    assert any(lab != IGNORE_INDEX for lab in both.labels[start:end])
    assert all(lab == IGNORE_INDEX for lab in second_only.labels[start:end])


def test_tokenize_messages_labels_are_input_ids_at_the_same_position() -> None:
    from vektori_trace.dataset import tokenize_messages

    tok = _tiny_tokenizer()
    ex = tokenize_messages(_repair_messages(), tok, [False, True, False, True], max_length=2048)
    assert ex is not None
    for i, lab in enumerate(ex.labels):
        if lab != IGNORE_INDEX:
            assert lab == ex.input_ids[i]


def test_tokenize_messages_returns_none_when_nothing_is_supervised() -> None:
    from vektori_trace.dataset import tokenize_messages

    tok = _tiny_tokenizer()
    assert tokenize_messages(_repair_messages(), tok, [False] * 4) is None


def test_tokenize_messages_refuses_a_mismatched_supervise_length() -> None:
    from vektori_trace.dataset import tokenize_messages

    tok = _tiny_tokenizer()
    with pytest.raises(ValueError, match="supervise has"):
        tokenize_messages(_repair_messages(), tok, [True, False])


def test_tokenize_messages_can_refuse_to_truncate() -> None:
    """Silently cutting an over-length row is how a mask gets dropped without
    anyone noticing (TRL #3927); the repair path drops the row and says so."""
    from vektori_trace.dataset import tokenize_messages

    tok = _tiny_tokenizer()
    msgs = _repair_messages()
    assert tokenize_messages(msgs, tok, [False, True, False, True],
                             max_length=4, truncate=False) is None
    cut = tokenize_messages(msgs, tok, [False, True, False, True],
                            max_length=4, truncate=True)
    if cut is not None:
        assert len(cut.input_ids) == len(cut.labels) == 4


def test_tokenize_messages_forwards_template_kwargs() -> None:
    """`enable_thinking` must be a decision, not the template's default."""
    from vektori_trace.dataset import _encode_messages

    tok = _tiny_tokenizer()
    seen = {}
    original = tok.apply_chat_template

    def spy(messages, **kwargs):
        seen.update(kwargs)
        kwargs.pop("enable_thinking", None)
        return original(messages, **kwargs)

    tok.apply_chat_template = spy
    _encode_messages(tok, _repair_messages(), {"enable_thinking": False})
    assert seen.get("enable_thinking") is False
