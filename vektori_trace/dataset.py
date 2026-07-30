"""Trajectory → tokenized SFT example with loss masking.

Correctness-critical (V0_PLAN.md Step 6): most tokens in agent trajectories are
environment observations. Training to predict those memorises this repo's test
output. Mask everything that is not a parent-agent assistant turn.

Decisions locked for v0 (stated up front, not chosen after seeing numbers):
- Allow-list: loss only on `role == "assistant"` AND `subagent_depth == 0`.
- `Turn.thinking` / reasoning_content: STRIP — do not put it in messages or
  the loss. Qwen3 has a reasoning channel, but agent ATIF thinking is not
  reliably aligned with that template; silent inclusion inside `content` is
  the failure mode this module exists to prevent.
- Do not use `Trace.condensed()` — it truncates and flattens roles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .schema import Turn

IGNORE_INDEX = -100

# Minimal chat template for offline tests (from-scratch tokenizers have none).
# Honours `add_generation_prompt` because the OPD loop depends on it: the student
# must be positioned to *open* an assistant turn, and a template that ignored the
# flag would silently have it continue the previous message instead
# (`distill.encode_prefix`).
TEST_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>\\n' }}"
    "{% if message.get('content') %}{{ message['content'] }}{% endif %}"
    "{% if message.get('tool_calls') %}"
    "{% for tc in message['tool_calls'] %}"
    "{{ '<|tool_call|>' + tc['function']['name'] + ' ' + tc['function']['arguments'] }}"
    "{% endfor %}"
    "{% endif %}"
    "{{ '<|end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)


def _require_train():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:  # pragma: no cover - env guard
        raise RuntimeError(
            "training extras required: install with `uv sync --extra train` "
            "(or `pip install 'vektori-trace[train]'`)"
        ) from e


def turns_to_messages(turns: list[Turn]) -> list[dict[str, Any]]:
    """Convert Turns to chat-template messages.

    Drops subagent-originated turns (depth > 0) and subagent marker systems.
    Strips `thinking` — never folded into `content`.
    """
    messages: list[dict[str, Any]] = []
    for t in turns:
        if t.subagent_depth > 0:
            continue
        if t.role == "system" and (t.content or "").startswith("[subagent"):
            continue
        msg: dict[str, Any] = {"role": t.role}
        if t.content is not None:
            msg["content"] = t.content
        elif t.tool_calls:
            msg["content"] = ""
        if t.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": (
                            tc.args
                            if isinstance(tc.args, str)
                            else json.dumps(tc.args or {})
                        ),
                    },
                }
                for tc in t.tool_calls
            ]
        if t.tool_call_id is not None:
            msg["tool_call_id"] = t.tool_call_id
        messages.append(msg)
    return messages


def _assistant_mask_for_messages(messages: list[dict[str, Any]]) -> list[bool]:
    """True iff this message's tokens should receive loss (parent assistant)."""
    return [m.get("role") == "assistant" for m in messages]


@dataclass
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


def _encode_messages(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    """Tokenize a message prefix; empty prefix → empty ids."""
    if not messages:
        return []
    try:
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
    except Exception:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = tokenizer.encode(text, add_special_tokens=False)
    # Some tokenizers return a BatchEncoding (dict-like: input_ids/attention_mask)
    # from apply_chat_template(tokenize=True) rather than a bare list. Iterating
    # a dict with list(...) silently yields its *keys*, not token ids — which
    # tokenize_sft_example can't distinguish from real (short, wrong) ids, so it
    # masks everything and returns None instead of erroring. Unwrap explicitly.
    if hasattr(encoded, "get") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if not isinstance(encoded, list):
        encoded = list(encoded)
    return encoded


def tokenize_sft_example(
    turns: list[Turn],
    tokenizer: Any,
    *,
    max_length: int = 4096,
    mask_prefix_messages: int = 0,
) -> TokenizedExample | None:
    """Tokenize a trajectory and mask non-assistant spans with IGNORE_INDEX.

    Uses prefix lengths of `apply_chat_template` (assumes a prefix-stable
    template, which HF chat templates are). Returns None if nothing trainable
    remains (no parent assistant turns).

    Teacher-continuation / ReOPD (PLAN.md): `mask_prefix_messages` masks the
    first N chat messages entirely (student prefix) — those tokens carry
    IGNORE_INDEX even if they are assistant turns. Continuation tokens keep loss.
    """
    _require_train()
    messages = turns_to_messages(turns)
    if not messages or not any(m.get("role") == "assistant" for m in messages):
        return None

    train_mask = _assistant_mask_for_messages(messages)
    if mask_prefix_messages < 0:
        raise ValueError(f"mask_prefix_messages must be >= 0, got {mask_prefix_messages}")
    if mask_prefix_messages:
        for i in range(min(mask_prefix_messages, len(train_mask))):
            train_mask[i] = False

    full_ids = _encode_messages(tokenizer, messages)
    if not full_ids:
        return None

    labels = [IGNORE_INDEX] * len(full_ids)
    prev_len = 0
    for i in range(len(messages)):
        prefix_ids = _encode_messages(tokenizer, messages[: i + 1])
        cur_len = len(prefix_ids)
        # Non-prefix-stable templates can shrink; refuse rather than mis-mask.
        if cur_len < prev_len:
            raise RuntimeError(
                "chat template is not prefix-stable — refusing to build labels "
                "(masking would be silently wrong)"
            )
        if train_mask[i]:
            # Align against full_ids: take the span [prev_len, cur_len).
            end = min(cur_len, len(full_ids))
            for j in range(prev_len, end):
                labels[j] = full_ids[j]
        prev_len = cur_len

    if len(full_ids) != prev_len:
        # Full encode disagreed with last prefix encode — don't guess.
        raise RuntimeError(
            "chat template prefix length != full length — refusing to build labels"
        )

    input_ids = full_ids
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    if not any(lab != IGNORE_INDEX for lab in labels):
        return None

    return TokenizedExample(
        input_ids=input_ids,
        labels=labels,
        attention_mask=[1] * len(input_ids),
    )


def tokenize_teacher_continuation(
    prefix_turns: list[Turn],
    continuation_turns: list[Turn],
    tokenizer: Any,
    *,
    max_length: int = 4096,
) -> TokenizedExample | None:
    """SFT example where the student prefix is masked; teacher continuation is not."""
    n_prefix_msgs = len(turns_to_messages(prefix_turns))
    return tokenize_sft_example(
        prefix_turns + continuation_turns,
        tokenizer,
        max_length=max_length,
        mask_prefix_messages=n_prefix_msgs,
    )


class LabelPreservingCollator:
    """Pad `labels` with IGNORE_INDEX. Stock DataCollatorForLanguageModeling
    regenerates labels from input_ids on pad and silently erases every mask."""

    def __init__(self, pad_token_id: int, *, label_pad_id: int = IGNORE_INDEX):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        _require_train()
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        batch_input, batch_labels, batch_attn = [], [], []
        for f in features:
            pad_n = max_len - len(f["input_ids"])
            batch_input.append(f["input_ids"] + [self.pad_token_id] * pad_n)
            batch_labels.append(f["labels"] + [self.label_pad_id] * pad_n)
            attn = f.get("attention_mask", [1] * len(f["input_ids"]))
            batch_attn.append(list(attn) + [0] * pad_n)
        return {
            "input_ids": torch.tensor(batch_input, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
        }


def build_sft_dataset(examples: list[TokenizedExample]) -> Any:
    """HF Dataset from already-tokenized examples. Lazy-imports datasets."""
    _require_train()
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "input_ids": [e.input_ids for e in examples],
            "labels": [e.labels for e in examples],
            "attention_mask": [e.attention_mask for e in examples],
        }
    )


__all__ = [
    "IGNORE_INDEX",
    "TEST_CHAT_TEMPLATE",
    "LabelPreservingCollator",
    "TokenizedExample",
    "build_sft_dataset",
    "tokenize_sft_example",
    "tokenize_teacher_continuation",
    "turns_to_messages",
]
