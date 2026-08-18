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

# Qwen3 emits this before the content of a *final* assistant turn. It is part of
# every supervised span and is deliberately kept out of the loss — see
# `tokenize_messages`.
THINK_WRAPPER_TEXT = "<think>\n\n</think>\n\n"

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
    # Trailing newline to match the message header above: a generation boundary
    # that differs from the template's own turn boundary is exactly the
    # off-by-one-token shift these offline tests exist to catch.
    "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
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


def _unwrap_ids(encoded: Any) -> list[int]:
    """Return a bare list of token ids from whatever apply_chat_template gave us.

    `apply_chat_template(tokenize=True)` returns a plain list on some versions, a
    `BatchEncoding` on others, and a `tokenizers.Encoding` on transformers 5.x.
    `list()` on the first two yields *dict keys* / an opaque object, not ids — a
    short wrong sequence that every downstream check happily accepts, which is
    how a prefix assert becomes theatre. Unwrap explicitly and refuse anything
    that is not a list of ints.
    """
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif hasattr(encoded, "get") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if not isinstance(encoded, list):
        encoded = list(encoded)
    # A BatchEncoding of one row nests: [[id, id, ...]].
    if len(encoded) == 1 and isinstance(encoded[0], list):
        encoded = encoded[0]
    if encoded and not all(isinstance(i, int) for i in encoded):
        raise TypeError(
            "apply_chat_template did not yield token ids "
            f"(first element is {type(encoded[0]).__name__}) — unwrap is wrong "
            "for this transformers version, and a prefix assert over these "
            "values would pass without checking anything"
        )
    return encoded


def _encode_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    template_kwargs: dict[str, Any] | None = None,
    *,
    add_generation_prompt: bool = False,
) -> list[int]:
    """Tokenize a message prefix; empty prefix → empty ids.

    `template_kwargs` are forwarded to `apply_chat_template`. Qwen3's template
    reads `enable_thinking` from there; leaving it unset takes the template's
    own default, which is not the same thing as choosing it.
    """
    if not messages:
        return []
    extra = dict(template_kwargs or {})
    # `chat_template` rides in template_kwargs so a caller can render against a
    # template other than the tokenizer's own — which is how the same messages
    # can be masked our way and TRL's way and compared position by position.
    try:
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt, **extra
        )
    except Exception:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, **extra
        )
        encoded = tokenizer.encode(text, add_special_tokens=False)
    return _unwrap_ids(encoded)


class NonLastSupervisionError(ValueError):
    """A row supervises a message that is not the last one.

    Qwen3's chat template renders an assistant turn differently depending on
    whether it is last (see `tokenize_messages`), so a supervised span that is
    not the final message cannot be located in the full render. Such a row is
    refused, never masked approximately.
    """


class PrefixInstabilityError(RuntimeError):
    """`render(messages[:-1])` is not a token-prefix of `render(messages)`."""


def _think_wrapper_ids(tokenizer: Any) -> list[int]:
    """Token ids for the empty reasoning block Qwen3 emits on a final turn.

    Derived from the tokenizer rather than hardcoded — on Qwen3-14B this is
    `[151667, 271, 151668, 271]` (`<think>`, `\n\n`, `</think>`, `\n\n`) and the
    caller asserts against what the template actually produced, so a tokenizer
    that disagrees fails on CPU instead of training something else.
    """
    return _unwrap_ids(tokenizer.encode(THINK_WRAPPER_TEXT, add_special_tokens=False))


def tokenize_messages(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    supervise: list[bool],
    *,
    max_length: int = 4096,
    template_kwargs: dict[str, Any] | None = None,
    truncate: bool = True,
    mask_think_wrapper: bool = True,
) -> TokenizedExample | None:
    """Tokenize a row whose **last** message is its one supervised target.

    Two encodes, not a per-message loop, and the reason is a property of Qwen3's
    template rather than a style preference. That template wraps an assistant
    turn in `<think>\n\n</think>\n\n` only when the turn is `loop.last`;
    `enable_thinking` gates the *generation prompt* and nothing else. So
    `render(messages[:i+1])` is **not** a token-prefix of `render(messages)` for
    any non-final assistant `i` — it is four tokens longer. A loop that measures
    prefix lengths and indexes them into the full render therefore walks each
    supervised span four tokens past `<|im_end|>` and into the next user turn.
    That is not hypothetical: it is what the previous corpus trained on, for
    3,396 of its 3,561 actions, and neither the length-monotonicity guard nor
    the TRL cross-check (which tolerates <4% extra) noticed.

    The construction that cannot have the bug is a row whose only supervised
    message is the last one:

        prefix = render(messages[:-1], add_generation_prompt=True)
        full   = render(messages)
        assert prefix == full[:len(prefix)]

    History stays visible and unsupervised; nothing about it needs to be
    locatable in the full render.

    `mask_think_wrapper` (default on) additionally keeps the four wrapper tokens
    out of the loss. They stay in `input_ids`, so the target is still
    conditioned on `</think>\n\n`, but supervising them would teach "open the
    reasoning block and close it immediately" on every single row — which sets
    `enable_thinking=True` while training the behaviour out. Think *length*
    belongs to the instruction-tuned prior; this function supervises the action.
    See `docs/SFT-SCRATCH-PLAN.md`.

    `truncate=False` returns None instead of cutting an over-length example.
    Silently truncating is how a mask gets dropped without anyone noticing
    (TRL #3927); this path would rather lose the row and say so.

    Raises `NonLastSupervisionError` if anything but the final message is
    supervised, and `PrefixInstabilityError` if the two encodes disagree.
    """
    if len(supervise) != len(messages):
        raise ValueError(
            f"supervise has {len(supervise)} entries for {len(messages)} messages"
        )
    if not messages or not any(supervise):
        return None
    bad = [i for i, sup in enumerate(supervise) if sup and i != len(messages) - 1]
    if bad:
        raise NonLastSupervisionError(
            f"messages {bad} are supervised but are not the last message "
            f"(index {len(messages) - 1}). Qwen3's template renders a non-final "
            "assistant turn without its <think> wrapper, so such a span cannot "
            "be located in the full render. Split the row so each supervised "
            "action is its own final message."
        )
    if messages[-1].get("role") != "assistant":
        raise NonLastSupervisionError(
            f"the supervised final message has role {messages[-1].get('role')!r}, "
            "not 'assistant'"
        )

    full_ids = _encode_messages(tokenizer, messages, template_kwargs)
    if not full_ids:
        return None
    if len(full_ids) > max_length and not truncate:
        return None

    # `add_generation_prompt=True` is what puts the prefix exactly where the
    # model will be asked to generate at serving time. Under
    # `enable_thinking=True` that render ends at `<|im_start|>assistant\n`, so
    # the whole target span — wrapper included — falls inside `full_ids` after it.
    prefix_ids = _encode_messages(
        tokenizer, messages[:-1], template_kwargs, add_generation_prompt=True
    )
    if full_ids[: len(prefix_ids)] != prefix_ids:
        diverge = next(
            (j for j in range(min(len(prefix_ids), len(full_ids)))
             if prefix_ids[j] != full_ids[j]),
            min(len(prefix_ids), len(full_ids)),
        )
        raise PrefixInstabilityError(
            "render(messages[:-1], add_generation_prompt=True) is not a token "
            f"prefix of render(messages): they diverge at token {diverge} of "
            f"{len(prefix_ids)}. Refusing to build labels — this is the failure "
            "the two-encode construction exists to catch, not to work around."
        )

    start = len(prefix_ids)
    if mask_think_wrapper:
        wrapper = _think_wrapper_ids(tokenizer)
        if full_ids[start : start + len(wrapper)] != wrapper:
            raise PrefixInstabilityError(
                f"expected the reasoning wrapper {wrapper} at token {start}, "
                f"found {full_ids[start : start + len(wrapper)]}. The template "
                "no longer emits <think></think> on a final assistant turn; "
                "re-derive the mask rather than training through this."
            )
        start += len(wrapper)

    labels = [IGNORE_INDEX] * start + full_ids[start:]

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


def tokenize_from_ids(
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
    *,
    max_length: int = 4096,
    mask_prompt: bool = True,
) -> TokenizedExample | None:
    """Build a training example from *sampled* ids — no re-tokenization.

    Phase 0.5 (`docs/PILOT.md`): OPD and GRPO must supervise the tokens the
    student actually emitted at inference. Re-encoding the text is close but
    not identical (retokenization drift); this path refuses to invent ids.

    Loss lands on completion tokens only when `mask_prompt` is True (the
    default). Empty completions yield None.
    """
    if not completion_token_ids:
        return None
    prompt = [int(x) for x in prompt_token_ids]
    completion = [int(x) for x in completion_token_ids]
    input_ids = prompt + completion
    labels = (
        [IGNORE_INDEX] * len(prompt) + list(completion) if mask_prompt else list(input_ids)
    )
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


def tokenize_from_captures(
    captures: list[Any],
    *,
    max_length: int = 4096,
) -> list[TokenizedExample]:
    """One TokenizedExample per captured completion (assistant turn).

    Accepts `CapturedCompletion` instances or dicts with `prompt_token_ids` /
    `token_ids`. Skips empty completions. Does not stitch multi-turn into one
    sequence — each model call is its own training sample, matching how OPD
    scores per on-policy step.
    """
    out: list[TokenizedExample] = []
    for cap in captures:
        if hasattr(cap, "prompt_token_ids") and hasattr(cap, "token_ids"):
            prompt_ids = list(cap.prompt_token_ids)
            token_ids = list(cap.token_ids)
        elif isinstance(cap, dict):
            prompt_ids = list(cap.get("prompt_token_ids") or [])
            token_ids = list(cap.get("token_ids") or [])
        else:
            raise TypeError(
                f"capture must be CapturedCompletion or dict, got {type(cap).__name__}"
            )
        ex = tokenize_from_ids(
            prompt_ids, token_ids, max_length=max_length, mask_prompt=True
        )
        if ex is not None:
            out.append(ex)
    return out


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
    "tokenize_from_captures",
    "tokenize_from_ids",
    "tokenize_sft_example",
    "tokenize_teacher_continuation",
    "turns_to_messages",
]
