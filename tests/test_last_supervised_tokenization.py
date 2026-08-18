"""Step 1 of `docs/SFT-SCRATCH-PLAN.md`: the row shapes we accept and refuse.

These run against the real `Qwen/Qwen3-14B` tokenizer, not the offline stub,
because the property under test *is* a property of that template: a non-final
assistant turn loses its `<think>` wrapper, so per-message prefix encoding
overshoots. A stub template that is prefix-stable would pass every one of these
while the real one fails, which is exactly how the bug survived a green suite.
"""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")

from vektori_trace.dataset import (  # noqa: E402
    IGNORE_INDEX,
    THINK_WRAPPER_TEXT,
    NonLastSupervisionError,
    tokenize_messages,
)

MODEL = "Qwen/Qwen3-14B"
KW = {"enable_thinking": True}

ACTION = '{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls -la\\n", "duration": 0.1}]}'


@pytest.fixture(scope="module")
def tok():
    return transformers.AutoTokenizer.from_pretrained(MODEL)


def _u(c):
    return {"role": "user", "content": c}


def _a(c):
    return {"role": "assistant", "content": c}


def test_turn_one_row_is_accepted(tok):
    """`[user, assistant]` — production turn 1. The shape Stage A is built from."""
    ex = tokenize_messages(
        [_u("spec + task + terminal"), _a(ACTION)], tok, [False, True],
        max_length=8192, template_kwargs=KW, truncate=False,
    )
    assert ex is not None
    assert any(lab != IGNORE_INDEX for lab in ex.labels)


def test_middle_supervision_is_refused(tok):
    """`[u, a, u]` supervising the middle turn — the exact leak we measured.

    The old loop produced a span ending `...<|im_end|>\\n<|im_start|>user\\ntotal`.
    """
    with pytest.raises(NonLastSupervisionError):
        tokenize_messages(
            [_u("spec"), _a(ACTION), _u("total 40\ndrwxr-xr-x")], tok, [False, True, False],
            max_length=8192, template_kwargs=KW, truncate=False,
        )


def test_packed_multi_action_row_is_refused(tok):
    """The shape the previous corpus was built in: several supervised actions."""
    msgs = [_u("spec"), _a(ACTION), _u("obs"), _a(ACTION)]
    with pytest.raises(NonLastSupervisionError):
        tokenize_messages(
            msgs, tok, [False, True, False, True],
            max_length=8192, template_kwargs=KW, truncate=False,
        )


def test_handoff_head_then_target_is_accepted(tok):
    """`[u, a_handoff, u, a_target]` — the 48 post-compaction firsts.

    A loop-based equality assert would have refused these and silently cut
    Stage A from 183 rows to 117.
    """
    msgs = [_u("spec"), _a("handoff answer prose"), _u("obs"), _a(ACTION)]
    ex = tokenize_messages(
        msgs, tok, [False, False, False, True],
        max_length=8192, template_kwargs=KW, truncate=False,
    )
    assert ex is not None
    assert any(lab != IGNORE_INDEX for lab in ex.labels)


def test_supervised_span_is_the_action_and_stops_at_im_end(tok):
    """The span must not run past `<|im_end|>` into the following turn."""
    msgs = [_u("spec"), _a(ACTION)]
    ex = tokenize_messages(
        msgs, tok, [False, True], max_length=8192, template_kwargs=KW, truncate=False,
    )
    span = [i for i, lab in enumerate(ex.labels) if lab != IGNORE_INDEX]
    assert span == list(range(span[0], len(ex.labels))), "span must be contiguous to the end"
    text = tok.decode([ex.input_ids[i] for i in span])
    assert text.startswith('{"analysis"')
    assert "<|im_start|>" not in text


def test_think_wrapper_is_context_not_target(tok):
    """The wrapper is in `input_ids`, out of `labels`.

    Supervising it would teach an empty reasoning block on every row — thinking
    enabled by flag, trained away in the weights.
    """
    msgs = [_u("spec"), _a(ACTION)]
    ex = tokenize_messages(
        msgs, tok, [False, True], max_length=8192, template_kwargs=KW, truncate=False,
    )
    wrapper = tok.encode(THINK_WRAPPER_TEXT, add_special_tokens=False)
    first = next(i for i, lab in enumerate(ex.labels) if lab != IGNORE_INDEX)
    assert ex.input_ids[first - len(wrapper) : first] == wrapper
    assert all(lab == IGNORE_INDEX for lab in ex.labels[first - len(wrapper) : first])


def test_wrapper_is_supervised_when_the_mask_is_off(tok):
    """The documented rollback: `mask_think_wrapper=False` puts it back in."""
    msgs = [_u("spec"), _a(ACTION)]
    ex = tokenize_messages(
        msgs, tok, [False, True], max_length=8192, template_kwargs=KW,
        truncate=False, mask_think_wrapper=False,
    )
    first = next(i for i, lab in enumerate(ex.labels) if lab != IGNORE_INDEX)
    assert tok.decode(ex.input_ids[first:]).startswith("<think>")


def test_the_old_loop_would_have_overshot(tok):
    """Pin the template property itself, so a Qwen3 update cannot hide it.

    `render([u, a])` is not a prefix of `render([u, a, u])`: the wrapper is on
    the last assistant turn only.
    """
    from vektori_trace.dataset import _encode_messages

    short = _encode_messages(tok, [_u("spec"), _a(ACTION)], KW)
    long = _encode_messages(tok, [_u("spec"), _a(ACTION), _u("obs")], KW)
    assert long[: len(short)] != short
    wrapper = tok.encode(THINK_WRAPPER_TEXT, add_special_tokens=False)
    assert len(short) - len(wrapper) < len(long)
