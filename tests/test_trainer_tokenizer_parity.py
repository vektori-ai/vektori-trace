"""The trainer's inlined tokenizer must match `dataset.tokenize_messages` exactly.

`scripts/sft_repair_train_modal.py` carries a copy of the masking logic because
Modal re-imports the module inside a container where the local package is not
installed. A copy can drift, and this one did: the plan's step 1 rewrote
`tokenize_messages` while `tokenize_row` kept the per-message length loop. The
per-row sha256 fingerprint would have caught it at GPU time — after the image
build and the model download were already paid for, and only if nobody passed a
flag to skip it. Catch it on CPU instead.
"""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("modal")

from scripts.sft_repair_train_modal import TEMPLATE_KWARGS, tokenize_row  # noqa: E402
from vektori_trace.dataset import tokenize_messages  # noqa: E402

MODEL = "Qwen/Qwen3-14B"
ACTION = '{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls -la\\n", "duration": 0.1}]}'


@pytest.fixture(scope="module")
def tok():
    return transformers.AutoTokenizer.from_pretrained(MODEL)


def _u(c):
    return {"role": "user", "content": c}


def _a(c):
    return {"role": "assistant", "content": c}


ROWS = [
    pytest.param([_u("spec + task + terminal"), _a(ACTION)], id="turn-1"),
    pytest.param(
        [_u("spec"), _a("handoff answer prose"), _u("obs"), _a(ACTION)],
        id="handoff-head",
    ),
    pytest.param(
        [_u("spec"), _a("not json at all"), _u("parse error feedback"), _a(ACTION)],
        id="parse-error-recovery",
    ),
]


@pytest.mark.parametrize("messages", ROWS)
def test_trainer_copy_matches_the_real_function(tok, messages):
    supervise = [False] * (len(messages) - 1) + [True]
    ours = tokenize_messages(
        messages, tok, supervise, max_length=40960,
        template_kwargs=TEMPLATE_KWARGS, truncate=False,
    )
    theirs = tokenize_row({"messages": messages, "supervise": supervise}, tok)
    assert ours is not None and theirs is not None
    assert theirs["input_ids"] == ours.input_ids
    assert theirs["labels"] == ours.labels
    assert theirs["attention_mask"] == ours.attention_mask


def test_trainer_copy_refuses_a_packed_row(tok):
    """The shape the previous corpus used. Both sides must refuse it."""
    messages = [_u("spec"), _a(ACTION), _u("obs"), _a(ACTION)]
    with pytest.raises(SystemExit, match="not the last"):
        tokenize_row({"messages": messages, "supervise": [False, True, False, True]}, tok)


def test_trainer_copy_masks_the_think_wrapper(tok):
    """Not just "some prefix is masked" — the wrapper specifically."""
    from vektori_trace.dataset import IGNORE_INDEX, THINK_WRAPPER_TEXT

    messages = [_u("spec"), _a(ACTION)]
    row = tokenize_row({"messages": messages, "supervise": [False, True]}, tok)
    wrapper = tok.encode(THINK_WRAPPER_TEXT, add_special_tokens=False)
    first = next(i for i, lab in enumerate(row["labels"]) if lab != IGNORE_INDEX)
    assert row["input_ids"][first - len(wrapper) : first] == wrapper
    assert all(lab == IGNORE_INDEX for lab in row["labels"][first - len(wrapper) : first])
    assert tok.decode(row["input_ids"][first:]).startswith('{"analysis"')
