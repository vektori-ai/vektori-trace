"""Each trainer's inlined tokenizer must match `dataset.tokenize_messages` exactly.

`scripts/sft_repair_train_modal.py`, `scripts/sft_stage_a_train_modal.py` and
`scripts/sft_stage_b_train_modal.py` each carry a copy of the masking logic
because
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

from scripts import (  # noqa: E402
    sft_repair_train_modal,
    sft_stage_a_train_modal,
    sft_stage_b_train_modal,
)
from scripts.sft_repair_train_modal import tokenize_row  # noqa: E402
from vektori_trace.dataset import tokenize_messages  # noqa: E402

# Both copies, checked against the same source of truth. A second copy is a
# second chance to drift.
TRAINERS = [
    pytest.param(sft_repair_train_modal, id="repair"),
    pytest.param(sft_stage_a_train_modal, id="stage-a"),
    # Stage B inlines the same copy at a different MAX_LENGTH. Its own docstring
    # claims the fingerprint covers drift, but that check runs inside the Modal
    # container, after the image build and the 14B download are paid for — and a
    # drifted copy is what burned a repair run.
    pytest.param(sft_stage_b_train_modal, id="stage-b"),
]

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


# --------------------------------------------------------------------------
# Stage A's copy, and the constants that make it Stage A rather than the repair
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mod", TRAINERS)
@pytest.mark.parametrize("messages", ROWS)
def test_every_trainer_copy_matches_the_real_function(tok, mod, messages):
    supervise = [False] * (len(messages) - 1) + [True]
    ours = tokenize_messages(
        messages, tok, supervise, max_length=mod.MAX_LENGTH,
        template_kwargs=mod.TEMPLATE_KWARGS, truncate=False,
    )
    theirs = mod.tokenize_row({"messages": messages, "supervise": supervise}, tok)
    assert ours is not None and theirs is not None
    assert theirs["input_ids"] == ours.input_ids
    assert theirs["labels"] == ours.labels
    assert theirs["attention_mask"] == ours.attention_mask


@pytest.mark.parametrize("mod", TRAINERS)
def test_thinking_stays_on_in_every_trainer(mod):
    assert mod.TEMPLATE_KWARGS == {"enable_thinking": True}


def test_stage_a_trains_at_8192_not_the_repair_length():
    """Amendment 2: the 18 recoveries that need ~33k are Stage B's. A Stage A
    run at 40960 is the repair run's length and would silently readmit them."""
    assert sft_stage_a_train_modal.MAX_LENGTH == 8192
    assert sft_repair_train_modal.MAX_LENGTH == 40960


def test_stage_a_defaults_to_bf16_and_the_repair_run_to_nf4():
    """The 39.6 GiB peak that justifies --nf4 is NF4 @ 40960. Stage A is 8192,
    so bf16 is probed first and the arm is decided by measurement (step 5)."""
    import inspect

    def default(mod, name):
        return inspect.signature(mod.train.get_raw_f()).parameters[name].default

    assert default(sft_stage_a_train_modal, "nf4") is False
    assert default(sft_repair_train_modal, "nf4") is True


def test_stage_a_hyperparameters_match_the_plan():
    m = sft_stage_a_train_modal
    assert (m.LORA_R, m.LORA_ALPHA, m.LORA_DROPOUT) == (32, 64, 0.05)
    assert m.LORA_TARGET_MODULES == "all-linear"

    import inspect

    sig = inspect.signature(m.train.get_raw_f()).parameters
    assert sig["epochs"].default == 4.0
    assert sig["lr"].default == 1e-4
    assert sig["grad_accum"].default == 8
    assert sig["save_steps"].default == 10
    assert sig["seed"].default == 0


def test_stage_a_reads_its_own_dataset_and_writes_its_own_adapter():
    """Pointing Stage A at the repaired jsonl, or writing over the repaired
    adapter, are both silent ways to run a different experiment."""
    a, r = sft_stage_a_train_modal, sft_repair_train_modal
    assert a.DATA_IN_VOLUME == "sft-stage-a/stage_a.jsonl"
    assert a.OUT_IN_VOLUME not in (r.OUT_IN_VOLUME, r.V1_IN_VOLUME)
    assert a.DATA_IN_VOLUME != r.DATA_IN_VOLUME
    assert not hasattr(a, "V1_IN_VOLUME")


# --------------------------------------------------------------------------
# The did-it-move check, which the first bf16 probe died in
# --------------------------------------------------------------------------


def test_moved_check_survives_a_model_that_changed_device_mid_run():
    """The bf16 arm passes no device_map, so the snapshot is taken on the host
    and Trainer moves the model to the GPU afterwards. A bare `.clone()` plus
    `torch.equal` then compares cpu against cuda:0 and raises — which is what
    killed the first probe *after* training had already succeeded.

    No GPU here, so `.to()` is exercised against a meta-free CPU copy; what is
    actually asserted is that neither side is trusted to already be on the same
    device as the other.
    """
    torch = pytest.importorskip("torch")
    from scripts.sft_stage_a_train_modal import _count_moved, _snapshot_trainable

    net = torch.nn.Linear(4, 3)
    net.bias.requires_grad_(False)  # only `weight` is "trainable"

    before = _snapshot_trainable(net)
    assert set(before) == {"weight"}
    assert all(t.device.type == "cpu" for t in before.values())

    assert _count_moved(net, before) == 0
    with torch.no_grad():
        net.weight.add_(1.0)
    assert _count_moved(net, before) == 1


def test_snapshot_does_not_alias_the_live_parameter():
    """A snapshot that shares storage with the parameter compares equal no
    matter what training did, and the probe would report `moved == 0` on a run
    that trained perfectly well."""
    torch = pytest.importorskip("torch")
    from scripts.sft_stage_a_train_modal import _count_moved, _snapshot_trainable

    net = torch.nn.Linear(4, 3)
    before = _snapshot_trainable(net)
    with torch.no_grad():
        net.weight.mul_(2.0)
    assert _count_moved(net, before) >= 1
