"""§6.4 student training proof — a mocked two-turn batch, end to end.

`docs/OPD-MULTITURN-PLAN.md` §6.4 lists six things to prove before any Harbor or
GPU work. This file proves each one on a *real* (tiny) model with a *real* LoRA
adapter and a *real* optimizer step, on CPU:

  1. rollout captures ck75 token ids and behaviour log probabilities;
  2. the later ck75 prefix contains the earlier Harbor observation;
  3. environment and template tokens have zero loss weight;
  4. the student recomputes current log probabilities with autograd enabled;
  5. a nonzero finite gradient reaches ck75 trainable parameters;
  6. exactly one optimizer step changes and reloads the adapter.

Deliberately a real model rather than a stub. A mocked `backward()` proves the
test harness works; it does not prove a gradient reaches LoRA parameters, which
is the thing that has to be true before spending on a GPU. `tiny-random-gpt2`
makes that affordable — the point is the plumbing, not the model.

What stays mocked is the *teacher*: §6.3 already proved DeepSeek scoring against
the live endpoint, so re-paying for it here would add nothing. Teacher logprobs
are frozen numbers with a deliberately different tokenisation from the student's,
so the cross-tokenizer chunk path is genuinely exercised.
"""

from __future__ import annotations

import math

import pytest

from vektori_trace.chunk_opd import clipped_is_policy_loss
from vektori_trace.opd_rollout import (
    RolloutError,
    TrajectoryRecord,
    TurnRecord,
    advantages_for_turn,
    assert_observation_carried,
    assert_single_policy_version,
    global_supervised_token_count,
)

torch = pytest.importorskip("torch", reason="§6.4 proof needs the train extra")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

TINY = "hf-internal-testing/tiny-random-gpt2"
V0 = "ck75-v0"


# ---------------------------------------------------------------------------
# A two-turn rollout whose turn 2 is conditioned on turn 1's observation
# ---------------------------------------------------------------------------


def _tokenize_bytes(text: str, chunk: int) -> tuple[list[bytes], list[int]]:
    """Split text into fixed-size byte chunks — a stand-in tokenisation.

    Using two different `chunk` values for student and teacher is what makes the
    alignment non-trivial (1:N, N:1, M:N chunks), which is the case the semantic
    prior exists for.
    """
    raw = text.encode()
    toks = [raw[i : i + chunk] for i in range(0, len(raw), chunk)]
    return toks, list(range(1, len(toks) + 1))


@pytest.fixture
def trajectory() -> TrajectoryRecord:
    """Two turns; turn 2's prefix quotes turn 1's Harbor output verbatim."""
    obs1 = "src/hatch/resolver.py\nsrc/hatch/cli.py"

    a1 = '{"cmd": "find /workspace -name \'*.py\'"}'
    a2 = '{"cmd": "sed -n \'1,40p\' src/hatch/resolver.py"}'

    b1, ids1 = _tokenize_bytes(a1, 4)
    b2, ids2 = _tokenize_bytes(a2, 4)

    t1 = TurnRecord(
        turn_index=0,
        student_prefix_text="<|system|>agent<|user|>find the resolver<|assistant|>",
        action_bytes=a1.encode(),
        action_token_ids=ids1,
        action_token_bytes=b1,
        behavior_logprobs=[-0.4 - 0.01 * i for i in range(len(ids1))],
        observation=obs1,
        policy_version=V0,
    )
    t2 = TurnRecord(
        turn_index=1,
        # The observation is carried forward verbatim — this is what makes the
        # trajectory state distribution student-owned.
        student_prefix_text=(
            "<|system|>agent<|user|>find the resolver<|assistant|>"
            + a1
            + "<|user|>"
            + obs1
            + "<|assistant|>"
        ),
        action_bytes=a2.encode(),
        action_token_ids=ids2,
        action_token_bytes=b2,
        behavior_logprobs=[-0.6 - 0.01 * i for i in range(len(ids2))],
        observation="def resolve(...):",
        policy_version=V0,
        termination_reason="natural",
    )
    return TrajectoryRecord(task="pypa__hatch-2086", turns=[t1, t2], policy_version=V0)


def _teacher_for(turn: TurnRecord) -> tuple[list[bytes], list[float]]:
    """A teacher tokenisation of the same bytes, at a different granularity."""
    toks, _ = _tokenize_bytes(turn.action_bytes.decode(), 3)
    # Deterministic but varied; the teacher is mildly more confident than the
    # student, so advantages should skew positive.
    lps = [-0.2 - 0.03 * (i % 7) for i in range(len(toks))]
    return toks, lps


# ---------------------------------------------------------------------------
# §6.4.1 — rollout captures ids and behaviour log probabilities
# ---------------------------------------------------------------------------


def test_rollout_captures_ids_and_behavior_logprobs(trajectory):
    for turn in trajectory.turns:
        assert turn.n_action_tokens > 0
        assert len(turn.behavior_logprobs) == turn.n_action_tokens
        assert all(math.isfinite(v) for v in turn.behavior_logprobs)
        # The ids reconstruct exactly the bytes that were executed.
        assert b"".join(turn.action_token_bytes) == turn.action_bytes
        assert turn.policy_version == V0


def test_token_bytes_disagreeing_with_the_action_is_refused():
    """§11: executed action differs from the recorded action."""
    with pytest.raises(RolloutError, match="not the bytes that were executed"):
        TurnRecord(
            turn_index=0,
            student_prefix_text="p",
            action_bytes=b"hello",
            action_token_ids=[1, 2],
            action_token_bytes=[b"hel", b"LO"],
            behavior_logprobs=[-0.1, -0.2],
            observation="",
            policy_version=V0,
        )


def test_missing_behavior_logprobs_are_refused():
    with pytest.raises(RolloutError, match="behaviour logprobs"):
        TurnRecord(
            turn_index=0,
            student_prefix_text="p",
            action_bytes=b"ab",
            action_token_ids=[1, 2],
            action_token_bytes=[b"a", b"b"],
            behavior_logprobs=[-0.1],
            observation="",
            policy_version=V0,
        )


def test_empty_action_is_refused():
    with pytest.raises(RolloutError, match="no action tokens"):
        TurnRecord(
            turn_index=0,
            student_prefix_text="p",
            action_bytes=b"",
            action_token_ids=[],
            action_token_bytes=[],
            behavior_logprobs=[],
            observation="",
            policy_version=V0,
        )


def test_mixed_policy_versions_are_refused(trajectory):
    """§7.4: one frozen version for the whole batch."""
    other = TrajectoryRecord(
        task="t2", turns=[trajectory.turns[0]], policy_version=V0
    )
    assert assert_single_policy_version([trajectory, other]) == V0

    rogue = TurnRecord(
        turn_index=0,
        student_prefix_text="p",
        action_bytes=b"ab",
        action_token_ids=[1, 2],
        action_token_bytes=[b"a", b"b"],
        behavior_logprobs=[-0.1, -0.2],
        observation="",
        policy_version="ck75-v1",
    )
    v1 = TrajectoryRecord(task="t3", turns=[rogue], policy_version="ck75-v1")
    with pytest.raises(RolloutError, match="mixes policy versions"):
        assert_single_policy_version([trajectory, v1])


# ---------------------------------------------------------------------------
# §6.4.2 — the later prefix contains the earlier observation
# ---------------------------------------------------------------------------


def test_later_prefix_contains_the_earlier_observation(trajectory):
    assert_observation_carried(trajectory)  # must not raise
    assert trajectory.turns[0].observation in trajectory.turns[1].student_prefix_text


def test_independent_turns_are_caught(trajectory):
    """Two single-turn rollouts must not pass as a multi-turn trajectory."""
    broken = TrajectoryRecord(
        task=trajectory.task,
        turns=[
            trajectory.turns[0],
            TurnRecord(
                turn_index=1,
                student_prefix_text="<|system|>agent<|user|>unrelated<|assistant|>",
                action_bytes=trajectory.turns[1].action_bytes,
                action_token_ids=trajectory.turns[1].action_token_ids,
                action_token_bytes=trajectory.turns[1].action_token_bytes,
                behavior_logprobs=trajectory.turns[1].behavior_logprobs,
                observation="",
                policy_version=V0,
            ),
        ],
        policy_version=V0,
    )
    with pytest.raises(RolloutError, match="not a multi-turn trajectory"):
        assert_observation_carried(broken)


# ---------------------------------------------------------------------------
# §6.4.3 — environment and template tokens carry zero loss weight
# ---------------------------------------------------------------------------


def test_only_sampled_action_tokens_are_supervised(trajectory):
    """The prefix is never handed to the loss, so it cannot be supervised.

    This is structural, not a mask that must be remembered: `TurnRecord` keeps
    prefix text and action ids in separate fields, and only the action ids reach
    `advantages_for_turn`.
    """
    for turn in trajectory.turns:
        tb, tl = _teacher_for(turn)
        adv = advantages_for_turn(turn, tb, tl)
        assert len(adv.advantages) == turn.n_action_tokens
        assert len(adv.supervised_mask) == turn.n_action_tokens
        # Nothing from the prefix could have entered: the counts match the
        # action exactly.
        assert adv.stats.n_supervised_tokens <= turn.n_action_tokens


def test_teacher_scoring_different_bytes_is_refused(trajectory):
    turn = trajectory.turns[0]
    with pytest.raises(RolloutError, match="different bytes than were executed"):
        advantages_for_turn(turn, [b"totally", b"other"], [-0.1, -0.2])


# ---------------------------------------------------------------------------
# The cross-tokenizer seam: alignment -> advantages
# ---------------------------------------------------------------------------


def test_seam_produces_finite_advantages_across_both_turns(trajectory):
    batches = []
    for turn in trajectory.turns:
        tb, tl = _teacher_for(turn)
        adv = advantages_for_turn(turn, tb, tl)
        assert all(math.isfinite(a) for a in adv.advantages)
        assert adv.stats.n_chunks > 0
        batches.append(adv)

    total = global_supervised_token_count(batches)
    assert total == sum(b.n_supervised for b in batches)
    assert total > 0
    # 4-byte student vs 3-byte teacher tokens: the alignment must not be all 1:1,
    # or the chunk path is not actually being exercised.
    assert any(b.stats.exact_1to1_token_fraction < 1.0 for b in batches)


# ---------------------------------------------------------------------------
# §6.4.4-6 — autograd, gradient to LoRA params, exactly one optimizer step
# ---------------------------------------------------------------------------


@pytest.fixture
def lora_model():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    try:
        base = AutoModelForCausalLM.from_pretrained(TINY)
    except Exception as e:  # pragma: no cover - offline / no cache
        pytest.skip(f"tiny model unavailable: {e}")
    model = get_peft_model(
        base,
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=0.0),
    )
    model.train()
    return model


def _current_logprobs(model, ids: list[int]):
    """log pi_current for each supplied id, with autograd live (§6.4.4)."""
    x = torch.tensor([ids])
    out = model(input_ids=x)
    # Next-token convention: logits at position t predict token t+1.
    logits = out.logits[:, :-1, :].float()
    targets = x[:, 1:]
    lp = torch.log_softmax(logits, dim=-1)
    return lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).squeeze(0), targets.shape[1]


def test_one_optimizer_step_moves_lora_and_gradients_are_finite(
    trajectory, lora_model
):
    """The whole §6.4 chain, on a real model: forward, loss, backward, step."""
    model = lora_model
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "LoRA attached no trainable parameters"

    before = [p.detach().clone() for p in trainable]

    # --- build detached advantages for every turn (teacher side is frozen data)
    batches = []
    for turn in trajectory.turns:
        tb, tl = _teacher_for(turn)
        batches.append(advantages_for_turn(turn, tb, tl))

    denom = global_supervised_token_count(batches)
    assert denom > 0

    # --- accumulate the globally normalised loss over both turns (§7.3 step 4)
    vocab = model.config.vocab_size
    total_loss = torch.tensor(0.0)
    for adv in batches:
        ids = [i % vocab for i in adv.action_token_ids]
        if len(ids) < 2:
            continue
        cur, n = _current_logprobs(model, ids)
        assert cur.requires_grad, "§6.4.4: current logprobs must carry autograd"

        beh = torch.tensor(adv.behavior_logprobs[:n], dtype=torch.float32)
        a = torch.tensor(adv.advantages[:n], dtype=torch.float32)
        mask = torch.tensor(
            [1.0 if s else 0.0 for s in adv.supervised_mask[:n]], dtype=torch.float32
        )
        # denominator=1.0 -> raw sum; one global division below.
        total_loss = total_loss + clipped_is_policy_loss(
            cur, beh, a, mask, denominator=1.0
        )

    loss = total_loss / denom
    assert torch.isfinite(loss), "§7.4: loss must be finite"

    loss.backward()

    # --- §6.4.5: a nonzero finite gradient reached trainable parameters
    grads = [p.grad for p in trainable if p.grad is not None]
    assert grads, "no LoRA parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads), "gradient is identically zero"

    # --- §6.4.6: exactly one optimizer step, and it moves the adapter
    opt = torch.optim.AdamW(trainable, lr=1e-2)
    opt.step()
    opt.zero_grad()

    moved = [
        not torch.equal(b, p.detach()) for b, p in zip(before, trainable, strict=True)
    ]
    assert any(moved), "one optimizer step did not change any LoRA parameter"


def test_adapter_round_trips_through_disk(trajectory, lora_model, tmp_path):
    """§6.4.6: the changed adapter can be saved and reloaded (publish v1)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = lora_model
    trainable = [p for p in model.parameters() if p.requires_grad]
    with torch.no_grad():
        for p in trainable:
            p.add_(torch.full_like(p, 0.01))

    out = tmp_path / "v1"
    model.save_pretrained(out)
    assert (out / "adapter_config.json").is_file()
    assert any(out.glob("adapter_model.*"))

    base = AutoModelForCausalLM.from_pretrained(TINY)
    reloaded = PeftModel.from_pretrained(base, out)
    got = {
        n: p for n, p in reloaded.named_parameters() if "lora" in n.lower()
    }
    assert got, "reloaded adapter exposes no LoRA parameters"
    assert all(torch.isfinite(p).all() for p in got.values())


def test_fully_masked_turn_contributes_no_gradient(lora_model):
    """A turn whose every position is a sentinel must not move the weights."""
    model = lora_model
    trainable = [p for p in model.parameters() if p.requires_grad]
    ids = [5, 6, 7, 8]

    cur, n = _current_logprobs(model, ids)
    beh = torch.full((n,), -0.5)
    a = torch.zeros(n)
    mask = torch.zeros(n)

    loss = clipped_is_policy_loss(cur, beh, a, mask)
    loss.backward()

    assert float(loss.detach()) == 0.0
    total = sum(
        float(p.grad.abs().sum()) for p in trainable if p.grad is not None
    )
    assert total == pytest.approx(0.0, abs=1e-9)
