"""OPD loop on a tiny from-scratch model with a fake teacher.

The loop under test is the one that runs on the GPU — only the model size and the
teacher's transport are swapped. What these assert is the part that cannot be
checked by reading the numbers off a real run: that the tokens the teacher scores
are exactly the tokens the student sampled, in order, and that the gradient
reaches the LoRA parameters.
"""

from __future__ import annotations

import json

import pytest

from vektori_trace.dataset import TEST_CHAT_TEMPLATE
from vektori_trace.distill import (
    OPDTrainConfig,
    encode_prefix,
    run_opd_training,
    write_opd_report,
)
from vektori_trace.reopd import build_reopd_example
from vektori_trace.schema import ToolCall, Turn
from vektori_trace.teacher import InMemoryIdScoringPool
from vektori_trace.train import LoraHyperparams

pytest.importorskip("torch")
pytest.importorskip("peft")


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
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tok.chat_template = TEST_CHAT_TEMPLATE
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel(
        GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=max(tok.vocab_size, 64), n_positions=512)
    )
    model = get_peft_model(
        model,
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, target_modules=["c_attn"]),
    )
    return model, tok


def _cfg(tmp_path, **kw):
    base = {
        "output_dir": tmp_path / "opd",
        "max_steps": 2,
        "examples_per_step": 1,
        "max_new_tokens": 6,
        "verify_tokenizers": False,  # the tiny model shares no vocab with the pilot pair
        "lora": LoraHyperparams(r=4, alpha=8, dropout=0.0, target_modules=["c_attn"]),
    }
    base.update(kw)
    return OPDTrainConfig(**base)


def test_encode_prefix_positions_the_student_to_open_a_turn(tiny):
    _, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)
    ids = encode_prefix(ex, tok, max_prefix_tokens=512)
    assert ids
    text = tok.decode(ids)
    # The prefix must end where an assistant turn begins, not mid-observation.
    assert text.rstrip().endswith("<|assistant|>")
    # The teacher's own action at this step must not leak into the prefix.
    assert "patching now" not in text


def test_encode_prefix_left_truncates_keeping_recent_context(tiny):
    _, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)
    full = encode_prefix(ex, tok, max_prefix_tokens=512)
    clipped = encode_prefix(ex, tok, max_prefix_tokens=5)
    assert len(clipped) == 5
    assert clipped == full[-5:]


def test_empty_prefix_is_refused(tiny):
    _, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=0)
    ex.prefix_turns = []
    with pytest.raises(ValueError, match="empty prefix"):
        encode_prefix(ex, tok, max_prefix_tokens=512)


def test_loop_trains_and_writes_an_adapter(tmp_path, tiny):
    model, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)
    pool = InMemoryIdScoringPool()

    result = run_opd_training([ex], pool, _cfg(tmp_path), model=model, tokenizer=tok)

    assert result.steps == 2
    assert result.final_loss is not None
    assert result.final_loss == result.final_loss  # not NaN
    assert result.adapter_dir.is_dir()
    assert (result.adapter_dir / "adapter_config.json").is_file()
    assert result.action_tokens_scored > 0
    assert pool.score_calls == 2
    assert result.provenance["loss"] == "reverse_kl_surrogate"


def test_teacher_scores_exactly_the_tokens_the_student_sampled(tmp_path, tiny):
    """The alignment that, if wrong, corrupts the objective without failing."""
    model, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)
    pool = InMemoryIdScoringPool()

    sampled: list[list[int]] = []
    import vektori_trace.distill as distill_mod

    original = distill_mod._sample_action

    def spy(m, t, prefix_ids, cfg):
        ids = original(m, t, prefix_ids, cfg)
        sampled.append(list(ids))
        return ids

    distill_mod._sample_action = spy
    try:
        run_opd_training(
            [ex], pool, _cfg(tmp_path, max_steps=1), model=model, tokenizer=tok
        )
    finally:
        distill_mod._sample_action = original

    assert sampled, "the loop never sampled"
    # Same ids, same order — not merely the same count.
    assert pool.last_tokens == sampled[-1]
    # And the prefix the teacher conditioned on is the encoded prefix, untruncated.
    assert pool.last_prompt_ids == encode_prefix(ex, tok, max_prefix_tokens=3584)


def test_gradient_reaches_the_lora_parameters(tmp_path, tiny):
    import torch

    model, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)
    before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad and "lora" in n
    }
    assert before, "no LoRA parameters to check"

    run_opd_training(
        [ex],
        InMemoryIdScoringPool(),
        _cfg(tmp_path, max_steps=3, learning_rate=1e-2),
        model=model,
        tokenizer=tok,
    )

    moved = [
        n
        for n, p in model.named_parameters()
        if n in before and not torch.allclose(before[n], p.detach())
    ]
    assert moved, "LoRA weights did not move — the objective is not connected"


def test_log_records_loss_and_monitoring_ratio_per_step(tmp_path, tiny):
    model, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)
    result = run_opd_training(
        [ex], InMemoryIdScoringPool(), _cfg(tmp_path, max_steps=2), model=model, tokenizer=tok
    )
    rows = [json.loads(line) for line in result.log_path.read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert "loss" in row
        assert "mean_log_ratio" in row
        assert row["action_tokens"] > 0
        assert row["lr"] > 0


def test_teacher_length_disagreement_is_fatal(tmp_path, tiny):
    """A teacher returning the wrong count must stop the run, not be truncated."""
    model, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)

    class _Short(InMemoryIdScoringPool):
        def score_ids(self, prompt_ids, tokens):
            return super().score_ids(prompt_ids, tokens)[:-1] or [-0.5]

    with pytest.raises(RuntimeError, match="logprobs for"):
        run_opd_training(
            [ex], _Short(), _cfg(tmp_path, max_new_tokens=8), model=model, tokenizer=tok
        )


def test_examples_per_step_accumulates_before_stepping(tmp_path, tiny):
    model, tok = tiny
    turns = _turns()
    examples = [
        build_reopd_example(turns, task="t", action_index=0),
        build_reopd_example(turns, task="t", action_index=1),
    ]
    pool = InMemoryIdScoringPool()
    result = run_opd_training(
        examples, pool, _cfg(tmp_path, max_steps=2, examples_per_step=2), model=model, tokenizer=tok
    )
    # Two optimizer steps, two examples each → four teacher round-trips.
    assert pool.score_calls == 4
    assert result.steps == 2


def test_no_examples_is_refused(tmp_path, tiny):
    model, tok = tiny
    with pytest.raises(ValueError, match="no ReOPD examples"):
        run_opd_training([], InMemoryIdScoringPool(), _cfg(tmp_path), model=model, tokenizer=tok)


def test_topk_objective_trains_and_is_recorded_as_a_different_loss(tmp_path, tiny):
    """The thunlp/OPD-style analytic path: top-K KL instead of a sampled estimate."""
    model, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)
    pool = InMemoryIdScoringPool()

    result = run_opd_training(
        [ex], pool, _cfg(tmp_path, top_k=4), model=model, tokenizer=tok
    )

    assert result.steps == 2
    assert result.final_loss is not None
    # A KL between two proper distributions is non-negative; the sampled-token
    # surrogate has no such guarantee, so this also distinguishes the two paths.
    assert result.final_loss >= -1e-6
    assert result.provenance["loss"] == "topk_reverse_kl"
    assert result.provenance["top_k"] == 4


def test_align_topk_rows_keeps_the_sampled_token_and_drops_the_least_probable():
    from vektori_trace.distill import align_topk_rows

    # Sampled token 7 is the *worst* entry — padding logic that ranked by logprob
    # alone would drop the very token the student sampled.
    rows = [{1: -0.1, 2: -0.2, 3: -5.0, 7: -9.0}]
    ids, lps = align_topk_rows(rows, [7], 3)
    assert ids[0][0] == 7  # sampled token first, always retained
    assert set(ids[0]) == {7, 1, 2}  # -5.0 dropped as least probable of the rest
    assert lps[0][0] == -9.0


def test_align_topk_rows_refuses_a_short_row():
    from vektori_trace.distill import align_topk_rows

    with pytest.raises(RuntimeError, match="fewer than top_k"):
        align_topk_rows([{7: -1.0, 1: -2.0}], [7], 4)


def test_topk_reverse_kl_is_zero_when_the_distributions_agree():
    import torch

    from vektori_trace.opd import topk_reverse_kl

    # Student logits that put a known distribution on ids 0 and 1.
    logits = torch.tensor([[[2.0, 1.0, -5.0]]])
    ids = torch.tensor([[[0, 1]]])
    student_lp = torch.log_softmax(logits, dim=-1)[0, 0, :2]
    # Teacher identical over the same set → KL 0.
    teacher = student_lp.detach().reshape(1, 1, 2)
    loss = topk_reverse_kl(logits, ids, teacher)
    assert abs(float(loss)) < 1e-5

    # Strictly positive when they differ. Note the comparison is over the
    # *renormalised* K set, so a teacher differing only by a constant offset
    # (here, the same 1.0 logit gap) is the same distribution and still scores 0 —
    # which is why this uses a uniform teacher rather than a shifted one.
    uniform = torch.tensor([[[-1.0, -1.0]]])
    assert float(topk_reverse_kl(logits, ids, uniform)) > 0
    shifted = torch.tensor([[[-1.0, -2.0]]])  # same gap as `logits` over {0,1}
    assert abs(float(topk_reverse_kl(logits, ids, shifted))) < 1e-5


def test_report_states_the_teacher_and_the_monitoring_scalar(tmp_path, tiny):
    model, tok = tiny
    ex = build_reopd_example(_turns(), task="t", action_index=1)
    result = run_opd_training(
        [ex], InMemoryIdScoringPool(), _cfg(tmp_path), model=model, tokenizer=tok
    )
    md_path = write_opd_report(result, tmp_path / "out")
    payload = json.loads((tmp_path / "out" / "opd.json").read_text())
    assert payload["provenance"]["teacher_model"] == "in-memory"
    assert payload["action_tokens_scored"] == result.action_tokens_scored
    assert "mean log ratio" in md_path.read_text()
