"""The scoring bridge and the optimizer step (plan §8.2, §8.3, §7.3).

The optimizer half runs on a real tiny model with a real LoRA adapter and a real
`AdamW.step()`. A mocked backward would prove the harness works, not that a
gradient reaches LoRA parameters — which is the property that has to hold before
a paid run.

The scoring half is tested against the pinned DeepSeek renderer and tokenizer,
with the teacher's *transport* stubbed: §6.3 already proved that against the
live endpoint, and re-paying for it here would add nothing.
"""

from __future__ import annotations

import pytest

from vektori_trace.replay_opd import SampledAction, build_replay_batch
from vektori_trace.replay_score import (
    ScoringError,
    score_action,
    score_replay_batch,
)
from vektori_trace.replay_select import ReplayPrefix
from vektori_trace.replay_train import (
    ReplayTrainConfig,
    ReplayTrainError,
    build_optimizer,
    current_logprobs,
    make_optimizer_step,
)

torch = pytest.importorskip("torch", reason="the step proof needs the train extra")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

TINY = "hf-internal-testing/tiny-random-gpt2"
V0 = "ck75-v0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prefix(task: str, trace: str, step: int = 2) -> ReplayPrefix:
    return ReplayPrefix(task=task, trace_id=trace, step_index=step, prefix_turns=[])


def _split(text: str, chunk: int) -> list[bytes]:
    raw = text.encode()
    return [raw[i : i + chunk] for i in range(0, len(raw), chunk)]


#: A Qwen-plausible tokenisation, written out rather than fixed-width split.
#: DeepSeek emits {"/cmd/":/ "/ls/ -/la/ /works/pace/"} for this string; a
#: fixed-width student split shares almost no boundary with it and collapses the
#: whole action into one 5:11 chunk, which the threshold-6 sentinel then drops —
#: correct behaviour that silently yields zero supervised tokens. A fixture must
#: not depend on it.
STUDENT_TOKENS = [b'{"', b"cmd", b'":', b' "ls', b" -la", b" /work", b"space", b'"}']


def _action(prefix, i, text=None) -> SampledAction:
    if text is None:
        toks = list(STUDENT_TOKENS)
        text = b"".join(toks).decode()
    else:
        toks = _split(text, 4)
    return SampledAction(
        prefix_id=prefix.prefix_id,
        sample_index=i,
        action_bytes=text.encode(),
        action_token_ids=list(range(2, len(toks) + 2)),
        action_token_bytes=toks,
        behavior_logprobs=[-0.5 - 0.01 * k for k in range(len(toks))],
        policy_version=V0,
        prompt_token_ids=[11, 12, 13, 14, 15],
    )


class FakePool:
    """`score_ids` with the real contract: one logprob per supplied token."""

    def __init__(self, value=-0.3, n_override=None):
        self.value = value
        self.n_override = n_override
        self.calls = []

    def score_ids(self, prompt_ids, tokens):
        self.calls.append((len(prompt_ids), len(tokens)))
        n = self.n_override if self.n_override is not None else len(tokens)
        return [self.value - 0.01 * i for i in range(n)]


@pytest.fixture(scope="module")
def teacher_tokenizer():
    from vektori_trace.vocab_bridge import load_tokenizer

    try:
        return load_tokenizer("deepseek-ai/DeepSeek-V4-Flash-0731")
    except Exception as e:  # pragma: no cover - offline
        pytest.skip(f"teacher tokenizer unavailable: {e}")


MESSAGES = [
    {"role": "system", "content": "You are a terminal agent."},
    {"role": "user", "content": "List the workspace."},
]


# ---------------------------------------------------------------------------
# Scoring bridge (§4 rendering contract)
# ---------------------------------------------------------------------------


def test_action_span_is_located_by_joint_tokenisation(teacher_tokenizer):
    from vektori_trace.replay_score import locate_action_span

    action = '{"cmd": "ls -la /workspace"}'
    prefix_ids, action_ids, dropped = locate_action_span(
        MESSAGES, action, teacher_tokenizer
    )

    assert prefix_ids and action_ids
    # The renderer closes the turn with EOS; ck75 never sampled it.
    assert dropped >= 1
    decoded = teacher_tokenizer.decode(action_ids, skip_special_tokens=False)
    assert decoded == action


def test_scored_span_excludes_the_turn_terminator(teacher_tokenizer):
    """Decoding with specials hidden would make an EOS-carrying span look exact."""
    from vektori_trace.replay_score import locate_action_span

    action = '{"cmd": "pytest -x"}'
    _pre, action_ids, _d = locate_action_span(MESSAGES, action, teacher_tokenizer)

    visible = teacher_tokenizer.decode(action_ids, skip_special_tokens=False)
    assert visible == action
    assert "end" not in visible.lower() or visible == action


def test_score_action_returns_bytes_and_logprobs(teacher_tokenizer):
    p = _prefix("t", "tr0")
    a = _action(p, 0)
    pool = FakePool()

    s = score_action(a, MESSAGES, teacher_tokenizer, pool)

    assert s.key == a.key
    assert len(s.teacher_logprobs) == s.n_teacher_tokens
    assert b"".join(s.teacher_token_bytes) == a.action_bytes
    assert s.n_prefix_tokens > 0
    # One request carrying prefix + action, exactly as §3 describes.
    assert len(pool.calls) == 1


def test_teacher_returning_the_wrong_count_is_refused(teacher_tokenizer):
    p = _prefix("t", "tr0")
    a = _action(p, 0)
    with pytest.raises(ScoringError, match="logprobs for"):
        score_action(a, MESSAGES, teacher_tokenizer, FakePool(n_override=3))


def test_non_finite_teacher_logprob_is_refused(teacher_tokenizer):
    class BadPool:
        def score_ids(self, prompt_ids, tokens):
            return [-0.3] * (len(tokens) - 1) + [float("-inf")]

    p = _prefix("t", "tr0")
    with pytest.raises(ScoringError, match="non-finite"):
        score_action(_action(p, 0), MESSAGES, teacher_tokenizer, BadPool())


def test_batch_scoring_produces_the_pairs_build_replay_batch_wants(teacher_tokenizer):
    prefixes = [_prefix(f"task{i}", f"tr{i}") for i in range(8)]
    actions = [_action(p, i) for p in prefixes for i in range(4)]
    msgs = {p.prefix_id: MESSAGES for p in prefixes}

    scored, ledger = score_replay_batch(actions, msgs, teacher_tokenizer, FakePool())

    assert set(scored) == {a.key for a in actions}
    # Per-trace share is 1/8 here; the default 35% cap is aimed at a real
    # batch where action lengths vary, so relax it for uniform fixtures.
    batch = build_replay_batch(prefixes, actions, scored, max_trace_share=0.5)
    assert batch.global_supervised_tokens > 0


def test_ledger_reports_repeated_prefix_cost(teacher_tokenizer):
    """§8.4: each prefix is re-sent once per sample, and that dominates."""
    prefixes = [_prefix(f"task{i}", f"tr{i}") for i in range(2)]
    actions = [_action(p, i) for p in prefixes for i in range(4)]
    msgs = {p.prefix_id: MESSAGES for p in prefixes}

    _scored, ledger = score_replay_batch(actions, msgs, teacher_tokenizer, FakePool())

    assert ledger["n_actions"] == 8
    assert ledger["n_teacher_requests"] == 8
    # 4 samples per prefix -> three of every four prefix sends are repeats.
    assert ledger["repeated_prefix_tokens"] == pytest.approx(
        ledger["unique_prefix_tokens"] * 3, rel=0.01
    )


def test_missing_prefix_render_is_refused(teacher_tokenizer):
    p = _prefix("t", "tr0")
    with pytest.raises(ScoringError, match="no rendered prefix"):
        score_replay_batch([_action(p, 0)], {}, teacher_tokenizer, FakePool())


# ---------------------------------------------------------------------------
# Optimizer step — real model, real LoRA, real step
# ---------------------------------------------------------------------------


@pytest.fixture
def lora_model():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    try:
        base = AutoModelForCausalLM.from_pretrained(TINY)
    except Exception as e:  # pragma: no cover - offline
        pytest.skip(f"tiny model unavailable: {e}")
    return get_peft_model(
        base,
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=0.0),
    )


def _batch(model, n_prefixes=8, n_per=4):
    """A scored, aligned batch whose ids are valid for the tiny model."""
    from vektori_trace.replay_score import score_replay_batch

    vocab = model.config.vocab_size
    prefixes = [_prefix(f"task{i}", f"tr{i}") for i in range(n_prefixes)]
    actions = []
    for p in prefixes:
        for i in range(n_per):
            a = _action(p, i)
            a.action_token_ids = [(t * 7 + 3) % vocab for t in a.action_token_ids]
            a.prompt_token_ids = [(t * 5 + 1) % vocab for t in range(6)]
            actions.append(a)
    # Teacher side: a different granularity, so the chunk path is exercised.
    scored = {}
    for a in actions:
        tb = _split(a.action_bytes.decode(), 3)
        scored[a.key] = (tb, [-0.3 - 0.02 * (k % 5) for k in range(len(tb))])
    return build_replay_batch(prefixes, actions, scored)


def test_one_step_moves_lora_and_reports_it(lora_model, tmp_path):
    cfg = ReplayTrainConfig(output_dir=tmp_path / "v_replay")
    opt = build_optimizer(lora_model, cfg)
    step = make_optimizer_step(lora_model, opt, cfg, save=False)

    batch = _batch(lora_model)
    rep = step(batch)

    assert rep["optimizer_steps"] == 1
    assert rep["lora_tensors_moved"] > 0
    assert rep["max_param_delta"] > 0
    assert rep["global_supervised_tokens"] == batch.global_supervised_tokens
    assert rep["supervised_positions_scored"] > 0
    import math

    assert math.isfinite(rep["loss"])


def test_current_logprobs_carry_autograd(lora_model):
    ids = [3, 4, 5, 6]
    lp = current_logprobs(lora_model, ids)
    assert lp.requires_grad
    assert lp.shape[0] == len(ids) - 1, "first id is context under next-token"


def test_saving_writes_an_adapter_that_reloads(lora_model, tmp_path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    out = tmp_path / "v_replay"
    cfg = ReplayTrainConfig(output_dir=out)
    opt = build_optimizer(lora_model, cfg)
    step = make_optimizer_step(lora_model, opt, cfg, save=True)

    rep = step(_batch(lora_model))
    assert rep["adapter_saved_to"] == str(out)
    assert (out / "adapter_config.json").is_file()

    base = AutoModelForCausalLM.from_pretrained(TINY)
    reloaded = PeftModel.from_pretrained(base, out)
    assert any("lora" in n.lower() for n, _ in reloaded.named_parameters())


def test_output_dir_equal_to_v0_is_refused(tmp_path):
    """§8.3: v0 must not be overwritten."""
    with pytest.raises(ReplayTrainError, match="must not be overwritten"):
        ReplayTrainConfig(adapter_path=str(tmp_path / "v0"), output_dir=tmp_path / "v0")


def test_a_batch_with_no_supervised_tokens_is_refused(lora_model, tmp_path):
    cfg = ReplayTrainConfig(output_dir=tmp_path / "v")
    opt = build_optimizer(lora_model, cfg)
    step = make_optimizer_step(lora_model, opt, cfg, save=False)

    batch = _batch(lora_model, n_prefixes=8, n_per=1)
    for adv in batch.advantages:
        adv.supervised_mask = [False] * len(adv.supervised_mask)

    # global_supervised_tokens goes to zero first, which is the earlier and
    # more informative refusal.
    with pytest.raises(ReplayTrainError, match="no supervised tokens"):
        step(batch)


def test_model_without_trainable_params_is_refused(tmp_path):
    from transformers import AutoModelForCausalLM

    try:
        base = AutoModelForCausalLM.from_pretrained(TINY)
    except Exception as e:  # pragma: no cover
        pytest.skip(str(e))
    for p in base.parameters():
        p.requires_grad_(False)

    cfg = ReplayTrainConfig(output_dir=tmp_path / "v")

    class _Opt:
        def zero_grad(self, set_to_none=True):
            pass

        def step(self):
            pass

    step = make_optimizer_step(base, _Opt(), cfg, save=False)
    with pytest.raises(ReplayTrainError, match="no trainable parameters"):
        step(_batch(base))


def test_gradient_is_globally_normalised_not_per_example(lora_model, tmp_path):
    """§7.3: one denominator across the batch.

    Two batches with the same total supervised tokens but different example
    counts must produce the same gradient scale. A per-example mean would not.
    """
    cfg = ReplayTrainConfig(output_dir=tmp_path / "v", max_grad_norm=None)

    def grad_norm_for(n_prefixes, n_per):
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM

        m = get_peft_model(
            AutoModelForCausalLM.from_pretrained(TINY),
            LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=0.0),
        )
        torch.manual_seed(0)
        opt = build_optimizer(m, cfg)
        params = [p for _n, p in m.named_parameters() if p.requires_grad]
        before = [p.detach().clone() for p in params]
        make_optimizer_step(m, opt, cfg, save=False)(_batch(m, n_prefixes, n_per))
        return max(
            float((p.detach() - b).abs().max())
            for b, p in zip(before, params, strict=True)
        )

    # Same 32 actions, arranged 8x4 and 16x2.
    a = grad_norm_for(8, 4)
    b = grad_norm_for(16, 2)
    assert a == pytest.approx(b, rel=0.35), (
        "the update scale must follow total supervised tokens, not example count"
    )


# ---------------------------------------------------------------------------
# Prefix conditioning — log pi_current must match log pi_old's conditioning
# ---------------------------------------------------------------------------


def test_every_action_token_is_scored_including_the_first(lora_model):
    """1:1 with the captured behaviour logprobs — no token dropped as context."""
    from vektori_trace.replay_train import action_logprobs_under_prefix

    prompt = [5, 6, 7, 8]
    action = [9, 10, 11]
    lp = action_logprobs_under_prefix(lora_model, prompt, action)

    assert lp.shape[0] == len(action), "all action tokens, first included"
    assert lp.requires_grad


def test_changing_the_prefix_changes_the_first_action_logprob(lora_model):
    """The regression test for the conditioning bug.

    log pi_old was captured after the full replay prefix. If log pi_current is
    computed without it, the importance ratio compares two different
    distributions — finitely, so nothing downstream reveals it. A different
    prefix must therefore produce a different first-token score.
    """
    from vektori_trace.replay_train import action_logprobs_under_prefix

    action = [9, 10, 11]
    a = action_logprobs_under_prefix(lora_model, [5, 6, 7, 8], action)
    b = action_logprobs_under_prefix(lora_model, [21, 22, 23, 24], action)

    assert float(a[0]) != pytest.approx(float(b[0]), abs=1e-9), (
        "the first action token is scored from the final prefix logit, so it "
        "must move when the prefix changes"
    )


def test_scoring_the_action_alone_differs_from_scoring_it_under_its_prefix(
    lora_model,
):
    """The old, wrong path and the correct one must not agree.

    If these matched, the bug would have been unobservable and this test
    worthless — so the assertion is that they differ.
    """
    from vektori_trace.replay_train import (
        action_logprobs_under_prefix,
        current_logprobs,
    )

    prompt = [5, 6, 7, 8]
    action = [9, 10, 11]

    correct = action_logprobs_under_prefix(lora_model, prompt, action)
    # What the buggy version did: score the action with no prefix at all, which
    # also silently drops the first token.
    naive = current_logprobs(lora_model, action)

    assert correct.shape[0] == len(action)
    assert naive.shape[0] == len(action) - 1
    assert float(correct[1]) != pytest.approx(float(naive[0]), abs=1e-9)


def test_missing_prompt_ids_are_refused(lora_model, tmp_path):
    """A batch whose actions lack prefixes cannot be trained on."""
    cfg = ReplayTrainConfig(output_dir=tmp_path / "v")
    opt = build_optimizer(lora_model, cfg)
    step = make_optimizer_step(lora_model, opt, cfg, save=False)

    batch = _batch(lora_model, n_prefixes=8, n_per=1)
    for adv in batch.advantages:
        adv.prompt_token_ids = None

    with pytest.raises(ReplayTrainError, match="no prompt_token_ids"):
        step(batch)


def test_action_logprobs_refuse_an_empty_prefix(lora_model):
    from vektori_trace.replay_train import action_logprobs_under_prefix

    with pytest.raises(ReplayTrainError, match="no prompt_token_ids"):
        action_logprobs_under_prefix(lora_model, [], [9, 10])


class TestLoadV0Placement:
    """`load_v0_for_training` refuses to load 14B weights without a real GPU.

    The bug this guards is silent, not loud: `from_pretrained` with no
    `device_map` puts the model on CPU and `torch.tensor(..., device=None)`
    agrees, so the run trains — correctly, and far too slowly to finish. These
    tests assert the refusal happens *before* any weight load, which is why
    they can run on a CPU-only machine with a nonexistent model id.
    """

    def _cfg(self, **kw):
        from vektori_trace.replay_train import ReplayTrainConfig

        return ReplayTrainConfig(
            base_model="does-not-exist/never-loaded",
            adapter_path="/nonexistent/adapter",
            **kw,
        )

    def test_unset_device_is_refused(self):
        import pytest

        from vektori_trace.replay_train import ReplayTrainError, load_v0_for_training

        with pytest.raises(ReplayTrainError, match="device is unset"):
            load_v0_for_training(self._cfg())

    def test_cpu_device_is_refused(self):
        import pytest

        from vektori_trace.replay_train import ReplayTrainError, load_v0_for_training

        with pytest.raises(ReplayTrainError, match="not a CUDA device"):
            load_v0_for_training(self._cfg(device="cpu"))

    def test_missing_adapter_still_refused_first(self):
        """adapter_path is checked before placement; order must not regress."""
        import pytest

        from vektori_trace.replay_train import ReplayTrainConfig, ReplayTrainError, load_v0_for_training

        cfg = ReplayTrainConfig(base_model="x", adapter_path=None, device="cuda")
        with pytest.raises(ReplayTrainError, match="adapter_path is required"):
            load_v0_for_training(cfg)

    def test_config_stays_constructible_for_cpu_tests(self):
        """The tiny CPU tests build a config and call make_optimizer_step."""
        cfg = self._cfg()
        assert cfg.device is None


class TestAdapterReloadable:
    """§8.4: v_replay must differ from v0 *and be reloadable*.

    "Differs" is proven from live tensors in the step; reloadability is a
    property of the bytes on disk. The two diverge exactly in the quiet cases —
    a config with no weights beside it, a truncated write — where the in-memory
    model is fine and nothing else in the run would notice.
    """

    def _good(self, tmp_path):
        import torch
        from safetensors.torch import save_file

        d = tmp_path / "v_replay"
        d.mkdir()
        (d / "adapter_config.json").write_text(
            '{"peft_type": "LORA", "r": 32, "lora_alpha": 64, '
            '"target_modules": ["q_proj"]}'
        )
        save_file({"base.lora_A.weight": torch.zeros(2, 2)},
                  str(d / "adapter_model.safetensors"))
        return d

    def test_good_adapter_reports_shape(self, tmp_path):
        from vektori_trace.replay_train import verify_adapter_reloadable

        got = verify_adapter_reloadable(self._good(tmp_path))
        assert got["peft_type"] == "LORA"
        assert got["r"] == 32
        assert got["n_tensors"] == 1
        assert got["weights_bytes"] > 0

    def test_config_without_weights_refused(self, tmp_path):
        """The failure a `moved > 0` check cannot see."""
        import pytest

        from vektori_trace.replay_train import ReplayTrainError, verify_adapter_reloadable

        d = tmp_path / "v_replay"
        d.mkdir()
        (d / "adapter_config.json").write_text("{}")
        with pytest.raises(ReplayTrainError, match="no adapter weights file"):
            verify_adapter_reloadable(d)

    def test_zero_byte_weights_refused(self, tmp_path):
        import pytest

        from vektori_trace.replay_train import ReplayTrainError, verify_adapter_reloadable

        d = tmp_path / "v_replay"
        d.mkdir()
        (d / "adapter_config.json").write_text("{}")
        (d / "adapter_model.safetensors").write_bytes(b"")
        with pytest.raises(ReplayTrainError, match="zero bytes"):
            verify_adapter_reloadable(d)

    def test_corrupt_weights_refused(self, tmp_path):
        import pytest

        from vektori_trace.replay_train import ReplayTrainError, verify_adapter_reloadable

        d = tmp_path / "v_replay"
        d.mkdir()
        (d / "adapter_config.json").write_text("{}")
        (d / "adapter_model.safetensors").write_bytes(b"not a safetensors file")
        with pytest.raises(ReplayTrainError, match="unreadable"):
            verify_adapter_reloadable(d)

    def test_missing_config_refused(self, tmp_path):
        import pytest

        from vektori_trace.replay_train import ReplayTrainError, verify_adapter_reloadable

        d = tmp_path / "v_replay"
        d.mkdir()
        with pytest.raises(ReplayTrainError, match="adapter_config.json is missing"):
            verify_adapter_reloadable(d)


class TestLogitsToKeep:
    """`logits_to_keep` must be a pure memory optimisation, not a value change.

    It slices hidden states before `lm_head`, so the full-sequence logit tensor
    is never allocated — 11.5 GiB down to 2.6 at a 31.6k prefix with a
    9,216-token action. That only helps if the scores are unchanged, which is
    what these pin.
    """

    def _tiny(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        torch.manual_seed(0)
        cfg = AutoConfig.for_model(
            "qwen3", vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16,
        )
        return AutoModelForCausalLM.from_config(cfg).eval()

    def _reference(self, model, prompt, action):
        """The pre-optimisation path: full logits, then slice."""
        import torch

        x = torch.tensor([list(prompt) + list(action)])
        with torch.no_grad():
            out = model(input_ids=x)
        logits = out.logits[0, len(prompt) - 1 : -1, :].float()
        targets = x[0, len(prompt):]
        return torch.log_softmax(logits, -1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)

    def test_scores_match_the_full_logit_path(self):
        import torch

        from vektori_trace.replay_train import action_logprobs_under_prefix

        model = self._tiny()
        prompt, action = list(range(5, 45)), [7, 11, 23, 4, 99]
        got = action_logprobs_under_prefix(model, prompt, action)
        want = self._reference(model, prompt, action)
        assert got.shape == want.shape == (len(action),)
        assert torch.allclose(got, want, atol=1e-6)

    def test_single_token_action(self):
        """n_action+1 = 2 kept positions; the off-by-one is easiest to get
        wrong at length 1."""
        import torch

        from vektori_trace.replay_train import action_logprobs_under_prefix

        model = self._tiny()
        prompt, action = list(range(5, 45)), [42]
        got = action_logprobs_under_prefix(model, prompt, action)
        assert got.shape == (1,)
        assert torch.allclose(got, self._reference(model, prompt, action), atol=1e-6)

    def test_gradient_still_reaches_parameters(self):
        import torch

        from vektori_trace.replay_train import action_logprobs_under_prefix

        model = self._tiny()
        model.train()
        out = action_logprobs_under_prefix(model, list(range(5, 45)), [7, 11, 23])
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)

    def test_config_defaults_to_checkpointing(self):
        from vektori_trace.replay_train import ReplayTrainConfig

        assert ReplayTrainConfig().gradient_checkpointing is True
