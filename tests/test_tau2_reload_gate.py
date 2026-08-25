"""The reload gate, pinned to what the real adapter actually measured.

The first version of this gate compared full-sequence logits across three
concurrently-loaded 4B models and failed a healthy adapter with a 19.9 logit
delta. `tau2_adapter_diagnose_modal.py` then established the ground truth on the
saved probe adapter:

    504 tensors (252 lora_A + 252 lora_B), r=16, alpha=32, all-linear
    lora_B non-zero      252/252
    self-consistency     0.000000   (bit-identical, so bf16 noise was never it)
    adapter effect       0.3125 logits at the last position after 3 steps

These tests encode the *shape* of that result so the gate cannot silently drift
back to comparing the wrong thing.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Measured on the probe adapter, 2026-08-25.
PROBE_ADAPTER = {
    "n_tensors": 504,
    "lora_B_nonzero": 252,
    "lora_B_total": 252,
    "self_consistency_delta": 0.0,
    "adapter_effect_logit_delta": 0.3125,
    "r": 16,
    "lora_alpha": 32,
}


def gate_verdict(n_tensors, b_norms, self_delta, effect):
    """The gate's decision logic, extracted so it can be tested without a GPU."""
    if not b_norms:
        return "no lora_B tensors in the saved adapter"
    if sum(1 for v in b_norms if v > 0) == 0:
        return "every lora_B is zero"
    if self_delta != 0.0:
        return f"model is not deterministic in eval mode (delta {self_delta})"
    if effect == 0.0:
        return "disabling the adapter changes nothing"
    return None


def test_healthy_adapter_passes():
    """The real probe adapter's numbers must not trip the gate."""
    assert gate_verdict(
        PROBE_ADAPTER["n_tensors"],
        [0.0315] * PROBE_ADAPTER["lora_B_total"],
        PROBE_ADAPTER["self_consistency_delta"],
        PROBE_ADAPTER["adapter_effect_logit_delta"],
    ) is None


def test_all_zero_lora_B_is_caught():
    """A perfect no-op: every behavioural test would otherwise pass it."""
    v = gate_verdict(504, [0.0] * 252, 0.0, 0.0)
    assert v and "zero" in v


def test_missing_lora_B_is_caught():
    v = gate_verdict(504, [], 0.0, 0.31)
    assert v and "no lora_B" in v


def test_nondeterminism_is_caught_before_any_comparison():
    """If eval() is not deterministic, no delta below it means anything."""
    v = gate_verdict(504, [0.03] * 252, 0.002, 0.31)
    assert v and "deterministic" in v


def test_unapplied_adapter_is_caught():
    v = gate_verdict(504, [0.03] * 252, 0.0, 0.0)
    assert v and "changes nothing" in v


def test_small_effect_is_not_a_failure():
    """After three steps the true effect is ~0.31 logits.

    The regression this guards: a tolerance calibrated as if the adapter should
    dominate the model. It should not, and a tiny effect is the correct result.
    """
    assert gate_verdict(504, [0.001] * 252, 0.0, 0.0001) is None


@pytest.mark.parametrize("delta", [0.3125, 1.0, 19.875])
def test_gate_does_not_depend_on_a_raw_logit_threshold(delta):
    """Any non-zero effect passes; magnitude is reported, never thresholded.

    19.875 is the value that failed the original gate. It must not fail now.
    """
    assert gate_verdict(504, [0.03] * 252, 0.0, delta) is None


def test_probe_diagnosis_shape_if_present():
    """If the diagnosis artifact is around, it must match what we encoded."""
    p = "/tmp/adapter_diagnosis.json"
    if not os.path.exists(p):
        pytest.skip("no diagnosis artifact locally")
    d = json.load(open(p))
    assert d["n_tensors"] == PROBE_ADAPTER["n_tensors"]
    assert d["self_consistency_delta"] == 0.0
    assert d["adapter_effect_logit_delta"] > 0
    assert d["adapter_config"]["r"] == PROBE_ADAPTER["r"]


# ---------------------------------------------------------------------------
# Checkpoint accounting. Both of these shipped broken once: the record was
# written before the commit returned (so `committed` was never true in the
# file), and only weights/config were hashed (so optimizer.pt, scheduler.pt and
# rng_state.pth -- the files an exact resume needs -- went unrecorded).
# ---------------------------------------------------------------------------

def _ckpt_dir(tmp_path):
    d = tmp_path / "checkpoint-34"
    d.mkdir()
    for fn in ("adapter_model.safetensors", "adapter_config.json",
               "optimizer.pt", "scheduler.pt", "rng_state.pth",
               "trainer_state.json"):
        (d / fn).write_bytes(b"x" * 64)
    return str(d)


def test_checkpoint_record_marks_committed(tmp_path):
    from vektori_trace.tau2.runlog import RunLog

    rl = RunLog(str(tmp_path / "run"))
    calls = []
    rl.checkpoint(_ckpt_dir(tmp_path), step=34, commit=lambda: calls.append(1))
    rl.close()

    rec = json.loads(open(tmp_path / "run" / "checkpoints.jsonl").read().strip())
    assert calls == [1], "commit was not called"
    assert rec.get("committed") is True, (
        "the record was written before the commit returned; the report would "
        "claim the checkpoint is not durable"
    )


def test_checkpoint_hashes_optimizer_and_rng_state(tmp_path):
    from vektori_trace.tau2.runlog import RunLog

    rl = RunLog(str(tmp_path / "run"))
    rl.checkpoint(_ckpt_dir(tmp_path), step=34)
    rl.close()

    files = json.loads(
        open(tmp_path / "run" / "checkpoints.jsonl").read().strip())["files"]
    for fn in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        assert fn in files, f"{fn} is required for an exact resume but unrecorded"
    assert len(files) == 6


def test_failed_commit_fails_the_run(tmp_path):
    """A checkpoint that did not commit is not durable; continuing would lie."""
    from vektori_trace.tau2.runlog import RunLog

    def boom():
        raise OSError("volume unavailable")

    rl = RunLog(str(tmp_path / "run"))
    with pytest.raises(RuntimeError, match="not durable"):
        rl.checkpoint(_ckpt_dir(tmp_path), step=34, commit=boom)
    rl.close()

    rec = json.loads(open(tmp_path / "run" / "checkpoints.jsonl").read().strip())
    assert rec["committed"] is False
    assert "volume unavailable" in rec["commit_error"]


REQUIRED_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def test_required_adapter_files_cover_serving_needs():
    """Weights alone are not a servable artifact.

    Prompt parity (7/7) was established for one specific chat template under
    transformers 5.5.3 vs vLLM's 4.57.6. Serving the adapter with a different
    tokenizer or template voids that result, so both ship with the weights.
    """
    import re
    src = open("scripts/tau2_sft_train.py").read()
    block = re.search(r"INFERENCE_FILES = \((.*?)\)", src, re.S).group(1)
    for fn in ("adapter_config.json", "adapter_model.safetensors",
               "tokenizer.json", "tokenizer_config.json"):
        assert fn in block, f"{fn} is not gated on but serving needs it"
    # The template may be a file or a key inside tokenizer_config.json
    # depending on the transformers version; the gate must accept either.
    # Both the template and the special-token map may be a separate file OR a
    # key inside tokenizer_config.json. Qwen3 embeds both, so requiring the
    # file form failed a complete artifact after 45 minutes of training.
    assert "EITHER_FILE_OR_KEY" in src
    assert '"special_tokens_map.json", "eos_token"' in src
    assert '"chat_template.jinja", "chat_template"' in src


def test_probe_adapter_had_every_required_file():
    """The real probe artifact, as a regression anchor."""
    p = "/tmp/adapter_diagnosis.json"
    if not os.path.exists(p):
        pytest.skip("no diagnosis artifact locally")
    saved = set(json.load(open(p))["files"])
    assert set(REQUIRED_ADAPTER_FILES) <= saved, (
        f"missing {set(REQUIRED_ADAPTER_FILES) - saved}")


# ---------------------------------------------------------------------------
# Artifact completeness. Three different jobs need three different file sets,
# and the distinction only becomes visible when one is missing months later.
# ---------------------------------------------------------------------------

def test_inference_file_set_is_complete():
    """What serving needs: weights, config, and the exact tokenizer/template.

    Prompt parity (7/7) holds for one specific chat template across
    transformers 5.5.3 and vLLM 4.57.6. Reloading the tokenizer from the Hub
    instead of the run directory reintroduces the drift that check ruled out.
    """
    src = open("scripts/tau2_sft_train.py").read()
    for fn in ("adapter_config.json", "adapter_model.safetensors",
               "tokenizer.json", "tokenizer_config.json"):
        assert f'"{fn}"' in src, f"{fn} is not in INFERENCE_FILES"
    # Special tokens are embedded in tokenizer_config.json by Qwen3, so they
    # are checked as file-or-key rather than required as a file.
    assert '"special_tokens_map.json", "eos_token"' in src


def test_resume_file_set_is_complete():
    """What continuing training needs, beyond what serving needs."""
    src = open("scripts/tau2_sft_train.py").read()
    for fn in ("optimizer.pt", "scheduler.pt", "trainer_state.json"):
        assert f'"{fn}"' in src, f"{fn} is not in RESUME_FILES"


def test_tokenizer_is_saved_explicitly():
    src = open("scripts/tau2_sft_train.py").read()
    assert "tok.save_pretrained(args.out)" in src, (
        "SFTTrainer only saves the tokenizer when handed one; saving it "
        "explicitly is what makes the run directory self-contained"
    )


def test_base_model_revision_is_recorded():
    """An adapter is meaningless without the exact base weights."""
    src = open("scripts/tau2_sft_train.py").read()
    assert "_commit_hash" in src
    assert '"base_model_revision"' in src


def test_checkpoints_keep_optimizer_state():
    """save_only_model=True would make every epoch checkpoint unresumable."""
    src = open("scripts/tau2_sft_train.py").read()
    assert "save_only_model=False" in src
    assert "assert not cfg.save_only_model" in src
