"""The Stage B trainer's CPU-checkable invariants.

The training loop needs a GPU, but the three things that would silently turn
this run into something else do not: the cold-share arithmetic that gates it,
which adapter it continues, and the sampler it declares. All three are checked
here so a GPU session is never the place a typo surfaces.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import sft_stage_b_train_modal as sb

SRC = Path("scripts/sft_stage_b_train_modal.py").read_text()


# ---- the number that gates the run -------------------------------------

def test_weighted_and_uniform_shares_differ_when_the_weights_do():
    """The whole reason the floor stalled: cold rows are short, so a uniform
    shuffle sees far fewer cold tokens than the solved mix intends."""
    weights = [3.0, 1.0, 1.0]
    supervised = [100, 400, 400]
    kinds = [sb.COLD_KIND, "later_last", "later_last"]
    weighted, uniform = sb.cold_shares(weights, supervised, kinds)
    # weighted: E[cold]=3/5*100=60, E[tok]=60+(1/5*400)*2=220
    assert weighted == pytest.approx(60 / 220)
    assert uniform == pytest.approx(100 / 900)
    assert weighted > uniform


def test_equal_weights_make_the_two_shares_identical():
    weights = [1.0, 1.0, 1.0]
    supervised = [100, 400, 400]
    kinds = [sb.COLD_KIND, "later_last", "later_last"]
    weighted, uniform = sb.cold_shares(weights, supervised, kinds)
    assert weighted == pytest.approx(uniform)


def test_misaligned_inputs_are_refused_rather_than_zipped_short():
    """These three lists index the same rows. Silently truncating to the
    shortest would compute a share over a subset and report it as the whole."""
    with pytest.raises(ValueError, match="index the same rows"):
        sb.cold_shares([1.0, 1.0], [100], [sb.COLD_KIND])


def test_zero_weights_are_refused():
    with pytest.raises(ValueError, match="nothing would be sampled"):
        sb.cold_shares([0.0, 0.0], [100, 100], [sb.COLD_KIND, "later_last"])


def test_no_cold_rows_is_a_zero_share_not_a_crash():
    weighted, uniform = sb.cold_shares([1.0], [100], ["later_last"])
    assert (weighted, uniform) == (0.0, 0.0)


def test_the_floor_here_matches_the_builder():
    from scripts import sft_stage_b_dataset as builder

    assert sb.COLD_TOKEN_FLOOR == builder.COLD_TOKEN_FLOOR
    assert sb.COLD_KIND == builder.COLD_REPLAY


# ---- what it continues --------------------------------------------------

def test_it_continues_stage_a_checkpoint_84():
    assert sb.BASE_ADAPTER_IN_VOLUME == "sft/qwen3-14b-stage-a-lora/checkpoint-84"


def test_it_never_constructs_a_fresh_adapter():
    """Stage A creates a LoRA; Stage B continuing one is the difference. A
    stray get_peft_model here would start from zero at Stage B's LR."""
    tree = ast.parse(SRC)
    called = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "get_peft_model" not in called
    assert "LoraConfig" not in called
    assert "PeftModel.from_pretrained" in SRC
    assert "is_trainable=True" in SRC


def test_the_v1_lineage_is_refused_by_name():
    """v1 / repaired / ck63 are the tool_call-envelope protocol. Continuing one
    would look like a normal run and undo what Stage A measured."""
    for forbidden in ("dsv4", "repaired", "checkpoint-63"):
        assert f'"{forbidden}"' in SRC


def test_stage_b_writes_beside_stage_a_never_over_it():
    assert sb.OUT_IN_VOLUME == "sft/qwen3-14b-stage-b-lora"
    assert sb.OUT_IN_VOLUME != "sft/qwen3-14b-stage-a-lora"


def test_the_length_covers_the_deferred_recoveries():
    """Amendment 2 moved 18 recoveries here; 14 exceed 8k, max ~33k."""
    assert sb.MAX_LENGTH == 40960


def test_the_sampler_seam_is_asserted_not_assumed():
    """An upstream rename of _get_train_sampler would restore uniform sampling
    silently, which is exactly the 14.5% share."""
    assert '_get_train_sampler' in SRC
    assert 'hasattr(SFTTrainer, "_get_train_sampler")' in SRC
    assert "WeightedRandomSampler" in SRC


# ---- interrupted-run resume --------------------------------------------

def _checkpoint(tmp_path: Path, step: int = 50) -> tuple[Path, str]:
    rel = f"{sb.OUT_IN_VOLUME}/checkpoint-{step}"
    path = tmp_path / rel
    path.mkdir(parents=True)
    for name in sb.RESUME_REQUIRED_FILES:
        (path / name).touch()
    (path / "trainer_state.json").write_text(json.dumps({
        "global_step": step,
        "max_steps": 93,
        "train_batch_size": 1,
        "epoch": step / 93,
    }))
    return path, rel


def test_resume_accepts_the_complete_stage_b_checkpoint(tmp_path):
    path, rel = _checkpoint(tmp_path)
    got, state, step = sb.validate_resume_checkpoint(
        rel, volume_mount=tmp_path, max_steps=93, batch_size=1
    )
    assert got == path
    assert state["global_step"] == step == 50


def test_resume_requires_rng_state_instead_of_silently_reseeding(tmp_path):
    path, rel = _checkpoint(tmp_path)
    (path / "rng_state.pth").unlink()
    with pytest.raises(ValueError, match="rng_state.pth"):
        sb.validate_resume_checkpoint(
            rel, volume_mount=tmp_path, max_steps=93, batch_size=1
        )


@pytest.mark.parametrize("rel", [
    "/adapters/sft/qwen3-14b-stage-b-lora/checkpoint-50",
    "sft/qwen3-14b-stage-a-lora/checkpoint-50",
    "sft/qwen3-14b-stage-b-lora/not-a-checkpoint",
])
def test_resume_refuses_paths_outside_the_stage_b_checkpoint_series(tmp_path, rel):
    with pytest.raises(ValueError, match="must be"):
        sb.validate_resume_checkpoint(
            rel, volume_mount=tmp_path, max_steps=93, batch_size=1
        )


def test_resume_directory_step_must_match_trainer_state(tmp_path):
    path, rel = _checkpoint(tmp_path)
    state = json.loads((path / "trainer_state.json").read_text())
    state["global_step"] = 49
    (path / "trainer_state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="global_step=49"):
        sb.validate_resume_checkpoint(
            rel, volume_mount=tmp_path, max_steps=93, batch_size=1
        )


def test_resume_refuses_schedule_or_batch_drift(tmp_path):
    _, rel = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="max_steps"):
        sb.validate_resume_checkpoint(
            rel, volume_mount=tmp_path, max_steps=94, batch_size=1
        )
    with pytest.raises(ValueError, match="train_batch_size"):
        sb.validate_resume_checkpoint(
            rel, volume_mount=tmp_path, max_steps=93, batch_size=2
        )


def test_resume_training_arguments_fail_closed_on_dynamics_changes():
    prior = SimpleNamespace(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
    )
    expected = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-5,
    }
    assert sb.resume_training_arg_mismatches(prior, expected) == [
        "gradient_accumulation_steps: checkpoint=8, current=4"
    ]


def test_stage_b_saves_every_ten_steps_and_checks_resume_completion():
    assert sb.DEFAULT_SAVE_STEPS == 10
    assert 'save_steps: int = DEFAULT_SAVE_STEPS' in SRC
    assert "final_step != max_steps" in SRC
