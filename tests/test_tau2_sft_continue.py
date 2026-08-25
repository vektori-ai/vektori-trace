"""Static launch invariants for CK35 -> C30 continued SFT."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TRAINER = (REPO / "scripts/tau2_sft_train.py").read_text()
LAUNCHER = (REPO / "scripts/tau2_sft_continue_modal.py").read_text()


def test_c30_requires_a_parent_adapter():
    assert 'args.partition == "C30" and not args.init_adapter' in TRAINER


def test_parent_is_loaded_trainable_without_optimizer_resume():
    assert "PeftModel.from_pretrained" in TRAINER
    assert "is_trainable=True" in TRAINER
    assert "trainer.train(resume_from_checkpoint" not in TRAINER
    assert '"optimizer_state_source": "fresh"' in TRAINER
    assert '"scheduler_state_source": "fresh"' in TRAINER


def test_launcher_is_ck35_to_c30_only():
    assert 'PARENT_IN_VOLUME = "tau2/runs/a_warm_20260825_003343/checkpoint-35"' in LAUNCHER
    assert '"--partition", "C30"' in LAUNCHER
    assert '"--init-adapter", parent' in LAUNCHER


def test_launcher_needs_explicit_gpu_approval():
    assert "if not yes:" in LAUNCHER
    assert "pass --yes" in LAUNCHER


def test_default_is_one_real_c30_epoch():
    assert "epochs: float = 1.0" in LAUNCHER


def test_parent_adapter_is_validated_and_hashed():
    assert "def parent_adapter_manifest" in TRAINER
    assert '"files_sha256": files' in TRAINER
    assert 'parent_base != base_model' in TRAINER
