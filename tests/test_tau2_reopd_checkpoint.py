"""Checkpoint save/restore: the difference between one update and thirty-two.

`replay_train` saves an adapter, which suffices for a single update. These tests
cover what a resumable run additionally needs, and the ways each piece fails
silently when it is missing.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from vektori_trace.tau2.reopd_checkpoint import (  # noqa: E402
    CheckpointError,
    load_checkpoint,
    restore_rng,
    rng_snapshot,
    save_checkpoint,
)
from vektori_trace.tau2.reopd_state import RunState, UpdateDir  # noqa: E402


class TinyModel(torch.nn.Module):
    """Stands in for the PEFT model; only save_pretrained is exercised."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(4))

    def save_pretrained(self, path):
        import os
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/adapter_config.json", "w") as fh:
            json.dump({"peft_type": "LORA"}, fh)
        with open(f"{path}/adapter_model.safetensors", "wb") as fh:
            fh.write(self.w.detach().numpy().tobytes())


def _model_opt():
    m = TinyModel()
    return m, torch.optim.AdamW(m.parameters(), lr=1e-4)


# --- what a checkpoint must contain --------------------------------------


def test_saves_everything_needed_to_resume(tmp_path):
    m, opt = _model_opt()
    cp = tmp_path / "checkpoint"
    state = save_checkpoint(m, opt, cp, update_index=3,
                            policy_version="update-3",
                            parent_policy_hash="ck35abc")

    assert (cp / "adapter_config.json").exists()
    assert (cp / "adapter_model.safetensors").exists()
    assert (cp / "optimizer.pt").exists()
    assert (cp / "state.json").exists()
    assert state["update_index"] == 3
    assert state["policy_version"] == "update-3"
    assert state["parent_policy_hash"] == "ck35abc"
    assert state["rng_state"]
    assert state["reload_verified"] is False       # until verify_reload runs


def test_scheduler_state_is_null_not_absent_when_there_is_none(tmp_path):
    """build_optimizer returns a bare AdamW; absent reads as 'forgot to save'."""
    m, opt = _model_opt()
    state = save_checkpoint(m, opt, tmp_path / "cp", update_index=0,
                            policy_version="v0", parent_policy_hash="p")
    assert "scheduler_state" in state
    assert state["scheduler_state"] is None
    assert state["has_scheduler"] is False


def test_adam_moments_survive_a_round_trip(tmp_path):
    """Resuming fresh discards the moments and changes the effective LR."""
    m, opt = _model_opt()
    m.w.grad = torch.ones(4)
    opt.step()                                     # populates exp_avg
    before = opt.state_dict()["state"][0]["exp_avg"].clone()

    save_checkpoint(m, opt, tmp_path / "cp", update_index=1,
                    policy_version="v1", parent_policy_hash="p")

    m2, opt2 = _model_opt()
    assert not opt2.state_dict()["state"]          # fresh: no moments
    load_checkpoint(tmp_path / "cp", m2, opt2)
    after = opt2.state_dict()["state"][0]["exp_avg"]
    assert torch.allclose(before, after)


def test_scheduler_round_trips(tmp_path):
    m, opt = _model_opt()
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    sched.step(); sched.step()
    state = save_checkpoint(m, opt, tmp_path / "cp", update_index=2,
                            policy_version="v2", parent_policy_hash="p",
                            scheduler=sched)
    assert state["has_scheduler"] is True

    m2, opt2 = _model_opt()
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.5)
    load_checkpoint(tmp_path / "cp", m2, opt2, scheduler=sched2)
    assert sched2.last_epoch == sched.last_epoch


# --- RNG ------------------------------------------------------------------


def test_rng_round_trip_reproduces_the_stream(tmp_path):
    """A resumed run must sample the stream it would have sampled."""
    import random
    random.seed(1234)
    torch.manual_seed(1234)
    snap = rng_snapshot()
    expected = [random.random() for _ in range(5)]

    for _ in range(20):
        random.random()
    restore_rng(snap)
    assert [random.random() for _ in range(5)] == expected


def test_rng_restored_by_load_checkpoint(tmp_path):
    import random
    m, opt = _model_opt()
    random.seed(99)
    save_checkpoint(m, opt, tmp_path / "cp", update_index=0,
                    policy_version="v0", parent_policy_hash="p")
    expected = [random.random() for _ in range(3)]

    random.seed(1)                                  # clobber it
    m2, opt2 = _model_opt()
    load_checkpoint(tmp_path / "cp", m2, opt2)
    assert [random.random() for _ in range(3)] == expected


# --- refusals -------------------------------------------------------------


def test_missing_optimizer_is_refused(tmp_path):
    m, opt = _model_opt()
    cp = tmp_path / "cp"
    save_checkpoint(m, opt, cp, update_index=0, policy_version="v0",
                    parent_policy_hash="p")
    (cp / "optimizer.pt").unlink()
    m2, opt2 = _model_opt()
    with pytest.raises(CheckpointError, match="discards Adam's moments"):
        load_checkpoint(cp, m2, opt2)


def test_missing_state_is_refused(tmp_path):
    m2, opt2 = _model_opt()
    with pytest.raises(CheckpointError, match="not a resumable checkpoint"):
        load_checkpoint(tmp_path / "nowhere", m2, opt2)


def test_expected_scheduler_missing_is_refused(tmp_path):
    m, opt = _model_opt()
    cp = tmp_path / "cp"
    save_checkpoint(m, opt, cp, update_index=0, policy_version="v0",
                    parent_policy_hash="p")          # saved without scheduler
    m2, opt2 = _model_opt()
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1)
    with pytest.raises(CheckpointError, match="scheduler.pt is absent"):
        load_checkpoint(cp, m2, opt2, scheduler=sched2)


# --- integration with the state machine ----------------------------------


def test_saved_checkpoint_still_fails_validation_until_reload_verified(tmp_path):
    """A checkpoint is not trusted until it is proven to reload."""
    rs = RunState(tmp_path / "run", n_updates=2)
    u = rs.update(0)
    m, opt = _model_opt()
    save_checkpoint(m, opt, u.checkpoint_path, update_index=0,
                    policy_version="v0", parent_policy_hash="p")
    with pytest.raises(Exception, match="reload-verified"):
        u.validate_checkpoint()


def test_reload_verified_checkpoint_passes_validation(tmp_path):
    rs = RunState(tmp_path / "run", n_updates=2)
    u = rs.update(0)
    m, opt = _model_opt()
    save_checkpoint(m, opt, u.checkpoint_path, update_index=0,
                    policy_version="v0", parent_policy_hash="p")
    sp = u.checkpoint_path / "state.json"
    st = json.loads(sp.read_text())
    st["reload_verified"] = True
    sp.write_text(json.dumps(st))
    assert u.validate_checkpoint()["update_index"] == 0
