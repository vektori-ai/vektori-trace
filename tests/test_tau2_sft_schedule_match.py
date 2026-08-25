"""Continued SFT consumes the same stream as the replay arm.

V2 s8 preregisters the match over updates, prefix exposures, sampling order,
effective batch size and LoRA capacity. Every test here covers one way the two
arms could diverge while both still logged "one epoch over C30".
"""

from __future__ import annotations

import json

import pytest

from vektori_trace.tau2.reopd_schedule import build_schedule, freeze_schedule

POOL = [f"{10 + i // 10}#{i % 10}" for i in range(289)]


def _sched(tmp_path):
    p = str(tmp_path / "schedule.json")
    freeze_schedule(p, build_schedule(POOL))
    return p, json.load(open(p))


def test_exposures_match_the_replay_arm(tmp_path):
    _, s = _sched(tmp_path)
    rows = [pid for u in s["updates"] for pid in u["prefix_ids"]]
    assert len(rows) == 512                      # not 289
    assert s["n_updates"] == 32                  # not ~37
    assert s["n_per_update"] == 16               # not 8


def test_repeated_prefixes_are_not_deduplicated(tmp_path):
    """A prefix drawn twice is two exposures; collapsing them halves the arm."""
    _, s = _sched(tmp_path)
    rows = [pid for u in s["updates"] for pid in u["prefix_ids"]]
    assert len(rows) > len(set(rows))
    from collections import Counter
    assert max(Counter(rows).values()) >= 2


def test_effective_batch_is_the_schedule_width(tmp_path):
    """batch_size 1 x grad_accum 16 = one optimizer step per update."""
    _, s = _sched(tmp_path)
    assert 1 * s["n_per_update"] == 16
    assert s["n_exposures"] // s["n_per_update"] == s["n_updates"]


def test_order_is_frozen_not_shuffled(tmp_path):
    _, s = _sched(tmp_path)
    rows = [pid for u in s["updates"] for pid in u["prefix_ids"]]
    assert rows[:289] == POOL


def test_trainer_forces_a_sequential_sampler():
    """HF Trainer defaults to RandomSampler, so ordering rows is not enough."""
    src = open("scripts/tau2_sft_train.py").read()
    assert "_FrozenOrderTrainer" in src
    assert "SequentialSampler" in src
    assert "_get_train_sampler" in src


def test_trainer_pins_max_steps_under_a_schedule():
    src = open("scripts/tau2_sft_train.py").read()
    assert "args.probe or args.schedule" in src


def test_both_arms_read_one_frozen_file(tmp_path):
    """Built once, read twice: a regenerated schedule breaks the match."""
    from vektori_trace.tau2.reopd_schedule import ScheduleError

    p, _ = _sched(tmp_path)
    with pytest.raises(ScheduleError, match="already frozen"):
        freeze_schedule(p, build_schedule(POOL, n_per_update=8))


# --- the runtime proofs ---------------------------------------------------


def test_order_assertion_checks_both_class_and_emitted_batch():
    """A correct sampler class does not prove a correct emitted order.

    accelerate can wrap the sampler and reorder or split batches, so the class
    check alone is necessary and not sufficient.
    """
    src = open("scripts/tau2_sft_train.py").read()
    assert "_assert_sequential_order" in src
    assert "SequentialSampler" in src
    assert "get_train_dataloader" in src
    # compares the first real batch against the rows the schedule put first
    assert "batch 0 position" in src


def test_order_assertion_runs_before_training():
    src = open("scripts/tau2_sft_train.py").read()
    assert src.index("_assert_sequential_order(trainer") < src.index("trainer.train(")


def test_order_assertion_uses_a_real_runlog_method():
    """RunLog has step/write_config/summary, not `event`."""
    from vektori_trace.tau2.runlog import RunLog
    src = open("scripts/tau2_sft_train.py").read()
    assert "runlog.event(" not in src
    assert hasattr(RunLog, "step")


def test_schedule_is_mandatory_for_the_control_arm():
    """A warning is too weak: an unmatched control reads like a control."""
    src = open("scripts/tau2_sft_continue_modal.py").read()
    assert "--schedule is required" in src
    assert "allow_unmatched" in src


def test_staging_pins_the_schedule_hash():
    src = open("scripts/tau2_stage_policy_modal.py").read()
    assert "24c0aa5395d69772" in src
    assert "SCHEDULE_IN_VOLUME" in src


# --- the schedule must actually take effect ------------------------------


def test_schedule_governs_the_step_count_not_the_epoch_math():
    """The run printed 'planned optimizer steps 36' while claiming a match.

    An anchor that silently matched nothing left the schedule block out
    entirely, so rows, batch size and step count were all epoch-derived. The
    step count must come from the schedule, and it must be computed BEFORE the
    epoch fallback.
    """
    src = open("scripts/tau2_sft_train.py").read()
    i_apply = src.index("schedule_meta = None")
    i_steps = src.index("planned optimizer steps")
    assert i_apply < i_steps, "schedule must be applied before steps is printed"
    assert 'steps = schedule_meta["n_updates"]' in src


def test_no_dead_schedule_branch_survives():
    src = open("scripts/tau2_sft_train.py").read()
    assert "if False:" not in src, "dead branch left in the trainer"


def test_schedule_meta_is_always_defined():
    """It is read unconditionally by write_config; an unset name is a NameError
    that only fires on the paid path."""
    src = open("scripts/tau2_sft_train.py").read()
    i_def = src.index("schedule_meta = None")
    i_use = src.index('"schedule": schedule_meta')
    assert i_def < i_use
