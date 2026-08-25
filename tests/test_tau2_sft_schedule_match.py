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
