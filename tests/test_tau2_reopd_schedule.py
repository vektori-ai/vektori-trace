"""The shared 32x16 schedule, and the ways two arms could silently diverge."""

from __future__ import annotations

import json

import pytest

from vektori_trace.tau2.reopd_schedule import (
    N_PER_UPDATE,
    N_UPDATES,
    ScheduleError,
    batch_for,
    build_schedule,
    freeze_schedule,
    load_schedule,
)

POOL = [f"{10 + i // 10}#{i % 10}" for i in range(289)]   # 289, like C30


def test_pilot_budget_is_32x16():
    """V2 section 8. An arm changing either number is a different experiment."""
    assert (N_UPDATES, N_PER_UPDATE) == (32, 16)


def test_shape_matches_the_preregistered_budget():
    s = build_schedule(POOL)
    assert s["n_updates"] == 32
    assert all(len(u["prefix_ids"]) == 16 for u in s["updates"])
    assert s["n_exposures"] == 512


def test_512_exposures_over_289_is_about_1_77_passes():
    s = build_schedule(POOL)
    assert s["n_pool"] == 289
    assert s["passes"] == pytest.approx(1.7716, abs=1e-3)


def test_wrapping_does_not_over_weight_the_head():
    """The imbalance would land exactly where the frozen order put late tasks."""
    s = build_schedule(POOL)
    # 512 draws over 289: some prefixes twice, some three times, none skipped
    assert s["n_unexposed"] == 0
    assert s["exposure_max"] - s["exposure_min"] <= 1


def test_order_is_the_frozen_order_not_a_reshuffle():
    s = build_schedule(POOL)
    flat = [pid for u in s["updates"] for pid in u["prefix_ids"]]
    assert flat[:289] == POOL


def test_batch_for_returns_one_update():
    s = build_schedule(POOL)
    assert batch_for(s, 0) == POOL[:16]
    assert batch_for(s, 1) == POOL[16:32]
    with pytest.raises(ScheduleError, match="no update 99"):
        batch_for(s, 99)


def test_hash_is_stable_across_rebuilds():
    assert build_schedule(POOL)["schedule_hash"] == build_schedule(POOL)["schedule_hash"]


def test_hash_changes_with_shape():
    a = build_schedule(POOL)
    b = build_schedule(POOL, n_per_update=8)
    assert a["schedule_hash"] != b["schedule_hash"]


def test_hash_changes_with_pool_order():
    """Two arms computing the order independently must not silently agree."""
    a = build_schedule(POOL)
    b = build_schedule(list(reversed(POOL)))
    assert a["schedule_hash"] != b["schedule_hash"]


# --- freeze / read twice --------------------------------------------------


def test_freeze_is_idempotent(tmp_path):
    p = str(tmp_path / "sched.json")
    s = build_schedule(POOL)
    assert freeze_schedule(p, s)["schedule_hash"] == s["schedule_hash"]
    assert freeze_schedule(p, build_schedule(POOL))["schedule_hash"] == s["schedule_hash"]


def test_freeze_refuses_a_different_schedule(tmp_path):
    """Regenerating per branch is what invalidates the comparison."""
    p = str(tmp_path / "sched.json")
    freeze_schedule(p, build_schedule(POOL))
    with pytest.raises(ScheduleError, match="already frozen"):
        freeze_schedule(p, build_schedule(POOL, n_per_update=8))


def test_second_arm_reads_the_same_stream(tmp_path):
    p = str(tmp_path / "sched.json")
    first = freeze_schedule(p, build_schedule(POOL))
    second = load_schedule(p, expect_hash=first["schedule_hash"], expect_pool=POOL)
    assert second["updates"] == first["updates"]


def test_load_refuses_wrong_hash(tmp_path):
    p = str(tmp_path / "sched.json")
    freeze_schedule(p, build_schedule(POOL))
    with pytest.raises(ScheduleError, match="!= expected"):
        load_schedule(p, expect_hash="0000000000000000")


def test_load_refuses_a_schedule_from_another_corpus(tmp_path):
    p = str(tmp_path / "sched.json")
    freeze_schedule(p, build_schedule(POOL))
    with pytest.raises(ScheduleError, match="outside the loaded pool"):
        load_schedule(p, expect_pool=["99#9"])


def test_load_refuses_a_missing_schedule(tmp_path):
    with pytest.raises(ScheduleError, match="no frozen schedule"):
        load_schedule(str(tmp_path / "absent.json"))


def test_rejects_empty_pool():
    with pytest.raises(ScheduleError, match="no prefixes"):
        build_schedule([])
