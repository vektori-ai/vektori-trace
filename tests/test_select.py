"""Unit tests for select.py — both conditions required (lacking-loss AND
in-band pass rate), and the held-out split. Pure functions over already-
computed inputs, no LLM/Docker involved."""

from __future__ import annotations

from pathlib import Path

from vektori_trace.evaluate.passrate import PassRate
from vektori_trace.select import held_out_split, select_training_tasks, write_selection_report


def _pr(task: str, passed: int, n: int) -> PassRate:
    return PassRate(task=task, passed=passed, n=n)


def test_select_requires_both_lacking_loss_and_in_band() -> None:
    lacking_loss_tasks = ["a", "b", "c"]
    pass_rates = {
        "a": _pr("a", 2, 10),  # 0.20, in band, lacking-loss -> selected
        "b": _pr("b", 8, 10),  # 0.80, out of band -> excluded despite lacking-loss
        "d": _pr("d", 3, 10),  # in band but NOT in lacking_loss_tasks -> excluded
    }
    # 'c' has no measured pass rate at all.

    selected = select_training_tasks(lacking_loss_tasks, pass_rates)

    assert selected == ["a"]


def test_select_excludes_tasks_missing_from_pass_rates() -> None:
    """Silence isn't evidence of trainability — a task with zero rollouts
    measured must not be assumed in-band."""
    selected = select_training_tasks(["a"], {})
    assert selected == []


def test_select_dedupes_repeated_task_ids() -> None:
    selected = select_training_tasks(["a", "a", "a"], {"a": _pr("a", 2, 10)})
    assert selected == ["a"]


def test_select_respects_a_custom_band() -> None:
    pass_rates = {"a": _pr("a", 5, 10)}  # 0.50
    assert select_training_tasks(["a"], pass_rates) == []  # outside default band
    assert select_training_tasks(["a"], pass_rates, band=(0.4, 0.6)) == ["a"]


def test_held_out_split_removes_excluded_before_splitting() -> None:
    train, holdout = held_out_split(
        ["a", "b", "c", "d"], exclude={"c"}, holdout_frac=0.5, seed=0
    )
    assert "c" not in train and "c" not in holdout
    assert sorted(train + holdout) == ["a", "b", "d"]


def test_held_out_split_is_deterministic_given_the_same_seed() -> None:
    ids = [f"t{i}" for i in range(20)]
    r1 = held_out_split(ids, holdout_frac=0.25, seed=42)
    r2 = held_out_split(ids, holdout_frac=0.25, seed=42)
    assert r1 == r2


def test_held_out_split_differs_across_seeds_on_a_large_enough_set() -> None:
    ids = [f"t{i}" for i in range(20)]
    r1 = held_out_split(ids, holdout_frac=0.25, seed=1)
    r2 = held_out_split(ids, holdout_frac=0.25, seed=2)
    assert r1 != r2


def test_held_out_split_sizes_match_the_fraction() -> None:
    ids = [f"t{i}" for i in range(10)]
    train, holdout = held_out_split(ids, holdout_frac=0.3, seed=0)
    assert len(holdout) == 3
    assert len(train) == 7


def test_held_out_split_train_and_holdout_partition_the_input() -> None:
    ids = ["a", "b", "c", "d", "e"]
    train, holdout = held_out_split(ids, holdout_frac=0.4, seed=7)
    assert set(train) | set(holdout) == set(ids)
    assert set(train) & set(holdout) == set()


def test_write_selection_report_notes_empty_selection(tmp_path: Path) -> None:
    md_path = write_selection_report(
        tmp_path,
        frontier_model="gpt-5",
        candidate_model="small",
        agent="claude-code",
        band=(0.10, 0.40),
        rollouts=8,
        seed=0,
        holdout_frac=0.2,
        pass_rates={"a": _pr("a", 9, 10)},
        lacking_loss_tasks=["a"],
        selected=[],
        train_ids=[],
        holdout_ids=[],
        exclude=set(),
    )
    text = md_path.read_text()
    assert "Nothing selected" in text
    assert (tmp_path / "selection.json").exists()


def test_write_selection_report_json_is_reproducible_from_disk(tmp_path: Path) -> None:
    import json

    write_selection_report(
        tmp_path,
        frontier_model="gpt-5",
        candidate_model="small",
        agent="claude-code",
        band=(0.10, 0.40),
        rollouts=8,
        seed=3,
        holdout_frac=0.2,
        pass_rates={"a": _pr("a", 2, 10)},
        lacking_loss_tasks=["a", "a"],
        selected=["a"],
        train_ids=["a"],
        holdout_ids=[],
        exclude={"z"},
    )
    data = json.loads((tmp_path / "selection.json").read_text())
    assert data["seed"] == 3
    assert data["selected"] == ["a"]
    assert data["excluded"] == ["z"]
    assert data["lacking_loss_tasks"] == ["a"]  # deduped
    assert data["pass_rates"]["a"]["rate"] == 0.2
