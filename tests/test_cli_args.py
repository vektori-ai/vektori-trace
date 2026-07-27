"""Threshold parsing at the CLI boundary.

`select_deficit`'s two thresholds are the only thing standing between a ranked
list and a confident report, so a value that silently disables one puts us back
where `scores[0]` was: reporting inverted evidence as a diagnosis. These are
cheap to reject at parse time and expensive to notice downstream.
"""

from __future__ import annotations

import argparse
import json

import pytest

from vektori_trace.cli import _min_gap_arg, _min_support_arg, build_parser


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf", "Infinity"])
def test_non_finite_gaps_are_rejected(value: str) -> None:
    """`float("nan")` parses, and then every `gap < nan` is False — so a NaN
    threshold doesn't loosen the filter, it removes it."""
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        _min_gap_arg(value)


@pytest.mark.parametrize("value", ["-0.1", "-1", "-1e9"])
def test_negative_gaps_are_rejected(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=">= 0"):
        _min_gap_arg(value)


@pytest.mark.parametrize("value", ["0", "0.2", "1", "0.999"])
def test_valid_gaps_survive(value: str) -> None:
    assert _min_gap_arg(value) == float(value)


def test_non_numeric_gap_is_rejected() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="not a number"):
        _min_gap_arg("high")


@pytest.mark.parametrize("value", ["0", "-1"])
def test_support_below_one_is_rejected(value: str) -> None:
    """min_support=0 admits a capability with no relevant traces on either
    side — a gap computed from nothing."""
    with pytest.raises(argparse.ArgumentTypeError, match=">= 1"):
        _min_support_arg(value)


def test_non_integer_support_is_rejected() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="not an integer"):
        _min_support_arg("3.5")


@pytest.mark.parametrize("value", ["1", "3", "10"])
def test_valid_support_survives(value: str) -> None:
    assert _min_support_arg(value) == int(value)


def test_the_parser_actually_uses_the_validators() -> None:
    """The validators are worthless if they aren't wired to the flags — this
    fails if a future arg is added back with a bare `type=float`."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["diagnose", "--manifest", "m.json", "--min-gap", "nan"])
    with pytest.raises(SystemExit):
        parser.parse_args(["diagnose", "--manifest", "m.json", "--min-support", "0"])


def test_defaults_still_parse() -> None:
    args = build_parser().parse_args(["diagnose", "--manifest", "m.json"])
    assert args.min_gap > 0
    assert args.min_support >= 1


def test_selftest_thresholds_are_validated_too() -> None:
    """The sweep scores recovery *against* these thresholds, so a NaN here
    doesn't just weaken the report — it makes every cell recover."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["selftest", "--min-gap", "nan"])
    with pytest.raises(SystemExit):
        parser.parse_args(["selftest", "--min-support", "0"])


def _manifest(tmp_path, entries: list[dict]):
    """A manifest plus the trace files it points at — enough for `cmd_diagnose`
    to load, which is all the model-pairing checks run before."""
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for i, e in enumerate(entries):
        p = traces_dir / f"t{i}.json"
        p.write_text(json.dumps({"runId": f"r{i}", "status": "success", "turns": []}))
        items.append({"path": str(p), **e})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(items))
    return path


def _diagnose_args(manifest, *extra: str):
    return build_parser().parse_args(["diagnose", "--manifest", str(manifest), *extra])


PAIRED = [
    {"outcome": "win", "model": "gpt-5", "task": "t1"},
    {"outcome": "loss", "model": "small", "task": "t1"},
]


def test_diagnose_rejects_one_model_flag_without_the_other(tmp_path) -> None:
    """The two contrasts are defined by which model produced which trace — one
    name on its own defines neither, and finding that out after labelling has
    cost an LLM call per trace."""
    from vektori_trace.cli import cmd_diagnose

    manifest = _manifest(tmp_path, PAIRED)
    assert cmd_diagnose(_diagnose_args(manifest, "--frontier-model", "gpt-5")) == 2
    assert cmd_diagnose(_diagnose_args(manifest, "--candidate-model", "small")) == 2


def test_diagnose_rejects_identical_frontier_and_candidate_model(tmp_path) -> None:
    from vektori_trace.cli import cmd_diagnose

    manifest = _manifest(tmp_path, PAIRED)
    args = _diagnose_args(manifest, "--frontier-model", "gpt-5", "--candidate-model", "gpt-5")
    assert cmd_diagnose(args) == 2


def test_diagnose_rejects_a_model_with_no_traces_in_the_manifest(tmp_path) -> None:
    """Cheap to catch here; expensive to discover as an empty contrast after
    every trace has been labelled."""
    from vektori_trace.cli import cmd_diagnose

    manifest = _manifest(tmp_path, PAIRED)
    args = _diagnose_args(manifest, "--frontier-model", "gpt-5", "--candidate-model", "typo-8b")
    assert cmd_diagnose(args) == 2


def test_diagnose_rejects_model_flags_against_a_mine_manifest(tmp_path) -> None:
    """`mine` writes no `model` field, so the pairing has nothing to work with."""
    from vektori_trace.cli import cmd_diagnose

    manifest = _manifest(tmp_path, [{"outcome": "win"}, {"outcome": "loss"}])
    args = _diagnose_args(manifest, "--frontier-model", "gpt-5", "--candidate-model", "small")
    assert cmd_diagnose(args) == 2


def test_diagnose_model_flags_default_to_none(tmp_path) -> None:
    """Neither flag given must stay the plain, model-blind path."""
    args = build_parser().parse_args(["diagnose", "--manifest", "m.json"])
    assert args.frontier_model is None
    assert args.candidate_model is None


def test_replay_requires_tasks_dir_and_both_models() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["replay", "--frontier-model", "gpt-5", "--candidate-model", "small"])
    with pytest.raises(SystemExit):
        parser.parse_args(["replay", "--tasks-dir", "t", "--candidate-model", "small"])
    with pytest.raises(SystemExit):
        parser.parse_args(["replay", "--tasks-dir", "t", "--frontier-model", "gpt-5"])


def test_replay_agent_defaults_to_claude_code() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["replay", "--tasks-dir", "t", "--frontier-model", "gpt-5", "--candidate-model", "small"]
    )
    assert args.agent == "claude-code"


def test_replay_rejects_identical_frontier_and_candidate_model(tmp_path) -> None:
    """A gap between a model and itself isn't a gap — cheap to reject before
    burning a real replay run to discover it."""
    from vektori_trace.cli import cmd_replay

    task = tmp_path / "tasks" / "t1"
    task.mkdir(parents=True)
    (task / "task.toml").write_text("x = 1")

    parser = build_parser()
    args = parser.parse_args(
        [
            "replay",
            "--tasks-dir",
            str(tmp_path / "tasks"),
            "--frontier-model",
            "same-model",
            "--candidate-model",
            "same-model",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert cmd_replay(args) == 2
