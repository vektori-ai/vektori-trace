"""Step B — gym import adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vektori_trace.gym_import import (
    UnverifiableGymTask,
    emit_harbor_task,
    import_gym,
    parse_gym_record,
)

_TEST_PATCH = (
    "diff --git a/tests/test_a.py b/tests/test_a.py\n"
    "--- a/tests/test_a.py\n"
    "+++ b/tests/test_a.py\n"
    "@@ -0,0 +1 @@\n"
    "+def test_a(): assert True\n"
)


def _complete_record(**over: object) -> dict:
    rec: dict = {
        "instance_id": "repo__task-1",
        "problem_statement": "fix it",
        "FAIL_TO_PASS": '["tests/test_a.py::test_a"]',
        "PASS_TO_PASS": ["tests/test_b.py::test_b"],
        "image": "ghcr.io/example:latest",
        "base_commit": "abc123",
        "test_patch": _TEST_PATCH,
        "patch": "diff --git a/x b/x\n",
        "repo": "example/repo",
    }
    rec.update(over)
    return rec


def test_parse_aliases() -> None:
    inst = parse_gym_record(_complete_record())
    assert inst.instance_id == "repo__task-1"
    assert inst.fail_to_pass == ["tests/test_a.py::test_a"]
    assert inst.pass_to_pass == ["tests/test_b.py::test_b"]
    assert inst.gold_patch.startswith("diff")


def test_emit_harbor_task_carries_a_real_verifier(tmp_path: Path) -> None:
    """PLAN.md's pass@k gate cannot tell an always-failing placeholder from a
    task outside the student's support, so imported tasks must ship the same
    graded F2P/P2P verifier the miner emits."""
    inst = parse_gym_record(_complete_record())
    task_dir = emit_harbor_task(inst, tmp_path)

    assert (task_dir / "task.toml").exists()
    assert (task_dir / "instruction.md").read_text().startswith("fix it")
    assert (task_dir / "environment" / "Dockerfile").exists()

    script = (task_dir / "tests" / "test.sh").read_text()
    assert "verifier.py" in script
    assert (task_dir / "tests" / "verifier.py").exists()
    assert json.loads((task_dir / "tests" / "f2p.json").read_text()) == [
        "tests/test_a.py::test_a"
    ]
    assert json.loads((task_dir / "tests" / "p2p.json").read_text()) == [
        "tests/test_b.py::test_b"
    ]
    # The base commit reaches the anti-tamper reset, the image the Dockerfile.
    assert "abc123" in script
    dockerfile = (task_dir / "environment" / "Dockerfile").read_text()
    assert "ghcr.io/example:latest" in dockerfile


def test_record_without_oracle_is_skipped_not_stubbed(tmp_path: Path) -> None:
    for missing, key in [
        ("no container image", "image"),
        ("no FAIL_TO_PASS", "FAIL_TO_PASS"),
        ("no base_commit", "base_commit"),
        ("no test_patch", "test_patch"),
    ]:
        inst = parse_gym_record(_complete_record(**{key: None}))
        with pytest.raises(UnverifiableGymTask, match=missing):
            emit_harbor_task(inst, tmp_path)


def test_import_jsonl_reports_skips(tmp_path: Path) -> None:
    src = tmp_path / "gym.jsonl"
    src.write_text(
        json.dumps(_complete_record(instance_id="good"))
        + "\n"
        + json.dumps({"instance_id": "bad", "problem": "p2"})
        + "\n"
    )
    result = import_gym(src, tmp_path / "tasks")
    assert len(result.tasks) == 1
    assert all((p / "task.toml").exists() for p in result.tasks)
    assert "bad" in result.skipped
    assert "image" in result.skipped["bad"]


def test_default_test_cmd_targets_f2p(tmp_path: Path) -> None:
    inst = parse_gym_record(_complete_record())
    script = (emit_harbor_task(inst, tmp_path) / "tests" / "test.sh").read_text()
    assert "pytest" in script
    assert "tests/test_a.py::test_a" in script


def test_explicit_test_cmd_is_honoured(tmp_path: Path) -> None:
    inst = parse_gym_record(_complete_record(test_cmd="tox -e py312"))
    script = (emit_harbor_task(inst, tmp_path) / "tests" / "test.sh").read_text()
    assert "tox -e py312" in script


def test_language_specific_test_command(tmp_path: Path) -> None:
    """The verifier picks its log parser from the test command, so a pytest
    default on a Go task parses nothing and grades every rollout 0.0."""
    inst = parse_gym_record(_complete_record(language="go", FAIL_TO_PASS=["TestFoo"]))
    script = (emit_harbor_task(inst, tmp_path) / "tests" / "test.sh").read_text()
    assert "go test" in script
    assert "pytest" not in script


def test_unknown_language_without_test_cmds_is_skipped(tmp_path: Path) -> None:
    inst = parse_gym_record(
        _complete_record(language="haskell", FAIL_TO_PASS=["Spec.hs"])
    )
    with pytest.raises(UnverifiableGymTask, match="no default for language"):
        emit_harbor_task(inst, tmp_path)

    # ...unless the record carries its own command.
    ok = parse_gym_record(
        _complete_record(
            language="haskell", FAIL_TO_PASS=["Spec.hs"], test_cmd="stack test"
        )
    )
    assert "stack test" in (
        emit_harbor_task(ok, tmp_path) / "tests" / "test.sh"
    ).read_text()


def test_unset_language_infers_python_only_from_pytest_node_ids(tmp_path: Path) -> None:
    py = parse_gym_record(_complete_record(FAIL_TO_PASS=["tests/test_a.py::test_a"]))
    assert py.default_test_cmd() is not None
    ambiguous = parse_gym_record(_complete_record(FAIL_TO_PASS=["SomeSuite.Case"]))
    assert ambiguous.default_test_cmd() is None
    with pytest.raises(UnverifiableGymTask):
        emit_harbor_task(ambiguous, tmp_path)
