"""scaffold_task writes a complete Harbor task dir with no harbor binary.

The LLM call is stubbed; what's under test is the file layout, which used to
depend on shelling out to `harbor task init` with flags that don't exist.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from vektori_trace import taskgen
from vektori_trace.diagnose import Capability, DeficitScore

GENERATED = {
    "task_description": "Read the traceback before editing.",
    "instruction_md": "# Task\nFix the failure.\n",
    "dockerfile": "FROM ubuntu:24.04\nRUN apt-get update\n",
    "test_outputs_py": "def test_it():\n    assert True\n",
    "solve_sh": "#!/bin/bash\necho solved\n",
}


@pytest.fixture
def deficit() -> DeficitScore:
    return DeficitScore(
        capability=Capability(
            id="reads_traceback", name="Reads the traceback", description="…"
        ),
        baseline_rate=0.1,
        incident_rate=0.8,
        gap=0.7,
        prevalence=0.5,
        priority=0.35,
        n_relevant_wins=10,
        n_relevant_losses=8,
    )


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(taskgen, "generate_task_files", lambda *a, **k: dict(GENERATED))


def test_scaffold_task_writes_a_full_task_dir(tmp_path: Path, deficit: DeficitScore) -> None:
    task_dir = taskgen.scaffold_task(deficit, tmp_path)

    assert task_dir == tmp_path / "reads-the-traceback"
    assert (task_dir / "task.toml").exists()
    assert (task_dir / "instruction.md").read_text() == GENERATED["instruction_md"]
    assert (task_dir / "environment" / "Dockerfile").read_text() == GENERATED["dockerfile"]
    assert (task_dir / "tests" / "test_outputs.py").read_text() == GENERATED["test_outputs_py"]
    assert (task_dir / "tests" / "test.sh").exists()


def test_solve_sh_is_the_generated_oracle_and_is_executable(
    tmp_path: Path, deficit: DeficitScore
) -> None:
    """The emitter's default solve.sh git-applies a gold patch. A synthesized
    task has no gold patch, so the generated script must win."""
    task_dir = taskgen.scaffold_task(deficit, tmp_path)
    solve = task_dir / "solution" / "solve.sh"

    assert solve.read_text() == GENERATED["solve_sh"]
    assert solve.stat().st_mode & 0o111
    # An empty patch.diff would look like a real (no-op) oracle to anything
    # that reads solution/ without checking.
    assert not (task_dir / "solution" / "patch.diff").exists()


def test_task_toml_records_the_diagnosis_it_came_from(
    tmp_path: Path, deficit: DeficitScore
) -> None:
    """Every number has to be re-derivable from artifacts on disk."""
    task_dir = taskgen.scaffold_task(deficit, tmp_path)
    cfg = tomllib.loads((task_dir / "task.toml").read_text())

    assert cfg["task"]["name"] == "vektori/reads-the-traceback"
    provenance = cfg["metadata"]["repo2env"]
    assert provenance["source"] == "taskgen"
    assert provenance["capability_id"] == "reads_traceback"
    assert provenance["gap"] == 0.7
    assert provenance["n_relevant_losses"] == 8
