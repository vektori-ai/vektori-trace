"""Emitted task.toml conforms to Harbor's actual schema.

The emitter deliberately doesn't *depend* on Harbor to write the format — the
whole point is that datasets stay portable. But not depending on it is not a
reason to guess at it, and asserting our own hand-written expectations proves
nothing. So the check loads emitted files through Harbor's real `TaskConfig`,
which is also what makes a schema change show up here rather than as tasks that
load with fields quietly missing.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from harbor.models.task.config import TaskConfig

from vektori_trace.mining.emitter import TASK_SCHEMA_VERSION, HarborTask, write_harbor_task


@pytest.fixture
def task() -> HarborTask:
    return HarborTask(
        name="psf__requests-1234",
        org="vektori",
        description="Fix the redirect handling",
        instruction="# Fix it\n",
        oracle_diff="diff --git a/x b/x\n",
        repo2env={"source": "pr_runtime"},
        keywords=["requests", "pr_runtime"],
        environment_dockerfile="FROM python:3.12-slim\n",
        test_script="#!/bin/bash\nexit 0\n",
    )


def _loaded(task: HarborTask, tmp_path: Path) -> TaskConfig:
    task_dir = write_harbor_task(task, tmp_path)
    raw = tomllib.loads((task_dir / "task.toml").read_text())
    return TaskConfig.model_validate(raw)


def test_emitted_task_loads_through_harbors_own_model(task, tmp_path: Path) -> None:
    cfg = _loaded(task, tmp_path)
    assert cfg.task is not None
    assert cfg.task.name == "vektori/psf__requests-1234"
    assert cfg.task.description == "Fix the redirect handling"


def test_schema_version_is_declared_under_the_current_key(task, tmp_path: Path) -> None:
    """`version` still works via Harbor's `handle_version_rename` shim, so this
    was never a live failure — but it is a deprecation shim, and we were
    declaring 1.0, which is not the schema we conform to."""
    task_dir = write_harbor_task(task, tmp_path)
    raw = tomllib.loads((task_dir / "task.toml").read_text())

    assert "version" not in raw
    assert raw["schema_version"] == TASK_SCHEMA_VERSION
    assert _loaded(task, tmp_path).schema_version == TASK_SCHEMA_VERSION


def test_keywords_land_where_harbor_reads_them(task, tmp_path: Path) -> None:
    """Under [metadata] they parsed fine and were never read, so every emitted
    task had `task.keywords == []`."""
    cfg = _loaded(task, tmp_path)
    assert cfg.task is not None
    assert cfg.task.keywords == ["requests", "pr_runtime"]


def test_our_provenance_survives_the_round_trip(task, tmp_path: Path) -> None:
    """[metadata] is Harbor's free-form table; our repo2env subtable is how a
    number stays re-derivable from the artifact."""
    cfg = _loaded(task, tmp_path)
    assert cfg.metadata["repo2env"]["source"] == "pr_runtime"
    assert cfg.metadata["repo2env"]["content_hash"].startswith("sha256:")
    assert cfg.metadata["difficulty"] == "medium"
    assert cfg.metadata["category"] == "bugfix"


def test_timeouts_survive_the_round_trip(task, tmp_path: Path) -> None:
    cfg = _loaded(task, tmp_path)
    assert cfg.agent.timeout_sec == 1800.0
    assert cfg.verifier.timeout_sec == 300.0


def test_text_only_task_also_conforms(tmp_path: Path) -> None:
    """The lite path emits no environment/ or tests/ at all."""
    lite = HarborTask(
        name="psf__requests-9",
        org="vektori",
        description="d",
        instruction="i",
        oracle_diff="diff",
        repo2env={},
    )
    cfg = _loaded(lite, tmp_path)
    assert cfg.task is not None
    assert cfg.metadata["repo2env"]["reward_kinds"] == ["diff_similarity"]
