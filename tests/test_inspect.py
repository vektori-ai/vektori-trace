"""The static audit of an emitted task.

Each check exists because the thing it catches fails *silently*: the task ships,
harbor runs it, the verifier scores 0, and it reads as a hard task rather than a
broken one. So every check gets a test that it actually fails when it should —
an audit that can only pass is not an audit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vektori_trace.mining.emitter import HarborTask, write_harbor_task
from vektori_trace.mining.env_guard import git_history_scrub
from vektori_trace.mining.inspect import audit_task, audit_tasks, failure_histogram
from vektori_trace.mining.pipeline import build_eval_script

BASE = "a" * 40

TEST_PATCH = """diff --git a/tests/test_calc.py b/tests/test_calc.py
--- a/tests/test_calc.py
+++ b/tests/test_calc.py
@@ -1,2 +1,5 @@
 from calc import add
+
+
+def test_add_negatives():
+    assert add(-1, -2) == -3
"""

ORACLE = """diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""

F2P = ["tests/test_calc.py::test_add_negatives"]


def _dockerfile(base: str = BASE) -> str:
    return (
        "FROM python:3.12-slim\n"
        "WORKDIR /workspace\n"
        f"RUN git reset --hard {base} && git clean -fdx\n" + git_history_scrub(base)
    )


def _write_task(
    tmp_path: Path,
    *,
    base: str = BASE,
    dockerfile: str | None = None,
    f2p: list[str] | None = None,
    shipped_f2p: list[str] | None = None,
    oracle: str = ORACLE,
    test_patch: str = TEST_PATCH,
    name: str = "probe",
) -> Path:
    f2p = F2P if f2p is None else f2p
    task = HarborTask(
        name=name,
        org="vektori",
        description="d",
        instruction="i",
        oracle_diff=oracle,
        repo2env={
            "pipeline": "pr_runtime",
            "pr_runtime": {"base_commit": base, "fail_to_pass": f2p, "pass_to_pass": []},
        },
        environment_dockerfile=_dockerfile(base) if dockerfile is None else dockerfile,
        test_script=build_eval_script(
            base, test_patch=test_patch, test_cmds=["pytest"], fail_to_pass=f2p
        ),
        aux_files={
            "tests/f2p.json": json.dumps(f2p if shipped_f2p is None else shipped_f2p),
            "tests/p2p.json": json.dumps([]),
        },
    )
    return write_harbor_task(task, tmp_path)


def test_a_well_formed_task_passes_every_check(tmp_path: Path) -> None:
    audit = audit_task(_write_task(tmp_path))
    assert audit.ok, audit.details


# ---------------------------------------------------------------------------
# 1. base commit right
# ---------------------------------------------------------------------------


def test_dockerfile_resetting_to_a_different_commit_is_caught(tmp_path: Path) -> None:
    """The oracle patch was generated against the declared base. Reset the tree
    somewhere else and the gold patch may not apply at all — a task whose own
    solution fails is unwinnable, and nothing downstream can tell."""
    audit = audit_task(_write_task(tmp_path, dockerfile=_dockerfile("b" * 40)))

    assert not audit.ok
    assert "dockerfile_resets_to_base_commit" in audit.failures


def test_a_missing_base_commit_is_caught(tmp_path: Path) -> None:
    audit = audit_task(_write_task(tmp_path, base=""))

    assert "base_commit_declared" in audit.failures


# ---------------------------------------------------------------------------
# 2. .git scrubbed
# ---------------------------------------------------------------------------


def test_an_unscrubbed_git_dir_is_caught(tmp_path: Path) -> None:
    """The tree can sit at the base commit while `.git` still holds origin/main,
    the fix commit and the hidden test. Reading the answer out of it needs no
    network — a documented, repeated SWE-bench incident."""
    naked = f"FROM python:3.12-slim\nRUN git reset --hard {BASE} && git clean -fdx\n"
    audit = audit_task(_write_task(tmp_path, dockerfile=naked))

    assert "git_history_scrubbed" in audit.failures
    assert "remote_removed" in audit.details["git_history_scrubbed"]


def test_a_partial_scrub_is_still_a_failure(tmp_path: Path) -> None:
    """Removing the remote but leaving tags and the reflog still leaves the fix
    reachable. A partial guard is not a guard."""
    partial = (
        f"FROM python:3.12-slim\nRUN git reset --hard {BASE} && git clean -fdx\n"
        "RUN git remote remove origin\n"
    )
    audit = audit_task(_write_task(tmp_path, dockerfile=partial))

    assert "git_history_scrubbed" in audit.failures


# ---------------------------------------------------------------------------
# 3. F2P names actually in the test patch
# ---------------------------------------------------------------------------


def test_an_f2p_file_not_in_the_test_patch_is_caught(tmp_path: Path) -> None:
    """The verifier looks up these exact node ids in the suite output. A name
    the test_patch never adds is never collected, never runs, never passes: the
    task scores 0 for everyone forever and reads as merely hard."""
    audit = audit_task(_write_task(tmp_path, f2p=["tests/test_ghost.py::test_nope"]))

    assert "f2p_files_in_test_patch" in audit.failures
    assert "test_ghost.py" in audit.details["f2p_files_in_test_patch"]


def test_no_declared_f2p_is_caught(tmp_path: Path) -> None:
    audit = audit_task(_write_task(tmp_path, f2p=[]))

    assert "f2p_declared" in audit.failures


def test_shipped_f2p_drifting_from_task_toml_is_caught(tmp_path: Path) -> None:
    """`tests/f2p.json` is what the verifier reads; `task.toml` is what every
    report quotes. When they disagree the task is graded on one and described by
    the other."""
    audit = audit_task(_write_task(tmp_path, shipped_f2p=["tests/test_calc.py::test_other"]))

    assert "shipped_f2p_matches_task_toml" in audit.failures


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------


def test_a_missing_oracle_is_caught(tmp_path: Path) -> None:
    audit = audit_task(_write_task(tmp_path, oracle=""))

    assert "oracle_patch_present" in audit.failures


def test_an_oracle_that_patches_the_graded_tests_is_caught(tmp_path: Path) -> None:
    """If the gold patch also rewrites the tests it is graded by, it satisfies
    F2P by moving the ruler — and the validity proof passes for the wrong
    reason."""
    audit = audit_task(_write_task(tmp_path, oracle=ORACLE + TEST_PATCH))

    assert "oracle_excludes_test_files" in audit.failures


# ---------------------------------------------------------------------------
# Corpus-level behaviour
# ---------------------------------------------------------------------------


def test_a_broken_task_does_not_stop_the_audit(tmp_path: Path) -> None:
    """The point is a histogram over a corpus, so one unreadable task must not
    take the rest of the audit with it."""
    good = _write_task(tmp_path / "a", name="good")
    broken = tmp_path / "b" / "broken"
    broken.mkdir(parents=True)

    audits = audit_tasks([good, broken])

    assert len(audits) == 2
    assert audits[0].ok
    assert not audits[1].ok


def test_a_task_with_no_toml_is_reported_not_raised(tmp_path: Path) -> None:
    empty = tmp_path / "nothing"
    empty.mkdir()

    audit = audit_task(empty)

    assert audit.failures == ["task_toml_present"]


def test_unparseable_toml_is_reported_not_raised(tmp_path: Path) -> None:
    d = tmp_path / "bad"
    d.mkdir()
    (d / "task.toml").write_text("this is not = = toml")

    audit = audit_task(d)

    assert "task_toml_parses" in audit.failures


def test_the_histogram_counts_per_check(tmp_path: Path) -> None:
    a = _write_task(tmp_path / "a", name="one", f2p=["tests/test_ghost.py::test_x"])
    b = _write_task(tmp_path / "b", name="two", f2p=["tests/test_ghost.py::test_y"])
    c = _write_task(tmp_path / "c", name="three")

    hist = failure_histogram(audit_tasks([a, b, c]))

    assert hist["f2p_files_in_test_patch"] == 2
    assert "git_history_scrubbed" not in hist


@pytest.mark.parametrize(
    "node_id,expected_file",
    [
        ("tests/test_calc.py::test_add_negatives", "tests/test_calc.py"),
        ("tests/test_calc.py::TestClass::test_method", "tests/test_calc.py"),
        ("tests/test_calc.py", "tests/test_calc.py"),
    ],
)
def test_node_id_forms_all_resolve_to_their_file(
    tmp_path: Path, node_id: str, expected_file: str
) -> None:
    """pytest node ids come in three shapes and the check has to read the path
    out of all of them — a parse miss here would look like a missing test."""
    audit = audit_task(_write_task(tmp_path, f2p=[node_id]))

    assert audit.checks["f2p_files_in_test_patch"], audit.details
