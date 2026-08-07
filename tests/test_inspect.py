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
        "RUN command -v tmux >/dev/null 2>&1 || apt-get install -y tmux || true\n"
        f"RUN git reset --hard {base} && git clean -fd\n" + git_history_scrub(base)
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
# 2b. git clean doesn't delete generated build artifacts
# ---------------------------------------------------------------------------


def test_git_clean_dash_x_is_caught(tmp_path: Path) -> None:
    """`-x` deletes git-ignored files. For hatch-vcs/setuptools_scm repos that
    includes the generated version file `import <pkg>` needs — prefect's
    `_build_info.py`, hatch's `_version.py` — so every rollout against that
    task silently scores reward=0 no matter what the agent does."""
    dockerfile = f"FROM python:3.12-slim\nRUN git reset --hard {BASE} && git clean -fdx\n"
    audit = audit_task(_write_task(tmp_path, dockerfile=dockerfile))

    assert "git_clean_preserves_build_artifacts" in audit.failures


def test_git_clean_dash_x_as_separate_token_is_caught(tmp_path: Path) -> None:
    """`-f -d -x` as separate tokens is exactly as unsafe as `-fdx` combined."""
    dockerfile = f"FROM python:3.12-slim\nRUN git reset --hard {BASE} && git clean -f -d -x\n"
    audit = audit_task(_write_task(tmp_path, dockerfile=dockerfile))

    assert "git_clean_preserves_build_artifacts" in audit.failures


def test_git_clean_dash_capital_x_is_caught(tmp_path: Path) -> None:
    """`-X` removes only ignored files — strictly the same failure mode as
    lowercase `-x` for this purpose."""
    dockerfile = f"FROM python:3.12-slim\nRUN git reset --hard {BASE} && git clean -fd -X\n"
    audit = audit_task(_write_task(tmp_path, dockerfile=dockerfile))

    assert "git_clean_preserves_build_artifacts" in audit.failures


def test_git_clean_dash_fd_passes(tmp_path: Path) -> None:
    """`-fd` (no `x`) is the safe form and must not be flagged."""
    dockerfile = f"FROM python:3.12-slim\nRUN git reset --hard {BASE} && git clean -fd\n"
    audit = audit_task(_write_task(tmp_path, dockerfile=dockerfile))

    assert "git_clean_preserves_build_artifacts" not in audit.failures


def test_git_clean_dash_dash_exclude_is_not_a_false_positive(tmp_path: Path) -> None:
    """`--exclude=<pattern>` contains the letter `x` but does the opposite of
    `-x`: it protects a pattern from the clean rather than deleting ignored
    files. Must not be flagged."""
    dockerfile = (
        f"FROM python:3.12-slim\n"
        f"RUN git reset --hard {BASE} && git clean -fd --exclude=.venv\n"
    )
    audit = audit_task(_write_task(tmp_path, dockerfile=dockerfile))

    assert "git_clean_preserves_build_artifacts" not in audit.failures


# ---------------------------------------------------------------------------
# 2c. tmux baked into the image
# ---------------------------------------------------------------------------


def test_missing_tmux_is_caught(tmp_path: Path) -> None:
    """terminus-2 drives the container through a tmux pane and nothing else, so
    an image without tmux fails every rollout at "Failed to start tmux session"
    — and the task's runtime allowlist has no PyPI/GitHub, so harbor's
    install-at-runtime fallback can't recover it."""
    dockerfile = f"FROM python:3.12-slim\nRUN git reset --hard {BASE} && git clean -fd\n"
    audit = audit_task(_write_task(tmp_path, dockerfile=dockerfile))

    assert "dockerfile_installs_tmux" in audit.failures


def test_tmux_mentioned_only_in_a_comment_is_not_enough(tmp_path: Path) -> None:
    """The emitted Dockerfile carries a long comment explaining why tmux must be
    baked in. A substring check would be satisfied by that comment alone, so
    deleting the RUN line and keeping the prose would pass the very audit meant
    to catch it."""
    dockerfile = (
        f"FROM python:3.12-slim\n"
        f"# Defensive: terminus-2 drives the container through a tmux pane, so\n"
        f"# tmux must be installed at build time or every rollout fails.\n"
        f"RUN git reset --hard {BASE} && git clean -fd\n"
    )
    audit = audit_task(_write_task(tmp_path, dockerfile=dockerfile))

    assert "dockerfile_installs_tmux" in audit.failures


def test_tmux_install_passes(tmp_path: Path) -> None:
    """The defensive install line the pipeline emits satisfies the check."""
    dockerfile = (
        f"FROM python:3.12-slim\n"
        f"RUN command -v tmux >/dev/null 2>&1 || \\\n"
        f"    (apt-get update && apt-get install -y --no-install-recommends tmux) || true\n"
        f"RUN git reset --hard {BASE} && git clean -fd\n"
    )
    audit = audit_task(_write_task(tmp_path, dockerfile=dockerfile))

    assert "dockerfile_installs_tmux" not in audit.failures


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


# ---------------------------------------------------------------------------
# The CLI guards these checks exist to drive
# ---------------------------------------------------------------------------


def test_dockerfile_without_test_cmd_fails_before_the_bootstrap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Otherwise it fails *after* every PR has been silently discarded: the
    supplied Dockerfile skips the agent that discovers the test command, F2P
    comes from running the suite, so every candidate skips as no_fail_to_pass
    and it reads as "this repo has no minable PRs"."""
    from vektori_trace.cli import main

    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12-slim\n")

    rc = main(["mine", "--repo", "o/n", "--dockerfile", str(df), "--out", str(tmp_path)])

    assert rc == 2
    assert "--dockerfile needs --test-cmd" in capsys.readouterr().err


def test_blank_test_cmds_do_not_satisfy_the_guard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from vektori_trace.cli import main

    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12-slim\n")

    rc = main(
        ["mine", "--repo", "o/n", "--dockerfile", str(df), "--test-cmd", "  ", "--out", str(tmp_path)]
    )

    assert rc == 2


# ---------------------------------------------------------------------------
# Provenance for the languages `--language` accepts, not just Python
# ---------------------------------------------------------------------------

GO_TEST_PATCH = """diff --git a/calc_test.go b/calc_test.go
--- a/calc_test.go
+++ b/calc_test.go
@@ -1,2 +1,6 @@
 package calc
+
+func TestAddNegatives(t *testing.T) {
+	if Add(-1, -2) != -3 { t.Fail() }
+}
"""

RUST_TEST_PATCH = """diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,2 +1,6 @@
 pub fn add(a: i32, b: i32) -> i32 { a - b }
+#[cfg(test)]
+mod tests {
+    #[test]
+    fn add_negatives() { assert_eq!(super::add(-1, -2), -3); }
+}
"""


def test_a_go_f2p_absent_from_the_patch_is_caught(tmp_path: Path) -> None:
    """Go node ids carry no file path, so the file check can say nothing about
    them. The old `.py`-only regex made `f2p_files` empty and the provenance
    check passed by default — silently, for exactly the runners least likely to
    have been eyeballed."""
    audit = audit_task(
        _write_task(tmp_path, f2p=["TestGhost"], test_patch=GO_TEST_PATCH)
    )

    assert "f2p_names_in_test_patch" in audit.failures
    assert "TestGhost" in audit.details["f2p_names_in_test_patch"]


def test_a_go_f2p_present_in_the_patch_passes(tmp_path: Path) -> None:
    audit = audit_task(
        _write_task(tmp_path, f2p=["TestAddNegatives"], test_patch=GO_TEST_PATCH)
    )

    assert audit.checks["f2p_names_in_test_patch"], audit.details


def test_a_rust_module_path_f2p_is_checked_by_symbol(tmp_path: Path) -> None:
    """`mod::tests::case` — no path, `::` separators."""
    good = audit_task(
        _write_task(tmp_path / "a", f2p=["tests::add_negatives"], test_patch=RUST_TEST_PATCH)
    )
    bad = audit_task(
        _write_task(tmp_path / "b", f2p=["tests::never_written"], test_patch=RUST_TEST_PATCH)
    )

    assert good.checks["f2p_names_in_test_patch"], good.details
    assert "f2p_names_in_test_patch" in bad.failures


def test_a_junit_style_f2p_is_checked_by_symbol(tmp_path: Path) -> None:
    """JUnit's `com.foo.BarTest#testBaz` separates with `#` and `.`."""
    patch = GO_TEST_PATCH.replace("TestAddNegatives", "testAddNegatives")
    audit = audit_task(
        _write_task(tmp_path, f2p=["com.foo.BarTest#testAddNegatives"], test_patch=patch)
    )

    assert audit.checks["f2p_names_in_test_patch"], audit.details


def test_python_f2p_is_still_checked_by_file_and_by_name(tmp_path: Path) -> None:
    audit = audit_task(_write_task(tmp_path))

    assert audit.checks["f2p_files_in_test_patch"]
    assert audit.checks["f2p_names_in_test_patch"]


def test_no_test_patch_at_all_is_not_a_silent_pass(tmp_path: Path) -> None:
    """Unverifiable is not the same as verified."""
    audit = audit_task(_write_task(tmp_path, test_patch=""))

    assert "f2p_names_in_test_patch" in audit.failures
    assert "unverifiable" in audit.details["f2p_names_in_test_patch"]


# ---------------------------------------------------------------------------
# Malformed metadata
# ---------------------------------------------------------------------------


def test_a_non_table_repo2env_does_not_abort_the_corpus(tmp_path: Path) -> None:
    """`repo2env = "bad"` is valid TOML, and a chained .get() on the string
    raises — which would take the whole histogram down with one bad task."""
    d = tmp_path / "weird"
    d.mkdir()
    (d / "task.toml").write_text('schema_version = "1.3"\n[metadata]\nrepo2env = "bad"\n')

    audit = audit_task(d)

    assert not audit.ok
    assert "metadata_well_formed" in audit.failures


def test_a_non_list_fail_to_pass_does_not_raise(tmp_path: Path) -> None:
    d = tmp_path / "weird2"
    d.mkdir()
    (d / "task.toml").write_text(
        'schema_version = "1.3"\n[metadata.repo2env.pr_runtime]\n'
        'base_commit = 12345\nfail_to_pass = "not-a-list"\n'
    )

    audit = audit_task(d)

    assert not audit.ok
    assert "base_commit_declared" in audit.failures


def test_malformed_metadata_still_lets_the_rest_of_the_corpus_audit(tmp_path: Path) -> None:
    good = _write_task(tmp_path / "g", name="good")
    weird = tmp_path / "w"
    weird.mkdir()
    (weird / "task.toml").write_text("[metadata]\nrepo2env = 42\n")

    audits = audit_tasks([good, weird, _write_task(tmp_path / "g2", name="good2")])

    assert len(audits) == 3
    assert audits[0].ok and audits[2].ok
    assert not audits[1].ok


# ---------------------------------------------------------------------------
# Two false positives the first real corpus taught us
# ---------------------------------------------------------------------------

MODIFIED_TEST_PATCH = """diff --git a/tests/test_calc.py b/tests/test_calc.py
--- a/tests/test_calc.py
+++ b/tests/test_calc.py
@@ -10,7 +10,7 @@ class TestThing:
     def test_repr(self):
-        assert repr(x) == "old"
+        assert repr(x) == "new"
"""


def test_a_parametrised_f2p_id_is_not_a_false_positive(tmp_path: Path) -> None:
    """`test_pickle[0-None]` is generated at run time from one `def test_pickle`
    in the source, so searching the patch for the expanded id can only ever
    miss. Keeping the suffix flagged 2 of 4 sound structlog tasks."""
    audit = audit_task(
        _write_task(tmp_path, f2p=["tests/test_calc.py::test_add_negatives[0-None]"])
    )

    assert audit.checks["f2p_names_in_test_patch"], audit.details


def test_a_go_subtest_id_is_not_a_false_positive(tmp_path: Path) -> None:
    audit = audit_task(
        _write_task(tmp_path, f2p=["TestAddNegatives/negative_case"], test_patch=GO_TEST_PATCH)
    )

    assert audit.checks["f2p_names_in_test_patch"], audit.details


def test_a_modified_test_counts_as_provenance(tmp_path: Path) -> None:
    """A PR that *modifies* an existing test moves it fail->pass just as
    legitimately as one that adds a new test — and the modified test's `def`
    line then appears only as a context line, never as an added one.
    `structlog-786`'s `test_repr` is exactly that case."""
    audit = audit_task(
        _write_task(
            tmp_path, f2p=["tests/test_calc.py::test_repr"], test_patch=MODIFIED_TEST_PATCH
        )
    )

    assert audit.checks["f2p_names_in_test_patch"], audit.details


def test_a_genuinely_absent_name_is_still_caught(tmp_path: Path) -> None:
    """The loosening must not cost the check its teeth."""
    audit = audit_task(
        _write_task(
            tmp_path, f2p=["tests/test_calc.py::test_never_written"],
            test_patch=MODIFIED_TEST_PATCH,
        )
    )

    assert "f2p_names_in_test_patch" in audit.failures
