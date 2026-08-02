"""The validation stage must not delete git-ignored build artifacts.

`git clean -fdx` removes ignored files. For any project that derives its
version from git, those are generated at install time and the package stops
importing without them: prefect's `.gitignore` lists
`src/prefect/_build_info.py`, written by `hatch_build.py`.

The failure is silent, which is why it needs a test. Deleting that file made
`import prefect` fail, which made pytest die while parsing config — before
collecting a single test. The stage emitted an empty START/END block,
`parse_logs` returned {}, F2P came out 0, and the PR skipped as
`no_fail_to_pass`. That bucket name reads as "this PR has no failing test"
when what actually happened is "validation never ran", and it zeroed the
yield of a whole prefect mining run (100 PRs, 0 tasks).

Measured in the prefect bootstrap image: `git clean -fd` removes nothing,
`git clean -fdx` removes exactly the two generated files.
"""

from __future__ import annotations

from vektori_trace.mining.validate import _build_stage_script

BASE = "d630146acb690f2967e89c3f429ec3425644d87a"


def _script() -> str:
    return _build_stage_script(
        BASE,
        apply_patch=None,
        apply_test_patch="--- a/tests/test_x.py\n+++ b/tests/test_x.py\n",
        test_cmds=["python -m pytest -q tests/test_x.py"],
    )


def test_git_clean_does_not_remove_ignored_files() -> None:
    """No `-x` on git clean: ignored files are build artifacts we need."""
    script = _script()
    clean_lines = [ln for ln in script.splitlines() if ln.startswith("git clean")]
    assert clean_lines, "expected a git clean step in the stage script"
    for line in clean_lines:
        flags = line.split()[2]  # e.g. "-fd"
        assert "x" not in flags, (
            f"git clean must not pass -x (found {flags!r}): it deletes generated "
            "build artifacts like prefect's src/prefect/_build_info.py, which "
            "silently zeroes F2P detection"
        )


def test_git_clean_still_removes_untracked_files() -> None:
    """Dropping -x must not turn the clean into a no-op."""
    script = _script()
    assert "git clean -fd" in script


def test_stage_script_still_resets_to_base_commit() -> None:
    assert f"git reset --hard {BASE}" in _script()
