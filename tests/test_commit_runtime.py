"""Tests for the commit_runtime pipeline.

The filters are exercised against a real git repository built in a tmpdir
rather than mocks, because the thing that broke `pr_runtime` for 100 straight
PRs was a *sandbox* behaviour no faked test could see. `git log` output
formatting, merge-commit parents and shallow-clone limits are exactly that
class of detail.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vektori_trace.mining.commit_runtime import (
    CommitRuntimePipeline,
    build_instruction_from_commit,
    leaked_identifiers,
)
from vektori_trace.mining.git_local import (
    CommitInfo,
    list_commits,
    show_diff,
)
from vektori_trace.mining.spec import CommitRuntimeOptions


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "t@example.com",
            "HOME": str(cwd),
        },
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    d = tmp_path / "repo"
    d.mkdir()
    _git(["init", "-q", "-b", "main"], d)
    (d / "app.py").write_text("def f():\n    return 1\n")
    _git(["add", "-A"], d)
    _git(["commit", "-q", "-m", "initial commit adding the app module"], d)

    (d / "app.py").write_text("def f():\n    return 2\n")
    (d / "test_app.py").write_text("def test_f():\n    assert f() == 2\n")
    _git(["add", "-A"], d)
    _git(["commit", "-q", "-m", "fix: f returned the wrong value on empty input"], d)
    return d


def test_list_commits_parses_subject_body_and_parents(repo: Path) -> None:
    commits = list_commits(repo, limit=10)
    assert len(commits) == 2
    head = commits[0]  # newest first
    assert head.subject == "fix: f returned the wrong value on empty input"
    assert head.parent_sha == commits[1].sha
    assert not head.is_merge
    # A root commit has no parent, so there is no "before" tree to fail in.
    assert commits[1].parent_sha == ""


def test_list_commits_survives_multiline_bodies(repo: Path) -> None:
    """Commit bodies contain newlines; a line-based parse would split records."""
    (repo / "app.py").write_text("def f():\n    return 3\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "fix: crash on reload\n\nLine one.\nLine two.\n"], repo)

    head = list_commits(repo, limit=1)[0]
    assert head.subject == "fix: crash on reload"
    assert "Line one." in head.body and "Line two." in head.body


def test_show_diff_is_parseable_by_the_pr_splitter(repo: Path) -> None:
    """`git show` output must match `gh pr diff` shape or the splitter breaks."""
    from vektori_trace.mining.pipeline import split_patch_and_test_patch

    head = list_commits(repo, limit=1)[0]
    patch, test_patch = split_patch_and_test_patch(show_diff(repo, head.sha))
    assert "app.py" in patch and "test_app.py" not in patch
    assert "test_app.py" in test_patch


def _commit(**kw) -> CommitInfo:
    base = {
        "sha": "a" * 40,
        "parent_sha": "b" * 40,
        "parents": ["b" * 40],
        "author_name": "Dev",
        "author_email": "dev@example.com",
        "authored_at": "2026-03-12T08:15:22Z",
        "subject": "fix: crash when the config file is missing entirely",
        "body": "",
    }
    return CommitInfo(**{**base, **kw})


def _pipeline(**opts) -> CommitRuntimePipeline:
    p = CommitRuntimePipeline.__new__(CommitRuntimePipeline)
    p.options = CommitRuntimeOptions(**opts)
    p.input = None
    return p


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"parents": ["b" * 40, "c" * 40]}, "merge_commit"),
        ({"parent_sha": "", "parents": []}, "root_commit"),
        ({"author_name": "dependabot[bot]"}, "excluded_author"),
        ({"author_name": "renovate[bot]"}, "excluded_author"),
        ({"subject": "wip"}, "message_too_short"),
        ({"subject": "feat: add a brand new exporting subsystem"}, "non_bug_type"),
    ],
)
def test_metadata_filter_rejects(kwargs: dict, expected: str) -> None:
    assert _pipeline()._metadata_filter(_commit(**kwargs)) == expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"subject": "update the docs for the new release process"}, "not_a_bugfix"),
    ],
)
def test_metadata_filter_rejects_with_keyword_gate_on(kwargs: dict, expected: str) -> None:
    assert _pipeline(require_bugfix_keyword=True)._metadata_filter(_commit(**kwargs)) == expected


@pytest.mark.parametrize(
    "subject",
    [
        "fix: crash when the config file is missing entirely",
        "correct the wrong offset used when parsing headers",  # keyword, no prefix
        "resolve the regression introduced in the parser rewrite",
    ],
)
def test_metadata_filter_accepts_bugfixes(subject: str) -> None:
    assert _pipeline()._metadata_filter(_commit(subject=subject)) is None


def test_bugfix_signal_required_without_prefix_or_issue() -> None:
    """A feature commit with no conventional-commit type must not sail through.

    Without a positive signal the candidate reaches validation, which costs a
    container run per commit — the expensive stage this filter exists to guard.
    """
    pipe = _pipeline(require_bugfix_keyword=True)
    plain = _commit(subject="add support for exporting reports as PDF files")
    assert pipe._metadata_filter(plain) == "not_a_bugfix"
    linked = _commit(
        subject="add support for exporting reports as PDF files",
        body="Closes #42",
    )
    assert pipe._metadata_filter(linked) is None


def test_cap_pass_to_pass_is_bounded_and_deterministic() -> None:
    """Two mines of one commit must produce the same P2P set, or tasks drift."""
    p2p = [f"tests/test_x.py::test_{i}" for i in range(200)]
    pipe = _pipeline(max_pass_to_pass=50)
    first = pipe._cap_pass_to_pass(list(p2p))
    second = pipe._cap_pass_to_pass(list(reversed(p2p)))
    assert len(first) == 50
    assert first == second

    assert len(_pipeline(max_pass_to_pass=0)._cap_pass_to_pass(list(p2p))) == 200


def test_instruction_fallback_strips_the_solution_pointer() -> None:
    """The raw-text path must not carry `Closes #N` or the CC prefix through."""
    text = build_instruction_from_commit(
        _commit(subject="fix: crash on missing config", body="Closes #42\n\nThe app died.")
    )
    assert "#42" not in text
    assert "Closes" not in text
    # The conventional-commit prefix is noise in a problem statement.
    assert "**Title:** crash on missing config" in text
    assert "bbbbbbbbbbbb" in text  # base commit is the parent, not the commit


_PATCH = """diff --git a/src/pkg/validators.py b/src/pkg/validators.py
--- a/src/pkg/validators.py
+++ b/src/pkg/validators.py
@@ -1,3 +1,5 @@
+def normalize_rrule_string(value):
+    if value is None:
+        return None
     return value.strip()
"""

_TEST_PATCH = """diff --git a/tests/test_validation.py b/tests/test_validation.py
--- a/tests/test_validation.py
+++ b/tests/test_validation.py
@@ -1,2 +1,4 @@
+def test_normalize_none():
+    assert normalize_rrule_string(None) is None
"""


def test_leaked_identifiers_flags_names_the_patch_introduces() -> None:
    leaky = "**Title:** normalize_rrule_string crashes when input is None"
    assert leaked_identifiers(leaky, _PATCH, _TEST_PATCH) == ["normalize_rrule_string"]


def test_leaked_identifiers_flags_file_stems_and_test_names() -> None:
    assert "validators" in leaked_identifiers("See validators for details.", _PATCH, "")
    assert "test_normalize_none" in leaked_identifiers(
        "Add test_normalize_none coverage.", "", _TEST_PATCH
    )


def test_leaked_identifiers_ignores_clean_symptom_prose() -> None:
    """The gate must not fire on a good instruction, or it drops every task."""
    clean = (
        "**Title:** Passing None to the recurrence rule validator raises an exception\n\n"
        "Providing None as input causes an error during validation. It should be "
        "handled gracefully instead."
    )
    assert leaked_identifiers(clean, _PATCH, _TEST_PATCH) == []


def test_leaked_identifiers_ignores_short_tokens() -> None:
    """Short names appear in ordinary prose; flagging them fires on everything."""
    patch = (
        "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n+def run(x):\n"
    )
    assert leaked_identifiers("The run failed with an error.", patch, "") == []


def test_leaky_instruction_is_dropped_not_downgraded(monkeypatch) -> None:
    """A verified-leaky synthesis must skip the task, never fall back to raw text.

    Falling back would emit the commit message that leaked in the first place —
    shipping exactly the gameable task synthesis exists to prevent.
    """
    pipe = _pipeline()
    pipe.input = type("I", (), {"llm": object()})()
    pipe._synthesized = pipe._synthesis_failures = pipe._leaked = pipe._leak_retries = 0
    pipe._llm_cost_usd = 0.0
    # Both the draft and the stricter retry keep leaking.
    monkeypatch.setattr(
        CommitRuntimePipeline,
        "_call_synthesis",
        lambda self, c, i, forbid: "**Title:** normalize_rrule_string is broken " * 3,
    )
    out = pipe._build_instruction(
        _commit(), issue=None, patch=_PATCH, test_patch=_TEST_PATCH
    )
    assert out is None
    assert pipe._leaked == 1


def test_clean_synthesis_is_kept(monkeypatch) -> None:
    pipe = _pipeline()
    pipe.input = type("I", (), {"llm": object()})()
    pipe._synthesized = pipe._synthesis_failures = pipe._leaked = pipe._leak_retries = 0
    pipe._llm_cost_usd = 0.0
    monkeypatch.setattr(
        CommitRuntimePipeline,
        "_call_synthesis",
        lambda self, c, i, forbid: (
            "**Title:** Validation raises on empty input\n\n"
            "## Description\n\nIt errors instead of returning a value."
        ),
    )
    out = pipe._build_instruction(
        _commit(), issue=None, patch=_PATCH, test_patch=_TEST_PATCH
    )
    assert out is not None and "# Issue" in out
    assert pipe._synthesized == 1 and pipe._leaked == 0


def test_instruction_prefers_the_issue_over_the_commit_message() -> None:
    """The issue describes the symptom; the commit message describes the fix."""
    text = build_instruction_from_commit(
        _commit(subject="fix: patch the validator to check for None"),
        issue=("App crashes on startup", "It exits with a traceback every time."),
    )
    assert "App crashes on startup" in text
    assert "validator" not in text


def test_ai_agent_bots_are_not_excluded() -> None:
    """`[bot]` is not a proxy for "not a real fix".

    Measured on the last 25 prefect commits: a blanket `[bot]` pattern excluded
    18, about half of them real fixes authored by `devin-ai-integration[bot]`.
    One was "Reject unknown fields for FlowRunFilter" (#22659) — a change
    pr_runtime had already emitted as a valid task, so the two pipelines
    disagreed about the same commit.
    """
    agent = _commit(
        author_name="devin-ai-integration[bot]",
        author_email="158243242+devin-ai-integration[bot]@users.noreply.github.com",
        subject="Reject unknown fields for FlowRunFilter and its nested criteria",
        body="Unknown keys were silently accepted and dropped instead of raising.",
    )
    assert _pipeline()._metadata_filter(agent) is None

    gha = _commit(
        author_name="github-actions[bot]",
        subject="fix: correct the stale lease-cleanup invariant",
        body="The documented invariant no longer matches the implementation.",
    )
    assert _pipeline()._metadata_filter(gha) is None


def test_dependency_bumps_are_rejected_on_content_not_just_author() -> None:
    """The author list is a cheap pre-filter, not the actual gate.

    A dependency bump from an unrecognised bot account must still be dropped,
    or narrowing the author list would open a hole.
    """
    bump = _commit(
        author_name="some-unknown-ci-account",
        subject="chore(deps): bump docker/login-action from 4 to 4.5.2",
    )
    assert _pipeline()._metadata_filter(bump) == "non_bug_type"


def test_exclude_authors_option_overrides_the_default_list() -> None:
    custom = _pipeline(exclude_authors=["desertaxle"])
    assert custom._metadata_filter(_commit(author_email="desertaxle@users.noreply.github.com")) == (
        "excluded_author"
    )
    # Overriding replaces the defaults, so dependabot is no longer named —
    # its `chore(deps)` content still is.
    assert custom._metadata_filter(_commit(author_name="dependabot[bot]")) is None
