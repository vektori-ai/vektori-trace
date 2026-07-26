"""`require_linked_issue` is a gate, not decoration.

The CLI's `--no-require-linked-issue` help text says it keeps "PRs with no
'Fixes #N' trailer" — which was only true once `_pre_filter` actually looked
at the option. Before that the flag changed nothing.
"""

from __future__ import annotations

import pytest

from vektori_trace.mining.bootstrap.spec import BootstrapResult, LanguageHint
from vektori_trace.mining.github import PullRequestSummary
from vektori_trace.mining.pipeline import PRRuntimePipeline
from vektori_trace.mining.spec import PipelineInput, PRRuntimeOptions, RepoSpec


def _pr(body: str) -> PullRequestSummary:
    return PullRequestSummary(
        number=7,
        title="Handle empty input",
        body=body,
        state="MERGED",
        merged_at="2025-01-01T00:00:00Z",
        base_ref="main",
        base_sha="a" * 40,
        head_sha="b" * 40,
        is_draft=False,
        url="https://github.com/o/n/pull/7",
        changed_files=["src/app.py"],
    )


def _pipeline(*, require_linked_issue: bool) -> PRRuntimePipeline:
    return PRRuntimePipeline(
        PipelineInput(repo=RepoSpec(url="o/n")),
        PRRuntimeOptions(require_linked_issue=require_linked_issue),
        bootstrap=BootstrapResult(
            image_digest="local/img@sha256:" + "0" * 64,
            image_tag="local/img:test",
            language=LanguageHint.PYTHON,
            repo="o/n",
            ref="c" * 40,
            rebuild_cmds=[],
            test_cmds=["pytest"],
            smoke_passed=True,
            iterations=1,
            build_time_sec=1.0,
            llm_provider="test/none",
        ),
    )


NO_TRAILER = "The parser crashes on empty input and it should not."
WITH_TRAILER = "The parser crashes on empty input. Fixes #42"


def test_pr_without_linked_issue_is_skipped_when_required():
    assert _pipeline(require_linked_issue=True)._pre_filter(_pr(NO_TRAILER)) == "no_linked_issue"


def test_pr_without_linked_issue_survives_when_not_required():
    assert _pipeline(require_linked_issue=False)._pre_filter(_pr(NO_TRAILER)) is None


def test_pr_with_linked_issue_survives_either_way():
    assert _pipeline(require_linked_issue=True)._pre_filter(_pr(WITH_TRAILER)) is None
    assert _pipeline(require_linked_issue=False)._pre_filter(_pr(WITH_TRAILER)) is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
