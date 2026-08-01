"""`require_linked_issue` is a gate, not decoration.

The CLI's `--no-require-linked-issue` help text says it keeps "PRs with no
'Fixes #N' trailer" — which was only true once `_pre_filter` actually looked
at the option. Before that the flag changed nothing.
"""

from __future__ import annotations

import pytest

from vektori_trace.mining.bootstrap.spec import BootstrapResult, LanguageHint
from vektori_trace.mining.github import PullRequestSummary
from vektori_trace.mining.pipeline import PRRuntimePipeline, _linked_issue_number
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


# A PR template that *instructs* the author to reference an issue. The example
# reference is the trap: matching it links every PR using this template to the
# same unrelated issue, and the task's instruction.md then describes a different
# bug entirely. Measured on pydantic, this inflated the linked-issue rate from
# 21% to 40% — every one of those extras pointing at issue #123.
PYDANTIC_TEMPLATE = (
    "<!-- Thank you for your contribution! -->\n\n"
    "## Change Summary\n\npydantic-core handles the casting of constraints.\n\n"
    "## Related issue number\n\n"
    '<!-- WARNING: please use "fix #123" style references so the issue is '
    "closed when this PR is merged. -->\n"
)


def test_pr_template_example_reference_is_not_a_linked_issue():
    assert _linked_issue_number(PYDANTIC_TEMPLATE) is None
    assert _pipeline(require_linked_issue=True)._pre_filter(
        _pr(PYDANTIC_TEMPLATE)
    ) == "no_linked_issue"


def test_full_github_issue_url_counts_as_a_link():
    """GitHub closes an issue from the URL spelling too, and repos use both."""
    body = "Fixes https://github.com/pydantic/pydantic/issues/13507.\n" + PYDANTIC_TEMPLATE
    assert _linked_issue_number(body) == 13507
    assert _pipeline(require_linked_issue=True)._pre_filter(_pr(body)) is None


def test_hash_spelling_still_wins_and_still_works():
    assert _linked_issue_number("Fixes #13450.") == 13450
    assert _linked_issue_number("closes #99") == 99
    assert _linked_issue_number("Just a refactor, no issue.") is None
