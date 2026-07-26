"""One sandbox is reused across every PR, so its dependencies can be wrong.

The sandbox is provisioned at bootstrap HEAD; each PR's stages run at that
PR's base commit. When the manifest moved between the two, the F2P/P2P oracle
was computed against dependency versions the base commit never asked for —
and nothing said so. Detection only: the task still ships, with the caveat.
"""

from __future__ import annotations

import pytest

from vektori_trace.mining.bootstrap.docker import ExecResult
from vektori_trace.mining.validate import (
    dependency_manifests_for,
    detect_dependency_drift,
    manifests_differ,
)

OLD = "requests==2.28.0\n"
NEW = "requests==2.31.0\n"


def test_identical_manifests_are_not_drift():
    assert manifests_differ({"requirements.txt": OLD}, {"requirements.txt": OLD}) is False


def test_changed_manifest_is_drift():
    assert manifests_differ({"requirements.txt": OLD}, {"requirements.txt": NEW}) is True


def test_manifest_absent_on_both_sides_is_not_drift():
    assert manifests_differ({"go.mod": None}, {"go.mod": None}) is False


def test_manifest_added_between_commits_is_drift():
    assert manifests_differ({"requirements.txt": None}, {"requirements.txt": NEW}) is True


def test_manifest_paths_per_language():
    assert "requirements.txt" in dependency_manifests_for("python")
    assert "go.mod" in dependency_manifests_for("go")
    assert dependency_manifests_for("unknown") == ()
    assert dependency_manifests_for(None) == ()


class FakeSandbox:
    """Answers `git show <ref>:<path>` from a per-ref content map."""

    def __init__(self, contents: dict[str, dict[str, str]]):
        self.contents = contents
        self.calls: list[str] = []

    def exec(self, command: str, *, timeout: int = 300) -> ExecResult:
        self.calls.append(command)
        ref, _, path = command.split("show ", 1)[1].partition(":")
        body = self.contents.get(ref, {}).get(path)
        if body is None:
            return ExecResult(exit_code=128, stdout="", stderr="does not exist", duration_sec=0.0)
        return ExecResult(exit_code=0, stdout=body, stderr="", duration_sec=0.0)


def test_detect_drift_reads_both_refs_via_git_show():
    sandbox = FakeSandbox(
        {"boot": {"requirements.txt": OLD}, "base": {"requirements.txt": NEW}}
    )
    assert (
        detect_dependency_drift(
            sandbox, bootstrap_ref="boot", base_commit="base", language="python"
        )
        is True
    )
    assert any("show boot:requirements.txt" in c for c in sandbox.calls)


def test_detect_no_drift_when_manifests_match():
    sandbox = FakeSandbox(
        {"boot": {"requirements.txt": OLD}, "base": {"requirements.txt": OLD}}
    )
    assert (
        detect_dependency_drift(
            sandbox, bootstrap_ref="boot", base_commit="base", language="python"
        )
        is False
    )


def test_same_commit_skips_the_check_entirely():
    sandbox = FakeSandbox({})
    assert (
        detect_dependency_drift(
            sandbox, bootstrap_ref="abc", base_commit="abc", language="python"
        )
        is False
    )
    assert sandbox.calls == []


def test_unreadable_sandbox_is_not_reported_as_drift():
    class Broken:
        def exec(self, command, *, timeout=300):
            raise RuntimeError("docker gone")

    assert (
        detect_dependency_drift(
            Broken(), bootstrap_ref="boot", base_commit="base", language="python"
        )
        is False
    )


def test_drift_flag_reaches_the_emitted_task_metadata():
    """The detection is only useful if a consumer can see it on the task."""
    from vektori_trace.mining.bootstrap.spec import BootstrapResult, LanguageHint
    from vektori_trace.mining.github import PullRequestSummary
    from vektori_trace.mining.pipeline import PRRuntimePipeline
    from vektori_trace.mining.spec import PipelineInput, PRRuntimeOptions, RepoSpec

    pipeline = PRRuntimePipeline(
        PipelineInput(repo=RepoSpec(url="o/n")),
        PRRuntimeOptions(),
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
    pr = PullRequestSummary(
        number=7,
        title="Fix it",
        body="Fixes #1",
        state="MERGED",
        merged_at="2025-01-01T00:00:00Z",
        base_ref="main",
        base_sha="a" * 40,
        head_sha="b" * 40,
        is_draft=False,
        url="https://github.com/o/n/pull/7",
        changed_files=["src/app.py"],
    )
    task = pipeline._build_task(
        pr,
        "patch",
        "test patch",
        fail_to_pass=["t"],
        pass_to_pass=[],
        validation_status="verified",
        dependency_drift_detected=True,
    )
    assert task.repo2env["pr_runtime"]["dependency_drift_detected"] is True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
