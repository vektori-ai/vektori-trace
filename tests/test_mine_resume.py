"""One bad PR must not end the sweep, and a crashed sweep must resume.

A mining run is hours of Docker work over hundreds of PRs. Before this, any
exception after the diff fetch — a Docker daemon hiccup, a network blip inside
`validate_pr` — propagated out of `run()` and every unprocessed PR was lost,
with no record of which ones had already been decided.
"""

from __future__ import annotations

import json

import pytest

from vektori_trace.mining.bootstrap.spec import BootstrapResult, LanguageHint
from vektori_trace.mining.github import PullRequestSummary
from vektori_trace.mining.pipeline import CHECKPOINT_FILENAME, PRRuntimePipeline
from vektori_trace.mining.spec import PipelineInput, PRRuntimeOptions, RepoSpec

DIFF = """diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -1,1 +1,1 @@
-old
+new
diff --git a/tests/test_app.py b/tests/test_app.py
--- a/tests/test_app.py
+++ b/tests/test_app.py
@@ -1,0 +1,2 @@
+def test_new():
+    assert True
"""


def _pr(number: int) -> PullRequestSummary:
    return PullRequestSummary(
        number=number,
        title=f"Fix a real bug {number}",
        body=f"Something is broken and this fixes it. Fixes #{number}",
        state="MERGED",
        merged_at="2025-01-01T00:00:00Z",
        base_ref="main",
        base_sha="a" * 40,
        head_sha="b" * 40,
        is_draft=False,
        url=f"https://github.com/o/n/pull/{number}",
        changed_files=["src/app.py", "tests/test_app.py"],
    )


class FakeSandbox:
    def cleanup(self) -> None:
        pass


class FakeProvider:
    """Stands in for `vektori_trace.mining.github`."""

    def __init__(self, prs):
        self.prs = prs
        self.diffs_fetched: list[int] = []

    def list_merged_prs(self, owner, name, **kwargs):
        return self.prs

    def fetch_pr_diff(self, owner, name, number, token=None):
        self.diffs_fetched.append(number)
        return DIFF

    def fetch_issue(self, owner, name, number, token=None):
        return None


def _bootstrap() -> BootstrapResult:
    return BootstrapResult(
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
    )


def _pipeline(monkeypatch, prs, *, validate_side_effect) -> tuple[PRRuntimePipeline, FakeProvider]:
    pipeline = PRRuntimePipeline(
        PipelineInput(repo=RepoSpec(url="o/n")),
        PRRuntimeOptions(require_new_test_funcs=False, require_linked_issue=False),
        bootstrap=_bootstrap(),
    )
    provider = FakeProvider(prs)
    monkeypatch.setattr("vektori_trace.mining.pipeline.github", provider)
    monkeypatch.setattr(
        "vektori_trace.mining.pipeline.resolve_repo_token", lambda *a, **k: None
    )
    monkeypatch.setattr(
        PRRuntimePipeline, "_start_validation_sandbox", lambda self: FakeSandbox()
    )
    monkeypatch.setattr("vektori_trace.mining.validate.validate_pr", validate_side_effect)
    return pipeline, provider


class _Outcome:
    status = "verified"
    reason = ""

    def __init__(self, f2p):
        self.fail_to_pass = f2p
        self.pass_to_pass = []


def test_one_pr_exception_does_not_kill_the_sweep(tmp_path, monkeypatch):
    """PR #2 blows up inside validation; #1 and #3 still emit."""

    def validate(*, base_commit, **kwargs):
        # The pipeline hands every PR the same base_sha, so key off call order.
        validate.calls += 1
        if validate.calls == 2:
            raise RuntimeError("docker daemon went away")
        return _Outcome(["tests/test_app.py::test_new"])

    validate.calls = 0

    pipeline, _ = _pipeline(monkeypatch, [_pr(1), _pr(2), _pr(3)], validate_side_effect=validate)
    result = pipeline.run(tmp_path)

    assert result.emitted == 2
    emitted_dirs = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert emitted_dirs == ["o__n-1", "o__n-3"]
    assert result.skip_reasons["unexpected_error:RuntimeError"] == 1


def test_checkpoint_records_every_pr_incrementally(tmp_path, monkeypatch):
    """The checkpoint is on disk while the sweep is still running, not after."""
    seen_during: list[dict] = []

    def validate(**kwargs):
        path = tmp_path / CHECKPOINT_FILENAME
        seen_during.append(
            json.loads(path.read_text())["processed"] if path.exists() else {}
        )
        return _Outcome(["tests/test_app.py::test_new"])

    pipeline, _ = _pipeline(monkeypatch, [_pr(1), _pr(2), _pr(3)], validate_side_effect=validate)
    pipeline.run(tmp_path)

    # By the time PR #3 is validated, #1 and #2 are already recorded.
    assert seen_during[0] == {}
    assert seen_during[1] == {"1": "emitted"}
    assert seen_during[2] == {"1": "emitted", "2": "emitted"}
    final = json.loads((tmp_path / CHECKPOINT_FILENAME).read_text())["processed"]
    assert final == {"1": "emitted", "2": "emitted", "3": "emitted"}


def test_second_run_skips_already_processed_prs(tmp_path, monkeypatch):
    """A restart resumes: recorded PRs aren't re-fetched or re-validated."""
    (tmp_path / CHECKPOINT_FILENAME).write_text(
        json.dumps({"version": 1, "processed": {"1": "emitted", "2": "no_test_patch"}})
    )

    def validate(**kwargs):
        return _Outcome(["tests/test_app.py::test_new"])

    pipeline, provider = _pipeline(
        monkeypatch, [_pr(1), _pr(2), _pr(3)], validate_side_effect=validate
    )
    result = pipeline.run(tmp_path)

    assert provider.diffs_fetched == [3]  # #1 and #2 never touched again
    assert result.emitted == 1
    assert result.skip_reasons["already_processed"] == 2


def test_corrupt_checkpoint_is_ignored(tmp_path, monkeypatch):
    (tmp_path / CHECKPOINT_FILENAME).write_text("{not json")

    def validate(**kwargs):
        return _Outcome(["tests/test_app.py::test_new"])

    pipeline, provider = _pipeline(monkeypatch, [_pr(1)], validate_side_effect=validate)
    result = pipeline.run(tmp_path)
    assert result.emitted == 1
    assert provider.diffs_fetched == [1]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
