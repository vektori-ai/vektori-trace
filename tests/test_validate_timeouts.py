"""A timed-out validation stage must drop the candidate, not ship a task.

`sandbox.exec` returns whatever the suite printed before it was killed, and
`parse_logs` reads that partial log perfectly happily. The two stages then
cover different, wall-clock-determined subsets of the suite, so F2P/P2P come
out of a comparison that never happened — and the task that ships looks
well-formed, with the verifier agreeing with its wrong oracle.
"""

from __future__ import annotations

import pytest

from vektori_trace.mining.bootstrap.docker import TIMEOUT_EXIT_CODE, ExecResult
from vektori_trace.mining.validate import validate_pr

# A pre-fix suite where one test fails, and the same suite passing post-fix.
PRE_LOG = """
tests/test_a.py::test_one PASSED
tests/test_a.py::test_two FAILED
tests/test_b.py::test_three PASSED
"""
POST_LOG = """
tests/test_a.py::test_one PASSED
tests/test_a.py::test_two PASSED
tests/test_b.py::test_three PASSED
"""
# What a killed run leaves behind: real, parseable, and missing the rest.
TRUNCATED_LOG = "tests/test_a.py::test_one PASSED\n"


def _ok(stdout: str = "") -> ExecResult:
    return ExecResult(exit_code=0, stdout=stdout, stderr="", duration_sec=0.1)


def _timeout(stdout: str, seconds: int = 600) -> ExecResult:
    return ExecResult(
        exit_code=TIMEOUT_EXIT_CODE,
        stdout=stdout,
        stderr=f"[timeout after {seconds}s]",
        duration_sec=float(seconds),
        timed_out=True,
    )


class FakeSandbox:
    """Answers the git-preflight probes, then returns scripted stage results."""

    def __init__(self, stages: list[ExecResult]):
        self.stages = list(stages)
        self.stage_calls = 0

    def exec(self, command: str, *, timeout: int = 300) -> ExecResult:
        # A validation stage is identified by the marker the stage script
        # echoes around the test run. Preflight commands (git install probe,
        # safe.directory, base-commit fetch) never carry it — and matching on
        # anything looser catches the stage script too, since it sets
        # safe.directory itself.
        if "R2E_START_TEST_OUTPUT" in command:
            self.stage_calls += 1
            return self.stages.pop(0)
        if "command -v git" in command or "cat-file" in command:
            return _ok("OK")
        return _ok()


def _validate(sandbox: FakeSandbox, **kw):
    return validate_pr(
        sandbox=sandbox,
        base_commit="a" * 40,
        patch="diff --git a/x b/x\n",
        test_patch="diff --git a/t b/t\n",
        test_cmds=["pytest -v"],
        **kw,
    )


def test_baseline_two_clean_stages_verify() -> None:
    """The control: without a timeout this corpus yields exactly one F2P."""
    outcome = _validate(FakeSandbox([_ok(PRE_LOG), _ok(POST_LOG)]))

    assert outcome.status == "verified"
    assert outcome.fail_to_pass == ["tests/test_a.py::test_two"]
    assert outcome.pass_to_pass == [
        "tests/test_a.py::test_one",
        "tests/test_b.py::test_three",
    ]


def test_pre_stage_timeout_drops_the_candidate() -> None:
    outcome = _validate(FakeSandbox([_timeout(TRUNCATED_LOG), _ok(POST_LOG)]))

    assert outcome.status == "failed"
    assert "timed out" in outcome.reason
    assert "pre-fix" in outcome.reason
    assert outcome.fail_to_pass == []


def test_pre_stage_timeout_does_not_run_the_post_stage() -> None:
    """No point burning another full suite run on a comparison already void."""
    sandbox = FakeSandbox([_timeout(TRUNCATED_LOG), _ok(POST_LOG)])
    _validate(sandbox)
    assert sandbox.stage_calls == 1


def test_post_stage_timeout_drops_the_candidate() -> None:
    outcome = _validate(FakeSandbox([_ok(PRE_LOG), _timeout(TRUNCATED_LOG)]))

    assert outcome.status == "failed"
    assert "post-fix" in outcome.reason
    assert outcome.fail_to_pass == []


def test_truncated_post_run_would_otherwise_ship_a_missing_regression_guard() -> None:
    """The concrete harm, shown by removing the guard.

    With a complete pre-run and a truncated post-run, `test_three` passed
    before and was never reached after. Without the timeout check it simply
    vanishes from P2P — the task ships with one fewer regression guard than it
    should have, and nothing downstream can tell.
    """
    from vektori_trace.mining.log_parsers import parse_logs

    pre_status = parse_logs(["pytest -v"], PRE_LOG)
    truncated_post = parse_logs(["pytest -v"], TRUNCATED_LOG)

    p2p_complete = [t for t, s in pre_status.items() if s == "PASSED" and truncated_post.get(t) == "PASSED"]
    assert "tests/test_b.py::test_three" not in p2p_complete
    assert p2p_complete == ["tests/test_a.py::test_one"]


def test_timeout_keeps_the_partial_log_for_diagnosis() -> None:
    """Dropped is not the same as unexplained — the operator still needs to see
    how far the suite got before deciding to raise the timeout."""
    outcome = _validate(FakeSandbox([_timeout(TRUNCATED_LOG)]))
    assert "test_one" in outcome.pre_log


def test_timeout_reason_names_the_configured_budget() -> None:
    outcome = _validate(FakeSandbox([_timeout(TRUNCATED_LOG, seconds=90)]), timeout=90)
    assert "90s" in outcome.reason


# ---------------------------------------------------------------------------
# The flag itself
# ---------------------------------------------------------------------------


def test_timed_out_result_is_not_ok() -> None:
    assert _timeout("").ok is False
    assert _ok().ok is True


def test_a_container_command_may_exit_124_without_us_killing_it() -> None:
    """Why `timed_out` exists rather than testing for 124.

    A script inside the container can run GNU `timeout` itself and legitimately
    exit 124. Treating the code as proof of our own timeout would drop a
    perfectly good candidate.
    """
    self_timed = ExecResult(
        exit_code=TIMEOUT_EXIT_CODE, stdout=PRE_LOG, stderr="", duration_sec=1.0
    )
    assert self_timed.timed_out is False

    outcome = _validate(FakeSandbox([self_timed, _ok(POST_LOG)]))
    assert outcome.status == "verified"


@pytest.mark.parametrize("code", [0, 1, 2, 5])
def test_nonzero_exit_is_normal_for_a_failing_suite(code: int) -> None:
    """pytest exits non-zero whenever tests fail, which is the *expected* state
    of the pre-fix stage. Only timeouts invalidate the comparison."""
    pre = ExecResult(exit_code=code, stdout=PRE_LOG, stderr="", duration_sec=1.0)
    outcome = _validate(FakeSandbox([pre, _ok(POST_LOG)]))
    assert outcome.status == "verified"
