"""Step A — trajectory resume consistency + desync reporting."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from vektori_trace.resume import (
    ResumeDesyncError,
    measure_desync_rate,
    replay_prefix,
    transcript_implied_files,
)
from vektori_trace.schema import ToolCall, Turn


@dataclass
class FakeExec:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class FakeSandbox:
    """Runs `python3 -c ...` writes for real in a temp dir so the quoting is
    exercised end to end; everything else is faked."""

    def __init__(self, diffs: list[str] | None = None, root: Path | None = None):
        self.cmds: list[str] = []
        self._diffs = list(diffs or [""])
        self._diff_i = 0
        self.root = root

    def exec(self, cmd: str, *, timeout: int = 120):
        self.cmds.append(cmd)
        if cmd.startswith("git add"):
            diff = self._diffs[min(self._diff_i, len(self._diffs) - 1)]
            self._diff_i += 1
            return FakeExec(stdout=diff)
        if self.root is not None and (cmd.startswith("python3 -c") or cmd.startswith("cat ")):
            p = subprocess.run(
                cmd, shell=True, cwd=self.root, capture_output=True, text=True
            )
            return FakeExec(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
        return FakeExec(stdout="ok")


def _turns_with_bash(*cmds: str) -> list[Turn]:
    turns = []
    for i, cmd in enumerate(cmds):
        turns.append(
            Turn(
                index=i,
                role="assistant",
                content=None,
                tool_calls=[ToolCall(id=f"c{i}", name="bash", args={"cmd": cmd})],
            )
        )
        turns.append(Turn(index=i + 100, role="tool", content="ok", tool_call_id=f"c{i}"))
    return turns


def test_replay_prefix_runs_tool_calls_and_asserts_diff() -> None:
    turns = _turns_with_bash("echo a", "echo b")
    expected = "diff --git a/x b/x\n+hello"
    sb = FakeSandbox(diffs=[expected])
    result = replay_prefix(turns, T=1, sandbox=sb, expected_diff=expected)
    assert result.consistent
    assert result.steps_replayed == 2
    assert any("echo a" in c for c in sb.cmds)


def test_desync_hard_fails() -> None:
    turns = _turns_with_bash("echo a")
    sb = FakeSandbox(diffs=["actual-diff"])
    with pytest.raises(ResumeDesyncError):
        replay_prefix(turns, T=0, sandbox=sb, expected_diff="expected-diff")


def test_desync_recorded_without_hard_fail() -> None:
    turns = _turns_with_bash("echo a")
    sb = FakeSandbox(diffs=["actual"])
    result = replay_prefix(
        turns, T=0, sandbox=sb, expected_diff="expected", hard_fail=False
    )
    assert not result.consistent
    assert result.desync_reason


def _write_turn(i: int, path: str, content: str) -> list[Turn]:
    return [
        Turn(
            index=i,
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id=f"w{i}",
                    name="write_file",
                    args={"path": path, "content": content},
                )
            ],
        ),
        Turn(index=i + 100, role="tool", content="ok", tool_call_id=f"w{i}"),
    ]


def test_write_file_replay_actually_writes_the_file(tmp_path: Path) -> None:
    """The generated shell command must survive the shell: `repr` renders
    newlines as backslash-n, which python then refuses to parse, and picks
    double quotes on apostrophes, which hands content to $-expansion."""
    content = "print('a $HOME b')\nx = 1\n"
    turns = _write_turn(0, "pkg/mod.py", content)
    sb = FakeSandbox(root=tmp_path)
    result = replay_prefix(turns, T=0, sandbox=sb)
    assert (tmp_path / "pkg" / "mod.py").read_text() == content
    assert result.consistent
    assert result.verified and result.checks_run == 1


def test_missing_expectation_is_unverified_not_consistent() -> None:
    """A replay with nothing checkable must not report consistent — the desync
    rate is the whole output of the Step-A spike."""
    turns = _turns_with_bash("echo a")
    result = replay_prefix(turns, T=0, sandbox=FakeSandbox(diffs=["whatever"]))
    assert not result.verified
    assert not result.consistent
    assert result.checks_run == 0
    assert result.unverifiable_steps == 1
    assert "unverified" in (result.desync_reason or "")


def test_file_desync_detected_from_transcript() -> None:
    """Sandbox that reports success but does not apply the write — exactly the
    silent desync bisection cannot tolerate."""
    turns = _write_turn(0, "a.txt", "expected\n")
    with pytest.raises(ResumeDesyncError):
        replay_prefix(turns, T=0, sandbox=FakeSandbox())

    result = replay_prefix(turns, T=0, sandbox=FakeSandbox(), hard_fail=False)
    assert result.verified and not result.consistent
    assert result.file_mismatches and "a.txt" in result.file_mismatches[0]


def test_transcript_implied_files_takes_last_write() -> None:
    turns = _write_turn(0, "a.txt", "first") + _write_turn(1, "a.txt", "second")
    assert transcript_implied_files(turns, T=0) == {"a.txt": "first"}
    assert transcript_implied_files(turns, T=1) == {"a.txt": "second"}
    assert transcript_implied_files(turns, T=-1) == {}


def test_unverified_excluded_from_desync_denominator() -> None:
    turns = _turns_with_bash("echo a")
    unverified = replay_prefix(turns, T=0, sandbox=FakeSandbox(diffs=["x"]))
    bad = replay_prefix(
        turns, T=0, sandbox=FakeSandbox(diffs=["x"]), expected_diff="y", hard_fail=False
    )
    stats = measure_desync_rate([unverified, bad])
    assert stats["n"] == 2
    assert stats["unverified"] == 1
    assert stats["verified"] == 1
    assert stats["desynced"] == 1
    assert stats["desync_rate"] == 1.0


def test_desync_rate_aggregation() -> None:
    turns = _turns_with_bash("echo a")
    ok = replay_prefix(
        turns,
        T=0,
        sandbox=FakeSandbox(diffs=["same"]),
        expected_diff="same",
        hard_fail=False,
    )
    bad = replay_prefix(
        turns,
        T=0,
        sandbox=FakeSandbox(diffs=["x"]),
        expected_diff="y",
        hard_fail=False,
    )
    stats = measure_desync_rate([ok, bad])
    assert stats["n"] == 2
    assert stats["desynced"] == 1
    assert stats["desync_rate"] == 0.5


def test_readonly_tools_are_skipped_not_counted_as_unsupported() -> None:
    """A read-only call changes nothing, so there is nothing to reproduce.
    Treating it as unsupported made every real scaffold report 100% desync."""
    turns = [
        Turn(
            index=0,
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="r", name="view", args={"path": "a.py"})],
        ),
        *_write_turn(1, "a.txt", "content\n"),
    ]
    sb = FakeSandbox(root=None)
    result = replay_prefix(turns, T=1, sandbox=sb, hard_fail=False)
    assert result.readonly_skipped == 1
    assert result.unsupported_skipped == []
    # The read-only call issued no shell command.
    assert not any("view" in c for c in sb.cmds)


def test_unsupported_tool_is_a_coverage_gap_not_a_desync() -> None:
    turns = [
        Turn(
            index=0,
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="x", name="browser_click", args={"sel": "#go"})],
        ),
        *_write_turn(1, "a.txt", "content\n"),
    ]
    # Must not raise, even with hard_fail=True: the replayer's gap is not the
    # trajectory's desync.
    result = replay_prefix(turns, T=1, sandbox=FakeSandbox(root=None), hard_fail=True)
    assert result.unsupported_skipped == ["browser_click"]
    assert result.verified is False
    assert result.consistent is False
    assert "coverage gap" in (result.desync_reason or "")

    stats = measure_desync_rate([result])
    assert stats["tool_coverage_blocked"] == 1
    assert stats["missing_tool_mappings"] == {"browser_click": 1}
    assert stats["desync_rate"] == 0.0  # excluded, not counted as a desync


def test_str_replace_is_replayed_and_predicted(tmp_path: Path) -> None:
    turns = [
        *_write_turn(0, "m.py", "def f():\n    return 1\n"),
        Turn(
            index=1,
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="e",
                    name="str_replace",
                    args={
                        "path": "m.py",
                        "old_str": "return 1",
                        "new_str": "return 2",
                    },
                )
            ],
        ),
    ]
    assert transcript_implied_files(turns, T=1) == {"m.py": "def f():\n    return 2\n"}
    result = replay_prefix(turns, T=1, sandbox=FakeSandbox(root=tmp_path))
    assert (tmp_path / "m.py").read_text() == "def f():\n    return 2\n"
    assert result.consistent


def test_str_replace_on_unknown_base_is_unpredictable_not_guessed() -> None:
    """Editing a file whose base content came from the repo leaves the result
    unknown — it must drop out of the expectation rather than be asserted."""
    turns = [
        Turn(
            index=0,
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="e",
                    name="str_replace",
                    args={"path": "repo.py", "old_str": "a", "new_str": "b"},
                )
            ],
        )
    ]
    assert transcript_implied_files(turns, T=0) == {}


def test_apply_patch_marks_its_files_unpredictable() -> None:
    patch = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    turns = [
        *_write_turn(0, "x.py", "a\n"),
        Turn(
            index=1,
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="p", name="apply_patch", args={"patch": patch})],
        ),
    ]
    # x.py was written earlier, but the patch makes its post-state unknown.
    assert transcript_implied_files(turns, T=0) == {"x.py": "a\n"}
    assert transcript_implied_files(turns, T=1) == {}
