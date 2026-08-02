"""Step A — trajectory state resume spike (PLAN.md).

Replay tool calls from a failed student trajectory into a fresh container up to
step T. Hard consistency assertion: post-replay `git diff` must match the
transcript-implied workspace state. Desync → hard fail; desync rate reported.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..schema import ToolCall, Turn

WRITE_TOOLS = ("write_file", "create_file", "write", "create")
EXEC_TOOLS = ("bash", "shell", "run_terminal_cmd", "execute")
EDIT_TOOLS = ("str_replace", "str_replace_editor", "edit_file", "edit", "replace")
PATCH_TOOLS = ("apply_patch", "patch")
# Tools that cannot change the workspace. Skipping them is not a gap in the
# replay — there is nothing to reproduce — so they must not count as unsupported.
# Treating them as unsupported made every real scaffold (which reads far more
# often than it writes) report a 100% desync rate, and PLAN.md's Step-A gate
# reads that number to decide whether ReOPD is mandatory.
READONLY_TOOLS = (
    "read_file",
    "read",
    "view",
    "cat",
    "open",
    "ls",
    "list_files",
    "glob",
    "grep",
    "search",
    "find",
    "think",
    "todo",
    "task_done",
)


class ResumeDesyncError(RuntimeError):
    """Post-replay git diff does not match the transcript-implied diff at T."""


class ExecSandbox(Protocol):
    def exec(self, cmd: str, *, timeout: int = 120) -> Any: ...


@dataclass
class ReplayStepResult:
    step_index: int
    tool_name: str
    ok: bool
    detail: str = ""


@dataclass
class ResumeResult:
    T: int
    steps_replayed: int
    consistent: bool
    expected_diff: str
    actual_diff: str
    step_results: list[ReplayStepResult] = field(default_factory=list)
    desync_reason: str | None = None
    # `consistent` is only meaningful when something was actually compared. A
    # replay with no checkable expectation is *unverified*, not consistent — the
    # difference is the whole point of the Step-A spike, and collapsing it makes
    # the measured desync rate identically zero.
    verified: bool = True
    checks_run: int = 0
    # Tools this replayer cannot reproduce, and read-only calls it correctly
    # skipped. `unsupported_skipped` is a coverage statistic about the replayer;
    # it is deliberately NOT a desync, because conflating the two makes the
    # Step-A desync rate a measure of tool coverage instead of trajectory
    # reproducibility. A prefix containing one is unverified, never inconsistent.
    unsupported_skipped: list[str] = field(default_factory=list)
    readonly_skipped: int = 0
    # Steps whose effect on the workspace the transcript cannot predict (opaque
    # shell commands). Reported so the coverage of the assertion is visible.
    unverifiable_steps: int = 0
    file_mismatches: list[str] = field(default_factory=list)


def assistant_tool_steps(turns: list[Turn]) -> list[tuple[int, ToolCall]]:
    """Ordered (turn_index, tool_call) for parent-agent tool invocations."""
    out: list[tuple[int, ToolCall]] = []
    for t in turns:
        if t.subagent_depth > 0:
            continue
        if t.role != "assistant":
            continue
        for tc in t.tool_calls:
            out.append((t.index, tc))
    return out


def _write_args(tc: ToolCall) -> tuple[str, str] | None:
    """(path, content) for a file-writing tool call, else None."""
    if (tc.name or "").lower() not in WRITE_TOOLS:
        return None
    args = tc.args if isinstance(tc.args, dict) else {}
    path = args.get("path") or args.get("file_path") or args.get("filename")
    if path is None:
        return None
    content = args.get("content") or args.get("file_text") or args.get("contents") or ""
    return str(path), str(content)


def _edit_args(tc: ToolCall) -> tuple[str, str, str] | None:
    """(path, old, new) for a string-replacement edit call, else None."""
    if (tc.name or "").lower() not in EDIT_TOOLS:
        return None
    args = tc.args if isinstance(tc.args, dict) else {}
    path = args.get("path") or args.get("file_path") or args.get("filename")
    old = args.get("old_str") or args.get("old_string") or args.get("old")
    new = args.get("new_str") or args.get("new_string") or args.get("new")
    if path is None or old is None:
        return None
    return str(path), str(old), str(new if new is not None else "")


def _patch_arg(tc: ToolCall) -> str | None:
    """The unified diff carried by an apply_patch-style call, else None."""
    if (tc.name or "").lower() not in PATCH_TOOLS:
        return None
    args = tc.args if isinstance(tc.args, dict) else {}
    patch = args.get("patch") or args.get("diff") or args.get("input")
    return str(patch) if patch else None


def is_readonly_tool(tc: ToolCall) -> bool:
    """True when the call cannot change the workspace, so replay may skip it."""
    return (tc.name or "").lower() in READONLY_TOOLS


def _tool_to_shell(tc: ToolCall) -> str | None:
    """Map a terminus-style tool call to a shell command. None = unsupported."""
    name = (tc.name or "").lower()
    args = tc.args if isinstance(tc.args, dict) else {}
    if name in EXEC_TOOLS:
        cmd = args.get("cmd") or args.get("command") or args.get("input")
        return str(cmd) if cmd is not None else None

    edit = _edit_args(tc)
    if edit is not None:
        path, old, new = edit
        # Exact single-occurrence replacement, matching str_replace semantics:
        # zero or multiple matches is an error, not a silent partial edit, or the
        # replayed workspace diverges from the transcript without anyone noticing.
        program = (
            "import pathlib,sys\n"
            "p=pathlib.Path(sys.argv[1]); s=p.read_text()\n"
            "old,new=sys.argv[2],sys.argv[3]\n"
            "n=s.count(old)\n"
            "if n!=1: sys.exit('str_replace matched %d times, expected 1' % n)\n"
            "p.write_text(s.replace(old,new,1))\n"
        )
        return "python3 -c " + " ".join(
            shlex.quote(a) for a in (program, path, old, new)
        )

    patch = _patch_arg(tc)
    if patch is not None:
        heredoc = "EOF_R2E_RESUME_PATCH"
        return f"git apply --verbose - <<'{heredoc}'\n{patch}\n{heredoc}"

    write = _write_args(tc)
    if write is not None:
        path, content = write
        # `shlex.quote`, not `repr`: repr renders the program's newlines as the
        # two characters \ and n, which the shell passes through literally and
        # python then parses as a syntax error. It also picks double quotes when
        # the content contains an apostrophe, which hands file content to the
        # shell for $-expansion.
        program = (
            "import pathlib,sys\n"
            "p=pathlib.Path(sys.argv[1])\n"
            "p.parent.mkdir(parents=True, exist_ok=True)\n"
            "p.write_text(sys.argv[2])\n"
        )
        return "python3 -c " + " ".join(
            shlex.quote(a) for a in (program, path, content)
        )
    return None


def transcript_implied_files(turns: list[Turn], T: int) -> dict[str, str]:
    """Final content of every file the transcript writes in the prefix ≤ T.

    This is the expectation the replay is checked against. Only file-writing tool
    calls are predictable from a transcript; shell commands are opaque, so this
    map is a lower bound on the workspace state and `unverifiable_steps` records
    how much of the prefix it does not cover.
    """
    steps = assistant_tool_steps(turns)
    to_run = steps[: T + 1] if T >= 0 else []
    files: dict[str, str] = {}
    unpredictable: set[str] = set()
    for _turn_i, tc in to_run:
        write = _write_args(tc)
        if write is not None:
            path, content = write
            files[path] = content
            unpredictable.discard(path)
            continue

        edit = _edit_args(tc)
        if edit is not None:
            path, old, new = edit
            # An edit is only predictable when the transcript already established
            # the file's full content in this prefix. Editing a file whose base
            # state came from the repo leaves the result unknown — drop it rather
            # than assert against a guess.
            if path in files and files[path].count(old) == 1:
                files[path] = files[path].replace(old, new, 1)
            else:
                files.pop(path, None)
                unpredictable.add(path)
            continue

        # A patch can touch files whose base content the transcript never showed.
        patch = _patch_arg(tc)
        if patch is not None:
            for path in _paths_in_patch(patch):
                files.pop(path, None)
                unpredictable.add(path)
    return files


def _paths_in_patch(unified_diff: str) -> list[str]:
    """`b/` paths touched by a unified diff."""
    out: list[str] = []
    for line in unified_diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip().split("\t")[0]
            if path.startswith("b/"):
                path = path[2:]
            if path and path != "/dev/null" and path not in out:
                out.append(path)
    return out


def replay_prefix(
    turns: list[Turn],
    T: int,
    sandbox: ExecSandbox,
    *,
    expected_diff: str | None = None,
    base_ref: str = "HEAD",
    hard_fail: bool = True,
) -> ResumeResult:
    """Replay tool calls for turn indices ≤ T into sandbox; assert git diff.

    `T` indexes assistant tool steps (0-based into `assistant_tool_steps`), not
    raw turn indices — matches bisection over action prefixes.
    """
    steps = assistant_tool_steps(turns)
    if T < -1:
        raise ValueError(f"T must be >= -1, got {T}")
    to_run = steps[: T + 1] if T >= 0 else []

    step_results: list[ReplayStepResult] = []
    skipped_readonly = 0
    unsupported: list[str] = []
    for step_i, (_turn_i, tc) in enumerate(to_run):
        # A read-only call changes nothing, so there is nothing to reproduce.
        # Skipping is not a coverage gap and must not be counted as one.
        if is_readonly_tool(tc):
            skipped_readonly += 1
            step_results.append(
                ReplayStepResult(
                    step_index=step_i,
                    tool_name=tc.name,
                    ok=True,
                    detail="read-only tool, no workspace effect",
                )
            )
            continue

        cmd = _tool_to_shell(tc)
        if cmd is None:
            # An unmapped tool is a gap in *this replayer*, not evidence that the
            # trajectory desynced. Conflating the two reports a desync rate that
            # measures tool coverage, and PLAN.md's Step-A gate reads that number
            # to decide whether ReOPD is mandatory. Recorded and skipped.
            unsupported.append(tc.name)
            step_results.append(
                ReplayStepResult(
                    step_index=step_i,
                    tool_name=tc.name,
                    ok=True,
                    detail=f"unsupported tool for resume, skipped: {tc.name}",
                )
            )
            continue

        exec_result = sandbox.exec(cmd)
        ok = bool(getattr(exec_result, "ok", getattr(exec_result, "exit_code", 1) == 0))
        detail = ""
        if not ok:
            detail = str(
                getattr(exec_result, "stderr", None)
                or getattr(exec_result, "stdout", None)
                or exec_result
            )[:500]
        step_results.append(
            ReplayStepResult(step_index=step_i, tool_name=tc.name, ok=ok, detail=detail)
        )

    def _git_diff() -> str:
        r = sandbox.exec(f"git add -A && git diff --cached {base_ref} || true")
        out = getattr(r, "stdout", None)
        if out is None:
            out = str(r)
        return out.strip()

    actual = _git_diff()

    checks = 0
    problems: list[str] = []

    # A tool call that failed on replay but succeeded in the original trajectory
    # *is* a desync, and for a bash-only prefix it is the only signal available.
    # It is asymmetric evidence, so it is not counted as a check: a step erroring
    # falsifies consistency, but every step exiting 0 proves nothing about the
    # resulting workspace state.
    failed_steps = [s for s in step_results if not s.ok]
    if failed_steps:
        problems.append(
            "replay step(s) failed: "
            + "; ".join(
                f"#{s.step_index} {s.tool_name}: {s.detail[:120]}" for s in failed_steps[:3]
            )
        )

    # Check 1 — every file the transcript wrote must hold the transcript's
    # content after replay. This is the assertion that can actually fail; the
    # git-diff comparison below only runs when a caller supplies a diff it
    # derived some other way.
    expected_files = transcript_implied_files(turns, T)
    mismatches: list[str] = []
    for path, content in expected_files.items():
        checks += 1
        got = _read_file(sandbox, path)
        if got is None:
            mismatches.append(f"{path}: missing after replay")
        elif got != content:
            mismatches.append(
                f"{path}: content differs (expected {len(content)}B, got {len(got)}B)"
            )
    if mismatches:
        problems.append("file state mismatch: " + "; ".join(mismatches))

    # Check 2 — caller-supplied diff, when there is one.
    expected = expected_diff if expected_diff is not None else ""
    if expected_diff is not None:
        checks += 1
        if _normalize_diff(actual) != _normalize_diff(expected_diff):
            problems.append("post-replay git diff != supplied expected diff")

    unverifiable = sum(
        1 for _turn_i, tc in to_run if (tc.name or "").lower() in EXEC_TOOLS
    )
    # Verified means the replay could have been shown wrong: either a positive
    # state check ran, or a step failure already showed it was. An unsupported
    # tool voids that: the workspace is missing an effect the transcript had, so
    # any comparison would be against a state we knowingly failed to build.
    verified = (checks > 0 or bool(failed_steps)) and not unsupported
    consistent = verified and not problems
    if unsupported:
        desync_reason = (
            "unverified: replayer has no mapping for "
            + ", ".join(sorted(set(unsupported)))
            + " — coverage gap, not a desync"
        )
    elif not verified:
        desync_reason = (
            "unverified: nothing replayed and no expected state to compare "
            f"({unverifiable} opaque shell step(s), no expected_diff supplied)"
        )
    else:
        desync_reason = "; ".join(problems) or None

    result = ResumeResult(
        T=T,
        steps_replayed=len(to_run),
        consistent=consistent,
        expected_diff=expected,
        actual_diff=actual,
        step_results=step_results,
        desync_reason=desync_reason,
        verified=verified,
        checks_run=checks,
        unverifiable_steps=unverifiable,
        file_mismatches=mismatches,
        unsupported_skipped=sorted(set(unsupported)),
        readonly_skipped=skipped_readonly,
    )
    # An unverified replay is not a desync — there is nothing to disagree with —
    # so it must not hard-fail the bisection. It is counted separately instead.
    if verified and not consistent and hard_fail:
        raise ResumeDesyncError(desync_reason or "desync")
    return result


def _read_file(sandbox: ExecSandbox, path: str) -> str | None:
    """Read a file back out of the sandbox; None when it does not exist."""
    r = sandbox.exec(f"cat -- {shlex.quote(path)}")
    # Default to failure, matching the step path: a result object exposing
    # neither `ok` nor `exit_code` must not be read as "file exists", or the
    # missing-file case is reported as a content mismatch against its repr.
    ok = bool(getattr(r, "ok", getattr(r, "exit_code", 1) == 0))
    if not ok:
        return None
    out = getattr(r, "stdout", None)
    return str(r) if out is None else out


def _normalize_diff(diff: str) -> str:
    lines = [ln.rstrip() for ln in diff.splitlines() if ln.strip()]
    # Drop index/blob noise that varies across replays of the same tree.
    keep = []
    for ln in lines:
        if ln.startswith("index ") or ln.startswith("diff --git"):
            continue
        keep.append(ln)
    return "\n".join(keep)


def measure_desync_rate(
    trials: list[ResumeResult],
) -> dict[str, float | int]:
    """Aggregate consistency outcomes into the desync-rate statistic.

    Unverified replays are excluded from the denominator rather than counted as
    consistent — the rate PLAN.md gates on is over replays that were actually
    checked. `unverified_rate` is reported beside it so a high one is visible.
    """
    n = len(trials)
    verified = [t for t in trials if t.verified]
    unverified = n - len(verified)
    desynced = sum(1 for t in verified if not t.consistent)
    blocked = sum(1 for t in trials if t.unsupported_skipped)
    missing_tools: dict[str, int] = {}
    for t in trials:
        for name in t.unsupported_skipped:
            missing_tools[name] = missing_tools.get(name, 0) + 1
    return {
        "n": n,
        "verified": len(verified),
        "unverified": unverified,
        "unverified_rate": (unverified / n) if n else 0.0,
        "desynced": desynced,
        "desync_rate": (desynced / len(verified)) if verified else 0.0,
        "consistent": len(verified) - desynced,
        # Coverage, reported beside the rate it would otherwise contaminate: a
        # high `tool_coverage_blocked` means the desync rate is measured on a
        # small, possibly unrepresentative slice, which is a different problem
        # from trajectories genuinely failing to replay.
        "tool_coverage_blocked": blocked,
        "missing_tool_mappings": dict(sorted(missing_tools.items())),
    }


__all__ = [
    "EDIT_TOOLS",
    "EXEC_TOOLS",
    "PATCH_TOOLS",
    "READONLY_TOOLS",
    "WRITE_TOOLS",
    "ExecSandbox",
    "ReplayStepResult",
    "ResumeDesyncError",
    "ResumeResult",
    "assistant_tool_steps",
    "is_readonly_tool",
    "measure_desync_rate",
    "replay_prefix",
    "transcript_implied_files",
]
