"""Local `git` queries for commit-level mining.

`pr_runtime` gets its candidates from the GitHub API, which already hands back
structured metadata. Commit mining has no such API: the history lives in a
clone, so this module is the equivalent "list candidates" layer, built on
`git log` / `git show` subprocesses.

The output of `show_diff` is deliberately the same shape as `gh pr diff`, so
`split_patch_and_test_patch()` and the rest of the `pipeline.py` helpers work
on commits unchanged.

Design credit: the commit-mining approach is R2E-Gym's (SWE-GEN, Jain et al.,
COLM '25) by way of Repo2RLEnv's `commit_runtime` pipeline (Apache-2.0).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


class GitError(RuntimeError):
    """Raised when a `git` subprocess returns non-zero."""


@dataclass(slots=True)
class CommitInfo:
    """One commit's metadata, parsed from `git log --format=...`."""

    sha: str  # full 40-char SHA
    parent_sha: str  # first parent ("" for a root commit — nothing to diff against)
    parents: list[str] = field(default_factory=list)  # >1 entry means a merge commit
    author_name: str = ""
    author_email: str = ""
    authored_at: str = ""  # ISO8601, e.g. "2026-03-12T08:15:22Z"
    subject: str = ""
    body: str = ""

    @property
    def is_merge(self) -> bool:
        return len(self.parents) > 1

    @property
    def message(self) -> str:
        """Subject + blank line + body, the way `git show -s` renders it."""
        return f"{self.subject}\n\n{self.body}" if self.body else self.subject

    @property
    def short_sha(self) -> str:
        return self.sha[:12]


# ASCII separators. Commit subjects and bodies contain newlines, tabs and
# almost every printable character, so a line-based format cannot be parsed
# unambiguously; the unit/record separators effectively never appear in real
# commit text.
_FIELD_SEP = "\x1f"
_RECORD_SEP = "\x1e"

_LOG_FORMAT = (
    _FIELD_SEP.join(
        [
            "%H",  # full SHA
            "%P",  # all parents, space-separated
            "%an",  # author name
            "%ae",  # author email
            "%aI",  # author date, ISO8601 strict
            "%s",  # subject
            "%b",  # body
        ]
    )
    + _RECORD_SEP
)

_EXPECTED_FIELDS = 7


def _run_git(args: list[str], cwd: Path, *, timeout: int = 60) -> str:
    """Run `git ...` in `cwd` and return stdout. Raises GitError on non-zero exit."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {' '.join(args)!r} timed out after {timeout}s") from exc
    if proc.returncode != 0:
        raise GitError(
            f"git {' '.join(args)!r} failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()[:400]}"
        )
    return proc.stdout


def clone_for_log(
    repo_url: str, dest: Path, *, depth: int = 200, branch: str | None = None
) -> Path:
    """Clone `repo_url` deep enough for `git log` to walk.

    The bootstrap clone is `--depth=1`, which has exactly one commit and is
    therefore useless for mining. `--filter=blob:none` keeps the clone cheap:
    we need commit metadata for every candidate but file contents only for the
    handful that survive filtering, and git fetches those lazily on `git show`.
    """
    dest.mkdir(parents=True, exist_ok=True)
    args = ["clone", "--filter=blob:none", f"--depth={depth}", "--no-tags"]
    if branch:
        args += ["--branch", branch]
    args += [repo_url, str(dest)]
    _run_git(args, dest.parent, timeout=600)
    return dest


def list_commits(
    clone_dir: Path,
    *,
    since: date | None = None,
    until: date | None = None,
    limit: int = 50,
    branch: str = "HEAD",
    first_parent: bool = True,
) -> list[CommitInfo]:
    """Return commits on `branch`, newest first.

    Range and count filtering happen inside git rather than in Python, so a
    repo with 200k commits costs the same as one with 200.

    `first_parent` follows only the mainline. Without it a single squashed
    merge expands into every commit of the branch it merged, which are
    intermediate states that were never independently tested — exactly the
    commits whose F2P validation would fail after paying for a container run.
    """
    args = [
        "log",
        f"--max-count={limit}",
        f"--format={_LOG_FORMAT}",
        "--no-decorate",
    ]
    if first_parent:
        args.append("--first-parent")
    if since is not None:
        args.append(f"--since={since.isoformat()}")
    if until is not None:
        args.append(f"--until={until.isoformat()}")
    args.append(branch)

    try:
        raw = _run_git(args, clone_dir, timeout=120)
    except GitError as exc:
        # An unknown branch/ref is a normal outcome for a caller probing a
        # default branch name, not a reason to abort the whole mine.
        logger.warning("git log failed in %s: %s", clone_dir, exc)
        return []
    return _parse_log_output(raw)


def _parse_log_output(raw: str) -> list[CommitInfo]:
    """Parse the record-separated `git log` stream into `CommitInfo`s."""
    out: list[CommitInfo] = []
    for rec in raw.split(_RECORD_SEP):
        if not rec.strip():
            continue
        fields = rec.lstrip("\n").split(_FIELD_SEP)
        if len(fields) < _EXPECTED_FIELDS:
            logger.debug("skipping malformed git log record: %r", rec[:100])
            continue
        sha, parents_str, author_name, author_email, authored_at, subject, body = fields[:7]
        parents = parents_str.split() if parents_str else []
        out.append(
            CommitInfo(
                sha=sha,
                parent_sha=parents[0] if parents else "",
                parents=parents,
                author_name=author_name,
                author_email=author_email,
                authored_at=authored_at,
                subject=subject,
                body=body.strip(),
            )
        )
    return out


def show_diff(clone_dir: Path, commit_sha: str, *, timeout: int = 60) -> str:
    """Return the unified diff a commit introduces, with no message header.

    `--format=` suppresses the commit-info block, leaving a bare
    `diff --git a/X b/Y` sequence — the same shape `gh pr diff` produces, so
    `split_patch_and_test_patch()` parses it without a special case.
    """
    return _run_git(
        ["show", "--format=", "--patch", "--no-color", commit_sha],
        clone_dir,
        timeout=timeout,
    )


def changed_files(clone_dir: Path, commit_sha: str) -> list[str]:
    """Return the repo-relative paths a commit touches."""
    raw = _run_git(
        ["show", "--no-color", "--format=", "--name-only", commit_sha],
        clone_dir,
        timeout=60,
    )
    return [line.strip() for line in raw.splitlines() if line.strip()]
