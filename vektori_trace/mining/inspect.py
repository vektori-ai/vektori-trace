"""Static audit of an emitted task directory.

The v0 plan's last step-3 item says to mine one repo and then *hand-inspect
three tasks*: base commit right, `.git` scrubbed, F2P names actually in the test
patch. Hand-inspection doesn't survive contact with fifty tasks and can't be
re-run after a refactor, so the three checks live here instead.

Deliberately static — it reads the emitted files and never starts a container.
`check-env` already answers "does the guard hold at runtime" by running one; the
question here is different and cheaper: **does what we wrote down agree with
itself.** A task whose `f2p.json` names a test the hidden test_patch never adds
is broken no matter how sound the container is, and it fails silently — the
verifier looks for the test, doesn't find it, and scores 0. Every task built
that way is an unwinnable task in the dataset, indistinguishable from a hard one.

Findings, not exceptions: the point is a histogram over a corpus, so one broken
task must not stop the audit of the rest.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# `+++ b/path/to/test_x.py` — the post-image path of each file a patch touches.
_PATCH_TARGET_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)

# A pytest node id: `path/to/test_x.py::TestCls::test_y` → the path part.
_NODE_PATH_RE = re.compile(r"^([^:]+?\.py)(?:::|$)")


@dataclass
class TaskAudit:
    """One task's findings. `ok` is the conjunction — anything false ships a
    task that cannot be won, or can be won by reading the answer."""

    task: str
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(self.checks.values())

    @property
    def failures(self) -> list[str]:
        return [k for k, v in self.checks.items() if not v]


def _record(audit: TaskAudit, name: str, ok: bool, detail: str) -> None:
    audit.checks[name] = ok
    audit.details[name] = detail


def audit_task(task_dir: Path) -> TaskAudit:
    """Read one emitted task dir and check it agrees with itself."""
    audit = TaskAudit(task=task_dir.name)

    toml_path = task_dir / "task.toml"
    if not toml_path.exists():
        _record(audit, "task_toml_present", False, "no task.toml")
        return audit

    try:
        cfg = tomllib.loads(toml_path.read_text())
    except (tomllib.TOMLDecodeError, OSError) as e:
        _record(audit, "task_toml_parses", False, f"{type(e).__name__}: {e}")
        return audit

    pr_meta = (
        (cfg.get("metadata") or {}).get("repo2env", {}).get("pr_runtime", {})
    )
    base_commit = pr_meta.get("base_commit", "")
    declared_f2p = pr_meta.get("fail_to_pass") or []

    dockerfile = _read(task_dir / "environment" / "Dockerfile")
    test_sh = _read(task_dir / "tests" / "test.sh")
    oracle = _read(task_dir / "solution" / "patch.diff")

    # --- 1. base commit right -------------------------------------------------
    # The Dockerfile must reset to the *same* commit task.toml claims. If they
    # disagree the oracle patch is applied to a tree it wasn't generated
    # against, so it may not apply at all — and a task whose own gold patch
    # fails is unwinnable.
    _record(
        audit,
        "base_commit_declared",
        bool(base_commit) and len(base_commit) == 40,
        base_commit or "absent",
    )
    _record(
        audit,
        "dockerfile_resets_to_base_commit",
        bool(base_commit) and f"git reset --hard {base_commit}" in dockerfile,
        f"reset to {base_commit[:12] or '?'} present in Dockerfile"
        if base_commit and f"git reset --hard {base_commit}" in dockerfile
        else "Dockerfile does not reset to the declared base commit",
    )

    # --- 2. .git scrubbed -----------------------------------------------------
    # The working tree can sit at the base commit while `.git` still holds the
    # future: origin/main, tags, the fix commit, the hidden test. Reading the
    # answer out of it needs no network at all, and it is a documented,
    # repeated SWE-bench incident.
    scrub_markers = {
        "remote_removed": "git remote remove origin",
        "branches_pruned": "git branch -D",
        "tags_deleted": "git tag -d",
        "reflog_expired": "git reflog expire",
        "gc_pruned": "git gc --prune=now",
    }
    missing = [k for k, marker in scrub_markers.items() if marker not in dockerfile]
    _record(
        audit,
        "git_history_scrubbed",
        not missing,
        "all scrub steps present" if not missing else f"missing: {', '.join(missing)}",
    )

    # --- 3. F2P names actually in the test patch ------------------------------
    # The graded verifier looks up these exact node ids in the suite output. A
    # name the hidden test_patch never adds is never collected, never runs,
    # never passes — the task scores 0 for everyone, forever, and reads as
    # merely hard.
    patch_files = set(_PATCH_TARGET_RE.findall(test_sh))
    f2p_files = {m.group(1) for f in declared_f2p if (m := _NODE_PATH_RE.match(f))}
    _record(
        audit,
        "f2p_declared",
        bool(declared_f2p),
        f"{len(declared_f2p)} F2P test(s)" if declared_f2p else "no F2P tests declared",
    )
    if f2p_files and patch_files:
        orphans = sorted(f2p_files - patch_files)
        _record(
            audit,
            "f2p_files_in_test_patch",
            not orphans,
            "every F2P file appears in the test patch"
            if not orphans
            else f"F2P files absent from the test patch: {', '.join(orphans)}",
        )
    else:
        # No embedded test_patch to compare against (e.g. the test file already
        # existed at base). Not a pass — state that the check couldn't run
        # rather than quietly counting it as one.
        _record(
            audit,
            "f2p_files_in_test_patch",
            not f2p_files or bool(patch_files),
            "no test_patch embedded in test.sh; F2P provenance unchecked",
        )

    # --- the shipped-artifact checks ------------------------------------------
    f2p_json = task_dir / "tests" / "f2p.json"
    shipped_f2p: list[str] = []
    if f2p_json.exists():
        try:
            shipped_f2p = json.loads(f2p_json.read_text())
        except (json.JSONDecodeError, OSError):
            shipped_f2p = []
    _record(
        audit,
        "shipped_f2p_matches_task_toml",
        sorted(shipped_f2p) == sorted(declared_f2p),
        "tests/f2p.json agrees with task.toml"
        if sorted(shipped_f2p) == sorted(declared_f2p)
        else f"task.toml declares {len(declared_f2p)}, tests/f2p.json ships {len(shipped_f2p)}",
    )

    # The oracle is what proves the task is solvable at all.
    _record(
        audit,
        "oracle_patch_present",
        bool(oracle.strip()),
        f"{len(oracle.splitlines())} line(s)" if oracle.strip() else "empty or missing",
    )

    # The oracle must not carry the tests it is graded by: applying it would
    # then satisfy the F2P set by rewriting the ruler.
    oracle_files = set(_PATCH_TARGET_RE.findall(oracle))
    leaked = sorted(oracle_files & patch_files)
    _record(
        audit,
        "oracle_excludes_test_files",
        not leaked,
        "oracle touches no graded test file"
        if not leaked
        else f"oracle also patches graded tests: {', '.join(leaked)}",
    )

    return audit


def audit_tasks(task_dirs: list[Path]) -> list[TaskAudit]:
    return [audit_task(d) for d in task_dirs]


def failure_histogram(audits: list[TaskAudit]) -> dict[str, int]:
    """Per-check failure counts across the corpus — which check, how often."""
    hist: dict[str, int] = {}
    for a in audits:
        for name in a.failures:
            hist[name] = hist.get(name, 0) + 1
    return hist


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


__all__ = ["TaskAudit", "audit_task", "audit_tasks", "failure_histogram"]
