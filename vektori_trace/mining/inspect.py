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

# Test-file extensions across the runners `--language` advertises. A node id
# that names a file lets us check the *file* appears in the hidden test patch.
_TEST_FILE_EXTS = (
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".scala",
    ".rb",
    ".php",
    ".cs",
    ".c",
    ".cc",
    ".cpp",
)

# The path part of an identifier that starts with one: pytest's
# `path/to/test_x.py::TestCls::test_y`, jest's `path/to/x.test.ts > suite > case`.
_NODE_PATH_RE = re.compile(
    r"^([^\s:>#]+?(?:" + "|".join(re.escape(e) for e in _TEST_FILE_EXTS) + r"))(?:[:>#]|$)"
)

# Everything that separates a test's name from its container across runners:
# pytest `::`, JUnit `#`, Rust/C++ `::`, Go/Java `.`, jest `>`.
_NODE_SEP_RE = re.compile(r"::|[#>.]")

# The body of the embedded test patch — added, removed and context lines
# alike. Restricting this to added lines is wrong: a PR that *modifies* an
# existing test moves it fail->pass just as legitimately as one that adds a new
# one, and the modified test's `def` line then appears only as context. Measured
# against the first real corpus, where `structlog-786`'s `test_repr` is exactly
# that case.
_PATCH_BODY_RE = re.compile(r"^[-+ ].*$", re.MULTILINE)


def _node_file(node_id: str) -> str | None:
    """The file an F2P identifier names, when it names one."""
    m = _NODE_PATH_RE.match(node_id.strip())
    return m.group(1) if m else None


def _node_symbol(node_id: str) -> str:
    """The test's own name — the last segment, whatever the separator.

    Go (`TestFoo`), Rust (`mod::tests::case`) and JUnit (`Cls#method`) node ids
    carry no path at all, so the file check can say nothing about them. The
    symbol is the one part every runner has, which makes it the only provenance
    check that works for every language `--language` accepts.

    The parametrisation suffix is dropped: pytest's `test_pickle[0-None]` and
    Go's `TestFoo/sub_case` are *generated at run time* from one `def
    test_pickle` / `func TestFoo` in the source, so searching the patch for the
    expanded id can only ever miss. Measured against the first real corpus —
    keeping it flagged 2 of 4 sound structlog tasks, and a check that fires on
    half a clean corpus is worse than no check.
    """
    parts = [p for p in _NODE_SEP_RE.split(node_id.strip()) if p]
    symbol = parts[-1] if parts else ""
    symbol = symbol.split("[", 1)[0]  # pytest parametrisation
    symbol = symbol.split("/", 1)[0]  # go subtests
    return symbol.strip()


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

    # Walk the metadata defensively. `[metadata]` with `repo2env = "bad"` is
    # syntactically valid TOML, and a chained `.get()` on the string raises —
    # which would abort `audit_tasks` and take the whole corpus histogram with
    # it. One malformed task must not stop the audit of the rest; that is the
    # entire reason these are findings and not exceptions.
    def _table(parent: object, key: str) -> dict:
        if not isinstance(parent, dict):
            return {}
        value = parent.get(key)
        return value if isinstance(value, dict) else {}

    pr_meta = _table(_table(_table(cfg, "metadata"), "repo2env"), "pr_runtime")
    if not pr_meta and isinstance(cfg.get("metadata"), dict):
        _record(
            audit,
            "metadata_well_formed",
            False,
            "no readable [metadata.repo2env.pr_runtime] table",
        )
    base_commit = pr_meta.get("base_commit", "")
    if not isinstance(base_commit, str):
        base_commit = ""
    declared_f2p = pr_meta.get("fail_to_pass") or []
    if not isinstance(declared_f2p, list):
        declared_f2p = []
    declared_f2p = [f for f in declared_f2p if isinstance(f, str)]

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
    patch_body = "\n".join(_PATCH_BODY_RE.findall(test_sh))
    f2p_files = {f for node in declared_f2p if (f := _node_file(node))}
    _record(
        audit,
        "f2p_declared",
        bool(declared_f2p),
        f"{len(declared_f2p)} F2P test(s)" if declared_f2p else "no F2P tests declared",
    )

    # File-level provenance, when the identifiers name files at all. pytest and
    # jest ids do; Go, Rust and JUnit ids do not.
    if f2p_files:
        orphans = sorted(f2p_files - patch_files)
        _record(
            audit,
            "f2p_files_in_test_patch",
            not orphans and bool(patch_files),
            "every F2P file appears in the test patch"
            if not orphans and patch_files
            else (
                "no test patch embedded in test.sh to check against"
                if not patch_files
                else f"F2P files absent from the test patch: {', '.join(orphans)}"
            ),
        )

    # Symbol-level provenance, which every runner supports. This is the check
    # that catches the failure that matters: a name the hidden test patch never
    # adds is never collected, never runs, never passes, so the task scores 0
    # for everyone forever and reads as merely hard. Previously the file regex
    # accepted only `.py`, so for every other language `f2p_files` came back
    # empty and the check passed by default — silently, for exactly the runners
    # least likely to have been eyeballed.
    #
    # "In the patch" means anywhere in it, not just its added lines: a PR that
    # modifies an existing test moves it fail->pass just as legitimately as one
    # that adds a new test.
    if declared_f2p:
        missing = sorted(
            {
                node
                for node in declared_f2p
                if (sym := _node_symbol(node)) and sym not in patch_body
            }
        )
        _record(
            audit,
            "f2p_names_in_test_patch",
            not missing and bool(patch_body.strip()),
            "every F2P test name is added by the test patch"
            if not missing and patch_body.strip()
            else (
                "no test patch embedded in test.sh; F2P provenance unverifiable"
                if not patch_body.strip()
                else f"F2P names the test patch never adds: {', '.join(missing[:3])}"
            ),
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
