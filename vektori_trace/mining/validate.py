"""Validate a candidate PR inside the bootstrap container.

Computes the FAIL_TO_PASS + PASS_TO_PASS sets that SWE-bench-style oracles
need. Workflow inside one container:

  1. Reset working tree to base_commit (`git reset --hard`)
  2. Apply test_patch only           → run tests → pre_status (per-test)
  3. Reset, then apply patch + test_patch → run tests → post_status
  4. F2P = tests that FAILED in (2) and PASS in (3)
     P2P = tests that PASSED in both (2) and (3)

We mirror SWE-bench's harness/test_spec/utils.py:make_eval_script_list_common
for the per-stage script, and harness/grading.py:get_logs_eval for the
status-extraction direction. Implementation is independent (no swebench
import); see references/SWE-bench/ for the reference code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from vektori_trace.mining.bootstrap.docker import DockerSandbox
from vektori_trace.mining.log_parsers import parse_logs
from vektori_trace.mining.pipeline import _GIT_CLEAN_EXCLUDES

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ValidationOutcome:
    status: str  # verified | partial | failed | skipped
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    pre_log: str = ""  # raw test output, pre-fix (truncated)
    post_log: str = ""  # raw test output, post-fix
    reason: str = ""  # populated on status != "verified"


_HEREDOC = "EOF_R2E_VALIDATE"


def _heredoc_apply(patch: str) -> str:
    return f"git apply --verbose --reject - <<'{_HEREDOC}'\n{patch}\n{_HEREDOC}"


def _build_stage_script(
    base_commit: str,
    *,
    apply_patch: str | None,
    apply_test_patch: str | None,
    test_cmds: list[str],
) -> str:
    """Build the shell script for one validation stage (pre-fix or post-fix).

    Always resets to base_commit first, then applies whichever patches the
    caller supplies (None ⇒ skip), then runs the test commands wrapped in
    START/END markers so the parser knows where output starts.

    Assumes the caller has already ensured `base_commit` is fetchable in the
    container's git object database (see `_fetch_base_commit`). The bootstrap
    image only ships the shallow-clone HEAD, so historical PR base commits
    aren't present until we fetch them explicitly.
    """
    parts: list[str] = [
        "set -uxo pipefail",
        "cd /workspace",
        "git config --global --add safe.directory /workspace",
        f"git reset --hard {base_commit}",
        # Keep language dep/build dirs (node_modules, target, vendor, ...) so
        # the clean doesn't wipe deps the suite needs — shared with
        # pipeline.py's Dockerfile emitter via _GIT_CLEAN_EXCLUDES so the two
        # can't drift apart again.
        #
        # NO `-x`. Ignored files are generated build artifacts, and removing
        # them breaks any project that derives its version from git:
        # prefect's `.gitignore` lists `src/prefect/_build_info.py`, written by
        # `hatch_build.py` at install time. `git clean -fdx` deleted it, so
        # `import prefect` failed, so pytest died during config parsing —
        # before collecting a single test. The stage emitted an empty
        # START/END block, `parse_logs` returned {}, and every PR skipped as
        # `no_fail_to_pass`: a plausible bucket name for a validation that
        # never ran. Measured on prefect: `-fd` removes nothing, `-fdx`
        # removes exactly those two generated files.
        f"git clean -fd {_GIT_CLEAN_EXCLUDES} || true",
    ]
    if apply_patch and apply_patch.strip():
        parts.append(_heredoc_apply(apply_patch))
    if apply_test_patch and apply_test_patch.strip():
        parts.append(_heredoc_apply(apply_test_patch))
    # Emit the markers to STDOUT via echo — NOT `: 'MARKER'`. With `set -x`,
    # a `:` no-op is traced to STDERR as `+ : MARKER`, while the test runner
    # writes to STDOUT. Slicing between stderr-only markers captured zero test
    # lines (the bug that silently zeroed F2P detection on any real suite).
    parts.append("echo R2E_START_TEST_OUTPUT")
    parts.append(" && ".join(test_cmds) if test_cmds else "echo 'no test_cmds'")
    parts.append("echo R2E_END_TEST_OUTPUT")
    return "\n".join(parts)


# Install git AND ca-certificates. Minimal images (esp. node:alpine) ship
# without CA certs, so `git fetch` over HTTPS dies with "server certificate
# verification failed. CAfile: none" — which silently breaks base_commit
# fetching during validation. Install both; refresh the CA store on alpine.
_GIT_DEFENSIVE_INSTALL = (
    "(command -v git >/dev/null 2>&1 && [ -e /etc/ssl/certs/ca-certificates.crt ]) || "
    "(apt-get update >/dev/null 2>&1 && "
    "apt-get install -y --no-install-recommends git ca-certificates >/dev/null 2>&1 && "
    "rm -rf /var/lib/apt/lists/*) || "
    "(apk add --no-cache git ca-certificates >/dev/null 2>&1 && "
    "update-ca-certificates >/dev/null 2>&1) || true"
)


def _ensure_git(sandbox: DockerSandbox) -> bool:
    """Install git in the sandbox if missing + mark the repo as safe.directory.

    Idempotent — no-op when git is already on PATH. Returns True if git is
    available after the call. Tries apt-get (Debian/Ubuntu) then apk (Alpine).
    Always marks /workspace as a safe directory afterward because Docker copies
    the repo in as root-owned, which triggers git's `detected dubious ownership`
    refusal otherwise.
    """
    check = sandbox.exec("command -v git >/dev/null 2>&1 && echo OK", timeout=10)
    if not (check.ok and "OK" in check.stdout):
        sandbox.exec(_GIT_DEFENSIVE_INSTALL, timeout=120)
        re = sandbox.exec("command -v git >/dev/null 2>&1 && echo OK", timeout=10)
        if not (re.ok and "OK" in re.stdout):
            return False
    sandbox.exec("git config --global --add safe.directory /workspace", timeout=10)
    return True


def _fetch_base_commit(sandbox: DockerSandbox, base_commit: str, *, timeout: int = 120) -> bool:
    """Make `base_commit` available in the container's git object db.

    Bootstrap shallow-clones at depth=1, so only the bootstrap-time HEAD is
    present. Without this fetch, `git reset --hard <historical_sha>` would
    fail silently or reset against the wrong tree, masking validation bugs.

    Strategy:
      0. Ensure git is installed (Python-slim bootstrap images often skip it).
      1. If the commit already exists locally (e.g. it IS bootstrap HEAD or
         we fetched it on a prior PR in this sandbox), skip — fast path.
      2. Else `git fetch --depth 1 origin <sha>` to pull just that one commit.
         Falls back to deepening the shallow clone if the server refuses
         a by-sha fetch.

    Returns True if base_commit is now reachable, False otherwise.
    """
    if not _ensure_git(sandbox):
        logger.warning("validate_pr: could not install git in sandbox; skipping fetch")
        return False
    # Fast path: already present
    have = sandbox.exec(
        f"git -C /workspace cat-file -e {base_commit} 2>/dev/null && echo OK",
        timeout=15,
    )
    if have.ok and "OK" in have.stdout:
        return True

    # Try direct by-sha fetch (works on GitHub since 2017's uploadpack.allowAnySHA1InWant)
    r = sandbox.exec(
        f"cd /workspace && git fetch --depth 1 origin {base_commit}",
        timeout=timeout,
    )
    if r.ok:
        return True
    logger.warning(
        "validate_pr: direct fetch of %s failed (%s); deepening clone",
        base_commit[:12],
        r.stderr.strip()[:200] if r.stderr else "",
    )
    # Fallback: unshallow. Slower but always works.
    r = sandbox.exec(
        "cd /workspace && git fetch --unshallow origin || git fetch --depth 1000 origin",
        timeout=timeout * 2,
    )
    if not r.ok:
        return False
    # Re-check
    have = sandbox.exec(
        f"git -C /workspace cat-file -e {base_commit} 2>/dev/null && echo OK",
        timeout=15,
    )
    return have.ok and "OK" in have.stdout


def _slice_test_output(output: str) -> str:
    """Trim to just the test-runner section between the START/END markers.

    The markers are echoed to stdout (see `_build_stage_script`); since
    `truncated()` puts stdout before stderr, `find` returns the stdout
    occurrence first — i.e. the real test section, not the `set -x` trace.
    """
    start = output.find("R2E_START_TEST_OUTPUT")
    end = output.find("R2E_END_TEST_OUTPUT")
    if start == -1:
        return output
    chunk = output[start:end] if end > start else output[start:]
    # Drop the marker line itself
    nl = chunk.find("\n")
    return chunk[nl + 1 :] if nl != -1 else chunk


def _timed_out(
    stage: str, timeout: int, *, pre_log: str = "", post_log: str = ""
) -> ValidationOutcome:
    """A timed-out stage cannot produce a task, only a plausible-looking one.

    `sandbox.exec` returns whatever the suite managed to print before it was
    killed, and `parse_logs` reads that partial log perfectly happily. The two
    stages then cover different, wall-clock-determined subsets of the suite, so
    the F2P/P2P sets are computed from a comparison that never happened: a test
    that passed in a complete pre-run and was never reached in a truncated
    post-run silently drops out of P2P, and the resulting task ships with a
    regression guard missing. Nothing downstream can detect this — the task
    looks well-formed, and the verifier agrees with it.

    So the candidate is dropped. Losing a PR to a slow suite is cheap; shipping
    a task whose oracle is wrong is not.
    """
    logger.warning("validate_pr: %s stage timed out after %ss — dropping candidate", stage, timeout)
    return ValidationOutcome(
        status="failed",
        reason=f"{stage} stage timed out after {timeout}s (partial test log is not comparable)",
        pre_log=pre_log,
        post_log=post_log,
    )


# ----- dependency drift ------------------------------------------------------
#
# The validation sandbox is provisioned ONCE, at bootstrap HEAD, and reused for
# every candidate PR — but each PR's stages run at that PR's own `base_sha`. If
# the dependency manifest moved between the two commits, the installed packages
# are not what the base commit asks for, and F2P/P2P can come out wrong with no
# signal that it happened. Re-provisioning per PR is far too expensive, so we
# DETECT the condition and record it on the emitted task as a caveat.
#
# Per language, the file that declares dependencies. Lock files are included
# where they're the thing that actually pins installed versions.
_DEPENDENCY_MANIFESTS: dict[str, tuple[str, ...]] = {
    "python": ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg"),
    "node": ("package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"),
    "go": ("go.mod", "go.sum"),
    "rust": ("Cargo.toml", "Cargo.lock"),
    "java": ("pom.xml", "build.gradle", "build.gradle.kts"),
    "c_cpp": ("CMakeLists.txt", "conanfile.txt"),
}


def dependency_manifests_for(language: str | None) -> tuple[str, ...]:
    """Manifest paths worth comparing for `language` (empty if unknown)."""
    return _DEPENDENCY_MANIFESTS.get((language or "").lower(), ())


def manifests_differ(
    bootstrap_manifests: dict[str, str | None],
    base_manifests: dict[str, str | None],
) -> bool:
    """True if any manifest's content differs between the two commits.

    A manifest missing at *both* commits says nothing and is ignored; a
    manifest present at one and absent at the other is drift (a dependency
    file was added or removed between the commits).
    """
    for path in set(bootstrap_manifests) | set(base_manifests):
        left = bootstrap_manifests.get(path)
        right = base_manifests.get(path)
        if left is None and right is None:
            continue
        if left != right:
            logger.info("dependency drift: %s differs between bootstrap HEAD and base commit", path)
            return True
    return False


def _read_manifests(
    sandbox: DockerSandbox, ref: str, paths: tuple[str, ...], *, timeout: int = 30
) -> dict[str, str | None]:
    """`git show <ref>:<path>` for each path. None when the file isn't there."""
    out: dict[str, str | None] = {}
    for path in paths:
        r = sandbox.exec(f"git -C /workspace show {ref}:{path}", timeout=timeout)
        out[path] = r.stdout if r.ok else None
    return out


def detect_dependency_drift(
    sandbox: DockerSandbox,
    *,
    bootstrap_ref: str,
    base_commit: str,
    language: str | None,
) -> bool:
    """Did the dependency manifest move between the sandbox's installed state
    (bootstrap HEAD) and this PR's base commit?

    Best-effort: any failure to read either side returns False. This is a
    caveat recorded on the task, never a reason to lose a candidate, so an
    unreadable manifest must not be louder than the task itself.
    """
    paths = dependency_manifests_for(language)
    if not paths or not bootstrap_ref or not base_commit:
        return False
    if bootstrap_ref == base_commit:
        return False
    try:
        at_bootstrap = _read_manifests(sandbox, bootstrap_ref, paths)
        at_base = _read_manifests(sandbox, base_commit, paths)
    except Exception as exc:
        logger.warning("dependency-drift check failed (%s); assuming no drift", exc)
        return False
    return manifests_differ(at_bootstrap, at_base)


def validate_pr(
    *,
    sandbox: DockerSandbox,
    base_commit: str,
    patch: str,
    test_patch: str,
    test_cmds: list[str],
    language: str | None = None,
    timeout: int = 600,
) -> ValidationOutcome:
    """Run the two-stage validation and return the resulting outcome.

    Re-uses a shared sandbox across PRs; the `git reset --hard` at the top of
    each stage script guarantees a clean working tree.

    `language` is the LanguageHint value (e.g. "python", "go") used as a
    fallback when test_cmds doesn't name a known runner. Almost always the
    runner is inferred from test_cmds — language is just a safety net.
    """
    if not test_cmds:
        return ValidationOutcome(
            status="failed",
            reason="bootstrap did not record any test_cmds",
        )

    # Ensure the base commit is in the container's object db before any
    # `git reset --hard <sha>`. Bootstrap shallow-clones at depth=1, so
    # historical PR base commits aren't present by default.
    if not _fetch_base_commit(sandbox, base_commit, timeout=timeout):
        return ValidationOutcome(
            status="failed",
            reason=f"could not fetch base_commit {base_commit[:12]} in sandbox",
        )

    # Stage 1: pre-fix (apply test_patch only) — captures the "buggy" baseline.
    pre_script = _build_stage_script(
        base_commit,
        apply_patch=None,
        apply_test_patch=test_patch,
        test_cmds=test_cmds,
    )
    logger.info("validate_pr: running pre-fix stage at %s", base_commit[:12])
    pre = sandbox.exec(pre_script, timeout=timeout)
    if pre.timed_out:
        return _timed_out("pre-fix", timeout, pre_log=pre.truncated(max_chars=5_000_000))
    # Parse the FULL output — a real suite emits 100k+ chars of per-test lines;
    # truncating to 20k elided the test section and zeroed F2P detection.
    pre_log = pre.truncated(max_chars=5_000_000)
    pre_status = parse_logs(test_cmds, _slice_test_output(pre_log), language=language)

    # If the test_patch itself failed to apply, no point continuing.
    if "error: patch failed" in pre_log.lower() or "patch does not apply" in pre_log.lower():
        return ValidationOutcome(
            status="failed",
            reason="test_patch failed to apply at base_commit",
            pre_log=pre_log,
        )

    # Stage 2: post-fix (apply both patch and test_patch).
    post_script = _build_stage_script(
        base_commit,
        apply_patch=patch,
        apply_test_patch=test_patch,
        test_cmds=test_cmds,
    )
    logger.info("validate_pr: running post-fix stage")
    post = sandbox.exec(post_script, timeout=timeout)
    if post.timed_out:
        return _timed_out(
            "post-fix",
            timeout,
            pre_log=pre_log,
            post_log=post.truncated(max_chars=5_000_000),
        )
    post_log = post.truncated(max_chars=5_000_000)
    post_status = parse_logs(test_cmds, _slice_test_output(post_log), language=language)

    if "error: patch failed" in post_log.lower() or "patch does not apply" in post_log.lower():
        return ValidationOutcome(
            status="failed",
            reason="gold patch failed to apply at base_commit",
            pre_log=pre_log,
            post_log=post_log,
        )

    # Compute F2P / P2P. Both sets are over the union of test names seen.
    #
    # F2P counts ERROR->PASSED as well as FAILED->PASSED: a new test that
    # references a symbol the fix introduces *errors* (import/collection
    # failure) at base_commit rather than asserting-failing. Restricting F2P
    # to FAILED-only silently drops these, killing otherwise-valid candidates
    # on `no_fail_to_pass`. Both transitions mean "broken at base, fixed by
    # the gold patch" — the FAIL_TO_PASS contract.
    fail_to_pass: list[str] = []
    pass_to_pass: list[str] = []
    for tname, pre_st in pre_status.items():
        post_st = post_status.get(tname)
        if pre_st in ("FAILED", "ERROR") and post_st == "PASSED":
            fail_to_pass.append(tname)
        elif pre_st == "PASSED" and post_st == "PASSED":
            pass_to_pass.append(tname)

    # Diagnostics: when yield is poor, this tells us whether the suite even
    # ran (test counts) and where transitions went — distinguishing "genuine
    # non-bugfix PR" from "validation didn't detect the failing test".
    logger.info(
        "validate_pr: parsed pre=%d post=%d tests; f2p=%d p2p=%d (pre statuses: %s)",
        len(pre_status),
        len(post_status),
        len(fail_to_pass),
        len(pass_to_pass),
        {s: sum(1 for v in pre_status.values() if v == s) for s in set(pre_status.values())},
    )

    if not fail_to_pass:
        return ValidationOutcome(
            status="failed",
            reason="no fail-to-pass tests after validation",
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            pre_log=pre_log,
            post_log=post_log,
        )

    return ValidationOutcome(
        status="verified",
        fail_to_pass=sorted(fail_to_pass),
        pass_to_pass=sorted(pass_to_pass),
        pre_log=pre_log,
        post_log=post_log,
    )
