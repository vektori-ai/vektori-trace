"""Commit-level mining — the sibling pipeline to `pr_runtime`.

`pr_runtime` can only see fixes that arrived as a merged PR carrying a linked
issue. That gate is there for a good reason (the issue is the problem statement
written *before* the fix existed), but it is expensive: an 800-PR prefect mine
dropped 434 candidates — 54% — as `no_linked_issue`, and it cannot see fixes
that were committed straight to main at all.

This pipeline walks `git log` instead, then hands the resulting
(patch, test_patch, base_commit) tuple to the *same* validation harness, so the
F2P/P2P oracle and the graded verifier are identical to `pr_runtime`'s. What
differs is everything upstream of validation:

  * candidates come from a clone, not the GitHub API
  * no reviewer has approved anything, so the metadata/structural filters do
    the work a PR template does for free
  * the problem statement is LLM-synthesized (see `_SYNTH_SYSTEM`)

Why synthesis is not optional here
----------------------------------
A commit message is written by the fixer, after the fix, and routinely names
the function that changed or pastes a changelog bullet describing the solution.
An agent can then score 1.0 by reading the prompt rather than solving anything,
which silently inflates every downstream pass@k number. Restating the commit as
the symptom a user would have reported removes that channel. Upstream's audit
of this change moved instruction cleanliness from ~33% to ~100%.

Design credit: commit-level curation is R2E-Gym's SWE-GEN (Jain et al.,
COLM '25); this follows the shape of Repo2RLEnv's `commit_runtime` (Apache-2.0).
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from vektori_trace.mining import github
from vektori_trace.mining.auth import resolve_repo_token
from vektori_trace.mining.bootstrap.spec import BootstrapResult
from vektori_trace.mining.emitter import HarborTask, write_harbor_task
from vektori_trace.mining.git_local import (
    CommitInfo,
    GitError,
    clone_for_log,
    list_commits,
    show_diff,
)
from vektori_trace.mining.llm import complete
from vektori_trace.mining.pipeline import (
    AGENT_ALLOWED_HOSTS,
    CHECKPOINT_FILENAME,
    MODEL_PATCH_PATH,
    _count_new_test_funcs,
    _diff_loc_changed,
    _difficulty_bucket,
    _files_in_patch,
    _load_checkpoint,
    _reflow_pr_body,
    _runtime_aux_files,
    _strip_info_leak,
    _word_count,
    _write_checkpoint,
    build_environment_dockerfile,
    build_eval_script,
    model_patch_collect,
    normalize_test_cmds_for_runtime,
    split_patch_and_test_patch,
    targeted_test_cmds_for_pr,
)
from vektori_trace.mining.result import PipelineResult
from vektori_trace.mining.spec import CommitRuntimeOptions, PipelineInput

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "0.1.0"

# `Closes #12` / `fixes [#12](url)` trailers. Kept out of the instruction: the
# number is a pointer to the answer, and in synthesized output it would also
# tempt the model to reference an issue the agent cannot read.
_CLOSES_RE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+\[?#\d+\]?(?:\([^)]*\))?",
    re.IGNORECASE,
)

# Conventional-commit prefix, stripped from the subject before it is shown.
_CC_PREFIX_RE = re.compile(
    r"^(?:fix|feat|chore|docs|refactor|test|perf|build|ci|style|revert)"
    r"(?:\([^)]+\))?!?:\s*",
    re.IGNORECASE,
)

# Conventional-commit types that are definitionally not bug fixes. Rejecting on
# the type is much more precise than keyword matching, when the repo uses them.
_NON_BUG_TYPE_RE = re.compile(
    r"^(?:chore|docs|feat|refactor|style|test|ci|build|perf|revert)"
    r"(?:\([^)]+\))?!?:\s*",
    re.IGNORECASE,
)

# For repos that don't use conventional commits, require *some* positive
# evidence this is a fix. Without it, feature work with no type prefix sails
# straight through to the expensive validation stage.
_BUGFIX_KEYWORD_RE = re.compile(
    r"\b(?:fix(?:e[sd])?|fixing|bug(?:fix|s)?|regression|crash(?:e[sd]|ing)?|broken|"
    r"incorrect(?:ly)?|wrong(?:ly)?|fail(?:s|ed|ing|ure)?|defect|hotfix|"
    r"raise[sd]?|error|exception|traceback|leak|deadlock|race)\b",
    re.IGNORECASE,
)

# Bot authors. Their commits are real and their diffs are large, which is the
# worst combination: they consume validation budget and never yield a bug fix.
_DEFAULT_BOT_PATTERNS = (
    "[bot]",
    "dependabot",
    "renovate",
    "github-actions",
    "pre-commit-ci",
)

_SYNTH_SYSTEM = """You are writing a GitHub bug report for an AI coding agent to fix.

You are given the commit message (and maybe a linked issue) that FIXED a bug.
Rewrite it into a clear problem statement describing ONLY the observed problem
and expected behavior — the symptom, as a user would report it BEFORE any fix
existed.

STRICT RULES:
- Describe the symptom + expected vs actual behavior. Include a short
  reproduction if one is evident.
- Do NOT describe the solution, the fix approach, or which functions/files/tests
  to change. Do NOT mention "fix", "patch", commit SHAs, PR/issue numbers,
  file names, test names, "Signed-off-by", or changelog bullets.
- Output exactly: a `**Title:**` line, then a `## Description` section. Markdown
  allowed. Nothing else. Keep it concise (2-6 sentences)."""

# Identifiers defined or modified by a patch. `+` lines only: a name the patch
# *introduces* is the answer, whereas a name in context was already in the repo
# and the agent can find it anyway.
_DEF_RE = re.compile(
    r"^\+\s*(?:def|class|func|fn|type|struct|interface)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)

# Identifiers short enough to appear in ordinary prose ("f", "id", "run") would
# make the leak gate fire on every instruction. Below this length the token
# carries little solution signal anyway.
_MIN_LEAK_TOKEN_LEN = 5


def leaked_identifiers(instruction: str, patch: str, test_patch: str) -> list[str]:
    """Return solution identifiers that survived into the instruction.

    The synthesis prompt *asks* the model not to name the fix. Measured on
    gpt-5-nano it complies partially: it drops file names, test names and issue
    numbers, but keeps function names and method calls — which is most of the
    patch. A task whose prompt names the function to change is solvable by
    reading rather than reasoning, and scores 1.0 either way, so nothing
    downstream can tell a real solve from a leaked one.

    Checked rather than trusted, because a silently gameable task inflates every
    pass@k number computed from it.
    """
    names: set[str] = set()
    for src in (patch, test_patch):
        names.update(_DEF_RE.findall(src or ""))
    for path in _files_in_patch(patch) + _files_in_patch(test_patch):
        stem = Path(path).stem
        if stem:
            names.add(stem)

    hits = []
    for name in names:
        if len(name) < _MIN_LEAK_TOKEN_LEN:
            continue
        if re.search(rf"\b{re.escape(name)}\b", instruction):
            hits.append(name)
    return sorted(hits)


_TASK_FOOTER = (
    "## Task\n\n"
    "Modify the repository so that the issue described above is resolved. "
    "The task's test suite verifies your patch by applying it on top of "
    "the base commit `{base}` and running the modified tests."
)


def _strip_commit_prefix(subject: str) -> str:
    """Drop the conventional-commit type prefix, keeping the description."""
    return _CC_PREFIX_RE.sub("", subject or "", count=1).strip()


def build_instruction_from_commit(
    commit: CommitInfo, *, issue: tuple[str, str] | None = None
) -> str:
    """Render a commit (or its linked issue) as a task prompt, without an LLM.

    This is the fallback path — used when synthesis is disabled or the LLM call
    fails. It is strictly worse than synthesis for the reason in the module
    docstring, so it scrubs what it can: `Closes #N`, cross-refs, SHAs and
    template noise.
    """
    if issue is not None:
        i_title, i_body = issue
        title = _strip_info_leak(i_title or "").strip()
        body = _reflow_pr_body(_strip_info_leak(_CLOSES_RE.sub("", i_body or ""))).strip()
    else:
        title = _strip_info_leak(_strip_commit_prefix(commit.subject)).strip()
        body = _reflow_pr_body(_strip_info_leak(_CLOSES_RE.sub("", commit.body or ""))).strip()
    if not title:
        title = _strip_commit_prefix(commit.subject) or "(no title)"

    parts = [f"# Issue\n\n**Title:** {title}"]
    if body:
        parts.append("## Description\n\n" + body)
    parts.append(_TASK_FOOTER.format(base=commit.parent_sha[:12]))
    return "\n\n".join(parts)


class CommitRuntimePipeline:
    """Commit-level mining with the same sandbox-verified F2P/P2P oracle."""

    def __init__(
        self,
        input: PipelineInput,
        options: CommitRuntimeOptions,
        bootstrap: BootstrapResult | None = None,
    ):
        if bootstrap is None:
            raise RuntimeError(
                "commit_runtime requires a BootstrapResult — run ensure_bootstrap() first"
            )
        self.input = input
        self.options = options
        self.bootstrap = bootstrap
        self._progress_cb = None
        self._sandbox = None
        self._llm_cost_usd = 0.0
        self._synthesized = 0
        self._synthesis_failures = 0
        self._leaked = 0
        self._leak_retries = 0

    def set_progress_callback(self, cb) -> None:
        self._progress_cb = cb

    def _emit_progress(self, name: str, outcome: str, reason: str = "") -> None:
        if self._progress_cb is not None:
            try:
                self._progress_cb(name=name, outcome=outcome, reason=reason)
            except Exception as exc:
                logger.debug("progress callback failed: %s", exc)

    # ----- instruction synthesis ---------------------------------------------

    def _call_synthesis(
        self, commit: CommitInfo, issue: tuple[str, str] | None, forbid: list[str]
    ) -> str | None:
        """One synthesis call. `forbid` names identifiers to keep out, if any."""
        src = f"Commit subject: {commit.subject}\n\nCommit body:\n{commit.body or ''}\n"
        if issue is not None:
            src += f"\nLinked issue title: {issue[0]}\nLinked issue body:\n{issue[1] or ''}\n"
        system = _SYNTH_SYSTEM
        if forbid:
            # The generic rule already said "don't name the fix" and was not
            # followed. Naming the exact tokens is a materially stronger
            # instruction than restating the policy.
            system += (
                "\n\nYou already produced a draft that leaked implementation details. "
                "These identifiers name the solution and MUST NOT appear in any form: "
                + ", ".join(forbid)
                + ". Describe the user-visible symptom only."
            )
        try:
            resp = complete(
                self.input.llm,
                system=system,
                user=src,
                max_tokens=self.options.max_llm_tokens,
                temperature=self.options.llm_temperature,
            )
        except Exception as exc:
            # One bad synthesis must not end the sweep.
            logger.warning("commit %s: synthesis failed: %s", commit.short_sha, exc)
            return None
        self._llm_cost_usd += getattr(resp, "cost_usd", 0.0) or 0.0
        return (resp.content or "").strip()

    def _synthesize_problem_statement(
        self,
        commit: CommitInfo,
        issue: tuple[str, str] | None,
        *,
        patch: str = "",
        test_patch: str = "",
    ) -> str | None:
        """LLM-rewrite the commit into a problem statement, then verify it.

        Returns the `**Title:** … ## Description …` block, or None when the
        call fails, comes back too thin, or still names the solution after a
        stricter retry. None means the caller must not emit a synthesized
        instruction for this commit.
        """
        if self.input.llm is None:
            return None

        body = self._call_synthesis(commit, issue, forbid=[])
        if body is None:
            self._synthesis_failures += 1
            return None
        if _word_count(body) < 10:
            logger.warning("commit %s: synthesis too thin", commit.short_sha)
            self._synthesis_failures += 1
            return None

        leaks = leaked_identifiers(body, patch, test_patch)
        if leaks:
            logger.info(
                "commit %s: draft leaked %s — retrying with an explicit blocklist",
                commit.short_sha,
                ", ".join(leaks),
            )
            retry = self._call_synthesis(commit, issue, forbid=leaks)
            if retry and _word_count(retry) >= 10:
                still = leaked_identifiers(retry, patch, test_patch)
                if not still:
                    self._synthesized += 1
                    self._leak_retries += 1
                    return retry
                leaks = still
            self._leaked += 1
            logger.warning(
                "commit %s: instruction still leaks %s after retry",
                commit.short_sha,
                ", ".join(leaks),
            )
            return None

        self._synthesized += 1
        return body

    def _build_instruction(
        self,
        commit: CommitInfo,
        *,
        issue: tuple[str, str] | None,
        patch: str = "",
        test_patch: str = "",
    ) -> str | None:
        """Return the instruction, or None when no clean one could be produced.

        None is a skip, not a fallback to raw commit text. The raw text is what
        leaked in the first place — emitting it after synthesis failed would
        ship exactly the gameable task the synthesis exists to prevent.
        """
        if self.options.synthesize_with_llm:
            synth = self._synthesize_problem_statement(
                commit, issue, patch=patch, test_patch=test_patch
            )
            if synth is None:
                return None
            return f"# Issue\n\n{synth}\n\n" + _TASK_FOOTER.format(base=commit.parent_sha[:12])
        return build_instruction_from_commit(commit, issue=issue)

    # ----- filters ------------------------------------------------------------

    def _metadata_filter(self, commit: CommitInfo) -> str | None:
        """Message-level filters. No clone read, no container — run these first."""
        if self.options.skip_merge_commits and commit.is_merge:
            return "merge_commit"
        if not commit.parent_sha:
            # A root commit has nothing to diff against, so there is no
            # "before" state for the tests to fail in.
            return "root_commit"

        author = f"{commit.author_name} {commit.author_email}".lower()
        patterns = [p.lower() for p in self.options.exclude_authors] or list(
            _DEFAULT_BOT_PATTERNS
        )
        if any(p in author for p in patterns):
            return "excluded_author"

        if _word_count(commit.message) < self.options.min_message_words:
            return "message_too_short"

        subject = commit.subject or ""
        if _NON_BUG_TYPE_RE.match(subject):
            return "non_bug_type"

        # Positive evidence required: a `fix:` prefix, a linked issue, or a
        # bugfix keyword somewhere in the message.
        has_fix_prefix = bool(re.match(r"^fix(?:\([^)]+\))?!?:", subject, re.IGNORECASE))
        has_issue_ref = bool(_CLOSES_RE.search(commit.message))
        has_keyword = bool(_BUGFIX_KEYWORD_RE.search(subject))
        if not (has_fix_prefix or has_issue_ref or has_keyword):
            return "not_a_bugfix"

        if (
            self.options.min_problem_statement_words > 0
            and _word_count(commit.message) < self.options.min_problem_statement_words
        ):
            return "problem_statement_too_short"
        return None

    def _structural_filter(self, source_patch: str, test_patch: str) -> str | None:
        """Diff-level filters. Cheap relative to a container run, so still pre-validation."""
        source_files = _files_in_patch(source_patch)
        if (
            self.options.skip_ci_only
            and source_files
            and all(p.startswith(".github/") for p in source_files)
        ):
            return "ci_only_patch"
        if len(source_files) > self.options.max_source_files_per_commit:
            # A sweeping refactor is not a bug fix, and its F2P set would
            # describe the refactor rather than a defect.
            return "too_many_source_files"
        if self.options.require_new_test_funcs and _count_new_test_funcs(test_patch) < 1:
            return "no_new_test_funcs"
        return None

    # ----- run loop -----------------------------------------------------------

    def run(self, out_dir: Path) -> PipelineResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        token = resolve_repo_token(self.input.repo, self.input.auth)
        self._token = token
        owner, name = self.input.repo.owner_name

        clone_root = Path(tempfile.mkdtemp(prefix="vektori-commit-mine-"))
        clone_dir = clone_root / "repo"
        skip_reasons: dict[str, int] = {}
        emitted = 0
        checkpoint = _load_checkpoint(out_dir)

        try:
            url = self._clone_url(owner, name, token)
            logger.info(
                "cloning %s/%s (depth=%d) for commit mining",
                owner,
                name,
                self.options.clone_depth,
            )
            try:
                clone_for_log(url, clone_dir, depth=self.options.clone_depth)
            except GitError as exc:
                raise RuntimeError(f"failed to clone {owner}/{name}: {exc}") from exc

            commits = list_commits(
                clone_dir,
                since=self.options.since,
                until=self.options.until,
                limit=self.options.limit,
                branch=self.options.branch,
                first_parent=self.options.first_parent,
            )
            logger.info("walking %d commits for %s/%s", len(commits), owner, name)

            for commit in commits:
                label = f"{owner}/{name}@{commit.short_sha}"
                if commit.sha in checkpoint:
                    skip_reasons["already_processed"] = (
                        skip_reasons.get("already_processed", 0) + 1
                    )
                    self._emit_progress(label, "skip", "already_processed")
                    continue
                try:
                    kind, key, message = self._process_commit(
                        commit, owner, name, clone_dir, out_dir
                    )
                except Exception as exc:
                    # Same contract as pr_runtime: one bad candidate is that
                    # candidate's skip, never the sweep's end.
                    logger.warning("commit %s failed: %s", commit.short_sha, exc)
                    kind, key, message = "error", f"unexpected_error:{type(exc).__name__}", str(exc)

                if kind == "emit":
                    emitted += 1
                else:
                    skip_reasons[key] = skip_reasons.get(key, 0) + 1
                self._emit_progress(label, kind, message)
                checkpoint[commit.sha] = key
                _write_checkpoint(out_dir, checkpoint)
        finally:
            if self._sandbox is not None:
                self._sandbox.cleanup()
                self._sandbox = None
            shutil.rmtree(clone_root, ignore_errors=True)

        if self.options.synthesize_with_llm:
            logger.info(
                "instruction synthesis: %d ok (%d needed a leak retry), "
                "%d failed, %d dropped as leaky, $%.4f",
                self._synthesized,
                self._leak_retries,
                self._synthesis_failures,
                self._leaked,
                self._llm_cost_usd,
            )

        return PipelineResult(
            candidates=len(commits),
            emitted=emitted,
            skipped=sum(skip_reasons.values()),
            out_dir=out_dir,
            skip_reasons=skip_reasons,
        )

    def _clone_url(self, owner: str, name: str, token: str | None) -> str:
        """HTTPS clone URL, with the token inlined only for private repos."""
        if token and self.input.repo.access == "private":
            return f"https://x-access-token:{token}@github.com/{owner}/{name}.git"
        return f"https://github.com/{owner}/{name}.git"

    def _process_commit(
        self,
        commit: CommitInfo,
        owner: str,
        name: str,
        clone_dir: Path,
        out_dir: Path,
    ) -> tuple[str, str, str]:
        """Filter, validate and (if it survives) emit one commit."""
        reason = self._metadata_filter(commit)
        if reason:
            return "skip", reason, reason

        try:
            diff = show_diff(clone_dir, commit.sha)
        except GitError as exc:
            logger.warning("commit %s: git show failed: %s", commit.short_sha, exc)
            return "error", "diff_fetch_failed", "diff_fetch_failed"

        patch, test_patch = split_patch_and_test_patch(diff)
        if not patch.strip():
            return "skip", "empty_source_patch", "empty_source_patch"
        if not test_patch.strip():
            return "skip", "no_test_patch", "no_test_patch"

        structural = self._structural_filter(patch, test_patch)
        if structural:
            return "skip", structural, structural

        fail_to_pass: list[str] = []
        pass_to_pass: list[str] = []
        validation_status = "skipped"
        if not self.options.skip_validation:
            if self._sandbox is None:
                self._sandbox = self._start_validation_sandbox()
            from vektori_trace.mining.validate import validate_pr

            targeted_cmds = targeted_test_cmds_for_pr(
                normalize_test_cmds_for_runtime(self.bootstrap.test_cmds),
                _files_in_patch(test_patch),
            )
            outcome = validate_pr(
                sandbox=self._sandbox,
                # The "before" state is the commit's first parent, which is
                # what `git show` diffed against — anything else would apply
                # the patch to a tree it was not written for.
                base_commit=commit.parent_sha,
                patch=patch,
                test_patch=test_patch,
                test_cmds=targeted_cmds,
                language=self.bootstrap.language.value,
                timeout=self.options.validation_timeout_sec,
            )
            fail_to_pass = outcome.fail_to_pass
            pass_to_pass = outcome.pass_to_pass
            validation_status = outcome.status
            if (
                self.options.require_fail_to_pass
                and len(fail_to_pass) < self.options.min_fail_to_pass
            ):
                return "skip", "no_fail_to_pass", outcome.reason or "no_fail_to_pass"

            pass_to_pass = self._cap_pass_to_pass(pass_to_pass)

        issue = self._fetch_linked_issue(commit, owner, name)
        instruction = self._build_instruction(
            commit, issue=issue, patch=patch, test_patch=test_patch
        )
        if instruction is None:
            # Verified-leaky or unsynthesizable. Dropping it costs one task;
            # keeping it costs the credibility of every pass@k computed from
            # the corpus, because a leaked task scores 1.0 without a solve.
            return "skip", "instruction_leaked", "instruction_leaked"

        task = self._build_task(
            commit,
            patch,
            test_patch,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            validation_status=validation_status,
            issue=issue,
            instruction=instruction,
        )
        write_harbor_task(task, out_dir)
        logger.info(
            "emitted task %s (F2P=%d, P2P=%d)", task.name, len(fail_to_pass), len(pass_to_pass)
        )
        return "emit", "emitted", task.name

    def _cap_pass_to_pass(self, pass_to_pass: list[str]) -> list[str]:
        """Bound the regression guard.

        The graded reward is `f2p_rate * p2p_rate`, so every extra P2P test is
        another chance for an unrelated flake to scale a correct solve down.
        Sorted before truncating so the kept subset is deterministic across
        runs — an unstable P2P set would make two mines of the same commit
        produce different tasks.
        """
        cap = self.options.max_pass_to_pass
        if cap > 0 and len(pass_to_pass) > cap:
            logger.info("capping P2P from %d to %d", len(pass_to_pass), cap)
            return sorted(pass_to_pass)[:cap]
        return pass_to_pass

    def _fetch_linked_issue(
        self, commit: CommitInfo, owner: str, name: str
    ) -> tuple[str, str] | None:
        """Fetch the linked issue when the commit references one.

        Worth the API call even with synthesis on: the issue is the symptom
        described before the fix existed, so it gives the rewrite far better
        source material than a one-line commit subject.
        """
        m = re.search(r"#(\d+)", commit.message or "")
        if not m or not _CLOSES_RE.search(commit.message or ""):
            return None
        try:
            return github.fetch_issue(
                owner, name, int(m.group(1)), token=getattr(self, "_token", None)
            )
        except Exception as exc:
            logger.debug("commit %s: issue fetch failed: %s", commit.short_sha, exc)
            return None

    def _start_validation_sandbox(self):
        from vektori_trace.mining.bootstrap.docker import DockerSandbox

        marker = Path(tempfile.mkdtemp(prefix="r2e-commit-runtime-"))
        (marker / ".keep").write_text("")
        return DockerSandbox.start(
            base_image=self.bootstrap.image_tag,
            repo_dir=marker,
            platform=self.input.bootstrap.platform,
        )

    # ----- task builder -------------------------------------------------------

    def _build_task(
        self,
        commit: CommitInfo,
        patch: str,
        test_patch: str,
        *,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        validation_status: str,
        issue: tuple[str, str] | None,
        instruction: str,
    ) -> HarborTask:
        owner, name = self.input.repo.owner_name
        # Short SHA, not a PR number — commits have no monotonic id, and the
        # SHA is what makes the task reproducible from the repo alone.
        task_id = f"{owner}__{name}-{commit.short_sha}"

        resolved_test_cmds = targeted_test_cmds_for_pr(
            normalize_test_cmds_for_runtime(self.bootstrap.test_cmds),
            _files_in_patch(test_patch),
        )
        eval_script = build_eval_script(
            base_commit=commit.parent_sha,
            test_patch=test_patch,
            test_cmds=resolved_test_cmds,
            language=self.bootstrap.language.value,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            model_patch_path=MODEL_PATCH_PATH,
        )
        image_ref = (
            self.bootstrap.image_digest
            if self.bootstrap.pushed_to_registry
            else self.bootstrap.image_tag
        )
        dockerfile = build_environment_dockerfile(
            bootstrap_image=image_ref, base_commit=commit.parent_sha
        )

        loc = _diff_loc_changed(patch)
        repo2env = {
            "pipeline": "commit_runtime",
            "pipeline_version": PIPELINE_VERSION,
            "repo": f"{owner}/{name}",
            "ref": commit.parent_sha,
            "reference": f"https://github.com/{owner}/{name}/commit/{commit.sha}",
            "source_access": self.input.repo.access,
            "built_at": datetime.now(UTC).isoformat(),
            # Only claimed when an LLM actually rewrote this task's
            # instruction. `pr_runtime` stamps this field whenever a model is
            # configured at all, which reads as a synthesis that never ran.
            **(
                {"synthesis_llm": self.input.llm.qualified_name}
                if (self.input.llm and self.options.synthesize_with_llm)
                else {}
            ),
            "reward_kinds": ["test_execution", "diff_similarity"],
            "commit_runtime": {
                "commit_sha": commit.sha,
                "parent_sha": commit.parent_sha,
                "authored_at": commit.authored_at,
                "base_commit": commit.parent_sha,
                "fail_to_pass": fail_to_pass,
                "pass_to_pass": pass_to_pass,
                "validation_status": validation_status,
                "bootstrap_image": self.bootstrap.image_digest,
                "linked_issue": bool(issue),
                "instruction_source": (
                    "llm_synthesis" if self.options.synthesize_with_llm else "raw_text"
                ),
                "pass_to_pass_capped": self.options.max_pass_to_pass > 0
                and len(pass_to_pass) >= self.options.max_pass_to_pass,
                "reward_mode": "graded",
            },
            "reward_calibration": {
                "f2p_count": len(fail_to_pass),
                "p2p_count": len(pass_to_pass),
                "source_files": len(_files_in_patch(patch)),
                "loc_changed": loc,
                "difficulty": _difficulty_bucket(len(fail_to_pass), loc),
            },
        }

        return HarborTask(
            name=task_id,
            org=self.input.output.org,
            description=_strip_commit_prefix(commit.subject) or task_id,
            instruction=instruction,
            oracle_diff=patch,
            repo2env=repo2env,
            difficulty="medium",
            category="bugfix",
            keywords=[name, "commit_runtime"],
            environment_dockerfile=dockerfile,
            test_script=eval_script,
            aux_files=(
                _runtime_aux_files(fail_to_pass, pass_to_pass) if fail_to_pass else {}
            ),
            environment_network_mode="allowlist",
            environment_allowed_hosts=list(AGENT_ALLOWED_HOSTS),
            agent_network_mode="allowlist",
            agent_allowed_hosts=list(AGENT_ALLOWED_HOSTS),
            verifier_network_mode="no-network",
            verifier_environment_mode="separate",
            verifier_collect=[{"command": model_patch_collect(commit.parent_sha)}],
            artifacts=[MODEL_PATCH_PATH],
        )


__all__ = [
    "CHECKPOINT_FILENAME",
    "CommitRuntimePipeline",
    "build_instruction_from_commit",
]
