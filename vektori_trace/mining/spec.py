"""Input contract for envgen — GitHub + pr_runtime only.

Trimmed from repo2rlenv's spec/input.py + spec/options.py: no PipelineName
registry, no GitLab/local source handling, no QA/Sandbox/OutputSpec generality
we don't use. One pipeline, one source kind.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_URL_PREFIXES = (
    ("https://github.com/", "https://github.com/"),
    ("http://github.com/", "https://github.com/"),
    ("git@github.com:", "git@github.com:"),
    ("ssh://git@github.com/", "git@github.com:"),
)


class RepoSpec(BaseModel):
    url: str
    ref: str = "HEAD"
    access: Literal["public", "private", "auto"] = "auto"
    auth_token_env: str | None = None

    @field_validator("url")
    @classmethod
    def normalize_url(cls, v: str) -> str:
        """Canonicalize to the bare repo root, dropping any browser path tail.

        Humans copy URLs out of the address bar, so `.../psf/requests/tree/main`
        and `.../psf/requests/pull/1234` are the common case, not the exception.
        Taking the last two path segments reads those as owner=('tree','pull'),
        so we take the *first* two after the host and discard the rest — which
        also makes `url` safe to hand straight to `git clone`.
        """
        v = v.strip()
        if not v:
            raise ValueError("repo url must not be empty")

        prefix = ""
        path = v
        for candidate, canonical in _URL_PREFIXES:
            if v.startswith(candidate):
                prefix, path = canonical, v[len(candidate) :]
                break
        else:
            # Anything carrying a scheme is a URL we don't support, not a bare
            # `owner/name`. Enumerating the four schemes we happen to know lets
            # `ftp://github.com/psf/requests` fall through to the bare branch,
            # where splitting on "/" yields owner='ftp:' — canonicalized to
            # `https://github.com/ftp:/github.com` and handed to `git clone`.
            if "://" in v or v.startswith("git@"):
                raise ValueError(f"only github.com repos are supported, got {v!r}")
            prefix = "https://github.com/"

        parts = [p for p in path.split("/") if p]
        if len(parts) < 2:
            raise ValueError(f"repo url must be 'owner/name' or a full GitHub URL, got {v!r}")
        owner, name = parts[0], parts[1].removesuffix(".git")
        if not owner or not name:
            raise ValueError(f"cannot parse owner/name from {v!r}")
        return f"{prefix}{owner}/{name}"

    @property
    def owner_name(self) -> tuple[str, str]:
        # `url` is canonical after validation: <prefix>owner/name, nothing else.
        for _, canonical in _URL_PREFIXES:
            if self.url.startswith(canonical):
                owner, name = self.url[len(canonical) :].split("/")
                return owner, name
        raise ValueError(f"cannot parse owner/name from {self.url!r}")


class LLMSpec(BaseModel):
    provider: str
    model: str
    api_key_env: str | None = None
    endpoint: str | None = None
    max_concurrent: int = 5
    timeout_sec: int = 120
    fallback: LLMSpec | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.provider}/{self.model}"


class AuthSpec(BaseModel):
    github_token_env: str = "GITHUB_TOKEN"
    use_gh_cli: bool = True
    build_secrets_env: dict[str, str] = Field(default_factory=dict)


class BootstrapSpec(BaseModel):
    enabled: bool = True
    max_iterations: int = 20
    max_seconds: int = 1800
    base_image: str | None = None
    user_dockerfile: Path | None = None
    # Only meaningful alongside `user_dockerfile`. The agent-driven path
    # discovers how to run the suite and reports it in `test_cmds`; a supplied
    # Dockerfile skips the agent, so nothing discovers anything and the field
    # came back empty — which silently made every PR fail validation with
    # `no_fail_to_pass`, since F2P is derived by *running* the suite. If you
    # own the image you have to say how to test it.
    user_test_cmds: list[str] = []
    user_language: Literal["python", "node", "go", "rust", "java", "c_cpp"] | None = None
    cache_dir: Path = Field(
        default_factory=lambda: Path(os.environ.get("R2E_CACHE_DIR", "./workspace/bootstrap"))
    )
    image_registry: str | None = None
    max_llm_spend_usd: float | None = 5.0
    platform: Literal["linux/amd64", "linux/arm64"] = "linux/amd64"
    languages_hint: list[str] | None = None


class OutputSpec(BaseModel):
    org: str = "default"


class PipelineInput(BaseModel):
    repo: RepoSpec
    llm: LLMSpec | None = None
    output: OutputSpec = Field(default_factory=OutputSpec)
    bootstrap: BootstrapSpec = Field(default_factory=BootstrapSpec)
    auth: AuthSpec = Field(default_factory=AuthSpec)


class PRRuntimeOptions(BaseModel):
    """Sandbox-verified PR mining: clones, applies diff, runs tests in the bootstrap image.

    Runs each candidate PR's tests inside the bootstrap container twice — once
    with only `test_patch` applied (captures which tests fail pre-fix), once
    with both `test_patch` and the gold `patch` applied (confirms which now
    pass). Tests that transition fail→pass become the FAIL_TO_PASS oracle;
    tests that pass both times become PASS_TO_PASS regression guards.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Mining ---
    limit: int = 50
    since: date | None = None
    until: date | None = None
    state: Literal["merged"] = "merged"
    skip_drafts: bool = True
    require_linked_issue: bool = True
    languages: list[str] = ["python"]

    # --- Validation ---
    require_fail_to_pass: bool = True
    min_fail_to_pass: int = 1
    validation_timeout_sec: int = 600
    skip_validation: bool = False

    # --- Quality (SWE-bench Lite-style sampling) ---
    lite_filter: bool = False
    max_source_files_per_pr: int = 50
    min_problem_statement_words: int = 0

    # --- Structural filters (cheap, applied before validation) ---
    require_new_test_funcs: bool = True
    skip_ci_only: bool = True


class CommitRuntimeOptions(BaseModel):
    """Commit-level mining — the sibling of `pr_runtime`, not a replacement.

    Walks `git log` instead of the PR list, then hands the resulting
    (patch, test_patch, base_commit) tuple to the *same* validation harness.
    Two reasons it exists:

      1. Yield. A `pr_runtime` mine of 800 prefect PRs dropped 434 (54%) as
         `no_linked_issue`. Those fixes are real and their tests are real; the
         only thing missing was a GitHub issue link.
      2. Reach. Repos that squash-merge or commit straight to main have fixes
         that are not behind a PR at all and are invisible to `pr_runtime`.

    The trade is signal quality: no reviewer approved a commit, so the filters
    below have to do the work a PR template does for free. They are applied
    before validation because validation costs a container run.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Mining ---
    limit: int = 50
    since: date | None = None
    until: date | None = None
    branch: str = "HEAD"
    # bootstrap clones --depth=1, which has nothing to walk. 200 covers roughly
    # a month of an active repo; raise it for a wider --since window.
    clone_depth: int = 200
    first_parent: bool = True

    # --- Filters (cheap, applied before validation) ---
    skip_merge_commits: bool = True
    min_message_words: int = 5  # drops "wip", "fmt", "typo", "address review"
    max_source_files_per_commit: int = 10
    exclude_authors: list[str] = []  # substring match on name or email; bots
    require_new_test_funcs: bool = True
    skip_ci_only: bool = True

    # --- Validation (mirrors PRRuntimeOptions) ---
    require_fail_to_pass: bool = True
    min_fail_to_pass: int = 1
    validation_timeout_sec: int = 600
    skip_validation: bool = False
    # Cap the PASS_TO_PASS guard. A whole-suite P2P of several hundred tests
    # multiplies flake risk into the graded reward (reward = f2p_rate *
    # p2p_rate) and lengthens every rollout, while adding little regression
    # signal past the first few dozen. 0 disables the cap.
    max_pass_to_pass: int = 50

    # --- Instruction synthesis ---
    # On by default, and the pipeline is unsound without it. Commit messages
    # are written by the fixer *after* the fix, so they routinely name the
    # function changed or paste a changelog bullet describing the solution —
    # an agent can score 1.0 by reading the prompt rather than solving. The
    # rewrite restates the symptom as a user would have reported it.
    synthesize_with_llm: bool = True
    # Deliberately low: with synthesis on, the LLM expands terse commits, so
    # this only needs to exclude the near-empty ones that synthesis cannot
    # rescue. A high floor here would re-create the yield problem this
    # pipeline exists to solve.
    min_problem_statement_words: int = 8
    llm_temperature: float = 0.3
    # Reasoning models bill thinking against this budget and emit the visible
    # answer only from what is left. Measured on gpt-5-nano (the default
    # synthesis model): 1024 produced empty content 3/3 times, 2048 succeeded
    # 3/3. 4096 leaves margin for long commit bodies, which lengthen the
    # reasoning. An empty completion is not an error — it arrives as a
    # too-thin synthesis and the task is dropped, so too small a budget looks
    # like "this repo has no clean instructions".
    max_llm_tokens: int = 4096


LLMSpec.model_rebuild()
