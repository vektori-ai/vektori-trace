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


class RepoSpec(BaseModel):
    url: str
    ref: str = "HEAD"
    access: Literal["public", "private", "auto"] = "auto"
    auth_token_env: str | None = None

    @field_validator("url")
    @classmethod
    def normalize_url(cls, v: str) -> str:
        v = v.strip()
        if "/" not in v:
            raise ValueError(f"repo url must be 'owner/name' or a full GitHub URL, got {v!r}")
        if not v.startswith(("http://", "https://", "git@")):
            v = f"https://github.com/{v}"
        return v.rstrip("/").removesuffix(".git")

    @property
    def owner_name(self) -> tuple[str, str]:
        path = self.url.replace("https://github.com/", "").replace("git@github.com:", "")
        parts = path.rstrip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"cannot parse owner/name from {self.url!r}")
        return parts[-2], parts[-1]


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


LLMSpec.model_rebuild()
