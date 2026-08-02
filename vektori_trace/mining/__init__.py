"""Sandboxed, test-verified RL-task generation from a real GitHub repo's PR history.

Simplified/moved from repo2rlenv (github.com/huggingface/Repo2RLEnv): the
pr_runtime and commit_runtime pipelines + their bootstrap/emitter/reward
machinery, GitHub-only.

The two pipelines are siblings, not alternatives. `pr_runtime` mines merged
PRs carrying a linked issue — high-quality problem statements, written before
the fix existed, but it cannot see fixes that never became such a PR.
`commit_runtime` walks `git log` to reach those, and pays for it by having to
synthesize the problem statement. They emit the same task shape, so their
output pools into one corpus.
"""

from vektori_trace.mining.bootstrap import BootstrapResult, LanguageHint, ensure_bootstrap
from vektori_trace.mining.commit_runtime import CommitRuntimePipeline
from vektori_trace.mining.pipeline import PRRuntimePipeline
from vektori_trace.mining.result import PipelineResult
from vektori_trace.mining.spec import (
    AuthSpec,
    BootstrapSpec,
    CommitRuntimeOptions,
    LLMSpec,
    OutputSpec,
    PipelineInput,
    PRRuntimeOptions,
    RepoSpec,
)

__all__ = [
    "AuthSpec",
    "BootstrapResult",
    "BootstrapSpec",
    "CommitRuntimeOptions",
    "CommitRuntimePipeline",
    "LLMSpec",
    "LanguageHint",
    "OutputSpec",
    "PRRuntimeOptions",
    "PRRuntimePipeline",
    "PipelineInput",
    "PipelineResult",
    "RepoSpec",
    "ensure_bootstrap",
]
