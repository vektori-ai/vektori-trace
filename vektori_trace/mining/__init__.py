"""Sandboxed, test-verified RL-task generation from a real GitHub repo's PR history.

Simplified/moved from repo2rlenv (github.com/huggingface/Repo2RLEnv): only the
pr_runtime pipeline + its bootstrap/emitter/reward machinery, GitHub-only.
"""

from vektori_trace.mining.bootstrap import BootstrapResult, LanguageHint, ensure_bootstrap
from vektori_trace.mining.pipeline import PRRuntimePipeline
from vektori_trace.mining.result import PipelineResult
from vektori_trace.mining.spec import (
    AuthSpec,
    BootstrapSpec,
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
