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

**Re-exports are lazy (PEP 562).** These names used to be imported eagerly,
which meant `import vektori_trace.mining.atif` — a self-contained trajectory
parser whose only dependencies are `json`, `pathlib` and our own `schema` —
pulled in the whole pipeline: pydantic specs, docker, GitHub auth. That is
merely slow for the CLI, but fatal on a training container, which carries
torch and no mining dependencies. It surfaced as `ModuleNotFoundError: No
module named 'pydantic'` after the image pull, on a GPU billing by the second.

The public API is unchanged: `from vektori_trace.mining import
PRRuntimePipeline` still works and still imports what it needs. It just no
longer happens as a side effect of touching an unrelated submodule.
"""

from __future__ import annotations

from typing import Any

#: Public name → the submodule that defines it. Resolved on first access.
_EXPORTS: dict[str, str] = {
    "BootstrapResult": "vektori_trace.mining.bootstrap",
    "LanguageHint": "vektori_trace.mining.bootstrap",
    "ensure_bootstrap": "vektori_trace.mining.bootstrap",
    "CommitRuntimePipeline": "vektori_trace.mining.commit_runtime",
    "PRRuntimePipeline": "vektori_trace.mining.pipeline",
    "PipelineResult": "vektori_trace.mining.result",
    "AuthSpec": "vektori_trace.mining.spec",
    "BootstrapSpec": "vektori_trace.mining.spec",
    "CommitRuntimeOptions": "vektori_trace.mining.spec",
    "LLMSpec": "vektori_trace.mining.spec",
    "OutputSpec": "vektori_trace.mining.spec",
    "PipelineInput": "vektori_trace.mining.spec",
    "PRRuntimeOptions": "vektori_trace.mining.spec",
    "RepoSpec": "vektori_trace.mining.spec",
}


def __getattr__(name: str) -> Any:
    """Import a re-exported name on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module), name)
    globals()[name] = value  # cache: subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


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
