"""Token resolution — gh CLI first, env var fallback. No secret ever logged."""

from __future__ import annotations

import os
import shutil
import subprocess

from vektori_trace.mining.spec import AuthSpec, RepoSpec


class AuthError(RuntimeError):
    pass


def resolve_github_token(repo: RepoSpec, auth: AuthSpec) -> str | None:
    """Return a GitHub token following the documented resolution order.

    Order:
      1. repo.auth_token_env (if explicitly set)
      2. `gh auth token` (if auth.use_gh_cli)
      3. $GITHUB_TOKEN
      4. None (anonymous)
    """
    if repo.auth_token_env:
        token = os.environ.get(repo.auth_token_env)
        if token:
            return token

    if auth.use_gh_cli and shutil.which("gh"):
        try:
            result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            pass

    return os.environ.get(auth.github_token_env)


def resolve_llm_api_key(provider: str, llm_api_key_env: str | None = None) -> str | None:
    """Return an LLM provider API key based on the provider name."""
    if llm_api_key_env:
        v = os.environ.get(llm_api_key_env)
        if v:
            return v

    defaults = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "huggingface": "HF_TOKEN",
        "together": "TOGETHER_API_KEY",
        "groq": "GROQ_API_KEY",
    }
    env_name = defaults.get(provider.lower())
    if env_name:
        return os.environ.get(env_name)
    return None


def auth_clone_url(repo_url: str, token: str | None) -> str:
    """Inject token into URL for private clone. Token never logged."""
    if not token:
        return repo_url
    if repo_url.startswith("https://github.com/"):
        return repo_url.replace("https://", f"https://x-access-token:{token}@")
    return repo_url


def resolve_repo_token(repo: RepoSpec, auth: AuthSpec) -> str | None:
    return resolve_github_token(repo, auth)
