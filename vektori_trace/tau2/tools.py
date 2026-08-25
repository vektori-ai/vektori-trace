"""The retail tool schema, taken from Tau2 itself.

Tau2 passes tools to the agent through the API's native `tools=` parameter
(`llm_agent.py`, `tools=self.tools`), built from `Tool.openai_schema`. Training
prompts must carry the same schema through Qwen's chat template, or the model is
trained without the tool definitions it will be served with.

This module imports the live domain rather than hard-coding a copy, so the
schema cannot drift from what serving sends. The extracted schema is hashed into
the manifest; a change in Tau2's revision changes the hash and invalidates the
corpus, which is the intended behaviour.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def load_domain_tools(domain: str = "retail") -> list[dict[str, Any]]:
    """Return the OpenAI-format tool schemas the agent is served with."""
    from tau2.registry import registry

    get_env = registry.get_env_constructor(domain)
    env = get_env()
    return [t.openai_schema for t in env.get_tools()]


def tools_hash(tools: list[dict[str, Any]]) -> str:
    payload = json.dumps(tools, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def mutating_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    """Tools whose names indicate they write to the database.

    Derived from the schema rather than a frozen list so a Tau2 revision that
    adds a mutating tool does not silently bypass the policy gates.
    """
    out = set()
    for t in tools:
        name = (t.get("function") or {}).get("name", "")
        if any(name.startswith(p) for p in
               ("cancel_", "modify_", "return_", "exchange_", "transfer_")):
            out.add(name)
    return out
