"""Thin OpenAI client wrapper used for diagnosis and task generation."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = os.environ.get("VEKTORI_MODEL", "gpt-5-nano")


def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running vektori-trace."
        )
    return OpenAI(api_key=api_key)


def call_json(
    system: str,
    user: str,
    schema_name: str,
    json_schema: dict[str, Any],
    model: str | None = None,
) -> dict[str, Any]:
    """Call the model and parse a JSON object matching json_schema (strict mode)."""
    client = _client()
    resp = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": json_schema,
                "strict": True,
            },
        },
    )
    content = resp.choices[0].message.content
    return json.loads(content)
