"""Trace schema: a run made up of turns, each with an optional role, content, and tool calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Outcome = Literal["win", "loss"]


@dataclass
class ToolCall:
    id: str
    name: str
    args: Any


@dataclass
class Turn:
    index: int
    role: str
    thinking: str | None = None
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Turn:
        return cls(
            index=d.get("index", 0),
            role=d.get("role", "unknown"),
            thinking=d.get("thinking"),
            content=d.get("content"),
            tool_calls=[
                ToolCall(id=tc.get("id", ""), name=tc.get("name", ""), args=tc.get("args"))
                for tc in d.get("toolCalls", []) or []
            ],
            tool_call_id=d.get("toolCallId"),
        )


@dataclass
class Trace:
    run_id: str
    status: str
    turns: list[Turn]
    outcome: Outcome
    source_path: Path

    @classmethod
    def load(cls, path: Path, outcome: Outcome | None = None) -> Trace:
        data = json.loads(path.read_text())
        status = data.get("status", "unknown")
        resolved_outcome = outcome or ("win" if status == "success" else "loss")
        return cls(
            run_id=data.get("runId", path.stem),
            status=status,
            turns=[Turn.from_dict(t) for t in data.get("turns", [])],
            outcome=resolved_outcome,
            source_path=path,
        )

    def condensed(self, max_turns: int = 60, max_field_chars: int = 400) -> str:
        """Compact text rendering of the trace for LLM context, truncated for cost control."""
        turns = self.turns
        if len(turns) > max_turns:
            head = turns[: max_turns // 2]
            tail = turns[-max_turns // 2 :]
            turns = head + [None] + tail  # type: ignore[list-item]

        lines: list[str] = []
        for t in turns:
            if t is None:
                lines.append(f"... [{len(self.turns) - max_turns} turns omitted] ...")
                continue
            piece = f"[{t.index}] {t.role}:"
            if t.thinking:
                piece += f" (thinking: {_truncate(t.thinking, max_field_chars)})"
            if t.content:
                piece += f" {_truncate(t.content, max_field_chars)}"
            for tc in t.tool_calls:
                args = _truncate(json.dumps(tc.args), 200) if tc.args is not None else ""
                piece += f"\n    tool_call {tc.name}({args})"
            lines.append(piece)
        return "\n".join(lines)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"


@dataclass
class ManifestEntry:
    path: Path
    outcome: Outcome | None = None


def load_manifest(manifest_path: Path) -> list[ManifestEntry]:
    data = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    entries = []
    for item in data:
        p = Path(item["path"])
        if not p.is_absolute():
            p = base / p
        entries.append(ManifestEntry(path=p, outcome=item.get("outcome")))
    return entries
