"""Compaction boundary detection (plan §15).

**This module detects boundaries. It does not reconstruct post-compaction
context, and a caller must not treat a detected boundary as a reconstructed
one.** The distinction is the whole reason this file is separate from
`replay_select`:

- *Detection* answers "did Harbor compact here, and at which replay step".
  That is settled: the marker is a `source: "system"` step carrying
  `extra.context_management = {"type": "compaction", "boundary": "replace"}`,
  and across the audited corpus every boundary is followed by an inlined
  `source: "user"` handoff message in the main trajectory.
- *Reconstruction* answers "what context did the agent actually have after the
  boundary". `reopd.prefix_turns_through_step` returns `turns[:end]` — a flat
  slice from index 0 — so a prefix at a post-boundary step still carries every
  pre-boundary turn. Since `boundary: "replace"` means Harbor *dropped* that
  history, marking such a prefix `post_compaction=True` would label it as the
  retained state while its content is the opposite.

So `post_compaction` set from this module means "this step follows boundary N",
which is a fact about position, not about content. Until the prefix builder
honours the replacement, a run must not claim post-compaction coverage
(§8.3/§8.4). `replay_select.select_replay_prefixes` refuses a non-zero
`require_post_compaction` for exactly that reason.

The retained state is *reachable*: it is the inlined user handoff message the
detector records as `handoff_turn_index`. Reconstruction would slice from
there rather than from 0. That change belongs to the prefix builder and is
deliberately not made here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Prose markers for the handoff message Terminus inlines after a boundary.
#: Used to *classify* and locate the retained state, never to gate training.
HANDOFF_MARKERS = (
    "Here are the answers the other agent provided",
    "You are picking up work from a previous AI agent",
    "The next agent has a few questions",
)


class CompactionError(RuntimeError):
    """A compaction record cannot be read as a boundary."""


@dataclass(frozen=True)
class CompactionBoundary:
    """One recorded compaction event in a trace.

    `index` is the ordinal within the trace (0-based), so compaction-1 and
    compaction-2 are distinguishable — §15 makes multiple boundaries
    first-class rather than collapsing them into one boolean.
    """

    index: int
    raw_step_pos: int
    raw_step_id: Any
    boundary_kind: str | None
    sidecars: tuple[str, ...] = ()
    #: Position in the *parsed turn* list, when it could be located.
    marker_turn_index: int | None = None
    #: The inlined user handoff turn — the retained state a future
    #: reconstruction would slice from. Recorded, not yet used.
    handoff_turn_index: int | None = None
    #: First replay step at or after this boundary.
    first_step_after: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "raw_step_pos": self.raw_step_pos,
            "raw_step_id": self.raw_step_id,
            "boundary_kind": self.boundary_kind,
            "sidecars": list(self.sidecars),
            "marker_turn_index": self.marker_turn_index,
            "handoff_turn_index": self.handoff_turn_index,
            "first_step_after": self.first_step_after,
            **self.meta,
        }


def boundaries_from_raw(trajectory_path: Path) -> list[CompactionBoundary]:
    """Read boundaries out of a raw `trajectory.json`.

    Keys off `extra.context_management`, which is structured data, rather than
    the marker's prose. Returns an empty list for a trace that never compacted.
    """
    path = Path(trajectory_path)
    try:
        steps = json.loads(path.read_text())["steps"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise CompactionError(f"{path}: unreadable trajectory: {e}") from e

    out: list[CompactionBoundary] = []
    for pos, step in enumerate(steps):
        cm = (step.get("extra") or {}).get("context_management")
        if not isinstance(cm, dict) or cm.get("type") != "compaction":
            continue
        sidecars: list[str] = []
        obs = step.get("observation") or {}
        for result in obs.get("results") or []:
            for ref in result.get("subagent_trajectory_ref") or []:
                p = ref.get("trajectory_path")
                if p:
                    sidecars.append(p)
        nxt = steps[pos + 1] if pos + 1 < len(steps) else None
        msg = (nxt.get("message") or "") if nxt else ""
        marker = next((m for m in HANDOFF_MARKERS if m in msg), None)
        out.append(
            CompactionBoundary(
                index=len(out),
                raw_step_pos=pos,
                raw_step_id=step.get("step_id"),
                boundary_kind=cm.get("boundary"),
                sidecars=tuple(sidecars),
                meta={
                    "next_source": nxt.get("source") if nxt else None,
                    "next_is_handoff": marker is not None,
                    "handoff_marker": marker,
                },
            )
        )
    return out


def locate_in_turns(
    boundaries: list[CompactionBoundary],
    turns: list[Any],
    steps: list[tuple[int, Any]],
) -> list[CompactionBoundary]:
    """Map raw boundaries onto parsed turns and replay step indices.

    Raw step positions and replay step indices are different coordinate
    systems: the parser splices subagent turns in and `assistant_tool_steps`
    counts only depth-0 assistant tool calls. Matching by prose on the depth-0
    system marker is what bridges them; depth>0 markers are ignored, since the
    same summarization sidecar appears several times and would otherwise be
    counted as extra boundaries.
    """
    marker_turns = [
        t
        for t in turns
        if getattr(t, "role", None) == "system"
        and getattr(t, "subagent_depth", 0) == 0
        and "context summarization" in (getattr(t, "content", "") or "").lower()
    ]

    out: list[CompactionBoundary] = []
    for b in boundaries:
        marker_idx = (
            marker_turns[b.index].index if b.index < len(marker_turns) else None
        )
        handoff_idx = None
        first_after = None
        if marker_idx is not None:
            for t in turns:
                if (
                    t.index > marker_idx
                    and getattr(t, "subagent_depth", 0) == 0
                    and getattr(t, "role", None) == "user"
                ):
                    handoff_idx = t.index
                    break
            after = [i for i, (ti, _) in enumerate(steps) if ti > marker_idx]
            first_after = after[0] if after else None
        out.append(
            CompactionBoundary(
                index=b.index,
                raw_step_pos=b.raw_step_pos,
                raw_step_id=b.raw_step_id,
                boundary_kind=b.boundary_kind,
                sidecars=b.sidecars,
                marker_turn_index=marker_idx,
                handoff_turn_index=handoff_idx,
                first_step_after=first_after,
                meta=dict(b.meta),
            )
        )
    return out


def post_compaction_steps(boundaries: list[CompactionBoundary]) -> set[int]:
    """Replay step indices that follow *some* boundary.

    Position only. See the module docstring: a step in this set has a prefix
    that still contains pre-boundary history, so it must not be reported as
    reconstructed post-compaction coverage.
    """
    return {b.first_step_after for b in boundaries if b.first_step_after is not None}


def reconstruction_is_implemented() -> bool:
    """Whether post-compaction prefixes are rebuilt from the retained state.

    Exists so a caller can assert on it rather than assume. Returns False:
    `prefix_turns_through_step` still slices from index 0 (plan §15).
    """
    return False


__all__ = [
    "CompactionBoundary",
    "CompactionError",
    "HANDOFF_MARKERS",
    "boundaries_from_raw",
    "locate_in_turns",
    "post_compaction_steps",
    "reconstruction_is_implemented",
]
