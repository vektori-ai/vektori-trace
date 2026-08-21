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

The retained state is *reachable*, but it is **not** just the inlined user
handoff message this detector records as `handoff_turn_index`. That message is
only the *tail* — the previous agent's answers. The head (the "you are picking
up work…" summary and the new agent's questions) lives in
`trajectory.summarization-N-questions.json`.

`scripts/sft_export_traces.py` already solved this for the SFT corpus:
`handoff_head()` loads the questions sidecar and `split_on_compaction()` starts
each new segment from it, discarding pre-boundary turns. The SFT repair plan
records that collapsing the head into the single main-trajectory user message
was considered and rejected — it leaves "Here are the answers…" dangling
against questions absent from the context.

That matters beyond provenance: ck75's own SFT was done on that reconstructed
geometry, so feeding it a flat pre-boundary prefix is off-distribution with
respect to its training. Reconstruction should therefore reuse the SFT
definition rather than invent a second one. That change belongs to the prefix
builder and is deliberately not made here.
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


def first_boundary_step(boundaries: list[CompactionBoundary]) -> int | None:
    """The earliest replay step affected by any boundary, or None.

    Every step at or after this one sits inside a compacted segment, so its
    prefix is built from context Harbor replaced. `post_compaction_steps`
    deliberately marks only `first_step_after` per boundary — a *position*, for
    reporting — which means later steps in the same segment are equally
    misreconstructed and carry no mark. Anything that needs to *exclude*
    affected states must use this, not the marks.
    """
    steps = [b.first_step_after for b in boundaries if b.first_step_after is not None]
    return min(steps) if steps else None


#: The system turn terminus-2 writes at every context reset. `atif` keeps it as
#: a depth-0 system turn, so it survives parsing as a split point. Kept
#: byte-identical to `sft_export_traces.BOUNDARY_MESSAGE`: the two paths must
#: split at the same places or the SFT corpus and the replay corpus describe
#: different conversations.
BOUNDARY_MESSAGE = "Performed context summarization and handoff to continue task."


def handoff_head(jobs_dir: Path, boundary_ordinal: int) -> list[Any]:
    """The opening of the conversation that follows compaction number N.

    Ported verbatim from `sft_export_traces.handoff_head`, which established
    the contract v1/ck75 was actually trained on. Its reasoning, unchanged:

        The main trajectory only holds the *tail* of a handoff — the step
        carrying the previous agent's answers. The head (the "you are picking
        up work…" message with the summary in it, and the new agent's
        questions) lives in `trajectory.summarization-N-questions.json`, which
        is why reading only `agent/trajectory.json` produces a conversation
        with no beginning.

    `SFT-REPAIR-PLAN.md` records that collapsing this head into the single
    main-trajectory user message was considered and rejected: it leaves "Here
    are the answers…" dangling against questions absent from the context.
    """
    from .mining.atif import find_trajectory, parse_trajectory_file

    main = find_trajectory(Path(jobs_dir))
    if main is None:
        return []
    path = main.parent / f"trajectory.summarization-{boundary_ordinal}-questions.json"
    if not path.exists():
        return []
    # Depth 0: this is the parent conversation resuming, not subagent chatter.
    return [t for t in parse_trajectory_file(path) if t.subagent_depth == 0]


def split_on_compaction(turns: list[Any], jobs_dir: Path) -> list[list[Any]]:
    """One list of turns per linear conversation the model actually saw.

    Ported from `sft_export_traces.split_on_compaction`. Each boundary ends the
    current segment and starts the next one from the questions-sidecar head, so
    a segment contains the retained state and what followed it — never the
    history `boundary: "replace"` discarded.

    Splits on `BOUNDARY_MESSAGE` rather than on `extra.context_management`
    because it operates on parsed `Turn`s, which do not carry `extra`. The
    structured predicate remains the detector of record
    (`boundaries_from_raw`); `assert_split_agrees_with_raw` checks the two
    agree on a given trace, since a disagreement would mean the corpus this
    run trains on differs from the one it reports.
    """
    segments: list[list[Any]] = []
    current: list[Any] = []
    boundary_ordinal = 0
    for turn in turns:
        if turn.subagent_depth > 0:
            continue  # summarization subagent transcript; reached via handoff_head
        content = turn.content or ""
        if turn.role == "system" and content.startswith("[subagent"):
            continue
        if turn.role == "system" and content.strip() == BOUNDARY_MESSAGE:
            segments.append(current)
            boundary_ordinal += 1
            current = list(handoff_head(jobs_dir, boundary_ordinal))
            continue
        current.append(turn)
    segments.append(current)
    return [s for s in segments if s]


def current_segment(turns: list[Any], jobs_dir: Path) -> list[Any]:
    """The latest reconstructed segment — the conversation now in context.

    After several boundaries only the most recent segment is live; earlier ones
    were replaced. The audited corpus has traces with up to four boundaries, so
    "latest" is the common case rather than an edge one.
    """
    segments = split_on_compaction(turns, jobs_dir)
    return segments[-1] if segments else []


def assert_split_agrees_with_raw(turns: list[Any], jobs_dir: Path) -> dict[str, Any]:
    """The prose split and the structured detector must find the same boundaries.

    `split_on_compaction` keys on `BOUNDARY_MESSAGE`; `boundaries_from_raw`
    keys on `extra.context_management`. They should agree exactly. A
    disagreement is a finding, not a nuisance: it means one of the two views of
    the corpus is wrong, and both are load-bearing — the split builds what the
    model sees, the detector builds what the report claims.
    """
    from .mining.atif import find_trajectory

    main = find_trajectory(Path(jobs_dir))
    if main is None:
        raise CompactionError(f"{jobs_dir}: no trajectory to reconcile against")
    raw = boundaries_from_raw(main)
    n_prose = sum(
        1
        for t in turns
        if getattr(t, "subagent_depth", 0) == 0
        and getattr(t, "role", None) == "system"
        and (getattr(t, "content", "") or "").strip() == BOUNDARY_MESSAGE
    )
    if n_prose != len(raw):
        raise CompactionError(
            f"{jobs_dir}: the message-based split finds {n_prose} boundaries but "
            f"extra.context_management finds {len(raw)}. The reconstructed "
            "conversation and the reported boundary inventory disagree."
        )
    return {"n_boundaries": len(raw), "agrees": True}


def reconstruction_is_implemented() -> bool:
    """Whether post-compaction prefixes are rebuilt from the retained state.

    Exists so a caller can assert on it rather than assume.

    True since the port of `sft_export_traces`' definition:
    `candidates_from_traces` enumerates compacted traces from
    `current_segment`, so a post-compaction candidate's prefix is the retained
    handoff state plus what followed it — not the history Harbor replaced.

    Consequences for callers, both deliberate: `select_replay_prefixes` stops
    excluding compacted candidates, and `require_post_compaction > 0` becomes
    satisfiable. Anything that reverts the enumeration must revert this too.
    """
    return True


__all__ = [
    "BOUNDARY_MESSAGE",
    "CompactionBoundary",
    "assert_split_agrees_with_raw",
    "current_segment",
    "handoff_head",
    "split_on_compaction",
    "CompactionError",
    "HANDOFF_MARKERS",
    "boundaries_from_raw",
    "first_boundary_step",
    "locate_in_turns",
    "post_compaction_steps",
    "reconstruction_is_implemented",
]
