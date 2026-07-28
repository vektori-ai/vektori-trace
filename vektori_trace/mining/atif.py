"""Parse Harbor's ATIF trajectory files into our Turn model.

Harbor writes trajectories as ATIF: an object with `steps[]`, not the JSON
array of turns the old code checked for. Every real trajectory therefore failed
that check and fell through to a fallback that packed the last 4,000 characters
of stdout into a single Turn — so the diagnosis LLM has been reading a stderr
tail and calling it a trajectory. Every capability label produced from a mined
run was made up from that.

There is no fallback here. A trajectory that cannot be parsed raises, because a
silently degraded trace is indistinguishable from a real one once it is written
to disk, and it poisons everything downstream while looking fine.

Three things the shape makes easy to get wrong, all of which drop the evidence
rather than the noise:

  * **`message` is a string OR a list of content parts.** Handling only the
    string case silently empties every step an agent sent as structured
    content.
  * **`observation.results[]` carry what actually happened.** The tool call
    says what the agent tried; the observation says whether it worked and why
    not. The traceback lives here. A parser that keeps `tool_calls` and drops
    observations produces trajectories where nothing ever fails, which is
    exactly backwards for diagnosing failure.
  * **Work happens in subagents.** Claude Code delegates, and the delegated
    turns live in separate files referenced by path. Not following them drops
    most of the trajectory on any run that used a subagent.

Harbor's own Pydantic models do the validation, so the schema is theirs to
change and ours to follow rather than duplicate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vektori_trace.schema import ToolCall, Turn

# Harbor is required to *produce* these files, so requiring it to read them
# costs nothing in practice — but the import failure should say so plainly.
try:  # pragma: no cover - import guard
    from harbor.models.trajectories import Trajectory

    _HARBOR_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - import guard
    Trajectory = None  # type: ignore[assignment]
    _HARBOR_IMPORT_ERROR = e


class TrajectoryParseError(RuntimeError):
    """A trajectory file could not be read as ATIF."""


# ATIF `source` -> our role vocabulary.
_ROLE = {"agent": "assistant", "user": "user", "system": "system"}

# Subagents can spawn subagents. The limit is a loop guard, not a judgement
# about how deep real work goes.
MAX_SUBAGENT_DEPTH = 5


def _require_harbor() -> None:
    if Trajectory is None:
        raise TrajectoryParseError(
            "harbor is required to parse ATIF trajectories "
            f"(install it: pip install harbor). Import failed: {_HARBOR_IMPORT_ERROR}"
        )


def _flatten_content(content: Any) -> str | None:
    """ATIF content is a string or a list of typed parts.

    Images can't go into a text trace, but dropping them entirely loses the
    fact that the agent looked at something, so they leave a marker.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content

    pieces: list[str] = []
    for part in content:
        # Pydantic model or plain dict, depending on where it came from.
        part_type = getattr(part, "type", None) or (
            part.get("type") if isinstance(part, dict) else None
        )
        if part_type == "text":
            text = getattr(part, "text", None) or (
                part.get("text") if isinstance(part, dict) else None
            )
            if text:
                pieces.append(text)
        elif part_type == "image":
            source = getattr(part, "source", None) or (
                part.get("source") if isinstance(part, dict) else None
            )
            media = getattr(source, "media_type", None) or (
                source.get("media_type") if isinstance(source, dict) else "image"
            )
            pieces.append(f"[image: {media}]")
    return "\n".join(pieces) if pieces else None


def parse_trajectory_file(
    path: Path,
    *,
    _depth: int = 0,
    _seen: set[Path] | None = None,
) -> list[Turn]:
    """Read one ATIF file (following subagents and continuations) into Turns."""
    _require_harbor()
    path = path.resolve()
    seen = _seen if _seen is not None else set()
    if path in seen:
        raise TrajectoryParseError(f"trajectory reference cycle at {path}")
    seen.add(path)

    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise TrajectoryParseError(f"could not read {path}: {e}") from e

    try:
        trajectory = Trajectory.model_validate(raw)
    except Exception as e:
        raise TrajectoryParseError(f"{path} is not valid ATIF: {e}") from e

    turns = _steps_to_turns(trajectory, base_dir=path.parent, depth=_depth, seen=seen)

    # A long run is split across files; stopping at the first truncates it.
    if trajectory.continued_trajectory_ref:
        cont = (path.parent / trajectory.continued_trajectory_ref).resolve()
        if cont.exists():
            turns += parse_trajectory_file(cont, _depth=_depth, _seen=seen)
        else:
            turns.append(
                _marker(len(turns), f"[continuation unavailable: {trajectory.continued_trajectory_ref}]")
            )

    return _reindex(turns)


def _marker(index: int, text: str, *, subagent_depth: int = 0) -> Turn:
    """A visible placeholder for work we know happened but cannot read.

    Silently omitting it would make the trajectory look complete and shorter
    than it was, which is a worse lie than admitting the gap.
    """
    return Turn(index=index, role="system", content=text, subagent_depth=subagent_depth)


def _steps_to_turns(
    trajectory: Any, *, base_dir: Path, depth: int, seen: set[Path]
) -> list[Turn]:
    turns: list[Turn] = []
    for step in trajectory.steps:
        role = _ROLE.get(step.source, step.source)
        content = _flatten_content(step.message)
        tool_calls = [
            ToolCall(id=tc.tool_call_id, name=tc.function_name, args=tc.arguments)
            for tc in (step.tool_calls or [])
        ]

        # Keep the step even when it carries only an observation: an empty
        # assistant turn followed by a traceback is still the shape of the
        # failure.
        if content is not None or step.reasoning_content or tool_calls:
            turns.append(
                Turn(
                    index=len(turns),
                    role=role,
                    thinking=step.reasoning_content,
                    content=content,
                    tool_calls=tool_calls,
                    subagent_depth=depth,
                )
            )

        observation = step.observation
        if observation is None:
            continue

        for result in observation.results or []:
            result_content = _flatten_content(result.content)
            if result_content is not None or result.source_call_id:
                turns.append(
                    Turn(
                        index=len(turns),
                        role="tool",
                        content=result_content,
                        tool_call_id=result.source_call_id,
                        subagent_depth=depth,
                    )
                )
            turns += _subagent_turns(result, base_dir=base_dir, depth=depth, seen=seen)

    return turns


def _subagent_turns(
    result: Any, *, base_dir: Path, depth: int, seen: set[Path]
) -> list[Turn]:
    refs = getattr(result, "subagent_trajectory_ref", None) or []
    if not refs:
        return []
    if depth >= MAX_SUBAGENT_DEPTH:
        return [
            _marker(
                0,
                f"[subagent trajectory omitted: depth > {MAX_SUBAGENT_DEPTH}]",
                subagent_depth=depth,
            )
        ]

    turns: list[Turn] = []
    for ref in refs:
        ref_path = getattr(ref, "trajectory_path", None)
        if not ref_path:
            turns.append(
                _marker(0, "[subagent trajectory referenced without a path]", subagent_depth=depth)
            )
            continue
        resolved = (base_dir / ref_path).resolve()
        if not resolved.exists():
            turns.append(
                _marker(0, f"[subagent trajectory missing: {ref_path}]", subagent_depth=depth)
            )
            continue
        if resolved in seen:
            turns.append(
                _marker(
                    0,
                    f"[subagent trajectory already included: {ref_path}]",
                    subagent_depth=depth,
                )
            )
            continue
        # Marker stays at parent depth (context only); the followed turns carry
        # depth+1 so dataset.py can drop them from the SFT loss.
        turns.append(_marker(0, f"[subagent trajectory: {ref_path}]", subagent_depth=depth))
        turns += parse_trajectory_file(resolved, _depth=depth + 1, _seen=seen)
    return turns


def _reindex(turns: list[Turn]) -> list[Turn]:
    for i, t in enumerate(turns):
        t.index = i
    return turns


def find_trajectory(job_dir: Path) -> Path | None:
    """The agent's own trajectory under a harbor job dir.

    Harbor writes `<job>/<timestamp>/<task>__<trial>/agent/trajectory.json`,
    with subagents alongside as `trajectory.<type>.json`. Only the top-level
    file is returned — the subagent files are reached by following refs, so
    picking one up directly here would duplicate its turns.
    """
    candidates = [p for p in job_dir.rglob("trajectory.json") if p.parent.name == "agent"]
    if not candidates:
        candidates = list(job_dir.rglob("trajectory.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_job_trajectory(job_dir: Path) -> list[Turn]:
    """Find and parse the trajectory for a harbor job. Raises if absent."""
    path = find_trajectory(job_dir)
    if path is None:
        raise TrajectoryParseError(f"no trajectory.json under {job_dir}")
    return parse_trajectory_file(path)
