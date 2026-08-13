"""Load raw ATIF trajectories for the dashboard (no Harbor dependency)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_JSON_WARN_RE = re.compile(
    r"Extra text detected before JSON|No valid JSON object found",
    re.IGNORECASE,
)
_THINK_RE = re.compile(r"<think>", re.IGNORECASE)


@dataclass
class StepView:
    step_id: int
    timestamp: str | None
    timestamp_dt: datetime | None
    source: str
    model_name: str | None
    message: str
    tool_calls: list[dict[str, Any]]
    observation_text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    has_think: bool
    has_json_warn: bool
    first_command_preview: str
    delta_sec: float | None = None
    near_summarization: bool = False


@dataclass
class LoadedTrajectory:
    path: Path
    session_id: str | None
    agent_name: str | None
    model_name: str | None
    steps: list[StepView]
    final_metrics: dict[str, Any]
    # Wall-clock bounds (lock/trial.log/summarization artifacts). This is the
    # real run length — trajectory.json often stops dumping steps earlier while
    # summarization / retries keep going until timeout.
    started_at: datetime | None
    ended_at: datetime | None
    duration_sec: float | None
    steps_started_at: datetime | None = None
    steps_ended_at: datetime | None = None
    steps_duration_sec: float | None = None
    summarization_indices: list[int] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _parse_ts(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    pieces.append(str(part["text"]))
                elif part.get("type") == "image":
                    pieces.append("[image]")
                else:
                    pieces.append(json.dumps(part, ensure_ascii=False))
            else:
                pieces.append(str(part))
        return "\n".join(pieces)
    return str(content)


def _observation_text(step: dict[str, Any]) -> str:
    obs = step.get("observation")
    if not obs:
        return ""
    results = obs.get("results") if isinstance(obs, dict) else None
    if not results:
        return ""
    chunks: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            chunks.append(str(result))
            continue
        call_id = result.get("source_call_id")
        body = _flatten_content(result.get("content"))
        if call_id:
            chunks.append(f"[{call_id}]\n{body}")
        else:
            chunks.append(body)
    return "\n\n".join(chunks)


def _first_command_preview(tool_calls: list[dict[str, Any]]) -> str:
    for tc in tool_calls:
        args = tc.get("arguments") or {}
        if isinstance(args, dict):
            keys = args.get("keystrokes")
            if keys is not None:
                preview = str(keys).replace("\n", "\\n")
                return preview if len(preview) <= 80 else preview[:77] + "..."
        name = tc.get("function_name") or tc.get("name") or "tool"
        return str(name)
    return ""


def list_summarization_indices(agent_dir: Path) -> list[int]:
    indices: set[int] = set()
    for path in agent_dir.glob("trajectory.summarization-*-summary.json"):
        # trajectory.summarization-1-summary.json
        m = re.search(r"summarization-(\d+)-summary", path.name)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def load_trajectory(path: Path) -> LoadedTrajectory:
    path = path.resolve()
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not an ATIF object")

    agent_block = raw.get("agent") or {}
    steps_raw = raw.get("steps") or []
    if not isinstance(steps_raw, list):
        raise ValueError(f"{path} has non-list steps")

    steps: list[StepView] = []
    prev_dt: datetime | None = None
    for step in steps_raw:
        if not isinstance(step, dict):
            continue
        message = _flatten_content(step.get("message"))
        # Some ATIF dumps put thinking in reasoning_content instead of message.
        reasoning = step.get("reasoning_content")
        if reasoning and isinstance(reasoning, str) and reasoning not in message:
            message = f"<think>\n{reasoning}\n</think>\n\n{message}" if message else reasoning

        tool_calls = step.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            tool_calls = []
        tool_calls = [tc for tc in tool_calls if isinstance(tc, dict)]

        obs_text = _observation_text(step)
        metrics = step.get("metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {}

        ts = step.get("timestamp")
        ts_dt = _parse_ts(ts)
        delta = None
        if ts_dt is not None and prev_dt is not None:
            delta = (ts_dt - prev_dt).total_seconds()
        if ts_dt is not None:
            prev_dt = ts_dt

        steps.append(
            StepView(
                step_id=int(step.get("step_id") or len(steps) + 1),
                timestamp=str(ts) if ts else None,
                timestamp_dt=ts_dt,
                source=str(step.get("source") or "unknown"),
                model_name=step.get("model_name"),
                message=message,
                tool_calls=tool_calls,
                observation_text=obs_text,
                prompt_tokens=_as_int(metrics.get("prompt_tokens")),
                completion_tokens=_as_int(metrics.get("completion_tokens")),
                has_think=bool(_THINK_RE.search(message)),
                has_json_warn=bool(_JSON_WARN_RE.search(obs_text)),
                first_command_preview=_first_command_preview(tool_calls),
                delta_sec=delta,
            )
        )

    final_metrics = raw.get("final_metrics") or {}
    if not isinstance(final_metrics, dict):
        final_metrics = {}

    steps_started = steps[0].timestamp_dt if steps else None
    steps_ended = steps[-1].timestamp_dt if steps else None
    steps_duration = None
    if steps_started and steps_ended:
        steps_duration = (steps_ended - steps_started).total_seconds()

    trial_dir = path.parent.parent if path.parent.name == "agent" else path.parent
    wall_start, wall_end = wall_clock_bounds(
        trial_dir,
        steps_started=steps_started,
        steps_ended=steps_ended,
    )
    wall_duration = None
    if wall_start and wall_end:
        wall_duration = (wall_end - wall_start).total_seconds()

    summ = list_summarization_indices(path.parent)
    return LoadedTrajectory(
        path=path,
        session_id=raw.get("session_id"),
        agent_name=agent_block.get("name"),
        model_name=agent_block.get("model_name"),
        steps=steps,
        final_metrics=final_metrics,
        started_at=wall_start,
        ended_at=wall_end,
        duration_sec=wall_duration,
        steps_started_at=steps_started,
        steps_ended_at=steps_ended,
        steps_duration_sec=steps_duration,
        summarization_indices=summ,
        raw=raw,
    )


def _mtime_dt(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return None


def _last_timestamp_in_atif(path: Path) -> datetime | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list):
        return None
    last: datetime | None = None
    for step in steps:
        if not isinstance(step, dict):
            continue
        ts = _parse_ts(step.get("timestamp"))
        if ts is not None and (last is None or ts > last):
            last = ts
    return last


def wall_clock_bounds(
    trial_dir: Path,
    *,
    steps_started: datetime | None = None,
    steps_ended: datetime | None = None,
) -> tuple[datetime | None, datetime | None]:
    """Real run window from artifacts, not just dumped agent steps.

    Start: earliest of lock/config mtime, result.started_at, first step.
    End: latest of trial.log mtime, trajectory/summarization timestamps,
    result.finished_at / exception time, last step.
    """
    candidates_start: list[datetime] = []
    candidates_end: list[datetime] = []

    if steps_started is not None:
        candidates_start.append(_as_utc(steps_started))  # type: ignore[arg-type]
    if steps_ended is not None:
        candidates_end.append(_as_utc(steps_ended))  # type: ignore[arg-type]

    for name in ("lock.json", "config.json"):
        mt = _mtime_dt(trial_dir / name)
        if mt is not None:
            candidates_start.append(mt)

    trial_log = trial_dir / "trial.log"
    mt = _mtime_dt(trial_log)
    if mt is not None:
        candidates_end.append(mt)

    traj = trial_dir / "agent" / "trajectory.json"
    mt = _mtime_dt(traj)
    if mt is not None:
        candidates_end.append(mt)

    agent_dir = trial_dir / "agent"
    if agent_dir.is_dir():
        for path in agent_dir.glob("trajectory.summarization-*.json"):
            mt = _mtime_dt(path)
            if mt is not None:
                candidates_end.append(mt)
            last_ts = _last_timestamp_in_atif(path)
            if last_ts is not None:
                candidates_end.append(last_ts)

    for result_path in (trial_dir / "result.json", trial_dir.parent / "result.json"):
        try:
            data = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        started = _parse_ts(data.get("started_at"))
        finished = _parse_ts(data.get("finished_at"))
        if started is not None:
            candidates_start.append(started)
        if finished is not None:
            candidates_end.append(finished)
        info = data.get("exception_info")
        if isinstance(info, dict):
            occurred = _parse_ts(info.get("occurred_at"))
            if occurred is not None:
                candidates_end.append(occurred)

    exc = trial_dir / "exception.txt"
    mt = _mtime_dt(exc)
    if mt is not None:
        candidates_end.append(mt)

    start = min(candidates_start) if candidates_start else None
    end = max(candidates_end) if candidates_end else None
    return start, end


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_summarization_episode(agent_dir: Path, index: int) -> dict[str, Path]:
    """Return paths for summary/questions/answers for one summarization episode."""
    base = f"trajectory.summarization-{index}"
    out: dict[str, Path] = {}
    for kind in ("summary", "questions", "answers"):
        path = agent_dir / f"{base}-{kind}.json"
        if path.is_file():
            out[kind] = path
    return out


def summarization_agent_message(path: Path) -> str:
    """Best-effort: last agent message from a summarization ATIF file, full text."""
    data = json.loads(path.read_text())
    steps = data.get("steps") or []
    if not isinstance(steps, list):
        return ""
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        if step.get("source") == "agent":
            return _flatten_content(step.get("message"))
    # Fall back to concatenating all non-copied agent-ish content.
    chunks: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("is_copied_context"):
            continue
        msg = _flatten_content(step.get("message"))
        if msg:
            chunks.append(msg)
    return "\n\n".join(chunks)
