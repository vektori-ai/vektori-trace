"""Walk passk_jobs trees and produce TrialRef entries for the dashboard."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .status import TrialStatus, classify_status, exception_summary

_ATTEMPT_RE = re.compile(r"^(?P<task>.+)-(?P<attempt>\d+)$")
_DEFAULT_MAIN_SUFFIX = "sg2g67v"


@dataclass(frozen=True)
class TrialRef:
    trial_id: str
    trial_dir: Path
    task: str
    attempt: int | None
    status: TrialStatus
    has_trajectory: bool
    trajectory_path: Path | None
    agent: str | None
    model: str | None
    started_at: str | None
    exception_one_liner: str | None
    n_steps: int | None
    is_preferred_default: bool

    @property
    def label(self) -> str:
        attempt = f"-{self.attempt}" if self.attempt is not None else ""
        return f"{self.trial_id} [{self.status}]{attempt}"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_attempt(path: Path, baseline_root: Path) -> tuple[str, int | None]:
    """Extract task stem and attempt index from .../passk_jobs/stage1/<task>-N/..."""
    try:
        rel = path.resolve().relative_to(baseline_root.resolve())
    except ValueError:
        return path.name, None
    parts = rel.parts
    # passk_jobs / stage1 / <task>-N / ...
    if len(parts) >= 3 and parts[0] == "passk_jobs":
        m = _ATTEMPT_RE.match(parts[2])
        if m:
            return m.group("task"), int(m.group("attempt"))
    return path.name.rsplit("__", 1)[0] if "__" in path.name else path.name, None


def _traj_meta(traj_path: Path) -> tuple[str | None, str | None, str | None, int | None]:
    data = _read_json(traj_path)
    if not data:
        return None, None, None, None
    agent_block = data.get("agent") or {}
    agent = agent_block.get("name")
    model = agent_block.get("model_name")
    steps = data.get("steps") or []
    n_steps = len(steps) if isinstance(steps, list) else None
    started = None
    if isinstance(steps, list) and steps:
        first = steps[0]
        if isinstance(first, dict):
            started = first.get("timestamp")
    return agent, model, started, n_steps


def _config_meta(trial_dir: Path) -> tuple[str | None, str | None, str | None]:
    cfg = _read_json(trial_dir / "config.json")
    if not cfg:
        return None, None, None
    agent_block = cfg.get("agent") or {}
    return (
        cfg.get("trial_name"),
        agent_block.get("name"),
        agent_block.get("model_name"),
    )


def _started_from_result(trial_dir: Path) -> str | None:
    for candidate in (trial_dir / "result.json", trial_dir.parent / "result.json"):
        data = _read_json(candidate)
        if not data:
            continue
        started = data.get("started_at")
        if started:
            return str(started)
    return None


def _trial_dirs_from_passk(passk_root: Path) -> set[Path]:
    """Collect trial directories that look like harbor trial leaves."""
    found: set[Path] = set()
    if not passk_root.is_dir():
        return found

    for traj in passk_root.rglob("trajectory.json"):
        if traj.parent.name != "agent":
            continue
        found.add(traj.parent.parent.resolve())

    for exc in passk_root.rglob("exception.txt"):
        # .../<trial_id>/exception.txt
        found.add(exc.parent.resolve())

    for result in passk_root.rglob("result.json"):
        data = _read_json(result)
        if not data:
            continue
        # Trial-level results have trial_name; job-level have n_total_trials.
        if "trial_name" in data or "exception_info" in data:
            found.add(result.parent.resolve())

    return found


def discover_trials(
    baseline_root: Path,
    *,
    preferred_suffix: str = _DEFAULT_MAIN_SUFFIX,
) -> list[TrialRef]:
    """Scan `<baseline_root>/passk_jobs` for trials."""
    baseline_root = baseline_root.resolve()
    passk_root = baseline_root / "passk_jobs"
    refs: list[TrialRef] = []

    for trial_dir in sorted(_trial_dirs_from_passk(passk_root), key=lambda p: str(p)):
        traj_path = trial_dir / "agent" / "trajectory.json"
        has_traj = traj_path.is_file()
        status = classify_status(trial_dir, has_trajectory=has_traj)

        cfg_name, cfg_agent, cfg_model = _config_meta(trial_dir)
        agent, model, started, n_steps = (None, None, None, None)
        if has_traj:
            agent, model, started, n_steps = _traj_meta(traj_path)

        trial_id = cfg_name or trial_dir.name
        task, attempt = _parse_attempt(trial_dir, baseline_root)
        # Prefer task stem from trial_id when path parse fell back oddly.
        if "__" in trial_id:
            task = trial_id.rsplit("__", 1)[0]

        refs.append(
            TrialRef(
                trial_id=trial_id,
                trial_dir=trial_dir,
                task=task,
                attempt=attempt,
                status=status,
                has_trajectory=has_traj,
                trajectory_path=traj_path if has_traj else None,
                agent=agent or cfg_agent,
                model=model or cfg_model,
                started_at=started or _started_from_result(trial_dir),
                exception_one_liner=exception_summary(trial_dir),
                n_steps=n_steps,
                is_preferred_default=trial_id.endswith(preferred_suffix)
                or preferred_suffix in trial_id,
            )
        )

    # Prefer incomplete/trajectory first, then by started_at descending.
    def sort_key(r: TrialRef) -> tuple:
        status_rank = {
            TrialStatus.INCOMPLETE: 0,
            TrialStatus.TRAJECTORY: 1,
            TrialStatus.INFRA_ERROR: 2,
            TrialStatus.NO_TRAJ: 3,
        }[r.status]
        return (0 if r.is_preferred_default else 1, status_rank, r.started_at or "", r.trial_id)

    refs.sort(key=sort_key)
    return refs


def default_trial_index(refs: list[TrialRef]) -> int:
    for i, r in enumerate(refs):
        if r.is_preferred_default:
            return i
    for i, r in enumerate(refs):
        if r.status == TrialStatus.INCOMPLETE and r.has_trajectory:
            return i
    for i, r in enumerate(refs):
        if r.has_trajectory:
            return i
    return 0
