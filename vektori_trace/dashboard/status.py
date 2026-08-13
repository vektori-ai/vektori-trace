"""Classify a discovered trial for the dashboard list."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any


class TrialStatus(StrEnum):
    TRAJECTORY = "trajectory"
    INCOMPLETE = "incomplete"
    INFRA_ERROR = "infra_error"
    NO_TRAJ = "no_traj"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _exception_message(trial_dir: Path) -> str | None:
    exc_path = trial_dir / "exception.txt"
    if exc_path.is_file():
        try:
            text = exc_path.read_text().strip()
        except OSError:
            return "exception.txt unreadable"
        return text.splitlines()[0] if text else "exception.txt (empty)"

    trial_result = _read_json(trial_dir / "result.json")
    if trial_result:
        info = trial_result.get("exception_info")
        if isinstance(info, dict):
            msg = info.get("exception_message") or info.get("exception_type")
            if msg:
                return str(msg)
    return None


def _job_unfinished(trial_dir: Path) -> bool:
    """True when the harbor job result still looks mid-flight."""
    job_result = _read_json(trial_dir.parent / "result.json")
    if not job_result:
        return False
    if job_result.get("finished_at") is None:
        stats = job_result.get("stats") or {}
        if int(stats.get("n_running_trials") or 0) > 0:
            return True
        # Unfinished with a trajectory present is still incomplete (cutoff).
        return True
    return False


def classify_status(trial_dir: Path, *, has_trajectory: bool) -> TrialStatus:
    exc = _exception_message(trial_dir)
    if exc and not has_trajectory:
        return TrialStatus.INFRA_ERROR
    if has_trajectory and _job_unfinished(trial_dir):
        return TrialStatus.INCOMPLETE
    if has_trajectory:
        # Infra error that still produced a traj (rare) — prefer showing traj.
        return TrialStatus.TRAJECTORY
    if exc:
        return TrialStatus.INFRA_ERROR
    return TrialStatus.NO_TRAJ


def exception_summary(trial_dir: Path) -> str | None:
    return _exception_message(trial_dir)
