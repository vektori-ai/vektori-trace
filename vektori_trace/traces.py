"""Load teacher trajectories from disk.

Deliberately outside `cli/`: the OPD training loop needs this on a GPU
container, and importing anything under `vektori_trace.cli` executes that
package's `__init__`, which pulls in every command module — including `llm.py`
and its `openai` dependency. A training image has no reason to carry an API
client, and a `ModuleNotFoundError` after the image pull is an expensive way to
discover that.

Imports here stay limited to `mining.atif` and `schema`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_teacher_trajectories(source: Path) -> list[tuple[str, list[Any]]]:
    """Teacher trajectories from harbor job dirs or ATIF JSON traces.

    Two shapes, because two things produce them: `replay` leaves harbor job
    directories, and mined/example traces are JSON. Anything unreadable is
    reported rather than skipped — a silently smaller corpus changes what the
    run measured.
    """
    from .mining.atif import TrajectoryParseError, parse_job_trajectory
    from .schema import Trace

    trajectories: list[tuple[str, list[Any]]] = []
    source = Path(source)
    if not source.is_dir():
        raise FileNotFoundError(f"teacher trace source not found: {source}")

    for path in sorted(source.iterdir()):
        if path.is_dir():
            try:
                trajectories.append((path.name, parse_job_trajectory(path)))
            except TrajectoryParseError as e:
                raise ValueError(f"{path}: not a readable harbor job dir ({e})") from e
        elif path.suffix == ".json":
            try:
                trace = Trace.load(path)
            except Exception as e:
                # Match the job-dir branch above: name the file that failed.
                # Trace.load raises schema/decode errors the caller's
                # FileNotFoundError/ValueError handler does not catch, so
                # without this an unreadable trace escapes as a bare traceback.
                raise ValueError(f"{path}: not a readable ATIF trace ({e})") from e
            trajectories.append((trace.task or path.stem, trace.turns))
    if not trajectories:
        raise ValueError(
            f"no trajectories under {source} — expected harbor job dirs or ATIF .json traces"
        )
    return trajectories


__all__ = ["load_teacher_trajectories"]
