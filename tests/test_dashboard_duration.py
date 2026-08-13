"""Dashboard trajectory wall-clock vs step-span duration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from vektori_trace.dashboard.load_atif import load_trajectory, wall_clock_bounds


def test_wall_clock_longer_than_steps_for_cutoff_trial() -> None:
    """sg2g67v: trajectory steps stop ~18m; trial.log/summarization run to ~29.5m."""
    traj_path = Path(
        "qwen-run/vektori-out/baseline/passk_jobs/stage1/"
        "prefecthq__prefect-65ea05bef8d9-0/"
        "prefecthq__prefect-65ea05bef8d9-terminus-2/2026-08-02__16-21-07/"
        "prefecthq__prefect-65ea05bef8d9__sg2g67v/agent/trajectory.json"
    )
    if not traj_path.is_file():
        return  # artifact not present in this checkout

    loaded = load_trajectory(traj_path)
    assert loaded.steps_duration_sec is not None
    assert loaded.duration_sec is not None
    assert abs(loaded.steps_duration_sec - 1107.3) < 1.0
    assert abs(loaded.duration_sec - 1772.0) < 2.0
    assert loaded.duration_sec - loaded.steps_duration_sec > 600
    assert hasattr(loaded, "steps_duration_sec")
    assert hasattr(loaded, "steps_started_at")
    assert hasattr(loaded, "steps_ended_at")


def test_wall_clock_bounds_uses_trial_log_mtime(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    agent = trial / "agent"
    agent.mkdir(parents=True)
    (trial / "lock.json").write_text("{}")
    # Fake early lock / late trial.log via explicit timestamps in ATIF + mtime.
    start = datetime(2026, 8, 2, 16, 21, 7, tzinfo=UTC)
    step_end = datetime(2026, 8, 2, 16, 40, 19, tzinfo=UTC)
    traj = {
        "schema_version": "ATIF-v1.7",
        "session_id": "t",
        "agent": {"name": "terminus-2", "model_name": "x"},
        "steps": [
            {
                "step_id": 1,
                "timestamp": start.isoformat(),
                "source": "user",
                "message": "hi",
            },
            {
                "step_id": 2,
                "timestamp": step_end.isoformat(),
                "source": "agent",
                "message": "bye",
                "metrics": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        ],
        "final_metrics": {},
    }
    (agent / "trajectory.json").write_text(json.dumps(traj))
    log = trial / "trial.log"
    log.write_text("done\n")
    # Set trial.log mtime later than last step; keep traj mtime at step_end
    # so it doesn't dominate the end bound with "now".
    later = datetime(2026, 8, 2, 16, 50, 39, tzinfo=UTC).timestamp()
    import os

    os.utime(agent / "trajectory.json", (step_end.timestamp(), step_end.timestamp()))
    os.utime(log, (later, later))
    os.utime(trial / "lock.json", (start.timestamp(), start.timestamp()))

    wall_start, wall_end = wall_clock_bounds(
        trial, steps_started=start, steps_ended=step_end
    )
    assert wall_start == start
    assert wall_end is not None
    assert abs(wall_end.timestamp() - later) < 1.0

    loaded = load_trajectory(agent / "trajectory.json")
    assert loaded.steps_duration_sec == (step_end - start).total_seconds()
    assert loaded.duration_sec is not None
    assert loaded.duration_sec > loaded.steps_duration_sec
