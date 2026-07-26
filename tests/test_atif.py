"""Parsing Harbor's ATIF trajectories.

The bug this replaces was silent: Harbor writes `{"steps": [...]}` and the old
code checked for a JSON array, so every real trajectory fell through to a
fallback that packed the tail of stdout into one Turn. The traces looked
ordinary on disk. Every capability label derived from a mined run was made up
from a stderr tail.

So these tests care most about the evidence surviving — observations, content
parts, subagent work — and about the parser refusing rather than degrading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vektori_trace.mining.atif import (
    TrajectoryParseError,
    find_trajectory,
    parse_job_trajectory,
    parse_trajectory_file,
)

AGENT = {"name": "claude_code", "version": "1.0.0"}


def _traj(steps: list[dict], **extra) -> dict:
    return {"schema_version": "ATIF-v1.7", "agent": AGENT, "steps": steps, **extra}


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


# ---------------------------------------------------------------------------
# The shape itself
# ---------------------------------------------------------------------------


def test_steps_become_turns_with_roles_mapped(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "trajectory.json",
        _traj(
            [
                {"step_id": 1, "source": "user", "message": "Fix the failing test"},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "Looking now.",
                    "reasoning_content": "I should read the traceback first",
                },
            ]
        ),
    )

    turns = parse_trajectory_file(path)

    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[0].content == "Fix the failing test"
    assert turns[1].thinking == "I should read the traceback first"
    assert [t.index for t in turns] == [0, 1]


def test_message_as_content_parts_is_not_dropped(tmp_path: Path) -> None:
    """`message` is a string OR a list of parts. Handling only the string case
    silently empties every structured step."""
    path = _write(
        tmp_path / "trajectory.json",
        _traj(
            [
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": [
                        {"type": "text", "text": "First I'll check the log."},
                        {"type": "image", "source": {"media_type": "image/png", "path": "s.png"}},
                        {"type": "text", "text": "Then patch it."},
                    ],
                }
            ]
        ),
    )

    turns = parse_trajectory_file(path)

    assert "First I'll check the log." in turns[0].content
    assert "Then patch it." in turns[0].content
    # The image can't go in a text trace, but the fact one was looked at should.
    assert "[image: image/png]" in turns[0].content


def test_tool_calls_are_preserved(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "trajectory.json",
        _traj(
            [
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": "Running the tests.",
                    "tool_calls": [
                        {
                            "tool_call_id": "c1",
                            "function_name": "bash",
                            "arguments": {"cmd": "pytest -q"},
                        }
                    ],
                }
            ]
        ),
    )

    turns = parse_trajectory_file(path)

    assert len(turns[0].tool_calls) == 1
    assert turns[0].tool_calls[0].name == "bash"
    assert turns[0].tool_calls[0].args == {"cmd": "pytest -q"}


def test_observations_become_their_own_turns(tmp_path: Path) -> None:
    """The tool call says what the agent tried; the observation says whether it
    worked. The traceback is the evidence — a parser that keeps calls and drops
    observations produces trajectories where nothing ever fails."""
    traceback = "Traceback (most recent call last):\n  AssertionError: 1 != 2"
    path = _write(
        tmp_path / "trajectory.json",
        _traj(
            [
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": "Running the tests.",
                    "tool_calls": [
                        {"tool_call_id": "c1", "function_name": "bash", "arguments": {}}
                    ],
                    "observation": {
                        "results": [{"source_call_id": "c1", "content": traceback}]
                    },
                }
            ]
        ),
    )

    turns = parse_trajectory_file(path)

    assert len(turns) == 2
    assert turns[1].role == "tool"
    assert turns[1].tool_call_id == "c1"
    assert "AssertionError" in turns[1].content


def test_observation_content_parts_are_flattened(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "trajectory.json",
        _traj(
            [
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": "go",
                    "tool_calls": [
                        {"tool_call_id": "c1", "function_name": "bash", "arguments": {}}
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": "c1",
                                "content": [{"type": "text", "text": "exit code 1"}],
                            }
                        ]
                    },
                }
            ]
        ),
    )

    turns = parse_trajectory_file(path)
    assert turns[-1].content == "exit code 1"


def test_system_steps_are_kept(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "trajectory.json",
        _traj([{"step_id": 1, "source": "system", "message": "context limit reached"}]),
    )
    turns = parse_trajectory_file(path)
    assert turns[0].role == "system"


# ---------------------------------------------------------------------------
# Subagents
# ---------------------------------------------------------------------------


def _delegating(ref_name: str) -> dict:
    return _traj(
        [
            {
                "step_id": 1,
                "source": "agent",
                "message": "Delegating the search.",
                "tool_calls": [{"tool_call_id": "c1", "function_name": "task", "arguments": {}}],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "c1",
                            "content": "done",
                            "subagent_trajectory_ref": [{"trajectory_path": ref_name}],
                        }
                    ]
                },
            }
        ]
    )


def test_subagent_trajectories_are_followed(tmp_path: Path) -> None:
    """Claude Code delegates. Not following the refs drops most of the work on
    any run that used a subagent."""
    _write(
        tmp_path / "trajectory.search.json",
        _traj(
            [
                {"step_id": 1, "source": "agent", "message": "Subagent grepping the repo."},
                {"step_id": 2, "source": "agent", "message": "Found it in utils.py."},
            ]
        ),
    )
    path = _write(tmp_path / "trajectory.json", _delegating("trajectory.search.json"))

    turns = parse_trajectory_file(path)
    text = " ".join(t.content or "" for t in turns)

    assert "Subagent grepping the repo." in text
    assert "Found it in utils.py." in text
    assert [t.index for t in turns] == list(range(len(turns)))


def test_missing_subagent_file_is_marked_not_silently_dropped(tmp_path: Path) -> None:
    """A shorter trajectory that looks complete is a worse lie than an
    admitted gap."""
    path = _write(tmp_path / "trajectory.json", _delegating("trajectory.gone.json"))

    turns = parse_trajectory_file(path)

    assert any("subagent trajectory missing" in (t.content or "") for t in turns)


def test_subagent_ref_without_a_path_is_marked(tmp_path: Path) -> None:
    payload = _traj(
        [
            {
                "step_id": 1,
                "source": "agent",
                "message": "Delegating.",
                "tool_calls": [
                    {"tool_call_id": "c1", "function_name": "task", "arguments": {}}
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "c1",
                            "content": "done",
                            "subagent_trajectory_ref": [{"trajectory_id": "only-an-id"}],
                        }
                    ]
                },
            }
        ]
    )
    turns = parse_trajectory_file(_write(tmp_path / "trajectory.json", payload))
    assert any("without a path" in (t.content or "") for t in turns)


def test_reference_cycle_does_not_hang(tmp_path: Path) -> None:
    _write(tmp_path / "trajectory.b.json", _delegating("trajectory.json"))
    path = _write(tmp_path / "trajectory.json", _delegating("trajectory.b.json"))

    turns = parse_trajectory_file(path)

    assert any("already included" in (t.content or "") for t in turns)


# ---------------------------------------------------------------------------
# Continuations
# ---------------------------------------------------------------------------


def test_continuation_files_are_followed(tmp_path: Path) -> None:
    """A long run is split across files; stopping at the first truncates it."""
    _write(
        tmp_path / "trajectory.cont.json",
        _traj([{"step_id": 1, "source": "agent", "message": "second half"}]),
    )
    path = _write(
        tmp_path / "trajectory.json",
        _traj(
            [{"step_id": 1, "source": "agent", "message": "first half"}],
            continued_trajectory_ref="trajectory.cont.json",
        ),
    )

    turns = parse_trajectory_file(path)
    text = " ".join(t.content or "" for t in turns)
    assert "first half" in text and "second half" in text


def test_missing_continuation_is_marked(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "trajectory.json",
        _traj(
            [{"step_id": 1, "source": "agent", "message": "first half"}],
            continued_trajectory_ref="nope.json",
        ),
    )
    turns = parse_trajectory_file(path)
    assert any("continuation unavailable" in (t.content or "") for t in turns)


# ---------------------------------------------------------------------------
# No fallback
# ---------------------------------------------------------------------------


def test_a_json_array_is_rejected(tmp_path: Path) -> None:
    """The old code's happy path. It is not ATIF and must not parse."""
    path = _write(tmp_path / "trajectory.json", {})
    path.write_text(json.dumps([{"index": 0, "role": "assistant", "content": "hi"}]))

    with pytest.raises(TrajectoryParseError):
        parse_trajectory_file(path)


def test_malformed_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    path.write_text("{not json")
    with pytest.raises(TrajectoryParseError):
        parse_trajectory_file(path)


def test_steps_missing_required_fields_raise(tmp_path: Path) -> None:
    """Validation is Harbor's models, so a schema change surfaces here rather
    than as quietly emptier traces."""
    path = _write(tmp_path / "trajectory.json", _traj([{"source": "agent"}]))
    with pytest.raises(TrajectoryParseError):
        parse_trajectory_file(path)


def test_empty_steps_raise(tmp_path: Path) -> None:
    path = _write(tmp_path / "trajectory.json", _traj([]))
    with pytest.raises(TrajectoryParseError):
        parse_trajectory_file(path)


# ---------------------------------------------------------------------------
# Locating the file in a job dir
# ---------------------------------------------------------------------------


def test_find_trajectory_prefers_the_agent_dir(tmp_path: Path) -> None:
    """Subagent files sit beside the main one. Picking one directly would
    duplicate its turns, since refs already pull it in."""
    trial = tmp_path / "2026-01-01__00-00-00" / "task__abc"
    _write(trial / "agent" / "trajectory.json", _traj([{"step_id": 1, "source": "agent", "message": "main"}]))
    _write(trial / "other" / "trajectory.json", _traj([{"step_id": 1, "source": "agent", "message": "not main"}]))

    found = find_trajectory(tmp_path)
    assert found is not None and found.parent.name == "agent"


def test_parse_job_trajectory_raises_when_absent(tmp_path: Path) -> None:
    with pytest.raises(TrajectoryParseError):
        parse_job_trajectory(tmp_path)


def test_harbor_runner_turns_a_parse_failure_into_an_infra_failure(tmp_path: Path) -> None:
    """An unparseable trajectory means we don't know what the agent did, which
    is not the same as the agent failing — so the task leaves the dataset
    rather than becoming a loss."""
    from vektori_trace.mining.miner import HarborTraceRunner, InfraFailure

    runner = HarborTraceRunner(agent="claude_code", jobs_dir=tmp_path)
    with pytest.raises(InfraFailure):
        runner._parse_turns(tmp_path)
