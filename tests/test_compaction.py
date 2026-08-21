"""Compaction boundary detection — detection only, not reconstruction (§15)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from vektori_trace.compaction import (
    CompactionError,
    boundaries_from_raw,
    locate_in_turns,
    post_compaction_steps,
    reconstruction_is_implemented,
)


def _raw(tmp_path, n_boundaries=1, handoff=True):
    steps: list[dict[str, Any]] = [
        {"step_id": 1, "source": "user", "message": "task"},
        {"step_id": 2, "source": "agent", "message": "act"},
    ]
    for b in range(n_boundaries):
        steps.append(
            {
                "step_id": 10 + b,
                "source": "system",
                "message": "Performed context summarization and handoff to continue task.",
                "extra": {"context_management": {"type": "compaction", "boundary": "replace"}},
                "observation": {
                    "results": [
                        {"subagent_trajectory_ref": [
                            {"trajectory_path": f"trajectory.summarization-{b+1}-summary.json"}
                        ]}
                    ]
                },
            }
        )
        steps.append(
            {
                "step_id": 20 + b,
                "source": "user",
                "message": (
                    "Here are the answers the other agent provided.\nstate"
                    if handoff else "unrelated"
                ),
            }
        )
        steps.append({"step_id": 30 + b, "source": "agent", "message": "continue"})
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "trajectory.json"
    p.write_text(json.dumps({"steps": steps}))
    return p


class TestRawDetection:
    def test_keys_off_structured_field_not_prose(self, tmp_path):
        """extra.context_management is the signal; the message text is not."""
        p = tmp_path / "t.json"
        p.write_text(json.dumps({"steps": [
            {"step_id": 1, "source": "system",
             "message": "Performed context summarization and handoff to continue task."},
        ]}))
        assert boundaries_from_raw(p) == []

    def test_finds_multiple_boundaries_with_ordinals(self, tmp_path):
        """§15: compaction-1 vs compaction-2 must stay distinguishable."""
        got = boundaries_from_raw(_raw(tmp_path, n_boundaries=2))
        assert [b.index for b in got] == [0, 1]
        assert all(b.boundary_kind == "replace" for b in got)
        assert got[0].sidecars == ("trajectory.summarization-1-summary.json",)
        assert got[1].sidecars == ("trajectory.summarization-2-summary.json",)

    def test_records_whether_handoff_followed(self, tmp_path):
        with_h = boundaries_from_raw(_raw(tmp_path / "a", n_boundaries=1))
        without = boundaries_from_raw(_raw(tmp_path / "b", n_boundaries=1, handoff=False))
        assert with_h[0].meta["next_is_handoff"] is True
        assert without[0].meta["next_is_handoff"] is False

    def test_no_compaction_is_empty_not_error(self, tmp_path):
        p = tmp_path / "t.json"
        p.write_text(json.dumps({"steps": [{"step_id": 1, "source": "agent", "message": "x"}]}))
        assert boundaries_from_raw(p) == []

    def test_unreadable_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        with pytest.raises(CompactionError):
            boundaries_from_raw(p)


@dataclass
class _T:
    index: int
    role: str
    content: str = ""
    subagent_depth: int = 0
    tool_calls: list = field(default_factory=list)


class TestLocate:
    def _turns(self):
        return [
            _T(0, "user", "task"),
            _T(1, "assistant", "a1"),
            _T(2, "system", "Performed context summarization and handoff to continue task."),
            _T(3, "system", "[subagent trajectory: x.json]"),
            _T(4, "user", "Here are the answers the other agent provided."),
            _T(5, "assistant", "a2"),
        ]

    def test_maps_marker_handoff_and_step(self, tmp_path):
        raw = boundaries_from_raw(_raw(tmp_path))
        turns = self._turns()
        steps = [(1, object()), (5, object())]
        got = locate_in_turns(raw, turns, steps)
        assert got[0].marker_turn_index == 2
        assert got[0].handoff_turn_index == 4
        assert got[0].first_step_after == 1

    def test_depth1_markers_do_not_create_extra_boundaries(self, tmp_path):
        """The same sidecar is spliced repeatedly; only depth-0 counts."""
        raw = boundaries_from_raw(_raw(tmp_path))
        turns = self._turns() + [
            _T(6, "system", "Performed context summarization and handoff to continue task.",
               subagent_depth=1),
        ]
        got = locate_in_turns(raw, turns, [(1, object()), (5, object())])
        assert len(got) == 1
        assert got[0].marker_turn_index == 2


class TestContract:
    def test_post_compaction_steps_is_positional(self, tmp_path):
        raw = boundaries_from_raw(_raw(tmp_path))
        located = locate_in_turns(
            raw,
            [_T(0, "user"), _T(1, "system",
                "Performed context summarization and handoff to continue task."),
             _T(2, "user", "Here are the answers the other agent provided."),
             _T(3, "assistant")],
            [(0, object()), (3, object())],
        )
        assert post_compaction_steps(located) == {1}

    def test_reconstruction_is_not_implemented(self):
        """Guards the §15 claim: detection landed, reconstruction did not.

        If someone implements replacement slicing, this test fails and forces
        the plan and the coverage claim to be updated together.
        """
        assert reconstruction_is_implemented() is False
