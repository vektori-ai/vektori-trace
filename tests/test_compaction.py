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

    def test_reconstruction_is_implemented(self):
        """Guards the §15 claim, now in the other direction.

        Reconstruction landed by porting `sft_export_traces`' definition, so
        this asserts True. Reverting the enumeration without reverting the flag
        would silently re-open post-compaction selection onto flat prefixes,
        which is the failure this pair of assertions exists to prevent.
        """
        assert reconstruction_is_implemented() is True


class TestReconstruction:
    """The ported SFT reconstruction: summary + new conversation, not both."""

    def _sidecar(self, d, ordinal, text="You are picking up work from a previous AI agent"):
        (d / f"trajectory.summarization-{ordinal}-questions.json").write_text(
            json.dumps({
                "schema_version": "ATIF-v1.7",
                "session_id": f"s-{ordinal}",
                "agent": {"name": "terminus-2-summarization-questions", "version": "2.0.0"},
                "steps": [
                    {"step_id": 1, "source": "user", "message": text},
                    {"step_id": 2, "source": "agent", "message": "I have questions."},
                ],
            })
        )

    def _trial(self, tmp_path, n_boundaries=1):
        """A trial dir shaped like Harbor's: <trial>/agent/trajectory.json."""
        agent = tmp_path / "trial" / "agent"
        agent.mkdir(parents=True)
        steps = [
            {"step_id": 1, "source": "user", "message": "the task"},
            {"step_id": 2, "source": "agent", "message": "pre-boundary action"},
        ]
        for b in range(1, n_boundaries + 1):
            steps.append({
                "step_id": len(steps) + 1, "source": "system",
                "message": "Performed context summarization and handoff to continue task.",
                "extra": {"context_management": {"type": "compaction", "boundary": "replace"}},
            })
            steps.append({"step_id": len(steps) + 1, "source": "user",
                          "message": "Here are the answers the other agent provided."})
            steps.append({"step_id": len(steps) + 1, "source": "agent",
                          "message": f"post-boundary-{b} action"})
            self._sidecar(agent, b, text=f"pickup-head-{b}")
        (agent / "trajectory.json").write_text(json.dumps({
            "schema_version": "ATIF-v1.7",
            "session_id": "sess",
            "agent": {"name": "terminus-2", "version": "2.0.0"},
            "steps": steps,
        }))
        return agent.parent

    def test_pre_boundary_turns_are_dropped(self, tmp_path):
        """The whole point: old conversation is replaced, not appended."""
        from vektori_trace.compaction import current_segment
        from vektori_trace.mining.atif import parse_job_trajectory

        trial = self._trial(tmp_path)
        seg = current_segment(parse_job_trajectory(trial), trial)
        text = " ".join((t.content or "") for t in seg)
        assert "pre-boundary action" not in text
        assert "post-boundary-1 action" in text

    def test_head_comes_from_the_questions_sidecar(self, tmp_path):
        """The inlined user message is the handoff *tail*; the head is the sidecar."""
        from vektori_trace.compaction import current_segment
        from vektori_trace.mining.atif import parse_job_trajectory

        trial = self._trial(tmp_path)
        seg = current_segment(parse_job_trajectory(trial), trial)
        text = " ".join((t.content or "") for t in seg)
        assert "pickup-head-1" in text
        # and the answers still follow it, not dangling
        assert "Here are the answers" in text

    def test_latest_segment_only_after_several_boundaries(self, tmp_path):
        """Up to four boundaries occur in the real corpus; only the last is live."""
        from vektori_trace.compaction import current_segment, split_on_compaction
        from vektori_trace.mining.atif import parse_job_trajectory

        trial = self._trial(tmp_path, n_boundaries=3)
        turns = parse_job_trajectory(trial)
        assert len(split_on_compaction(turns, trial)) == 4
        text = " ".join((t.content or "") for t in current_segment(turns, trial))
        assert "pickup-head-3" in text
        assert "post-boundary-3 action" in text
        for gone in ("pickup-head-1", "pickup-head-2", "post-boundary-1", "post-boundary-2"):
            assert gone not in text

    def test_no_subagent_noise_in_segment(self, tmp_path):
        from vektori_trace.compaction import current_segment
        from vektori_trace.mining.atif import parse_job_trajectory

        trial = self._trial(tmp_path)
        seg = current_segment(parse_job_trajectory(trial), trial)
        assert all(getattr(t, "subagent_depth", 0) == 0 for t in seg)
        assert not any((t.content or "").startswith("[subagent") for t in seg)

    def test_trace_without_compaction_is_unchanged(self, tmp_path):
        from vektori_trace.compaction import current_segment
        from vektori_trace.mining.atif import parse_job_trajectory

        agent = tmp_path / "trial" / "agent"
        agent.mkdir(parents=True)
        (agent / "trajectory.json").write_text(json.dumps({
            "schema_version": "ATIF-v1.7",
            "session_id": "sess",
            "agent": {"name": "terminus-2", "version": "2.0.0"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "task"},
                {"step_id": 2, "source": "agent", "message": "act"},
            ],
        }))
        trial = agent.parent
        turns = parse_job_trajectory(trial)
        seg = current_segment(turns, trial)
        assert len(seg) == len([t for t in turns if t.subagent_depth == 0])

    def test_split_agrees_with_structured_detector(self, tmp_path):
        """Prose split and extra.context_management must find the same boundaries."""
        from vektori_trace.compaction import assert_split_agrees_with_raw
        from vektori_trace.mining.atif import parse_job_trajectory

        trial = self._trial(tmp_path, n_boundaries=2)
        got = assert_split_agrees_with_raw(parse_job_trajectory(trial), trial)
        assert got == {"n_boundaries": 2, "agrees": True}

    def test_matches_the_original_sft_implementation(self, tmp_path):
        """Byte-identical to the function ck75's SFT corpus was built with.

        The exporter now imports from this module, so this pins the ported
        behaviour against an inline copy of the original source.
        """
        from vektori_trace.compaction import split_on_compaction
        from vektori_trace.mining.atif import (
            find_trajectory,
            parse_job_trajectory,
            parse_trajectory_file,
        )

        BOUNDARY = "Performed context summarization and handoff to continue task."

        def original_handoff_head(jobs_dir, ordinal):
            main = find_trajectory(jobs_dir)
            if main is None:
                return []
            path = main.parent / f"trajectory.summarization-{ordinal}-questions.json"
            if not path.exists():
                return []
            return [t for t in parse_trajectory_file(path) if t.subagent_depth == 0]

        def original_split(turns, jobs_dir):
            segments, current, ordinal = [], [], 0
            for turn in turns:
                if turn.subagent_depth > 0:
                    continue
                content = turn.content or ""
                if turn.role == "system" and content.startswith("[subagent"):
                    continue
                if turn.role == "system" and content.strip() == BOUNDARY:
                    segments.append(current)
                    ordinal += 1
                    current = list(original_handoff_head(jobs_dir, ordinal))
                    continue
                current.append(turn)
            segments.append(current)
            return [s for s in segments if s]

        trial = self._trial(tmp_path, n_boundaries=2)
        turns = parse_job_trajectory(trial)
        mine = split_on_compaction(turns, trial)
        theirs = original_split(turns, trial)
        assert len(mine) == len(theirs)
        for a, b in zip(mine, theirs):
            assert [(t.role, t.content) for t in a] == [(t.role, t.content) for t in b]
