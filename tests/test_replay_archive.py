"""§10 per-example archival."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from vektori_trace.replay_archive import (
    ArchiveError,
    build_example_record,
    read_examples,
    write_examples,
)


@dataclass
class _Prefix:
    prefix_id: str = "tr1@7"
    task: str = "pallets__click-3704"
    trace_id: str = "tr1"
    step_index: int = 7


@dataclass
class _Stats:
    def to_dict(self) -> dict[str, Any]:
        return {"chunk_kinds": {"1:1": 2}, "aligned_fraction": 1.0}


@dataclass
class _Adv:
    advantages: list[float] = field(default_factory=lambda: [0.5, -0.25])
    supervised_mask: list[bool] = field(default_factory=lambda: [True, True])
    stats: Any = field(default_factory=_Stats)

    @property
    def n_supervised(self) -> int:
        return sum(1 for s in self.supervised_mask if s)


@dataclass
class _Action:
    sample_index: int = 0
    action_bytes: bytes = b'{"cmd": "ls"}'
    action_token_ids: list[int] = field(default_factory=lambda: [11, 22])
    action_token_bytes: list[bytes] = field(default_factory=lambda: [b'{"cmd', b'": "ls"}'])
    behavior_logprobs: list[float] = field(default_factory=lambda: [-0.1, -0.2])
    policy_version: str = "ck75-v0"
    prompt_token_ids: list[int] | None = field(default_factory=lambda: [1, 2, 3])
    termination_reason: str = "stop"


def _record(**kw):
    return build_example_record(prefix=_Prefix(), action=_Action(), advantages=_Adv(), **kw)


class TestRecord:
    def test_action_bytes_survive_exactly(self):
        """Bytes round-trip through base64, not str.

        An action is bytes; some tokens split a UTF-8 sequence. Decoding to str
        to store it would alter the thing whose exactness is the point.
        """
        raw = b"\xff\xfe not utf-8 \x00"
        rec = build_example_record(
            prefix=_Prefix(),
            action=_Action(action_bytes=raw),
            advantages=_Adv(),
        )
        assert base64.b64decode(rec.payload["action_bytes_b64"]) == raw

    def test_advantage_length_mismatch_refused(self):
        """A record that misaligns advantages and tokens is worse than none."""
        with pytest.raises(ArchiveError, match="advantages for"):
            build_example_record(
                prefix=_Prefix(),
                action=_Action(),
                advantages=_Adv(advantages=[0.1]),
            )

    def test_behaviour_logprobs_are_archived(self):
        """log pi_old cannot be recomputed once the policy updates."""
        rec = _record()
        assert rec.payload["behavior_logprobs"] == [-0.1, -0.2]

    def test_prefixes_stored_as_hash_plus_bounds(self):
        """A ~95k-token prefix is not stored verbatim 32 times."""
        text = "x" * 500_000
        rec = _record(student_prefix_text=text, teacher_prefix_text=text[:10])
        assert rec.payload["student_prefix_n_chars"] == 500_000
        assert len(rec.payload["student_prefix_head"]) == 2000
        assert rec.payload["student_prefix_sha256"] != rec.payload["teacher_prefix_sha256"]

    def test_optional_teacher_fields_omitted_when_absent(self):
        rec = _record()
        assert "teacher_logprobs" not in rec.payload


class TestWriteRead:
    def test_roundtrip_and_index(self, tmp_path):
        recs = [
            build_example_record(
                prefix=_Prefix(prefix_id=f"tr{i}@3", trace_id=f"tr{i}", task=f"task{i%2}"),
                action=_Action(sample_index=i),
                advantages=_Adv(),
            )
            for i in range(4)
        ]
        summary = write_examples(tmp_path / "ex.jsonl", recs)
        assert summary["n_examples"] == 4
        assert summary["global_supervised_tokens"] == 8
        assert set(summary["supervised_tokens_by_task"]) == {"task0", "task1"}
        assert summary["max_task_share"] == 0.5

        back = read_examples(tmp_path / "ex.jsonl")
        assert len(back) == 4
        assert back[0]["key"] == "tr0@3#0"

    def test_truncated_final_line_still_reads(self, tmp_path):
        """A crash mid-write must not cost the completed examples."""
        p = tmp_path / "ex.jsonl"
        write_examples(p, [_record()])
        with p.open("a") as fh:
            fh.write('{"key": "partial", "act')

        back = read_examples(p)
        assert len(back) == 1
        assert back[0]["key"] == "tr1@7#0"

    def test_dominance_is_visible_in_index(self, tmp_path):
        """§8.4: no task may dominate the supervised-token count."""
        recs = [
            build_example_record(
                prefix=_Prefix(prefix_id=f"t@{i}", task="hog" if i else "other"),
                action=_Action(sample_index=i),
                advantages=_Adv(),
            )
            for i in range(4)
        ]
        summary = write_examples(tmp_path / "ex.jsonl", recs)
        assert summary["max_task_share"] == 0.75
