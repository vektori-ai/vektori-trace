"""Per-example archival for a replay OPD run (plan §10).

§10 requires every replay example to archive the material needed to reproduce
it: canonical messages, both rendered prefixes, exact sampled bytes, token ids
and per-token log probabilities, chunk membership, masks, and the teacher's
request metadata. `run_replay_chunk_opd` returns *aggregate* statistics — chunk
kind proportions, advantage magnitudes, tokens by prefix and step — which are
what you read to interpret a run. They are not what you read to re-derive one.

The distinction matters more here than in a normal training run. If the update
turns out to be wrong, the aggregate report says *that* something is off; only
the per-example record says *where*. And behaviour log probabilities cannot be
recomputed after the fact — the sampling policy is frozen and then updated — so
an example not archived at write time is gone.

Format: one JSON Lines file, one object per supervised action. JSONL rather
than a single document because a 32-action run is ~100 MB of token-level detail
and a partially-written file should still be readable up to its last complete
line — the failure mode this exists to serve is a crash midway.

Bytes are stored base64 rather than decoded text: an action is bytes, some of
them are not valid UTF-8 mid-token, and round-tripping through `str` would
silently alter the thing whose exactness is the point.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class ArchiveError(RuntimeError):
    """An example cannot be archived faithfully."""


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass
class ExampleRecord:
    """One archived replay example. Field names match §10's bullet list."""

    prefix_id: str
    sample_index: int
    task: str
    trace_id: str
    step_index: int
    policy_version: str
    payload: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.prefix_id}#{self.sample_index}"


def build_example_record(
    *,
    prefix: Any,
    action: Any,
    advantages: Any,
    canonical_messages: list[dict[str, Any]] | None = None,
    student_prefix_text: str | None = None,
    teacher_prefix_text: str | None = None,
    teacher_token_bytes: list[bytes] | None = None,
    teacher_logprobs: list[float] | None = None,
    teacher_request: dict[str, Any] | None = None,
    compaction: dict[str, Any] | None = None,
) -> ExampleRecord:
    """Assemble the §10 record for one action.

    Everything is copied, not referenced: an archive that holds live objects is
    an archive that changes when the batch does.

    Rendered prefixes are stored as a hash plus a bounded head/tail rather than
    in full. A single prefix measured ~95k tokens, so 32 verbatim copies would
    dominate the artifact while adding nothing a hash does not already prove —
    the hash detects any drift between the two renderings, which is what §4's
    contract actually requires. Pass the full text explicitly if a specific
    investigation needs it.
    """
    if advantages is None:
        raise ArchiveError(f"{prefix.prefix_id}: no advantages to archive")

    n = len(action.action_token_ids)
    if len(advantages.advantages) != n:
        raise ArchiveError(
            f"{prefix.prefix_id}#{action.sample_index}: {len(advantages.advantages)} "
            f"advantages for {n} action tokens — the record would misalign the "
            "very quantities it exists to make checkable"
        )

    payload: dict[str, Any] = {
        "key": f"{prefix.prefix_id}#{action.sample_index}",
        "task": prefix.task,
        "trace_id": prefix.trace_id,
        "step_index": prefix.step_index,
        "policy_version": action.policy_version,
        "termination_reason": action.termination_reason,
        # -- the exact sampled bytes (§10) --------------------------------
        "action_bytes_b64": _b64(action.action_bytes),
        "action_bytes_sha256": _sha(action.action_bytes),
        "action_n_bytes": len(action.action_bytes),
        # -- student side --------------------------------------------------
        "student_token_ids": list(action.action_token_ids),
        "student_token_bytes_b64": [_b64(b) for b in action.action_token_bytes],
        "behavior_logprobs": list(action.behavior_logprobs),
        "prompt_token_ids_n": (
            len(action.prompt_token_ids) if action.prompt_token_ids else 0
        ),
        "prompt_token_ids_sha256": (
            _sha(json.dumps(list(action.prompt_token_ids)).encode())
            if action.prompt_token_ids
            else None
        ),
        # -- credit assignment ---------------------------------------------
        "advantages": list(advantages.advantages),
        "supervised_mask": list(advantages.supervised_mask),
        "n_supervised": advantages.n_supervised,
        "chunk_stats": advantages.stats.to_dict(),
    }

    if teacher_token_bytes is not None:
        payload["teacher_token_bytes_b64"] = [_b64(b) for b in teacher_token_bytes]
    if teacher_logprobs is not None:
        payload["teacher_logprobs"] = list(teacher_logprobs)
    if teacher_request is not None:
        payload["teacher_request"] = dict(teacher_request)
    if canonical_messages is not None:
        payload["canonical_messages"] = canonical_messages
    if compaction is not None:
        payload["compaction"] = dict(compaction)

    for name, text in (
        ("student_prefix", student_prefix_text),
        ("teacher_prefix", teacher_prefix_text),
    ):
        if text is None:
            continue
        raw = text.encode()
        payload[f"{name}_sha256"] = _sha(raw)
        payload[f"{name}_n_chars"] = len(text)
        payload[f"{name}_head"] = text[:2000]
        payload[f"{name}_tail"] = text[-2000:]

    return ExampleRecord(
        prefix_id=prefix.prefix_id,
        sample_index=action.sample_index,
        task=prefix.task,
        trace_id=prefix.trace_id,
        step_index=prefix.step_index,
        policy_version=action.policy_version,
        payload=payload,
    )


def write_examples(path: Path, records: Iterable[ExampleRecord]) -> dict[str, Any]:
    """Write records as JSON Lines; return an index of what was written.

    The index is the cheap thing to read back — it answers "is every action
    here, and does any one of them dominate the denominator" without parsing
    100 MB of token detail.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    index: list[dict[str, Any]] = []
    n_bytes = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            line = json.dumps(rec.payload, sort_keys=True)
            fh.write(line + "\n")
            n_bytes += len(line) + 1
            index.append(
                {
                    "key": rec.key,
                    "task": rec.task,
                    "trace_id": rec.trace_id,
                    "step_index": rec.step_index,
                    "n_supervised": rec.payload.get("n_supervised"),
                    "action_bytes_sha256": rec.payload.get("action_bytes_sha256"),
                }
            )

    total = sum(e["n_supervised"] or 0 for e in index)
    by_task: dict[str, int] = {}
    for e in index:
        by_task[e["task"]] = by_task.get(e["task"], 0) + (e["n_supervised"] or 0)

    return {
        "path": str(path),
        "n_examples": len(index),
        "bytes_written": n_bytes,
        "global_supervised_tokens": total,
        "supervised_tokens_by_task": by_task,
        # §8.4: "no task or single trace dominates the global supervised-token
        # count". Reported here so the claim is checkable from the index alone.
        "max_task_share": (
            round(max(by_task.values()) / total, 4) if total and by_task else None
        ),
        "index": index,
    }


def read_examples(path: Path) -> list[dict[str, Any]]:
    """Read back a JSONL archive, tolerating a truncated final line.

    A crash mid-write leaves a partial last line; refusing to read the other 31
    complete examples because of it would defeat the reason this is JSONL.
    """
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return out


__all__ = [
    "ArchiveError",
    "ExampleRecord",
    "build_example_record",
    "read_examples",
    "write_examples",
]
