"""Durable per-update state for a multi-update Tau2 ReOPD run.

A 32-update run dispatches paid teacher calls in every update. The single
requirement that shapes this whole module is that a crash must never cause a
score to be bought twice, and must never cause a half-written update to be
mistaken for a finished one.

The state machine, per update:

    PLANNED -> SAMPLED -> SCORED -> TRAINED

Each transition writes its outputs to a temporary path, validates them, then
atomically renames a marker into place. A restart reads markers, never logs:
a terminal log can end mid-write, and inferring "it looked like it finished"
from one is how a run silently repeats a paid stage.

Why markers and not a single run-state file
-------------------------------------------
One mutable state file has to be rewritten at every transition, so a crash
during that rewrite loses the whole run's position. Independent
write-once markers cannot lose earlier state no matter when the process dies.

What `SCORED` guarantees
------------------------
That `scores.jsonl` holds a complete, fingerprint-matched score for every
action in `actions.jsonl`. Partial score files are normal -- scoring persists
incrementally so a failure at request 30 keeps the 29 already billed -- but
they are only *reused*, never treated as a finished stage.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

#: Ordered. `reached()` compares by index, so a stage implies its predecessors.
STAGES = ("PLANNED", "SAMPLED", "SCORED", "TRAINED")


class ReOPDStateError(RuntimeError):
    """A run directory is inconsistent with what its markers claim."""


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON so a reader never sees a partial file.

    `os.replace` is atomic within a filesystem, and the fsync before it is what
    makes the content durable rather than merely visible -- without it a power
    loss can leave a renamed but empty file, which is worse than no file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """One durable line. Every paid result goes through here immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, tolerating a torn final line.

    A crash mid-append can leave a partial last line. Dropping it is correct:
    the row was never durably written, so whatever produced it will be redone.
    Raising instead would strand a run that is otherwise fully recoverable.
    """
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return rows


@dataclass(frozen=True)
class UpdateDir:
    """One update's directory, and the only thing allowed to interpret it."""

    root: Path
    index: int

    @property
    def path(self) -> Path:
        return self.root / f"update-{self.index:03d}"

    @property
    def actions_path(self) -> Path:
        return self.path / "actions.jsonl"

    @property
    def scores_path(self) -> Path:
        return self.path / "scores.jsonl"

    @property
    def report_path(self) -> Path:
        return self.path / "report.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.path / "checkpoint"

    def marker(self, stage: str) -> Path:
        if stage not in STAGES:
            raise ReOPDStateError(f"unknown stage {stage!r}")
        return self.path / f".{stage}"

    def mark(self, stage: str, payload: dict[str, Any] | None = None) -> None:
        """Record a stage as complete. Written last, after its outputs exist."""
        atomic_write_json(self.marker(stage), payload or {"stage": stage})

    def reached(self, stage: str) -> bool:
        """True when `stage` or any later stage is marked complete."""
        if stage not in STAGES:
            raise ReOPDStateError(f"unknown stage {stage!r}")
        want = STAGES.index(stage)
        return any(self.marker(s).exists() for s in STAGES[want:])

    def stage(self) -> str | None:
        """The furthest stage reached, or None if the update has not started."""
        for s in reversed(STAGES):
            if self.marker(s).exists():
                return s
        return None

    def validate(self) -> None:
        """Fail when a marker claims more than the directory can support.

        The failure this catches: a marker written before its outputs were
        durable, so a restart skips a stage whose results do not exist. Better
        to stop than to train on an absent batch.
        """
        if self.reached("SAMPLED") and not self.actions_path.exists():
            raise ReOPDStateError(
                f"update {self.index} is marked SAMPLED but {self.actions_path.name} "
                "is missing; the marker is ahead of its outputs"
            )
        if self.reached("SCORED"):
            n_actions = len(read_jsonl(self.actions_path))
            n_scores = len(read_jsonl(self.scores_path))
            if n_scores < n_actions:
                raise ReOPDStateError(
                    f"update {self.index} is marked SCORED with {n_scores} scores "
                    f"for {n_actions} actions; a short batch silently changes the "
                    "global denominator"
                )
        if self.reached("TRAINED") and not self.checkpoint_path.exists():
            raise ReOPDStateError(
                f"update {self.index} is marked TRAINED but has no checkpoint; "
                "the next update would resume from the wrong policy version"
            )


class RunState:
    """The run directory: a frozen manifest plus one directory per update."""

    def __init__(self, root: str | Path, *, n_updates: int) -> None:
        self.root = Path(root)
        self.n_updates = n_updates

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def update(self, i: int) -> UpdateDir:
        if not 0 <= i < self.n_updates:
            raise ReOPDStateError(f"update {i} outside 0..{self.n_updates - 1}")
        return UpdateDir(self.root, i)

    def updates(self) -> Iterator[UpdateDir]:
        for i in range(self.n_updates):
            yield self.update(i)

    def freeze_manifest(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write the manifest once; refuse to change it afterwards.

        Restarting a run under changed settings -- a different cap, batch shape,
        or prefix pool -- produces one artifact trained under two recipes, and
        nothing in the checkpoint would record the split. Comparing the incoming
        payload against the frozen one is what makes a resume safe.
        """
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text())
            drift = {
                k: (existing.get(k), payload.get(k))
                for k in set(existing) | set(payload)
                if k not in ("started_at",) and existing.get(k) != payload.get(k)
            }
            if drift:
                raise ReOPDStateError(
                    f"run manifest already frozen with different settings: {drift}. "
                    "Resuming would train one artifact under two recipes; start a "
                    "new run id instead."
                )
            return existing
        atomic_write_json(self.manifest_path, payload)
        return payload

    def resume_point(self) -> int:
        """The first update that is not fully TRAINED.

        Validates every earlier update on the way, so a corrupt run is caught
        before it dispatches anything paid rather than at the point of use.
        """
        for u in self.updates():
            u.validate()
            if not u.reached("TRAINED"):
                return u.index
        return self.n_updates

    def paid_scores(self, i: int) -> dict[str, Any]:
        """Scores already bought for update `i`, keyed by action key.

        This is the money-saving path: `score_replay_batch` skips any action
        whose key appears here. A partial file from a crashed attempt is exactly
        what this is for.
        """
        return {r["key"]: r for r in read_jsonl(self.update(i).scores_path)
                if "key" in r}

    def summary(self) -> dict[str, Any]:
        stages = {}
        for u in self.updates():
            s = u.stage()
            if s:
                stages[u.index] = s
        return {
            "root": str(self.root),
            "n_updates": self.n_updates,
            "resume_at": self.resume_point(),
            "stages": stages,
        }


__all__ = [
    "RunState",
    "STAGES",
    "UpdateDir",
    "ReOPDStateError",
    "append_jsonl",
    "atomic_write_json",
    "read_jsonl",
]
