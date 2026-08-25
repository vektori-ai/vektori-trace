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
        # Fsync the directory too. Without it the rename itself can be lost on
        # power failure, leaving the marker absent or -- worse -- present but
        # empty, which is the state `validate()` exists to make impossible.
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
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
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            # Only the LAST line may be torn: that is the one an interrupted
            # append was writing. A malformed line anywhere else means the file
            # was corrupted after it was durably written, and silently dropping
            # the remainder would hide however many paid results follow it.
            if i != len(lines) - 1:
                raise ReOPDStateError(
                    f"{path}: malformed JSON at line {i + 1} of {len(lines)}. "
                    "Only a torn final line is recoverable; a bad line in the "
                    "middle means the file was corrupted after it was written."
                ) from e
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
            actions = read_jsonl(self.actions_path)
            scores = read_jsonl(self.scores_path)

            # Exact key-set equality, not a count. A count comparison accepts a
            # duplicate score standing in for a missing one, and accepts scores
            # for actions that are not in this batch at all -- both of which
            # change the global denominator while looking complete.
            a_keys = [r.get("key") for r in actions]
            s_keys = [r.get("key") for r in scores]
            dupes = {k for k in s_keys if s_keys.count(k) > 1}
            if dupes:
                raise ReOPDStateError(
                    f"update {self.index}: duplicate score keys {sorted(dupes)[:4]}; "
                    "a duplicate can mask a missing score in a count check"
                )
            missing = set(a_keys) - set(s_keys)
            foreign = set(s_keys) - set(a_keys)
            if missing:
                raise ReOPDStateError(
                    f"update {self.index} is marked SCORED but {len(missing)} "
                    f"actions have no score: {sorted(missing)[:4]}"
                )
            if foreign:
                raise ReOPDStateError(
                    f"update {self.index}: score keys outside this batch: "
                    f"{sorted(foreign)[:4]}"
                )

            # Fingerprints bind a score to the exact action bytes and teacher
            # it was bought for. A stale score that survives a resume grades an
            # action the current policy never sampled.
            by_key = {r.get("key"): r for r in actions}
            for s in scores:
                want = by_key[s["key"]].get("score_fingerprint")
                got = s.get("fingerprint")
                if want and got and want != got:
                    raise ReOPDStateError(
                        f"update {self.index}: score for {s['key']} was bought "
                        f"for a different action or teacher ({got} != {want})"
                    )

        if self.reached("TRAINED"):
            self.validate_checkpoint()

    #: What a resumable checkpoint must contain. Model weights alone are not
    #: enough: resuming with a fresh optimizer discards Adam's moments, which
    #: silently changes the effective learning rate for the updates that follow.
    CHECKPOINT_REQUIRED = ("adapter_config.json", "optimizer.pt", "state.json")

    def validate_checkpoint(self) -> dict[str, Any]:
        """Fail unless the checkpoint can actually resume the run.

        An empty directory passing this check is the failure mode that turns a
        crash at update 20 into a silent restart from an untrained adapter.
        """
        cp = self.checkpoint_path
        if not cp.exists():
            raise ReOPDStateError(
                f"update {self.index} is marked TRAINED but has no checkpoint; "
                "the next update would resume from the wrong policy version"
            )
        missing = [f for f in self.CHECKPOINT_REQUIRED if not (cp / f).exists()]
        # Adapter weights are written under either name depending on peft
        # version, so they are checked as an either/or rather than by name.
        if not any((cp / n).exists()
                   for n in ("adapter_model.safetensors", "adapter_model.bin")):
            missing.append("adapter_model.safetensors")
        if missing:
            raise ReOPDStateError(
                f"update {self.index} checkpoint is incomplete, missing "
                f"{missing}. A checkpoint without optimizer, scheduler and RNG "
                "state cannot resume the run it claims to belong to."
            )

        state = json.loads((cp / "state.json").read_text())
        for field in ("update_index", "policy_version", "parent_policy_hash",
                      "rng_state", "scheduler_state"):
            if field not in state:
                raise ReOPDStateError(
                    f"update {self.index} checkpoint state.json lacks {field!r}"
                )
        if int(state["update_index"]) != self.index:
            raise ReOPDStateError(
                f"checkpoint in update-{self.index:03d} claims update "
                f"{state['update_index']}; the run would resume at the wrong point"
            )
        if not state.get("reload_verified"):
            raise ReOPDStateError(
                f"update {self.index} checkpoint was never reload-verified. A "
                "saved adapter that does not change logits on reload is the "
                "failure that makes a whole run a no-op."
            )
        return state


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
