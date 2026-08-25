"""The one batch schedule both continuation arms consume.

V2 §8 preregisters the match between continued SFT and replay OPD over
*updates, prefix exposures, sampling order, effective batch size and LoRA
capacity* -- explicitly not over token counts, which cannot be matched because
sampled and recorded actions differ in length.

    continued SFT   32 updates x 16 states = 512 recorded-expert exposures
    replay OPD      32 updates x 16 states = 512 sampled-student exposures

Built once, read twice. A branch that derives its own order breaks the match as
surely as one that uses a different row set, and nothing in either training log
would show it: both would report 32 updates over C30 and both would look
correct.

Why this is not `cycle_updates` alone
-------------------------------------
`c30_loader.cycle_updates` produces the batches. This module *freezes* them:
it writes the exact prefix ids per update to a hashable artifact so the second
arm can prove it consumed the same stream rather than merely computing one the
same way. A shared function is a claim about code; a shared file is evidence.

The 32x16 shape against a 289-prefix pool
-----------------------------------------
512 exposures over 289 prefixes is ~1.77 passes. The pool does not divide
evenly, so wrapping continues from where the previous pass ended rather than
restarting at the head -- otherwise the first 223 prefixes would be seen twice
and the tail once, and the imbalance would sit exactly where the frozen
task-first order put the later tasks.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

#: V2 §8's pilot budget. Both arms, no exceptions: an arm that changes either
#: number is a different experiment and must say so.
N_UPDATES = 32
N_PER_UPDATE = 16


class ScheduleError(RuntimeError):
    """A schedule is missing, malformed, or disagrees with the frozen one."""


def build_schedule(
    prefix_ids: list[str],
    *,
    n_updates: int = N_UPDATES,
    n_per_update: int = N_PER_UPDATE,
) -> dict[str, Any]:
    """Chunk the frozen sampling order into `n_updates` fixed-size batches.

    `prefix_ids` must already be in the manifest's frozen order -- this does not
    shuffle. The manifest's order is task-first/position-second, so consecutive
    slices are task-balanced by construction; reshuffling here would discard the
    property the freeze exists to guarantee.
    """
    if not prefix_ids:
        raise ScheduleError("no prefixes to schedule")
    if n_updates <= 0 or n_per_update <= 0:
        raise ScheduleError("n_updates and n_per_update must both be > 0")

    updates, i = [], 0
    n = len(prefix_ids)
    for u in range(n_updates):
        batch = [prefix_ids[(i + k) % n] for k in range(n_per_update)]
        updates.append({"update": u, "prefix_ids": batch})
        i = (i + n_per_update) % n

    exposures: dict[str, int] = {}
    for up in updates:
        for pid in up["prefix_ids"]:
            exposures[pid] = exposures.get(pid, 0) + 1

    payload = {
        "n_updates": n_updates,
        "n_per_update": n_per_update,
        "n_exposures": n_updates * n_per_update,
        "n_pool": n,
        "passes": round(n_updates * n_per_update / n, 4),
        "updates": updates,
        "exposure_min": min(exposures.values()),
        "exposure_max": max(exposures.values()),
        "n_unexposed": n - len(exposures),
    }
    payload["schedule_hash"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "schedule_hash"},
                   sort_keys=True).encode()
    ).hexdigest()[:16]
    return payload


def freeze_schedule(path: str, schedule: dict[str, Any]) -> dict[str, Any]:
    """Write the schedule once; refuse to overwrite a different one.

    The second arm to run must read this file, not rebuild it. Refusing an
    overwrite is what makes "built once, read twice" enforceable rather than a
    convention someone remembers.
    """
    if os.path.exists(path):
        existing = json.load(open(path))
        if existing.get("schedule_hash") != schedule["schedule_hash"]:
            raise ScheduleError(
                f"a different schedule is already frozen at {path} "
                f"({existing.get('schedule_hash')} != {schedule['schedule_hash']}). "
                "Both continuation arms must consume the identical stream; "
                "regenerating it per branch invalidates the comparison."
            )
        return existing
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(schedule, fh, indent=1)
    return schedule


def load_schedule(
    path: str,
    *,
    expect_hash: str | None = None,
    expect_pool: list[str] | None = None,
) -> dict[str, Any]:
    """Read the frozen schedule and prove it belongs to this prefix pool."""
    if not os.path.exists(path):
        raise ScheduleError(f"no frozen schedule at {path}")
    sched = json.load(open(path))

    if expect_hash and sched.get("schedule_hash") != expect_hash:
        raise ScheduleError(
            f"schedule hash {sched.get('schedule_hash')} != expected {expect_hash}"
        )
    if expect_pool is not None:
        pool = set(expect_pool)
        used = {pid for up in sched["updates"] for pid in up["prefix_ids"]}
        foreign = used - pool
        if foreign:
            raise ScheduleError(
                f"schedule references {len(foreign)} prefixes outside the loaded "
                f"pool: {sorted(foreign)[:4]}. It was frozen against a different "
                "corpus."
            )
    return sched


def batch_for(schedule: dict[str, Any], update: int) -> list[str]:
    """The prefix ids for one update, by index."""
    for up in schedule["updates"]:
        if up["update"] == update:
            return list(up["prefix_ids"])
    raise ScheduleError(f"schedule has no update {update}")


def describe(schedule: dict[str, Any]) -> str:
    return (
        f"{schedule['n_updates']} updates x {schedule['n_per_update']} states "
        f"= {schedule['n_exposures']} exposures over {schedule['n_pool']} "
        f"prefixes ({schedule['passes']} passes, "
        f"{schedule['exposure_min']}-{schedule['exposure_max']} each), "
        f"hash {schedule['schedule_hash']}"
    )


__all__ = [
    "N_PER_UPDATE",
    "N_UPDATES",
    "ScheduleError",
    "batch_for",
    "build_schedule",
    "describe",
    "freeze_schedule",
    "load_schedule",
]
