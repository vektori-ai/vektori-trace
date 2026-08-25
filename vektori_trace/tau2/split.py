"""Deterministic 30/30/16/38 partitioning of the Tau2 retail task set.

The split is the experiment's contamination boundary, so it is built by code
that asserts its own invariants rather than by prose that describes them:

    W30 SFT  |  C30 adaptation  |  S16 selection  |  F38 final test  = 114

Ordering matters and is not negotiable (V2 4.2):

1. audit eligibility first -- train tasks must be *made of* eligible traces;
2. reserve every contaminated diagnostic for S16 before choosing train60,
   or one of them can land in W30/C30 by luck;
3. choose train60 from the remaining eligible pool;
4. halve it into W30/C30, balanced against each other;
5. allocate everything left to S16/F38 by family and difficulty.

DeepSeek success gates *eligibility* only. It must never decide W30-vs-C30 or
S16-vs-F38: within a pool, allocation is by normalized family and public
difficulty band under a fixed seed.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

SEED = 20260824

# Already inspected and design-influencing (V2 4). These may never enter the
# blind final test, and reserving them first is what guarantees it.
CONTAMINATED = ("57", "73", "75", "93")

SIZES = {"W30": 30, "C30": 30, "S16": 16, "F38": 38}


class SplitInvariantError(AssertionError):
    """A partition invariant does not hold. Never caught -- the split is wrong."""


@dataclass
class TaskMeta:
    """What allocation is allowed to look at."""

    task_id: str
    family: str                 # normalized workflow family
    difficulty: str             # public difficulty band, or "unknown"
    eligible: bool              # has a fully eligible trace
    n_decisions: int = 0
    has_mutation: bool = False
    has_trace: bool = False


@dataclass
class Split:
    W30: list[str] = field(default_factory=list)
    C30: list[str] = field(default_factory=list)
    S16: list[str] = field(default_factory=list)
    F38: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[str]]:
        return {"W30": self.W30, "C30": self.C30, "S16": self.S16, "F38": self.F38}

    def manifest_hash(self) -> str:
        payload = json.dumps(
            {k: sorted(v, key=_k) for k, v in self.as_dict().items()},
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def _k(t: str) -> int:
    return int(t) if t.isdigit() else 0


def _stratify(tasks: list[TaskMeta], n: int, rng: random.Random) -> list[str]:
    """Take `n` tasks, spreading families and difficulty bands evenly.

    Round-robin over (family, difficulty) buckets rather than sampling: with 16
    slots and a dozen families, independent sampling routinely leaves a family
    entirely on one side.
    """
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for t in tasks:
        buckets[(t.family, t.difficulty)].append(t.task_id)
    for v in buckets.values():
        rng.shuffle(v)

    order = sorted(buckets, key=lambda b: (-len(buckets[b]), b))
    taken: list[str] = []
    while len(taken) < n:
        progressed = False
        for b in order:
            if buckets[b] and len(taken) < n:
                taken.append(buckets[b].pop())
                progressed = True
        if not progressed:
            break
    return sorted(taken, key=_k)


def _balanced_halve(tasks: list[TaskMeta], rng: random.Random,
                    near_duplicates: dict[str, str] | None = None
                    ) -> tuple[list[str], list[str]]:
    """Split train60 into two halves matched on family, difficulty and mutation.

    Near-duplicate templates stay on the same side: if W30 holds a near-copy of
    a C30 task, `A_warm` has effectively trained on an "unseen" adaptation task
    and the adaptation measurement is contaminated.
    """
    groups: dict[str, list[TaskMeta]] = defaultdict(list)
    for t in tasks:
        key = (near_duplicates or {}).get(t.task_id, t.task_id)
        groups[key].append(t)

    blocks = sorted(groups.values(),
                    key=lambda g: (g[0].family, g[0].difficulty, _k(g[0].task_id)))
    rng.shuffle(blocks)
    blocks.sort(key=lambda g: -len(g))          # place big blocks first

    a: list[TaskMeta] = []
    b: list[TaskMeta] = []
    for blk in blocks:
        target = a if _cost(a, blk) <= _cost(b, blk) else b
        if len(target) + len(blk) > len(tasks) // 2:
            target = b if target is a else a
        target.extend(blk)

    return (sorted((t.task_id for t in a), key=_k),
            sorted((t.task_id for t in b), key=_k))


def _cost(side: list[TaskMeta], blk: list[TaskMeta]) -> float:
    """How much adding `blk` unbalances this side. Lower is better."""
    fam = sum(1 for t in side if t.family == blk[0].family)
    dif = sum(1 for t in side if t.difficulty == blk[0].difficulty)
    mut = sum(1 for t in side if t.has_mutation)
    return len(side) * 1.0 + fam * 2.0 + dif * 1.0 + (mut if blk[0].has_mutation else 0)


def build_split(metas: dict[str, TaskMeta], *, seed: int = SEED,
                near_duplicates: dict[str, str] | None = None) -> Split:
    rng = random.Random(seed)
    all_ids = sorted(metas, key=_k)

    total = sum(SIZES.values())
    if len(all_ids) != total:
        raise SplitInvariantError(
            f"{len(all_ids)} tasks supplied but the split needs exactly {total}"
        )

    # 1. contaminated diagnostics -> S16, before anything else looks at the pool
    reserved = [t for t in CONTAMINATED if t in metas]
    pool = [metas[t] for t in all_ids if t not in reserved]

    # 2. train60 from eligible tasks only
    eligible = [t for t in pool if t.eligible]
    if len(eligible) < SIZES["W30"] + SIZES["C30"]:
        raise SplitInvariantError(
            f"only {len(eligible)} eligible non-reserved tasks; need "
            f"{SIZES['W30'] + SIZES['C30']}. Collect more DeepSeek trials on "
            "training candidates rather than shrinking the evaluation partitions."
        )
    train60 = _stratify(eligible, SIZES["W30"] + SIZES["C30"], rng)

    # 3. balanced halving
    w30, c30 = _balanced_halve([metas[t] for t in train60], rng, near_duplicates)

    # 4. everything else -> S16 (reserved first) then F38
    rest = [metas[t] for t in all_ids if t not in set(train60) | set(reserved)]
    s16_extra = _stratify(rest, SIZES["S16"] - len(reserved), rng)
    s16 = sorted(reserved + s16_extra, key=_k)
    f38 = sorted((t.task_id for t in rest if t.task_id not in set(s16_extra)), key=_k)

    split = Split(W30=w30, C30=c30, S16=s16, F38=f38)
    assert_invariants(split, metas)
    return split


def assert_invariants(split: Split, metas: dict[str, TaskMeta]) -> None:
    """Every property the experiment's validity rests on. Raises, never warns."""
    parts = split.as_dict()

    for name, want in SIZES.items():
        got = len(parts[name])
        if got != want:
            raise SplitInvariantError(f"{name} has {got} tasks, expected {want}")

    seen: dict[str, str] = {}
    for name, ids in parts.items():
        for t in ids:
            if t in seen:
                raise SplitInvariantError(
                    f"task {t} appears in both {seen[t]} and {name}"
                )
            seen[t] = name

    total = sum(len(v) for v in parts.values())
    if total != len(metas):
        raise SplitInvariantError(
            f"partitions cover {total} tasks but {len(metas)} were supplied"
        )

    if set(split.W30) & set(split.C30):
        raise SplitInvariantError("W30 and C30 overlap; the adaptation stage is void")

    train = set(split.W30) | set(split.C30)
    for t in train:
        if not metas[t].eligible:
            raise SplitInvariantError(f"training task {t} has no eligible trace")

    for t in CONTAMINATED:
        if t in metas and t not in split.S16:
            raise SplitInvariantError(
                f"contaminated diagnostic {t} is in "
                f"{seen.get(t, 'no partition')}, must be in S16"
            )

    if train & (set(split.S16) | set(split.F38)):
        raise SplitInvariantError("a training task also appears in an eval partition")


def balance_report(split: Split, metas: dict[str, TaskMeta]) -> dict[str, Any]:
    """Composition of every partition, so imbalance is visible not assumed."""
    out: dict[str, Any] = {}
    for name, ids in split.as_dict().items():
        ms = [metas[t] for t in ids]
        fam: dict[str, int] = defaultdict(int)
        dif: dict[str, int] = defaultdict(int)
        for m in ms:
            fam[m.family] += 1
            dif[m.difficulty] += 1
        out[name] = {
            "n": len(ms),
            "families": dict(sorted(fam.items())),
            "difficulty": dict(sorted(dif.items())),
            "with_mutation": sum(1 for m in ms if m.has_mutation),
            "with_trace": sum(1 for m in ms if m.has_trace),
            "eligible": sum(1 for m in ms if m.eligible),
        }
    return out
