#!/usr/bin/env python3
"""Freeze the C30 replay-prefix manifest (V2 §7.2). CPU only, no GPU, no Modal.

Both continuation branches -- continued SFT and replay OPD -- must consume the
*identical* prefix stream. If they diverge, the comparison measures the stream
rather than the objective, and nothing in either training log would show it.
V2 §7.2 puts it plainly: the manifest is built once and read twice, never
regenerated per branch. This script is the "built once".

What it does NOT do is build rows. The corpus already contains every C30 row:
`rows.tokenized.jsonl` holds all partitions and the split is a *filter* over it,
which is exactly how `tau2_sft_train.load_rows` reads W30. Rebuilding rows here
would produce a second, subtly different corpus for the adaptation half -- the
confound V2 §6.2 forbids. So this selects, verifies, orders, and hashes.

Why all 289 rows, and not the ≤180 that V2 §6.2 describes
---------------------------------------------------------
§6.2's `min(6, n)` coverage selector was superseded when the corpus was built:
W30 uses **every genuine decision** (273 rows over 30 tasks), on the grounds that
capping at six would discard ~40% of paid-for supervision. C30 must match,
because §6.2's own closing requirement is that the representation be identical
on both sides of the train split:

    W30   30 tasks   273 rows   33,817 supervised tokens
    C30   30 tasks   289 rows   34,893 supervised tokens

Supervised mass differs by ~3%, so the halves are well matched without any
capping. Applying the selector to C30 alone would introduce precisely the
asymmetry the plan warns about. If the cap is ever reinstated it must be applied
to *both* halves, which means retraining `A_warm`.

The sampling order is frozen here too, not chosen at training time. V2 §8
requires task-first/position-second sampling so every task has support and a few
long traces cannot dominate; a branch that reshuffles independently breaks the
match as surely as a different row set would.

    python scripts/tau2_freeze_c30_prefixes.py --artifacts /data/tau2/artifacts_16384
    python scripts/tau2_freeze_c30_prefixes.py --verify      # re-derive and compare
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

# The seed the split was built under (vektori_trace.tau2.split.SEED). Reused so
# the sampling order is reproducible from the manifest alone.
SEED = 20260824

PARTITION = "C30"
IGNORE = -100

# Refused by name. C30 is the adaptation half; W30 already trained `A_warm`, and
# S16/F38 are evaluation partitions whose contents must never enter an
# optimizer.
ALLOWED_PARTITIONS = ("C30",)


class FreezeError(RuntimeError):
    """An invariant does not hold. Never caught -- the manifest would be wrong."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_c30_rows(artifacts: str) -> tuple[list[dict], dict]:
    """Every C30 row from the frozen corpus, with the split's own invariants.

    Deliberately mirrors `tau2_sft_train.load_rows`: same manifest, same hash
    verification, same partition-isolation assertions, same sort. W30 was read
    this way to train `A_warm`, so C30 is read this way to adapt from it.
    """
    man_path = os.path.join(artifacts, "task_split_manifest.json")
    rows_path = os.path.join(artifacts, "rows.tokenized.jsonl")
    sem_path = os.path.join(artifacts, "rows.semantic.jsonl")
    hash_path = os.path.join(artifacts, "artifact_hashes.json")
    for p in (man_path, rows_path, sem_path, hash_path):
        if not os.path.exists(p):
            raise FreezeError(f"missing {p}")

    manifest = json.load(open(man_path))

    # The corpus must be byte-identical to what the split was frozen against.
    # A rebuilt corpus under the same manifest hash is a different experiment.
    frozen = json.load(open(hash_path))
    for fn, want in frozen.items():
        got = sha256_file(os.path.join(artifacts, fn))
        if got != want:
            raise FreezeError(
                f"{fn} hash {got[:16]} != frozen {want[:16]}; the corpus moved "
                f"under the split"
            )

    want_tasks = set(manifest["partitions"][PARTITION])
    other: set[str] = set()
    for name, ids in manifest["partitions"].items():
        if name != PARTITION:
            other |= set(ids)

    rows = [json.loads(line) for line in open(rows_path)]
    rows = [r for r in rows if r["task_id"] in want_tasks]
    if not rows:
        raise FreezeError(f"no rows for {PARTITION}")

    seen = {r["task_id"] for r in rows}
    if seen - want_tasks:
        raise FreezeError(f"rows outside {PARTITION}: {sorted(seen - want_tasks)}")
    if seen & other:
        # The assertion that makes the adaptation stage an adaptation stage
        # (V2 §4). Enforced in code, never by convention.
        raise FreezeError(
            f"{PARTITION} tasks also present in another partition: "
            f"{sorted(seen & other)} -- W30 ∩ C30 must be empty or `A_warm` has "
            f"already trained on an 'unseen' adaptation task"
        )

    for r in rows:
        n = len(r["input_ids"])
        if n != len(r["labels"]):
            raise FreezeError(f"{r['task_id']}#{r['position']}: length mismatch")
        if not any(l != IGNORE for l in r["labels"]):
            raise FreezeError(f"{r['task_id']}#{r['position']}: no supervised tokens")

    rows.sort(key=lambda r: (int(r["task_id"]) if r["task_id"].isdigit() else 0,
                             r["position"]))
    return rows, manifest


def sampling_order(rows: list[dict], seed: int) -> list[str]:
    """Frozen prefix order: task first, position second (V2 §8).

    Sampling a flat row list would let a few long traces dominate a 32-update
    budget. Cycling tasks in a shuffled order, and taking each task's positions
    in a shuffled order within it, gives every task support.

    This is deliberately NOT the ReOPD paper's step-decaying prefix
    distribution. That is a later preregistered ablation; calling a
    coverage-balanced sampler "paper-identical" would misdescribe the run.
    """
    rng = random.Random(seed)
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)

    tasks = sorted(by_task, key=lambda t: int(t) if t.isdigit() else 0)
    rng.shuffle(tasks)
    queues = {}
    for t in tasks:
        positions = sorted(by_task[t], key=lambda r: r["position"])
        rng.shuffle(positions)
        queues[t] = positions

    order: list[str] = []
    while any(queues[t] for t in tasks):
        for t in tasks:
            if queues[t]:
                r = queues[t].pop(0)
                order.append(f"{r['task_id']}#{r['position']}")
    return order


def build_manifest(rows: list[dict], split_manifest: dict, seed: int) -> dict:
    prefixes = []
    for r in rows:
        supervised = sum(1 for l in r["labels"] if l != IGNORE)
        prefixes.append({
            "prefix_id": f"{r['task_id']}#{r['position']}",
            "task_id": r["task_id"],
            "position": r["position"],
            "action_type": r["action_type"],
            "tool_names": r.get("tool_names") or [],
            # The corpus row's own hash. Lets a branch prove at load time that
            # the row it is training on is the row that was frozen.
            "semantic_hash": r["semantic_hash"],
            "n_tokens": len(r["input_ids"]),
            "n_supervised_tokens": supervised,
        })

    order = sampling_order(rows, seed)
    if sorted(order) != sorted(p["prefix_id"] for p in prefixes):
        raise FreezeError("sampling order does not cover the prefix set exactly")

    payload = {
        "partition": PARTITION,
        "split_manifest_hash": split_manifest["manifest_hash"],
        "seed": seed,
        "n_tasks": len({p["task_id"] for p in prefixes}),
        "n_prefixes": len(prefixes),
        "n_supervised_tokens": sum(p["n_supervised_tokens"] for p in prefixes),
        "prefixes": prefixes,
        "sampling_order": order,
    }
    payload["prefix_manifest_hash"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "prefix_manifest_hash"},
                   sort_keys=True).encode()
    ).hexdigest()[:16]
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="/data/tau2/artifacts_16384")
    ap.add_argument("--out", default=None,
                    help="default: <artifacts>/c30_prefix_manifest.json")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--verify", action="store_true",
                    help="re-derive and compare against the existing manifest "
                         "instead of writing; exits non-zero on any drift")
    a = ap.parse_args()

    out = a.out or os.path.join(a.artifacts, "c30_prefix_manifest.json")

    rows, split_manifest = load_c30_rows(a.artifacts)
    manifest = build_manifest(rows, split_manifest, a.seed)

    per_task: dict[str, int] = {}
    for p in manifest["prefixes"]:
        per_task[p["task_id"]] = per_task.get(p["task_id"], 0) + 1
    counts = sorted(per_task.values())
    types: dict[str, int] = {}
    for p in manifest["prefixes"]:
        types[p["action_type"]] = types.get(p["action_type"], 0) + 1

    print(f"=== C30 prefix manifest (seed={a.seed}) ===")
    print(f"  split manifest   {manifest['split_manifest_hash']}")
    print(f"  tasks            {manifest['n_tasks']}")
    print(f"  prefixes         {manifest['n_prefixes']}")
    print(f"  supervised toks  {manifest['n_supervised_tokens']:,}")
    print(f"  rows per task    min {counts[0]}, median {counts[len(counts)//2]}, "
          f"max {counts[-1]}")
    print(f"  action types     " + ", ".join(f"{k}={v}" for k, v in sorted(types.items())))
    print(f"  prefix hash      {manifest['prefix_manifest_hash']}")

    if a.verify:
        if not os.path.exists(out):
            print(f"\nno manifest at {out} to verify against", file=sys.stderr)
            return 1
        existing = json.load(open(out))
        drift = []
        for key in ("prefix_manifest_hash", "n_prefixes", "n_tasks",
                    "n_supervised_tokens", "split_manifest_hash", "seed"):
            if existing.get(key) != manifest[key]:
                drift.append(f"  {key}: frozen={existing.get(key)!r} "
                             f"rebuilt={manifest[key]!r}")
        if existing.get("sampling_order") != manifest["sampling_order"]:
            drift.append("  sampling_order differs")
        if drift:
            print("\nDRIFT -- the frozen manifest and a fresh rebuild disagree:",
                  file=sys.stderr)
            print("\n".join(drift), file=sys.stderr)
            return 1
        print("\nverified: rebuild is identical to the frozen manifest")
        return 0

    if os.path.exists(out):
        # Refuse to silently replace a manifest a branch may already have
        # consumed. Both branches must read the same bytes; a second freeze
        # mid-experiment is how they stop doing so.
        existing = json.load(open(out))
        if existing.get("prefix_manifest_hash") != manifest["prefix_manifest_hash"]:
            print(f"\n{out} already exists with hash "
                  f"{existing.get('prefix_manifest_hash')} != "
                  f"{manifest['prefix_manifest_hash']}.\nRefusing to overwrite: a "
                  f"branch may already have trained against it. Delete it "
                  f"deliberately if the corpus genuinely changed.", file=sys.stderr)
            return 1
        print(f"\n{out} already frozen with an identical hash; nothing to do")
        return 0

    json.dump(manifest, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    print("  Both continuation branches read this file. Build once, read twice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
