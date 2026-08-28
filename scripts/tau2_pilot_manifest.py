#!/usr/bin/env python3
"""Freeze a pilot's preregistration: every episode named before update 0 runs.

Why this file exists at all
---------------------------
`plan_update` crosses `task_ids x seeds` fresh at every update, so a run
configured with 4 tasks and 2 seeds samples *the same 8 scenarios* ten times --
only the episode id carries the update index. That is a data-reuse study, not
80 distinct episodes, and comparing its exposure to ReOPD's 289 distinct
prefixes would be wrong.

So the plan is frozen here, in full, before anything is sampled: all 80
(task, seed) pairs, assigned to updates, written into the manifest, hashed. A
run that cannot show which episodes it *intended* cannot claim afterwards that
what it got was the design.

Training tasks come from C30. The engineering-proof tasks (57, 73, 75, 93) are
S16 and must never appear here: training on them would contaminate the held-out
comparison the entire efficacy claim rests on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def load_split(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def build_plans(
    c30: list[str], *, n_updates: int, episodes_per_update: int,
    seeds: list[int], rng: random.Random,
) -> list[list[dict]]:
    """Assign distinct (task, seed) pairs across updates.

    Every pair is used at most once across the whole run: repetition would make
    later updates re-visit a scenario the policy has already been trained on,
    which is a different experiment (data reuse) and must be chosen
    deliberately rather than fallen into.

    Tasks are shuffled once with a recorded seed and dealt round-robin, so each
    update's batch spans distinct tasks -- the pre-rollout balance mechanism
    that replaces the share gate we demoted to telemetry.
    """
    pairs = [(t, s) for s in seeds for t in c30]
    need = n_updates * episodes_per_update
    if len(pairs) < need:
        raise SystemExit(
            f"need {need} distinct (task, seed) pairs but C30 x seeds gives "
            f"only {len(pairs)}. Add a seed or reduce the schedule -- reusing "
            "a pair silently turns this into a data-reuse study."
        )
    rng.shuffle(pairs)

    # Deal so that no update repeats a TASK, even under a different seed. Two
    # seeds of one task inside one batch concentrate that task's gradient share
    # for a reason the schedule chose rather than the policy earned -- and with
    # the share gate now telemetry, this frozen plan is the only thing
    # enforcing batch balance.
    plans: list[list[dict]] = []
    remaining = list(pairs)
    for u in range(n_updates):
        block: list[tuple[str, int]] = []
        seen: set[str] = set()
        deferred: list[tuple[str, int]] = []
        while remaining and len(block) < episodes_per_update:
            t, s = remaining.pop(0)
            if t in seen:
                deferred.append((t, s))
                continue
            block.append((t, s))
            seen.add(t)
        remaining = deferred + remaining
        if len(block) < episodes_per_update:
            raise SystemExit(
                f"update {u} could only fill {len(block)}/{episodes_per_update} "
                "episodes with distinct tasks; add a seed or widen the pool"
            )
        plans.append([
            {
                "episode_id": f"u{u:03d}-task{t}-seed{s}",
                "task_id": t,
                "seed": s,
            }
            for t, s in block
        ])
    return plans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True,
                    help="task_split_manifest.json (the frozen split)")
    ap.add_argument("--out", required=True, help="manifest.json to write")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n-updates", type=int, default=10)
    ap.add_argument("--episodes-per-update", type=int, default=8)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--plan-seed", type=int, default=20260829,
                    help="RNG seed for the task->update assignment, recorded "
                         "in the manifest so the plan is reproducible")
    ap.add_argument("--parent", default="tau2/runs/a_sft_new_ck35_r2/checkpoint-32")
    ap.add_argument("--parent-adapter-hash", default="3869b147ab7ce5d2")
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--max-action-tokens", type=int, default=4096)
    ap.add_argument("--max-input-tokens", type=int, default=16384)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    split = load_split(Path(a.split))
    parts = split["partitions"]
    c30 = [str(t) for t in parts["C30"]]
    s16 = [str(t) for t in parts["S16"]]

    overlap = sorted(set(c30) & set(s16))
    if overlap:
        raise SystemExit(f"C30 and S16 overlap on {overlap}; the split is bad")

    # The engineering proof trained on S16 tasks. That was fine for a mechanism
    # test and fatal for a preregistered one.
    banned = {"57", "73", "75", "93"}
    leaked = sorted(banned & set(c30))
    if leaked:
        raise SystemExit(
            f"engineering-proof tasks {leaked} appear in C30; they are S16 and "
            "training on them would contaminate the held-out comparison"
        )

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    rng = random.Random(a.plan_seed)
    plans = build_plans(
        c30, n_updates=a.n_updates,
        episodes_per_update=a.episodes_per_update, seeds=seeds, rng=rng,
    )

    flat = [(p["task_id"], p["seed"]) for blk in plans for p in blk]
    assert len(flat) == len(set(flat)), "duplicate (task, seed) in the plan"
    used_tasks = sorted({t for t, _ in flat}, key=int)
    if set(used_tasks) - set(c30):
        raise SystemExit("a planned task is outside C30")

    manifest = {
        "kind": "tau2_live_opd_pilot",
        "run_id": a.run_id,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "domain": "retail",
        "n_updates": a.n_updates,
        "episodes_per_update": a.episodes_per_update,
        "base_model": "Qwen/Qwen3-4B",
        "student_tokenizer": "Qwen/Qwen3-4B",
        "parent": "/" + a.parent.lstrip("/"),
        "adapter_hash": a.parent_adapter_hash,
        "learning_rate": a.learning_rate,
        "temperature": a.temperature,
        "max_turns": a.max_turns,
        "max_action_tokens": a.max_action_tokens,
        "max_input_tokens": a.max_input_tokens,
        "require_reasoning": True,
        "teacher_model": "accounts/fireworks/models/deepseek-v4-flash-0731",
        "teacher_tokenizer": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "teacher_renderer": "deepseek-v4-native",
        "prompt_boundary": "generation-prompt (no empty-think wrapper)",
        "loss_aggregation": "token-mean over one global supervised-token "
                            "denominator (unchanged from the mechanism proof)",
        "share_metrics": "telemetry only; balance is enforced by this frozen "
                         "plan, not by rejecting realized batches",
        # The plan itself.
        "plan_seed": a.plan_seed,
        "seeds": seeds,
        "task_pool": "C30",
        "task_ids": used_tasks,
        "plans_by_update": plans,
        "n_planned_episodes": len(flat),
        "split_manifest_hash": split.get("manifest_hash"),
        "tools_hash": split.get("tools_hash"),
        # Held out. Named here so the run records what it must not touch.
        "eval_pool": "S16",
        "eval_task_ids": s16,
        "eval_seeds": [0, 1],
        "eval_note": "evaluated ONCE on the untouched parent and once on the "
                     "final checkpoint; never inspected mid-run",
        "sealed_pool": "F38",
    }
    manifest["plan_hash"] = hashlib.sha256(
        json.dumps(manifest["plans_by_update"], sort_keys=True).encode()
    ).hexdigest()[:16]

    print(f"run          : {a.run_id}")
    print(f"schedule     : {a.n_updates} x {a.episodes_per_update} = "
          f"{len(flat)} episodes")
    print(f"distinct     : {len(set(flat))} (task, seed) pairs, "
          f"{len(used_tasks)} distinct tasks from C30")
    print(f"seeds        : {seeds}")
    print(f"plan_hash    : {manifest['plan_hash']}")
    print(f"split hash   : {manifest['split_manifest_hash']}")
    print(f"eval (S16)   : {len(s16)} tasks x {manifest['eval_seeds']} seeds")
    print()
    for u, blk in enumerate(plans):
        print(f"  u{u:03d}: " + ", ".join(
            f"{p['task_id']}/s{p['seed']}" for p in blk))

    if a.dry_run:
        print("\nDRY RUN -- nothing written")
        return 0

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"\nfrozen -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
