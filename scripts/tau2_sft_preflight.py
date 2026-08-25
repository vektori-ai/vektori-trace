#!/usr/bin/env python3
"""Prove every W30 row before a GPU is allocated. CPU only, no GPU, no Modal.

Each check corresponds to a way this repository has previously trained on the
wrong thing and found out late:

- TRL #3927: `assistant_only_loss` silently drops the mask past `max_length`,
  and an all-IGNORE batch reports a plausible loss while learning nothing.
- Qwen3's template wraps `<think></think>` around the *last* assistant turn
  only, so labels built per-message overshoot into the following user turn.
- A stock collator rebuilds `labels` from `input_ids` and destroys the -100
  prompt mask, supervising the entire prompt.

Exit code 0 means every row is safe to train on. Anything else means stop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IGNORE = -100


def fail(msg: str, failures: list[str]) -> None:
    failures.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="/data/tau2/artifacts_16384")
    ap.add_argument("--partition", default="W30")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument("--decode-sample", type=int, default=40,
                    help="rows to decode-verify in full (slow); 0 = all")
    args = ap.parse_args()

    rows_path = os.path.join(args.artifacts, "rows.tokenized.jsonl")
    sem_path = os.path.join(args.artifacts, "rows.semantic.jsonl")
    man_path = os.path.join(args.artifacts, "task_split_manifest.json")
    hash_path = os.path.join(args.artifacts, "artifact_hashes.json")
    for p in (rows_path, sem_path, man_path):
        if not os.path.exists(p):
            print(f"missing {p}", file=sys.stderr)
            return 2

    failures: list[str] = []

    # --- frozen artifact hashes -----------------------------------------
    if os.path.exists(hash_path):
        frozen = json.load(open(hash_path))
        for fn, want in frozen.items():
            fp = os.path.join(args.artifacts, fn)
            if not os.path.exists(fp):
                fail(f"hash manifest names {fn}, which is missing", failures)
                continue
            got = hashlib.sha256(open(fp, "rb").read()).hexdigest()
            if got != want:
                fail(f"{fn} hash {got[:16]} != frozen {want[:16]}", failures)
        print(f"artifact hashes: checked {len(frozen)}")
    else:
        print("artifact_hashes.json absent — corpus is not frozen", file=sys.stderr)
        failures.append("no artifact_hashes.json")

    manifest = json.load(open(man_path))
    want_tasks = set(manifest["partitions"][args.partition])
    print(f"{args.partition}: {len(want_tasks)} tasks, "
          f"manifest_hash={manifest['manifest_hash']}")

    sem = {}
    for line in open(sem_path):
        r = json.loads(line)
        sem[(r["task_id"], r["position"])] = r

    rows = [json.loads(l) for l in open(rows_path)]
    rows = [r for r in rows if r["task_id"] in want_tasks]
    if not rows:
        print(f"no rows for {args.partition}", file=sys.stderr)
        return 2
    print(f"rows in partition: {len(rows)}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    think_ids = tok("<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]

    seen_tasks = set()
    sup_total = 0
    lengths, sup_counts = [], []
    kinds = Counter()
    decoded = 0
    n_decode = len(rows) if args.decode_sample == 0 else args.decode_sample
    step = max(1, len(rows) // max(1, n_decode))

    for i, r in enumerate(rows):
        tag = f"{r['task_id']}#{r['position']}"
        ids, labs = r["input_ids"], r["labels"]
        am = r.get("attention_mask") or [1] * len(ids)
        seen_tasks.add(r["task_id"])
        kinds[r["action_type"]] += 1

        # 1. shapes agree
        if not (len(ids) == len(labs) == len(am)):
            fail(f"{tag}: lengths {len(ids)}/{len(labs)}/{len(am)} disagree", failures)
            continue

        # 2. no row exceeds the pinned context
        if len(ids) > args.max_length:
            fail(f"{tag}: {len(ids)} tokens > max_length {args.max_length}", failures)
        lengths.append(len(ids))

        # 3. supervised labels equal their input_ids; masked ones are exactly -100
        sup_idx = [j for j, l in enumerate(labs) if l != IGNORE]
        if not sup_idx:
            fail(f"{tag}: zero supervised tokens", failures)
            continue
        bad = [j for j, l in enumerate(labs) if l != IGNORE and l != ids[j]]
        if bad:
            fail(f"{tag}: {len(bad)} supervised labels differ from input_ids", failures)
        if any(l != IGNORE for l in labs if l < 0 and l != IGNORE):
            fail(f"{tag}: a masked label is not exactly -100", failures)

        # 4. the supervised span is contiguous and terminal
        if sup_idx != list(range(sup_idx[0], sup_idx[-1] + 1)):
            fail(f"{tag}: supervised span is not contiguous", failures)
        if sup_idx[-1] != len(labs) - 1:
            fail(f"{tag}: supervised span does not reach the end of the row", failures)
        sup_counts.append(len(sup_idx))
        sup_total += len(sup_idx)

        # 5. the think wrapper sits before the span and is NOT supervised
        start = sup_idx[0]
        w = len(think_ids)
        if start >= w and ids[start - w:start] == think_ids:
            if any(labs[j] != IGNORE for j in range(start - w, start)):
                fail(f"{tag}: the <think> wrapper is supervised", failures)
        else:
            fail(f"{tag}: no masked <think> wrapper immediately before the target",
                 failures)

        # 6. the span decodes to the intended assistant action
        if i % step == 0 and decoded < n_decode:
            decoded += 1
            text = tok.decode([ids[j] for j in sup_idx], skip_special_tokens=False)
            s = sem.get((r["task_id"], r["position"]))
            if s is None:
                fail(f"{tag}: no semantic row to compare against", failures)
            else:
                content = (s["target"].get("content") or "").strip()
                if content and content[:60] not in text:
                    fail(f"{tag}: decoded span does not contain the target text",
                         failures)
                for c in (s["target"].get("tool_calls") or []):
                    if c["function"]["name"] not in text:
                        fail(f"{tag}: tool {c['function']['name']} missing from span",
                             failures)

    # 7. partition isolation
    stray = seen_tasks - want_tasks
    if stray:
        fail(f"rows from tasks outside {args.partition}: {sorted(stray)}", failures)
    other = set()
    for name, ids_ in manifest["partitions"].items():
        if name != args.partition:
            other |= set(ids_)
    leak = seen_tasks & other
    if leak:
        fail(f"tasks also present in another partition: {sorted(leak)}", failures)

    lengths.sort(); sup_counts.sort()
    def q(v, p): return v[min(len(v) - 1, int(len(v) * p))] if v else 0
    print(f"\ntokens/row     min={lengths[0]} p50={q(lengths,.5)} "
          f"p90={q(lengths,.9)} max={lengths[-1]}")
    print(f"supervised/row min={sup_counts[0]} p50={q(sup_counts,.5)} "
          f"max={sup_counts[-1]}  total={sup_total:,}")
    print(f"action types   {dict(kinds)}")
    print(f"tasks covered  {len(seen_tasks)}/{len(want_tasks)}")
    print(f"decode-verified {decoded} rows")

    if failures:
        print(f"\nFAILED: {len(failures)} problem(s)")
        for f in failures[:25]:
            print(f"  {f}")
        return 1
    print("\nPREFLIGHT PASSED — every row is safe to train on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
