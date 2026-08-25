#!/usr/bin/env python3
"""Subset a frozen Phase 7 manifest to its 45 selection prefixes.

`phase7_eval.py` generates from every entry in `manifest["prefixes"]` and uses
`selection_prefix_ids` only for the pass/fail bar. Handing it the frozen 60
therefore pays for 15 tripwires per arm -- including `first_edit` and
`long_context`, whose prompts run 32-35k tokens and will evict the KV cache at
concurrency 4. The canary in OPD-MULTITURN-PLAN.md 8.4 is the 45, so the file
the driver reads must *be* the 45.

The parent sha is recorded, not inherited: this file's own sha necessarily
differs from the frozen manifest's, and a subset that silently kept the parent
digest would be claiming to be something it is not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vektori_trace.evaluate.phase7 import selection_prefix_ids

EXPECT = 45


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect", type=int, default=EXPECT)
    args = ap.parse_args()

    raw = args.source.read_bytes()
    parent_sha = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    prefixes = manifest["prefixes"]

    keep = set(selection_prefix_ids(prefixes))
    subset = [p for p in prefixes if p["prefix_id"] in keep]

    # `selection` is written per entry by phase7_manifest.py; if it disagrees
    # with the category/suite filter the manifest is not the one this canary
    # was frozen against.
    flagged = {p["prefix_id"] for p in prefixes if p.get("selection")}
    if flagged and flagged != keep:
        print(
            f"selection flag disagrees with category filter: "
            f"{len(flagged)} flagged vs {len(keep)} filtered",
            file=sys.stderr,
        )
        return 1

    if len(subset) != args.expect:
        print(
            f"selection set is {len(subset)}, not {args.expect}: refusing to "
            f"write. The canary bar is defined on {args.expect} prefixes.",
            file=sys.stderr,
        )
        return 1

    manifest["prefixes"] = subset
    manifest["parent_manifest_sha256"] = parent_sha
    manifest["parent_manifest_path"] = str(args.source)
    manifest["subset"] = "selection_prefix_ids"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(manifest, indent=2, sort_keys=True).encode()
    args.out.write_bytes(body)

    from collections import Counter
    cells = Counter((p["suite"], p["category"]) for p in subset)
    print(f"parent      {args.source}  sha256 {parent_sha}")
    print(f"parent size {len(prefixes)} prefixes")
    print(f"subset      {args.out}  sha256 {hashlib.sha256(body).hexdigest()}")
    print(f"subset size {len(subset)} prefixes")
    for (suite, cat), n in sorted(cells.items()):
        print(f"  {suite:16} {cat:18} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
