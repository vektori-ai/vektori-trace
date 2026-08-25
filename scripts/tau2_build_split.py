#!/usr/bin/env python3
"""Freeze the Tau2 retail 30/30/16/38 manifest. CPU only, no GPU, no Modal.

Consumes `eligibility_report.json` from `tau2_build_corpus.py` and Tau2's own
task definitions, and emits a hashed manifest whose invariants are asserted
rather than described. Run it twice: the manifest hash must be identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vektori_trace.tau2.split import (
    CONTAMINATED, SEED, assert_invariants, balance_report, build_split,
)
from vektori_trace.tau2.difficulty import difficulty_bands
from vektori_trace.tau2.taskmeta import build_task_metas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="/data/tau2/artifacts_16384")
    ap.add_argument("--survey", default="docs/tau2-teacher-survey.json")
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--results", default="/data/tau2/data/tau2/results/final",
                    help="Tau2 frontier results, the source of real difficulty bands")
    ap.add_argument("--allow-proxy-difficulty", action="store_true",
                    help="proceed on proxy bands when the real ones are "
                         "unavailable; the split is then NOT difficulty-balanced")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.artifacts, "task_split_manifest.json")

    rep_path = os.path.join(args.artifacts, "eligibility_report.json")
    if not os.path.exists(rep_path):
        print(f"missing {rep_path}; run tau2_build_corpus.py first", file=sys.stderr)
        return 2
    rep = json.load(open(rep_path))

    # Fully eligible == structural + policy + rendering. The corpus builder
    # already applied the render gate whole-trace, so this list is final.
    eligible = set(rep.get("fully_eligible_task_ids") or [])
    traced = set((rep.get("per_task") or {}).keys())
    decisions = {t: v.get("n_decisions", 0)
                 for t, v in (rep.get("per_task") or {}).items()}
    print(f"eligible={len(eligible)}  traced={len(traced)}")

    from tau2.registry import registry
    tasks = [t.model_dump() for t in registry.get_tasks_loader(args.domain)()]
    print(f"{args.domain} tasks: {len(tasks)}")

    bands, band_meta = difficulty_bands(args.results, args.domain)
    covered = sum(1 for t in tasks if str(t.get("id")) in bands)
    print(f"difficulty bands: {len(bands)} tasks rated from "
          f"{len(band_meta['files'])} frontier result files "
          f"({covered}/{len(tasks)} of this domain)")
    if covered < len(tasks):
        msg = (f"only {covered}/{len(tasks)} tasks have a real difficulty band; "
               "the rest fall back to a reference-action-count proxy")
        if not args.allow_proxy_difficulty:
            print("\n" + msg, file=sys.stderr)
            print("Re-run with --allow-proxy-difficulty to accept a split that "
                  "is NOT difficulty-balanced, and record that fact.",
                  file=sys.stderr)
            return 2
        print(f"  WARNING: {msg}")

    metas = build_task_metas(tasks, eligible=eligible, traced=traced,
                             survey_path=args.survey, decisions=decisions,
                             bands=bands)

    import collections
    fams = collections.Counter(m.family for m in metas.values())
    print(f"\nfamilies ({len(fams)}):")
    for f, c in fams.most_common():
        print(f"  {f:44s} {c:3d}")

    split = build_split(metas, seed=args.seed)
    assert_invariants(split, metas)          # belt and braces

    print(f"\n=== SPLIT (seed={args.seed}, hash={split.manifest_hash()}) ===")
    for name, ids in split.as_dict().items():
        print(f"  {name}: {len(ids):3d}  {sorted(ids, key=lambda x: int(x) if x.isdigit() else 0)}")

    bal = balance_report(split, metas)
    print("\n=== BALANCE ===")
    for name, b in bal.items():
        print(f"  {name}: n={b['n']} eligible={b['eligible']} traced={b['with_trace']} "
              f"mutation={b['with_mutation']}")
        print(f"        difficulty {b['difficulty']}")

    w30f = set(bal["W30"]["families"])
    c30f = set(bal["C30"]["families"])
    print(f"\n  W30/C30 family overlap: {len(w30f & c30f)} shared, "
          f"{len(w30f ^ c30f)} on one side only")

    manifest = {
        "seed": args.seed,
        "domain": args.domain,
        "manifest_hash": split.manifest_hash(),
        "sizes": {k: len(v) for k, v in split.as_dict().items()},
        "contaminated_reserved_to_S16": [t for t in CONTAMINATED if t in metas],
        "eligibility_report_hash": hashlib.sha256(
            open(rep_path, "rb").read()).hexdigest()[:16],
        "difficulty_source": band_meta,
        "difficulty_coverage": f"{covered}/{len(tasks)}",
        "tools_hash": rep.get("tools_hash"),
        "max_length": rep.get("max_length"),
        "partitions": {k: sorted(v, key=lambda x: int(x) if x.isdigit() else 0)
                       for k, v in split.as_dict().items()},
        "task_meta": {t: {"family": m.family, "difficulty": m.difficulty,
                          "eligible": m.eligible, "has_trace": m.has_trace,
                          "has_mutation": m.has_mutation,
                          "n_decisions": m.n_decisions}
                      for t, m in sorted(metas.items(),
                                         key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)},
        "balance": bal,
        "invariants_asserted": [
            "sizes are exactly 30/30/16/38 and sum to 114",
            "no task appears in two partitions",
            "W30 and C30 are disjoint",
            "every training task has a fully eligible trace",
            "contaminated diagnostics 57/73/75/93 are in S16",
            "no training task appears in S16 or F38",
        ],
    }
    json.dump(manifest, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    print(f"  manifest_hash {split.manifest_hash()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
