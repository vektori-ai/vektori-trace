"""Metadata-only manifest correction: record the contract versions actually used.

Usage:
  python scripts/pilotd_amend_manifest.py <manifest.json> <out.json> [--apply]

The frozen manifest recorded `parser_version: v2` / `projection_version: v3`,
but the code that sampled and scored this run is v3/v4, and update-0's paid
score rows prove it: all 75 score fingerprints equal their action row's
`score_fingerprint`, and `live_score_fingerprint` binds both versions. Had the
scores been bought under v2/v3 every fingerprint would differ.

This corrects the two labels and preserves the superseded values. It touches
NOTHING that determines the experiment: not the schedule, not the plan hash,
not the task/seed pairs, not policy weights, not captured actions, not score
fingerprints. Without `--apply` it only writes the corrected copy for review.
"""
import argparse
import json
import sys

sys.path.insert(0, "/data/vektori-trace")

from vektori_trace.tau2.live_agent import PARSER_VERSION
from vektori_trace.tau2.live_projection import PROJECTION_VERSION
from vektori_trace.tau2.live_score import SCORE_ALGORITHM

AMENDMENT = {
    "date": "2026-08-29",
    "kind": "metadata_only_contract_label_correction",
    "what_changed": [
        "top-level parser_version: v2 -> v3",
        "top-level projection_version: v3 -> v4",
    ],
    "superseded": {"parser_version": "v2", "projection_version": "v3"},
    "why": (
        "The manifest was frozen carrying the pre-repair contract labels. The code "
        "that sampled and scored every action in this run is parser v3 / projection "
        "v4 (restart gates 1 and 5). Update-0's 75 paid score rows each carry a "
        "fingerprint equal to their action row's score_fingerprint, and "
        "live_score_fingerprint binds PARSER_VERSION and PROJECTION_VERSION -- so "
        "the scores were demonstrably bought under v3/v4. The labels were wrong, "
        "not the run."
    ),
    "unchanged": [
        "plan_hash and the frozen 80 (task, seed) pairs",
        "plans_by_update -- every update's roster",
        "parent adapter and all policy weights",
        "captured actions, token ids and behaviour logprobs",
        "score fingerprints and every paid teacher score",
        "score_algorithm (chunk-v2, already correct)",
        "okay_token_prediction, recorded before update 0 was trained",
    ],
    "evidence": (
        "scripts/pilotd_scorerow.py over update-000/scores.jsonl: 75 rows, "
        "score_algorithm chunk-v2 on all, fingerprint matched 75 / mismatched 0."
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    m = json.load(open(args.manifest))
    before = (m.get("parser_version"), m.get("projection_version"), m.get("score_algorithm"))
    print("manifest before : parser=%s projection=%s score=%s" % before)
    print("code contract   : parser=%s projection=%s score=%s"
          % (PARSER_VERSION, PROJECTION_VERSION, SCORE_ALGORITHM))

    if before[0] == PARSER_VERSION and before[1] == PROJECTION_VERSION:
        print("already correct; nothing to amend")
        return

    plan_before = m.get("plan_hash")
    m["parser_version"] = PARSER_VERSION
    m["projection_version"] = PROJECTION_VERSION
    m.setdefault("amendments", []).append(AMENDMENT)

    assert m.get("plan_hash") == plan_before, "plan_hash must not change"
    assert m.get("score_algorithm") == SCORE_ALGORITHM

    with open(args.out, "w") as fh:
        json.dump(m, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("wrote %s" % args.out)
    print("manifest after  : parser=%s projection=%s score=%s"
          % (m["parser_version"], m["projection_version"], m["score_algorithm"]))
    print("plan_hash unchanged: %s" % m["plan_hash"])
    if not args.apply:
        print("\n(dry run -- pass --apply after review, then upload to the volume)")


main()
