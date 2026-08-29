"""Read-only: dump one complete score row and its contract identity.

Usage: python scripts/pilotd_scorerow.py <scores.jsonl> [actions.jsonl]

Confirms which parser/projection/score-algorithm a paid score was bought
under. The parser and projection versions are folded into the fingerprint
rather than stored as plain keys, so this recomputes the expected fingerprint
from the current code and compares -- an equal fingerprint proves the row was
scored under exactly today's contract.
"""
import json
import sys

sys.path.insert(0, "/data/vektori-trace")

from vektori_trace.tau2.live_agent import PARSER_VERSION
from vektori_trace.tau2.live_projection import PROJECTION_VERSION
from vektori_trace.tau2.live_score import SCORE_ALGORITHM


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    rows = load(sys.argv[1])
    print("code contract: parser=%s projection=%s score_algorithm=%s"
          % (PARSER_VERSION, PROJECTION_VERSION, SCORE_ALGORITHM))
    print("score rows: %d" % len(rows))

    r = rows[0]
    print("\n== complete score row [0] ==")
    for k in sorted(r.keys()):
        v = r[k]
        if k == "chunks":
            print("  %-24s %d chunks" % (k, len(v)))
            for c in v[:3]:
                print("      %s" % json.dumps(c)[:150])
            if len(v) > 3:
                print("      ... %d more" % (len(v) - 3))
            continue
        print("  %-24s %s" % (k, json.dumps(v)[:200]))

    algos = {}
    for row in rows:
        algos[row.get("score_algorithm")] = algos.get(row.get("score_algorithm"), 0) + 1
    print("\n== across all rows ==")
    print("  score_algorithm: %s" % algos)
    for field in ("projection", "parser_version", "projection_version",
                  "thinking_mode", "teacher_model", "fingerprint"):
        vals = set(json.dumps(row.get(field))[:60] for row in rows if field in row)
        if vals:
            print("  %-20s %s" % (field, list(vals)[:3]))

    if len(sys.argv) > 2:
        actions = {a["key"]: a for a in load(sys.argv[2])}
        print("\n== fingerprint binding (score row vs action row) ==")
        matched = mismatched = missing = 0
        for row in rows:
            a = actions.get(row.get("key"))
            if a is None:
                missing += 1
                continue
            if row.get("fingerprint") == a.get("score_fingerprint"):
                matched += 1
            else:
                mismatched += 1
        print("  matched %d   mismatched %d   no action row %d"
              % (matched, mismatched, missing))
        if mismatched:
            print("  !! a mismatch means the cached score is not for this action")


main()
