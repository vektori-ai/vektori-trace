"""Read-only: dump a failed turn's raw generation. No spend.

Usage: python scripts/pilotd_show_failed_turn.py <turns.jsonl> [turn_index]

A cap termination is archived as a FailedTurn before the parse error
propagates -- the generation was paid for and its bytes exist nowhere else.
This prints what the model actually wrote, so a 4096-token overrun can be
diagnosed rather than guessed at.
"""
import base64
import json
import sys


def decode(row):
    for k in ("action_bytes_b64", "raw_sampled_bytes_b64", "raw_bytes_b64"):
        if row.get(k):
            return base64.b64decode(row[k]).decode("utf-8", "replace")
    for k in ("raw_sampled_bytes", "raw", "text"):
        if isinstance(row.get(k), str):
            return row[k]
    return None


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    want = int(sys.argv[2]) if len(sys.argv) > 2 else None
    print("rows in archive: %d" % len(rows))
    print("row kinds: %s" % sorted({r.get("kind", "?") for r in rows}))
    for r in rows:
        ti = r.get("turn_index")
        print("  turn %-4s kind=%-14s finish=%-8s tokens=%s"
              % (ti, r.get("kind", "?"), r.get("finish_reason"),
                 len(r.get("sampled_token_ids") or r.get("action_token_ids") or [])))

    targets = [r for r in rows
               if (want is None or r.get("turn_index") == want)
               and (r.get("finish_reason") == "length"
                    or "fail" in str(r.get("kind", "")).lower())]
    if not targets:
        targets = [r for r in rows if want is not None and r.get("turn_index") == want]
    if not targets:
        print("\nno failed/capped turn found")
        return

    for r in targets:
        raw = decode(r)
        print("\n" + "=" * 70)
        print("turn %s  finish_reason=%s  kind=%s"
              % (r.get("turn_index"), r.get("finish_reason"), r.get("kind")))
        n = len(r.get("sampled_token_ids") or r.get("action_token_ids") or [])
        print("tokens: %d" % n)
        if r.get("failure_reason"):
            print("reason: %s" % str(r["failure_reason"])[:300])
        if raw is None:
            print("!! no raw bytes in this row; keys = %s" % sorted(r.keys()))
            continue
        print("raw bytes: %d chars" % len(raw))
        print("=" * 70)
        print(raw)


main()
