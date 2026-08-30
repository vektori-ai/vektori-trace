"""Census every episode outcome across a run's updates. No spend.

Usage: python scripts/pilotd_failure_census.py <run_id> [run_id ...]

Counts sampled / failed / discarded per update and classifies each failure by
its archived `failure_kind` -- cap_termination vs no_reasoning vs other -- so a
reported failure rate comes from artifacts rather than from log reading across
several re-rolled run ids.
"""
import json
import os
import subprocess
import sys
import tempfile

VOL = "vektori-trace-adapters"
BASE = "tau2/live-opd"
MODAL = "/data/vektori-trace/.venv/bin/modal"


def vget(path, dest):
    r = subprocess.run([MODAL, "volume", "get", VOL, path, dest],
                       capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0


def rows(p):
    out = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    for run in sys.argv[1:]:
        print("\n===== %s =====" % run)
        for u in range(10):
            uu = "update-%03d" % u
            with tempfile.TemporaryDirectory() as td:
                ep = os.path.join(td, "e.jsonl")
                if not vget("%s/%s/%s/live_archive/episodes.jsonl" % (BASE, run, uu), ep):
                    continue
                latest = {}
                for r in rows(ep):
                    latest[r["episode_id"]] = r
                st = {}
                for r in latest.values():
                    st[r.get("status")] = st.get(r.get("status"), 0) + 1
                kinds = {}
                for eid, r in sorted(latest.items()):
                    if r.get("status") != "failed":
                        continue
                    tp = os.path.join(td, "t.jsonl")
                    if not vget("%s/%s/%s/live_archive/turns/%s.jsonl"
                                % (BASE, run, uu, eid), tp):
                        kinds.setdefault("archive_missing", []).append(eid)
                        continue
                    for tr in rows(tp):
                        if tr.get("kind") == "failed_turn":
                            k = tr.get("failure_kind") or "unknown"
                            kinds.setdefault(k, []).append(
                                "%s@t%s(%s)" % (eid, tr.get("turn_index"),
                                                tr.get("finish_reason")))
                print("  %s  %s" % (uu, st))
                for k, v in sorted(kinds.items()):
                    print("      %-18s %d  %s" % (k, len(v), ", ".join(v)))


main()
