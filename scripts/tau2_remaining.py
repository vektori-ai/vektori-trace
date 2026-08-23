#!/usr/bin/env python3
"""Emit the task ids of a tau2 domain that no results file has covered yet.

tau2 v0.2.0 cannot resume non-interactively: a colliding --save-to prompts
"resume the run? (y/n)" on stdin, and under stdin=DEVNULL that is an EOFError
rather than a resume. So the way to extend a partial sweep without paying twice
for finished work is to compute the complement here and pass it as explicit
--task-ids into a fresh --save-to.

    python3 scripts/tau2_remaining.py --domain retail          # ids, space-separated
    python3 scripts/tau2_remaining.py --domain telecom --count # just how many
"""
from __future__ import annotations

import argparse
import glob
import json
import shlex
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--tau2-dir", default="/data/tau2")
    ap.add_argument("--count", action="store_true", help="print the count, not the ids")
    a = ap.parse_args()

    root = Path(a.tau2_dir)
    all_ids = [str(t["id"]) for t in
               json.loads((root / "data" / "tau2" / "domains" / a.domain /
                           "tasks.json").read_text())]

    # Any simulation already on disk counts as done, whichever run produced it --
    # results files are per-run, and a domain may be spread over several.
    done: set[str] = set()
    for f in glob.glob(str(root / "data" / "simulations" / "*.json")):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue                      # a run still writing its file
        for s in d.get("simulations", []):
            done.add(str(s.get("task_id")))

    todo = [t for t in all_ids if t not in done]
    if a.count:
        print(f"{a.domain}: {len(all_ids)} total, {len(all_ids)-len(todo)} done, "
              f"{len(todo)} remaining")
    else:
        # Telecom ids carry | and [] -- quote or the shell mangles them.
        print(" ".join(shlex.quote(t) for t in todo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
