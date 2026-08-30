"""Print the frozen task splits: which tasks are training vs evaluation.

Usage: python scripts/pilotd_show_splits.py <manifest.json> [id ...]

Any ids given are looked up against both pools, so "was task 57 trained on?"
is answered from the manifest rather than from memory.
"""
import json
import sys


def main():
    m = json.load(open(sys.argv[1]))
    train = [str(t) for t in (m.get("task_ids") or [])]
    ev = [str(t) for t in (m.get("eval_task_ids") or [])]
    print("train pool %s : %d tasks" % (m.get("task_pool"), len(train)))
    print("  %s" % " ".join(train))
    print("eval pool  %s : %d tasks" % (m.get("eval_pool"), len(ev)))
    print("  %s" % " ".join(ev))
    print("sealed pool %s (never used here)" % m.get("sealed_pool"))
    overlap = sorted(set(train) & set(ev))
    print("train/eval overlap: %s" % (overlap or "none"))

    for q in sys.argv[2:]:
        where = []
        if q in train:
            where.append("TRAIN (%s)" % m.get("task_pool"))
        if q in ev:
            where.append("EVAL (%s)" % m.get("eval_pool"))
        print("  task %-4s -> %s" % (q, " + ".join(where) or "in NEITHER pool"))


main()
