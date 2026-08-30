"""Projected teacher/student log-likelihood gap for a scored update. No spend.

Usage: python scripts/pilotd_kl_gap.py <scores.jsonl> <actions.jsonl> [label]

This is the quantity OPD actually optimises. Reverse KL pushes the student
toward the teacher on states the student visits, so the direct question --
"is the student getting closer to DeepSeek?" -- is the per-token gap

    gap_i = L_T(chunk) / n_chunk  -  behaviour_logprob_i

averaged over supervised tokens. It must SHRINK across updates if OPD is
working. It is reported as the projected gap, never as "teacher KL": the
tokenizers differ, so this is a byte-chunk projection, not exact tokenwise KL.

Reported per payload kind too, because reasoning and visible content can move
in opposite directions and a single mean would hide that.
"""
import base64
import collections
import json
import statistics
import sys


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    scores = {r["key"]: r for r in load(sys.argv[1])}
    actions = {r["key"]: r for r in load(sys.argv[2])}
    label = sys.argv[3] if len(sys.argv) > 3 else ""

    gaps = []
    by_kind = collections.defaultdict(list)
    t_sum = s_sum = 0.0
    n = 0
    for key, sc in scores.items():
        a = actions.get(key)
        if a is None:
            continue
        lps = a.get("behavior_logprobs") or []
        for ch in sc.get("chunks") or []:
            idxs = [i for i in (ch.get("student_idx") or []) if i < len(lps)]
            if not idxs:
                continue
            tl = ch.get("teacher_logprobs") or []
            L_T = sum(tl)
            # teacher mass spread evenly over the chunk's student tokens, which
            # is the only defensible per-token attribution across tokenizers
            per_t = L_T / len(idxs)
            for i in idxs:
                g = per_t - lps[i]
                gaps.append(g)
                by_kind[ch.get("kind", "?")].append(g)
                t_sum += per_t
                s_sum += lps[i]
                n += 1

    if not gaps:
        print("no supervised tokens"); return
    gaps.sort()
    print("== projected teacher/student gap %s ==" % label)
    print("  supervised tokens : %d" % n)
    print("  mean gap          : %+.5f   <- shrinking toward 0 = OPD working" % (sum(gaps) / n))
    print("  median gap        : %+.5f" % statistics.median(gaps))
    print("  mean log p_teacher: %+.5f" % (t_sum / n))
    print("  mean log p_student: %+.5f" % (s_sum / n))
    print("  teacher more likely than student on %.1f%% of tokens"
          % (100.0 * sum(1 for g in gaps if g > 0) / n))
    for q in (0.05, 0.25, 0.75, 0.95):
        print("    p%-4s           : %+.5f" % (int(q * 100), gaps[int(q * (len(gaps) - 1))]))
    print("  by payload kind:")
    for k, v in sorted(by_kind.items()):
        print("    %-12s n=%-7d mean=%+.5f  median=%+.5f"
              % (k, len(v), sum(v) / len(v), statistics.median(v)))


main()
