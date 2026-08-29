"""Read-only: the advantage distribution of a scored update. No spend.

Usage: python scripts/pilotd_advantages.py <scores.jsonl> <actions.jsonl> [label]

Recomputes chunk advantages exactly as training would -- one ratio per aligned
chunk, delegated to chunk_opd -- and reports the distribution, the extreme
tails, and how much of the total absolute advantage lands on the `Okay` opener.
"""
import base64
import collections
import json
import statistics
import sys

sys.path.insert(0, "/data/vektori-trace")


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    scores = {r["key"]: r for r in load(sys.argv[1])}
    actions = {r["key"]: r for r in load(sys.argv[2])}
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    print("== advantages %s ==" % label)
    print("score rows %d   action rows %d" % (len(scores), len(actions)))

    rows = []          # (advantage, key, idx, token_text)
    okay_abs = 0.0
    total_abs = 0.0
    per_action = []

    for key, s in scores.items():
        a = actions.get(key)
        if a is None:
            continue
        lps = a.get("behavior_logprobs") or []
        tb = a.get("action_token_bytes_b64") or []

        def tok(i):
            if i < len(tb):
                try:
                    return base64.b64decode(tb[i]).decode("utf-8", "replace")
                except Exception:
                    return "?"
            return "?"

        n_sup = 0
        for ch in s.get("chunks") or []:
            idxs = ch.get("student_idx") or []
            tlps = ch.get("teacher_logprobs") or []
            L_T = sum(tlps)
            L_S = sum(lps[i] for i in idxs if i < len(lps))
            if L_S == 0:
                continue
            ratio = L_T / L_S
            for i in idxs:
                if i >= len(lps):
                    continue
                adv = (ratio - 1.0) * lps[i]
                t = tok(i)
                rows.append((adv, key, i, t))
                total_abs += abs(adv)
                if t.strip().lower() == "okay":
                    okay_abs += abs(adv)
                n_sup += 1
        per_action.append((n_sup, key))

    if not rows:
        print("no advantages computed")
        return
    vals = [r[0] for r in rows]
    vals_sorted = sorted(vals)
    print("\nsupervised tokens : %d" % len(vals))
    print("mean              : %+.5f" % statistics.mean(vals))
    print("median            : %+.5f" % statistics.median(vals))
    print("stdev             : %.5f" % statistics.pstdev(vals))
    print("min / max         : %+.4f / %+.4f" % (vals_sorted[0], vals_sorted[-1]))
    for q in (0.001, 0.01, 0.25, 0.75, 0.99, 0.999):
        print("  p%-6s        : %+.5f" % (q * 100, vals_sorted[int(q * (len(vals_sorted) - 1))]))
    pos = sum(1 for v in vals if v > 0)
    print("positive / negative: %d / %d" % (pos, len(vals) - pos))

    print("\n== 10 most-negative supervised tokens ==")
    for adv, key, i, t in sorted(rows)[:10]:
        print("  %+9.4f  %-26s idx=%-4d %r" % (adv, key, i, t[:24]))
    print("\n== 10 most-positive ==")
    for adv, key, i, t in sorted(rows)[-10:][::-1]:
        print("  %+9.4f  %-26s idx=%-4d %r" % (adv, key, i, t[:24]))

    print("\n== the Okay question ==")
    n_okay = sum(1 for r in rows if r[3].strip().lower() == "okay")
    print("  Okay tokens              : %d / %d = %.2f%% of supervised"
          % (n_okay, len(vals), 100.0 * n_okay / len(vals)))
    print("  share of total |advantage|: %.2f%%" % (100.0 * okay_abs / total_abs))
    okv = [r[0] for r in rows if r[3].strip().lower() == "okay"]
    if okv:
        print("  Okay advantage median    : %+.4f" % statistics.median(okv))
        print("  Okay advantage min / max : %+.4f / %+.4f" % (min(okv), max(okv)))
    neg10 = sorted(rows)[:10]
    print("  Okay in 10 most-negative : %d/10"
          % sum(1 for r in neg10 if r[3].strip().lower() == "okay"))

    zero = [k for n, k in per_action if n == 0]
    print("\nactions with 0 supervised tokens: %d %s"
          % (len(zero), sorted(zero)[:5]))


main()
