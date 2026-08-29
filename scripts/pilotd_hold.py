"""Read-only: verify a live-OPD update's rollout hold. No spend, no GPU.

Usage: python scripts/pilotd_hold.py <report.json> <actions.jsonl> [episodes.jsonl]

Checks the structural gate (sampling provenance, roster, format failures) and
computes the preregistered `Okay` measurements from the captured tokens.
"""
import base64
import collections
import json
import statistics
import sys


def load_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def show(label, value):
    print("  %-34s %s" % (label, value))


def main():
    report = json.load(open(sys.argv[1]))
    actions = load_jsonl(sys.argv[2])
    episodes = load_jsonl(sys.argv[3]) if len(sys.argv) > 3 else []

    print("== report ==")
    for k in ("update_index", "policy_version", "adapter_hash", "parent_adapter_hash",
              "sampled_from_adapter", "n_episodes", "n_turns", "n_failed",
              "n_discarded", "max_trace_share", "generation_config"):
        if k in report:
            show(k, json.dumps(report[k])[:200])

    if episodes:
        seen = {}
        for r in episodes:
            seen[r["episode_id"]] = r
        status = collections.Counter(r["status"] for r in seen.values())
        print("\n== episodes ==")
        show("distinct", len(seen))
        show("by status", dict(status))
        show("turns total", sum(r.get("num_turns", 0) for r in seen.values()))
        pv = set(r.get("policy_version") for r in seen.values() if r.get("policy_version"))
        show("policy_version(s)", pv or "not recorded per-episode")

    # --- the preregistered Okay measurements -------------------------------
    print("\n== reasoning boundaries and the Okay opener ==")
    openers = collections.Counter()
    okay_lp = []
    n_react = 0
    missing_reasoning = 0
    for a in actions:
        txt = a.get("reasoning_text")
        if txt is None:
            b64 = a.get("reasoning_b64") or a.get("action_b64")
            if b64:
                try:
                    txt = base64.b64decode(b64).decode("utf-8", "replace")
                except Exception:
                    txt = None
        if not txt or not txt.strip():
            missing_reasoning += 1
            continue
        n_react += 1
        first = txt.strip().split(None, 1)[0].strip(",.:;!?").lower()
        openers[first] += 1

        lps = a.get("behavior_logprobs") or []
        toks = a.get("sampled_token_texts") or a.get("sampled_tokens") or []
        for i, t in enumerate(toks[:6]):
            if isinstance(t, str) and t.strip().lower() == "okay" and i < len(lps):
                okay_lp.append(lps[i])
                break

    show("actions", len(actions))
    show("with reasoning", n_react)
    show("missing/empty reasoning", missing_reasoning)
    if n_react:
        okay_n = openers.get("okay", 0)
        show("begin with 'Okay'", "%d/%d = %.1f%%" % (okay_n, n_react, 100.0 * okay_n / n_react))
    print("  top openers:")
    for w, c in openers.most_common(8):
        print("      %-14s %d" % (w, c))
    if okay_lp:
        show("median logprob('Okay')", "%.5f  (n=%d)" % (statistics.median(okay_lp), len(okay_lp)))
        show("min / max", "%.5f / %.5f" % (min(okay_lp), max(okay_lp)))
    else:
        show("logprob('Okay')", "no Okay token located in first 6 positions")


main()
