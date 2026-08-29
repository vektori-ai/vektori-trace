"""Read-only: verify a live-OPD rollout hold. No spend, no GPU.

Usage:
  python scripts/pilotd_hold.py <report.json> <actions.jsonl> <episodes.jsonl> \
      [--expect-adapter HASH]

Structural gate: sampling provenance (every episode on the expected adapter),
one policy version, complete roster, zero format failures/discards.

Preregistered `Okay` measurements: opener frequency over reasoning turns, the
median behaviour logprob of the `Okay` token, reasoning-boundary validity, the
missing/empty reasoning rate, and whether another opener replaced it.

Reasoning spans come from the repository's own parser (`split_generation` /
PARSER_VERSION), never a local re-implementation, so this measures what the
scorer would actually see.
"""
import argparse
import base64
import collections
import json
import statistics
import sys

sys.path.insert(0, "/data/vektori-trace")

from vektori_trace.tau2.live_agent import PARSER_VERSION, split_generation
from vektori_trace.tau2.live_projection import PROJECTION_VERSION
from vektori_trace.tau2.live_score import SCORE_ALGORITHM


def load_jsonl(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def show(label, value):
    print("  %-32s %s" % (label, value))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("actions")
    ap.add_argument("episodes")
    ap.add_argument("--expect-adapter", default=None)
    args = ap.parse_args()

    report = json.load(open(args.report))
    actions = load_jsonl(args.actions)
    episodes = load_jsonl(args.episodes)

    print("contract: parser=%s projection=%s score=%s"
          % (PARSER_VERSION, PROJECTION_VERSION, SCORE_ALGORITHM))

    print("\n== batch report ==")
    for k in ("ok", "requested", "trainable", "trainable_turns", "failed",
              "failed_turns", "discarded", "duplicates", "problems", "policy_versions"):
        if k in report:
            show(k, json.dumps(report[k])[:160])

    # --- structural: sampling provenance ----------------------------------
    seen = {}
    for r in episodes:
        seen[r["episode_id"]] = r
    print("\n== episodes (%d distinct) ==" % len(seen))
    adapters = collections.Counter(r.get("adapter_hash") for r in seen.values())
    pvs = collections.Counter(r.get("policy_version") for r in seen.values())
    gens = collections.Counter(r.get("gen_config_hash") for r in seen.values())
    status = collections.Counter(r.get("status") for r in seen.values())
    show("status", dict(status))
    show("adapter_hash", dict(adapters))
    show("policy_version", dict(pvs))
    show("gen_config_hash", dict(gens))
    show("require_reasoning", dict(collections.Counter(
        r.get("require_reasoning") for r in seen.values())))
    show("failed turns (sum)", sum(r.get("num_failed_turns", 0) for r in seen.values()))
    rewards = [r.get("reward") for r in seen.values() if r.get("reward") is not None]
    if rewards:
        show("reward", "%s/%s = %.3f" % (int(sum(rewards)), len(rewards),
                                         sum(rewards) / len(rewards)))

    ok = True
    if args.expect_adapter:
        bad = {h: n for h, n in adapters.items() if h != args.expect_adapter}
        if bad:
            ok = False
            print("  !! FAIL episodes not on expected adapter %s: %s"
                  % (args.expect_adapter, bad))
        else:
            print("  OK  all %d episodes sampled from %s"
                  % (len(seen), args.expect_adapter))
    if len(pvs) > 1:
        ok = False
        print("  !! FAIL multiple policy versions in one batch")
    if len(gens) > 1:
        ok = False
        print("  !! FAIL multiple generation configs in one batch")

    # --- the preregistered Okay measurements -------------------------------
    print("\n== reasoning boundaries and the Okay opener ==")
    openers = collections.Counter()
    okay_lp = []
    n_reasoning = 0
    n_empty = 0
    n_unparsed = 0
    for a in actions:
        raw = base64.b64decode(a["action_bytes_b64"]).decode("utf-8", "replace")
        try:
            text, _content, _tools = split_generation(raw)
        except Exception:
            n_unparsed += 1
            continue
        if not text or not text.strip():
            n_empty += 1
            continue
        n_reasoning += 1
        openers[text.strip().split(None, 1)[0].strip(",.:;!?").lower()] += 1

        # locate the Okay token by decoding per-token bytes
        tb = a.get("action_token_bytes_b64") or []
        lps = a.get("behavior_logprobs") or []
        for i, chunk in enumerate(tb[:6]):
            try:
                tok = base64.b64decode(chunk).decode("utf-8", "replace")
            except Exception:
                continue
            if tok.strip().lower() == "okay" and i < len(lps):
                okay_lp.append(lps[i])
                break

    show("actions", len(actions))
    show("parsed with reasoning", n_reasoning)
    show("empty/missing reasoning", n_empty)
    show("parser refused", n_unparsed)
    if n_empty or n_unparsed:
        ok = False
        print("  !! a reasoning-required run must have zero empty/refused")
    if n_reasoning:
        n_okay = openers.get("okay", 0)
        show("begin with 'Okay'", "%d/%d = %.1f%%"
             % (n_okay, n_reasoning, 100.0 * n_okay / n_reasoning))
    print("  openers:")
    for w, c in openers.most_common(10):
        print("      %-16s %3d  (%.1f%%)" % (w, c, 100.0 * c / max(n_reasoning, 1)))
    if okay_lp:
        show("median logprob('Okay')", "%.5f  (n=%d)" % (statistics.median(okay_lp), len(okay_lp)))
        show("min / max", "%.5f / %.5f" % (min(okay_lp), max(okay_lp)))
    else:
        show("logprob('Okay')", "not located in first 6 tokens")

    print("\n== VERDICT ==")
    print("  structural gate: %s" % ("PASS" if ok and report.get("ok") else "REVIEW"))


main()
