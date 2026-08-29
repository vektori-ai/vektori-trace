"""Verify a dry-run's skips/exclusions match the manifest's declared policy.

Usage: python scripts/pilotd_verify_exclusions.py <manifest.json> <actions.jsonl>

The dry-run's own ACCEPTANCE line hardcodes "no payload skips", so it cannot
distinguish a DECLARED skip from an undeclared one, and it prints only a
truncated per-action listing. This re-derives every action's projection from
the captured bytes and checks that each skipped payload matches a declared
exclusion class -- which is the check the amended manifest actually asks for.
"""
import base64
import json
import sys

sys.path.insert(0, "/data/vektori-trace")

from vektori_trace.tau2.live_agent import split_generation

HERMES = ("<tool_call>", "</tool_call>")


def main():
    manifest = json.load(open(sys.argv[1]))
    declared = manifest.get("declared_exclusions", {})
    print("declared classes: %s" % sorted(declared.keys()))

    actions = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
    print("actions: %d" % len(actions))

    flagged = []
    for a in actions:
        raw = base64.b64decode(a["action_bytes_b64"]).decode("utf-8", "replace")
        try:
            reasoning, content, tools = split_generation(raw)
        except Exception as e:
            flagged.append((a["key"], "parser_refused", str(e)[:80], 0))
            continue
        if not reasoning:
            flagged.append((a["key"], "no_reasoning", "", 0))
            continue
        hit = [m for m in HERMES if m in reasoning]
        if hit:
            flagged.append((a["key"], "hermes_markup_inside_reasoning_span",
                            "markup=%s at %d, %d tool_calls, content=%db"
                            % (hit[0], reasoning.find(hit[0]), len(tools),
                               len(content.encode()) if content else 0),
                            len(a.get("action_token_ids") or [])))

    print("\n== actions whose reasoning payload cannot align ==")
    if not flagged:
        print("  none")
    for key, cls, detail, ntok in flagged:
        covered = "DECLARED" if cls in declared else "*** UNDECLARED ***"
        print("  %-28s %-38s %s" % (key, cls, covered))
        if detail:
            print("      %s  (%d tokens)" % (detail, ntok))

    undeclared = [f for f in flagged if f[1] not in declared]
    n_ep = len(set(a["episode_id"] for a in actions))
    ep_hit = len(set(k.split("@")[0] for k, _, _, _ in flagged))
    print("\n== against the declared stop rules ==")
    print("  affected actions   : %d / %d = %.2f%%"
          % (len(flagged), len(actions), 100.0 * len(flagged) / max(len(actions), 1)))
    print("  affected episodes  : %d / %d   (stop rule: > 2/8)" % (ep_hit, n_ep))
    print("  undeclared classes : %d" % len(undeclared))
    print("\n  VERDICT: %s" % ("all skips covered by declared policy"
                               if not undeclared else "UNDECLARED SKIP PRESENT"))


main()
