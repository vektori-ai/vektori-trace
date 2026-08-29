"""Read-only: print a run's per-update roster from the frozen manifest.

Usage: python scripts/pilotd_plan.py <manifest.json> <update>
No spend, no GPU. Verifies the requested update's pairs are disjoint from
every earlier update's, which is what makes it a fresh on-policy batch.
"""
import json
import sys


def norm(p):
    if isinstance(p, dict):
        return (p.get("task_id", p.get("task")), p.get("seed"))
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        return (p[0], p[1])
    return (p, None)


def main():
    m = json.load(open(sys.argv[1]))
    u = int(sys.argv[2])
    print("okay_token_prediction:", json.dumps(m.get("okay_token_prediction"))[:500])
    print("require_reasoning:", m.get("require_reasoning"),
          "| parser:", m.get("parser_version"),
          "| projection:", m.get("projection_version"),
          "| score_algo:", m.get("score_algorithm"))
    print("max_action_tokens:", m.get("max_action_tokens"),
          "| max_turns:", m.get("max_turns"),
          "| temperature:", m.get("temperature"),
          "| thinking:", m.get("thinking_mode"))
    print("retry_policy:", json.dumps(m.get("retry_policy"))[:300])
    print("declared_exclusions:", json.dumps(m.get("declared_exclusions"))[:300])
    print("share_metrics:", json.dumps(m.get("share_metrics"))[:300])
    print("n_planned_episodes:", m.get("n_planned_episodes"),
          "| n_updates:", m.get("n_updates"),
          "| per_update:", m.get("episodes_per_update"))

    pbu = m.get("plans_by_update")
    if pbu is None:
        print("!! no plans_by_update")
        return
    keys = sorted(pbu, key=lambda k: int(k)) if isinstance(pbu, dict) else list(range(len(pbu)))
    cur = pbu[str(u)] if isinstance(pbu, dict) else pbu[u]
    print("\n== update %d: %d episodes ==" % (u, len(cur)))
    for p in cur:
        print("   ", norm(p))
    seen = set()
    for k in keys:
        if int(k) >= u:
            continue
        for p in (pbu[k] if isinstance(pbu, dict) else pbu[int(k)]):
            seen.add(norm(p))
    overlap = seen & set(norm(p) for p in cur)
    print("\nprior-update pairs: %d   overlap with update %d: %d" % (len(seen), u, len(overlap)))
    if overlap:
        print("!! OVERLAP", sorted(overlap))


main()
