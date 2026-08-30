"""Read a Tau2 eval Results file: reward, messages, and reasoning if present.

Usage: python scripts/pilotd_show_eval.py <results.json> [--full]

Reasoning lives in each assistant message's `raw_data`, which is where vLLM's
`reasoning_content` lands. `content` is what Tau2 renders and what the customer
sees -- an empty `content` with text in `raw_data` is the unclosed-<think>
deployment defect, not missing data.
"""
import json
import sys


def main():
    d = json.load(open(sys.argv[1]))
    full = "--full" in sys.argv
    print("top-level keys:", sorted(d.keys()))
    sims = d.get("simulations") or []
    print("simulations:", len(sims))
    if not sims:
        print("(no simulations -- the run failed before completing one)")
        for k in ("info", "tasks"):
            if k in d:
                print(" %s: %s" % (k, json.dumps(d[k])[:400]))
        return

    for s in sims:
        ri = s.get("reward_info") or {}
        print("\n=== task %s trial %s ===" % (s.get("task_id"), s.get("trial")))
        print("  reward           : %s" % ri.get("reward"))
        print("  termination      : %s" % s.get("termination_reason"))
        msgs = s.get("messages") or []
        print("  messages         : %d" % len(msgs))
        n_empty = 0
        n_reason = 0
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            content = m.get("content")
            raw = m.get("raw_data") or {}
            rc = None
            ch = (raw.get("choices") or [{}])[0] if isinstance(raw, dict) else {}
            if isinstance(ch, dict):
                rc = (ch.get("message") or {}).get("reasoning_content")
            if rc:
                n_reason += 1
            if not content and not m.get("tool_calls"):
                n_empty += 1
                print("    !! msg %d: EMPTY content, no tool calls; "
                      "reasoning_content=%s chars"
                      % (i, len(rc) if rc else 0))
            if full:
                print("    [%d] content=%r" % (i, (content or "")[:160]))
                if rc:
                    print("         reasoning=%r" % rc[:200])
                for tc in (m.get("tool_calls") or []):
                    print("         tool=%s" % tc.get("name"))
        print("  assistant msgs with reasoning_content: %d" % n_reason)
        print("  assistant msgs empty (deployment defect): %d" % n_empty)


main()
