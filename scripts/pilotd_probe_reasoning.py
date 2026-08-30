"""Find where reasoning actually lives in a Tau2 Results file.

Usage: python scripts/pilotd_probe_reasoning.py <results.json>

Walks each assistant message and reports every key present, plus any nested
key whose name mentions reasoning/thinking, rather than assuming one path.
"""
import json
import sys


def walk(o, path=""):
    """Yield (path, value) for keys mentioning reasoning/thinking/think."""
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{path}.{k}" if path else k
            if any(w in k.lower() for w in ("reason", "think")):
                yield p, v
            yield from walk(v, p)
    elif isinstance(o, list):
        for i, v in enumerate(o[:3]):
            yield from walk(v, f"{path}[{i}]")


def main():
    d = json.load(open(sys.argv[1]))
    sims = d.get("simulations") or []
    print("simulations: %d" % len(sims))
    if not sims:
        return
    s = sims[0]
    msgs = s.get("messages") or []
    asst = [m for m in msgs if m.get("role") == "assistant"]
    print("assistant messages in sim 0: %d" % len(asst))
    if not asst:
        return

    m = asst[0]
    print("\n== first assistant message: top-level keys ==")
    for k in sorted(m):
        v = m[k]
        kind = type(v).__name__
        prev = json.dumps(v)[:90] if not isinstance(v, (dict, list)) else f"<{kind}>"
        print("  %-16s %-8s %s" % (k, kind, prev))

    raw = m.get("raw_data")
    if isinstance(raw, dict):
        print("\n== raw_data keys ==")
        print("  %s" % sorted(raw))
        ch = (raw.get("choices") or [{}])
        if ch and isinstance(ch[0], dict):
            print("  choices[0] keys: %s" % sorted(ch[0]))
            mm = ch[0].get("message")
            if isinstance(mm, dict):
                print("  choices[0].message keys: %s" % sorted(mm))
    else:
        print("\n!! raw_data is %s, not a dict" % type(raw).__name__)

    print("\n== any reasoning/thinking key anywhere in this message ==")
    found = list(walk(m))
    if not found:
        print("  none")
    for p, v in found:
        n = len(v) if isinstance(v, str) else "-"
        prev = (v[:120] if isinstance(v, str) else json.dumps(v)[:120])
        print("  %s  (len=%s)\n    %r" % (p, n, prev))

    print("\n== content of each assistant message ==")
    for i, a in enumerate(asst):
        c = a.get("content")
        tc = a.get("tool_calls") or []
        print("  [%d] content=%s tools=%d  %r"
              % (i, len(c) if c else 0, len(tc), (c or "")[:80]))


main()
