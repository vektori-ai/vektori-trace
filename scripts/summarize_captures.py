#!/usr/bin/env python3
"""Did the run actually record what it claimed to?

Reads `token_captures.jsonl` (+ the `capture_failures.jsonl` ledger next to it)
and prints the numbers that decide whether the captures are trainable:

- how many completions were captured, and how many upstream calls produced no
  capture at all (the ledger -- silence there is the only proof of completeness);
- per-position top-K coverage and the *set of widths actually returned*, which is
  how a deployment quietly serving K=5 for a K=10 request becomes visible;
- whether the alternatives are keyed by `token_id`. String-keyed alternatives
  cannot be mapped back to ids without a local tokenizer, which is exactly the
  boundary-shift hazard `teacher_fireworks.py` exists to avoid.

    python3 scripts/summarize_captures.py vektori-out/teacher-dsv4/captures
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = Path(sys.argv[1])
    cap_path = d / "token_captures.jsonl"
    fail_path = d / "capture_failures.jsonl"

    if not cap_path.exists():
        print(f"NO CAPTURES: {cap_path} does not exist")
        return 1

    caps = []
    bad = 0
    for line in cap_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            caps.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1

    fails = []
    if fail_path.exists():
        for line in fail_path.read_text().splitlines():
            if line.strip():
                try:
                    fails.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    models = Counter(c.get("model") or "?" for c in caps)
    n_prompt = sum(len(c.get("prompt_token_ids") or []) for c in caps)
    n_gen = sum(len(c.get("token_ids") or []) for c in caps)

    positions = with_top = alts = alts_id = 0
    widths: Counter = Counter()
    detail_present = 0
    for c in caps:
        det = c.get("logprob_detail")
        if det is None:
            continue
        detail_present += 1
        for row in det:
            positions += 1
            top = row.get("top") or []
            if top:
                with_top += 1
                widths[len(top)] += 1
            for a in top:
                alts += 1
                if a.get("token_id") is not None:
                    alts_id += 1

    print(f"captures file        {cap_path}")
    print(f"completions captured {len(caps)}" + (f"  ({bad} unparseable lines)" if bad else ""))
    print(f"capture FAILURES     {len(fails)}" + ("" if not fails else "   <-- calls that reached the model but were not recorded"))
    print(f"models               {dict(models)}")
    print(f"prompt tokens        {n_prompt:,}")
    print(f"generated tokens     {n_gen:,}")
    print()
    print(f"with logprob_detail  {detail_present}/{len(caps)} completions")
    print(f"scored positions     {positions:,}")
    cov = (with_top / positions) if positions else 0.0
    print(f"positions with top-K {with_top:,}  ({cov:.1%})")
    print(f"top-K widths seen    {dict(sorted(widths.items()))}")
    print(f"alternatives         {alts:,}")
    idcov = (alts_id / alts) if alts else 0.0
    print(f"  keyed by token_id  {alts_id:,}  ({idcov:.1%})")
    print()

    ok = True
    if not caps:
        print("VERDICT: nothing captured.")
        ok = False
    if fails:
        print(f"WARN: {len(fails)} upstream call(s) produced no capture. First:")
        print("      " + json.dumps(fails[0])[:400])
        ok = False
    if positions and cov < 1.0:
        print(f"WARN: only {cov:.1%} of scored positions carry alternatives.")
        ok = False
    if len(widths) > 1:
        print(f"WARN: inconsistent top-K width across positions: {dict(widths)}")
        ok = False
    if alts and idcov < 1.0:
        print(f"WARN: {alts - alts_id:,} alternative(s) have no token_id -- "
              "string-keyed, not usable for id-level alignment.")
        ok = False
    print("VERDICT:", "captures look complete and trainable" if ok else "INCOMPLETE -- see warnings")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
