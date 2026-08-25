#!/usr/bin/env python3
"""Per-domain rendered-context census. CPU only, no GPU, no Modal.

The 16,384 pin was measured on retail alone. Airline and telecom have their own
policies, tool schemas, conversation lengths and observation sizes, so retail's
number is not transferable. This renders each domain with its *own* policy and
tools through the pinned Qwen template and reports retention at 8K/12K/16K/32K.

Retention is reported by **complete trace**, not by row: a partial trajectory
distorts the corpus, so a trace that loses any decision is lost entirely.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vektori_trace.tau2.export import build_row
from vektori_trace.tau2.normalize import (
    GreetingProvenanceError, MalformedTraceError, normalize_trace, select_trace,
)
from vektori_trace.tau2.tools import load_domain_tools, tools_hash

CAPS = (8192, 12288, 16384, 32768)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="retail,airline,telecom")
    ap.add_argument("--sims", default="/data/tau2/data/simulations")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="/data/tau2/artifacts/context_census.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    report = {}
    for domain in args.domains.split(","):
        domain = domain.strip()
        files = sorted(glob.glob(os.path.join(args.sims, f"flash_{domain}*.json")))
        if not files:
            print(f"{domain}: no simulations found, skipping")
            continue

        schemas = load_domain_tools(domain)
        policy = None
        by_task = {}
        for f in files:
            d = json.load(open(f))
            env = (d.get("info") or {}).get("environment_info") or {}
            if policy is None and env.get("policy"):
                policy = env["policy"]
            for sim in d.get("simulations", []):
                if (sim.get("reward_info") or {}).get("reward") not in (1, 1.0):
                    continue
                try:
                    tr = normalize_trace(sim, os.path.basename(f))
                except (GreetingProvenanceError, MalformedTraceError):
                    continue
                by_task.setdefault(tr.task_id, []).append(tr)

        if policy is None or not by_task:
            print(f"{domain}: no policy or no passing traces, skipping")
            continue

        sel = {t: select_trace(v) for t, v in by_task.items()}
        policy_tokens = len(tok(policy, add_special_tokens=False)["input_ids"])
        schema_tokens = len(tok(json.dumps(schemas), add_special_tokens=False)["input_ids"])

        # Render once at the largest cap; every smaller cap is then a threshold.
        lengths: dict[str, list[int]] = {}
        for t, tr in sel.items():
            ls = []
            for d in tr.decisions:
                r = build_row(d, tok, max_length=10**9, system=policy,
                              tools=schemas, check_tools=False)
                ls.append(r.n_total if r else 10**9)
            lengths[t] = ls

        allrows = [n for ls in lengths.values() for n in ls]
        allrows.sort()

        def q(p):
            return allrows[min(len(allrows) - 1, int(len(allrows) * p))]

        dom = {
            "n_tasks": len(sel),
            "n_rows": len(allrows),
            "policy_tokens": policy_tokens,
            "tool_schema_tokens": schema_tokens,
            "tools_hash": tools_hash(schemas),
            "n_tools": len(schemas),
            "row_tokens": {"min": allrows[0], "p50": q(.5), "p90": q(.9),
                           "p99": q(.99), "max": allrows[-1]},
            "retention": {},
        }
        for cap in CAPS:
            traces_kept = [t for t, ls in lengths.items() if all(n <= cap for n in ls)]
            rows_kept = sum(1 for n in allrows if n <= cap)
            dom["retention"][str(cap)] = {
                "complete_traces": len(traces_kept),
                "complete_traces_pct": round(100 * len(traces_kept) / len(sel), 1),
                "rows_if_partial_allowed": rows_kept,
                "rows_from_complete_traces": sum(len(lengths[t]) for t in traces_kept),
            }
        report[domain] = dom

        print(f"\n=== {domain.upper()} ===")
        print(f"  tasks={dom['n_tasks']} rows={dom['n_rows']} "
              f"tools={dom['n_tools']} policy={policy_tokens}tok "
              f"schema={schema_tokens}tok")
        print(f"  row tokens: {dom['row_tokens']}")
        print(f"  {'cap':>7}  {'traces':>14}  {'rows(complete)':>15}")
        for cap in CAPS:
            r = dom["retention"][str(cap)]
            print(f"  {cap:>7}  {r['complete_traces']:>4}/{dom['n_tasks']:<3} "
                  f"({r['complete_traces_pct']:>5.1f}%)  "
                  f"{r['rows_from_complete_traces']:>15}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
