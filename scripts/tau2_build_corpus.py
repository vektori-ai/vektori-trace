#!/usr/bin/env python3
"""Build the Tau2 retail SFT corpus from raw simulations. CPU only.

Order matters and is the one in `docs/CLEA-QWEN3-4B-EXPERIMENT-V2.md` 4.2:
eligibility cannot follow the split, because train60 must be *made of* fully
eligible traces. But rendering eligibility needs the tokenizer, so the audit
runs before allocation and the render gate runs per row afterwards.

    raw inventory
      -> normalize + structural/policy audit
      -> reserve contaminated diagnostics for S16
      -> select train60 -> W30/C30
      -> export every genuine renderable decision
      -> freeze manifests + census

Nothing here authorizes a GPU run. It emits artifacts and numbers; V2's claims
are then updated *from* those numbers rather than predicted ahead of them.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vektori_trace.tau2.eligibility import audit
from vektori_trace.tau2.export import _serving_parser, build_row, census
from vektori_trace.tau2.normalize import (
    GreetingProvenanceError,
    MalformedTraceError,
    normalize_trace,
    select_trace,
)

# Already inspected and design-influencing: they may not enter the blind test.
CONTAMINATED = {"57", "73", "75", "93"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", default="/data/tau2/data/simulations/flash_retail*.json")
    ap.add_argument("--out", default="/data/tau2/artifacts")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-length", type=int, required=True,
                    help="pinned context; rows above it are rejected, never cut")
    ap.add_argument("--allow-structural-parser", action="store_true",
                    help="accept the structural round trip when the vLLM "
                         "serving parser is unavailable; the corpus is then "
                         "NOT serving-parity verified")
    ap.add_argument("--no-tokenize", action="store_true",
                    help="stop after the semantic stage (no tokenizer needed)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- stage 1: inventory + normalize ---------------------------------
    files = sorted(glob.glob(args.sims))
    if not files:
        print(f"no simulations matched {args.sims}", file=sys.stderr)
        return 2

    by_task: dict[str, list] = {}
    raw_by_key: dict[tuple, dict] = {}
    failures: list[dict] = []
    for f in files:
        d = json.load(open(f))
        for sim in d.get("simulations", []):
            if (sim.get("reward_info") or {}).get("reward") not in (1, 1.0):
                continue
            try:
                tr = normalize_trace(sim, os.path.basename(f))
            except (GreetingProvenanceError, MalformedTraceError) as e:
                failures.append({"task_id": str(sim.get("task_id")),
                                 "file": os.path.basename(f),
                                 "error": type(e).__name__, "detail": str(e)})
                continue
            by_task.setdefault(tr.task_id, []).append(tr)
            raw_by_key[(tr.task_id, tr.trace_hash)] = sim

    print(f"files={len(files)} tasks_with_passing_trace={len(by_task)} "
          f"normalization_failures={len(failures)}")
    for fl in failures[:10]:
        print("  FAIL", fl["task_id"], fl["error"], fl["detail"][:110])

    # ---- stage 2: audit EVERY trace, then select among the survivors ----
    # Selecting first would discard a task whose chosen trace fails policy while
    # a sibling passes. Five retail tasks have more than one passing trace.
    from vektori_trace.tau2.tools import (load_domain_tools,
                                      mutating_tool_names, tools_hash)
    schemas = load_domain_tools("retail")
    mutating = frozenset(mutating_tool_names(schemas))
    print(f"retail tool schema: {len(schemas)} tools, hash={tools_hash(schemas)}, "
          f"mutating={sorted(mutating)}")

    all_verdicts: dict[str, dict[str, Any]] = {}
    selected, verdicts = {}, {}
    for t, traces in by_task.items():
        scored = []
        for tr in traces:
            v = audit(tr, raw_by_key[(tr.task_id, tr.trace_hash)], mutating)
            all_verdicts.setdefault(t, {})[tr.trace_hash] = {
                "gates": v.gates,
                "diagnostic_confirmation": v.diagnostic_confirmation,
                "diagnostic_confirmation_notes": v.diagnostic_confirmation_notes,
                "evidence": v.evidence, "notes": v.notes}
            scored.append((tr, v))
        passing = [(tr, v) for tr, v in scored if v.eligible()]
        pool = passing or scored
        tr = select_trace([x[0] for x in pool])
        selected[t] = tr
        verdicts[t] = next(v for x, v in pool if x.trace_hash == tr.trace_hash)

    eligible = sorted([t for t, v in verdicts.items() if v.eligible()],
                      key=lambda x: int(x) if x.isdigit() else 0)
    structural = [t for t, v in verdicts.items() if v.structural and not v.eligible()]

    import collections as _c

    def _k(x): return int(x) if x.isdigit() else 0

    conf = _c.Counter(v.diagnostic_confirmation for v in verdicts.values())
    print(f"\ndiagnostic confirmation (NOT a gate): {dict(conf)}")
    print("  This classifier does not decide eligibility. An earlier version "
          "did and\n  rejected 64/73 traces; manual inspection showed it was "
          "rejecting clearly\n  compliant sequences because its proposal "
          "taxonomy conflated overlapping\n  mutation categories. These "
          "outcomes measure classifier uncertainty, not\n  trace "
          "non-compliance.")

    # Manual-review queue, in the priority order agreed 2026-08-24.
    queue: dict[str, list[str]] = {}
    for t, v in verdicts.items():
        why = v.needs_manual_review()
        if why:
            queue.setdefault(why, []).append(t)

    import random
    rng = random.Random(20260824)
    passes = sorted([t for t, v in verdicts.items()
                     if v.diagnostic_confirmation == "pass"], key=_k)
    nonmut = sorted([t for t, v in verdicts.items()
                     if v.diagnostic_confirmation == "not_applicable"], key=_k)
    if passes:
        queue["sample_heuristic_pass"] = rng.sample(passes, min(5, len(passes)))
    if nonmut:
        queue["control_non_mutation"] = rng.sample(nonmut, min(5, len(nonmut)))

    print("\nmanual-review queue:")
    for why in ("executed_after_decline", "heuristic_fail", "heuristic_uncertain",
                "sample_heuristic_pass", "control_non_mutation"):
        ts = sorted(queue.get(why, []), key=_k)
        if ts:
            print(f"  {why:26s} {len(ts):3d}  {ts[:14]}{' ...' if len(ts) > 14 else ''}")

    uncertain = sorted(queue.get("heuristic_uncertain", []), key=_k)

    gate_fail: dict[str, list[str]] = {}
    for t, v in verdicts.items():
        for g in v.failed():
            gate_fail.setdefault(g, []).append(t)

    print(f"\nselected one trace per task : {len(selected)}")
    print(f"fully eligible (struct+policy): {len(eligible)}")
    print(f"structural only               : {len(structural)}  {structural[:20]}")
    if gate_fail:
        print("\nfailed gates:")
        for g, ts in sorted(gate_fail.items(), key=lambda x: -len(x[1])):
            print(f"  {g:38s} {len(ts):3d}  {sorted(ts, key=lambda x:int(x) if x.isdigit() else 0)[:12]}")

    # ---- decision census (pre-render) -----------------------------------
    dec_counts = [len(selected[t].decisions) for t in eligible]
    if dec_counts:
        dec_counts.sort()
        def q(x): return dec_counts[min(len(dec_counts) - 1, int(len(dec_counts) * x))]
        print(f"\ngenuine decisions per eligible trace: min={dec_counts[0]} "
              f"p50={q(.5)} p90={q(.9)} max={dec_counts[-1]} total={sum(dec_counts)}")
        print(f"projected W30 rows if 30 tasks drawn at the mean: "
              f"{sum(dec_counts)/len(dec_counts)*30:.0f}")

    reserved = sorted(CONTAMINATED & set(eligible), key=lambda x: int(x))
    print(f"\ncontaminated diagnostics among eligible (reserved for S16): {reserved}")
    print(f"train60 candidates after reservation: {len(eligible) - len(reserved)}")

    if args.no_tokenize:
        print("\n--no-tokenize: stopping before the render gate")
        return 0

    # ---- stage 3: render gate over every eligible decision --------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    policy = tools = None
    for f in files:
        info = (json.load(open(f)).get("info") or {})
        env = info.get("environment_info") or {}
        if env.get("policy"):
            policy = env["policy"]
            break
    if policy is None:
        print("no environment policy found in any simulation info", file=sys.stderr)
        return 2

    parser = _serving_parser(tok)
    if parser is None and not args.allow_structural_parser:
        print("the vLLM serving tool parser is unavailable in this environment. "
              "Re-run where vLLM is installed, or pass --allow-structural-parser "
              "to accept a structural check and record that the corpus is NOT "
              "serving-parity verified.", file=sys.stderr)
        return 2
    parser_mode = "serving" if parser is not None else "structural"
    print(f"tool round-trip parser: {parser_mode}")

    # Rendering eligibility is a property of the whole trace, not of one row.
    # Section 5 requires the exact rendered history to fit; a trace that loses
    # any decision to over-length or a failed round trip is ineligible, and none
    # of its rows enter the corpus. Partial trajectories would let the census and
    # the eligibility report disagree about the same task.
    rows, rejected, render_failed = [], [], {}
    for t in eligible:
        trace_rows, why = [], []
        for d in selected[t].decisions:
            try:
                r = build_row(d, tok, max_length=args.max_length,
                              system=policy, tools=schemas, parser=parser,
                              require_parser=not args.allow_structural_parser)
            except Exception as e:
                why.append({"position": d.position, "reason": type(e).__name__,
                            "detail": str(e)[:300]})
                continue
            if r is None:
                why.append({"position": d.position, "reason": "over_length",
                            "detail": f"> {args.max_length}"})
                continue
            trace_rows.append(r)
        if why:
            render_failed[t] = why
            rejected.extend({"task_id": t, **w} for w in why)
        else:
            rows.extend(trace_rows)

    renderable = [t for t in eligible if t not in render_failed]
    print(f"\nrender gate: {len(renderable)}/{len(eligible)} traces fully renderable; "
          f"{len(render_failed)} rejected whole")
    for t, why in list(render_failed.items())[:8]:
        print(f"  task {t}: {len(why)} decision(s) failed, e.g. {why[0]['reason']}")

    rep = census(rows, rejected)
    print(f"\n=== RENDERED CORPUS (all eligible tasks, max_length={args.max_length}) ===")
    for k in ("n_rows", "n_tasks", "n_rejected", "rows_per_task",
              "total_tokens", "target_tokens", "over_4096", "over_8192"):
        print(f"  {k:26s} {rep[k]}")
    print(f"  action_types               {rep['action_types']}")
    if rejected:
        print("\n  rejections by reason:")
        import collections
        for r, c in collections.Counter(x["reason"] for x in rejected).most_common():
            print(f"    {r:24s} {c}")

    rep["parser_mode"] = parser_mode
    rep["tools_hash"] = tools_hash(schemas)
    rep["max_length"] = args.max_length
    rep["render_failed_tasks"] = {t: w for t, w in render_failed.items()}
    rep["fully_eligible_task_ids"] = renderable
    json.dump(rep, open(os.path.join(args.out, "data_census.json"), "w"), indent=1)

    # The semantic corpus: the messages themselves, so the rows can be rebuilt,
    # inspected and diffed without the tokenizer.
    with open(os.path.join(args.out, "rows.semantic.jsonl"), "w") as fh:
        for t in renderable:
            for d in selected[t].decisions:
                fh.write(json.dumps({
                    "task_id": d.task_id, "position": d.position,
                    "message_index": d.message_index,
                    "action_type": d.action_type, "tool_names": d.tool_names,
                    "prompt": d.prompt, "target": d.target,
                    "semantic_hash": d.semantic_hash(),
                }, default=str) + "\n")

    # The training corpus: input_ids and labels, which is what the trainer eats.
    with open(os.path.join(args.out, "rows.tokenized.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps({
                "task_id": r.task_id, "position": r.position,
                "action_type": r.action_type, "tool_names": r.tool_names,
                "semantic_hash": r.semantic_hash,
                "input_ids": r.input_ids, "labels": r.labels,
                "attention_mask": [1] * len(r.input_ids),
            }) + "\n")

    # Eligibility is written last, because rendering is one of its gates.
    json.dump(
        {"n_files": len(files),
         "source_hashes": {os.path.basename(f): sha256_file(f) for f in files},
         "tools_hash": tools_hash(schemas), "parser_mode": parser_mode,
         "max_length": args.max_length,
         "normalization_failures": failures,
         "n_tasks_passing": len(by_task),
         "n_structurally_and_policy_eligible": len(eligible),
         "n_fully_eligible": len(renderable),
         "fully_eligible_task_ids": renderable,
         "render_failed": render_failed,
         "diagnostic_confirmation_outcomes": dict(conf),
         "diagnostic_confirmation_note": (
             "NOT an eligibility gate. An earlier version gated on this "
             "classifier and rejected 64/73 traces; manual inspection showed it "
             "was rejecting clearly compliant sequences because its proposal "
             "taxonomy conflated overlapping mutation categories. These outcomes "
             "measure classifier uncertainty, not trace non-compliance. "
             "Confirmation is evaluated primarily through Tau2's official "
             "communication checks, supplemented by targeted manual review."),
         "manual_review_queue": {k: sorted(v, key=_k) for k, v in queue.items()},
         "structural_only": sorted(structural, key=lambda x: int(x) if x.isdigit() else 0),
         "failed_gates": {g: sorted(ts, key=lambda x: int(x) if x.isdigit() else 0)
                          for g, ts in gate_fail.items()},
         "all_trace_verdicts": all_verdicts,
         "per_task": {t: {"trial": tr.trial, "seed": tr.seed,
                          "trace_hash": tr.trace_hash,
                          "source_file": tr.source_file,
                          "n_decisions": len(tr.decisions),
                          "gates": verdicts[t].gates,
                          "diagnostic_confirmation": verdicts[t].diagnostic_confirmation,
                          "diagnostic_confirmation_notes": verdicts[t].diagnostic_confirmation_notes,
                          "manual_confirmation": verdicts[t].manual_confirmation,
                          "evidence": verdicts[t].evidence,
                          "unaudited": verdicts[t].unaudited,
                          "notes": verdicts[t].notes}
                      for t, tr in selected.items()}},
        open(os.path.join(args.out, "eligibility_report.json"), "w"), indent=1)

    hashes = {}
    for fn in ("eligibility_report.json", "data_census.json",
               "rows.semantic.jsonl", "rows.tokenized.jsonl"):
        hashes[fn] = sha256_file(os.path.join(args.out, fn))
    json.dump(hashes, open(os.path.join(args.out, "artifact_hashes.json"), "w"),
              indent=1)

    print(f"\nwrote {args.out}/: eligibility_report.json, data_census.json, "
          "rows.semantic.jsonl, rows.tokenized.jsonl, artifact_hashes.json")
    for k, v in hashes.items():
        print(f"  {k:28s} {v[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
