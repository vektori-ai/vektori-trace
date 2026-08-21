#!/usr/bin/env python3
"""Action-token lengths across stored trajectories — where the per-turn cap comes from.

Plan §7.1 says to use "the task-derived per-turn token cap already validated for
ck75" and not to return to the previous 256-token cap. This is how that number
is derived rather than guessed: tokenize every assistant action in the stored
corpus and read the cap off the tail of the distribution.

The finding this was written to produce (`docs/action-length-measurement.md`):
the median action is ~605 tokens, and the previous run's 256-token cap cut
**69%** of actions mid-sequence. A truncated action still aligns and still
yields a finite loss, so nothing downstream reveals it.

CPU only. Reads trajectories off disk, tokenizes locally, allocates nothing.

    .venv/bin/python scripts/measure_action_lengths.py \\
        --corpus /data/vektori-out/dsv4-corpus60 \\
        --corpus /data/vektori-out/dsv4-corpus60-b \\
        --out /data/action_lengths.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Caps worth reporting a truncation rate for. 256 is the previous run's default
#: and is here so its cost stays visible rather than becoming folklore.
CANDIDATE_CAPS = (256, 512, 1024, 2048, 4096, 8192)


def measure(corpora: list[Path], tokenizer, *, passing_only: bool, limit: int | None):
    from vektori_trace.evaluate.resume import assistant_tool_steps
    from vektori_trace.mining.atif import parse_job_trajectory
    from vektori_trace.replay_corpus import load_corpus

    traces = []
    for root in corpora:
        traces.extend(load_corpus(root, passing_only=passing_only))
    if limit is not None:
        traces = traces[:limit]

    lengths: list[int] = []
    per_trace: list[dict[str, Any]] = []
    for rec in traces:
        try:
            turns = parse_job_trajectory(rec.trial_dir)
        except Exception:
            continue
        by_index = {t.index: t for t in turns}
        got: list[int] = []
        for turn_index, _call in assistant_tool_steps(turns):
            turn = by_index.get(turn_index)
            if turn is None:
                continue
            # Both halves are sampled tokens. Dropping `thinking` would
            # understate the cap for any reasoning model by most of its output.
            text = (turn.thinking or "") + (turn.content or "")
            if text:
                got.append(len(tokenizer(text)["input_ids"]))
        lengths.extend(got)
        per_trace.append(
            {"trace_id": rec.trace_id, "task": rec.task, "n_actions": len(got)}
        )
    return lengths, per_trace, len(traces)


def summarize(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"n_actions": 0}
    s = sorted(lengths)
    n = len(s)

    def q(p: float) -> int:
        return s[min(n - 1, int(n * p))]

    return {
        "n_actions": n,
        "min": s[0],
        "median": q(0.5),
        "mean": round(statistics.mean(s), 1),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "p999": q(0.999),
        "max": s[-1],
        "truncation_by_cap": {
            str(c): {
                "n_truncated": sum(1 for x in s if x > c),
                "rate": round(sum(1 for x in s if x > c) / n, 4),
            }
            for c in CANDIDATE_CAPS
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True, type=Path)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-14B",
                    help="the STUDENT's tokenizer — the cap binds the student")
    ap.add_argument("--passing-only", action="store_true", default=True)
    ap.add_argument("--all-traces", dest="passing_only", action="store_false")
    ap.add_argument("--limit", type=int, default=None,
                    help="traces to read; omit for the whole corpus")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )
    lengths, per_trace, n_traces = measure(
        args.corpus, tok, passing_only=args.passing_only, limit=args.limit
    )
    rep = summarize(lengths)
    rep["n_traces"] = n_traces
    rep["tokenizer"] = args.tokenizer
    rep["passing_only"] = args.passing_only
    rep["per_trace"] = per_trace

    if not lengths:
        print("no actions measured — wrong corpus path, or no passing traces")
        return 1

    print(f"traces: {n_traces}   actions: {rep['n_actions']}")
    print(f"median {rep['median']}  mean {rep['mean']}  max {rep['max']}")
    for k in ("p90", "p95", "p99", "p999"):
        print(f"  {k}: {rep[k]}")
    print("\ntruncation by cap:")
    for cap, row in rep["truncation_by_cap"].items():
        note = "  <-- previous run's default" if cap == "256" else ""
        print(f"  {cap:>5}: {row['n_truncated']:>5} ({row['rate']:.1%}){note}")

    print(
        "\nPick a cap above p99.9. It is a loop guard, not a length budget: an "
        "uncapped degenerate sample generates until context exhaustion."
    )
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
