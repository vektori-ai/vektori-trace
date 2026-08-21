#!/usr/bin/env python3
"""Archive a reproducible census of the replay candidate pool.

Two questions, one artifact:

1. **Context budget.** What does each candidate prefix cost in *rendered Qwen
   tokens*, and how many clear `prefix_tokens + max_new_tokens <= max_model_len`?
   Character counts are not a proxy — the dry run used to report
   `approx_prefix_chars` and that is not a number an approval can rest on.
2. **Compaction boundaries.** Where are they, how many per trace, and what
   shape does the trajectory take immediately after one?

Written as a checked-in script rather than a scratchpad probe because a number
quoted from a vanished shell is not evidence. The output records corpus paths,
tokenizer and renderer hashes, the code commit, quantiles, and every
per-candidate measurement, so the census can be re-derived and diffed.

    .venv/bin/python scripts/census_replay_prefixes.py --out /data/prefix_token_census.json
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_CORPUS = ["/data/vektori-out/dsv4-corpus60", "/data/vektori-out/dsv4-corpus60-b"]

#: Markers Terminus uses for the handoff message that follows a boundary. Used
#: only to *classify* the post-boundary shape in this report; nothing in the
#: training path keys off prose.
HANDOFF_MARKERS = (
    "Here are the answers the other agent provided",
    "You are picking up work from a previous AI agent",
    "The next agent has a few questions",
)


def _renderer_sha(*fns) -> str:
    """Hash the source of the functions that turn turns into a prompt.

    The rendered prefix depends on this code as much as on the tokenizer, so a
    census that pinned only the tokenizer would call two different renderings
    the same measurement.
    """
    h = hashlib.sha256()
    for fn in fns:
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()


def _tokenizer_sha(tok) -> str | None:
    try:
        vocab = tok.get_vocab()
    except Exception:
        return None
    blob = json.dumps(sorted(vocab.items()), separators=(",", ":")).encode()
    h = hashlib.sha256(blob)
    tpl = getattr(tok, "chat_template", None)
    if tpl:
        h.update(tpl.encode())
    return h.hexdigest()


def census_compaction(corpus_roots: list[str]) -> dict[str, Any]:
    """Boundary inventory, using the same detector the runtime uses."""
    from vektori_trace.compaction import CompactionError, boundaries_from_raw

    n_traces = 0
    boundaries = 0
    shapes: Counter = Counter()
    per_trace: Counter = Counter()
    rows: list[dict[str, Any]] = []

    for root in corpus_roots:
        for traj in sorted(Path(root).rglob("trajectory.json")):
            if traj.parent.name != "agent":
                continue
            try:
                steps = json.load(traj.open())["steps"]
            except Exception:
                continue
            # Exactly the production predicate. Duplicating the condition here
            # let the census count records the runtime detector would ignore,
            # so the two could disagree about how many boundaries exist.
            try:
                found = boundaries_from_raw(traj)
            except CompactionError:
                shapes["unreadable_trajectory"] += 1
                continue
            if not found:
                continue
            n_traces += 1
            per_trace[len(found)] += 1
            for b in found:
                boundaries += 1
                if b.meta.get("next_is_handoff"):
                    shapes["user_handoff_inlined"] += 1
                elif b.meta.get("next_source") is None:
                    shapes["boundary_is_last_step"] += 1
                else:
                    shapes[f"next_is_{b.meta.get('next_source')}_unrecognised"] += 1
                rows.append({"trial": str(traj.parent.parent), **b.to_dict(),
                             **b.meta})

    return {
        "traces_with_compaction": n_traces,
        "total_boundaries": boundaries,
        "boundaries_per_trace": dict(sorted(per_trace.items())),
        "shape_after_boundary": dict(shapes),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", default=None)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-14B")
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--max-new-tokens", type=int, default=9216)
    ap.add_argument("--out", default="/data/prefix_token_census.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    roots = args.corpus or DEFAULT_CORPUS

    import transformers

    from vektori_trace.dataset import turns_to_messages
    from vektori_trace.opd_manifest import require_commit
    from vektori_trace.replay_context import measure_prefix, summarize_budgets
    from vektori_trace.replay_corpus import candidates_from_traces, load_corpus

    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    traces = []
    for r in roots:
        traces.extend(load_corpus(Path(r), passing_only=True))
    cands = candidates_from_traces(traces)
    if args.limit:
        cands = cands[: args.limit]
    print(f"traces={len(traces)} candidates={len(cands)}", file=sys.stderr)

    rows: list[dict[str, Any]] = []
    budgets = []
    for i, c in enumerate(cands):
        if i % 500 == 0:
            print(f"  {i}/{len(cands)}", file=sys.stderr)
        try:
            msgs = turns_to_messages(c.prefix_turns)
            if not msgs:
                rows.append({"prefix_id": c.prefix_id, "error": "no messages"})
                continue
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            b = measure_prefix(
                c.prefix_id, prompt, tok,
                max_new_tokens=args.max_new_tokens,
                max_model_len=args.max_model_len,
            )
        except Exception as e:
            rows.append({"prefix_id": c.prefix_id, "error": f"{type(e).__name__}: {e}"})
            continue
        budgets.append(b)
        rows.append(
            {
                "prefix_id": c.prefix_id,
                "task": c.task,
                "trace_id": c.trace_id,
                "step": c.step_index,
                "n_turns": len(c.prefix_turns),
                "post_compaction": c.post_compaction,
                **b.to_dict(),
            }
        )

    toks = sorted(b.prefix_tokens for b in budgets)
    fitting = [r for r in rows if r.get("fits")]

    def q(p: float):
        return toks[min(len(toks) - 1, int(len(toks) * p))] if toks else None

    report = {
        "provenance": {
            "corpus_roots": roots,
            "tokenizer": args.tokenizer,
            "tokenizer_sha256": _tokenizer_sha(tok),
            "renderer_sha256": _renderer_sha(turns_to_messages),
            "vektori_trace_commit": require_commit(),
            "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "budget_for_prefix": args.max_model_len - args.max_new_tokens,
        },
        "context": {
            **summarize_budgets(budgets),
            "n_candidates": len(cands),
            "n_measured": len(budgets),
            "n_errors": len(rows) - len(budgets),
            "quantiles": {
                "min": toks[0] if toks else None,
                "p25": q(0.25), "median": q(0.5), "p75": q(0.75),
                "p90": q(0.90), "p99": q(0.99),
                "max": toks[-1] if toks else None,
                "mean": round(statistics.mean(toks)) if toks else None,
            },
            "overflow_rate": (
                round(1 - len(fitting) / len(budgets), 4) if budgets else None
            ),
            "distinct_tasks_fitting": len({r["task"] for r in fitting}),
            "distinct_traces_fitting": len({r["trace_id"] for r in fitting}),
        },
        "compaction": census_compaction(roots),
        "rows": rows,
    }
    # Per-candidate rows are the bulk; keep them out of the printed summary.
    summary = {k: v for k, v in report.items() if k != "rows"}
    summary["compaction"] = {
        k: v for k, v in report["compaction"].items() if k != "rows"
    }
    print(json.dumps(summary, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
