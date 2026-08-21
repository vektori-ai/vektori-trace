"""Export a finished pass@k sweep as an SFT set for TRL.

Rejection-sampling SFT: keep the rollouts that passed, rebuild what the model
actually saw, emit chat messages. The token ids under `captures/` belong to
whichever model ran the sweep and mean nothing to a student with a different
vocab, so this path goes through text and lets the student's tokenizer decide.

The subtlety is context compaction. terminus-2 ran with
`max_input_tokens: 40448` and `enable_summarize=True`, so it summarized and
reset the conversation whenever the prompt passed ~32.4k — and with
`linear_history=False` it wrote every step into one `trajectory.json` anyway.
Read as-is, 35 of the 117 passing rollouts describe a conversation no model ever
saw. This script splits them back at the compaction boundaries, which is what
`linear_history=True` would have written at runtime.

    python scripts/sft_export_traces.py --run /data/vektori-out/dsv4-corpus60 \
        --run /data/vektori-out/dsv4-corpus60-b --out-dir /data/sft
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vektori_trace.dataset import turns_to_messages
from vektori_trace.mining.atif import (
    TrajectoryParseError,
    find_trajectory,
    parse_job_trajectory,
    parse_trajectory_file,
)
from vektori_trace.schema import Turn

# The reconstruction lives in `vektori_trace.compaction` now, so replay OPD and
# this exporter split at exactly the same places. Re-exported here because this
# script's behaviour must not change: v1/ck75 were trained on its output.
from vektori_trace.compaction import (  # noqa: E402
    BOUNDARY_MESSAGE,
    handoff_head,
    split_on_compaction,
)

# The compaction trigger: max_input_tokens 40448 minus the 8000-free-token
# threshold. No rebuilt segment can legitimately exceed it.
COMPACTION_TRIGGER = 40448 - 8000


@dataclass
class Segment:
    task: str
    jobs_dir: str
    rollout_index: int | None
    segment_index: int
    messages: list[dict] = field(default_factory=list)
    n_tokens: int = 0
    n_supervised: int = 0
    problems: list[str] = field(default_factory=list)


def all_rollouts(run_dir: Path) -> list[dict]:
    """Every recorded rollout. Prefer passk_log.jsonl — run A was killed before
    it wrote a report, but every rollout is in the log with its jobs_dir."""
    log = run_dir / "passk_log.jsonl"
    if log.exists():
        return [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]
    return json.loads((run_dir / "passk.json").read_text()).get("rollouts", [])


def select_rollouts(run_dir: Path, which: str = "passing") -> list[dict]:
    """Rollouts of one outcome class.

    Training only ever wanted `passing`. Phase 7 needs `failing` too: every one
    of the 26 held-out tasks has zero passing rollouts, so a prefix from outside
    the training distribution can only come from a rollout that failed. The
    prefix is still a real prompt — an outcome label says nothing about whether
    the conversation leading up to a turn is well formed.
    """
    rows = all_rollouts(run_dir)
    if which == "all":
        return rows
    want = which == "passing"
    return [r for r in rows if bool(r.get("passed")) is want]


def passing_rollouts(run_dir: Path) -> list[dict]:
    """Rollouts that passed."""
    return select_rollouts(run_dir, "passing")




def build_segments(rollouts: list[dict]) -> tuple[list[Segment], list[str]]:
    out: list[Segment] = []
    skipped: list[str] = []
    for r in rollouts:
        task = r["task"]
        jobs_dir = Path(r["jobs_dir"])
        if not jobs_dir.exists():
            skipped.append(f"{task}: {jobs_dir} is gone")
            continue
        try:
            turns = parse_job_trajectory(jobs_dir)
        except TrajectoryParseError as e:
            skipped.append(f"{task}: {e}")
            continue
        for i, seg_turns in enumerate(split_on_compaction(turns, jobs_dir)):
            messages = turns_to_messages(seg_turns)
            if not any(m.get("role") == "assistant" for m in messages):
                skipped.append(f"{task} seg{i}: no assistant turns")
                continue
            out.append(
                Segment(
                    task=task,
                    jobs_dir=str(jobs_dir),
                    rollout_index=r.get("rollout_index"),
                    segment_index=i,
                    messages=messages,
                )
            )
    return out, skipped


def training_chat_template(tok) -> str:
    """Qwen3's stock template has no `{% generation %}` markers, so
    `return_assistant_tokens_mask` would silently return all zeros. TRL swaps in
    a patched template when `assistant_only_loss=True`; use the same one here so
    the measured supervised fraction is the one training will actually see."""
    from trl.chat_template_utils import get_training_chat_template, has_generation_markers

    if has_generation_markers(tok.chat_template):
        return tok.chat_template
    return get_training_chat_template(tok)


def measure(segments: list[Segment], tok, template: str) -> None:
    for seg in segments:
        enc = tok.apply_chat_template(
            seg.messages,
            chat_template=template,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        seg.n_tokens = len(enc["input_ids"])
        seg.n_supervised = sum(enc["assistant_masks"])


def check_validity(segments: list[Segment], max_length: int) -> dict:
    """Cheap structural checks. Each records onto the segment and the report;
    none of them are fatal, because a surprise here is information."""
    for seg in segments:
        roles = [m["role"] for m in seg.messages]
        if seg.n_tokens > COMPACTION_TRIGGER:
            seg.problems.append(
                f"over compaction trigger ({seg.n_tokens} > {COMPACTION_TRIGGER}) — split may be wrong"
            )
        if seg.n_tokens > max_length:
            seg.problems.append(f"over max_length ({seg.n_tokens} > {max_length})")
        if seg.n_supervised == 0:
            seg.problems.append("no supervised tokens")
        for a, b in pairwise(roles):
            if a == "user" and b == "user":
                seg.problems.append("two consecutive user turns")
                break
        open_calls = {
            tc["id"] for m in seg.messages for tc in (m.get("tool_calls") or [])
        }
        orphans = [
            m["tool_call_id"]
            for m in seg.messages
            if m.get("tool_call_id") and m["tool_call_id"] not in open_calls
        ]
        if orphans:
            seg.problems.append(f"{len(orphans)} tool results with no matching call")

    failing = [s for s in segments if s.problems]
    return {
        "segments": len(segments),
        "with_problems": len(failing),
        "by_problem": dict(
            Counter(p.split(" (")[0] for s in failing for p in s.problems)
        ),
        "failures": [
            {
                "task": s.task,
                "jobs_dir": s.jobs_dir,
                "segment_index": s.segment_index,
                "n_tokens": s.n_tokens,
                "problems": s.problems,
            }
            for s in failing
        ],
    }


def stats(segments: list[Segment], max_length: int) -> dict:
    lengths = sorted(s.n_tokens for s in segments)
    sup = sum(s.n_supervised for s in segments)
    tot = sum(s.n_tokens for s in segments)
    per_task = Counter(s.task for s in segments)
    return {
        "segments": len(segments),
        "tasks": len(per_task),
        "rollouts": len({s.jobs_dir for s in segments}),
        "tokens": {
            "min": lengths[0],
            "median": statistics.median(lengths),
            "p90": lengths[int(0.9 * (len(lengths) - 1))],
            "max": lengths[-1],
            "total": tot,
        },
        "supervised_tokens": sup,
        "supervised_fraction": round(sup / tot, 4) if tot else 0.0,
        "truncated_at": {
            str(n): sum(1 for x in lengths if x > n)
            for n in (4096, 8192, 16384, 32768, max_length)
        },
        "segments_per_task": dict(per_task.most_common()),
        "task_skew": {
            "max": max(per_task.values()),
            "min": min(per_task.values()),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, action="append", required=True,
                    help="a pass@k output dir (repeatable)")
    ap.add_argument("--model", default="Qwen/Qwen3-14B", help="student, for its tokenizer")
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument("--per-task-cap", type=int, default=None,
                    help="max segments kept per task (default: all)")
    ap.add_argument("--stats", action="store_true", help="report and exit")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    rollouts: list[dict] = []
    for run in args.run:
        got = passing_rollouts(run)
        print(f"{run}: {len(got)} passing rollouts")
        rollouts.extend(got)

    segments, skipped = build_segments(rollouts)
    for s in skipped:
        print(f"  skip {s}", file=sys.stderr)
    print(f"\n{len(rollouts)} rollouts -> {len(segments)} segments "
          f"over {len({s.task for s in segments})} tasks")
    multi = Counter(s.jobs_dir for s in segments)
    print(f"rollouts by segment count: {dict(sorted(Counter(multi.values()).items()))}")

    if args.per_task_cap:
        kept, seen = [], Counter()
        for seg in segments:
            if seen[seg.task] >= args.per_task_cap:
                continue
            seen[seg.task] += 1
            kept.append(seg)
        print(f"per-task cap {args.per_task_cap}: {len(segments)} -> {len(kept)}")
        segments = kept

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    template = training_chat_template(tok)
    measure(segments, tok, template)

    report = check_validity(segments, args.max_length)
    summary = stats(segments, args.max_length)

    print(f"\ntokens:     min {summary['tokens']['min']}  "
          f"median {summary['tokens']['median']:.0f}  "
          f"p90 {summary['tokens']['p90']}  max {summary['tokens']['max']}")
    print(f"supervised: {summary['supervised_tokens']:,} of "
          f"{summary['tokens']['total']:,} "
          f"({100 * summary['supervised_fraction']:.1f}%)")
    print(f"truncated at {args.max_length}: "
          f"{summary['truncated_at'][str(args.max_length)]}/{len(segments)}")
    print(f"segments per task: max {summary['task_skew']['max']}, "
          f"min {summary['task_skew']['min']}")
    print(f"\nvalidity: {report['with_problems']}/{report['segments']} segments "
          f"have problems {report['by_problem'] or ''}")

    if args.stats or not args.out_dir:
        if not args.out_dir:
            print("\n(no --out-dir, nothing written)", file=sys.stderr)
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = args.out_dir / "sft_train.jsonl"
    with train.open("w") as f:
        for seg in segments:
            f.write(json.dumps({"task": seg.task, "messages": seg.messages}) + "\n")
    manifest = args.out_dir / "segments_manifest.jsonl"
    with manifest.open("w") as f:
        for seg in segments:
            f.write(json.dumps({
                "task": seg.task,
                "jobs_dir": seg.jobs_dir,
                "rollout_index": seg.rollout_index,
                "segment_index": seg.segment_index,
                "n_messages": len(seg.messages),
                "n_tokens": seg.n_tokens,
                "n_supervised": seg.n_supervised,
                "problems": seg.problems,
            }) + "\n")
    (args.out_dir / "export_stats.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "validity_report.json").write_text(json.dumps(report, indent=2))
    (args.out_dir / "skipped.json").write_text(json.dumps(skipped, indent=2))
    for p in (train, manifest, args.out_dir / "export_stats.json",
              args.out_dir / "validity_report.json"):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
