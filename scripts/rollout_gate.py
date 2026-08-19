#!/usr/bin/env python3
"""Judge one live harbor rollout against step 7 of `docs/SFT-SCRATCH-PLAN.md`.

Step 7 asks three things of a real trajectory: the JSON parses, the keys
execute, and the model does not fall into a parser loop. That is a different
question from the Phase 7 sweep, which replays a frozen prefix and grades one
raw completion string.

**What a trajectory can and cannot answer.** Harbor does not record the raw
completion. `trajectory.json` holds its *parsed decomposition*: `message` is the
analysis/plan prose, and `tool_calls` are synthetic (`call_0_1`,
`bash_command`) wrapping the `keystrokes`/`duration` that
`TerminusJSONPlainParser` pulled out of the response. So commands appearing in
a turn is positive evidence the parser accepted it — a fenced, prose-wrapped or
`<tool_call>`-enveloped response yields no commands at all. What it cannot show
is whether the accepted JSON was *bare*: `native_json` and the strict field
gates need the raw text, which only the capture proxy records. Grading the
decomposition as if it were a completion scores every correct turn as a total
format failure; this script used to do exactly that.

    python scripts/rollout_gate.py --jobs-dir <passk_jobs> --out gates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _trajectories(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("trajectory.json") if p.parent.name == "agent"]
        or list(root.rglob("trajectory.json"))
    )


def judge(traj_path: Path) -> dict:
    data = json.loads(traj_path.read_text())
    steps = data.get("steps", data) if isinstance(data, dict) else data
    turns = []
    for step in steps:
        if step.get("source") not in ("agent", "assistant"):
            continue
        calls = step.get("tool_calls") or []
        keystrokes = [
            (c.get("arguments") or {}).get("keystrokes", "") for c in calls
        ]
        metrics = step.get("metrics") or {}
        turns.append(
            {
                "turn": len(turns) + 1,
                # The parser produced executable commands, so it accepted the
                # response. No commands with a non-empty message is the
                # signature of a response the parser could not use.
                "accepted": bool(calls),
                "n_commands": len(calls),
                "keystrokes": keystrokes,
                "message_chars": len(step.get("message") or ""),
                "reasoning_chars": len(step.get("reasoning_content") or ""),
                "completion_tokens": metrics.get("completion_tokens"),
            }
        )

    # A parser loop is consecutive turns the parser could not use. One stumble
    # is recoverable and the corpus contains recoveries; two in a row is the
    # loop step 7 names.
    longest = cur = 0
    for t in turns:
        cur = 0 if t["accepted"] else cur + 1
        longest = max(longest, cur)

    return {
        "trajectory": str(traj_path),
        "turns": turns,
        "summary": {
            "n_turns": len(turns),
            "n_accepted": sum(1 for t in turns if t["accepted"]),
            "n_commands": sum(t["n_commands"] for t in turns),
            "longest_unparsed_run": longest,
            "parser_loop": longest >= 2,
            "turn_1_accepted": turns[0]["accepted"] if turns else None,
            "total_completion_tokens": sum(
                t["completion_tokens"] or 0 for t in turns
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jobs-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", default="ck84")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    paths = _trajectories(args.jobs_dir)
    if not paths:
        print(f"no trajectory.json under {args.jobs_dir}", file=sys.stderr)
        return 2

    report = {"checkpoint": args.checkpoint, "rollouts": []}
    ok = True
    for p in paths:
        r = judge(p)
        report["rollouts"].append(r)
        s = r["summary"]
        print(f"\n{p.parent.parent.name}")
        for t in r["turns"]:
            first = (t["keystrokes"][0] if t["keystrokes"] else "").strip()[:46]
            print(
                f"  turn {t['turn']:>3}  {'ACCEPTED' if t['accepted'] else 'UNPARSED'}"
                f"  cmds={t['n_commands']}  tok={t['completion_tokens']}"
                f"  {first!r}"
            )
        print(
            f"  -> {s['n_accepted']}/{s['n_turns']} turns accepted, "
            f"{s['n_commands']} commands executed, "
            f"parser loop: {'YES' if s['parser_loop'] else 'no'}"
        )
        ok = ok and s["n_accepted"] == s["n_turns"] and not s["parser_loop"]

    print(
        "\nnative_json / required_fields are NOT decided here — a trajectory "
        "holds no raw completion. Run behind `vektori-trace capture-proxy` to "
        "grade those."
    )
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
