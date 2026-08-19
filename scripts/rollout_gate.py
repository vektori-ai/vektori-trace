#!/usr/bin/env python3
"""Grade every assistant turn of one harbor rollout against the Phase 7 gates.

Step 7 of `docs/SFT-SCRATCH-PLAN.md` asks a different question than the Phase 7
sweep does. The sweep replays frozen prefixes and grades one completion each;
this grades a *live* trajectory, where every turn after the first is conditioned
on the model's own previous output. That is the only place a format that holds
for one turn and collapses on turn 4 can show up.

Gates: format tier only, plus `orientation` on turn 1 (a cold start is an
orientation prefix by construction). Behaviour gates for edit/test are not
applied — Stage A was not asked to teach them.

    python scripts/rollout_gate.py --jobs-dir <passk_jobs>/stage1 --out gates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vektori_trace.evaluate.phase7 import grade


def _agent_steps(job_dir: Path) -> tuple[Path, list[dict]]:
    """The agent steps of one harbor trial, straight out of trajectory.json.

    Deliberately not `atif.parse_job_trajectory`: that normalises into Turns for
    mining, and the three fields this has to keep apart — `message` (the raw
    completion), `reasoning_content` (the think channel, which vLLM returns
    *outside* the message) and `tool_calls` (the v1 envelope) — are exactly what
    normalisation merges. Grading the merged text would score a tool-call
    envelope as if the model had emitted nothing.
    """
    candidates = [
        p for p in job_dir.rglob("trajectory.json") if p.parent.name == "agent"
    ] or list(job_dir.rglob("trajectory.json"))
    if not candidates:
        raise FileNotFoundError(f"no trajectory.json under {job_dir}")
    path = sorted(candidates)[0]
    data = json.loads(path.read_text())
    steps = data.get("steps", data) if isinstance(data, dict) else data
    return path, [s for s in steps if s.get("source") in ("agent", "assistant")]


def grade_job(job_dir: Path, *, checkpoint: str) -> list[dict]:
    _, steps = _agent_steps(job_dir)
    rows: list[dict] = []
    n = 0
    for step in steps:
        message = step.get("message") or ""
        tool_calls = step.get("tool_calls") or []
        reasoning = step.get("reasoning_content") or ""
        if not message.strip() and not tool_calls:
            continue
        n += 1
        res = grade(
            message,
            prefix_id=f"{job_dir.name}#turn{n}",
            checkpoint=checkpoint,
            # Turn 1 of a live rollout *is* the cold start. Later turns carry
            # whatever state the model has produced, so only format applies.
            category="orientation" if n == 1 else "rollout",
            suite="rollout",
            # The repo is on disk from turn 0; cloning is a real failure here
            # even before the terminal has shown it.
            git_present=True,
        )
        # The v1 regression put the action in an OpenAI tool_call instead of the
        # message body. harbor's terminus parser never sees it, so `message`
        # alone would grade as "emitted nothing" rather than "emitted the wrong
        # protocol". Fail the envelope gate on the field it actually lands in.
        if tool_calls:
            res.gates["no_legacy_envelope"] = False
            res.notes.append(f"{len(tool_calls)} tool_call(s) on the wire")
        rows.append(
            {
                "turn": n,
                "prefix_id": res.prefix_id,
                "passed": res.passed,
                "failed_gates": res.failed_gates,
                "gates": res.gates,
                "n_commands": res.n_commands,
                "first_command": res.first_command,
                "parser_error": res.parser_error,
                "parser_warning": res.parser_warning,
                "n_tool_calls": len(tool_calls),
                # vLLM returns the think channel outside the message when the
                # template opens it, so this is not len(res.think_body).
                "think_chars": len(reasoning) or len(res.think_body),
                "completion_chars": len(message),
                "notes": res.notes,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-dir", type=Path, required=True,
                    help="a harbor job dir, or any parent of one")
    ap.add_argument("--checkpoint", default="ck84")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    job_dirs = sorted(
        {p.parent.parent for p in args.jobs_dir.rglob("trajectory.json")}
    )
    if not job_dirs:
        print(f"no trajectory.json under {args.jobs_dir}", file=sys.stderr)
        return 2

    report: dict = {"checkpoint": args.checkpoint, "jobs": {}}
    all_rows: list[dict] = []
    for jd in job_dirs:
        rows = grade_job(jd, checkpoint=args.checkpoint)
        report["jobs"][str(jd)] = rows
        all_rows += rows
        print(f"\n{jd}")
        for r in rows:
            status = "PASS" if r["passed"] else "FAIL " + ",".join(r["failed_gates"])
            print(f"  turn {r['turn']:>3}  {status:<50} "
                  f"cmds={r['n_commands']} tc={r['n_tool_calls']} think={r['think_chars']}c "
                  f"{(r['first_command'] or '')[:40]!r}")

    n = len(all_rows)
    ok = sum(1 for r in all_rows if r["passed"])
    parser_errors = sum(1 for r in all_rows if r["parser_error"])
    report["summary"] = {
        "turns": n,
        "passed": ok,
        "parser_errors": parser_errors,
        # A parser loop is the failure step 7 names: the model gets a parse
        # error back and answers it with another unparseable action. Two in a
        # row is the signature, one is a recoverable stumble.
        "consecutive_parser_errors": max(
            (
                len(run)
                for run in _runs([bool(r["parser_error"]) for r in all_rows])
            ),
            default=0,
        ),
        "turn_1_passed": all_rows[0]["passed"] if all_rows else None,
    }
    print(f"\n{ok}/{n} turns clear the gates; {parser_errors} parser errors, "
          f"longest parser-error run {report['summary']['consecutive_parser_errors']}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0 if ok == n else 1


def _runs(flags: list[bool]) -> list[list[bool]]:
    """Maximal runs of True."""
    out: list[list[bool]] = []
    cur: list[bool] = []
    for f in flags:
        if f:
            cur.append(f)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
