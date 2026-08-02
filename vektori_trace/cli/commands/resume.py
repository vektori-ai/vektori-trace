"""`resume-check` and `bisect` — sandbox replay over a manifest."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from .._shared import _load_traces


def _sandbox_for_task(task_dir: Path, platform: str = "linux/amd64"):
    """Start a fresh container for one task's environment image.

    Step A replays into a *fresh* container per prefix — reusing one would let an
    earlier probe's writes leak into the next, which is the desync the assertion
    is supposed to detect.
    """
    import tempfile

    from ...mining.bootstrap.docker import DockerSandbox

    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        raise FileNotFoundError(f"no environment/Dockerfile in {task_dir}")
    image = ""
    for line in dockerfile.read_text().splitlines():
        if line.strip().upper().startswith("FROM "):
            image = line.split(None, 1)[1].strip()
            break
    if not image:
        raise ValueError(f"no FROM line in {dockerfile}")
    marker = Path(tempfile.mkdtemp(prefix="r2e-resume-"))
    (marker / ".keep").write_text("")
    return DockerSandbox.start(base_image=image, repo_dir=marker, platform=platform)


def cmd_resume_check(args: argparse.Namespace) -> int:
    """Step A — replay trajectory prefixes into fresh containers, report desync.

    PLAN.md calls this the spike that gates the design: "High desync makes ReOPD
    mandatory rather than merely preferable — that must be known in week 1."
    """
    from ...resume import assistant_tool_steps, measure_desync_rate, replay_prefix

    traces = _load_traces(Path(args.manifest))
    if args.model:
        traces = [t for t in traces if t.model == args.model]
    traces = [t for t in traces if t.task]
    if not traces:
        print("error: no traces with a task id in the manifest", file=sys.stderr)
        return 2
    if args.limit:
        traces = traces[: args.limit]

    tasks_dir = Path(args.tasks_dir)
    results = []
    per_trace: list[dict[str, Any]] = []
    for trace in traces:
        task_dir = tasks_dir / str(trace.task)
        if not task_dir.is_dir():
            print(f"  skip {trace.run_id}: no task dir {task_dir}", file=sys.stderr)
            continue
        n_steps = len(assistant_tool_steps(trace.turns))
        if n_steps == 0:
            continue
        # Probe prefixes across the trajectory rather than only the full one:
        # desync is a function of how far in you replay.
        fractions = [f for f in (0.25, 0.5, 1.0) if int(n_steps * f) > 0]
        for frac in fractions:
            T = int(n_steps * frac) - 1
            sandbox = None
            try:
                sandbox = _sandbox_for_task(task_dir, platform=args.platform)
                res = replay_prefix(trace.turns, T, sandbox, hard_fail=False)
            except Exception as e:
                print(f"  {trace.run_id} T={T}: infra failure: {e}", file=sys.stderr)
                continue
            finally:
                if sandbox is not None:
                    with contextlib.suppress(Exception):
                        sandbox.cleanup()
            results.append(res)
            per_trace.append(
                {
                    "run_id": trace.run_id,
                    "task": trace.task,
                    "T": T,
                    "steps": n_steps,
                    "verified": res.verified,
                    "consistent": res.consistent,
                    "checks_run": res.checks_run,
                    "unsupported_skipped": res.unsupported_skipped,
                    "readonly_skipped": res.readonly_skipped,
                    "desync_reason": res.desync_reason,
                }
            )

    if not results:
        print("error: no prefix was replayed", file=sys.stderr)
        return 1
    stats = measure_desync_rate(results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "resume-check.json"
    path.write_text(json.dumps({"stats": stats, "replays": per_trace}, indent=2) + "\n")
    print(f"resume check: {path}")
    print(json.dumps(stats, indent=2))
    if stats["verified"] == 0:
        print(
            "warning: nothing was verifiable — the desync rate is undefined, not 0",
            file=sys.stderr,
        )
    return 0


def cmd_bisect(args: argparse.Namespace) -> int:
    """Step D — verifier-guided bisection to the forking step."""
    import shlex
    import subprocess

    from ...intervene import bisect_forking_step, make_resume_fn

    # The teacher must continue *from the replayed prefix*. `validity.run_trial`
    # starts its own fresh container and cannot be handed one, so using it here
    # would run the teacher from scratch and ignore T entirely — every probe
    # would return the same answer and the located "forking step" would be an
    # artifact of the search, not of the trajectory. Until a scaffold that
    # accepts a seeded container exists, the continuation backend is supplied by
    # the caller: a command receiving the task dir and a JSON prefix, exiting 0
    # when the mined verifier passes.
    if not args.continuation_cmd:
        print(
            "error: --continuation-cmd is required. Bisection needs a teacher "
            "continuation that starts from the replayed prefix at step T; there "
            "is no harbor entrypoint for that yet, so the backend is external. "
            "The command receives {task_dir} and {prefix_json} and must exit 0 "
            "iff the verifier passes.",
            file=sys.stderr,
        )
        return 2

    traces = _load_traces(Path(args.manifest))
    traces = [t for t in traces if t.task and t.outcome == "loss"]
    if args.model:
        traces = [t for t in traces if t.model == args.model]
    if not traces:
        print("error: no failed traces with task ids in the manifest", file=sys.stderr)
        return 2
    if args.limit:
        traces = traces[: args.limit]

    tasks_dir = Path(args.tasks_dir)
    jobs = Path(args.out) / "bisect_jobs"
    rows: list[dict[str, Any]] = []
    for trace in traces:
        task_dir = tasks_dir / str(trace.task)
        if not task_dir.is_dir():
            continue

        def continue_with_teacher(turns, T, _task_dir=task_dir, _trace=trace):
            from ...resume import assistant_tool_steps

            jobs.mkdir(parents=True, exist_ok=True)
            prefix_path = jobs / f"{_trace.run_id}-T{T}.json"
            prefix_path.write_text(
                json.dumps(
                    {
                        "task": str(_trace.task),
                        "run_id": _trace.run_id,
                        "T": T,
                        "n_action_steps": len(assistant_tool_steps(turns)),
                        "teacher_model": args.teacher_model,
                        "prefix_turns": [
                            {
                                "index": t.index,
                                "role": t.role,
                                "content": t.content,
                                "tool_calls": [
                                    {"id": tc.id, "name": tc.name, "args": tc.args}
                                    for tc in t.tool_calls
                                ],
                            }
                            for t in turns
                        ][: T + 1 if T >= 0 else 0],
                    },
                    indent=2,
                )
                + "\n"
            )
            cmd = args.continuation_cmd.format(
                task_dir=shlex.quote(str(_task_dir)),
                prefix_json=shlex.quote(str(prefix_path)),
            )
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return proc.returncode == 0

        resume_fn = None
        sandbox = None
        if args.replay_prefix:
            try:
                sandbox = _sandbox_for_task(task_dir, platform=args.platform)
                resume_fn = make_resume_fn(sandbox, hard_fail=True)
            except Exception as e:
                print(f"  {trace.run_id}: sandbox unavailable: {e}", file=sys.stderr)
        try:
            result = bisect_forking_step(
                trace.turns,
                resume=resume_fn,
                continue_with_teacher=continue_with_teacher,
                samples_per_probe=args.samples_per_probe,
                verify_probes=args.verify_probes,
            )
        finally:
            if sandbox is not None:
                with contextlib.suppress(Exception):
                    sandbox.cleanup()
        rows.append(
            {
                "run_id": trace.run_id,
                "task": trace.task,
                "steps": result.steps,
                "forking_step": result.forking_step,
                "largest_recoverable_T": result.largest_recoverable_T,
                "monotone": result.monotone,
                "non_monotone_fraction": result.non_monotone_fraction,
                "sample_disagreements": result.sample_disagreements,
                "teacher_continuations": result.teacher_continuations,
                "budget_ok": result.budget_ok,
                "continuation_budget_ok": result.continuation_budget_ok,
                "dropped": result.dropped,
                "drop_reason": result.drop_reason,
                "resume_unverified": result.resume_unverified,
            }
        )

    if not rows:
        print("error: no trajectory was bisected", file=sys.stderr)
        return 1
    located = [r for r in rows if r["forking_step"] is not None and not r["dropped"]]
    summary = {
        "n": len(rows),
        "located": len(located),
        "dropped": sum(1 for r in rows if r["dropped"]),
        "non_monotone": sum(1 for r in rows if not r["monotone"]),
        "resume_unverified": sum(1 for r in rows if r["resume_unverified"]),
        "total_teacher_continuations": sum(r["teacher_continuations"] for r in rows),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bisection.json"
    path.write_text(json.dumps({"summary": summary, "trajectories": rows}, indent=2) + "\n")
    print(f"bisection report: {path}")
    print(json.dumps(summary, indent=2))
    return 0
