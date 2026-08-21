"""Loading the stored DeepSeek corpus into replay candidates (plan §8).

Walks a Harbor pass@k output tree, parses each trial's trajectory with the
repo's existing ATIF parser, and yields `ReplayPrefix` candidates. It reads the
corpus; it does not choose from it — `replay_select` does that.

Two judgements live here rather than in the caller:

**Passing trajectories first.** §8 says to "begin with valid prefixes from
passing trajectories because their environment histories and compaction
boundaries are already reconstructable". A failed trajectory's later states can
be arbitrarily broken — the agent may have corrupted the repo — so its prefixes
describe a state no sensible policy would be in. `outcome` is read from the
trial's `result.json` and defaults to *excluding* non-passes.

**A trace's own last step is not a replay state.** At the final step there is no
next action for ck75 to take that the trace can meaningfully be said to precede;
`max_step` is bounded to `n_steps - 1` so the pool never contains one.

The plan's own warning applies to anything built on this: the corpus was *run*
on 60 tasks, but only 34 have a pass. Do not report 60.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .replay_select import ReplayPrefix, enumerate_prefixes


class CorpusError(RuntimeError):
    """The stored corpus cannot be read as replay states."""


@dataclass
class TraceRecord:
    """One stored trial: where it is, what it did, and how long it ran."""

    task: str
    trace_id: str
    trial_dir: Path
    n_steps: int
    passed: bool | None
    #: The stored agent action at each step, kept only so the driver can assert
    #: ck75 did not reproduce it (`assert_action_is_student_sampled`). Never a
    #: training target.
    stored_actions: dict[int, bytes] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def _read_outcome(trial_dir: Path) -> bool | None:
    """Whether this trial passed, from its `result.json`.

    Harbor writes the verdict at `verifier_result.rewards.reward`. A *pass* is
    `reward == 1.0` and nothing else: this corpus also contains fractional
    rewards (0.5, 0.125, 0.777...) which are partial credit — some F2P tests
    passing — and treating those as passes would put prefixes from a trace that
    never fixed the bug into a pool §8 says should start from passing runs.

    Returns None when the file is missing, unreadable, or carries no reward.
    That is "unknown", not "failed": 246 of this corpus's 479 result files have
    no reward at all (mostly `AgentTimeoutError`, where the verifier never ran),
    and binning them with real failures would distort any pass rate quoted from
    it.
    """
    path = trial_dir / "result.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    reward = _extract_reward(data)
    if reward is None:
        return None
    return float(reward) >= 1.0


def _extract_reward(data: dict[str, Any]) -> float | None:
    """The trial's scalar reward, across the shapes this corpus uses.

    `verifier_result.rewards.reward` is the current Harbor layout. The flatter
    fallbacks are older shapes; they are checked after it so a file carrying
    both cannot have the stale one win.
    """
    vr = data.get("verifier_result")
    if isinstance(vr, dict):
        rewards = vr.get("rewards")
        if isinstance(rewards, dict):
            val = rewards.get("reward")
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
        val = vr.get("reward")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)

    for key in ("reward", "passed", "success"):
        val = data.get(key)
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        if isinstance(val, (int, float)):
            return float(val)
    return None


def iter_trials(root: Path) -> Iterator[Path]:
    """Every Harbor trial directory under `root`.

    A trial is identified by containing `agent/trajectory.json`, which is the
    layout `mining.atif.find_trajectory` documents. Directories without one are
    skipped rather than raising: a pass@k tree also holds job logs and configs.
    """
    if not root.is_dir():
        raise CorpusError(f"corpus root is not a directory: {root}")
    seen: set[Path] = set()
    for traj in sorted(root.rglob("trajectory.json")):
        if traj.parent.name != "agent":
            continue
        trial = traj.parent.parent
        if trial not in seen:
            seen.add(trial)
            yield trial


def load_trace(trial_dir: Path, *, keep_stored_actions: bool = True) -> TraceRecord:
    """Parse one trial into a `TraceRecord` using the repo's ATIF parser."""
    from .evaluate.resume import assistant_tool_steps
    from .mining.atif import parse_job_trajectory

    try:
        turns = parse_job_trajectory(trial_dir)
    except Exception as e:  # parser raises several types
        raise CorpusError(f"{trial_dir}: {type(e).__name__}: {e}") from e

    steps = assistant_tool_steps(turns)
    task = _task_name(trial_dir)

    stored: dict[int, bytes] = {}
    if keep_stored_actions:
        by_index = {t.index: t for t in turns}
        for t, (turn_index, _call) in enumerate(steps):
            turn = by_index.get(turn_index)
            if turn is None:
                continue
            text = turn.content or ""
            if text:
                stored[t] = text.encode()

    return TraceRecord(
        task=task,
        trace_id=trial_dir.name,
        trial_dir=trial_dir,
        n_steps=len(steps),
        passed=_read_outcome(trial_dir),
        stored_actions=stored,
        meta={"turns": len(turns)},
    )


def _task_name(trial_dir: Path) -> str:
    """`pypa__hatch-2086` from a trial dir named `pypa__hatch-2086__Ann59jw`."""
    name = trial_dir.name
    if "__" in name:
        parts = name.split("__")
        if len(parts) >= 3:
            return "__".join(parts[:-1])
    return name


def load_corpus(
    root: Path,
    *,
    passing_only: bool = True,
    min_steps: int = 2,
    limit: int | None = None,
) -> list[TraceRecord]:
    """Every usable trace under `root`.

    `passing_only` implements §8's "begin with valid prefixes from passing
    trajectories". Traces whose outcome is *unknown* are excluded under this
    flag too — an unverified trace cannot be claimed as a pass.

    `min_steps=2` drops traces too short to have a replay state that is neither
    the cold start nor the final step.
    """
    traces: list[TraceRecord] = []
    for trial in iter_trials(root):
        try:
            rec = load_trace(trial)
        except CorpusError:
            continue
        if rec.n_steps < min_steps:
            continue
        if passing_only and rec.passed is not True:
            continue
        traces.append(rec)
        if limit is not None and len(traces) >= limit:
            break
    return traces


def candidates_from_traces(
    traces: list[TraceRecord],
    *,
    min_step: int = 1,
    max_step: int | None = None,
) -> list[ReplayPrefix]:
    """Flatten traces into the candidate pool `replay_select` chooses from.

    The last step of each trace is excluded: there is no meaningful "next
    action" for ck75 to take at a state the trace itself never continued from.
    """
    out: list[ReplayPrefix] = []
    for rec in traces:
        from .mining.atif import parse_job_trajectory

        try:
            turns = parse_job_trajectory(rec.trial_dir)
        except Exception:
            continue
        upper = rec.n_steps - 1
        if max_step is not None:
            upper = min(upper, max_step)
        if upper <= min_step:
            continue

        # Compaction boundaries, derived rather than left empty. This marks a
        # step as *following* a boundary; it does not rebuild the retained
        # state — `prefix_turns_through_step` still slices from index 0, so the
        # prefix carries pre-boundary history that Harbor replaced. See
        # `compaction` module docstring and plan §15: `select_replay_prefixes`
        # refuses a non-zero `require_post_compaction` precisely because this
        # flag is positional, not reconstructed.
        comp_steps: set[int] = set()
        try:
            from .compaction import (
                boundaries_from_raw,
                locate_in_turns,
                post_compaction_steps,
            )
            from .evaluate.resume import assistant_tool_steps
            from .mining.atif import find_trajectory

            traj_path = find_trajectory(rec.trial_dir)
            if traj_path is not None:
                raw = boundaries_from_raw(traj_path)
                if raw:
                    located = locate_in_turns(
                        raw, turns, assistant_tool_steps(turns)
                    )
                    comp_steps = post_compaction_steps(located)
        except Exception:
            # A trace whose boundaries cannot be read is not a fatal corpus
            # error: it simply contributes no post-compaction candidates.
            comp_steps = set()

        out.extend(
            enumerate_prefixes(
                rec.task,
                rec.trace_id,
                turns,
                min_step=min_step,
                max_step=upper,
                compaction_steps=comp_steps,
            )
        )
    return out


def corpus_report(traces: list[TraceRecord]) -> dict[str, Any]:
    """Length and outcome distribution — the input to any kappa decision.

    Reported rather than assumed: the plan quotes "14-25-turn cases", and a
    schedule tuned on a different length distribution is not transferable. The
    step histogram here is what a calibrated prefix schedule should be derived
    from.
    """
    if not traces:
        return {"n_traces": 0}
    steps = sorted(r.n_steps for r in traces)
    hist: dict[str, int] = {}
    for n in steps:
        hist[str(n)] = hist.get(str(n), 0) + 1
    tasks = {r.task for r in traces}
    return {
        "n_traces": len(traces),
        "n_tasks": len(tasks),
        "n_passed": sum(1 for r in traces if r.passed is True),
        "n_failed": sum(1 for r in traces if r.passed is False),
        "n_unknown": sum(1 for r in traces if r.passed is None),
        "steps_min": steps[0],
        "steps_max": steps[-1],
        "steps_median": steps[len(steps) // 2],
        "steps_mean": sum(steps) / len(steps),
        "step_count_histogram": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
        "total_candidate_steps": sum(max(0, n - 1) for n in steps),
    }


__all__ = [
    "CorpusError",
    "TraceRecord",
    "candidates_from_traces",
    "corpus_report",
    "iter_trials",
    "load_corpus",
    "load_trace",
]
