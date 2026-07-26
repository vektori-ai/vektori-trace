"""Step 4's headline number: the frontier-vs-candidate pass-rate gap on the same
mined tasks, on one pinned scaffold. Computed before any diagnosis runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .schema import Trace

MIN_TASKS_FOR_FRAMING = 50
MIN_MEANINGFUL_GAP = 0.10


@dataclass
class GapResult:
    agent: str  # the one scaffold pinned across both arms
    frontier_model: str
    candidate_model: str
    frontier_attempted: int  # total tasks the frontier produced a judgeable trace for
    candidate_attempted: int  # same, for the candidate
    paired_n: int  # tasks BOTH arms judged — the basis for every rate below
    frontier_wins: int  # wins within the paired set only
    candidate_wins: int  # wins within the paired set only

    @property
    def frontier_rate(self) -> float | None:
        return (self.frontier_wins / self.paired_n) if self.paired_n else None

    @property
    def candidate_rate(self) -> float | None:
        return (self.candidate_wins / self.paired_n) if self.paired_n else None

    @property
    def gap(self) -> float | None:
        fr, cr = self.frontier_rate, self.candidate_rate
        return None if fr is None or cr is None else fr - cr


def compute_gap(
    traces: list[Trace], frontier_model: str, candidate_model: str, agent: str
) -> GapResult:
    """Pass rate per model, computed ONLY over tasks both models actually
    attempted (`paired_n`). Comparing each arm's own unrestricted rate would
    let asymmetric InfraFailure attrition (e.g. the frontier OOMing on the
    slow-building tasks) bias the gap — the same failure mode Step 4 exists to
    avoid by construction, since "the same mined tasks" is the whole point.

    frontier_attempted/candidate_attempted are reported separately so a run
    with heavy, lopsided attrition is visible even though it doesn't move the
    reported rate.
    """
    frontier = [t for t in traces if t.model == frontier_model]
    candidate = [t for t in traces if t.model == candidate_model]
    frontier_by_task = {t.task: t for t in frontier if t.task is not None}
    candidate_by_task = {t.task: t for t in candidate if t.task is not None}
    paired_tasks = frontier_by_task.keys() & candidate_by_task.keys()

    return GapResult(
        agent=agent,
        frontier_model=frontier_model,
        candidate_model=candidate_model,
        frontier_attempted=len(frontier),
        candidate_attempted=len(candidate),
        paired_n=len(paired_tasks),
        frontier_wins=sum(1 for task in paired_tasks if frontier_by_task[task].outcome == "win"),
        candidate_wins=sum(1 for task in paired_tasks if candidate_by_task[task].outcome == "win"),
    )


def write_gap_report(result: GapResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = asdict(result)
    report["frontier_rate"] = result.frontier_rate
    report["candidate_rate"] = result.candidate_rate
    report["gap"] = result.gap
    json_path = out_dir / "gap.json"
    json_path.write_text(json.dumps(report, indent=2))

    lines = ["# Vektori-trace replay: the gap number\n"]
    lines.append(f"Scaffold (pinned across both arms): `{result.agent}`\n")
    lines.append(
        f"- frontier (`{result.frontier_model}`): {format_rate(result.frontier_rate)} "
        f"({result.frontier_wins}/{result.paired_n} paired; {result.frontier_attempted} attempted)"
    )
    lines.append(
        f"- candidate (`{result.candidate_model}`): {format_rate(result.candidate_rate)} "
        f"({result.candidate_wins}/{result.paired_n} paired; {result.candidate_attempted} attempted)"
    )
    lines.append(f"- gap: {format_rate(result.gap)}")
    lines.append(f"- paired tasks (both arms judged): {result.paired_n}\n")

    advisories = []
    if result.paired_n < MIN_TASKS_FOR_FRAMING:
        advisories.append(
            f"Only {result.paired_n} paired task(s) — measure the real gap on "
            f"≥{MIN_TASKS_FOR_FRAMING} before believing any framing (V0_PLAN.md Step 4)."
        )
    if result.gap is not None and abs(result.gap) < MIN_MEANINGFUL_GAP:
        advisories.append(
            f"Gap is under {MIN_MEANINGFUL_GAP:.0%} — change the candidate, not the "
            "story (V0_PLAN.md Step 4)."
        )
    if advisories:
        lines.append("## Advisories\n")
        for a in advisories:
            lines.append(f"- {a}")

    md_path = out_dir / "gap.md"
    md_path.write_text("\n".join(lines))
    return md_path


def format_rate(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2%}"


__all__ = ["MIN_MEANINGFUL_GAP", "MIN_TASKS_FOR_FRAMING", "GapResult", "compute_gap", "format_rate", "write_gap_report"]
