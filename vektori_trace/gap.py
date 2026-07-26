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
    frontier_wins: int
    frontier_n: int
    candidate_wins: int
    candidate_n: int
    paired_n: int  # tasks where BOTH arms produced a judgeable trace

    @property
    def frontier_rate(self) -> float | None:
        return (self.frontier_wins / self.frontier_n) if self.frontier_n else None

    @property
    def candidate_rate(self) -> float | None:
        return (self.candidate_wins / self.candidate_n) if self.candidate_n else None

    @property
    def gap(self) -> float | None:
        fr, cr = self.frontier_rate, self.candidate_rate
        return None if fr is None or cr is None else fr - cr


def compute_gap(
    traces: list[Trace], frontier_model: str, candidate_model: str, agent: str
) -> GapResult:
    """Pass rate per model, plus the count of tasks both models actually
    attempted (the pairing Step 5's McNemar test needs — same task, two
    models). A task missing from one side (an InfraFailure excluded it there)
    doesn't count as paired even if the other side judged it fine.
    """
    frontier = [t for t in traces if t.model == frontier_model]
    candidate = [t for t in traces if t.model == candidate_model]
    frontier_tasks = {t.task for t in frontier if t.task is not None}
    candidate_tasks = {t.task for t in candidate if t.task is not None}

    return GapResult(
        agent=agent,
        frontier_model=frontier_model,
        candidate_model=candidate_model,
        frontier_wins=sum(1 for t in frontier if t.outcome == "win"),
        frontier_n=len(frontier),
        candidate_wins=sum(1 for t in candidate if t.outcome == "win"),
        candidate_n=len(candidate),
        paired_n=len(frontier_tasks & candidate_tasks),
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
        f"- frontier (`{result.frontier_model}`): {_fmt(result.frontier_rate)} "
        f"({result.frontier_wins}/{result.frontier_n})"
    )
    lines.append(
        f"- candidate (`{result.candidate_model}`): {_fmt(result.candidate_rate)} "
        f"({result.candidate_wins}/{result.candidate_n})"
    )
    lines.append(f"- gap: {_fmt(result.gap)}")
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


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2%}"


__all__ = ["GapResult", "compute_gap", "write_gap_report"]
