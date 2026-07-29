"""Step E — ground diagnose.py labels against bisection forking steps (PLAN.md).

Compare capability labels to execution-established forking steps; measure blur
and suggest a recalibrated min_gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GroundingPair:
    task: str
    capability: str
    forking_step: int | None
    label_agrees: bool | None  # None = no human/heuristic judgment supplied
    diagnose_gap: float | None = None


@dataclass
class GroundingReport:
    n: int
    agreement_rate: float | None
    blur: float  # mean |gap| on disagreements; 0 if none
    suggested_min_gap: float
    pairs: list[GroundingPair]
    underpowered: bool


def ground_diagnosis(
    pairs: list[GroundingPair],
    *,
    current_min_gap: float,
    agreement_floor: float = 0.70,
) -> GroundingReport:
    """Compare diagnose labels against forking steps.

    `label_agrees` should be set by the caller (human inspection or a heuristic
    that maps forking-step context → capability). This module aggregates only.

    `suggested_min_gap` recalibrates in **one direction only: upward**, and its
    evidence is **disagreements only**.

    - It is raised to the measured blur when agreement falls below
      `agreement_floor` *and* that blur exceeds `current_min_gap`; otherwise it is
      returned unchanged. It is never lowered, so high agreement cannot loosen the
      threshold — a labeller that agrees is not evidence that a smaller gap is
      safe, only that the current one has not been shown unsafe.
    - `blur` is the mean |diagnose_gap| over pairs where the label *disagreed*
      with the forking step. Agreeing pairs contribute nothing, so the suggestion
      is calibrated against observed label failures rather than the gap
      distribution as a whole.
    """
    judged = [p for p in pairs if p.label_agrees is not None]
    n = len(pairs)
    if not judged:
        return GroundingReport(
            n=n,
            agreement_rate=None,
            blur=0.0,
            suggested_min_gap=current_min_gap,
            pairs=pairs,
            underpowered=True,
        )

    agrees = sum(1 for p in judged if p.label_agrees)
    rate = agrees / len(judged)
    disagreements = [p for p in judged if not p.label_agrees]
    blur_vals = [
        abs(p.diagnose_gap) for p in disagreements if p.diagnose_gap is not None
    ]
    blur = (sum(blur_vals) / len(blur_vals)) if blur_vals else 0.0

    # Recalibrate: if agreement below floor, raise min_gap toward measured blur.
    suggested = current_min_gap
    if rate < agreement_floor and blur > current_min_gap:
        suggested = blur

    return GroundingReport(
        n=n,
        agreement_rate=rate,
        blur=blur,
        suggested_min_gap=suggested,
        pairs=pairs,
        underpowered=len(judged) < 10,
    )


def report_to_dict(report: GroundingReport) -> dict[str, Any]:
    return {
        "n": report.n,
        "agreement_rate": report.agreement_rate,
        "blur": report.blur,
        "suggested_min_gap": report.suggested_min_gap,
        "underpowered": report.underpowered,
        "pairs": [
            {
                "task": p.task,
                "capability": p.capability,
                "forking_step": p.forking_step,
                "label_agrees": p.label_agrees,
                "diagnose_gap": p.diagnose_gap,
            }
            for p in report.pairs
        ],
    }


__all__ = [
    "GroundingPair",
    "GroundingReport",
    "ground_diagnosis",
    "report_to_dict",
]
