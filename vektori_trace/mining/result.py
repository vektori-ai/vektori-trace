"""Uniform result shape for a mining run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PipelineResult:
    """candidates: discovered before filtering. emitted: tasks written to out_dir.
    skipped: sum of skip-reason counts. skip_reasons: per-reason counts."""

    candidates: int
    emitted: int
    skipped: int
    out_dir: Path
    skip_reasons: dict[str, int]
