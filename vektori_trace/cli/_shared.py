"""Helpers shared by more than one command module."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..schema import Trace, load_manifest


def _load_traces(manifest_path: Path) -> list[Trace]:
    entries = load_manifest(manifest_path)
    return [Trace.load(e.path, outcome=e.outcome, model=e.model, task=e.task) for e in entries]


def _check_replay_models(args: argparse.Namespace, traces: list[Trace]) -> str | None:
    """Validate `--frontier-model`/`--candidate-model`, or None if they're fine.

    Checked before the proposer runs: every failure here is knowable from the
    manifest alone, and discovering one after labelling has cost an LLM call per
    trace is pure waste.
    """
    frontier, candidate = args.frontier_model, args.candidate_model
    if (frontier is None) != (candidate is None):
        missing = "--candidate-model" if frontier else "--frontier-model"
        return (
            f"{missing} is required alongside the other — the two contrasts are "
            "defined by which model produced which trace, so one name on its own "
            "names nothing."
        )
    if frontier is None:
        return None
    if frontier == candidate:
        return (
            f"--frontier-model and --candidate-model are the same ({frontier!r}) — "
            "there is no cross-model contrast between a model and itself."
        )
    present = {t.model for t in traces}
    for flag, model in (("--frontier-model", frontier), ("--candidate-model", candidate)):
        if model not in present:
            known = ", ".join(sorted(m for m in present if m)) or "none (no 'model' field set)"
            return (
                f"{flag} {model!r} has no traces in the manifest. Models present: {known}. "
                "A manifest without models is a `mine` manifest; the two-contrast path "
                "needs one from `replay`."
            )
    return None
