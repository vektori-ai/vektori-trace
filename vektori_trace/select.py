"""Step 6's task selection: both conditions required — the diagnosed deficit was
lacking on the task, AND the candidate's own pass rate on it sits in the
trainable band. Neither alone is enough: lacking-but-untrainable wastes a
rejection-sampling run on a task that can never pass; in-band-but-irrelevant
trains on nothing the diagnosis said mattered."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .passrate import PASSRATE_MAX, PASSRATE_MIN, PassRate


def select_training_tasks(
    lacking_loss_tasks: list[str],
    pass_rates: dict[str, PassRate],
    *,
    band: tuple[float, float] = (PASSRATE_MIN, PASSRATE_MAX),
) -> list[str]:
    """`lacking_loss_tasks` are the task ids behind a diagnosis's
    `chosen_deficit.lacking_loss_run_ids` (resolved to task, the pairing key —
    see `gap.compute_gap`), deduplicated by the caller if the same task produced
    more than one lacking-loss trace. A task missing from `pass_rates` (no
    rollouts measured, or all unjudgeable) is excluded rather than assumed
    in-band — silence isn't evidence of trainability."""
    return sorted(
        task
        for task in dict.fromkeys(lacking_loss_tasks)
        if task in pass_rates and pass_rates[task].in_band(band)
    )


def held_out_split(
    task_ids: list[str],
    *,
    exclude: set[str] = frozenset(),
    holdout_frac: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Carve the held-out slice before training, from the excluded set removed
    first — mining an excluded repo/PR (e.g. SWE-bench Verified) into either
    side means training on, or measuring against, the answers.

    Deterministic in `seed` alone: same input list + same seed always produces
    the same split, and the seed is written into every run's report rather than
    left implicit, per V0_PLAN.md's "if a number isn't re-derivable from disk,
    it doesn't count."
    """
    remaining = sorted(t for t in task_ids if t not in exclude)
    rng = random.Random(seed)
    rng.shuffle(remaining)
    n_holdout = round(len(remaining) * holdout_frac)
    holdout = sorted(remaining[:n_holdout])
    train = sorted(remaining[n_holdout:])
    return train, holdout


def write_selection_report(
    out_dir: Path,
    *,
    frontier_model: str,
    candidate_model: str,
    agent: str,
    band: tuple[float, float],
    rollouts: int,
    seed: int,
    holdout_frac: float,
    pass_rates: dict[str, PassRate],
    lacking_loss_tasks: list[str],
    selected: list[str],
    train_ids: list[str],
    holdout_ids: list[str],
    exclude: set[str],
) -> Path:
    """Everything needed to re-derive `selected`/`train_ids`/`holdout_ids` from
    disk without rerunning anything — per-task pass rates included, not just the
    tasks that cleared the band, so a later reviewer can see what missed and by
    how much."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "frontier_model": frontier_model,
        "candidate_model": candidate_model,
        "agent": agent,
        "band": {"min": band[0], "max": band[1]},
        "rollouts": rollouts,
        "seed": seed,
        "holdout_frac": holdout_frac,
        "excluded": sorted(exclude),
        "lacking_loss_tasks": sorted(set(lacking_loss_tasks)),
        "pass_rates": {
            task: {"passed": pr.passed, "n": pr.n, "rate": pr.rate}
            for task, pr in sorted(pass_rates.items())
        },
        "selected": selected,
        "train": train_ids,
        "holdout": holdout_ids,
    }
    json_path = out_dir / "selection.json"
    json_path.write_text(json.dumps(report, indent=2))

    lines = ["# Vektori-trace training task selection\n"]
    lines.append(f"Scaffold: `{agent}`  ·  candidate: `{candidate_model}`  ·  frontier: `{frontier_model}`\n")
    lines.append(
        f"- lacking-loss tasks (deficit was lacking): {len(set(lacking_loss_tasks))}\n"
        f"- band: [{band[0]:.2f}, {band[1]:.2f}], {rollouts} rollout(s)/task\n"
        f"- selected (both conditions met): {len(selected)}\n"
        f"- excluded before split: {len(exclude)}\n"
        f"- held out (frac={holdout_frac}, seed={seed}): {len(holdout_ids)} of {len(train_ids) + len(holdout_ids)}\n"
    )
    if not selected:
        lines.append(
            "\n**Nothing selected.** Either no lacking-loss task's pass rate landed "
            "in the band, or none were measured. Per V0_PLAN.md: an empty band means "
            "nothing trainable at this model+scaffold — widen the replay before "
            "assuming this candidate has nothing to train from.\n"
        )
    md_path = out_dir / "selection.md"
    md_path.write_text("\n".join(lines))
    return md_path


__all__ = ["held_out_split", "select_training_tasks", "write_selection_report"]
