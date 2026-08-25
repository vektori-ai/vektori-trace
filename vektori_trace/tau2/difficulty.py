"""Real public difficulty bands from Tau2's own frontier results.

`docs/tau2-retail20.json` carries hand-copied bands for 20 retail tasks, which
left 94 of 114 unbanded and forced the split to stratify on a proxy. The full
results those bands were derived from ship with the benchmark:

    data/tau2/results/final/{model}_{domain}_default_*_4trials.json

Three frontier models x 4 trials = up to 12 observations per task. This computes
the same measure over every task, so allocation stratifies on the real thing.

It is a CEILING measure: those models sit well above any open model, so it
orders difficulty without predicting Qwen or Flash performance. That is exactly
what stratification needs -- a consistent ordering to balance across.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

# The three models the published bands were computed from. `default` runs only:
# the `no-user`, `op` and `base` variants change the task, not its difficulty.
FRONTIER = ("claude-3-7-sonnet", "gpt-4.1-2025", "o4-mini")


def _is_frontier_default(path: str, domain: str) -> bool:
    name = os.path.basename(path)
    if f"_{domain}_default_" not in name:
        return False
    return any(name.startswith(m) for m in FRONTIER)


def frontier_pass_rates(results_dir: str, domain: str) -> dict[str, float]:
    """Per-task pass rate over all frontier trials, or {} if unavailable."""
    hits: dict[str, list[int]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if not _is_frontier_default(path, domain):
            continue
        try:
            data = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        for sim in data.get("simulations", []):
            tid = str(sim.get("task_id"))
            reward = (sim.get("reward_info") or {}).get("reward")
            if reward is None:
                continue
            hits[tid].append(1 if reward in (1, 1.0) else 0)
    return {t: sum(v) / len(v) for t, v in hits.items() if v}


def band_for(rate: float) -> str:
    """Bands matching the published cut points (easy 100 / med 66-92 / hard <=42)."""
    if rate >= 0.95:
        return "easy"
    if rate >= 0.60:
        return "med"
    if rate > 0.0:
        return "hard"
    return "unsolved"


def difficulty_bands(results_dir: str, domain: str) -> tuple[dict[str, str], dict]:
    """Map task_id -> band, plus provenance for the manifest."""
    rates = frontier_pass_rates(results_dir, domain)
    bands = {t: band_for(r) for t, r in rates.items()}
    files = [os.path.basename(p) for p in sorted(glob.glob(
        os.path.join(results_dir, "*.json"))) if _is_frontier_default(p, domain)]
    meta = {
        "source": "tau2 data/tau2/results/final frontier default runs",
        "files": files,
        "n_tasks_with_rate": len(rates),
        "cut_points": {"easy": ">=0.95", "med": ">=0.60", "hard": ">0.0",
                       "unsolved": "==0.0"},
        "caveat": ("a CEILING measure over frontier models; it orders difficulty "
                   "but does not predict open-model performance"),
    }
    return bands, meta
