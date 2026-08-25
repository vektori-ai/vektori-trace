"""Task family and difficulty, derived from Tau2's own task definitions.

Allocation may look at these and at nothing else. In particular it may not look
at whether DeepSeek passed the task: that gates *eligibility* to be a training
task, and using it again to decide W30-vs-C30 or S16-vs-F38 would make the
partitions differ by teacher competence rather than by design.

Family comes from the reference action sequence in `evaluation_criteria`, which
is the benchmark's own statement of what the task requires. Difficulty is the
public frontier band from `docs/tau2-teacher-survey.json` where available; tasks
outside the surveyed 20 are "unknown" and are stratified as their own band so
they spread evenly rather than clumping.
"""

from __future__ import annotations

import json
import os
from typing import Any

MUTATION_PREFIXES = ("cancel_", "modify_", "return_", "exchange_")


def task_family(task: dict[str, Any]) -> str:
    """A normalized workflow family for one task.

    Two tasks share a family when they require the same *kind* of work: the set
    of mutating operations plus whether the flow is multi-step. Naming it from
    the reference actions keeps the grouping tied to the benchmark rather than
    to a hand-written taxonomy.
    """
    crit = task.get("evaluation_criteria") or {}
    actions = crit.get("actions") or []
    names = [a.get("name", "") for a in actions if a.get("requestor") == "assistant"]

    mutations = sorted({n for n in names if n.startswith(MUTATION_PREFIXES)})
    if not mutations:
        return "readonly_lookup"

    short = [m.replace("_pending_order", "").replace("_delivered_order", "")
              .replace("_items", "").replace("_", "") for m in mutations]
    base = "+".join(sorted(set(short)))
    return f"{base}__multi" if len(mutations) > 1 else base


def has_mutation(task: dict[str, Any]) -> bool:
    crit = task.get("evaluation_criteria") or {}
    return any((a.get("name") or "").startswith(MUTATION_PREFIXES)
               for a in (crit.get("actions") or []))


def n_reference_actions(task: dict[str, Any]) -> int:
    crit = task.get("evaluation_criteria") or {}
    return len([a for a in (crit.get("actions") or [])
                if a.get("requestor") == "assistant"])


def load_difficulty_bands(survey_path: str) -> dict[str, str]:
    """Public frontier difficulty band per task id, from the pinned survey."""
    if not os.path.exists(survey_path):
        return {}
    d = json.load(open(survey_path))
    retail = ((d.get("domains") or {}).get("retail") or {})
    bands = retail.get("bands") or d.get("bands") or {}
    out: dict[str, str] = {}
    for band, ids in bands.items():
        label = band.split()[0]        # "easy 100%" -> "easy"
        for t in ids:
            out[str(t)] = label
    return out


def difficulty_for(task_id: str, bands: dict[str, str],
                   n_actions: int) -> str:
    """Band if surveyed, else a proxy from reference-action count.

    The proxy is coarse and is labelled so, but leaving 94 tasks in one
    "unknown" bucket would let long multi-step tasks pile into one partition.
    """
    if task_id in bands:
        return bands[task_id]
    if n_actions <= 2:
        return "proxy_short"
    if n_actions <= 5:
        return "proxy_medium"
    return "proxy_long"


def build_task_metas(tasks: list[dict[str, Any]], *, eligible: set[str],
                     traced: set[str], survey_path: str,
                     decisions: dict[str, int] | None = None,
                     bands: dict[str, str] | None = None):
    """Assemble the TaskMeta table the split builder consumes.

    `bands` should be the real frontier bands from `difficulty.py`. The survey
    file is a fallback and covers only 20 retail tasks, which left 94 of 114 on
    a proxy and produced a split whose evaluation half was materially harder
    than its training half.
    """
    from .split import TaskMeta

    if bands is None:
        bands = load_difficulty_bands(survey_path)
    metas: dict[str, TaskMeta] = {}
    for t in tasks:
        tid = str(t.get("id"))
        n_act = n_reference_actions(t)
        metas[tid] = TaskMeta(
            task_id=tid,
            family=task_family(t),
            difficulty=difficulty_for(tid, bands, n_act),
            eligible=tid in eligible,
            n_decisions=(decisions or {}).get(tid, 0),
            has_mutation=has_mutation(t),
            has_trace=tid in traced,
        )
    return metas
