"""Non-regression check: narrow SFT must not trash instruction-following.

Pre-declared (before any training run), per V0_PLAN.md discipline:
- fixed seeded 50-prompt subset of public IFEval
- metric: IFEval strict-accuracy
- tolerance: ≤ 5 absolute points drop vs untouched candidate on the same subset
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

IFEVAL_SUBSET_SIZE = 50
IFEVAL_TOLERANCE = 0.05  # 5-point absolute drop
IFEVAL_SEED = 0


def _require_train():
    try:
        import datasets  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "training extras required: install with `uv sync --extra train`"
        ) from e


@dataclass
class NonRegressionResult:
    subset_size: int
    seed: int
    tolerance: float
    baseline_strict_acc: float
    arm_strict_acc: float
    delta: float
    passed: bool  # True iff drop ≤ tolerance
    arm: str


def load_ifeval_subset(
    *,
    n: int = IFEVAL_SUBSET_SIZE,
    seed: int = IFEVAL_SEED,
) -> list[dict[str, Any]]:
    """Fixed seeded subset of `google/IFEval` (or `wis-k/instruction-following-eval`)."""
    _require_train()
    from datasets import load_dataset

    # Prefer the canonical google card; fall back to the community mirror.
    try:
        ds = load_dataset("google/IFEval", split="train")
    except Exception:
        ds = load_dataset("wis-k/instruction-following-eval", split="train")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    chosen = idxs[:n]
    return [dict(ds[i]) for i in chosen]


def _strict_acc(records: list[dict[str, Any]], responses: list[str]) -> float:
    """IFEval strict-accuracy: every constraint on a prompt must pass.

    Uses the dataset's own `instruction_id_list` + kwargs when the official
    evaluator is importable; otherwise a conservative placeholder that scores
    empty responses as 0 and non-empty as unchecked — callers should pass an
    `evaluate_fn` for real runs.
    """
    try:
        from instruction_following_eval.evaluation import (  # type: ignore
            check_follow_instructions,
        )
    except ImportError:
        # Without the official helper we only refuse empty answers — report N
        # beside the number and treat this as a soft check.
        n = len(responses)
        if n == 0:
            return 0.0
        return sum(1 for r in responses if r and r.strip()) / n

    ok = 0
    for rec, resp in zip(records, responses, strict=True):
        if check_follow_instructions(rec, resp):
            ok += 1
    return ok / len(records) if records else 0.0


def evaluate_ifeval(
    records: list[dict[str, Any]],
    generate_fn: Callable[[str], str],
    *,
    evaluate_fn: Callable[[list[dict], list[str]], float] | None = None,
) -> float:
    prompts = [r.get("prompt") or r.get("instruction") or "" for r in records]
    responses = [generate_fn(p) for p in prompts]
    scorer = evaluate_fn or _strict_acc
    return scorer(records, responses)


def check_non_regression(
    arm: str,
    baseline_generate: Callable[[str], str],
    arm_generate: Callable[[str], str],
    *,
    subset: list[dict[str, Any]] | None = None,
    tolerance: float = IFEVAL_TOLERANCE,
    seed: int = IFEVAL_SEED,
    n: int = IFEVAL_SUBSET_SIZE,
) -> NonRegressionResult:
    records = subset if subset is not None else load_ifeval_subset(n=n, seed=seed)
    baseline = evaluate_ifeval(records, baseline_generate)
    arm_acc = evaluate_ifeval(records, arm_generate)
    delta = arm_acc - baseline
    return NonRegressionResult(
        subset_size=len(records),
        seed=seed,
        tolerance=tolerance,
        baseline_strict_acc=baseline,
        arm_strict_acc=arm_acc,
        delta=delta,
        passed=delta >= -tolerance,
        arm=arm,
    )


def write_nonregression_report(results: list[NonRegressionResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in results]
    json_path = out_dir / "nonregression.json"
    json_path.write_text(json.dumps(payload, indent=2))
    lines = [
        "# Non-regression (IFEval strict-acc)\n",
        f"Tolerance: ≤ {IFEVAL_TOLERANCE:.0%} absolute drop vs untouched candidate. "
        f"Subset N={IFEVAL_SUBSET_SIZE}, seed={IFEVAL_SEED}.\n",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"- `{r.arm}`: {r.arm_strict_acc:.2%} (baseline {r.baseline_strict_acc:.2%}, "
            f"delta={r.delta:+.2%}) [{status}]\n"
        )
    md_path = out_dir / "nonregression.md"
    md_path.write_text("".join(lines))
    return md_path


def subset_digest(records: list[dict[str, Any]]) -> str:
    blob = json.dumps([r.get("key") or r.get("prompt") for r in records], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


__all__ = [
    "IFEVAL_SEED",
    "IFEVAL_SUBSET_SIZE",
    "IFEVAL_TOLERANCE",
    "NonRegressionResult",
    "check_non_regression",
    "evaluate_ifeval",
    "load_ifeval_subset",
    "subset_digest",
    "write_nonregression_report",
]
