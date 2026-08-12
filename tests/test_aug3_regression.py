"""Gate 1: the real Aug-3 Qwen3-14B artifacts must now grade as ungradeable.

This is the regression that motivated the fix, run against the actual job dirs
the sweep produced rather than a synthetic fixture. `passk.json` from that run
reports `c=0, n=4` and `curves {"1": 0.0, "4": 0.0}` — a confident-looking
"the model solved none of these" — while every one of the four
`reward-details.json` files says the verifier could not grade its own work.

Skipped when the artifacts are not in the checkout (they are gitignored).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vektori_trace.evaluate.validity import (
    UNJUDGEABLE_STATUSES,
    _eval_is_untrustworthy,
    _find_parse_status,
    _find_reward,
)

ARTIFACTS = (
    Path(__file__).resolve().parent.parent / "qwen-stuff-14B" / "qwen14b" / "passk_jobs" / "stage1"
)


def _rollout_dirs() -> list[Path]:
    if not ARTIFACTS.is_dir():
        return []
    return sorted(d for d in ARTIFACTS.iterdir() if d.is_dir())


pytestmark = pytest.mark.skipif(
    not _rollout_dirs(),
    reason=f"Aug-3 artifacts not in this checkout ({ARTIFACTS})",
)


def test_all_four_rollouts_were_ungradeable() -> None:
    dirs = _rollout_dirs()
    assert len(dirs) == 4, f"expected the 4 Aug-3 rollouts, got {len(dirs)}"

    for d in dirs:
        reward = _find_reward(d)
        status = _find_parse_status(d)
        assert reward == 0.0, f"{d.name}: expected the recorded 0.0, got {reward}"
        assert status == "fallback_exitcode", f"{d.name}: parse_status={status}"
        assert status in UNJUDGEABLE_STATUSES
        assert _eval_is_untrustworthy(d), (
            f"{d.name}: verifier disclaimed its reward but we would still grade it"
        )


def test_the_recorded_report_disagrees_with_the_artifacts() -> None:
    """Documents the bug's effect: passk.json counted 4 ungradeable rollouts as
    4 model failures. If this ever stops being true the sweep was re-run and
    this test should be re-pointed, not deleted."""
    report = json.loads((ARTIFACTS.parent.parent / "passk.json").read_text())
    task = next(iter(report["stage1"].values()))

    assert task["n"] == 4 and task["c"] == 0
    assert task["curves"]["1"] == 0.0
    assert all(r["passed"] is False for r in report["rollouts"])

    # ...yet not one of them was gradeable.
    assert all(_eval_is_untrustworthy(d) for d in _rollout_dirs())


def test_empty_patch_rollout_is_indistinguishable_without_the_fix() -> None:
    """Rollout 2 submitted a 0-byte model_patch.diff and "failed" exactly like
    the others — the proof the failure was environmental, not the model."""
    empty = [
        d for d in _rollout_dirs() for p in d.rglob("model_patch.diff") if p.stat().st_size == 0
    ]
    assert empty, "expected at least one rollout with an empty patch"
    for d in empty:
        assert _find_reward(d) == 0.0
        assert _eval_is_untrustworthy(d)
