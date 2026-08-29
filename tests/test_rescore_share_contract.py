"""Analysis must use the training contract for share enforcement.

Update 0 of pilot_10x8_20260829d died with "task share 0.287 exceeds 0.1875"
AFTER all 75 teacher scores were bought and written -- `rescore` enforced a
limit that `run_projected_train_stage` passes `enforce_shares=False` to skip.
Its comment claimed parity with training while omitting that argument.

Enforcement is wrong on the live path: an episode's length is an OUTCOME, so
rejecting a batch after seeing its realized length is outcome-dependent
selection that excludes the long, hard trajectories OPD exists to learn from.
Balance is enforced before the rollout, by the frozen schedule.

This is restart gate 6 in docs/TAU2-OPD-DEEP-DIVE.md.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MODAL = Path(__file__).parent.parent / "scripts" / "tau2_live_opd_modal.py"
TRAIN = Path(__file__).parent.parent / "vektori_trace" / "tau2" / "live_train.py"


def _calls(src: str, name: str):
    tree = ast.parse(src)
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", None)) == name]


def _kw(call, key):
    for k in call.keywords:
        if k.arg == key:
            return k.value
    return None


class TestParity:
    def test_training_disables_enforcement(self):
        calls = _calls(TRAIN.read_text(), "build_projected_batch")
        assert calls, "training no longer builds a projected batch"
        for c in calls:
            v = _kw(c, "enforce_shares")
            assert v is not None and v.value is False, (
                "training must pass enforce_shares=False")

    def test_analysis_matches_training(self):
        calls = _calls(MODAL.read_text(), "build_projected_batch")
        assert calls, "rescore no longer builds a projected batch"
        for c in calls:
            v = _kw(c, "enforce_shares")
            assert v is not None, (
                "rescore omits enforce_shares, so it defaults to enforcing "
                "and refuses batches training would accept -- after the "
                "teacher has already been paid")
            assert v.value is False


class TestBehaviour:
    """The rule itself, on the shape that failed."""

    @staticmethod
    def _check(worst, limit, enforce):
        if enforce and worst > limit:
            raise ValueError(f"task share {worst:.3f} exceeds {limit}")
        return True

    def test_the_observed_failure_no_longer_refuses(self):
        assert self._check(0.287, 0.1875, enforce=False)

    def test_enforcement_would_still_refuse(self):
        with pytest.raises(ValueError, match="0.287"):
            self._check(0.287, 0.1875, enforce=True)

    def test_shares_are_still_reported(self):
        """Not enforcing is not the same as not measuring."""
        src = MODAL.read_text()
        i = src.index("enforce_shares=False")
        assert "spread" in src[i:i + 3000] or "share" in src[i:i + 3000]


class TestDeclaredSkips:
    """A preregistered exclusion must not fail the stage that declared it.

    The manifest permits exactly one payload skip -- the interleaved-reasoning
    exclusion on u000-task76-seed0@9#0. `rescore` hard-raised on ANY skip, so
    update 0 failed its own preregistration after the teacher was paid.
    """

    def test_undeclared_skips_still_fail(self):
        src = MODAL.read_text()
        assert "UNDECLARED payload skips" in src
        i = src.index("undeclared = [sk for sk in skips")
        assert "raise RuntimeError" in src[i:i + 400]

    def test_declared_skips_are_read_from_the_manifest(self):
        src = MODAL.read_text()
        i = src.index("declared_skips = set()")
        window = src[i:i + 900]
        assert "declared_exclusions" in window
        assert "identities" in window

    def test_declared_skips_are_reported_not_hidden(self):
        src = MODAL.read_text()
        assert "declared skips   :" in src

    @staticmethod
    def _filter(skips, declared):
        return [sk for sk in skips if sk[0] not in declared]

    def test_the_observed_case_passes(self):
        skips = [("u000-task76-seed0@9#0", "reasoning", None)]
        assert self._filter(skips, {"u000-task76-seed0@9#0"}) == []

    def test_a_new_skip_fails(self):
        skips = [("u000-task76-seed0@9#0", "reasoning", None),
                 ("u000-task44-seed1@2#0", "reasoning", None)]
        out = self._filter(skips, {"u000-task76-seed0@9#0"})
        assert len(out) == 1 and out[0][0] == "u000-task44-seed1@2#0"

    def test_no_declaration_means_any_skip_fails(self):
        skips = [("u000-task76-seed0@9#0", "reasoning", None)]
        assert len(self._filter(skips, set())) == 1
