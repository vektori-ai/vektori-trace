"""`--start-at` must not step over an update that never finished.

Skipping one leaves a hole every later update is parented on: the run reports
N updates of on-policy training while some prefix of them never happened, and
nothing downstream can distinguish that from a clean run.
"""

from __future__ import annotations

import pytest


def _guard(start_at, next_update, allow_gap=False):
    """The production rule, extracted for test."""
    if next_update is not None and start_at > next_update and not allow_gap:
        raise SystemExit(
            f"--start-at {start_at} would skip update {next_update}"
        )
    return True


class TestGuard:
    def test_resuming_at_next_update_is_allowed(self):
        assert _guard(3, 3)

    def test_restarting_an_earlier_update_is_allowed(self):
        """Re-running a completed update is safe; stages are idempotent."""
        assert _guard(1, 3)

    def test_zero_from_scratch_is_allowed(self):
        assert _guard(0, 0)

    def test_skipping_an_untrained_update_is_refused(self):
        with pytest.raises(SystemExit, match="would skip update 0"):
            _guard(5, 0)

    def test_skipping_by_one_is_refused(self):
        with pytest.raises(SystemExit, match="would skip update 3"):
            _guard(4, 3)

    def test_allow_gap_overrides_deliberately(self):
        assert _guard(5, 0, allow_gap=True)

    def test_unknown_next_update_does_not_block(self):
        """A status without next_update must not become an unresumable run."""
        assert _guard(5, None)


def test_flag_is_wired_into_the_parser():
    import ast
    from pathlib import Path

    src = Path(__file__).parent.parent / "scripts" / "tau2_pilot_orchestrate.py"
    tree = ast.parse(src.read_text())
    flags = {
        a.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "add_argument"
        for a in node.args
        if isinstance(a, ast.Constant) and isinstance(a.value, str)
    }
    assert "--allow-gap" in flags
    assert "--start-at" in flags
