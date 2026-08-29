"""`--max-episodes` may shrink a batch only for a diagnostic run.

Batch completeness is what stops an update from training on whichever
episodes happened to succeed -- a self-selected subset biased toward easy
tasks. But a diagnostic (does the parser hold? does the scorer produce sane
advantages?) does not need eight episodes, and paying for eight to answer a
one-episode question is waste.

The fence is the run id rather than a boolean, because a flag can be passed by
accident to the real run, while a run id is written into every artifact -- so a
shrunken batch identifies itself forever after.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "_m", Path(__file__).parent.parent / "scripts" / "tau2_live_opd_modal.py")


def _fence():
    """Import just the predicate, without Modal's decorators executing."""
    src = (Path(__file__).parent.parent / "scripts"
           / "tau2_live_opd_modal.py").read_text()
    start = src.index("def _is_diagnostic_run(")
    # Stop at whatever follows the helper -- it sits above rollout_only's
    # @app.function decorator, so slicing to "def rollout_only(" would drag
    # the decorator in and try to execute Modal at import time.
    end = src.index("@app.function(", start)
    ns: dict = {}
    exec(compile(src[start:end], "<fence>", "exec"), ns)
    return ns["_is_diagnostic_run"]


IS_DIAG = _fence()


class TestFence:
    @pytest.mark.parametrize("run_id", [
        "canary_chunkv2_20260829",
        "diag_parser_check",
        "pilot_10x8_20260829b_canary",
        "DIAG_UPPER",
    ])
    def test_diagnostic_ids_allowed(self, run_id):
        assert IS_DIAG(run_id)

    @pytest.mark.parametrize("run_id", [
        "pilot_10x8_20260829b",
        "pilot_10x8_20260829",
        "two_update_proof_1",
        "",
    ])
    def test_preregistered_ids_refused(self, run_id):
        assert not IS_DIAG(run_id)

    def test_the_real_pilot_cannot_be_shrunk(self):
        """The specific id this experiment will run under."""
        assert not IS_DIAG("pilot_10x8_20260829b")


def test_shrink_logic_is_guarded_in_source():
    """The guard must actually gate the shrink, not merely exist."""
    src = (Path(__file__).parent.parent / "scripts"
           / "tau2_live_opd_modal.py").read_text()
    i = src.index("if max_episodes and max_episodes < len(block):")
    window = src[i:i + 700]
    assert "_is_diagnostic_run(run_id)" in window
    assert "raise ValueError" in window
    # the shrink itself must come after the guard
    assert window.index("raise ValueError") < window.index("block = block[:max_episodes]")
