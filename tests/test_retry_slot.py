"""Carrying valid episodes forward while re-rolling one refused slot.

An update is refused unless every planned episode is `sampled`. Rerunning all
eight after a single format failure discards seven already valid, independent,
paid episodes -- same parent, same policy version, same generation config --
and adds sampling variation for nothing.

The retry is legitimate only if it stays visible: the failed attempt is
preserved, the resampled slot is declared, and the batch can never be
described as eight clean first-attempt episodes.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "scripts" / "tau2_live_opd_modal.py"


def _fn():
    tree = ast.parse(SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "retry_slot":
            return node
    raise AssertionError("retry_slot not found")


def _src():
    node = _fn()
    return ast.get_source_segment(SRC.read_text(), node)


class TestExposedToModal:
    def test_has_the_app_function_decorator(self):
        """Without it `modal run ...::retry_slot` cannot find the function."""
        node = _fn()
        assert node.decorator_list, "retry_slot is not registered with Modal"

    def test_signature(self):
        args = [a.arg for a in _fn().args.args]
        for want in ("source_run_id", "dest_run_id", "update", "episode_id",
                     "attempt"):
            assert want in args


class TestProvenanceIsMandatory:
    def test_writes_retry_provenance(self):
        assert "retry_provenance.json" in _src()

    def test_declares_conditional_sampling(self):
        s = _src()
        assert "CONDITIONAL SAMPLING" in s
        assert "format-failure rate" in s

    def test_records_which_slot_was_retried(self):
        s = _src()
        assert '"retried_episode_id"' in s
        assert '"retry_attempt"' in s
        assert '"carried_episodes"' in s


class TestRefusals:
    def test_refuses_same_run_id(self):
        s = _src()
        assert "source_run_id == dest_run_id" in s
        assert "evidence" in s

    def test_refuses_existing_destination(self):
        assert "FileExistsError" in _src()

    def test_requires_all_three_ids(self):
        assert "are required" in _src()

    def test_refuses_a_mixed_batch(self):
        """One policy, one adapter, one recipe -- or it is not one batch."""
        s = _src()
        for field in ("adapter_hash", "policy_version", "gen_config_hash",
                      "require_reasoning"):
            assert field in s
        assert "disagree on" in s


class TestCopiesOnlySamplingEvidence:
    def test_carries_only_sampled_episodes(self):
        s = _src()
        assert 'r.get("status") == "sampled"' in s

    def test_excludes_the_retried_episode(self):
        # multi-slot retry (2026-08-30): the single-id comparison became a set
        # membership test, which is the same exclusion over more than one slot.
        assert "e not in retry_ids" in _src()

    def test_names_what_it_refuses_to_copy(self):
        s = _src()
        for banned in ("scores", "checkpoint", "optimizer state",
                       "SCORED marker", "TRAINED marker"):
            assert banned in s, f"not_copied manifest omits {banned}"

    def test_no_copy_call_targets_scores_or_checkpoints(self):
        """Check the CALLS, not the prose -- the docstring mentions both."""
        import ast as _ast
        node = _fn()
        targets = []
        for n in _ast.walk(node):
            if isinstance(n, _ast.Call):
                fn = n.func
                name = getattr(fn, "attr", None) or getattr(fn, "id", "")
                if name in ("copy2", "copytree"):
                    targets.append(_ast.dump(n))
        joined = " ".join(targets)
        for banned in ("scores", "checkpoint", "optimizer", "SCORED", "TRAINED"):
            assert banned not in joined, (
                f"a copy call references {banned!r}: {joined[:200]}")
        assert targets, "expected at least one copy call for turn files"

    def test_copies_planned_marker_only(self):
        s = _src()
        assert '".PLANNED"' in s
        assert '".SAMPLED"' not in s


class TestSemantics:
    """Why leaving the slot empty is what triggers the resample."""

    def test_absent_slot_is_the_mechanism(self):
        from vektori_trace.tau2 import live_rollout
        src = Path(live_rollout.__file__).read_text()
        assert "if plan.episode_id in existing:" in src, (
            "capture_live_update must skip present ids -- that is what makes "
            "an absent slot resample, with no change to its contract"
        )


def test_provenance_shape_is_complete():
    """The fields a reader needs to know this batch was not clean."""
    s = _src()
    for key in ('"source_run_id"', '"dest_run_id"', '"retried_episode_id"',
                '"retry_attempt"', '"n_carried"', '"declared"', '"not_copied"'):
        assert key in s, f"provenance is missing {key}"
