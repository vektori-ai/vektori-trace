"""The driver's refusals and its resume arithmetic.

No GPU, no endpoint, no teacher. What is tested here is the bookkeeping that
decides whether a resumed run re-buys teacher calls or trains from the wrong
parent -- the failures that cost money or invalidate the arm.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "tau2_reopd_train", REPO / "scripts" / "tau2_reopd_train.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

from vektori_trace.tau2.reopd_sample import ReOPDSampleError  # noqa: E402
from vektori_trace.tau2.reopd_state import RunState, append_jsonl  # noqa: E402


# --- the constants are the preregistered ones ----------------------------


def test_action_cap_is_the_tau2_derived_one():
    """592 stored max; 2048 is ~3.5x. Not the 256 that truncated 68% before."""
    assert driver.MAX_ACTION_TOKENS == 2048


def test_required_context_covers_the_longest_prefix_plus_the_cap():
    need = driver.LONGEST_PREFIX_TOKENS + driver.MAX_ACTION_TOKENS
    assert need == 14_928
    assert driver.REQUIRED_CONTEXT >= need
    # the 12288 corpus dir sitting next door would NOT fit this
    assert 12_288 < need


# --- endpoint verification -----------------------------------------------


def _models_response(monkeypatch, payload):
    import urllib.request

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(payload).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: R())


def test_endpoint_with_enough_context_passes(monkeypatch):
    _models_response(monkeypatch, {"data": [
        {"id": "ck35", "max_model_len": 16384}]})
    out = driver.verify_endpoint("http://x", "ck35")
    assert out["max_model_len"] == 16384


def test_endpoint_too_short_is_refused(monkeypatch):
    """A 12288 server drops tokens from the FRONT, silently."""
    _models_response(monkeypatch, {"data": [
        {"id": "ck35", "max_model_len": 12288}]})
    with pytest.raises(ReOPDSampleError, match="context 12288 < 14928"):
        driver.verify_endpoint("http://x", "ck35")


def test_endpoint_serving_another_model_is_refused(monkeypatch):
    """vLLM resolves an unknown name to the base; the adapter does nothing."""
    _models_response(monkeypatch, {"data": [
        {"id": "Qwen/Qwen3-4B", "max_model_len": 16384}]})
    with pytest.raises(ReOPDSampleError, match="does not advertise"):
        driver.verify_endpoint("http://x", "ck35")


def test_endpoint_without_a_reported_context_is_refused(monkeypatch):
    """Trusting the launch flag hides a --max-model-len typo."""
    _models_response(monkeypatch, {"data": [{"id": "ck35"}]})
    with pytest.raises(ReOPDSampleError, match="did not report a context"):
        driver.verify_endpoint("http://x", "ck35")


# --- resume arithmetic ----------------------------------------------------


def _finish(rs, i, n=2):
    u = rs.update(i)
    u.mark("PLANNED")
    for k in range(n):
        append_jsonl(u.actions_path, {"key": f"{i}#{k}"})
    u.mark("SAMPLED")
    for k in range(n):
        append_jsonl(u.scores_path, {"key": f"{i}#{k}"})
    u.mark("SCORED")
    cp = u.checkpoint_path
    cp.mkdir(parents=True, exist_ok=True)
    (cp / "adapter_config.json").write_text("{}")
    (cp / "adapter_model.safetensors").write_text("w")
    (cp / "optimizer.pt").write_text("o")
    (cp / "state.json").write_text(json.dumps({
        "update_index": i, "policy_version": f"reopd-u{i:03d}",
        "parent_policy_hash": "ck35", "rng_state": {"python": "x"},
        "scheduler_state": None, "reload_verified": True,
    }))
    u.mark("TRAINED")


def test_a_32_update_run_resumes_where_it_died(tmp_path):
    rs = RunState(tmp_path / "run", n_updates=32)
    for i in range(20):
        _finish(rs, i)
    assert rs.resume_point() == 20


def test_scores_paid_before_a_crash_are_not_re_bought(tmp_path):
    """The money-losing failure: re-dispatching already-billed calls."""
    rs = RunState(tmp_path / "run", n_updates=32)
    for i in range(20):
        _finish(rs, i)
    # update 20 died mid-scoring: 9 of 16 paid, no SCORED marker
    u = rs.update(20)
    u.mark("PLANNED")
    for k in range(16):
        append_jsonl(u.actions_path, {"key": f"20#{k}"})
    u.mark("SAMPLED")
    for k in range(9):
        append_jsonl(u.scores_path, {"key": f"20#{k}"})

    assert rs.resume_point() == 20
    assert len(rs.paid_scores(20)) == 9      # reused, not re-dispatched
    assert not u.reached("SCORED")


def test_completed_run_reports_done(tmp_path):
    rs = RunState(tmp_path / "run", n_updates=3)
    for i in range(3):
        _finish(rs, i)
    assert rs.resume_point() == 3


# --- the manifest pins the recipe ----------------------------------------


def test_resuming_under_a_changed_recipe_is_refused(tmp_path):
    """One artifact trained under two recipes, nothing recording the split."""
    from vektori_trace.tau2.reopd_state import ReOPDStateError

    rs = RunState(tmp_path / "run", n_updates=32)
    rs.freeze_manifest({"max_action_tokens": 2048, "n_per_update": 16})
    with pytest.raises(ReOPDStateError, match="two recipes"):
        rs.freeze_manifest({"max_action_tokens": 512, "n_per_update": 16})
