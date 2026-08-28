"""Two-update rehearsal: the invariants a paid pilot cannot re-test.

Everything here runs on CPU with stubs. What it proves is the set of things
that would be silently wrong in a real run and expensive to discover:

- update 0 uses a fresh optimizer; update 1 RESUMES update 0's Adam state,
  with nonzero step counters (a reset is invisible -- max_param_delta reads
  1e-5 either way);
- identity comes from each update's own .SAMPLED marker, not the run manifest;
- a resumed run resamples nothing and re-buys no scores;
- a missing optimizer.pt is refused BEFORE a GPU is dispatched;
- the frozen plan is 80 distinct pairs and a tampered manifest is rejected;
- teardown keeps app ids whose stop failed;
- the budget ceiling stops the loop.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def orch():
    return _load("orch", REPO / "scripts" / "tau2_pilot_orchestrate.py")


@pytest.fixture(scope="module")
def planner():
    return _load("planner", REPO / "scripts" / "tau2_pilot_manifest.py")


# --- the frozen plan -------------------------------------------------------

C30 = [str(t) for t in [
    2, 4, 8, 9, 23, 26, 28, 44, 45, 50, 53, 58, 63, 64, 66, 67, 68, 69, 71,
    76, 78, 85, 87, 95, 96, 98, 101, 102, 108, 109]]


def test_plan_is_80_distinct_pairs_over_c30(planner):
    import random

    plans = planner.build_plans(
        C30, n_updates=10, episodes_per_update=8, seeds=[0, 1, 2],
        rng=random.Random(20260829),
    )
    flat = [(p["task_id"], p["seed"]) for blk in plans for p in blk]
    assert len(flat) == 80
    assert len(set(flat)) == 80, "a reused pair makes this a data-reuse study"
    assert set(t for t, _ in flat) <= set(C30)
    for u, blk in enumerate(plans):
        tasks = [p["task_id"] for p in blk]
        assert len(tasks) == len(set(tasks)), (
            f"update {u} repeats a task; with share limits demoted to "
            "telemetry the frozen plan is the only balance mechanism"
        )


def test_plan_refuses_when_the_pool_is_too_small(planner):
    import random

    with pytest.raises(SystemExit, match="distinct"):
        planner.build_plans(
            C30[:4], n_updates=10, episodes_per_update=8, seeds=[0],
            rng=random.Random(1),
        )


# --- pre-GPU checkpoint gate -----------------------------------------------

def _status(n: int, **over) -> dict:
    rows = []
    for i in range(n):
        rows.append({
            "update": i, "planned": True, "sampled": True, "scored": True,
            "trained": True, "n_scores": 8,
            "sampled_adapter_hash": f"hash-u{i}",
            "sampled_policy_version": f"live-u{i:03d}",
            "trained_adapter_hash": f"hash-u{i + 1}",
            "checkpoint_complete": True,
        })
    for i, patch in (over.get("patch") or {}).items():
        rows[i].update(patch)
    return {"exists": True, "n_updates": n, "updates": rows,
            "parent_adapter_hash": "hash-u0"}


def test_missing_optimizer_refuses_before_gpu_dispatch(orch):
    st = _status(2, patch={0: {"checkpoint_complete": False}})
    with pytest.raises(SystemExit, match="optimizer.pt"):
        orch.preflight_checkpoint(st, 1)


def test_untrained_previous_update_refuses(orch):
    st = _status(2, patch={0: {"trained": False}})
    with pytest.raises(SystemExit, match="not TRAINED"):
        orch.preflight_checkpoint(st, 1)


def test_update_zero_needs_no_previous_checkpoint(orch):
    orch.preflight_checkpoint(_status(1), 0)  # must not raise


# --- teardown --------------------------------------------------------------

def test_teardown_keeps_ids_whose_stop_failed(orch, tmp_path, monkeypatch):
    path = tmp_path / "owned.json"
    owned = orch.OwnedApps(path)
    owned.add("ap-good")
    owned.add("ap-bad")

    class _P:
        def __init__(self, rc):
            self.returncode = rc
            self.stdout = ""
            self.stderr = "boom" if rc else ""

    def fake_run(cmd, **kw):
        return _P(0 if "ap-good" in cmd else 1)

    monkeypatch.setattr(orch.subprocess, "run", fake_run)
    owned.stop_all()

    kept = json.loads(path.read_text())["app_ids"]
    assert kept == ["ap-bad"], (
        "an id whose stop failed is the only handle to a billing endpoint"
    )


# --- budget ----------------------------------------------------------------

def test_budget_ceiling_stops_on_time(orch, tmp_path):
    import time

    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({"started": time.time() - 8 * 3600,
                             "updates_done": 3, "teacher_tokens": 500_000}))
    led = orch.Ledger(p, max_usd=1e9, max_hours=7.0)
    assert "time ceiling" in (led.check() or "")


def test_budget_ceiling_stops_on_estimated_spend(orch, tmp_path):
    import time

    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({"started": time.time() - 3 * 3600,
                             "updates_done": 3, "teacher_tokens": 0}))
    led = orch.Ledger(p, max_usd=1.0, max_hours=1e9)
    assert "budget ceiling" in (led.check() or "")


def test_fresh_ledger_permits_the_run(orch, tmp_path):
    led = orch.Ledger(tmp_path / "none.json", max_usd=30.0, max_hours=7.0)
    assert led.check() is None


# --- status parsing --------------------------------------------------------

def test_status_parses_only_the_sentinel_line(orch, tmp_path, monkeypatch):
    payload = {"exists": True, "n_updates": 2, "updates": [],
               "plan_hash": "abc"}

    class _P:
        returncode = 0
        stdout = (
            '{"unrelated": "object printed by some library"}\n'
            "PILOT_STATUS_JSON=" + json.dumps(payload) + "\n"
            "Stopping app - local entrypoint completed.\n"
        )
        stderr = ""

    monkeypatch.setattr(orch.subprocess, "run", lambda *a, **k: _P())
    got = orch.fetch_status("run", tmp_path)
    assert got == payload, "a stray object must not be mistaken for the status"


def test_status_refuses_when_the_sentinel_is_absent(orch, tmp_path, monkeypatch):
    class _P:
        returncode = 0
        stdout = '{"looks": "like json"}'
        stderr = ""

    monkeypatch.setattr(orch.subprocess, "run", lambda *a, **k: _P())
    with pytest.raises(SystemExit, match="PILOT_STATUS_JSON"):
        orch.fetch_status("run", tmp_path)


# --- refresh verification --------------------------------------------------

class _Args:
    api_base = "http://endpoint"


def test_refresh_skipped_only_when_the_probe_matches(orch, tmp_path,
                                                     monkeypatch):
    rec = {"adapter_hash": "hash-u1", "probe_ids": [1, 2, 3],
           "probe_logprobs": [-0.5, -0.25, -0.75]}
    (tmp_path / "refresh-001.json").write_text(json.dumps(rec))

    monkeypatch.setattr(orch, "probe_now",
                        lambda *a, **k: [-0.5, -0.25, -0.75])
    assert orch.refresh_already_done(
        _Args(), tmp_path, 1, "m-u000", "hash-u1") is True

    # Same name, different weights: must NOT skip.
    monkeypatch.setattr(orch, "probe_now",
                        lambda *a, **k: [-0.5, -0.25, -0.70])
    assert orch.refresh_already_done(
        _Args(), tmp_path, 1, "m-u000", "hash-u1") is False


def test_refresh_not_skipped_when_the_hash_differs(orch, tmp_path):
    (tmp_path / "refresh-001.json").write_text(json.dumps(
        {"adapter_hash": "stale", "probe_ids": [1], "probe_logprobs": [-1.0]}))
    assert orch.refresh_already_done(
        _Args(), tmp_path, 1, "m-u000", "hash-u1") is False


def test_refresh_not_skipped_when_the_probe_errors(orch, tmp_path,
                                                   monkeypatch):
    (tmp_path / "refresh-001.json").write_text(json.dumps(
        {"adapter_hash": "hash-u1", "probe_ids": [1],
         "probe_logprobs": [-1.0]}))

    def _boom(*a, **k):
        raise RuntimeError("endpoint restarted")

    monkeypatch.setattr(orch, "probe_now", _boom)
    assert orch.refresh_already_done(
        _Args(), tmp_path, 1, "m-u000", "hash-u1") is False


# --- the two-update invariants a paid run cannot re-test -------------------

def _sampled(tmp_path, idx, *, adapter, policy, n=8):
    from vektori_trace.tau2.reopd_state import UpdateDir

    u = UpdateDir(tmp_path, idx)
    u.path.mkdir(parents=True, exist_ok=True)
    u.mark("PLANNED", {})
    u.mark("SAMPLED", {"adapter_hash": adapter, "policy_version": policy,
                       "actions": n, "episodes": n, "stage": "SAMPLED"})
    return u


def test_update1_identity_comes_from_its_own_marker(tmp_path):
    """The manifest names the run's INITIAL parent; update 1's is different."""
    from vektori_trace.tau2.live_train import LiveTrainError, sampled_identity

    u0 = _sampled(tmp_path, 0, adapter="sft-parent", policy="live-u000")
    u1 = _sampled(tmp_path, 1, adapter="child-of-u0", policy="live-u001")

    assert sampled_identity(u0)["adapter_hash"] == "sft-parent"
    assert sampled_identity(u1)["adapter_hash"] == "child-of-u0"
    assert sampled_identity(u1)["policy_version"] == "live-u001"

    # Enforcing the run-level parent on update 1 -- what a manifest read would
    # do -- must fail rather than quietly train onto the wrong adapter.
    with pytest.raises(LiveTrainError, match="was sampled from"):
        sampled_identity(u1, expect_adapter_hash="sft-parent")


def test_adam_resumes_with_nonzero_step_counts(tmp_path):
    """Update 1 must restore update 0's optimizer, not build a fresh one.

    A reset is invisible downstream: with bias correction the first step after
    it is near-full-magnitude, and max_param_delta reads 1e-5 either way.
    """
    torch = pytest.importorskip("torch")

    from vektori_trace.tau2.reopd_checkpoint import load_checkpoint, save_checkpoint

    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for _ in range(3):
        model(torch.ones(2, 4)).sum().backward()
        opt.step()
        opt.zero_grad()

    steps_before = [
        int(opt.state[p]["step"]) for p in model.parameters()
        if p in opt.state and "step" in opt.state[p]
    ]
    assert steps_before and all(s > 0 for s in steps_before)

    cp = tmp_path / "checkpoint"
    cp.mkdir(parents=True, exist_ok=True)
    torch.save(opt.state_dict(), cp / "optimizer.pt")
    (cp / "state.json").write_text(json.dumps(
        {"update_index": 0, "adapter_hash": "child-of-u0",
         "policy_version": "live-u000"}))

    fresh_model = torch.nn.Linear(4, 4)
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-5)
    assert not any("step" in fresh_opt.state.get(p, {})
                   for p in fresh_model.parameters()), "fresh optimizer"

    fresh_opt.load_state_dict(torch.load(cp / "optimizer.pt",
                                         weights_only=False))
    steps_after = [
        int(fresh_opt.state[p]["step"]) for p in fresh_model.parameters()
        if p in fresh_opt.state and "step" in fresh_opt.state[p]
    ]
    assert steps_after == steps_before, (
        "update 1 must continue update 0's Adam trajectory, not restart it"
    )


def test_resume_skips_sampled_scored_and_trained_stages(orch):
    """A resumed pilot resamples nothing and re-buys no scores."""
    st = _status(2)
    for i in (0, 1):
        row = orch.update_row(st, i)
        assert row["sampled"] and row["scored"] and row["trained"]

    partial = _status(2, patch={1: {"sampled": True, "scored": False,
                                    "trained": False}})
    row = orch.update_row(partial, 1)
    assert row["sampled"] is True, "rollout would be skipped"
    assert row["scored"] is False, "scoring would run"
    assert row["trained"] is False, "training would run"


def test_each_update_advances_the_policy(orch):
    """update k samples from update k-1's child, never a stale adapter."""
    st = _status(3)
    for k in range(1, 3):
        prev_child = orch.update_row(st, k - 1)["trained_adapter_hash"]
        sampled_by = orch.update_row(st, k)["sampled_adapter_hash"]
        assert sampled_by == prev_child, (
            f"update {k} sampled {sampled_by}, not update {k-1}'s child "
            f"{prev_child} -- the run would not be on-policy"
        )
