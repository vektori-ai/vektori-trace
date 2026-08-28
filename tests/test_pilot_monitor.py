"""Read-only pilot monitor: parsers, stall, alerts, no teardown authority."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "pilot_monitor", REPO / "scripts" / "pilot_monitor.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses + PEP 604 unions look up sys.modules[cls.__module__].
    sys.modules["pilot_monitor"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mon():
    return _load()


@pytest.fixture(scope="module")
def orch():
    spec = importlib.util.spec_from_file_location(
        "orch", REPO / "scripts" / "tau2_pilot_orchestrate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _snap(mon, tmp_path, *, now=1_700_000_000.0, billing=2, tmux=("serve", "pilot"),
          run_log="", serve_log="", ledger=None, status=None, env=False,
          update_log=None, age_run=60.0, age_status=60.0):
    state = tmp_path
    if ledger is not None:
        (state / "ledger.json").write_text(json.dumps(ledger))
    if status is not None:
        (state / "status.json").write_text(json.dumps(status))
        os_mtime = now - age_status
        os.utime(state / "status.json", (os_mtime, os_mtime))
    if run_log:
        (state / "pilot_run.log").write_text(run_log)
        os.utime(state / "pilot_run.log", (now - age_run, now - age_run))
    if serve_log:
        (state / "serve.log").write_text(serve_log)
    if env:
        (state / "pilot_env.sh").write_text("STUDENT_API_BASE=http://x\n")
    if update_log is not None:
        (state / "update-000.log").write_text(update_log)
        os.utime(state / "update-000.log", (now, now))
    modal = mon.ModalView(billing=billing, ephemeral=billing or 0, ids=("ap-AAA",))
    return mon.take_snapshot(
        state, now=now, modal=modal, tmux=tmux, modal_bin=Path("/dev/null"),
    )


def test_source_has_no_teardown_authority(mon):
    src = (REPO / "scripts" / "pilot_monitor.py").read_text()
    for needle in (
        "modal app stop", "tmux kill", "os.kill", "SIGKILL", "Popen.*kill",
        "app stop",
    ):
        assert needle not in src
    watch = (REPO / "scripts" / "pilot_watch.sh").read_text()
    assert "scratchpad" in watch.lower() or "retired" in watch.lower()
    assert "/tmp/claude" not in watch
    assert "pilotstate" not in watch


def test_parse_app_list_plain_table(mon):
    text = """\
 App ID                          Description     State      Tasks  Created at
 ap-abc123                       serve-student   ephemeral  1      2026-08-29
 ap-def456                       rollout_only    ephemeral  1      2026-08-29
 ap-old                          leftover        stopped    0      2026-08-28
"""
    v = mon.parse_app_list(text)
    assert v.ephemeral == 2
    assert v.billing == 2
    assert v.ids == ("ap-abc123", "ap-def456")


def test_parse_app_list_rich_and_zero_tasks(mon):
    text = (
        "│ App ID      │ Description │ State     │ Tasks │ Created at │\n"
        "│ ap-one      │ serve       │ ephemeral │ 1     │ 2026-08-29 │\n"
        "│ ap-idle     │ leftover    │ ephemeral │ 0     │ 2026-08-29 │\n"
    )
    v = mon.parse_app_list(text)
    assert v.ephemeral == 2
    assert v.billing == 1
    assert v.ids == ("ap-one", "ap-idle")


def test_parse_app_list_unknown_tasks_fail_closed(mon):
    v = mon.parse_app_list("ap-xyz  something  ephemeral  (running)\n")
    assert v.ephemeral == 1
    assert v.billing == 1


def test_parse_tmux_ls_named_sessions_only(mon):
    text = "serve: 1 windows (created Sat)\npilot: 1 windows\nother: 1 windows\n"
    assert mon.parse_tmux_ls(text) == ("serve", "pilot")
    assert mon.parse_tmux_ls("serve\npilot\n") == ("serve", "pilot")
    assert mon.parse_tmux_ls("") == ()
    assert mon.parse_tmux_ls("foo: 1 windows\n") == ()


def test_live_hours_ignore_stale_saved_field(mon):
    started = 1_700_000_000.0
    now = started + 0.5 * 3600
    view = mon.live_ledger(
        {"started": started, "hours": 0.01, "estimate_usd": 0.02,
         "max_usd": 30, "max_hours": 7, "updates_done": 0, "teacher_tokens": 0},
        now,
    )
    assert view.hours == pytest.approx(0.5, abs=1e-6)
    assert view.estimate_usd == pytest.approx(0.5 * 1.95, abs=1e-6)
    assert view.hours != 0.01


def test_live_estimate_matches_orchestrator_ledger(mon, orch, tmp_path):
    started = time.time() - 3600
    payload = {"started": started, "updates_done": 2, "teacher_tokens": 1_000_000}
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps(payload))
    led = orch.Ledger(p, max_usd=30.0, max_hours=7.0)
    now = time.time()
    view = mon.live_ledger(json.loads(p.read_text()), now)
    assert view.hours == pytest.approx(led.hours, abs=0.02)
    assert view.estimate_usd == pytest.approx(led.estimate_usd(), abs=0.05)


def test_stall_requires_both_run_log_and_trained_count_quiet(mon):
    now = 1_700_000_000.0
    assert not mon.stall_from_mtimes(
        run_mtime=now - 60, status_mtime=now - 25 * 60,
        complete=False, now=now,
    )
    assert not mon.stall_from_mtimes(
        run_mtime=now - 25 * 60, status_mtime=now - 60,
        complete=False, now=now,
    )
    assert mon.stall_from_mtimes(
        run_mtime=now - 21 * 60, status_mtime=now - 21 * 60,
        complete=False, now=now,
    )
    assert not mon.stall_from_mtimes(
        run_mtime=now - 21 * 60, status_mtime=now - 21 * 60,
        complete=True, now=now,
    )
    assert not mon.stall_from_mtimes(
        run_mtime=now - 21 * 60, status_mtime=None,
        complete=False, now=now,
    )


def test_gpu_without_tmux_is_the_critical_flag(mon, tmp_path):
    snap = _snap(
        mon, tmp_path, billing=2, tmux=(),
        run_log="=== update 0 ===\ndispatch rollout_only: --run-id x\n",
        ledger={"started": 1_700_000_000.0 - 600, "updates_done": 0,
                "max_usd": 30, "max_hours": 7, "teacher_tokens": 0},
        status={"exists": True, "n_updates": 10, "trained_updates": [],
                "updates": [], "complete": False},
    )
    flags = mon.classify(snap)
    assert any(f.code == "GPU_NO_TMUX" and f.severity == "critical" for f in flags)
    line = mon.compact_line(snap, flags)
    assert "GPU-NO-TMUX" in line
    assert mon.exit_code(flags) == 2


def test_gpus_with_tmux_are_not_the_leak(mon, tmp_path):
    snap = _snap(
        mon, tmp_path, billing=2, tmux=("serve", "pilot"),
        run_log="=== update 0 ===\ndispatch rollout_only: --run-id x\n",
        ledger={"started": 1_700_000_000.0 - 600, "updates_done": 0,
                "max_usd": 30, "max_hours": 7},
        status={"exists": True, "n_updates": 10, "trained_updates": [],
                "updates": [], "complete": False},
    )
    assert not any(f.code == "GPU_NO_TMUX" for f in mon.classify(snap))


def test_hours_warn_at_80_percent_of_ceiling(mon, tmp_path):
    now = 1_700_000_000.0
    snap = _snap(
        mon, tmp_path, now=now, billing=1, tmux=("serve", "pilot"),
        run_log="=== update 8 ===\n",
        ledger={"started": now - 5.6 * 3600, "updates_done": 8,
                "max_usd": 1e9, "max_hours": 7, "teacher_tokens": 0},
        status={"exists": True, "n_updates": 10, "trained_updates": list(range(8)),
                "updates": [{"update": i, "trained": True} for i in range(8)],
                "complete": False},
    )
    codes = {f.code for f in mon.classify(snap)}
    assert "HOURS_WARN" in codes
    assert snap.ledger.hours == pytest.approx(5.6, abs=1e-6)


def test_budget_warn_at_80_percent_of_ceiling(mon, tmp_path):
    now = 1_700_000_000.0
    # $24 / $1.95/h ≈ 12.31h of serving. Raise max_hours so this isolates spend.
    hours = 24.0 / 1.95
    snap = _snap(
        mon, tmp_path, now=now, billing=1, tmux=("serve", "pilot"),
        run_log="=== update 8 ===\n",
        ledger={"started": now - hours * 3600, "updates_done": 0,
                "max_usd": 30, "max_hours": 100, "teacher_tokens": 0},
        status={"exists": True, "n_updates": 10, "trained_updates": [],
                "updates": [], "complete": False},
    )
    codes = {f.code for f in mon.classify(snap)}
    assert "BUDGET_WARN" in codes
    assert snap.ledger.estimate_usd == pytest.approx(24.0, abs=0.02)


def test_serve_alert_refus_not_refund(mon):
    assert mon.SERVE_ALERT_RE.search("refusing to report an endpoint")
    assert mon.SERVE_ALERT_RE.search("Refusing to guess one")
    assert not mon.SERVE_ALERT_RE.search("please issue a refund")
    assert not mon.SERVE_ALERT_RE.search("the customer wants a refund today")


def test_run_alert_orchestrator_tokens_not_dialogue(mon):
    run = mon.RUN_ALERT_RE
    assert run.search("STOPPING before update 4: budget ceiling")
    assert run.search("rollout failed (rc=1) -- archived turns kept")
    assert run.search("!! 1 app id(s) NOT confirmed stopped")
    assert run.search("Traceback (most recent call last):")
    assert run.search("LiveBatchError: identity mismatch")
    assert run.search("LiveTrainError: missing optimizer")
    assert not run.search("Error: user asked to refuse a refund")
    assert not run.search("refusing to help with a return")
    assert not run.search("I will refuse that request")


def test_alert_re_never_binds_update_logs(mon):
    assert mon.alert_re_for("serve.log") is mon.SERVE_ALERT_RE
    assert mon.alert_re_for("pilot_run.log") is mon.RUN_ALERT_RE
    assert mon.alert_re_for("update-000.log") is None
    assert mon.alert_re_for("update-009.log") is None


def test_snapshot_does_not_alert_on_update_log_dialogue(mon, tmp_path):
    snap = _snap(
        mon, tmp_path, billing=1, tmux=("serve", "pilot"),
        run_log="=== update 0 ===\ndispatch rollout_only: x\n",
        serve_log="UP in 180s\nendpoint is live.\n",
        update_log="user: I refuse that. I want a refund.\nError: store closed\n",
        env=True,
        ledger={"started": 1_700_000_000.0 - 600, "updates_done": 0,
                "max_usd": 30, "max_hours": 7},
        status={"exists": True, "n_updates": 10, "trained_updates": [],
                "updates": [{"update": 0, "sampled": False, "scored": False,
                             "trained": False}],
                "complete": False},
    )
    assert snap.serve_alerts == ()
    assert snap.run_alerts == ()
    assert snap.serve_up_s == 180
    assert snap.env_present
    assert not any(f.code in ("SERVE_LOG", "RUN_LOG") for f in mon.classify(snap))


def test_snapshot_picks_stage_from_newer_update_log(mon, tmp_path):
    snap = _snap(
        mon, tmp_path, billing=2, tmux=("serve", "pilot"),
        run_log="=== update 0 ===\ndispatch rollout_only: --run-id x\n",
        update_log="rollout took 800s: 8 episodes, 40 actions\n",
        ledger={"started": 1_700_000_000.0 - 900, "updates_done": 0,
                "max_usd": 30, "max_hours": 7},
        status={"exists": True, "n_updates": 10, "trained_updates": [],
                "updates": [], "complete": False},
    )
    assert snap.current_update == 0
    assert "rollout took" in snap.last_stage


def test_stall_flag_from_old_mtimes(mon, tmp_path):
    now = 1_700_000_000.0
    snap = _snap(
        mon, tmp_path, now=now, billing=1, tmux=("serve", "pilot"),
        run_log="=== update 0 ===\ndispatch rollout_only: x\n",
        ledger={"started": now - 1800, "updates_done": 0,
                "max_usd": 30, "max_hours": 7},
        status={"exists": True, "n_updates": 10, "trained_updates": [],
                "updates": [], "complete": False},
        age_run=21 * 60,
        age_status=21 * 60,
    )
    assert any(f.code == "STALL" for f in mon.classify(snap))


def test_health_exit_healthy(mon, tmp_path):
    snap = _snap(
        mon, tmp_path, billing=2, tmux=("serve", "pilot"),
        run_log="=== update 0 ===\ndispatch rollout_only: x\n",
        serve_log="UP in 183s\n",
        env=True,
        ledger={"started": 1_700_000_000.0 - 600, "updates_done": 0,
                "max_usd": 30, "max_hours": 7},
        status={"exists": True, "n_updates": 10, "trained_updates": [],
                "updates": [{"update": 0, "sampled": False, "scored": False,
                             "trained": False}],
                "complete": False},
    )
    flags = mon.classify(snap)
    assert mon.exit_code(flags) == 0
    text = mon.health_report(snap, flags)
    assert "HEALTHY" in text
    assert "GPUs" in text


def test_alert_once_sees_serve_refus_not_update_log(mon, tmp_path, capsys):
    (tmp_path / "serve.log").write_text("refusing to report an endpoint\n")
    (tmp_path / "pilot_run.log").write_text("=== update 0 ===\n")
    (tmp_path / "update-000.log").write_text("I refuse. Refund please.\n")
    rc = mon.main(["--state-dir", str(tmp_path), "--repo", str(REPO),
                   "alert", "--once"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "serve.log" in out
    assert "refusing to report" in out
    assert "Refund please" not in out
