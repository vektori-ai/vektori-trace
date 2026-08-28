"""The Fireworks parity probe: gates, checks, and archive hygiene.

Mocked end to end -- no network, no key, no spend.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "scripts" / "tau2_fireworks_parity_probe.py"


def _run(*args: str, env=None):
    return subprocess.run(
        [sys.executable, str(PROBE), *args],
        capture_output=True, text=True, env=env, cwd=str(REPO),
    )


def test_refuses_to_send_without_yes(tmp_path):
    r = _run("--out", str(tmp_path))
    assert r.returncode != 0
    assert "refusing to send a paid request" in (r.stdout + r.stderr)


def test_dry_run_sends_nothing_and_archives(tmp_path):
    r = _run("--dry-run", "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr[-800:]
    files = list(tmp_path.glob("parity_*.json"))
    assert len(files) == 1
    rep = json.loads(files[0].read_text())
    assert rep["mode"] == "dry-run"
    assert rep["thinking_mode"] == "thinking"
    assert "score_ids" in rep["route"]
    assert set(rep["cases"]) == {"reasoning_plus_content", "reasoning_plus_tool_call"}


def test_offline_checks_cover_the_known_failure(tmp_path):
    """`</think><think>` is the duplicated boundary that made the naive
    thinking-mode fix fail. The probe must assert its absence explicitly."""
    r = _run("--dry-run", "--out", str(tmp_path))
    rep = json.loads(list(tmp_path.glob("parity_*.json"))[0].read_text())
    for case in rep["cases"].values():
        checks = case["offline_checks"]
        assert checks["no_duplicated_think_boundary"] is True
        assert checks["joint_extends_prefix"] is True
        assert checks["action_is_non_empty"] is True


def test_prefix_ends_at_the_thinking_boundary(tmp_path):
    """In thinking mode the teacher prompt must end at `<think>`, not
    `</think>` -- the chat-mode default is what put a closed reasoning block
    in front of an action that opens one."""
    r = _run("--dry-run", "--out", str(tmp_path))
    rep = json.loads(list(tmp_path.glob("parity_*.json"))[0].read_text())
    for case in rep["cases"].values():
        assert case["prefix_tail"].endswith("<think>")


def test_archive_never_contains_the_api_key(tmp_path, monkeypatch):
    secret = "fw_SECRET_DO_NOT_LEAK_12345"
    import os
    env = dict(os.environ, FIREWORKS_API_KEY=secret)
    r = _run("--dry-run", "--out", str(tmp_path), env=env)
    assert r.returncode == 0
    blob = list(tmp_path.glob("parity_*.json"))[0].read_text()
    assert secret not in blob
    assert secret not in r.stdout


def test_paid_mode_requires_a_key(tmp_path, monkeypatch):
    import os
    env = {k: v for k, v in os.environ.items() if k != "FIREWORKS_API_KEY"}
    r = _run("--yes", "--out", str(tmp_path), env=env)
    assert r.returncode != 0
    assert "FIREWORKS_API_KEY is not set" in (r.stdout + r.stderr)
