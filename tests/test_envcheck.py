"""The environment-integrity probe and its verdicts.

The probe itself needs Docker and is exercised by `vektori-trace check-env`.
What's tested here is everything around it: that the probe task is built from
the *real* guard functions rather than a copy that could drift, and that the
findings say what the container actually reported — a probe that grades itself
leniently is worse than no probe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vektori_trace.mining.env_guard import AGENT_ALLOWED_HOSTS, FIX_SOURCE_HOSTS
from vektori_trace.runtime.envcheck import (
    PROBE_ALLOWED_HOST,
    PROBE_BASE_SHA,
    PROBE_FIX_SHA,
    PROBE_SENTINEL,
    build_probe_task,
    evaluate_probe,
    probe_dockerfile,
)

# What a fully sound container reports.
CLEAN_PROBE = {
    "head_sha": PROBE_BASE_SHA,
    "all_log": "db711bf base commit",
    "remotes": "",
    "sentinel_hit": "",
    "fix_object_present": "no",
    # Under an allowlist the sidecar refuses the connection; the name may well
    # still resolve. Resolution is reported but no longer decides anything.
    "host_resolution": "\n".join(f"{h} -> 140.82.121.4" for h in FIX_SOURCE_HOSTS),
    "fix_source_http": "\n".join(f"{h} -> 000" for h in FIX_SOURCE_HOSTS),
    "allowed_http_code": "401",
}


def _findings(**overrides) -> dict[str, bool]:
    probe = {**CLEAN_PROBE, **overrides}
    return {f.name: f.ok for f in evaluate_probe(probe)}


def test_clean_container_passes_everything() -> None:
    assert all(_findings().values())


# ---------------------------------------------------------------------------
# The findings must actually fail
# ---------------------------------------------------------------------------


def test_dockerfile_not_running_is_caught() -> None:
    """The scenario the whole check exists for: if the compose file replaced
    the Dockerfile there'd be no repo, no base commit, and every emitted task
    would be invalid."""
    assert _findings(head_sha="none")["dockerfile_ran"] is False
    assert _findings(head_sha="deadbeef")["dockerfile_ran"] is False


def test_unscrubbed_history_is_caught() -> None:
    assert _findings(fix_object_present="yes")["future_commits_pruned"] is False
    assert _findings(sentinel_hit="6872f93 the fix")["sentinel_unreachable"] is False


def test_surviving_remote_is_caught() -> None:
    assert _findings(remotes="origin\thttps://github.com/x/y.git (fetch)")["remote_removed"] is False


# ---------------------------------------------------------------------------


def test_dockerfile_uses_the_real_history_scrub() -> None:
    """A probe that tested a copy of the scrub would keep passing after the
    real one broke."""
    from vektori_trace.mining.env_guard import git_history_scrub

    dockerfile = probe_dockerfile()
    for line in git_history_scrub(PROBE_BASE_SHA).strip().splitlines():
        assert line in dockerfile


def test_dockerfile_plants_a_fix_the_scrub_has_to_remove() -> None:
    dockerfile = probe_dockerfile()
    assert PROBE_SENTINEL in dockerfile
    assert f"git reset --hard {PROBE_BASE_SHA}" in dockerfile
    assert "git remote add origin" in dockerfile


def test_probe_task_declares_the_real_allowlist(tmp_path: Path) -> None:
    """From the same constant mined tasks use, not a copy that can drift from
    what ships — the probe has to be testing the policy we actually emit."""
    import tomllib

    task_dir = build_probe_task(tmp_path)
    cfg = tomllib.loads((task_dir / "task.toml").read_text())

    assert cfg["environment"]["network_mode"] == "allowlist"
    assert cfg["environment"]["allowed_hosts"] == list(AGENT_ALLOWED_HOSTS)
    assert cfg["agent"]["network_mode"] == "allowlist"


def test_probe_task_no_longer_ships_a_compose_overlay(tmp_path: Path) -> None:
    """The denylist is gone. Leaving the overlay behind would put a second,
    weaker control in the chain and make it ambiguous which one a passing
    check-env was actually measuring."""
    task_dir = build_probe_task(tmp_path)
    assert not (task_dir / "environment" / "docker-compose.yaml").exists()


def test_probe_task_is_a_complete_harbor_task(tmp_path: Path) -> None:
    task_dir = build_probe_task(tmp_path)
    assert (task_dir / "task.toml").exists()
    assert (task_dir / "environment" / "Dockerfile").exists()
    assert (task_dir / "tests" / "test.sh").exists()
    assert (task_dir / "solution" / "solve.sh").stat().st_mode & 0o111
    assert not (task_dir / "solution" / "patch.diff").exists()


def test_probe_test_script_reports_even_when_checks_fail(tmp_path: Path) -> None:
    """The probe is an instrument. It always exits 0 and always scores 1.0, so
    a failed check produces findings instead of a dead job with no report."""
    task_dir = build_probe_task(tmp_path)
    test_sh = (task_dir / "tests" / "test.sh").read_text()
    assert "exit 0" in test_sh
    assert 'echo "1.0" > /logs/verifier/reward.txt' in test_sh
    assert PROBE_FIX_SHA in test_sh


# ---------------------------------------------------------------------------
# The guard itself
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The network policy, as the probe sees it
# ---------------------------------------------------------------------------


def test_a_reachable_fix_source_is_caught() -> None:
    """One host answering is the whole leak: `pip download <pkg>==<fixed>` only
    needs files.pythonhosted.org."""
    leaky = "\n".join(
        f"{h} -> {'200' if h == 'files.pythonhosted.org' else '000'}" for h in FIX_SOURCE_HOSTS
    )
    findings = _findings(fix_source_http=leaky)

    assert findings["fix_sources_unreachable"] is False


@pytest.mark.parametrize("code", ["200", "301", "403", "404"])
def test_any_real_status_counts_as_reachable(code: str) -> None:
    """403 and 404 are servers answering. A server that answers can serve a
    wheel — only a failed connection (000) is a block."""
    raw = "\n".join(f"{h} -> {code}" for h in FIX_SOURCE_HOSTS)

    assert _findings(fix_source_http=raw)["fix_sources_unreachable"] is False


@pytest.mark.parametrize("code", ["000", "", "curl_failed"])
def test_failed_connections_count_as_blocked(code: str) -> None:
    raw = "\n".join(f"{h} -> {code}" for h in FIX_SOURCE_HOSTS)

    assert _findings(fix_source_http=raw)["fix_sources_unreachable"] is True


def test_an_empty_report_is_not_a_pass() -> None:
    """Nothing measured is not the same as nothing reachable."""
    assert _findings(fix_source_http="")["fix_sources_unreachable"] is False


def test_blocking_everything_fails_the_control() -> None:
    """The finding that stops "block the whole internet" from looking like a
    correct policy. It passes the contamination check and breaks every
    installed agent, which npm-installs itself and calls its model API from
    inside the container."""
    findings = _findings(allowed_http_code="000")

    assert findings["fix_sources_unreachable"] is True
    assert findings["allowed_host_reachable"] is False


def test_the_control_passes_on_any_answer_from_the_allowed_host() -> None:
    """401 is what an unauthenticated call to the model API returns, and it
    proves reachability exactly as well as 200 would."""
    for code in ("200", "401", "404"):
        assert _findings(allowed_http_code=code)["allowed_host_reachable"] is True


def test_the_control_host_is_actually_on_the_allowlist() -> None:
    """A control that names a host the policy never permitted would fail for
    the wrong reason and read as a broken allowlist."""
    assert PROBE_ALLOWED_HOST in AGENT_ALLOWED_HOSTS


def test_the_probe_measures_every_fix_source_individually(tmp_path: Path) -> None:
    """A partial policy is a different bug from an absent one, and sampling two
    hosts averages the difference away."""
    task_dir = build_probe_task(tmp_path)
    test_sh = (task_dir / "tests" / "test.sh").read_text()

    for host in FIX_SOURCE_HOSTS:
        assert host in test_sh


def test_a_concatenated_curl_result_is_still_read_as_blocked() -> None:
    """`curl -w '%{http_code}'` prints `000` *and* exits non-zero on a refused
    connection, so a `|| echo ...` fallback concatenated onto it. A real
    container reported every correctly-blocked host as `000curl_failed`, and
    the check called that reachable — a false alarm on a sound policy, which
    with `mine` refusing to replay on a failed check is as costly as a miss."""
    raw = "\n".join(f"{h} -> 000curl_failed" for h in FIX_SOURCE_HOSTS)

    assert _findings(fix_source_http=raw)["fix_sources_unreachable"] is True


def test_the_probe_does_not_concatenate_onto_a_curl_code(tmp_path: Path) -> None:
    """Fixed at the source too, so the parser's tolerance is a backstop rather
    than the only thing standing between us and a wrong verdict."""
    task_dir = build_probe_task(tmp_path)
    test_sh = (task_dir / "tests" / "test.sh").read_text()

    assert "|| echo curl_failed" not in test_sh
