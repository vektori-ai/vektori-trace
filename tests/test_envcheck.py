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

from vektori_trace.envcheck import (
    PROBE_BASE_SHA,
    PROBE_FIX_SHA,
    PROBE_SENTINEL,
    build_probe_task,
    evaluate_probe,
    probe_dockerfile,
)
from vektori_trace.mining.env_guard import FIX_SOURCE_HOSTS, egress_guard_compose

# What a fully sound container reports.
CLEAN_PROBE = {
    "head_sha": PROBE_BASE_SHA,
    "all_log": "db711bf base commit",
    "remotes": "",
    "sentinel_hit": "",
    "fix_object_present": "no",
    "github_ip": "0.0.0.0",
    "pypi_ip": "0.0.0.0",
    "host_resolution": "\n".join(f"{h} -> 0.0.0.0" for h in FIX_SOURCE_HOSTS),
    "pypi_http_code": "000",
    "github_http_code": "000",
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


def test_ipv6_leak_is_caught() -> None:
    """The bug this found in the field. An IPv4-only blackhole leaves the AAAA
    record intact, so a host that looks guarded still resolves to something
    routable and `pip download` reads the fix out of the wheel."""
    leaky = "\n".join(
        f"{h} -> 0.0.0.0" if h != "pypi.org" else "pypi.org -> 2a04:4e42:600::223"
        for h in FIX_SOURCE_HOSTS
    )
    findings = _findings(host_resolution=leaky)

    assert findings["all_guarded_hosts_blackholed"] is False
    # The overlay *was* applied — the guard is partial, not absent, and the
    # two must not be conflated or the fix gets aimed at the wrong thing.
    assert findings["compose_overlay_applied"] is True


def test_reachable_fix_source_is_caught() -> None:
    """Resolution is not reachability. A real HTTP status means a real server
    answered, whatever the resolver claimed."""
    assert _findings(pypi_http_code="200")["fix_sources_unreachable"] is False
    assert _findings(github_http_code="301")["fix_sources_unreachable"] is False


@pytest.mark.parametrize("code", ["000", "", None])
def test_blocked_codes_count_as_unreachable(code) -> None:
    assert _findings(pypi_http_code=code, github_http_code=code)["fix_sources_unreachable"]


def test_both_blackhole_families_are_accepted() -> None:
    """`::` is as much a blackhole as `0.0.0.0`, and which one `getent` returns
    first is not something we control."""
    v6 = "\n".join(f"{h} -> ::" for h in FIX_SOURCE_HOSTS)
    findings = _findings(github_ip="::", host_resolution=v6)
    assert findings["compose_overlay_applied"]
    assert findings["all_guarded_hosts_blackholed"]


def test_unresolvable_host_counts_as_blackholed() -> None:
    """Not resolving at all is blocked too."""
    raw = "\n".join(f"{h} -> " for h in FIX_SOURCE_HOSTS)
    assert _findings(host_resolution=raw)["all_guarded_hosts_blackholed"]


def test_empty_resolution_report_is_not_a_pass() -> None:
    """A probe that reported nothing must not read as a clean bill of health."""
    assert _findings(host_resolution="")["all_guarded_hosts_blackholed"] is False


# ---------------------------------------------------------------------------
# The probe task is built from the real thing
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


def test_probe_task_ships_the_real_egress_guard(tmp_path: Path) -> None:
    task_dir = build_probe_task(tmp_path)
    shipped = (task_dir / "environment" / "docker-compose.yaml").read_text()
    assert shipped == egress_guard_compose()


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


def test_egress_guard_maps_every_host_in_both_families() -> None:
    compose = egress_guard_compose()
    for host in FIX_SOURCE_HOSTS:
        assert f'"{host}:0.0.0.0"' in compose
        assert f'"{host}:::"' in compose


def test_egress_guard_is_an_overlay_not_a_replacement() -> None:
    """It must not carry `build:` or `image:`. Harbor merges this after the
    build compose file, so a build key here would override the Dockerfile —
    which is exactly the failure mode the container check rules out."""
    compose = egress_guard_compose()
    assert "build:" not in compose
    assert "image:" not in compose
    assert "extra_hosts:" in compose
