"""Tripwires around the network policy's design decisions.

The denylist is gone: Harbor's default-deny `network_mode = "allowlist"` blocks
everything not named, including hosts nobody thought of, which a denylist can
never do. These pin the harbor facts the switch depends on, so a harbor upgrade
that withdraws them fails here rather than in a mining run.
"""

from __future__ import annotations

import inspect

from harbor.environments.base import BaseEnvironment
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkMode

from vektori_trace.mining.env_guard import AGENT_ALLOWED_HOSTS, FIX_SOURCE_HOSTS


def test_local_docker_can_enforce_an_allowlist() -> None:
    """The property everything here rests on, pinned.

    `network_allowlist` is set from `_enable_egress_control`, which is on for a
    non-Windows container on Linux when the task requests egress control. This
    was *not* true on harbor 0.14, which is what the v0 plan was written
    against — which is why the plan's "switch to allowlist" was deferred, and
    why it is possible now.

    Pinned as a test rather than a note because the denylist is only justified
    while the migration is outstanding. If harbor drops the capability, this
    fails and the justification comes back.
    """
    src = inspect.getsource(DockerEnvironment.capabilities.fget)
    assert "network_allowlist=self._enable_egress_control" in src, (
        "local Docker's allowlist capability changed shape — recheck whether "
        "the env_guard denylist is still the right control"
    )

    gate = inspect.getsource(DockerEnvironment.__init__)
    assert "_egress_control_kernel_support" in gate
    assert 'sys.platform == "linux"' in gate


def test_harbor_rejects_an_unenforceable_policy_rather_than_downgrading() -> None:
    """Why a wrong choice fails loudly instead of leaving tasks unprotected —
    the property that makes a default-deny policy safe to depend on."""
    src = inspect.getsource(BaseEnvironment.validate_network_policy_support)
    assert "raise ValueError" in src
    assert "NetworkMode.ALLOWLIST" in src


def test_allowlist_is_a_real_mode() -> None:
    assert NetworkMode.ALLOWLIST.value == "allowlist"


def test_no_network_is_a_real_mode() -> None:
    """The verifier's policy. Its dependencies are in the image and the suite
    runs offline, so it needs nothing — and it is the container that decides
    the reward."""
    assert NetworkMode.NO_NETWORK.value == "no-network"


def test_docker_can_actually_disable_the_network() -> None:
    """`no-network` is refused rather than downgraded if the provider can't do
    it, so asking for it on a provider that can't is a loud failure — but on
    local Docker it must work, or every mined task fails to start."""
    src = inspect.getsource(DockerEnvironment.capabilities.fget)  # type: ignore[attr-defined]
    assert "disable_internet=self._enable_egress_control" in src


def test_egress_control_engages_for_any_non_public_policy() -> None:
    """The sidecar is what enforces this, and it is only wired in when some
    phase asks for a non-public policy. If that predicate ever narrows, tasks
    would keep loading and quietly lose their guard."""
    src = inspect.getsource(DockerEnvironment._requires_egress_control)
    assert "!= NetworkMode.PUBLIC" in src


def test_harbor_enforces_the_policy_with_a_proxy_not_etc_hosts() -> None:
    """The reason this replaces the denylist rather than joining it: services
    are routed through a sidecar (`network_mode: service:<sidecar>`), so a host
    absent from the allowlist is blocked whether or not we thought of it.
    `extra_hosts` could only ever pin names we enumerated."""
    src = inspect.getsource(DockerEnvironment._write_egress_control_services_compose_file)
    assert "network_mode" in src
    assert "service:" in src


def test_the_allowlist_never_contains_a_fix_source() -> None:
    """The one invariant that makes the whole thing worth doing. Adding
    `pypi.org` here to fix a broken agent would silently reopen the leak the
    guard exists to close — `pip download <pkg>==<fixed>` reads the fix
    straight out of the wheel."""
    for host in FIX_SOURCE_HOSTS:
        assert host not in AGENT_ALLOWED_HOSTS

    # Not just exact matches: a wildcard that swallows a fix source is the same
    # hole with extra steps.
    for pattern in (p for p in AGENT_ALLOWED_HOSTS if p.startswith("*.")):
        suffix = pattern[1:]
        for host in FIX_SOURCE_HOSTS:
            assert not host.endswith(suffix), f"{pattern} would permit {host}"


def test_the_allowlist_covers_what_an_installed_agent_needs() -> None:
    """An allowlist that blocks the agent's own registry or model API breaks
    every run — the failure mode that made this look like a trade-off."""
    assert "api.anthropic.com" in AGENT_ALLOWED_HOSTS
    assert "api.openai.com" in AGENT_ALLOWED_HOSTS
    assert "registry.npmjs.org" in AGENT_ALLOWED_HOSTS


def test_fix_sources_still_name_what_must_be_blocked() -> None:
    """No longer used to build a policy — it's the assertion side of
    `check-env`, and what "every fix source" means when the probe runs."""
    for host in ("pypi.org", "files.pythonhosted.org", "github.com", "codeload.github.com"):
        assert host in FIX_SOURCE_HOSTS
