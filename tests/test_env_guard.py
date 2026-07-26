"""Tripwires around the egress guard's design decisions.

The guard is a denylist; Harbor's default-deny `network_mode = "allowlist"` is
strictly better and, as of harbor 0.20, available on local Docker. It was not
on 0.14, which is what the v0 plan was written against. Those are facts about
harbor, not about us, so they are pinned here — the migration should be driven
by the suite rather than by whoever remembers.
"""

from __future__ import annotations

import inspect

import pytest
from harbor.environments.base import BaseEnvironment
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkMode

from vektori_trace.mining.env_guard import FIX_SOURCE_HOSTS, egress_guard_compose


def test_local_docker_can_enforce_an_allowlist() -> None:
    """The migration target, pinned.

    `network_allowlist` is set from `_enable_egress_control`, which is on for a
    non-Windows container on Linux when the task requests egress control. This
    was *not* true on harbor 0.14, which is what the v0 plan was written
    against — so the plan's "switch to allowlist, delete env_guard" is
    available now and was not then.

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
    """Why a wrong choice here fails loudly instead of leaving tasks
    unprotected — the property that makes the migration safe to attempt."""
    src = inspect.getsource(BaseEnvironment.validate_network_policy_support)
    assert "raise ValueError" in src
    assert "NetworkMode.ALLOWLIST" in src


def test_allowlist_is_a_real_mode() -> None:
    assert NetworkMode.ALLOWLIST.value == "allowlist"


def test_guard_covers_the_hosts_that_serve_a_published_fix() -> None:
    for host in ("pypi.org", "files.pythonhosted.org", "github.com", "codeload.github.com"):
        assert host in FIX_SOURCE_HOSTS


@pytest.mark.parametrize("host", FIX_SOURCE_HOSTS)
def test_every_host_is_blackholed_in_both_address_families(host: str) -> None:
    """An IPv4-only mapping leaves the AAAA record intact, which is how
    `pip download` kept reaching PyPI through a guard that looked applied."""
    compose = egress_guard_compose()
    assert f'      - "{host}:0.0.0.0"' in compose
    assert f'      - "{host}:::"' in compose


def test_guard_targets_the_agent_service() -> None:
    compose = egress_guard_compose()
    assert "services:" in compose
    assert "  main:" in compose
    assert "    extra_hosts:" in compose


def test_custom_host_list_is_honoured() -> None:
    compose = egress_guard_compose(("example.com",))
    assert '"example.com:0.0.0.0"' in compose
    assert "pypi.org" not in compose
