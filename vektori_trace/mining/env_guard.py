"""Anti-contamination egress guard for sandbox-verified tasks.

A published fix lives on the package index and the code host, so an agent with
open network egress can fetch the gold patch (and often the hidden test) for the
very repo it is being asked to fix. We saw this empirically: an agent blocked
from the web fell back to `git diff origin/main` (closed by the git-history
scrub in `build_environment_dockerfile`), and when that was closed it ran
`pip download <pkg>==<fixed>` and read the fix straight out of the wheel.

This module ships a per-task `environment/docker-compose.yaml` overlay that
Harbor merges into its compose chain. It blackholes the fix-bearing hosts
(PyPI + GitHub + their CDNs) by mapping them to 0.0.0.0, so `pip download`,
`git fetch github.com`, `curl raw.githubusercontent...` and WebFetch against
those hosts all fail, while general internet (the model API, the agent's own
installer, apt) stays reachable.

This is a *denylist*, which is the realistic control at the docker-compose
layer (`extra_hosts` can add host->ip pins but cannot default-deny). It closes
the obvious, observed leak paths and, crucially, keeps a hosted agent like
claude-code runnable (a full `allow_internet=false` block breaks the agent's
own install + API access).

Harbor's `network_mode = "allowlist"` supersedes this — migration pending
-------------------------------------------------------------------------
A default-deny allowlist is strictly better than a denylist, and the v0 plan
calls for switching to it and deleting this module. As of harbor 0.20 that is
available on local Docker: `DockerEnvironment` sets `network_allowlist` from
`_enable_egress_control`, which is on for a non-Windows container on Linux
whenever the task actually requests egress control. (It was *not* available on
0.14, which is what the plan was written against — worth knowing, because
`BaseEnvironment.validate_network_policy_support` raises rather than silently
downgrading, so on an older harbor the switch fails loudly instead of leaving
tasks unprotected.)

This module therefore stays only until the switch is made and verified in a
container: an allowlist that blocks the fix sources but also blocks the
agent's own installer or model API breaks every run, and that trade-off is
what needs measuring before the denylist comes out. `test_env_guard.py` pins
the support fact so the migration is driven by the suite rather than by
memory.

The stricter forms remain beyond that: a default-deny egress proxy, or a PyPI
mirror frozen to the task's base date so even the index lacks the fix.
"""

from __future__ import annotations

# Hosts that serve the published fix / hidden tests for an open-source repo.
# Blackholed to 0.0.0.0 in the agent's container at run time.
FIX_SOURCE_HOSTS: tuple[str, ...] = (
    "pypi.org",
    "pypi.python.org",
    "files.pythonhosted.org",
    "github.com",
    "api.github.com",
    "raw.githubusercontent.com",
    "codeload.github.com",
    "objects.githubusercontent.com",
)


def git_history_scrub(base_commit: str) -> str:
    """Return Dockerfile RUN lines that strip the repo to `base_commit`.

    The working tree may be reset to the base commit, but `.git` still holds
    the *future*: `origin/main`, tags, the fix commit, and the hidden test. An
    agent can `git diff origin/main HEAD` or `git show origin/main:<testfile>`
    to read the answer with zero network. This is a documented, repeated
    incident on SWE-bench (`git log --all | grep`). We remove the remote,
    delete every ref that points past the base, expire the reflog, and gc so
    the future commits are pruned, while keeping `base_commit` itself reachable
    (the verifier resets test files via `git checkout <base_commit> -- ...`).

    Append these lines AFTER the `git reset --hard <base_commit>` line.
    """
    return (
        f"RUN git checkout -q -B base {base_commit} \\\n"
        f"    && git remote remove origin 2>/dev/null || true\n"
        f"RUN for ref in $(git branch --format='%(refname:short)' | grep -vx base); do "
        f'git branch -D "$ref" 2>/dev/null || true; done \\\n'
        f"    && git tag -l | xargs -r git tag -d >/dev/null 2>&1 || true \\\n"
        f"    && git for-each-ref --format='%(refname)' refs/remotes 2>/dev/null "
        f"| xargs -r -n1 git update-ref -d 2>/dev/null || true \\\n"
        f"    && git reflog expire --expire=now --all 2>/dev/null || true \\\n"
        f"    && git gc --prune=now 2>/dev/null || true\n"
    )


def egress_guard_compose(hosts: tuple[str, ...] = FIX_SOURCE_HOSTS) -> str:
    """Return `environment/docker-compose.yaml` content blackholing `hosts`.

    Harbor includes a task's `environment/docker-compose.yaml` in its compose
    `-f` chain, so the `extra_hosts` entries land on the agent's `main` service.
    (Verified in a container by `vektori-trace check-env`, not assumed: the
    Dockerfile still builds and this file is merged as an overlay on top.)

    Each host is mapped twice, to `0.0.0.0` and to `::`. A single IPv4 mapping
    leaves the AAAA record untouched, and every host here has one — `pypi.org`,
    `files.pythonhosted.org` and `raw.githubusercontent.com` all resolved to
    routable IPv6 addresses through a guard that looked like it was working.
    Whether that leak is exploitable depends on whether the Docker network has
    IPv6 egress, which is a property of whoever's machine this runs on; a
    contamination control that holds only on hosts without IPv6 is not a
    control.
    """
    lines = [
        "# Auto-generated by Repo2RLEnv: anti-contamination egress guard.",
        "# Blackholes the hosts that serve this repo's published fix and hidden",
        "# tests, so an agent cannot fetch the answer (pip download / git fetch /",
        "# WebFetch against these fail). General internet stays up so the agent",
        "# can install itself and reach its model API. See pipelines/_env_guard.py.",
        "# Both families are mapped: an IPv4-only blackhole leaves AAAA intact.",
        "services:",
        "  main:",
        "    extra_hosts:",
    ]
    for h in hosts:
        lines.append(f'      - "{h}:0.0.0.0"')
        lines.append(f'      - "{h}:::"')
    return "\n".join(lines) + "\n"
