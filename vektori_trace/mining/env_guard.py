"""Anti-contamination guards for sandbox-verified tasks.

Two independent leaks, closed two different ways.

**The repo's own history.** The working tree sits at the base commit, but `.git`
still holds the future: `origin/main`, tags, the fix commit, the hidden test.
Reading the answer out of it needs no network at all. `git_history_scrub` prunes
it at image-build time.

**The network.** A published fix lives on the package index and the code host, so
an agent with open egress can fetch the gold patch — and often the hidden test —
for the very repo it is being asked to fix. Observed empirically: an agent
blocked from `git diff origin/main` fell back to `pip download <pkg>==<fixed>`
and read the fix straight out of the wheel.

Network policy is Harbor's, not ours
------------------------------------
This used to ship a per-task `docker-compose.yaml` mapping the fix-bearing hosts
to `0.0.0.0` and `::`. That was a *denylist*, which is what `extra_hosts` can
express, and a denylist is only ever as good as its enumeration — every CDN
alias, mirror and IP literal not on the list was a hole, and the list could only
be extended by thinking of things.

Harbor 0.20 enforces `network_mode = "allowlist"` on local Docker with a
default-deny sidecar proxy that eligible services route through
(`network_mode: service:<sidecar>`). Everything not named is blocked, including
hosts nobody thought of. `BaseEnvironment.validate_network_policy_support`
raises rather than silently downgrading, so on a provider that can't enforce it
the task fails loudly instead of running unprotected.

What made the migration look blocked, and why it wasn't
-------------------------------------------------------
The open question was: an allowlist that blocks the fix sources but also blocks
the agent's own installer or model API breaks every run. It dissolves once you
notice policy is settable **per phase** — `[environment]`, `[agent]` and
`[verifier]` each take their own — and that the two phases have completely
different needs:

* The **verifier** needs nothing. Dependencies are installed into the image at
  build time and the suite runs offline, so it gets `no-network`. This is the
  container that decides the reward, which makes it the one worth hardening
  most, and it costs nothing to harden.
* The **agent** depends on its class. Harbor's own scaffolds (`terminus-2`) call
  the model from the harness process on the host and send the container only
  shell commands, so that container needs no egress either. Installed agents
  (`claude-code`, `codex`) install a CLI *inside* the container and call the API
  from there, so they need the registry and the model endpoint — and an
  allowlist can grant exactly those two without ever granting PyPI or GitHub,
  which is precisely what a denylist could not do safely.

So the trade-off that needed measuring turned out not to be a trade-off. What
still needs measuring is that the policy actually holds in a container, which is
`vektori-trace check-env`'s job.

The stricter forms remain beyond this: a PyPI mirror frozen to the task's base
date, so even the index lacks the fix.
"""

from __future__ import annotations

# Hosts an agent must reach to function, and nothing else. Under
# `network_mode = "allowlist"` everything absent from this list is blocked by
# default — including the fix sources, which is the point: we no longer have to
# enumerate what to block, only what to permit.
#
# Deliberately absent, and never to be added: pypi.org, files.pythonhosted.org,
# github.com and their CDNs. Those serve the published fix.
AGENT_ALLOWED_HOSTS: tuple[str, ...] = (
    # Model APIs, for agents that call them from inside the container.
    "api.anthropic.com",
    "api.openai.com",
    "*.openai.azure.com",
    # Installer registries for the `installed` agent family, which npm/pip
    # themselves into the container at run time.
    "registry.npmjs.org",
    "*.registry.npmjs.org",
    # Telemetry/auth endpoints those CLIs refuse to start without.
    "statsig.anthropic.com",
    "sentry.io",
)

# The hosts a contamination check must prove are unreachable. Not a denylist —
# nothing consumes this to *build* a policy. It is the assertion side of
# `check-env`: under a correct allowlist every one of these must fail, and this
# list is what "every one" means.
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
