# Network policy: from denylist to default-deny

**Date:** 2026-07-26 · harbor 0.20.0 · Docker + buildx 0.35.0

The v0 plan's last step-3 item: *"`network_mode = "allowlist"`, delete
`env_guard.py`. The default is `public`, so mined tasks currently have full
access to github.com — and the current denylist only works on local Docker
anyway."*

## What was blocking it, and why it wasn't

`env_guard.py`'s own docstring recorded the open question:

> an allowlist that blocks the fix sources but also blocks the agent's own
> installer or model API breaks every run, and that trade-off is what needs
> measuring before the denylist comes out.

It reads like a trade-off between contamination and a working agent. It isn't,
for two reasons that only show up in harbor's source.

**Policy is settable per phase.** `[environment]`, `[agent]` and `[verifier]`
each take their own `network_mode` and `allowed_hosts`
(`PhaseNetworkPolicyConfig`). One policy for both phases would have to be the
looser of the two — which is exactly how you end up concluding a denylist is the
only option.

**The two phases have opposite needs.**

| phase | needs | policy |
|---|---|---|
| verifier | nothing — dependencies are installed into the image at build time and the suite runs offline | `no-network` |
| agent | depends entirely on its class (below) | `allowlist` |

And the agent's needs are not what the docstring assumed:

- Harbor's own scaffolds (`terminus-2`) call the model **from the harness process
  on the host** via litellm; the container receives only shell commands. That
  container needs no egress at all.
- Installed agents (`claude-code`, `codex`) `install()` a CLI **into** the
  container and call the API from inside it. Those need the registry and the
  model endpoint — two hosts, neither of which is PyPI or GitHub.

So the allowlist grants exactly what an agent needs while denying the fix
sources, which is precisely what a denylist could not do safely. The verifier —
the container that decides the reward, and therefore the one most worth
hardening — gets nothing at all, for free.

## Why default-deny is not just tidier

The old guard shipped `environment/docker-compose.yaml` mapping eight hosts to
`0.0.0.0` and `::`. That is what `extra_hosts` can express, and it is only ever
as good as its enumeration: every CDN alias, regional mirror and IP literal not
on the list was a hole, and the list could only grow by someone thinking of
something. The IPv6 half of it was itself a bug found in the field — the guard
looked correct for months while `pypi.org`'s AAAA record stayed routable.

Harbor 0.20 routes eligible services through a **default-deny sidecar proxy**
(`network_mode: service:<sidecar>`). A host absent from the allowlist is blocked
whether or not anyone thought of it. And
`BaseEnvironment.validate_network_policy_support` **raises** rather than silently
downgrading, so a provider that cannot enforce the policy fails the task loudly
instead of running it unprotected.

## Measured in a container

`vektori-trace check-env`, against a probe built from the same constants mined
tasks use:

```text
[PASS] dockerfile_ran: HEAD=db711bf3... (expected db711bf3...)
[PASS] fix_sources_unreachable: all 8 fix-source host(s) unreachable
[PASS] allowed_host_reachable: api.openai.com -> 401
[PASS] future_commits_pruned: fix object reachable: no
[PASS] sentinel_unreachable: git log -SSECRET_FIX: <no hits>
[PASS] remote_removed: git remote -v: <none>
```

**The two network findings only mean something together.** A policy of "block
everything" passes `fix_sources_unreachable` and breaks every installed agent;
`allowed_host_reachable` is the control that catches it, the same shape as the
honest-agent control that makes the reward-hack probe meaningful.

And on a real mined task, with the verifier on `no-network`:

```json
{"reward": 1.0, "resolved": true,
 "f2p_total": 2, "f2p_passed": 2, "p2p_total": 139, "p2p_passed": 139,
 "regressions": [], "parse_status": "ok", "runner": "pytest"}
```

The suite genuinely runs offline. That was the assumption the whole design rests
on, and it is now measured rather than assumed.

## Two things only running it could have found

**`docker buildx` is a hard prerequisite.** Harbor builds the egress sidecar with
`docker buildx build --file=...`. Without the plugin the trial dies at `unknown
flag: --file` — before any container starts, with an error that names neither
buildx nor the network policy. Now documented in the README with the one-liner
that fixes it.

**A correctly blocked host was reported as reachable.** The probe ran
`curl -w '%{http_code}' ... || echo curl_failed`, and on a refused connection
curl prints `000` *and* exits non-zero — so both fired and the value was
`000curl_failed`, which matched neither the blocked set nor a real status. Every
one of the eight hosts was correctly blocked and the check said otherwise.

That is a false alarm rather than a miss, but with `mine` now refusing to replay
against a failed check, a false alarm blocks the corpus just as effectively. The
shell no longer concatenates, and `_blocked` matches by prefix so it no longer
depends on the shell not doing it.

## What `env_guard.py` still does

The module keeps `git_history_scrub` — the `.git` leak is unrelated to the
network and closed at image-build time. What went is `egress_guard_compose` and
the denylist it built. `FIX_SOURCE_HOSTS` stays, with a different job: nothing
consumes it to *build* a policy any more, it is the assertion side of
`check-env` — the list of hosts that must all be unreachable.

One invariant is worth stating plainly, and has a test: **no fix source may ever
appear in `AGENT_ALLOWED_HOSTS`**, including via a wildcard that swallows one.
Adding `pypi.org` there to unbreak an agent would silently reopen the exact leak
the guard exists to close.

## Still open

- The stricter form remains: a PyPI mirror frozen to the task's base date, so
  even the index lacks the fix. The allowlist makes it unnecessary for the hosts
  we know about; a frozen mirror would make it unnecessary in general.
- `AGENT_ALLOWED_HOSTS` is currently the union of what Claude Code and Codex
  need. It has not been exercised by a real installed-agent run under the
  policy — `terminus-2` doesn't touch it, since it calls the model from the host.
  Worth confirming before step 4's paired replay picks its scaffold.
