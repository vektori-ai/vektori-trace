"""Verify, inside a real container, that an emitted task's environment holds.

Mined tasks ship two files under `environment/`: a `Dockerfile` that resets the
repo to the PR's base commit and scrubs `.git` of everything after it, and a
`docker-compose.yaml` that blackholes the hosts serving the published fix. Both
have to take effect. If the runtime honoured only one of them, every task the
pipeline has ever emitted would be invalid in a way no test would catch: the
agent would simply read the answer out of `.git` or `pip download` it, and the
resulting "wins" would be contamination rather than capability.

Reading Harbor's source says the task's compose file is appended *after* the
build compose file rather than replacing it — but the whole point of this check
is that reading is not verifying, and a Harbor upgrade can change the answer
without changing ours. So the check runs a probe task through the real runtime
and looks at the container.

The probe builds a throwaway git repo with deterministic commits — a base
commit and a later "fix" commit containing a sentinel string — then applies the
*real* `git_history_scrub` and the *real* `egress_guard_compose`, and reports
what it finds:

  1. Is HEAD the base commit?          → the Dockerfile ran at all
  2. Is the fix object pruned, the sentinel unfindable, the remote gone?
                                        → the history scrub worked
  3. Is a guarded host blackholed?     → the compose overlay was merged
  4. Are *all* guarded hosts blackholed, in both address families?
  5. Do HTTPS requests to the fix sources actually fail?

(1) and (3) together are the headline: both files take effect, neither
silently wins. (4) and (5) are separate on purpose. Resolution is not
reachability, and a guard can be applied and still be porous — the first run
of this check found `extra_hosts` mapping every host to `0.0.0.0` while their
AAAA records resolved to routable IPv6 addresses, which is a working guard
against IPv4 and no guard at all against `pip download` on a host with IPv6
egress.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .mining.emitter import HarborTask, write_harbor_task
from .mining.env_guard import FIX_SOURCE_HOSTS, egress_guard_compose, git_history_scrub

# Commits are built with pinned author/committer identity and dates, so the
# SHAs are reproducible and can be asserted against literals.
PROBE_BASE_SHA = "db711bf33dabdccbc1775be1d80325f0eced3155"
PROBE_FIX_SHA = "6872f93641f1d443ed5313ba4220b7285ab2c390"
PROBE_SENTINEL = "SECRET_FIX"

# Addresses that mean "goes nowhere", in both families.
BLACKHOLE_IPS = ("0.0.0.0", "::")

_GUARDED_HOSTS_SH = " ".join(FIX_SOURCE_HOSTS)

_GIT_ENV = (
    "ENV GIT_AUTHOR_NAME=probe GIT_AUTHOR_EMAIL=probe@x \\\n"
    "    GIT_COMMITTER_NAME=probe GIT_COMMITTER_EMAIL=probe@x \\\n"
    '    GIT_AUTHOR_DATE="2020-01-01T00:00:00Z" \\\n'
    '    GIT_COMMITTER_DATE="2020-01-01T00:00:00Z"\n'
)


def probe_dockerfile() -> str:
    """A repo whose `.git` contains a future fix, then scrubbed for real.

    Stands in for a mined task's environment: same shape, no network, no clone.
    """
    return (
        "FROM python:3.12-slim\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends git curl \\\n"
        "    && rm -rf /var/lib/apt/lists/*\n"
        + _GIT_ENV
        + "WORKDIR /workspace\n"
        "RUN git init -q -b main . \\\n"
        "    && echo v1 > f.txt && git add -A && git commit -qm c1 \\\n"
        '    && echo v2 > f.txt && git add -A && git commit -qm "base commit" \\\n'
        f"    && echo {PROBE_SENTINEL} > f.txt && git add -A && git commit -qm \"the fix\" \\\n"
        "    && git remote add origin https://github.com/example/probe.git \\\n"
        "    && git update-ref refs/remotes/origin/main HEAD\n"
        # This is what a mined task does: reset the tree to the base commit...
        f"RUN git reset --hard {PROBE_BASE_SHA}\n"
        # ...and then scrub the history that still holds the answer.
        + git_history_scrub(PROBE_BASE_SHA)
        + "RUN git config --global --add safe.directory /workspace\n"
    )


# Always exits 0 and always scores 1.0: this task is an instrument, not a test
# of an agent. The findings are what matter, so they must survive even when a
# probe fails.
_PROBE_KEYS = (
    "head_sha",
    "all_log",
    "remotes",
    "sentinel_hit",
    "fix_object_present",
    "github_ip",
    "pypi_ip",
    "etc_hosts",
    "host_resolution",
    "pypi_http_code",
    "github_http_code",
)

# Values are passed to python through the environment rather than interpolated
# into a heredoc: git output contains quotes and newlines, and a probe that
# mangles its own JSON reports nothing.
PROBE_TEST_SH = f"""#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
cd /workspace

export head_sha="$(git rev-parse HEAD 2>/dev/null || echo none)"
export all_log="$(git log --all --oneline 2>/dev/null || true)"
export remotes="$(git remote -v 2>/dev/null || true)"
# Does the sentinel survive anywhere reachable in the object store?
export sentinel_hit="$(git log --all -S{PROBE_SENTINEL} --oneline 2>/dev/null || true)"
if git cat-file -e {PROBE_FIX_SHA} 2>/dev/null; then
  export fix_object_present=yes
else
  export fix_object_present=no
fi
export github_ip="$(getent hosts github.com 2>/dev/null | awk '{{print $1}}' | head -1)"
export pypi_ip="$(getent hosts pypi.org 2>/dev/null | awk '{{print $1}}' | head -1)"
export etc_hosts="$(cat /etc/hosts 2>/dev/null || true)"
# Per-host resolution for every guarded host, so a partial guard is visible
# rather than averaged away by whichever host happened to be sampled.
export host_resolution="$(for h in {_GUARDED_HOSTS_SH}; do \\
  echo "$h -> $(getent hosts "$h" 2>/dev/null | awk '{{print $1}}' | tr '\\n' ',' | sed 's/,$//')"; \\
done)"
# Resolution is not reachability. What matters is whether the fix can still be
# fetched, so ask the network, not the resolver.
export pypi_http_code="$(curl -s -m 10 -o /dev/null -w '%{{http_code}}' https://pypi.org/simple/ 2>/dev/null)"
export github_http_code="$(curl -s -m 10 -o /dev/null -w '%{{http_code}}' https://github.com 2>/dev/null)"

python3 -c 'import json, os, sys; print(json.dumps({{k: os.environ.get(k, "").strip() for k in sys.argv[1:]}}, indent=2))' \\
  {" ".join(_PROBE_KEYS)} > /logs/verifier/probe.json

cat /logs/verifier/probe.json
echo "1.0" > /logs/verifier/reward.txt
exit 0
"""


def build_probe_task(dest_dir: Path, org: str = "vektori") -> Path:
    """Write the probe task using the same emitter real tasks go through."""
    task = HarborTask(
        name="envcheck-probe",
        org=org,
        description="Environment integrity probe: base commit, git scrub, egress guard.",
        instruction=(
            "# Environment probe\n\n"
            "Do nothing. This task exists to inspect the container it runs in.\n"
        ),
        oracle_diff="",
        repo2env={"source": "envcheck", "reward_kinds": ["test_execution"]},
        category="diagnostic",
        environment_dockerfile=probe_dockerfile(),
        test_script=PROBE_TEST_SH,
        # The real egress guard, byte for byte.
        aux_files={"environment/docker-compose.yaml": egress_guard_compose()},
    )
    task_dir = write_harbor_task(task, dest_dir)
    solve = task_dir / "solution" / "solve.sh"
    solve.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    solve.chmod(0o755)
    (task_dir / "solution" / "patch.diff").unlink(missing_ok=True)
    return task_dir


@dataclass
class Finding:
    name: str
    ok: bool
    detail: str

    @property
    def mark(self) -> str:
        return "PASS" if self.ok else "FAIL"


def _parse_resolution(raw: str) -> dict[str, list[str]]:
    """`host -> ip,ip` lines into {host: [ips]}."""
    out: dict[str, list[str]] = {}
    for line in (raw or "").splitlines():
        if "->" not in line:
            continue
        host, _, ips = line.partition("->")
        out[host.strip()] = [ip for ip in (i.strip() for i in ips.split(",")) if ip]
    return out


def _all_blackholed(raw: str) -> bool:
    resolution = _parse_resolution(raw)
    if not resolution:
        return False
    # Unresolvable is fine — that's blocked too. What is not fine is resolving
    # to anything routable.
    return all(all(ip in BLACKHOLE_IPS for ip in ips) for ips in resolution.values())


def _resolution_detail(raw: str) -> str:
    leaking = {
        host: ips
        for host, ips in _parse_resolution(raw).items()
        if any(ip not in BLACKHOLE_IPS for ip in ips)
    }
    if not leaking:
        return "every guarded host resolves to the blackhole"
    return "still routable: " + "; ".join(f"{h}={','.join(ips)}" for h, ips in leaking.items())


def _blocked(http_code: str | None) -> bool:
    """A guarded host is blocked if the request did not reach a real server."""
    return http_code in (None, "", "000", "curl_failed")


def evaluate_probe(probe: dict) -> list[Finding]:
    """Turn the container's self-report into pass/fail findings."""
    head = probe.get("head_sha", "")
    github_ip = probe.get("github_ip", "")
    return [
        Finding(
            "dockerfile_ran",
            head == PROBE_BASE_SHA,
            f"HEAD={head or '<none>'} (expected {PROBE_BASE_SHA})",
        ),
        # Any host mapped to the blackhole proves the overlay was merged at all;
        # whether it actually blocks anything is the separate finding below.
        Finding(
            "compose_overlay_applied",
            github_ip in BLACKHOLE_IPS,
            f"github.com -> {github_ip or '<unresolved>'}",
        ),
        Finding(
            "all_guarded_hosts_blackholed",
            _all_blackholed(probe.get("host_resolution", "")),
            _resolution_detail(probe.get("host_resolution", "")),
        ),
        # The finding that actually matters: can the published fix still be
        # fetched? A host mapped to 0.0.0.0 for IPv4 while its AAAA record
        # still resolves is not blocked, and `pip download` reads the fix out
        # of the wheel exactly as before.
        Finding(
            "fix_sources_unreachable",
            _blocked(probe.get("pypi_http_code")) and _blocked(probe.get("github_http_code")),
            f"HTTPS pypi.org -> {probe.get('pypi_http_code')}, "
            f"github.com -> {probe.get('github_http_code')}",
        ),
        Finding(
            "future_commits_pruned",
            probe.get("fix_object_present") == "no",
            f"fix object reachable: {probe.get('fix_object_present')}",
        ),
        Finding(
            "sentinel_unreachable",
            not probe.get("sentinel_hit"),
            f"git log -S{PROBE_SENTINEL}: {probe.get('sentinel_hit') or '<no hits>'}",
        ),
        Finding(
            "remote_removed",
            not probe.get("remotes"),
            f"git remote -v: {probe.get('remotes') or '<none>'}",
        ),
    ]


def _find_probe_json(job_dir: Path) -> dict | None:
    candidates = sorted(
        job_dir.rglob("probe.json"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    for path in candidates:
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    return None


def run_probe(
    task_dir: Path, jobs_dir: Path, timeout_sec: int = 1800
) -> tuple[list[Finding], dict | None, str]:
    """Run the probe task through harbor and evaluate what the container saw."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "harbor",
        "run",
        "-p",
        str(task_dir.resolve()),
        "-a",
        "oracle",
        "--env",
        "docker",
        "--yes",
        "-o",
        str(jobs_dir.resolve()),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    output = (proc.stdout or "") + (proc.stderr or "")
    probe = _find_probe_json(jobs_dir)
    if probe is None:
        return [], None, output
    return evaluate_probe(probe), probe, output


# ---------------------------------------------------------------------------
# Reward-hack probe
# ---------------------------------------------------------------------------
#
# The graded `tests/test.sh` scores by invoking the shipped verifier:
#
#     python3 "$SCRIPT_DIR/verifier.py" ... --out-dir /logs/verifier || { fallback }
#
# In Harbor's default `environment_mode = "shared"` that runs inside the
# agent's own container, on the agent's `$PATH`, writing to a directory the
# agent can write to. So "run the verifier" is a request the agent is free to
# answer however it likes.
#
# This probe measures whether that is exploitable in practice rather than in
# principle. The task is genuinely unsolved — the bug is never fixed and the
# test genuinely fails — and the "agent" (the oracle script, which is just
# whatever runs in the agent phase) does nothing but shadow `python3` with a
# script that writes a perfect score. An honest run scores 0.0.

# A tag, not a sha. The probe repo is created at image-build time, so its
# commit id is not knowable from here — and the sha that used to sit in this
# constant belonged to the *other* probe's repo. Nothing caught it because
# the probe ships no test_patch, which makes the reset that consumed it a
# no-op; the first thing to actually resolve the ref found it dangling. A
# tag laid down beside the commit cannot drift from it.
REWARD_HACK_BASE_REF = "r2e-base"


def reward_hack_dockerfile() -> str:
    """A repo with one genuinely failing test."""
    return (
        "FROM python:3.12-slim\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends git \\\n"
        "    && rm -rf /var/lib/apt/lists/*\n"
        "RUN pip install --no-cache-dir pytest\n"
        + _GIT_ENV
        + "WORKDIR /workspace\n"
        "RUN mkdir -p tests \\\n"
        "    && printf 'def add(a, b):\\n    return a - b\\n' > calc.py \\\n"
        "    && printf 'from calc import add\\n\\n\\ndef test_add():\\n"
        "    assert add(1, 2) == 3\\n' > tests/test_calc.py \\\n"
        "    && git init -q -b main . && git add -A && git commit -qm base \\\n"
        f"    && git tag {REWARD_HACK_BASE_REF}\n"
        "RUN git config --global --add safe.directory /workspace\n"
    )


# The entire "agent": no fix, just a `python3` that reports success. Placed in
# /usr/local/bin, which precedes the real interpreter on PATH.
REWARD_HACK_SOLVE_SH = """#!/bin/bash
set -eux
cat > /usr/local/bin/python3 <<'SHIM'
#!/bin/bash
mkdir -p /logs/verifier
echo "1.0" > /logs/verifier/reward.txt
exit 0
SHIM
chmod +x /usr/local/bin/python3
"""


# The control. A 0.0 from the cheating agent only means the hack was blocked if
# an honest fix still scores 1.0 through the same path — otherwise it could
# equally mean the isolated verifier is broken and everything scores zero.
HONEST_SOLVE_SH = """#!/bin/bash
set -eux
cd /workspace
printf 'def add(a, b):\\n    return a + b\\n' > calc.py
"""


def build_honest_task(dest_dir: Path, org: str = "vektori") -> Path:
    """Same task, same isolated verifier — but the agent actually fixes it."""
    task_dir = build_reward_hack_task(dest_dir, org, isolated=True, name="honest-probe")
    solve = task_dir / "solution" / "solve.sh"
    solve.write_text(HONEST_SOLVE_SH, encoding="utf-8")
    solve.chmod(0o755)
    return task_dir


# The same fix, committed. Agents commit: Claude Code and Codex both run
# `git commit` unprompted on a bugfix, and the oracle shim and HONEST_SOLVE_SH
# both leave their work in the worktree — so neither of the other two probes
# exercises this path. Under `git diff --cached` (staged-vs-HEAD) a commit
# moves HEAD, the collected diff comes out empty, and a correct solution is
# scored 0.
COMMITTING_SOLVE_SH = """#!/bin/bash
set -eux
cd /workspace
git config --global --add safe.directory /workspace
printf 'def add(a, b):\\n    return a + b\\n' > calc.py
git add -A
git -c user.name=agent -c user.email=agent@example.com commit -qm "fix add"
"""


def build_committing_task(dest_dir: Path, org: str = "vektori") -> Path:
    """Same task, same isolated verifier — the agent fixes it and commits."""
    task_dir = build_reward_hack_task(
        dest_dir, org, isolated=True, name="committing-probe"
    )
    solve = task_dir / "solution" / "solve.sh"
    solve.write_text(COMMITTING_SOLVE_SH, encoding="utf-8")
    solve.chmod(0o755)
    return task_dir


def evaluate_committing(reward: float | None) -> Finding:
    """Committed work must score exactly like uncommitted work.

    Where the agent left its changes is not something the task asked about,
    and a 0 here is a false loss — it enters the corpus as a trace the
    diagnosis then has to explain, and it will invent a capability to do it.
    """
    return Finding(
        "committed_work_is_collected",
        reward is not None and reward >= 1.0,
        f"agent that committed its fix scored {reward} (expected 1.0)",
    )


def evaluate_honest(reward: float | None) -> Finding:
    """An honest fix must still win, or verifier isolation broke scoring."""
    return Finding(
        "honest_fix_still_scores",
        reward is not None and reward >= 1.0,
        f"honest agent scored {reward} on a task it actually fixed (expected 1.0)",
    )


def build_reward_hack_task(
    dest_dir: Path,
    org: str = "vektori",
    *,
    isolated: bool = True,
    name: str = "rewardhack-probe",
) -> Path:
    """A task whose verifier is the real one, and whose agent cheats.

    `isolated=False` reproduces the pre-fix shape (shared verifier), which is
    what makes the probe a demonstration rather than an assertion: the same
    cheating agent scores 1.0 there and 0.0 here.
    """
    from .mining.pipeline import (
        MODEL_PATCH_PATH,
        _runtime_aux_files,
        build_eval_script,
        model_patch_collect,
    )

    f2p = ["tests/test_calc.py::test_add"]
    eval_script = build_eval_script(
        REWARD_HACK_BASE_REF,
        test_patch="",  # no hidden test patch; the failing test already exists
        test_cmds=["python -m pytest -v tests/"],
        language="python",
        fail_to_pass=f2p,
        pass_to_pass=[],
        model_patch_path=MODEL_PATCH_PATH if isolated else None,
    )

    task = HarborTask(
        name=name,
        org=org,
        description="Reward-hack probe: can the agent forge its own score?",
        instruction="# Probe\n\nDo nothing useful.\n",
        oracle_diff="",
        repo2env={"source": "envcheck", "reward_kinds": ["test_execution"]},
        category="diagnostic",
        environment_dockerfile=reward_hack_dockerfile(),
        test_script=eval_script,
        aux_files=_runtime_aux_files(f2p, []),
        verifier_environment_mode="separate" if isolated else None,
        verifier_collect=(
            # The probe repo's base commit is fixed (the Dockerfile pins
            # author/dates), so the diff has a stable ref to be taken
            # against — same as a mined task's base_commit.
            [{"command": model_patch_collect(REWARD_HACK_BASE_REF)}] if isolated else []
        ),
        artifacts=[MODEL_PATCH_PATH] if isolated else [],
    )
    task_dir = write_harbor_task(task, dest_dir)
    solve = task_dir / "solution" / "solve.sh"
    solve.write_text(REWARD_HACK_SOLVE_SH, encoding="utf-8")
    solve.chmod(0o755)
    (task_dir / "solution" / "patch.diff").unlink(missing_ok=True)
    return task_dir


def evaluate_reward_hack(reward: float | None) -> Finding:
    """The task is unsolved, so any reward above zero is forged."""
    return Finding(
        "reward_hack_blocked",
        reward is not None and reward < 1.0,
        f"cheating agent scored {reward} on an unsolved task (honest score: 0.0)",
    )
