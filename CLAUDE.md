# vektori-trace

## Golden corpus — use this, nothing else

**`/data/vektori-trace/cs/corpus50_v3`** on the EC2 box is the canonical task
corpus. 60 tasks, 5 repos (click 19, jinja 16, hatch 10, prefect 9, anyio 6),
tmux baked into every environment Dockerfile, mined 2026-08-06.

Layout is **per-repo** — point tools at `<root>/<repo>/mined_tasks`, *not*
`<root>/mined_tasks`, which does not exist and will silently find nothing.

Audit: 49 pass, 11 fail. All 11 fail only `f2p_names_in_test_patch`, a
pytest-parametrize false positive (F2P names are expanded `test_foo[Arg]` while
the test patch has the base `def test_foo`).

**No tenacity, no structlog.** Deliberately dropped; jinja/prefect/hatch
replaced them. Do not reintroduce them from an old mine.

Full per-task inventory (audit status, F2P/P2P counts, legacy registry) lives in
`docs/mined-tasks-inventory.md` — gitignored, local to each checkout, so
regenerate it from the box's `mine-report.json` files if it is missing.

## Legacy — kept for provenance, never used

Each of these carries a `LEGACY.md`. Do not point `replay`, `passk`, or
`diagnose` at them:

- `/data/mining/out/*` (39 tasks, no tmux → every rollout fails)
- `cs/corpus50` (v1) — **keep**: holds `pallets__jinja-1505`, the only task with
  a clean recorded pass
- `cs/corpus50_v2`, `cs/smoke`, `cs/qwen_smoketest_tasks`
- locally: `vektori-out-*`, `qwen-stuff-14B/`, `workspace/`

## GPU and Modal spend

**Never start a Modal endpoint, allocate a GPU, or launch a pass@k / rollout
sweep without explicit per-run approval.** An approved plan containing a GPU
step is not approval to execute that step.

**Tearing things down needs no approval** — kill idle endpoints and containers
the moment a run finishes or is abandoned, proactively, every time.

## Reading a pass@k result

One sample of n yields every k ≤ n — `pass@k = 1 − C(n−c,k)/C(n,k)`
(`evaluate/passk.py`). Do not run separate sweeps per k. `pass_at_k` returns
`None` when `k > n`.

Before trusting any rate, check `passk.json` for `no_gradeable_rollouts` and
`infra_failures`, and confirm the per-rollout `parse_status` is not
`fallback_exitcode`. A verifier that crashed before collecting tests used to be
recorded as a model failure; that is fixed, but old reports on disk still carry
the bad numbers.

`--no-escalate` unless you mean it: escalation fires on `c == 0` regardless of
how small stage 1 was, turning a 4-rollout smoke test into 32 rollouts per
failing task.

## The box

EC2 `i-0a348ff3d7be9769a` (ap-south-1), reachable **only via AWS SSM** —
`aws ssm send-command --document-name AWS-RunShellScript`. No direct SSH.
Repo lives at `/data/vektori-trace`. Heredocs are unreliable over SSM and the
box's python3 has no `tomllib`; keep remote commands simple.
