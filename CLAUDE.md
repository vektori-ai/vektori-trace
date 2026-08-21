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

## The OPD plan of record

**`docs/OPD-MULTITURN-PLAN.md` is authoritative.** Read it before touching
anything OPD — the loss, alignment, teacher scoring, prefix selection, or a GPU
run. It supersedes `FINAL-PLAN.md` and `docs/OPD-RUN-PLAN.md`, which are
provenance only; `docs/OPD.md` remains a method survey, not a design.

Three things settled 2026-08-21, each of which cost a wrong turn to find:

- **Replay-prefix only.** Live multi-turn Harbor OPD is out of scope. The
  trained artifact is `v_replay`, parented on frozen `v0` — *not* on a
  live-updated adapter, because a v1-parented result cannot answer whether
  replay itself helps.
- **The loss is `chunk_opd.py`**, the semantic-prior port of arXiv:2606.09456
  (`vektori_trace/opd_reference/`, pinned at `927a8264`). `cross_kl.py`'s
  estimator B is legacy/diagnostic and `assert_chunk_loss_selected` refuses it
  on a cross-tokenizer training path.
- **Fireworks has no `/tokenize`.** Teacher ids come from the pinned local
  encoder (`encoding_dsv4` + `providers/teacher/cross.py`) and go to
  `/completions` as an integer array. §6.3 probe passed 2026-08-21.

Do not reintroduce the 256-token per-turn cap: a truncated action still aligns
and still yields a finite loss, so the error is invisible downstream.
`FireworksOPDConfig.max_new_tokens` still defaults to it, and
`validate_cross_opd_config` is what refuses it.

## The SFT plan of record

**`docs/SFT-SCRATCH-PLAN.md` is authoritative.** Read it before touching
anything SFT — dataset, tokenizer, trainer, Phase 7 gates, or a GPU run. It
supersedes `PLAN.md`, `V0_PLAN.md`, `FINAL-PLAN.md`, `docs/SFT-REPAIR-PLAN.md`
and `docs/CL-PLAN.md`, which are provenance only.

Two invariants that cost three runs to find, both verified 2026-08-18:

- Qwen3's chat template wraps `<think>\n\n</think>\n\n` around the **last**
  assistant turn only, and `enable_thinking` gates nothing but the generation
  prompt. Per-message prefix encoding is therefore **not** prefix-stable —
  labels overshoot 4 tokens into the next user turn. Supervise the last message
  only, with a two-encode equality assert.
- Thinking stays **on**. Because the template puts the wrapper in every target
  span, supervising it would train empty thinking. Masking it is the plan's
  proposal, pending explicit sign-off — see "Decisions taken here" in that file
  rather than treating it as settled.
