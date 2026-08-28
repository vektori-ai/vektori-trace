# Pilot handoff — live OPD 10×8

Branch `feat/tau2-live-opd` · HEAD `8bcb047` · 2026-08-29 ~04:50 IST

---

## 0. TEARDOWN FIRST — read this before anything else

**A running pilot means two Modal GPUs are billing.** Before ending any
session, and immediately on any failure:

```bash
cd /data/vektori-trace && .venv/bin/modal app list | grep ephemeral
# every row with Tasks>=1 is billing. Expect ZERO when the run is done.
.venv/bin/modal app stop <app-id> -y        # -y is MANDATORY; silent no-op without it
cat /data/tau2/pilot-state/owned_apps.json  # non-empty => a stop FAILED, do it by hand
```

Check from **both** the box and locally — separate Modal clients, and one can
show a stale view. Re-check after ~60 s: an app can report `stopped` while its
container drains.

The orchestrator stops the endpoint it owns on every exit path, and *keeps* any
id whose stop failed rather than forgetting it. Do not rely on that alone.

---

## 1. What is running right now

| | |
| --- | --- |
| run id | `pilot_10x8_20260829` |
| plan hash | `aa9251ccb6d566fa` (80 distinct C30 pairs) |
| parent | `3869b147ab7ce5d2` — untouched `A_sft_new` |
| host | EC2 `i-0a348ff3d7be9769a`, **SSM only**, tmux `serve` + `pilot` |
| state dir | `/data/tau2/pilot-state` |
| endpoint app | `ap-Te8Ouvb9zU3GQ3Em94ZcT6` (serving L40S, owned by the pilot) |
| started | 23:08 UTC 2026-08-28 |
| ceilings | `--max-usd 30`, `--max-hours 7` |
| spend so far tonight | ~$3.40 total (mechanism proof ~$2.75 + pilot so far) |

Expected: 10 updates × 8 episodes ≈ 4–5.5 h, **$14–22**.

### Watch it

```bash
tail -f /data/tau2/pilot-state/pilot_run.log      # stage transitions
tail -f /data/tau2/pilot-state/update-000.log     # per-update detail
cat    /data/tau2/pilot-state/ledger.json         # hours / estimated spend
tmux attach -t pilot                              # Ctrl-B D detaches WITHOUT killing
```

`pilot_env.sh` appears only when the endpoint genuinely serves — it is the
readiness signal, not the log.

---

## 2. The experiment

**10 updates × 8 episodes = 80 distinct C30 `(task, seed)` pairs**, frozen in
`docs/prereg/pilot_10x8_20260829.manifest.json` *before* update 0 ran.

Per update: `refresh → rollout → score → train`, each its own Modal function so
the training GPU is allocated for ~2 min instead of held through everything.

- **C30** (train): 30 tasks, seeds 0/1/2. No pair reused; no task repeated
  within an update.
- **S16** (held out, 16 tasks incl. `56`): `0,1,21,30,32,38,43,56,57,59,73,75,84,92,93,110`.
  Inspected **once**, after update 9. Looking earlier converts it into a
  checkpoint-selection set and destroys the comparison.
- **F38**: sealed.
- Loss: **token-mean over one global supervised-token denominator**, unchanged
  from the mechanism proof. Do not change it mid-run.

**The engineering proof (`b6c160fcf9792e92`, `01a95c2368661cd7`) trained on
57/73/75/93, which are S16.** Fine for a mechanism test, fatal for this one —
which is why the pilot restarts from the untouched parent and those two
checkpoints are evidence only, never pilot lineage.

---

## 3. Resuming after a crash

Retries are **stage-local**. Never delete an update directory; the artifacts
are the recovery mechanism.

```bash
cd /data/vektori-trace
.venv/bin/python scripts/tau2_pilot_orchestrate.py \
  --run-id pilot_10x8_20260829 --n-updates 10 \
  --api-base "$STUDENT_API_BASE" --reload-url "$STUDENT_RELOAD_URL" \
  --student-model "$STUDENT_MODEL" \
  --serve-app-id <fresh id> --own-endpoint \
  --state-dir /data/tau2/pilot-state --max-usd 30 --max-hours 7
```

It reads stage markers off the **volume** (`pilot_status`), so it resamples
nothing and re-buys no scores. `--start-at N` only if you must force it.

If the endpoint died, start a new one first and pass its **fresh** app id —
`pilot_app_id.txt` can hold a stale value from an aborted attempt.

---

## 4. Bugs found tonight — each was invisible in the logs

| bug | why it mattered |
| --- | --- |
| **Fabricated endpoint URL** (`8bcb047`) | `get_web_url()` lives on the class's Function, not the instance's bound method; the fallback *invented* a URL missing the workspace prefix, class segment and `-dev` suffix. Every request 404s while vLLM reports healthy — reads as a model failure. Now raises instead of guessing. |
| **Fresh Adam every update** (`26d2ed3`) | `one_step` hardcoded `resume_from=None`, so chaining via `--parent-override` gave correct weights and a reset optimizer. Ten "iterative" updates would be ten independent first steps. `max_param_delta` reads 1e-5 either way. |
| **Identity from the run manifest** (`221daf9`) | The manifest names the run's *initial* parent, so update k>0 got the wrong adapter hash and policy version. Now from each update's own `.SAMPLED`. |
| **Same 8 scenarios ×10** (`c7a8fda`) | `plan_update` crosses the same tasks/seeds every update; only the episode id carried the index. Was data reuse dressed as 80 episodes. |
| **Missing `--tool-call-parser`** (`b5b777c`) | vLLM 400s on *any* request carrying `tools`. `/models`, `/health` and plain completions all look fine until the first tool turn. |
| **Share gate unsatisfiable** (`324da9b`) | `1.5/n` tightens as fast as added episodes dilute; refused every batch size 4–16 when one task runs long. Now telemetry — rejecting on realized length selects against hard tasks, which is what OPD exists to learn from. |

---

## 5. Facts worth not re-deriving

- **~8.0 turns/episode**, measured over 10 real episodes. The docs' "~13" was an
  estimate. So 10×8 ≈ **640 actions**, near ReOPD's 512 — 5×8 would have been
  ~320, well short.
- **Task 93 took 46.2% of supervised tokens** in one engineering update. That is
  the gradient share; **turn share (44.1%) is a different and worse proxy** —
  task 57 was 14.7% of turns but 24.5% of gradient.
- Tinker sums token losses (`loss:sum`) and bounds length with `max_turns`,
  **not** share rejection. Our global token-mean has the same relative
  weighting; the denominator changes scale, not which task dominates.
- vLLM serves adapters **prefixed**: `a-sft-new` → `Qwen3-4B-a-sft-new`. An
  unknown name silently resolves to the base model.
- `serve_student.py` fires a real completion before declaring itself up. That
  smoke test caught the URL bug before the pilot could roll out against a 404.

---

## 6. When the run finishes

1. **Verify teardown** (§0). Zero ephemeral apps, from both clients, twice.
2. **Terminal refresh** — the loop exits after update 9 trains without serving
   that checkpoint. Run `refresh_only --update 10` before evaluating, then the
   format probes.
3. **Paired S16 evaluation** — untouched parent vs update 9, same 16 tasks,
   same 2 seeds, same generation settings, both adapters on one instance.
   Report paired per-episode outcomes and discordant-pair counts, not a
   difference of aggregates. A non-significant result is **inconclusive**, not
   "no effect" — with ~32 paired episodes only a decisive imbalance will show.
4. Do **not** infer efficacy from training telemetry. Healthy gradients prove
   the mechanism runs, nothing more.

---

## 7. Open

- **Cumulative Modal spend** never pulled from modal.com. The $30 ceiling is
  per-run and cannot see the account total.
- **Grok's box-side monitors** — read-only pollers to be committed under
  `scripts/`. `scripts/pilot_watch.sh` currently hardcodes a laptop scratchpad
  path and is **useless on the box**: fix or delete it.
- Alerting on bare `refus`/`Error` false-positives constantly — Tau2 retail
  dialogue is full of "refund", "return", "refusal". Match orchestrator strings
  (`STOPPING`, `failed (rc=`, `!!`, `Traceback`) in `pilot_run.log` only.
