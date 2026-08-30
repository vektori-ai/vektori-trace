# Watching a live-OPD pilot run

Run as `ubuntu` on the box. Everything here is read-only — no spend, no GPU.

```bash
cd /data/vektori-trace
set -a; . ./.env; set +a          # needed for any `modal` command
```

Current run: `pilot_10x8_20260829e`, chain log
`/data/tau2/pilotd-state/chain_u3to9.log`.

---

## 0. The one command

```bash
watch -n 10 'echo "=== chain ==="; tail -4 /data/tau2/pilotd-state/chain_u3to9.log; \
echo "=== newest activity ==="; ls -t /data/tau2/pilotd-state/u[0-9]_*.log 2>/dev/null | head -1 | xargs tail -2 | cut -c1-140; \
echo "=== alive ==="; pgrep -fc tau2_pilot_chain'
```

Always shows something moving, whichever stage is running.

---

## 1. Why the chain log looks frozen

`chain_u3to9.log` only writes at **stage boundaries**. The per-action progress
goes to separate files. A long silence is normal.

| Stage | Duration | Writes to chain log? | Where the live output is |
| --- | ---: | --- | --- |
| Serve endpoint | ~5 min | no | `/data/tau2/chain_u<N>_serve.log` |
| Rollout, 8 episodes | ~18–23 min | no | `u<N>_rollout.log` |
| Dry-run + gates | ~2 min | **yes**, several lines | `u<N>_dryrun.log` |
| Scoring (paid) | ~8 min | one line at the end | `u<N>_score.log` |
| Training | ~5 min | **yes**, at the end | `u<N>_train.log` |

So expect ~25 quiet minutes from `UPDATE N`, then a burst.

---

## 2. Per stage

### Sampling / rollout

```bash
# episodes as they start, and the final tally
tail -f /data/tau2/pilotd-state/u3_rollout.log | grep --line-buffered -E \
  "u00[0-9]-task[0-9]+-seed[0-9]+|rollout took|teardown|failed|discarded"

# how many episodes so far (want 8)
grep -aoE "u003-task[0-9]+-seed[0-9]+" /data/tau2/pilotd-state/u3_rollout.log | sort -u | wc -l

# the model's actual words
tail -f /data/tau2/pilotd-state/u3_rollout.log | grep --line-buffered -A2 "Role.AGENT"

# endpoint startup
tail -f /data/tau2/chain_u3_serve.log
```

Healthy finish:
`rollout took 1379.8s: 8 episodes, 85 turns, 0 failed, 0 discarded`

### Gates (free, right after rollout)

```bash
tail -20 /data/tau2/pilotd-state/chain_u3to9.log
cat /data/tau2/pilotd-state/u3_dryrun.log        # retention + accounting
cat /data/tau2/pilotd-state/u3_exclusions.log    # which actions carry markup
```

Binding rules — any one halts the chain:

```text
retention              < 90%
affected actions       > 10%
excluded reasoning     > 10% of reasoning bytes
undeclared exclusion class
repeated cap terminations        (never raise the cap)
episode spread         DIAGNOSTIC only since the 2026-08-30 amendment
```

### Scoring (paid, ~$0.13/update)

```bash
tail -f /data/tau2/pilotd-state/u3_score.log | grep --line-buffered "scoring "
```

Prints `scoring 25/85 (324s, ~188,747 teacher tokens, ~$0.0415)`.

```bash
grep -aE "teacher tokens|est cost|structural weight" /data/tau2/pilotd-state/u3_score.log | tail -3
```

### Training

```bash
tail -f /data/tau2/pilotd-state/u3_train.log | grep --line-buffered -E \
  "parent:|trained in|loss|grad_norm|adapter=|weights moved|reload_verified"
```

Healthy finish:
`trained in 306.7s; loss=0.4954 adapter=f1bf130b66bbcff8 supervised=53780`

---

## 3. Progress and money

```bash
grep -c "OK ----" /data/tau2/pilotd-state/chain_u3to9.log     # updates done
grep -E "HALT" /data/tau2/pilotd-state/chain_u3to9.log        # empty = healthy
grep -aE "est cost" /data/tau2/pilotd-state/u*_score.log      # teacher spend

.venv/bin/modal app list | grep ephemeral
#   1–2 during a stage = normal
#   0 between updates  = normal
#   anything after the chain exits = STOP IT
```

### Teardown check (do this whenever the chain ends)

```bash
pgrep -fa 'serve_student|modal run|tau2_pilot_chain'   # want empty
.venv/bin/modal app list | grep -c ephemeral           # want 0
```

If an endpoint is somehow still up:

```bash
.venv/bin/modal app stop -y <ap-XXXX>
```

---

## 4. Run state and lineage

```bash
bash scripts/pilotc_status.sh pilot_10x8_20260829e   # episodes per update
bash scripts/pilotc_status.sh pilot_10x8_20260829d   # updates 0–1

# NOTE: that script hardcodes /data/tau2/pilotc-state for scratch, so its
# "latest log" section tails an unrelated old run. The per-update episode
# table it pulls from the volume IS correct. Ignore the last section.

# stage markers for one update
.venv/bin/modal volume ls vektori-trace-adapters \
  tau2/live-opd/pilot_10x8_20260829e/update-003
```

Lineage so far — each rollout samples from the previous update's child:

```text
3869b147ab7ce5d2  (SFT parent)
  -> 073237e3fdf791b1   update 0   loss 0.5311  grad 0.6398
  -> 7161d364fd097137   update 1   loss 0.5668  grad 0.6276
  -> f1bf130b66bbcff8   update 2   loss 0.4954  grad 0.5661
```

---

## 5. Analysis (any time, free)

```bash
# advantage distribution + the Okay share
.venv/bin/python scripts/pilotd_advantages.py /tmp/sc1.jsonl /tmp/a1.jsonl 'UPDATE 1'

# skips vs the declared policy
.venv/bin/python scripts/pilotd_verify_exclusions.py /tmp/mf3.json /tmp/a1.jsonl

# one action's parse and byte spans
.venv/bin/python scripts/pilotd_diag_action.py /tmp/a1.jsonl 'u001-task4-seed1@4#0'

# a failed turn's raw generation (e.g. the update-2 cap loop)
.venv/bin/python scripts/pilotd_show_failed_turn.py /tmp/ft.jsonl 3

# score rows and fingerprint binding
.venv/bin/python scripts/pilotd_scorerow.py /tmp/sc1.jsonl /tmp/a1.jsonl
```

Pull artifacts for an update first:

```bash
R=tau2/live-opd/pilot_10x8_20260829e/update-003
.venv/bin/modal volume get vektori-trace-adapters $R/actions.jsonl /tmp/a3.jsonl --force
.venv/bin/modal volume get vektori-trace-adapters $R/scores.jsonl  /tmp/sc3.jsonl --force
```

---

## 6. If it halts

The `HALT` line names the rule. Nothing is lost: stage markers make the rerun
skip completed work — it will not re-roll episodes or re-pay the teacher.

```bash
grep -B4 "HALT" /data/tau2/pilotd-state/chain_u3to9.log
bash scripts/tau2_pilot_chain.sh pilot_10x8_20260829e <N> 9   # resume
```

A **format failure** (cap termination) is different — it needs the one
preregistered slot retry, which is a manual sequence: new run id, stage the
manifest, `retry_slot`, then resume. Stop rather than retry a second time.
