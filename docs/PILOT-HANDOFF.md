# Tau2 live-OPD pilot — handoff (2026-08-29)

**Stop point.** Everything below is verified. The next action is a **single
episode re-roll** (task 95 / seed 1), not an eight-episode rollout.

Authority for design and restart gates: `docs/TAU2-OPD-DEEP-DIVE.md`.
This file is the operational state and the exact commands to resume.

---

## 1. State

| | |
| --- | --- |
| Branch | `feat/tau2-live-opd` |
| Contract | parser **v3**, projection **v3**, scoring **chunk-v2** |
| Parent adapter | `3869b147ab7ce5d2` — untouched, never trained |
| Frozen plan | `aa9251ccb6d566fa`, 80 distinct C30 `(task, seed)` pairs |
| Running | nothing — 0 ephemeral Modal apps, 0 GPUs |
| Spend | ~$1.00 this session (rollout + serving; $0.0155 teacher) |

Quarantined, never a parent: checkpoint `90e4511f26247f33` (trained under
projection v1, whose dominant gradient was a serialization artifact). All v1
and v2 score rows are refused by fingerprint.

---

## 2. What the next run is

Update 0 of run `c` produced **7 sampled + 1 failed**. The seven are valid and
paid for; only task 95's slot is unusable.

```text
u000-task44-seed1   sampled   8 turns      u000-task76-seed0   sampled  10
u000-task53-seed2   sampled   8            u000-task108-seed1  sampled   8
u000-task68-seed0   sampled   9            u000-task109-seed0  sampled  14
u000-task71-seed1   sampled   9            u000-task95-seed1   FAILED    7
```

All seven share one adapter, one policy version (`live-u000`), one generation
config, `require_reasoning=True`. Rerunning them would discard valid work and
add sampling variation that answers nothing.

**So: carry the seven forward, re-roll one.**

---

## 3. Commands (run on the box, as `ubuntu`)

```bash
cd /data/vektori-trace
git -c safe.directory=/data/vektori-trace fetch -q origin feat/tau2-live-opd
git -c safe.directory=/data/vektori-trace reset --hard origin/feat/tau2-live-opd
set -a; . ./.env; set +a
```

### 3.1 Stage the destination run (free)

`retry_slot` refuses to run without the destination's frozen manifest.

```bash
cp docs/prereg/pilot_10x8_20260829d.manifest.json /tmp/md.json
.venv/bin/modal run scripts/tau2_live_opd_modal.py::stage_manifest \
  --run-id pilot_10x8_20260829d --manifest-json "$(cat /tmp/md.json)"
```

### 3.2 Carry the seven, leave task 95 empty (free)

```bash
.venv/bin/modal run scripts/tau2_live_opd_modal.py::retry_slot \
  --source-run-id pilot_10x8_20260829c \
  --dest-run-id   pilot_10x8_20260829d \
  --update 0 --episode-id u000-task95-seed1 --attempt 2
```

Writes `retry_provenance.json`. Refuses on: a non-`failed` episode, `attempt`
other than 2, a carried set that is not the planned roster minus the retried
slot, a plan-hash mismatch, a missing turn file, or any leak of the retried
episode into the destination.

### 3.3 Start the endpoint (**first spend**, L40S ≈ $1.95/h)

```bash
tmux new-session -d -s pd-serve 'cd /data/vektori-trace && set -a && . ./.env && set +a && \
  .venv/bin/python scripts/serve_student.py \
    --base-model Qwen/Qwen3-4B \
    --adapter sft=/adapters/tau2/runs/a_sft_new_ck35_r2/checkpoint-32 \
    --gpu L40S --max-model-len 24576 --max-lora-rank 16 \
    --reasoning-parser qwen3 \
    --write-env /data/tau2/pilotd_student.env \
    --write-app-id /data/tau2/pilotd_app_id.txt \
    --gpu-log /data/tau2/pilotd_gpu.jsonl --max-hours 2 \
    2>&1 | tee /data/tau2/pilotd_serve.log'

# ready when this prints a URL (~2 min):
grep STUDENT_API_BASE /data/tau2/pilotd_student.env
```

### 3.4 Re-roll ONLY task 95

The absent slot is what makes `capture_live_update` resample it; the seven
present ids are skipped.

```bash
set -a; . /data/tau2/pilotd_student.env; set +a
mkdir -p /data/tau2/pilotd-state
tmux new-session -d -s pd-u0 "cd /data/vektori-trace && set -a && . ./.env && . /data/tau2/pilotd_student.env && set +a && \
  .venv/bin/modal run scripts/tau2_live_opd_modal.py::rollout_only \
    --run-id pilot_10x8_20260829d --update 0 \
    --api-base \"\$STUDENT_API_BASE\" --student-model Qwen3-4B-sft \
    --adapter-hash-expect 3869b147ab7ce5d2 \
    > /data/tau2/pilotd-state/u0_retry.log 2>&1"

tail -f /data/tau2/pilotd-state/u0_retry.log | grep --line-buffered -E "rolling out|rollout took|Traceback|episode"
```

**Tear down the moment it finishes** — the scorer needs no GPU:

```bash
tmux kill-session -t pd-u0 2>/dev/null; tmux kill-session -t pd-serve 2>/dev/null
for a in $(.venv/bin/modal app list 2>/dev/null | grep ephemeral | grep -oE "ap-[A-Za-z0-9]+"); do
  .venv/bin/modal app stop -y "$a"; done
.venv/bin/modal app list | grep -c ephemeral      # must print 0
```

### 3.5 Free check before paying the teacher

```bash
.venv/bin/modal run scripts/tau2_live_opd_modal.py::scoring_dryrun \
  --run-id pilot_10x8_20260829d --update 0
```

Six gates must read **True**: all actions scored, token accounting complete,
supervision retained, no index-1 boundary newline, no payload skips, interior
whitespace kept. Expect ~96% retention.

### 3.6 Then the update-0 hold

```bash
.venv/bin/modal run scripts/tau2_live_opd_modal.py::rescore \
  --run-id pilot_10x8_20260829d --update 0     # ≈ $0.02
```

**Stop and inspect before any optimizer step** (§5).

### Progress at any time

```bash
bash scripts/pilotc_status.sh pilot_10x8_20260829d
```

---

## 4. What was fixed today, and why each mattered

| Defect | Consequence if unfixed | Commit |
| --- | --- | --- |
| Live path took one L_T/L_S ratio **per token**, not per chunk | Opposing gradients at exact teacher/student agreement. `[-0.5,-1.0,-1.5]` vs `L_T=-3.0` → chunk rule `[0,0,0]`, per-token `[-0.5,0,+0.5]`. Equal logprobs agree either way, which is why it hid | `4b82d09` |
| Parser required literal `</think>` | Discarded episodes whose reasoning was present and coherent | `eb2a79e` |
| Score fingerprints ignored parser/projection/thinking mode | A resume would silently reuse scores bought under a different contract | `eb2a79e` |
| Boundary `\n` carried teacher credit | Largest gradient in the batch was "delete the newline your template requires" (−15..−23, 11/11 turns) | `19ac0c2` |
| Trim applied to student side only | Teacher token straddled the payload start; retention **96.5% → 11.1%** | `45c4b46` |
| Span claimed bytes its tokens no longer covered | Failure moved to `payload_bytes_disagree`, still 11.1% | `26da903` |
| `verify_episode` required literal `</think>` | Failed tasks 68 and 108 whose reasoning **was** captured (2,193 / 2,056 chars). Update 0 was really 7/8, not 5/8 | `3a85022` |
| `_resolve_reasoning` accepted whitespace-only reasoning as `closed` | A reasoning-required run could accept a turn that reasoned about nothing — the empty SFT wrapper | `3a85022` |
| `loss`/`grad_norm` reported `None` | No per-update signal for a stop condition | `19ac0c2` |
| `--start-at` ignored `next_update` | Could skip an untrained update, leaving every later one parented on a checkpoint that never existed | `2b66a13` |

**Two bugs were caught only by the free dry-run on real captured bytes**, not by
unit tests: the one-sided normalization and the span/token disagreement. Run
§3.5 before every paid rescore.

---

## 5. The two holds (preregistered, not optional)

**After update-0 SCORED:** 100% planned coverage · no payload skips · zero
boundary-newline/markup/tool-JSON weight · exact accounting · all advantages
finite · no zero-supervision actions · negative tail semantic rather than
serialization.

**After update-1 SAMPLED:** sampled from update-0's verified child hash · valid
explicit or implicit reasoning boundaries · tools executable · no format
regression · complete roster.

Updates 2–9 are **not authorized** until both pass.

---

## 6. Retry policy (manifest `pilot_10x8_20260829d`)

Format failures (`status="failed"`) are a measurement **of the policy**:
one slot-level retry, **stop on ≥2 in any attempt, or any failure on the
retry**. Infrastructure failures (`status="discarded"`) get two retries with
backoff, then stop. Never conflated; the archive already distinguishes them.

At a 12% per-episode format-failure rate: P(≥1 in 8) = **64%**, P(≥2) = 25%,
P(both attempts fail) = **41%**. So ~2 in 3 updates fail the gate on first
attempt. That is the real tax of `require_reasoning` on this SFT lineage, and
it is a measurement, not an obstacle.

---

## 7. What this run cannot claim

It is an **instrumentation, stability and sizing** run. It cannot establish
efficacy: 16 S16 episodes cannot separate a modest improvement from sampling
variation, and there is no matched continued-SFT control. Batches are
conditioned on format validity, so it stays signal-seeking.

Do not size an extension from external rollout counts — our correlated
multi-turn Tau2 turns are not comparable to independent single-turn rollouts.
Size it from this run's measured gradient variance, policy movement and
evaluation variance. A balanced 10×8 extension to 20×8 is predeclared and
would also be exploratory.

---

## 8. Open, not fixed

- **Task 95's form is a real model failure.** 925 chars: opens `<think>`,
  reasons, then drifts into addressing the user, with no closing tag and no
  tool call. There is no non-arbitrary boundary; refusing is correct. It may
  recur on the re-roll — if it does, stop rather than resample again.
- **`</think>` is accommodated, not taught.** Markup carries zero OPD weight by
  construction, so live OPD provides no positive target for the closing tag.
  Any formatting effect is indirect.
- **HTTP 408/500** seen twice; timeout cause never correlated with request
  duration.
- Cost figures are **estimates**, never reconciled against a Modal invoice.
