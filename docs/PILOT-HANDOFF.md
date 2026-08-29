# Tau2 live-OPD pilot — handoff (2026-08-29)

**Stop point.** Update 0 is TRAINED. The next action is the **update-1
rollout**, then stop again before its scoring hold.

Design authority: `docs/TAU2-OPD-DEEP-DIVE.md`.
Preregistration: `docs/prereg/pilot_10x8_20260829d.manifest.json`.

---

## 1. State

| | |
| --- | --- |
| Run | `pilot_10x8_20260829d`, plan `aa9251ccb6d566fa`, 80 frozen pairs |
| Contract | parser **v3**, projection **v4**, scoring **chunk-v2** |
| Parent | `3869b147ab7ce5d2` (untouched SFT) |
| **Update-0 child** | **`073237e3fdf791b1`** ← update 1 must sample from this |
| Update 0 | `.PLANNED → .SAMPLED → .SCORED → .TRAINED`, checkpoint written |
| Running | nothing — 0 ephemeral apps |
| Spend | ~$1.45 (teacher $0.1187; rest serving + training GPU) |

Quarantined, never a parent: `90e4511f26247f33` (trained under projection v1).

---

## 2. Update-0 results

**Rollout** — 8/8 episodes, 75 actions, 0 failed, 0 discarded. Task 95 needed
one slot-level retry (`retry_slot`, attempt 2); the failed attempt is preserved
in run `c` and counts toward the reported format-failure rate.

**Scoring** — 75/75 actions, 539,348 teacher tokens, **$0.1187**.

```text
retention        54,359 / 59,746 = 91.0%
accounting       54,359 supervised + 5,387 excluded = 59,746   (exact)
structural weight zero
payload skips    1 — the declared u000-task76-seed0@9#0 exclusion
credits finite   54,359 / 54,359
```

**Training** — one step, cached scores only:

```text
loss             0.531058203373816
grad_norm        0.6398276090621948
parent           3869b147ab7ce5d2  ->  child 073237e3fdf791b1
weights moved    True
reload_verified  True
teacher calls    75 reused, 0 new
n_examples       75      supervised tokens 54,359
```

---

## 3. Commands — the update-1 rollout

Run **as `ubuntu` on the box**. There is no `su` in these; you are already
that user.

```bash
cd /data/vektori-trace
git -c safe.directory=/data/vektori-trace fetch -q origin feat/tau2-live-opd
git -c safe.directory=/data/vektori-trace reset --hard origin/feat/tau2-live-opd
set -a; . ./.env; set +a
```

### 3.1 Serve the UPDATE-0 CHILD (first spend, L40S ≈ $1.95/h)

Note the adapter path — the child checkpoint, **not** the SFT parent.

```bash
tmux new-session -d -s pd1-serve 'cd /data/vektori-trace && set -a && . ./.env && set +a && \
  .venv/bin/python scripts/serve_student.py \
    --base-model Qwen/Qwen3-4B \
    --adapter sft=/adapters/tau2/live-opd/pilot_10x8_20260829d/update-000/checkpoint \
    --gpu L40S --max-model-len 24576 --max-lora-rank 16 \
    --reasoning-parser qwen3 \
    --write-env /data/tau2/pilotd1_student.env \
    --write-app-id /data/tau2/pilotd1_app_id.txt \
    --gpu-log /data/tau2/pilotd1_gpu.jsonl --max-hours 2 \
    2>&1 | tee /data/tau2/pilotd1_serve.log'

# ready when this prints a URL (~2 min):
grep STUDENT_API_BASE /data/tau2/pilotd1_student.env
```

### 3.2 Roll out update 1

`--adapter-hash-expect` is the **child** hash. That is what proves update 1
sampled from update 0's checkpoint rather than the parent — the whole point of
the on-policy claim.

```bash
set -a; . /data/tau2/pilotd1_student.env; set +a
mkdir -p /data/tau2/pilotd-state
tmux new-session -d -s pd1-u1 "cd /data/vektori-trace && set -a && . ./.env && . /data/tau2/pilotd1_student.env && set +a && \
  .venv/bin/modal run scripts/tau2_live_opd_modal.py::rollout_only \
    --run-id pilot_10x8_20260829d --update 1 \
    --api-base \"\$STUDENT_API_BASE\" --student-model Qwen3-4B-sft \
    --adapter-hash-expect 073237e3fdf791b1 \
    > /data/tau2/pilotd-state/u1_rollout.log 2>&1"

tail -f /data/tau2/pilotd-state/u1_rollout.log \
  | grep --line-buffered -E "rolling out|rollout took|episode|reward|Traceback"
```

### 3.3 Tear down the moment it lands

```bash
tmux kill-session -t pd1-u1 2>/dev/null; tmux kill-session -t pd1-serve 2>/dev/null
for a in $(.venv/bin/modal app list 2>/dev/null | grep ephemeral | grep -oE "ap-[A-Za-z0-9]+"); do
  .venv/bin/modal app stop -y "$a"; done
.venv/bin/modal app list | grep -c ephemeral      # must print 0
```

### 3.4 STOP — the update-1 hold

Do not score or train. Verify first (§4), then decide.

### Progress / billing, any time

```bash
bash scripts/pilotc_status.sh pilot_10x8_20260829d
.venv/bin/modal app list | grep ephemeral
```

---

## 4. The update-1 hold

**Structural** — all must hold:

- sampled from `073237e3fdf791b1` (the verified child), not the parent;
- valid explicit or implicit reasoning boundaries;
- tools executable;
- no format regression versus update 0;
- complete planned roster (8/8).

**The `Okay` measurements** — preregistered before the step, so neither
outcome can be rationalised afterwards (manifest → `okay_token_prediction`):

- fraction of reasoning turns beginning with `Okay`;
- median student behaviour logprob for `Okay` (update 0: **−0.01057**);
- reasoning-boundary validity and the missing/empty `<think>` rate;
- whether another opener simply replaces `Okay`.

If update 1 is later scored, also compare its advantage distribution, its
share of total absolute advantage, and whether the negative tail is less
concentrated.

**Prediction on record:** `Okay` frequency or confidence declines while
boundary validity stays intact. If its tail stays dominant across *subsequent*
scored updates, the unclamped ratio may be amplifying — but one update is an
early indication, not grounds to redesign the loss. `opd_loss_max_clamp` stays
`None`; no post-hoc clamp.

---

## 5. Why `Okay` is supervised and the boundary newline is not

Both are reachable learned behaviour. An earlier draft claimed the newline was
"template-forced and unactionable" — **that was wrong**: it appears in
`sampled_token_ids` with a behaviour logprob, so the model generated it; its
probability is merely ~1.0. A genuinely template-inserted byte would live in
the rendered prompt and carry no sampled logprob.

The newline is excluded for **comparability**: Qwen and DeepSeek assign that
byte different structural roles, so its teacher score is not comparable across
the two renderings. `Okay` is retained because it is comparable authored
text — same bytes, same role, a real disagreement about style. Masking it
would be deciding which of the teacher's preferences count, which is the
opposite of what OPD is for.

Update-0 evidence: all ten most-negative supervised tokens are `Okay` at index
2, −18.2 to −22.0.

---

## 6. Fixes behind this run

| Defect | Consequence if unfixed | Commit |
| --- | --- | --- |
| One `L_T/L_S` ratio per **token**, not per chunk | Opposing gradients at exact agreement | `4b82d09` |
| Parser required literal `</think>` | Discarded episodes whose reasoning was present | `eb2a79e` |
| Fingerprints ignored parser/projection/thinking mode | A resume reuses scores from another contract | `eb2a79e` |
| Boundary `\n` carried credit | Batch's dominant gradient was a serialization artifact | `19ac0c2` |
| Trim applied to student side only | Retention **96.5% → 11.1%** | `45c4b46` |
| Span claimed bytes its tokens lost | Still 11.1%, different error | `26da903`, `bc33290` |
| `verify_episode` required literal `</think>` | Failed 2 valid episodes; update 0 was really 7/8 | `3a85022` |
| Whitespace-only reasoning accepted as `closed` | A reasoning-required run accepts empty reasoning | `3a85022` |
| `loss`/`grad_norm` reported `None` | No per-update signal for a stop condition | `19ac0c2` |
| `--start-at` ignored `next_update` | Could skip an untrained update | `2b66a13` |
| `rescore` enforced shares, training did not | Died **after** the teacher was paid | `e474457` |
| `rescore` refused its own declared skip | Update 0 failed its own preregistration | `cd325bd` |

**Reverted deliberately:** `980e06f`, multi-segment payloads. It passed every
mechanical gate at 94.8% retention and was semantically contaminated — Hermes
`<tool_call>` markup reached DeepSeek inside `reasoning_content`, and
`_locate`'s cursor-less `find()` could map identical segments to the same
occurrence. A stub-teacher dry-run cannot see either. The conservative skip
costs 4.7% of reasoning bytes; the alternative applied contaminated credit.

**Run the free dry-run before every paid rescore.** Four defects surfaced only
on real captured bytes:

```bash
.venv/bin/modal run scripts/tau2_live_opd_modal.py::scoring_dryrun \
  --run-id pilot_10x8_20260829d --update 1
```

---

## 7. Retry and stop rules (frozen)

**Format failures** (`status="failed"`) are a measurement *of the policy*: one
slot-level retry via `retry_slot`; stop on ≥2 in any attempt, or any failure
on the retry. **Infrastructure** (`status="discarded"`): two retries with
backoff, then stop. Never conflated.

**Retention:** stop if any update falls below **0.90** (update 0: 0.9098), or
if interleaved-tool exclusions exceed 2/8 episodes or 10% of reasoning bytes.

At a 12% per-episode format-failure rate: P(≥1 in 8) = 64%, P(≥2) = 25%,
P(both attempts fail) = 41%. So ~2 in 3 updates fail the gate on first attempt.
That is the tax of `require_reasoning` on this lineage — a measurement, not an
obstacle.

---

## 8. What this run cannot claim

An **instrumentation, stability and sizing** run. It cannot establish efficacy:
16 S16 episodes cannot separate a modest improvement from sampling variation,
and there is no matched continued-SFT control. Batches are conditioned on
format validity, so it stays signal-seeking.

Do not size an extension from external rollout counts — correlated multi-turn
Tau2 turns are not comparable to independent single-turn rollouts. Size it from
this run's measured gradient variance, policy movement and evaluation variance.
A balanced 10×8 extension to 20×8 is predeclared and would also be exploratory.

---

## 9. Open

- **Interleaved tool calls inside `<think>`** — a *malformed generation*, not a
  representation problem: a real tool call ends the action. The structural fix
  is to enforce the protocol at generation time and parse into ordered
  ReasoningText/ToolCall/VisibleContent nodes, not to grow the scorer into a
  parser maze. 1/75 actions so far; true frequency unknown.
- **`</think>` is accommodated, not taught.** Markup carries zero OPD weight by
  construction, so there is no positive target for the closing tag.
- **HTTP 408/500** seen twice; never correlated with request duration.
- **Costs are estimates**, never reconciled against a Modal invoice. The
  teacher figure ($0.1187) is measured; GPU time is inferred from wall clock.
