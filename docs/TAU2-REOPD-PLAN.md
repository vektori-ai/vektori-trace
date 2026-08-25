# Tau2 retail ReOPD — the run plan

Rewritten 2026-08-26. The previous version of this file described telecom,
Qwen3-14B, official `train`/`test` partitions and A100 training. None of that is
this experiment; it predated the corpus, the split and the model choice, and it
was the most discoverable ReOPD document in the repo, so it is replaced rather
than marked superseded.

`docs/CLEA-QWEN3-4B-EXPERIMENT-V2.md` remains the plan of record for the
experiment design. This file is the *operational* plan for the ReOPD arm: what
runs, in what order, against which frozen artifacts, and what must be true
before anything is paid for.

**No GPU, endpoint or paid teacher call is authorized by this file.**

## 1. The question this arm answers

> Starting from the same Tau2 SFT checkpoint and using the same expert replay
> states, does replay OPD outperform compute-matched continued SFT on unseen
> Tau2 retail tasks?

The claim is `A_reopd` vs `A_sft_new`. It is **not** `A_reopd` vs `A_warm`:
that would compare 512 extra exposures against none, which says nothing about
the objective. `A_warm` is the shared floor both branches start from.

## 2. What is fixed

| | |
|---|---|
| Domain | Tau2 **retail** (not telecom — telecom is the later CL extension) |
| Student | Qwen3-4B + LoRA, parent **CK35** (`tau2/runs/a_warm_20260825_003343/checkpoint-35`) |
| Teacher | DeepSeek V4 Flash, `accounts/fireworks/models/deepseek-v4-flash-0731` |
| Split | W30 / C30 / S16 / F38 = 114, manifest `b741bfceb1f3d027` |
| Prefix pool | **289 frozen C30 prefixes** over 30 tasks, manifest `8e78c7b96161d024` |
| Corpus | `/data/tau2/artifacts_16384`, byte-hashed in `artifact_hashes.json` |
| Policy | sha256 `2c9652af…`, 6,699 chars, identical across all 73 eligible traces |
| Schedule | **32 updates × 16 states**, one student action per state |
| Action cap | **2,048 tokens** (stored max action is 592; a cap hit invalidates the sample) |
| Endpoint | **16,384-token context**, required and verified, not assumed |
| Sampling | task-first frozen order, **not** the paper's `kappa^t` |
| Branch state | fresh optimizer, scheduler and RNG from CK35; no momentum from warm SFT |

### Why 16,384

The longest C30 prefix is **12,880 tokens**. With the 2,048 action cap that
needs 14,928. A 12,288-token server does not fit it and would silently drop
tokens from the *front* of the prompt — sampling a state the run cannot
describe, with every downstream assertion still passing. `artifacts_12288`
exists on the box, which makes this a live hazard rather than a hypothetical.
The controller reads the server's reported context and refuses if it is short.

### Why not `kappa^t`

The ReOPD paper samples position `t` with probability proportional to
`kappa^t`, default `0.6`. That makes position 1 about 99× as likely as position
10, and puts ~99.4% of the mass in the first ten positions. V2 §8 instead
specifies task-first / position-second sampling so every task has support and a
few long traces cannot dominate a 32-update budget.

This is a deliberate departure and must be reported as one. The `kappa^t`
schedule is a later preregistered ablation; calling a coverage-balanced sampler
paper-identical would misdescribe the run.

## 3. Budget matching

V2 §8 preregisters the match over updates, prefix exposures, sampling order,
effective batch size and LoRA capacity — **not** token counts, which cannot be
matched because sampled and recorded actions differ in length.

```text
A_sft_new   32 updates x 16 states = 512 recorded-expert exposures
A_reopd     32 updates x 16 states = 512 sampled-student exposures
```

512 exposures over 289 prefixes is ~1.77 passes. The schedule wraps from where
the previous pass ended rather than restarting, so no prefix is exposed more
than one time above any other.

Both arms read the **same frozen schedule file**. Built once, read twice: a
branch that regenerates its own order breaks the match as surely as one using a
different row set, and both would still log "32 updates over C30".

> **Open item.** Continued SFT currently runs `--epochs 1.0` and lets TRL batch
> and shuffle — roughly 37 steps at effective batch 8. That matches on none of
> updates, exposures or order. It has not run yet, so it must be retargeted to
> this schedule before it does.

Supervised tokens, generated tokens, GPU time and teacher cost are **reported,
not forced**.

## 4. What one update does

```text
16 frozen C30 prefixes (from the schedule)
  -> student samples ONE action each, from the frozen prompt token ids
  -> DeepSeek scores those exact bytes under its own render of the same state
  -> cross-tokenizer chunk-OPD loss, one optimizer step
  -> save adapter + optimizer + scheduler + RNG + update index
  -> reload and prove the adapter changed the logits
  -> serve the new checkpoint for the next update
```

The history is an offline teacher replay; only the action is on-policy. This is
**not** a live Tau2 rollout and must not be described as one — the environment
is never queried after a student action, and the next state is not a
continuation of it.

## 5. The two contexts must match

The student is sampled from `prompt_token_ids` taken verbatim from the frozen
corpus. The teacher re-renders the same state under its own template. Those two
renderings have to describe the same context, and the corpus does not store it
in one place:

```text
rows.tokenized.jsonl   input_ids  = [system + tools] + prompt + [target]
rows.semantic.jsonl    prompt     = the history ONLY
```

The system policy and the retail tool schemas are absent from the semantic row
and must be reconstructed. Two bugs here were found and fixed during the build,
both of which produce a finite loss, a successful alignment and clean logs:

1. `canonical_messages` omitted the policy entirely.
2. Tools were attached to the prefix object, but `render_teacher_prefix` takes
   only `(messages, thinking_mode)` — `encoding_dsv4` reads `msg["tools"]` off
   the system turn, so they never reached DeepSeek.

**Neither is caught by any hash check.** The gate that catches both is render
parity: re-render `[system + tools] + prompt` under the pinned Qwen tokenizer
and require byte equality with the frozen `prompt_token_ids`.

```text
PASSED 2026-08-26: 289/289 prefixes re-render exactly
```

That result proves jointly the policy string, the tool schemas, the template
settings, the tokenizer and the label-derived action boundary.

## 6. Prerequisites, and their status

| # | Item | Status |
|---|---|---|
| 1 | Trace provenance from `eligibility_report.json → per_task → trace_hash` | done |
| 2 | Policy recovered, hashed, all-289 render parity | **passed** |
| 3 | Shared 32×16 schedule frozen | done |
| 4 | Continued SFT retargeted to that schedule | **open** |
| 5 | Tau2 teacher-render / action-span integration test | done |
| 6 | Controller bound to a verified 16,384-token endpoint | **open** |
| 7 | CK35 sampling-diversity canary (V2 §7.1a) | **not run** |

### 7 is a stop condition, not paperwork

`A_warm` is a launchpad. If warm SFT collapsed the action distribution onto one
memorised continuation per state, the student samples the same action every
time, the teacher scores the same thing every time, and the replay gradient is
flat. ReOPD would then fail for a reason that has nothing to do with the
objective under test, after paying for 512 teacher calls.

This is a live risk: CK35 trained on 273 rows with a rank-16 adapter and the
loss went flat after roughly ten of 105 steps. `scripts/tau2_swr_canary.py`
measures it — byte-distinct and canonical-distinct action rates at C30
prefixes, no teacher calls. Note it is checkpoint-only; V2 §7.1a's gate is
`A0`-relative, so a pass here does not fully discharge it, but a failure is
decisive either way.

## 7. Software

Benchmark-specific, in `vektori_trace/tau2/`:

| module | role |
|---|---|
| `c30_loader` | joins manifest + tokenized + semantic; policy/tool reconstruction; render parity |
| `reopd_schedule` | the frozen 32×16 stream both arms read |
| `reopd_sample` | byte-exact sampling from frozen prompt ids |
| `reopd_state` | durable `PLANNED → SAMPLED → SCORED → TRAINED` |
| `reopd_checkpoint` | adapter + optimizer + scheduler + RNG + reload proof |

Shared cross-tokenizer machinery, reused unchanged: `chunk_opd` (the loss, a
port of the pinned reference at `927a8264`), `align` (byte alignment),
`replay_score` (teacher scoring), `replay_sample.token_bytes_from_ids`,
`replay_train` (optimizer step), `dataset.tokenize_messages` (the renderer the
corpus and CK35 were built with).

These implement mathematical and transport contracts, not another benchmark's
behaviour. Reimplementing them would mean re-deriving the paper port and losing
the diff against upstream. Any package reorganisation waits until after the
experiment.

Not yet built: `scripts/tau2_reopd_train.py`, the driver.

## 8. Run order

```text
1. CK35 diversity canary                    one endpoint, no teacher spend
2. verify a 16,384-token endpoint            probe the 12,880-token prefix
3. two-prefix live canary                    small paid spend, needs approval
     2 prefixes -> 1 action each -> score -> one update -> checkpoint
     -> reload -> confirm logits changed
4. 32-update A_reopd run                     separate approval
5. A_sft_new on the identical schedule
6. evaluate both on S16; F38 stays sealed
```

Each of 1, 3 and 4 is a separate authorization. An approved plan containing a
GPU step is not approval to execute it.

## 9. Circuit breakers

New paid dispatch stops, artifacts are preserved, on any of: an authentication
failure; a score or alignment corruption; a ratio/clip anomaly; OOM; a missing
or mismatched policy version; a cap hit; a prompt-id mismatch against the frozen
manifest; or an evaluation regression.

A crash must resume from the durable markers and **reuse every score already
paid for**, never re-dispatch them.

## 10. What would falsify the arm

The method claim is not supported if gains require selection or test trace
access, disappear under policy-compliant scoring, occur only on already-seen
tasks, are explained by truncated or invalid actions, or if ReOPD cannot beat
the shared SFT warm start and is clearly worse than continued SFT at the matched
budget.

A flat gradient traceable to a collapsed `A_warm` is **not** a result about
ReOPD; it is a stop condition on the checkpoint (see §6).
