# Pilot handoff — live OPD, halted before restart

Branch `feat/tau2-live-opd` · HEAD `fcad207` · written 2026-08-29 ~05:15 IST

**Authority:** `docs/TAU2-OPD-DEEP-DIVE.md` § "Pilot incident" is the record of
what the pilot established and what must be fixed. This file is the operational
companion: current state, teardown, and what was verified first-hand versus
what is assumed. Where the two disagree, the deep dive wins.

---

## 0. TEARDOWN — check before ending any session

```bash
cd /data/vektori-trace && .venv/bin/modal app list | grep ephemeral
# any row with Tasks>=1 is billing. Expect ZERO.
.venv/bin/modal app stop <app-id> -y          # -y MANDATORY, silent no-op without it
cat /data/tau2/pilot-state/owned_apps.json    # non-empty => a stop FAILED, do it by hand
```

Check from **both** the box and locally — separate Modal clients. Re-check after
~60 s; an app can report `stopped` while its container drains.

**State at time of writing: 0 ephemeral apps, no tmux sessions, nothing
billing.** Verified from both clients.

---

## 1. Where things stand

**The pilot is halted and must not restart** until the gates in the deep dive
pass. Update 0 failed during rollout. **No teacher call and no optimizer step
ever ran**, so no trained artifact is contaminated.

| | |
| --- | --- |
| run id | `pilot_10x8_20260829` (failed, keep as evidence) |
| plan hash | `aa9251ccb6d566fa` — 80 distinct C30 `(task, seed)` pairs |
| parent | `3869b147ab7ce5d2`, untouched `A_sft_new` |
| stage reached | `.PLANNED` only; `.SAMPLED` absent |
| spend tonight | **~$3.60** total (mechanism proof ~$2.75 + failed pilot ~$0.85) |
| host | EC2 `i-0a348ff3d7be9769a`, **SSM only**; state in `/data/tau2/pilot-state` |

Do not delete or edit the failed archive. Restart is a **new run id** carrying
the same 80 pairs and plan hash, per the deep dive.

---

## 2. Update 0, exactly as archived

| episode | status | turns | outcome |
| --- | --- | ---: | --- |
| task 95 / seed 1 | sampled | 10 | reward 1.0 |
| task 71 / seed 1 | sampled | 9 | reward 1.0 |
| task 53 / seed 2 | sampled | 8 | reward 1.0 |
| task 108 / seed 1 | sampled | 8 | reward 1.0 |
| task 44 / seed 1 | failed | 3 | unclosed `<think>` |
| task 68 / seed 0 | failed | 3 | unclosed `<think>` |
| task 76 / seed 0 | failed | 2 | unclosed `<think>` |
| task 109 / seed 0 | discarded | 1 | HTTP 408 |

**Read the denominators the way the deep dive states them**, not the way I first
reported them: 3/7 non-infrastructure *episodes* (42.9%), but only 3 of ~43
generated *turns* (~7%). I conflated those two rates earlier; the turn rate is
the one that describes how often the parser is wrong.

`4/4 completed episodes scored reward 1.0` is a **conditional** result on the
episodes that finished — not a batch success rate.

---

## 3. Verified first-hand (I ran these)

- **Raw generations of all three failures**, read off the volume via
  `inspect_failed_turns`. Each: one `<think>`, zero `</think>`, 288–1,347 tokens
  of coherent reasoning, one or more valid `<tool_call>` blocks, and
  `finish_reason: stop`. So "dropped reasoning" and "truncated" are both false.
- **The prompt does not pre-open `<think>`** — `apply_chat_template(...,
  enable_thinking=True)` ends at `<|im_start|>assistant\n`. The model emits both
  delimiters itself.
- **SFT masked the wrapper.** `dataset.py` defines
  `THINK_WRAPPER_TEXT = "<think>\n\n</think>\n\n"` and `mask_think_wrapper=True`
  keeps those four tokens out of the loss — the closing tag was context, never a
  supervised target.
- **The N:1 advantage defect**, by reading both implementations:
  `chunk_opd.assign_chunk_advantages` sums `L_S` over the chunk;
  `live_batch` divided the chunk's `L_T` across its student tokens and took a
  fresh ratio per token. Worked example, verified numerically — 3 student
  tokens with **unequal** logprobs `[-0.5, -1.0, -1.5]`, teacher agreeing
  exactly (`L_T = L_S = -3.0`): the chunk rule gives `[0, 0, 0]`, the per-token
  rule gives `[-0.5, 0, +0.5]`. Opposing gradients at exact agreement.

  **Correction to an earlier version of this file:** it used three *equal*
  logprobs (`-1.0` each) and claimed the live path produced `A_i = -2.0`. That
  is wrong — with equal logprobs both rules return `[0, 0, 0]`, and a
  regression test built on that example would have passed against the defect.
  The distinguishing case requires unequal student logprobs within one chunk.

  **Repaired 2026-08-29 (`4b82d09`)** — chunks persisted whole through
  scoring, persistence and resume; arithmetic delegated to `chunk_opd`.
- **Endpoint URL fabrication** (fixed, `8bcb047`): the fallback invented a URL
  missing the workspace prefix, class segment and `-dev` suffix, so every
  request 404'd while vLLM logged `UP in 183s`. Never fabricate the URL —
  resolve it.

  **Not established:** that `get_web_url()` works only on the class's Function
  and not on a bound instance method. Both a laptop and a box probe returned a
  correct URL from the bound class method, so that explanation is unproven and
  should not be repeated as the root cause. What is verified is the fabricated
  fallback and its 404s.
- **Teardown works.** The orchestrator stopped its endpoint on failure both
  times; `owned_apps.json` came back empty.

## 3b. NOT verified — do not repeat these as facts

- **Whether the 4 successful episodes closed their think tags.** Never checked.
  If some turns close and some do not, the form is stochastic; if the successes
  never opened `<think>` at all, the story is different again. This is free to
  check from the archive and has not been done.
- **Whether SFT masking *causes* the unclosed form.** Plausible and consistent,
  but it is a hypothesis. The deep dive says so; I stated it more firmly than
  the evidence supports at one point.
- **Whether the HTTP 408 is random.** Needs correlation with request duration
  and client/server timeouts before being called infrastructure noise.
- **Cumulative Modal spend.** Never pulled from modal.com. The $30 ceiling is
  per-run and cannot see the account total.
- **The earlier claim that one OPD update broke `</think>`** (in the older
  handoff) is an unproven causal hypothesis. The untouched parent produces the
  same form.

---

## 4. Restart gates

See `docs/TAU2-OPD-DEEP-DIVE.md` § "Mandatory restart gates" — eight items.
The two that block everything:

1. **N:1 / M:N advantages must match `chunk_opd.py` exactly.** Carry aligned
   chunk membership through projected scoring rather than collapsing to
   per-index. Add the regression test where equal aggregate likelihoods produce
   zero advantage for every token in the chunk. Until this passes, a finite loss
   and a healthy gradient norm are **not** evidence the update direction is
   right.
2. **Parser parity with pinned vLLM.** Port the narrow upstream behavior
   (vLLM `92762ed`): first valid `<tool_call>` opening is the implicit end of an
   unclosed `<think>`; spans stay disjoint; invent no bytes; refuse the
   ambiguous cases. Not a general "reasoning to end of generation" rule — that
   would sweep the tool calls into the reasoning payload.

The root cause of the parser drift is worth keeping in mind: live capture posts
pre-tokenized ids to `/completions`, so vLLM's `qwen3` and `hermes` parsers
never run. `split_generation()` is a client-side reimplementation that fell
behind upstream.

---

## 5. Operational facts worth not re-deriving

- vLLM serves adapters **prefixed**: `a-sft-new` → `Qwen3-4B-a-sft-new`. An
  unknown name silently resolves to the base model. This cost one launch.
- `serve_student.py` writes `pilot_env.sh` only after a real completion
  succeeds. That smoke test is the readiness signal, not the log line.
- `--tool-call-parser hermes` is required or vLLM 400s on any request carrying
  `tools`; `/models`, `/health` and plain completions all look fine until the
  first tool turn.
- **~8 turns/episode**, measured. The proposal's "~13" was an estimate. 10×8 is
  therefore ~640 actions.
- Failed and discarded episodes are **terminal**. A resume reloads them, refuses
  to resample, and fails the same 8/8 check. `--start-at 0` does not force
  resampling.
- The alerting substrings `refus` and `Error` false-positive constantly on
  `update-NNN.log` — it is retail dialogue full of "refund", "return",
  "refusal". Match orchestrator strings (`STOPPING`, `failed (rc=`, `!!`,
  `Traceback`) in `pilot_run.log` only.
- `scripts/pilot_watch.sh` hardcodes a laptop scratchpad path and is **useless
  on the box**. Fix or delete it.

---

## 6. Commits this session

| commit | what |
| --- | --- |
| `6047ccd` | gate 4 — canary from SAMPLED-only evidence |
| `7409099` | `--rescore-only` — paid scoring, no optimizer step |
| `cf54d32` | `--one-step-only` — one step from cached scores |
| `b5b777c` | `--tool-call-parser`, or every tool turn is a 400 |
| `115c151` | `--parent-override`, or update 2 retrains from the SFT adapter |
| `324da9b` | live share limits are telemetry, not refusal |
| `26d2ed3` | `one_step` must resume Adam, not just the weights |
| `221daf9` | per-update identity + `rollout_only` |
| `47bb680` | pilot orchestrator, resume and scoped teardown |
| `c7a8fda` | freeze 80 distinct C30 plans before update 0 |
| `ea01216` | rehearsal, ledger fixes, frozen preregistration |
| `1a568d6` | `stage_manifest` |
| `8bcb047` | never fabricate the endpoint URL |
| `fcad207` | `inspect_failed_turns` — read raw bytes before diagnosing |

Full suite at `ea01216`: **1,982 passed, 2 skipped, 0 failed**.

---

## 7. The two-update mechanism proof is now suspect

`b6c160fcf9792e92` and `01a95c2368661cd7` were trained through the same
projected path that carries the N:1 defect, so **their loss and gradient
numbers were computed with inflated ratios**. Treat those runs as evidence that
the *plumbing* executes end to end, not as evidence about the update direction.

They also trained on tasks 57/73/75/93, which are **S16** — fine for a mechanism
test, disqualifying for the pilot lineage. The pilot restarts from the untouched
parent and those two checkpoints are never in it.
