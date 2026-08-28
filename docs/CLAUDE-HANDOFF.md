# Claude handoff — Tau2 live OPD

Branch: `feat/tau2-live-opd` · HEAD `98abea3` · 2026-08-29
Full suite: **1,957 passed, 2 skipped, 0 failed**

---

## 1. Where this stands in one paragraph

The live OPD loop runs end to end — rollout → score → optimizer step →
checkpoint → serving reload → next rollout — and a real two-update run on
2026-08-28 executed every stage with a verified weight change. That run then
exposed a **scoring defect**: DeepSeek was asked for the likelihood of *raw Qwen
bytes*, including control markup it never emits. The defect is diagnosed,
fixed, and wired into production, but **the repaired scorer has never made a
real teacher call**. Post-fix advantages are unknown.

---

## 2. The defect, and the evidence

`score_replay_batch` scored the complete raw action. The pinned reference
forbids this (`opd_reference/reward_manager_opd.py:337` — *"response content
only, no chat template special tokens on either side"*), and upstream
*translates* chat-template markers between model families rather than passing
one model's markup to the other.

Measured over the 31 archived actions of update 0:

| class | n | mean advantage | worst |
| --- | ---: | ---: | ---: |
| **markup** (`<tool_call>`, `<|im_end|>`) | 80 | **−24.47** | **−55.29** |
| reasoning | 11,707 | −0.60 | −24.50 |
| content | 1,760 | −0.70 | −27.06 |
| tool_json | 631 | −0.28 | −13.25 |

The twelve worst advantages in the entire batch are **all `<tool_call>`**
(−55.3 … −46.4). DeepSeek serves tool calls as a DSML block
(`<｜DSML｜invoke name=...>`) and has never emitted Qwen's `<tool_call>`
wrapper, so those numbers are a verdict on *notation*, not on the decision.

**Upstream repairs `<|im_end|>` (template mapping) but offers nothing for the
Qwen-JSON vs DeepSeek-DSML tool mismatch** — verified: no occurrence of
`tool_call`, `DSML` or `invoke name` anywhere in the vendored revision. That
half is ours.

### What the defect is *not*

- **Not** "no positive signal." Token-level signs were 3,388 positive /
  10,713 negative / 77 zero. All 31 *action means* were negative, but a mean
  over ~457 tokens says nothing about the signs beneath it.
- **Not** a magnitude problem. `grad_norm` 0.731, `clip_fraction` 0.005,
  `max_param_delta` 1e-5 — a healthy, tiny step. Lowering the LR would have
  wasted a run.
- Negative action means are *expected*: for `x ~ π_S`,
  `E[log π_T − log π_S] = −KL(π_S‖π_T) ≤ 0`.

### The observed consequence

After one update, three of four episodes opened `<think>`, reasoned coherently,
then emitted `<tool_call>` **without closing `</think>`**. The reasoning gate
refused the batch. `</think>` itself was only mildly penalised (mean −0.221,
27/31 negative) — far less than `<tool_call>` at −55 — so the likeliest
mechanism is that a handful of extreme unclamped advantages dominated the
structural gradient, with `</think>` breaking as collateral because SFT
*masked* it from the loss and never anchored it.

---

## 3. What was built

| file | purpose |
| --- | --- |
| `tau2/live_projection.py` | which byte ranges may carry teacher credit |
| `tau2/live_score.py` | score the DeepSeek-native equivalent, map back per payload |
| `tau2/live_batch.py` | `TurnAdvantages` from pre-aligned credit, no raw re-align |
| `tau2/live_token_classes.py` | advantage attribution by token class |
| `scripts/tau2_fireworks_parity_probe.py` | integer-ID scoring compatibility |

All glue. **No new tokenizer, parser or aligner** — reuses `split_generation`,
`encode_messages`/`render_teacher_prefix`, and `align_by_bytes`.

The projection supervises **reasoning + visible content** (byte-identical in
both renderings) and excludes tool-call serialization (semantics correspond,
bytes do not). A token straddling a payload boundary is dropped whole — half a
token cannot carry half an advantage.

### Production wiring (this was the gap; it is closed)

`train_live_update` now calls `run_projected_score_stage` and
`run_projected_train_stage`. **Zero live callers of `score_replay_batch`**,
verified by grep *and* by a test that monkeypatches it to raise and drives the
real driver. `diagnose` was the last leak and is fixed.

---

## 4. Verified results

| gate | result |
| --- | --- |
| Token reconciliation | 14,206 = 13,417 supervised + 761 masked + 28 sentinel |
| Projection, offline | 31/31 actions, **94.45%** retained, 0 payload skips |
| Scoring dry-run (stub teacher) | 31/31, $0, accounting complete |
| Fireworks probe (**paid**, passed) | echoed ids match, 19/19 and 59/59 logprobs, finite |
| Masked-advantage preview | mean −0.733 → −0.580; p1 −9.18 → −6.82; −55 gone |

The preview is a **historical masked analysis only** — those numbers still come
from chat-mode scoring. Re-scored values may move either direction.

### Fixed along the way

- **Parent adapter hash** derived from weights, never an optional flag
  (update 0 archived `""`).
- **Task share** derived from the batch; the replay default of 0.5 rejected a
  legitimate single-task live batch.
- **Payload ambiguity**: content is searched only after the reasoning span, and
  a repeated content string is refused rather than resolved by position.

---

## 5. Remaining gates before the ~$0.04 re-score

1. ~~Full suite~~ — **done, 1,957 passed**.
2. ~~Payload ambiguity~~ — **done, `98abea3`**.
3. **31 actions through the real `train_live_update`** with real DeepSeek
   tokenizer, fake pool, fake trainer, `score_replay_batch` patched to raise.
   Require: 31/31 scored, no payload skipped, nonzero supervision each,
   complete accounting, zero markup weight, rows validate, resume makes zero
   teacher calls.
4. **Fresh canary run directory** from the untouched `A_sft_new` parent. Copy
   **only** the immutable `SAMPLED` actions and rendered histories — *not*
   `SCORED`/`TRAINED` markers, scores, checkpoint or optimizer state, or the
   old `TRAINED` marker short-circuits the repair.
5. **Task-share policy for the pilot**: validate against the *preregistered
   planned roster*, not the tasks that happened to survive — otherwise a
   rollout that lost a task entirely passes its own concentration check. A
   single-task diagnostic may explicitly allow 1.0.

**Do not mutate or resume the old failed run** (`two_update_proof_20260828_182302`)
— preserve it as evidence.

Then: re-score 31 → inspect real post-fix advantages → decide on a clamp →
one step from untouched parent → reload + format probes → one fresh episode →
second update → 5×8 pilot.

---

## 6. Operational notes

- Box: EC2 `i-0a348ff3d7be9769a`, **SSM only**. Git there fails under root
  ("dubious ownership") — run as `sudo -u ubuntu -H`. Heredocs over SSM are
  unreliable; base64 instead.
- `modal app stop` **silently aborts without `-y`**.
- Endpoint: `serve_student.py --adapter a-sft-new=/adapters/tau2/runs/a_sft_new_ck35_r2/checkpoint-32 --gpu L40S --max-model-len 24576 --reasoning-parser qwen3`.
  Adapter is **rank 16** (vLLM default; no `--max-lora-rank` needed).
- Tau2 is **not** on the Modal volume — the image pins `tau2 @ f8de30c`.
  Domain data *is* staged at `tau2/tau2_data` with `TAU2_DATA_DIR` set;
  `user_simulator/` is required alongside `domains/retail/`.
- User simulator runs on **Fireworks**, not OpenAI — only `fireworks-api-key`
  exists as a Modal secret. Key lives in `/data/vektori-trace/.env`.
- Free inspectors: `--preflight-only`, `--projection-only`, `--reconcile-only`,
  `--scoring-dryrun-only`, `--predict-only`, `--classify-only`, `--show`,
  `--telemetry-only`.
- Spend gates: `--canary`, `--two-update-proof`, `--yes`. Nothing allocates a
  GPU without one.

### Measured costs

One episode ≈ 9 turns, ~70 s. Update 0 (4 episodes, 31 turns): rollout 288 s,
scoring 139 s / $0.043, training 99 s, **peak 12.9 GiB** on an L40S. Two-update
proof total ≈ **$1.15**. The 12.9 GiB answers the deep-dive's open question:
the live batch is **not** memory-bound at this scale.

---

## 7. Open questions

- **Does masking suffice for `</think>`?** Masking removes direct pressure but
  not indirect parameter movement. A format anchor may be needed; unmeasured.
- **Is a clamp warranted?** `advantage_clamp` was `None`. Even after masking,
  `min` ≈ −25 survives. Decide *after* seeing real post-fix advantages.
- **Did this defect contribute to the ReOPD failure?** Plausible — ReOPD used
  the same raw-action scorer across 32 updates — but unproven, and that run
  also had scale and control problems. Its archive has not been run through
  the classifier.
- **Do tool-call semantics deserve supervision?** v1 excludes them (631
  tokens). Their semantics correspond; their bytes do not.
