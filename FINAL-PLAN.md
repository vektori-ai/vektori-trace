# Cross-tokenizer OPD — full build plan

**This file is the single source of truth for the cross-tokenizer OPD work.**
It is self-contained: a reader with **zero prior context** can build the whole
thing from this document alone, without opening another doc. It deliberately
duplicates rather than references — where it must point outward it points at
**code**, which is verifiable, never at another doc's prose, so a disagreement
elsewhere in the repo can never stall someone mid-build.

Sections 1–10 are the reference material (what the system is, what was measured,
the maths, the failure modes). "Deliverable 2 — the code" onward is the build
itself. Nothing here has been implemented yet; this is the plan.

---

## Context — why this work exists

### What `vektori-trace` is

The product claim: *given a capability a small model lacks, say **which
intervention will fix it** — before burning weeks discovering it by exhaustion.*
Teams try RL, find the gradient is zero; try distillation, find the teacher never
demonstrated the behaviour. `PLAN.md` argues the decision is decidable by
measurement:

- **RL's gradient** is `Σ A_t ∇log π_s(y_t)`. When every rollout in a group
  fails, `A_t = 0` and the gradient is **identically zero**. RL sharpens within
  the model's existing support; it cannot create support.
- **OPD's gradient** is `Σ (log π_s − log π_t) ∇log π_s`. Nonzero wherever
  student and teacher disagree, **regardless of outcome**. It moves mass into
  regions the student assigns ≈0 probability.
- **OPD is bounded by the *teacher's* support, not the student's.**

`pass@k` measures support directly, so `pass@k` decides the intervention. The
experiment that tests this is arm **B1 (routed: RL on in-support, OPD on
out-of-support)** against **B2 (anti-routed: identical tasks, compute and method
mix, assignments inverted)** — B2 is the control that makes the claim real.

**This plan builds the OPD half.** If OPD does not run, B1/B4 do not exist and
the central experiment cannot be scored.

### The pipeline OPD sits at the end of

```
mine PRs → audit → replay → diagnose deficit → route (RL | OPD | quarantine)
                                                        ↓
                                                  THIS PLAN
```

Mining takes merged GitHub PRs with a linked issue and a test diff, and turns
each into a Harbor task: agent container (repo at base commit, git history
scrubbed, network allowlist) + a separate verifier container (applies the patch,
runs pytest, requires F2P tests to flip fail→pass and P2P tests not to regress).

### Why cross-tokenizer, and why it breaks everything

Every OPD primitive in the repo assumes **teacher and student score the same
token ids**. `tokenizer_check.check_tokenizers` hard-fails otherwise, before any
GPU is allocated — that gate is deliberate and stays.

The teacher is now **DeepSeek-V4-Flash-0731**, a different tokenizer family from the
Qwen3 student. Shared-id scoring is structurally impossible: sending the
student's ids to the teacher asks it to score *different strings* than the
student sampled. The danger is that this does not crash — mis-scored tokens still
produce a finite, plausible-looking loss. Almost every failure mode in §10 has
that shape, which is why the design is assert-heavy.

### Outcome

A cross-tokenizer path that aligns the two tokenizations **by bytes**, supervises
~100% of sampled tokens with the **same reverse-KL objective the repo already
declares**, and hard-fails on every misalignment mode rather than training on
garbage.

### Work already done this session — do not redo

P1 and P2, the two decisive measurements, are **run and passed** (§4). Artifacts:

```
/tmp/claude-1000/-home-alex-hunterz-Desktop-projects-vektori-trace/\
d3feceee-2544-460c-98d1-41b78c2a134d/scratchpad/
    p2_align_report.py                      # the measurement, ~180 lines, reruns in seconds
    Qwen_Qwen3-14B.json                     # student tokenizer
    deepseek-ai_DeepSeek-V4-Flash.json      # teacher tokenizer (identical to -0731's)
```

Superseded planning file: `~/.claude/plans/resilient-prancing-ritchie.md`. Its
content is folded in here. **Do not treat it as a second source of truth.**

---

## Deliverable 0 — documentation hygiene (do first; it governs everything else)

Several repo docs predate the current architecture and contradict it. **Do not
delete them.** Mark them, and make the new doc independent of them.

**Governing rule for `FINAL-PLAN.md` (this file): it is self-contained. It
duplicates rather than references.** Where it must point outward it points at
**code** — verifiable, and the actual ground truth — never at another doc's
prose. A disagreement between two docs must never be able to stall someone
mid-build.

**Banner to add at the top of each stale doc:**

```
> **STALE — do not build from this.** Superseded by `FINAL-PLAN.md`.
> Kept for history. No longer true: <one line>.
```

| file | the line |
|---|---|
| `docs/PILOT.md` | Modal spend-order; 30B→8B same-family Qwen teacher; H100+A100. Teacher is hosted, there is no Modal, and the pair is cross-tokenizer. |
| `docs/AWS.md` | Assumes two GPU instances, one *serving the teacher* for `prompt_logprobs`. That instance no longer exists. |
| `docs/AWS_PORT_PLAN.md` *(untracked)* | "Port Modal → AWS EC2." Its own successor `opd-aws-fireworks.md` opens "EC2 is out." |
| `docs/opd-aws-fireworks.md` *(untracked)* | Bedrock teacher + Fireworks-hosted student. Teacher is Fireworks; student is local — §9 shows the hosted student *cannot* carry this loss. |
| `docs/TECHNICAL_NOTES.md` *(untracked)* | Near-duplicate of `PAPER_BRAINDUMP.md`: same title, same six headings; `PAPER_BRAINDUMP.md` has a seventh ("Cost basis"). **Confirm with the author before bannering.** |

**Valid, safe to reference:** `docs/OPD.md` (method reference, corrected in
`bcf37c0`), `docs/HOSTED_TEACHERS.md`, `PLAN.md` / `V0_PLAN.md` (mining +
diagnosis halves), `docs/network-policy.md`.

`docs/mined-tasks-inventory.md` *(untracked)* was on that list and no longer is:
it inventories the `vektori-out-*` anyio/click/tenacity corpus, which the
`commit_runtime` prefect tasks superseded. See the open item below.

**Fix dangling pointers** so nothing routes a reader into a bannered doc:
`cli.py` help strings citing `PLAN.md Step H` / `docs/PILOT.md`, and `docs/AWS.md`'s
opening line.

---

# Deliverable 1 — the reference material (§§1–10 of this file)

## 1. Glossary

| term | meaning |
|---|---|
| **OPD** | On-policy distillation. Student acts, teacher scores *the student's own tokens*, gradient from the disagreement. |
| **ReOPD** | "Replayed" OPD: the prefix is a frozen, pre-collected teacher trajectory, so no containers run in the training loop. `reopd.py`. |
| **π_s, π_t** | Student and teacher next-token distributions. |
| **echo scoring** | Asking an inference API for logprobs of tokens **you supplied** rather than tokens it sampled. Fireworks `/completions` with `echo=True`. The precondition for OPD against a hosted teacher. |
| **prompt_logprobs** | vLLM's name for the same capability. |
| **span** | A maximal run of student tokens and teacher tokens covering **identical bytes**. The unit of supervision here. |
| **granularity** | `spans / student tokens`. 1.0 = perfect 1↔1; near 0 = one giant span, i.e. sequence-level not token-level. |
| **ATIF** | The trace JSON format (`{runId, status, turns:[…]}`). Parsed via Harbor's own Pydantic models. |
| **Harbor** | Task/execution framework. Produces the job dirs that become teacher trajectories. |
| **F2P / P2P** | Fail-to-pass / pass-to-pass tests. The verifier's grading contract. |
| **arms A0–A4, B1–B4** | The evaluation matrix in `PLAN.md`. B1 routed vs B2 anti-routed is the claim. |
| **LoRA** | Low-rank adapters; base weights frozen. |
| **top-K / `top_logprobs`** | How many alternatives per position the teacher returns. **Capped at 5** on Fireworks serverless. |

## 2. Configuration

- **Teacher: DeepSeek-V4-Flash-0731 on Fireworks serverless.** No dedicated
  deployment → `top_logprobs` hard-capped at 5 (`TOP_LOGPROBS_CAP = 5`,
  `teacher_fireworks.py`). Treat K=5 as fixed. Fireworks serves **FP8**, so
  teacher logprobs carry quantisation noise inside a term the student
  differentiates through — record it in provenance.
- **Student: Qwen3-8B + LoRA, single GPU, bf16.** Already
  `tokenizer_check.PILOT_STUDENT`, so no constant changes. Qwen3-14B is a later
  swap and is **free on everything here** — Qwen3-8B and Qwen3-14B ship
  byte-identical tokenizers (vocab, merges, added tokens, pre-tokenizer,
  normalizer all compare equal). Only the GPU shape moves.
- **No FSDP, no multi-GPU.** An 8B LoRA fits one card. FSDP + LoRA + a custom
  loss is a known source of silent breakage; dropping it removes a bug class from
  a design whose failures are already mostly silent.
- **Objective: on-policy reverse KL. Not parameterized.** No `beta` field, no
  forward-KL code path. §6 records why forward KL was rejected so it does not get
  relitigated.
- **Cost shape:** the teacher is queried **every step**, so *teacher latency*,
  not student FLOPs, sets step time.
- **Env:** `FIREWORKS_API_KEY` required. Also used in the repo:
  `OPENAI_API_KEY` (diagnosis), `VEKTORI_MODEL`.
- **Install:** `uv sync --extra train` (torch, transformers, peft, accelerate,
  datasets). `tokenizers` comes in transitively. Offline stages need no GPU and
  no network.

## 3. The existing loop this plugs into

All of this is already built and working.

```
harbor job dirs / ATIF .json traces
  → cli._load_teacher_trajectories             (cli.py:915)
  → reopd.iter_reopd_examples → ReOPDStepExample
       {task, step_index, prefix_turns, teacher_action_turn, later_teacher_turns}
  → distill.run_opd_training                   (distill.py:243)
      per optimizer step, per example (examples_per_step=4 default):
        encode_prefix(...)          chat template, add_generation_prompt=True
        _sample_action(...)         student samples, do_sample=True, temperature=1.0
        pool.score_ids(prefix_ids, action_ids)    teacher scores THOSE ids
        _student_action_logits(...) forward with grad → logits [1, A, V]
        opd.reverse_kl_surrogate(...) → (loss / examples_per_step).backward()
      then: clip_grad_norm_, optimizer.step(), cosine schedule
  → adapter written to out/adapter, opd_log.jsonl per step, opd.json report
```

**Invariants the new path must preserve:**

- `add_generation_prompt=True` positions the student to *open* an assistant turn.
  Without it every sampled token is scored in the wrong role.
- `temperature=1.0` keeps the sample on-policy. Any other value makes the
  surrogate weight logprobs of a policy that is not π_s. Rejected at config
  construction (`OPDTrainConfig.__post_init__`).
- `_student_action_logits` cross-checks its slice against `reopd.reopd_loss_mask`
  instead of trusting the arithmetic. **Model the cross-tokenizer branch's
  indexing assertion on exactly this.**
- The teacher never re-encodes text; it scores supplied ids.
- Empty samples are counted (`skipped_empty_samples`), never treated as
  agreement.
- NaN loss aborts *before* `optimizer.step()`.

**Test posture:** `tests/test_distill.py` runs the **real loop** on a tiny
from-scratch GPT-2 with `teacher.InMemoryIdScoringPool` — no GPU, no network,
deterministic per-token values so a test can prove token *order* was preserved,
not just count. **Match this for every new module.**

**File inventory (what to reuse, not rewrite):**

| file | role |
|---|---|
| `opd.py` | `reverse_kl_surrogate` (the objective), `topk_reverse_kl`, `token_logprobs`, `grpo_advantage`, configs |
| `distill.py` | the OPD training loop, `encode_prefix`, `_sample_action`, `align_topk_rows` |
| `reopd.py` | `ReOPDStepExample`, `build_reopd_example`, `iter_reopd_examples`, `reopd_loss_mask`, `TeacherContinuationExample` |
| `teacher.py` | `VllmTeacherPool`, `_post_json`, `TeacherScoringError`, `InMemoryIdScoringPool` |
| `teacher_fireworks.py` | hosted teacher; echo request shape, `_align_scored_entries`, `_entry_logprob`, `TOP_LOGPROBS_CAP` |
| `teacher_bedrock.py` | Bedrock CMI teacher (unused here) |
| `student_fireworks.py` | hosted student — **cannot carry this loss**, see §9 |
| `tokenizer_check.py` | `check_tokenizers` (same-vocab gate), `fingerprint_tokenizer`, `PILOT_*`/`SCALE_*` |
| `dataset.py` | loss masking: `role=="assistant" AND subagent_depth==0` |
| `cli.py` | `_load_teacher_trajectories` (:915), `_teacher_pool_for` (:953), `cmd_distill` (:991), parser (:2140) |

**Merged history that matters:** #17 CI (ruff + pytest on every PR) · #18 the OPD
loop, teacher scoring, AWS path · #19 hosted teachers (Fireworks, Bedrock) and a
Fireworks student · #20 doc reframe. #19 was merged without review and **has
never hit a live endpoint** — P0 is its first real test.

## 4. Measured facts — P1 and P2 are done and passed

Offline, 2026-07-31, against the real `tokenizer.json` of Qwen3 and
`deepseek-ai/DeepSeek-V4-Flash`. Zero API calls, zero GPU. Rerun:
`python p2_align_report.py`.

**These measurements were taken against the unsuffixed `deepseek-ai/DeepSeek-V4-Flash`
repo and carry over to `-0731` unchanged**: the two ship a byte-identical
`tokenizer.json` (verified 2026-08-01 — vocab 128 000, merges 127 741, 1 283
added tokens, same bytes). Only the vendored `encoding_dsv4.py` differs, and
only in `reasoning_effort`, which is never passed. Note `config.json` reports
`vocab_size: 129280`: that is the padded embedding matrix, not the tokenizer.
Ids in the pad range never appear in a `tokenizer.json`-derived byte table, so
they cannot be mapped — they fall into estimator (A)'s `other` bucket, which
stays exact.

**Both tokenizers are ByteLevel BPE** — this is what makes byte alignment valid,
and it is asserted, not assumed:

| | Qwen3-8B / 14B (identical) | DeepSeek-V4-Flash-0731 |
|---|---|---|
| model | BPE, 151 643 vocab | BPE, 127 997 vocab |
| decoder | ByteLevel | ByteLevel |
| normalizer | **NFC** | none |
| digit pre-split | `\p{N}` (one digit) | **`\p{N}{1,3}`** (up to three) |
| `add_prefix_space` | false | **true** |
| added tokens | 26 | 1 283 |

All 151 643 / 127 997 pieces round-trip through the ByteLevel table with **zero
failures** → `id → bytes` is total on both sides.

**Exact-byte single-token overlap: 84 030 tokens** — 55.4% of Qwen's vocab,
65.6% of DeepSeek's. Buildable offline because both models are open-weights.

**Alignment on real content** (granularity = spans / student tokens):

| content | student toks | granularity | 1↔1 spans | oversize | desync | byte mismatch |
|---|---|---|---|---|---|---|
| python source | 58 862 | **0.980** | 91.3% | 0 | 0 | 0 |
| trace JSON | 1 390 | **0.986** | 90.6% | 0 | 0 | 0 |
| markdown prose | 20 368 | **0.945** | 92.8% | 0 | 0 | 0 |
| tool-call JSON args | 2 150 | **0.977** | 88.1% | 0 | 0 | 0 |
| numeric-heavy | 1 200 | **0.667** | 75.0% | 0 | 0 | 0 |

Pre-registered floor was 0.5. **~90% of spans are 1↔1**, so the analytic
estimator is the main path, not the lucky case.

**V4-Flash vs V3:** BPE `vocab` and `merges` byte-identical; only `added_tokens`
grew 818 → 1283 (465 new: `<think>`, `</think>`, `<dsml:`, `</dsml:`,
`<|place_holder_mm_span_NNNN|>`). The new specials matter only for the masking
list (#4).

**Three findings, now requirements:**

1. **Numbers are the weak spot and they combine correctly anyway.**
   ```
   'x = 1234 + 56789'
     student: x | Ġ= | Ġ | 1 | 2 | 3 | 4 | Ġ+ | Ġ | 5 | 6 | 7 | 8 | 9
     teacher: x | Ġ= | Ġ | 123    | 4 | Ġ+ | Ġ | 567   | 89
   ```
   Byte offsets agree at every boundary, so `1234` closes as a 3↔1 span then a
   1↔1 span. Nothing dropped, nothing desynced. Numeric spans are ≤3 student
   tokens, far under the 8-token threshold. **Do not normalize numbers out of the
   data** — that changes what the student trains on to flatter a metric. Log
   granularity *by content type* instead.
2. **Qwen normalizes NFC, DeepSeek does not.** Non-NFC input shifts bytes between
   the two. #2 is a verified hazard, not a precaution.
3. **DeepSeek's decoder sets `add_prefix_space=true`, `trim_offsets=true`** —
   `decode()` silently eats a leading space. This is why §8 forbids `decode()`
   and builds byte tables from vocab *pieces*. The zero-mismatch numbers above
   used pieces; a `decode()`-based table would not have got them.

## 5. Why a top-5 teacher is not a problem

The blocking belief was that top-5 logprobs cannot support real distillation
math. It dissolves with one observation:

**A top-K response is an *exact coarse-grained* distribution, not an approximate
full one.** Partition the teacher's vocabulary into `K+1` events — each reported
token, plus "anything else":

```
P_t(event i) = exp(lp_i)              exact, reported
P_t(other)   = 1 − Σ_i exp(lp_i)      exact, arithmetic
```

Nothing is assumed about the ~123K untracked ids. They are **not assumed zero** —
they are lumped into one event whose *total* mass is known exactly. Apply the
same partition to the student (full logits locally → exact) and both sides are
genuine distributions summing to 1.

- **KL over a coarse-graining is a lower bound on the true KL** (data-processing
  inequality), monotone in K. Training on it never asserts anything false.
- Both alternatives are wrong. TRL's `_compute_jsd_loss_for_matched_tokens` feeds
  an *unnormalized* sub-vector to JSD (sums to <1, not a distribution).
  Renormalizing over the mapped set asserts unmapped mass is zero, so the student
  pays nothing for leaving the shared support. The `other` bucket is neither — it
  is the actual probability of the actual event.

**K=5 costs tightness, not validity.**

**Precedent:** Phan, Khisti, Ullrich (ICLR 2026) do cross-tokenizer distillation
from stored **top-K=20** teacher probabilities and beat ULD on
GSM8K/HumanEval/MBPP. Top-K is a working regime in the literature.

## 6. The objective — reverse KL, two estimators

```
reverse KL = Σ_e π_s(e) · ( log π_s(e) − log π_t(e) )
             └─ weighted by the STUDENT's probability
```

Mode-seeking: the student commits to one teacher mode rather than hedging across
all of them. Three reasons it is right here:

1. **Coverage.** Reverse KL is the direction *both* estimators can express, so
   **100% of spans are supervised.** Forward KL is computable only by estimator
   (A), which would silently drop the ~10% of spans (A) cannot reach — and §4
   shows those are the numeric and rare-identifier ones. That is #10,
   coverage-conditional bias, for no gain.
2. **GKD measured the divergence and found no general winner** — optimal
   interpolation varies by task; XSum best near reverse; reverse better for
   larger students; "mode-seeking divergences typically outperform their
   mean-seeking counterparts."
3. **The repo is built for it** — `reverse_kl_surrogate`, `topk_reverse_kl`,
   `student_fireworks.opd_loss_fn`, and `docs/OPD.md`'s capacity argument. No
   pre-registration change, no second code path to test.

**Forward KL: considered, rejected, closed.** The argument for it: under a top-K
teacher, forward KL weights by *teacher* probability, concentrating gradient on
the K events measured exactly, while reverse KL weights by *student* probability
whose tail sits in the lumped `other` bucket. Both cross-tokenizer papers do land
on forward KL. But GKD's broader ablation disagrees, the effect is
task-dependent, and the coverage cost is concrete while the benefit is
speculative for our tasks. **Not implemented.**

**Monitoring obligation.** Reverse KL is mode-seeking, so it collapses entropy by
construction. Log `student_entropy` from step 0. If it falls monotonically the
student is mode-collapsing — and `opd.grpo_advantage` returns `[0.0] * n` when
group reward std `< 1e-8`, so a collapsed student turns the downstream GRPO arm
into a no-op. A tripwire to watch, **not** a reason to change the objective.

**This is not SFT** — worth pre-empting, because "forward KL is SFT" gets used to
argue about the wrong axis. SFT does minimize forward KL, but that identity holds
*only* when forward KL is estimated with one sample drawn from the teacher's own
trajectories. What makes SFT memorize is (a) **off-policy data** — a fixed
dataset of teacher trajectories, so the student never sees the broken states it
will itself produce (Chu et al., *SFT Memorizes, RL Generalizes*, ICML 2025; GKD
found the same independently), and (b) **one-hot targets** — a zero-entropy
target that forces overconfidence, where distilling against a distribution is a
documented regularizer. We are on-policy with distributional targets: neither
mechanism is present.

| | states from | divergence | teacher signal |
|---|---|---|---|
| SFT / SeqKD | teacher | forward KL | 1 sampled token |
| ImitKD | **student** | forward KL | full distribution |
| GKD | student | JSD(β) | full distribution |
| MiniLLM / True OPD | student | reverse KL | full distribution |
| **this plan** | **student** | **reverse KL** | **coarse-grained distribution** |

### (A) Analytic — coarse-grained reverse KL. Primary path, ~90% of spans.

Available when a span is one student token ↔ one teacher token (byte-identical by
construction of the alignment). Build the `K+1` partition from the teacher's
top-K, map each reported teacher id to a student id via the exact-byte map (§8),
drop teacher entries with no byte-identical student token — their mass falls into
`other`, which stays exact:

```
mapped_t = [lp_i for mapped teacher entries]
other_t  = log1p(−Σ exp(mapped_t))                 # fp32
mapped_s = [student logprob at the mapped student id, ...]
other_s  = log1p(−Σ exp(mapped_s))

loss_A = Σ_e π_s(e) · (log π_s(e) − log π_t(e).detach())
```

Differentiable through `mapped_s`/`other_s`; teacher side detached. This is
`opd.topk_reverse_kl`'s math with renormalization replaced by the `other` event —
a strictly correct version of what the repo ships. Zero sampling variance.

**Coverage gate:** if mapped teacher mass falls below ~0.9 of the reported top-K
mass, demote the span to (B) rather than train on a mostly-`other` distribution.

*Still unmeasured:* a 1↔1 span guarantees its own token is in the map (being 1↔1
*means* those bytes are a single token in both vocabs), so §4's ~90% is settled
for the observed token. How many top-K **alternatives** map is not — static bound
65.6%, probability-weighted likely higher since common tokens overlap more. P4
logs the distribution.

### (B) Sampled — policy-gradient surrogate.

Covers multi-token spans plus 1↔1 spans failing the coverage gate — the ~10% (A)
cannot reach. **This is what makes coverage 100%.**

```
logP_s = Σ log π_s(student tokens in span)   # differentiable
logP_t = Σ log π_t(teacher tokens in span)   # scalar, detached, from echo
```

Feed to **`opd.reverse_kl_surrogate` unmodified**. Each sum is
`log π(that side's canonical tokenization of the span text)`; both factorizations
run over the identical byte string. Named caveat: a teacher's probability of a
*byte string* is the sum over its **cover encodings** (Phan et al., Lemma 1), not
of one tokenization — for a complete span on a shared byte boundary the canonical
path carries essentially all the mass, so this is exact for practical purposes.
State it; do not assert exactness.

**(A) and (B) estimate the same quantity** — the analytic low-variance form and
the single-sample form of one reverse KL — so together they supervise every span
without mixing objectives. Which ran on which span is a variance statistic, not a
different loss.

**Rejected, for the record:** span-level Bernoulli KL over `{span text, else}`
for multi-token spans. `exp(Σ log π)` over a 4-token span is ~1e-6, both
Bernoullis degenerate near 0, and the gradient vanishes exactly on the long spans
carrying the most bytes.

**Normalization:** one shared global denominator per optimizer step — total
supervised student tokens across the batch, across **both** estimators, never
per-estimator. Otherwise an A/B mix drifting step to step silently rescales the
learning rate.

## 7. Alignment — byte-offset two-pointer merge

Both vocabs are byte-level BPE: every id decodes to a deterministic, context-free
byte string, and concatenated token bytes equal the source text's UTF-8 bytes.
So alignment is a two-pointer walk over cumulative byte end-offsets — O(n+m),
exact, no fuzzy matching:

```python
while s < n_s and t < n_t:
    s_end, t_end = s_off[s], t_off[t]          # cumulative byte ends
    if   s_end < t_end: s += 1
    elif s_end > t_end: t += 1
    else:
        s += 1; t += 1
        spans.append(Span(student=range(s_start, s), teacher=range(t_start, t)))
        s_start, t_start = s, t
```

Confirmed identical to TRL's `_align_by_byte_offsets` on `main`. **Four
deliberate departures from TRL**, each load-bearing:

- **Both sides' offsets come from local tokenizers, not API strings.** The
  teacher side is tokenized locally and sent as an *integer array*
  (`prompt=[ids...]`, the shape `teacher_fireworks.py` already uses), so offsets
  come from the same `id→bytes` table that produced the ids. Reconstructing
  offsets from returned token strings is a *drift check* (P3), never critical
  path.
- **No trailing catch-all.** TRL dumps leftovers into one giant final group. An
  oversized tail is a desync, not a merge: any span exceeding
  `max_span_student_tokens` (default 8) raises `AlignmentError` with the span
  text.
- **Bayesian indexing, not TRL's default `observed`.** Echo returns the logprob
  *of* each token, so `lp[k]` scores `token_ids[k]` directly; `observed`
  (`lp[k]` predicts `k+1`) would silently score every span one position off.
  Because we send `echo=True` over the **full** prompt, the first completion
  token has a real conditional logprob — no span-0 hole, no guess.
- **No sequence packing.** Byte alignment is per-sample; TRL raises on it.

**EOS is stripped on both sides before alignment, and counted** — it has no byte
extent and is the likeliest source of a phantom zero-width span.

**Prefix asymmetry is accepted and explicit.** Student prefix rendered with
Qwen3's chat template; teacher prefix with `encode_messages` from the vendored
`encoding_dsv4.py`. Control tokens differ — sharing ids is impossible, and
sharing the rendered *string* would feed the teacher malformed control tokens.
Only the sampled action bytes are aligned; the conditioning contexts are each
model's correct rendering of the same conversation. Truncation must therefore
happen at a **shared message boundary**, never independently by token count per
side.

**The teacher has no Jinja chat template.** `deepseek-ai/DeepSeek-V4-Flash-0731` ships
`encoding/encoding_dsv4.py` instead (`encode_messages`,
`parse_message_from_completion_text`); `apply_chat_template` does not exist on
the teacher side. **Vendor the file into `vektori_trace/`, pin and hash it** —
the hash goes in the bridge and in provenance (#7). Format:

```
<｜begin▁of▁sentence｜>{system}<｜User｜>{user}<｜Assistant｜></think>{response}<｜end▁of▁sentence｜>
```

`thinking_mode` selects whether `</think>` closes immediately (chat) or the model
opens a reasoning block. **A pre-registration decision, not a default** — it
changes what the teacher is asked to score. Repo trajectories are non-thinking →
`chat`. Record it either way.

## 8. The exact-byte single-token map

Used **only** by estimator (A), only for top-K alternatives, **never as the
primary alignment.** A static token→token map cannot be the alignment mechanism:
BPE merges are context-sensitive (`" Face"` alone vs inside `"Hugging Face"`), so
a map built from isolated encodings predicts tokenizations the teacher never
produced — failing worst on long identifiers, paths, and JSON-escaped tool args,
the highest-signal positions in an agentic trajectory.

The narrow version is sound: `teacher_id → student_id` **iff both decode to the
identical byte string and that string is a single token in both vocabs.**
Context-independence follows from the byte-level property, and the map is
consulted only inside an already byte-verified 1↔1 span.

Build from `convert_ids_to_tokens` + the ByteLevel unicode→byte table. **Never
`tokenizer.decode()`** — it strips specials, maps unrelated ids to `''`, and on
the DeepSeek side eats leading spaces.

Special tokens are excluded from the map entirely and masked out of the loss;
count `special_tokens_masked` — likely the largest single coverage loss in
tool-call-heavy trajectories, invisible unless counted.

**No similarity metric anywhere.** Not Jaccard, not edit distance, not
embeddings. Correspondence is `bytes(a) == bytes(b)`; alignment is equality of
cumulative byte offsets. The 84 030 figure is a descriptive set intersection,
never consulted at training time.

## 9. Which backend can carry this loss

- **`distill.py` (local student) — the only one that can run estimator (A).**
  `_student_action_logits` returns `logits [1, A, V]`: the full distribution at
  every action position, which is what the `K+1` partition needs.
- **`student_fireworks.py` (hosted student) — cannot.** `run_fireworks_opd`
  trains via `forward_backward_custom`, and `opd_loss_fn(data, logprobs_list)`
  receives **per-token logprobs only, not logits**. The partition cannot be built
  from scalars, so that path carries (B) alone — silently demoting 100% of spans
  to the high-variance branch. **A hard constraint. Record it so nobody proposes
  it as a cheaper host.**
- **`teacher_fireworks.py` — reuse wholesale**: `_post_json`, the echo request
  shape, `_align_scored_entries`'s id-verified alignment, `_entry_logprob`'s
  `logprob`-not-`sampling_logprob` discipline (the latter is temperature-warped
  and would make the objective a KL to the wrong teacher), `TOP_LOGPROBS_CAP`.
- **`echo_mode`** defaults to `"full"` (the shape Fireworks' own distillation
  cookbook uses, therefore the one known to work) but `"last"` (`echo_last=N`)
  ships logprobs for the action only instead of a 3.5k-token prefix. Same
  information for our purposes. **Verify in P0** — a free win on the loop's
  bottleneck.

## 10. Hard-fail list — asserts, not warnings

Each produces a *finite, plausible-looking* loss if it passes silently. That is
the whole reason the design is assert-heavy.

1. Either tokenizer not ByteLevel → abort at bridge build.
2. Byte-length disagreement between the two rendered streams — assert before
   merging. *(Verified hazard: Qwen NFC vs DeepSeek none.)*
3. Prefix/action byte junction not on a teacher token boundary.
4. Special tokens inside the aligned region — partition around, mask, count.
   Include V4-Flash's 465 new added tokens.
5. Span exceeding `max_span_student_tokens` → desync; raise with the span text.
6. `granularity` below `min_alignment_granularity` → abort. Near-zero granularity
   is *sequence-level* distillation being reported as token-level OPD: valid
   arithmetic, false claim.
7. Encoder drift — `encoding_dsv4.py` hash in bridge + provenance; mismatch
   aborts.
8. Provider tokenizer revision drift — echoed strings vs local HF tokenizer.
9. Quantized teacher logprobs (FP8) — record magnitude in provenance. Two runs
   differing only here are not comparable.
10. **Coverage-conditional bias** — dropped spans concentrate on numerics and
    unusual identifiers, i.e. exactly where the models disagree most. Report
    `bytes_aligned / bytes_total` and `frac_dropped` by content type every run.
11. `Σ exp(mapped_t) > 1` (fp8 noise pushing `other` negative) → clamp to a floor
    and count; above 1% of positions the logprobs are too noisy for (A).
12. Off-by-one span reindexing.

All arithmetic in `float32` — `1 − Σexp(...)` near 1.0 loses too much in bf16.

---

# Deliverable 2 — the code

## Modules to create

**`vektori_trace/vocab_bridge.py`** — static, offline, no network, no torch.
Promote `p2_align_report.py`:
```
ByteTable                 id → bytes + fingerprint (mirror tokenizer_check's hashing)
build_byte_table(tok)     from convert_ids_to_tokens + ByteLevel table
validate_byte_table       round-trip assert over a real corpus — non-negotiable gate
assert_byte_level(tok)    refuse non-ByteLevel tokenizers outright
build_exact_token_map     §8's narrow map, specials excluded
CrossTokenizerBridge      artifact: both tables + map + both fingerprints
                          + encoding_dsv4 hash + thinking_mode; save/load JSON
check_cross_tokenizer     sibling to tokenizer_check.check_tokenizers
```
`tokenizer_check.check_tokenizers` **keeps hard-failing the same-vocab case.** Do
not weaken it — the same-vocab path is still the default and its guarantee is
stronger.

**`vektori_trace/align.py`** — pure, per-step, no network, no torch:
```
Span(student_idx, teacher_idx, byte_start, byte_end)
Alignment(spans, n_student_tokens, n_teacher_tokens, granularity, dropped)
align_by_bytes(...)      raises AlignmentError; never best-effort
classify_spans(...)      → ESTIMATOR_A | ESTIMATOR_B | DROP, with a reason string
span_logprob_sums(...)   Σ per side per span
```

**`vektori_trace/cross_kl.py`** — the loss; torch, shape-only, CPU-testable:
```
coarse_grained_reverse_kl(student_logprobs_full, mapped_pairs, teacher_topk)  # (A)
span_surrogate(...)                          # (B), wraps opd.reverse_kl_surrogate
cross_step_loss(spans, ...) -> (loss, CrossStepStats)    # one global denominator
```
`CrossStepStats`: `n_spans`, `granularity`, `frac_A`, `frac_B`, `frac_dropped`
(+ content-type breakdown), `bytes_aligned/bytes_total`, `special_tokens_masked`,
`mean_mapped_teacher_mass`, `student_entropy`.

**`vektori_trace/teacher_cross.py`** — wraps any `IdScoringPool`
(`FireworksTeacherPool` today) with the teacher tokenizer + encoder. **Extend
`teacher_fireworks.py`; do not rewrite it.** The one real change: the integer
`prompt` is built from **DeepSeek ids of the re-rendered teacher-side text**, so
this sits downstream of the encoder rather than assuming shared vocab like
`FireworksTeacherPool` does today. Same duck-typed contract `distill.py` expects
(`score_ids`, `score_ids_topk`, `generate`, `provenance`) so it drops into
`run_opd_training` unchanged. Exposes `probe_echo_support()` = the P0 probe.

**`vektori_trace/encoding_dsv4.py`** — vendored from the HF repo, hashed, pinned.

## Modules to modify

**`vektori_trace/distill.py`** — surgical; same-vocab path untouched:
- `OPDTrainConfig` gains `cross_tokenizer: bool`, `bridge_path`,
  `thinking_mode: str = "chat"`, `min_alignment_granularity: float = 0.5`,
  `max_span_student_tokens: int = 8`, `cross_top_k: int = 5`. **No `beta`.**
- New `encode_prefix_pair(...)` → student ids + teacher ids, truncated at a
  shared message boundary.
- One branch in the step body; provenance records
  `loss: "cross_tokenizer_reverse_kl"` + the bridge fingerprint.
- `verify_tokenizers` in the cross path calls `check_cross_tokenizer` — **not**
  `check_tokenizers`, and **not** nothing.
- `align_topk_rows` **cannot** be reused: it always keeps "the sampled token",
  which does not exist in the teacher's vocabulary here. Leave it untouched for
  the same-vocab path.

**`vektori_trace/cli.py`** — `build-bridge`, `align-report` (offline), extend
`probe-teacher` with `--echo`, and `distill --cross-tokenizer --bridge
--teacher-api-base --min-granularity`. Follow the existing `_teacher_pool_for`
fail-fast pattern (`cli.py:953`): every backend is probed before any GPU time,
because finding out late costs whatever the student instance has already billed.

## Build order

Stages 0–4 need **no API access and no model path.**

| Stage | Work | Blocked on |
|---|---|---|
| 0 | Banners on the stale docs (Deliverable 0) | nothing |
| 1 | `vocab_bridge.py` | nothing |
| 2 | `align.py` + tests. **Equivalence oracle (#12) written first** | nothing |
| 3 | Vendor + hash `encoding_dsv4.py`; teacher-prefix builder + cache | nothing |
| 4 | `cross_kl.py` + `distill.py` branch + P5/P6 fixtures | nothing |
| 5 | `teacher_cross.py` + **P0 echo probe** + P3/P4 | Fireworks model path |
| 6 | P7 real trajectory | P0 passing |

### Stage 3 detail — teacher prefix: cache it

**Network cost is identical either way** — the echo call must carry
`prefix_ids + action_ids` regardless, since the teacher needs the conditioning to
produce conditional logprobs. Caching saves ~1 ms of local tokenization against a
~1 s round-trip, so **speed decides nothing.**

Re-rendering each step re-derives conditioning that is supposed to be frozen. If
anything in the render path is non-deterministic (a date, a locale — DeepSeek's
format has a `<｜latest_reminder｜>` role for exactly that), the teacher conditions
on different text at step 200 than at step 5, the loss stays finite, and the
drift is invisible.

**Cache, for determinism.** `ReOPDStepExample` prefixes are frozen by
construction. Key on `(task, step_index, encoding_dsv4 hash, thinking_mode)`.
~29 MB for 1000 examples × 3584 ids. **A cache miss on an already-seen example is
a hard error** — that miss means the prefix changed, the exact bug the cache
exists to surface.

### If P0 fails

Do not build a degraded loss. Fall back to methods already implemented and
tokenizer-agnostic by construction: teacher-intervention/DAgger
(`reopd.TeacherContinuationExample`) or rejection-sampling SFT. Different methods
that already work, **not** a weakened version of this one. Not automatic — it
surfaces as a decision point.

---

# Verification

**Test-first, non-negotiable — #12's equivalence oracle:**
> Run the cross-tokenizer path with an **identical-tokenizer pair** and assert it
> equals plain `reverse_kl_surrogate` to float tolerance.

Write this **before any loss code**. It is the only test that catches off-by-one
span reindexing, which otherwise yields finite, plausible losses forever.

| | Check | Cost | Status / gate |
|---|---|---|---|
| P1 | Byte tables, ByteLevel assert, corpus round-trip | — | **DONE, passed** (§4) |
| P2 | Offline alignment, granularity by content type | — | **DONE, passed** (§4) |
| P0 | One echo call to V4-Flash-0731 on Fireworks with `teacher_fireworks.py`'s exact shape: `prompt=[ids]`, `max_tokens=1`, `temperature=0`, `echo=True`, `logprobs=True`, `raw_output=True`, `top_logprobs=5`. Confirm `choices[0].logprobs.content` carries `token_id` per entry. Also test `echo_last`. | 1 call | fail → fallback above |
| P3 | Echoed token strings vs local HF tokenizer, 100 strings | ~1 call | any diff → provider drift, stop |
| P4 | Same prompt scored 5×; determinism / fp8 noise; log mapped-alternative coverage | 5 calls | record in provenance |
| P5 | Full path incl. one LoRA step, tiny model + fixture, no GPU (mirror `tests/test_distill.py`) | free | — |
| P6 | Score a teacher continuation vs a deliberately corrupted one; span log-ratio must clearly favor the teacher's own text. **The only test that catches a scrambled alignment.** | ~2 calls | — |
| P7 | One real 40-step agentic trajectory, per-step stats logged | modest | — |

```bash
uv sync --extra train
uv run pytest tests/ -q                       # full suite, offline
uv run ruff check .                           # CI runs both (PR #17)
uv run vektori-trace align-report --bridge …  # P2, offline, no model calls
uv run vektori-trace probe-teacher --echo …   # P0
uv run vektori-trace distill --cross-tokenizer --bridge … \
    --teacher-backend fireworks --teacher-model-id accounts/…
```

---

# Open items — none block stages 0–4

- **Fireworks model path** for V4-Flash-0731 (`accounts/fireworks/models/…`) — stage 5
  only.
- **Confirm `thinking_mode="chat"`** — it changes what the teacher scores.
- **GPU card for the 8B student.** 1×A10G (24 GB) is *tight*: ~16.3 GB bf16
  weights + LoRA optimizer state + activations + ~0.6 GB KV cache during
  `_sample_action`. Needs gradient checkpointing, little headroom. 1×L40S (48 GB)
  or 1×A100-40G is the safe call.
- **`TECHNICAL_NOTES.md` vs `PAPER_BRAINDUMP.md`** — confirm which is superseded
  before bannering.
- **The mined tasks are unmeasured against the student.** Still true, but the
  corpus has changed. `docs/mined-tasks-inventory.md` *(untracked, and now
  superseded)* describes 22 tasks from 4 small pure-Python libraries (anyio,
  click, tenacity, structlog) — a 2026-07-31 snapshot of the local
  `vektori-out-*` runs, taken before the `commit_runtime` pipeline (#26/#27).
  What exists now is **48 task dirs / 46 unique ids, all `prefecthq/prefect`**,
  under `cs/smoke/*/mined_tasks/` on the AWS box — untracked and present on that
  EBS volume only. Both corpora are mining-only (`--no-replay`) with **no
  pass-rate run**. This gates the plan's value: if the student already solves
  them there is no gap to distil into, and the deficit-band selection in
  `select.py` exists precisely to answer this. Risks are unchanged and if
  anything sharper on prefect — contamination especially, since it is a popular
  repo with public PRs, so a pass may be recall rather than capability.
  **Run `passrate` + `select` before spending on stage 5+.**
