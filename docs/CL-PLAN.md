# The Support Ladder — continual learning for a coding agent

**Date:** 2026-08-17 · target: CLEA @ NeurIPS 2026 (deadline **Sep 4 AOE**)
· supersedes the v1 draft of this file · notes it answers: `continual-learning.md`,
`docs/THESIS.md`

---

## 0. One paragraph

A task stream is not uniformly learnable. For some tasks the student already
samples a passing trajectory and RL has a gradient; for others it never does and
RL's advantage is **identically zero**, so only a teacher-bounded signal moves
the weights. But teacher-bounded signals are exactly the ones that move the model
furthest in KL from base — and KL from base is what predicts forgetting
([RL's Razor](https://arxiv.org/abs/2509.04259)). So the stability–plasticity
dilemma has a hidden third axis: **support**. We measure the support profile of a
real repo task stream, define a routing policy over a KL-ordered ladder of
training signals, and run it for two rounds with a full retention matrix.

---

## 1. What is actually new, stated against the 2026 literature

Checked, not assumed. The zero-gradient mechanism is **no longer novel** — there
is a 2026 cluster on rescuing all-fail RL groups:

| Neighbour | What it does | Why we are not it |
|---|---|---|
| [Boundary-Aware Curriculum RL](https://arxiv.org/pdf/2606.22317) | large-k pass@k locates the capacity boundary → teacher guidance past it → RL consolidation | **closest work — read the PDF before writing.** Single-task math, one shot, no retention, no stream profile |
| [RSTG / Distill Where You Fail](https://arxiv.org/html/2608.00782) | SFT on teacher traces for zero-variance negative groups | per-batch rescue trick, not a policy over a stream; no forgetting axis |
| [LSPO](https://arxiv.org/html/2607.27787) | LoRA scaffold recovers gradient on zero-reward cliffs | same |
| [RL-ZVP / No Prompt Left Behind](https://arxiv.org/pdf/2509.21880) | extracts signal from zero-variance prompts | same |
| [TRACE](https://arxiv.org/html/2604.05336) (Mirhoseini, CLEA keynote) | failures → capabilities → synthetic env → per-capability LoRA via RL → MoE | assumes every deficit is RL-trainable; measures once, never re-evaluates |
| [SWE-Bench-CL](https://arxiv.org/pdf/2507.00014) | chronological SWE-bench sequences, CL metrics | **memory only, zero weight updates**; §8 proposes a protocol it never runs |
| [Limit of RLVR](https://arxiv.org/pdf/2504.13837) | RL ↑pass@1, ↓pass@k; bounded by base support | our premise, cited not re-proven |

**Three claims that remain unclaimed:**

- **C1 — the support profile.** On a real mined-repo task stream with an
  executable verifier, what fraction is gradient-dead for RL? Nobody has
  published this number. Every agentic-CL result implicitly assumes it is 0%.
- **C2 — Razor extends to on-policy distillation.** Razor tested SFT vs RL. OPD
  is on-policy (KL-minimal mechanism) but teacher-bounded (creates support). Its
  position on the forgetting-vs-KL curve is untested, and it is the interesting
  point.
- **C3 — routing as a CL policy.** Enter each task at the highest rung with a
  live gradient; beat the deployed full-ladder recipe on retention *and* compute
  at equal capability gain.

C1 alone is a ≤5pp opinion paper. C1+C2 is the research paper. C3 is the upside.

---

## 2. The ladder

Rungs, ordered by KL from base — which is Razor's predictor of forgetting.

| Rung | Data | Policy | Creates support? | KL from base | Forgetting |
|---|---|---|---|---|---|
| **SFT** | teacher traces, hard labels | off-policy | yes | high, unbounded | worst |
| **OPD** | student rollouts, teacher per-token reverse KL | on-policy | yes, teacher-bounded | moderate | middle |
| **RL** | student rollouts, verifier reward | on-policy | **no** — advantage ≡ 0 at c=0 | minimal | best |
| **NONE** | — | — | — | 0 | none |

**The policy:** *a task's support level sets the lowest rung you may enter at;
enter at the highest rung that still has a nonzero gradient. Every rung you drop
below that buys plasticity with retention.*

This is the reframe that matters. `routing.py`'s `{RL, OPD, QUARANTINE, NONE}` is
already this ladder; it was written as a switch and is now a policy with a stated
mechanism.

---

## 3. Buckets (unchanged from `routing.py`, restated in CL terms)

Measured for student and teacher on the same tasks, same scaffold, same verifier.

| Bucket | Condition | Rung | CL reading |
|---|---|---|---|
| **R4** | student pass@1 ≥ 0.75 | NONE | **train nothing, measure every round.** This is where forgetting appears |
| **R1** | pass@1 ≤ 0.25, c₃₂ > 0 | RL | in support, unreliable — RL has gradient |
| **R7** | 0.25 < pass@1 < 0.75 | RL | mid-band; not pre-registered, tagged as extension |
| **R2** | student c₃₂ = 0, teacher c₃₂ > 1/32 | OPD (→RL) | **gradient-dead for RL.** The paper's subject |
| **R3** | student c₃₂ = 0, teacher c₃₂ ≤ 1/32 | QUARANTINE | nobody's support — not a training problem |
| **R5** | student in support, teacher ≈ 0 | QUARANTINE | nothing to distil from |
| **R0** | passes only at k>8 | held | luck control → independent n=32 re-sample |
| **R6** | never measured | QUARANTINE | no claim |

**Support is never proven absent, only bounded.** c = 0/32 ⇒ p < 0.087 at 95%.
That is why the rule is a *count against a stated stratum* (≤1/32), not a rate
you can quietly evaluate at n=8. `routing.py` already enforces this in integer
arithmetic; do not relax it.

---

## 4. The loop

Per round *t*:

1. **MEASURE (student only).** Two-stage pass@k. Teacher is measured **once,
   ever** — it does not change; re-measuring it every round is pure waste.
2. **BUCKET.** Every task → one bucket, at a fixed stratum. Bucket labels from
   round *t* are only comparable to round *t−1* if the stratum matched.
3. **PICK.** This round's target slice = one diagnosed capability deficit
   (`diagnose.py`), split train/holdout by seeded `select.held_out_split`.
   **Holdout tasks never enter training or a replay buffer, ever.**
4. **ROUTE.** Each train task enters at its highest live rung (§2).
5. **TRAIN.** One LoRA, continuously updated across rounds (v1 default; the
   per-round-adapter variant is an ablation, not v1).
6. **RE-MEASURE.** Round-*t* holdout, **every previous round's holdout**, the
   R4 never-trained set, and IFEval (`nonregression.py`). → one row of `A[t][j]`.
7. **RE-BUCKET → the CL result.** Tasks that moved R2→R1 are forward transfer
   *with a named mechanism*: distillation created support that RL can now
   exploit. This movement is the thing that makes it a loop rather than a screen.

### Eval budget discipline — the part that kills naive implementations

Step 6 is quadratic in *T* and is the dominant cost, not training. Three rules:

- **Teacher: once.** Never re-measured.
- **Deep vs shallow.** Full n only on the current holdout + all prior holdouts +
  R4. Everything else gets a shallow tripwire (n=2) whose only job is to detect a
  cell that moved enough to deserve escalation.
- **Escalate on zero only.** Stage 1 at n=4 on all tasks; escalate to the deep
  stratum only where c=0. This is what `passk.py` already implements.
  `--no-escalate` everywhere else.

---

## 5. The retention matrix

`A[t][j]` = pass@k on **holdout of round j** after training rounds 1..t.
Row 0 = frozen student. Lower triangle = forgetting, upper = transfer.

- **AA** `= (1/T) Σ_j A[T][j]`
- **Forgetting** `F_j = max_{t<T} A[t][j] − A[T][j]`, report `F = mean_j F_j`
- **BWT** `= (1/(T−1)) Σ_{j<T} (A[T][j] − A[j][j])`
- **FWT** `= (1/(T−1)) Σ_{j>1} (A[j−1][j] − A[0][j])`
- **Base-skill retention** — IFEval strict-acc, ≤5pt tolerance, **per round**,
  not just at the end (`nonregression.py`)
- **KL(π_t ‖ π_0) on the target slice** — the Razor predictor. Log it per arm per
  round. Without it C2 is unfalsifiable.
- **Cost per solved task**, per arm. The enterprise metric and a listed CLEA topic.

Cells are **rates with n, c, CI** — not binary — so a 6-task holdout still
carries signal.

> **Gradeability screen, mandatory, before any cell counts:**
> `no_gradeable_rollouts`, `infra_failures`, `parse_status == fallback_exitcode`.
> A crashed verifier read as "zero support" would fabricate the paper's headline
> number, and a crashed verifier read as forgetting would fabricate its second.
> This is the single highest-risk failure mode in the entire plan.

---

## 6. Arms

| Arm | What | Why |
|---|---|---|
| **A0** frozen | row 0 of every matrix | lower bound; partly on disk already |
| **A_full** | SFT→OPD→RL on the whole slice, every task | **what people actually deploy.** The strawman that isn't one |
| **A_routed** | R4 skipped, R1→RL, R2→OPD→RL, R3 quarantined | the policy |
| **A_rl** *(stretch)* | always-RL sequential | TRACE-agent proxy; shows the gradient-dead fraction is lost |
| **A_joint** *(stretch)* | all rounds at once | upper bound, no forgetting possible |

Matched compute between A_full and A_routed, stated in GPU-seconds and tokens.
A_routed is expected to win on **retention and cost at equal target gain** —
two axes, and the second is free money for an enterprise workshop.

**If budget covers only two arms: A0 + A_routed**, and the paper is "first
retention matrix for routed agent training" without the routing-pays claim.

---

## 7. Round definition

**Rounds = diagnosed capability deficits.** Each task assigned to exactly one
round by its top-ranked deficit. Keeps the router central (each round has a mixed
R1/R2/R3 population) and is TRACE-native.

Rejected for v1, kept as ablation: rounds = repos (click → jinja → hatch →
prefect → anyio). Cleaner distribution shift, but each round has one dominant
deficit so the router does nothing inside it.

**T = 2.** Two rounds is the minimum that is literally continual and the maximum
the eval budget supports. Round order fixed by deficit rank, stated up front.

---

## 8. Power — read before committing GPU

corpus50_v3 is **60 tasks**, minus 3 network-blocked hatch tasks. T=2 → ~28/round
→ ~8-task holdouts.

- **Well-powered: the support profile.** Bucket proportions at N=57 give CIs of
  roughly ±7–12pp. "38% of the stream is gradient-dead" is defensible with a
  stated interval. **This is the headline and it is comfortably resolvable.**
- **Underpowered: the training delta.** An 8-task holdout cannot resolve a 10pp
  effect. Pre-declare that only ≥25pp effects are resolvable, *before* the run.
  Underpowered is a first-class result (V0_PLAN discipline).

Corpus expansion to ~150 is the single highest-leverage no-GPU action, and it is
out of scope for Sep 4. If the deadline slips, it becomes P0.

---

## 9. Build steps

**P0 — no GPU, do now, blocks everything**

1. `vektori_trace/cl/rounds.py` — assign tasks → rounds from `diagnose` output,
   enforce disjointness, emit `rounds.json` (seed, order, train/holdout).
2. `vektori_trace/evaluate/retention.py` — build `A[t][j]` from `passk.json`;
   AA / F / BWT / FWT with CIs + the §5 gradeability screen. Test against
   synthetic matrices.
3. `vektori_trace/cl/profile.py` — the support-profile histogram with CIs, and
   the round-over-round **bucket transition matrix** (the R2→R1 result).
4. KL logging: `KL(π_t ‖ π_0)` on the target slice, per arm per round. Reuse
   `cross_kl.py`.
5. `vektori trace cl matrix` / `cl profile` CLI + markdown/json report.
6. Backfill row 0 from rollouts already on disk. Free, and a patchy frozen row
   sizes the eval bill.
7. **Pre-register and commit**: buckets, thresholds, stratum, round order, seed,
   arm definitions, matched-compute definition, resolvable effect size.
8. Read [Boundary-Aware Curriculum RL](https://arxiv.org/pdf/2606.22317) and
   [RSTG](https://arxiv.org/html/2608.00782) in full. Write related work — it is
   fully determined already and it is the part that gets the paper accepted.

**P1 — GPU + API spend, needs explicit per-run approval** · the support profile.
Two-stage sweep, student + teacher, all 57 tasks. §10 has the cost model.
**This is the only step that must land.**

**P2 — GPU, per-run approval** · round 1, A_routed vs A_full, matched compute.
Then matrix row 1 + IFEval + KL. **Stop and read before P3.**

**P3 — GPU, per-run approval** · round 2, both arms → the 2×2 matrix and the
bucket-transition matrix.

**P4** · optional backends, only when a trigger fires (§12).

CLAUDE.md rule applies throughout: **approval of this plan is not approval to run
any step in P1+.** Tear down Modal endpoints the moment each round finishes.

---

## 10. Cost model — the gate

Two-stage: n=4 on all 57, escalate to n=32 only where c=0.

- **Student:** 228 + (fraction c=0) × 28 × 57. At ~50% → **~1,030 rollouts**.
- **Teacher:** same shape, but API (DeepSeek via Fireworks) → dollars, not
  GPU-hours, and **paid once for the whole project**.

At the recorded unit cost (memory: L40S @ 2 workers, ~$12 / 4.5h for 24 rollouts
≈ $0.50/rollout) that is **~$500 for the student profile alone**, before any
training. Three levers, decide before approving:

1. Drop the escalation stratum 32 → 16 (halves the tail; weakens the bound to
   p < 0.17 and **requires re-registering the rule** — `routing.py` handles a
   non-32 stratum correctly as a rate).
2. Drop the student 14B → 8B (halves everything; 14B already fails at
   orientation per `qwen14b_fails_at_orientation`, so 8B may be the more honest
   subject anyway).
3. Raise rollout throughput (more workers per GPU) — engineering, not science.

**Nothing runs until this number is approved.**

---

## 11. Falsifiers, stated now

| Claim | Dies if | Then |
|---|---|---|
| Support separates | nearly all tasks land in one bucket | no routing decision exists; publish the profile as a negative result about real task streams |
| C1 is interesting | gradient-dead fraction ≈ 0% | RL-only CL was fine all along — a genuinely useful negative |
| C2 — Razor extends to OPD | OPD's (KL, forgetting) point is off the Razor curve | still publishable: Razor's single-curve claim does not generalise to teacher-bounded on-policy objectives |
| C3 — routing pays | A_routed ≈ A_full on AA, F, and cost | the loop's central claim fails; publish it — most useful negative in the space |
| Rounds are separable | deficit assignment can't be made disjoint | fall back to repo-rounds |
| Cells are trustworthy | any cell's rate moves when infra failures are excluded | halt; fix the screen before anything else |
| Sequential training forgets at all | `F ≈ 0` for every arm | real finding about LoRA-scale agent training; the paper becomes C1+that |

---

## 12. Optional stack — trigger conditions only

Add a backend when its trigger fires in the matrix, never before.

| Observation | Then add |
|---|---|
| `F` large across rounds | ER replay stratified by capability → DER++ |
| Round-*t* merge undoes round-*t−1* | O-LoRA orthogonal stacking, or Agent-Dice merge |
| IFEval / general coding drops while deficits improve | OPLoRA (project off dominant singular directions) |
| Adapter pile becomes the bottleneck | OSFT |

DER++ is near-free for us: the OPD route already captures teacher logprobs
(`distill.py`), so logit-replay needs storage, not new inference.

**Out of v1 entirely:** EWC, meta-rationales/RCL, ContDa, JitRL, harness-edit
routes, corpus expansion, T=3, A_joint.

---

## 13. Paper

**Title direction:** *The Support Ladder: what you can teach an agent, and what
it costs you in what it already knew.*

**Positioning for CLEA:** TRACE finds *what* to teach. We ask whether teaching is
*possible*, and what it costs in retention. Mirhoseini keynotes — cite TRACE as
the neighbour we extend, never as a competitor. Bing Liu keynotes — CLOB/CIS is
the CL frame. Enterprise angle: post-cutoff mined repos are contamination-free
the way a private enterprise repo is, and cost-per-solved-task is the buyer's
metric.

**Skeleton (write once, degrade to the short form by deletion — decide which on
the day P3 either lands or doesn't):**

1. The decision nobody can make — RL vs distillation, the zero-gradient argument,
   cited to the 2026 cluster rather than re-proven
2. Support is measurable — pass@k as estimator, two-stage escalation, luck controls
3. The ladder and the routing policy — pre-registered thresholds
4. The corpus — mined, post-cutoff, contamination-free; mining filters
5. **The support profile** ← C1, the headline
6. **Retention: does the ladder predict forgetting?** ← C2, with KL on the x-axis
7. **Does routing pay?** ← C3, A_routed vs A_full, retention + cost
8. Bucket transitions across rounds — forward transfer with a mechanism
9. What we could not establish — the falsifiers that fired, and the ones this N
   cannot test

§9 is not a weakness section. It is the reason anyone believes §5–§8.

**Track:** research paper (5–9pp) if the matrix lands; opinion paper (≤5pp) on C1
alone if it doesn't. The profile is the submission floor.

---

## Unresolved questions

1. Approve the ~1,030-rollout / ~$500 profile sweep? (§10 — gates everything)
2. Student: Qwen3-14B, or 8B to halve every sweep?
3. Escalation stratum: keep 32, or re-register at 16?
4. A_full = SFT→OPD→RL, or is always-RL the more honest strawman for a coding agent?
5. Arms if budget is tight: A0+A_routed, or A_routed+A_full and no frozen row?
6. One continuously-updated LoRA across rounds (v1 default), or one per round?
7. If only C1 lands — submit the opinion paper, or push the matrix and risk it?
