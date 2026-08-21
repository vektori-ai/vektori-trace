Capability Routing — Design Doc
===============================

> **Superseded for OPD execution.** This remains background routing context only;
> the current replay-prefix-only OPD experiment is specified in
> `docs/OPD-MULTITURN-PLAN.md`.

**Date:** 2026-07-28 · **Branch:** `step6-train-arms` · **Supersedes:** the Step-6
experiment design in `V0_PLAN.md`. Mining, verification and diagnosis sections of
`V0_PLAN.md` remain authoritative.

Companion: [`docs/OPD.md`](docs/OPD.md) — the OPD method survey this design rests
on. Read it for anything about distillation mechanics.

Execution: [`docs/PILOT.md`](docs/PILOT.md) — what runs first and on which GPU.
It picks the smallest configuration that can execute this design (30B MoE
teacher → 8B student) and orders the steps so each one's result is available to
the next. It also records the trainer decision: **OPD is written
against `train.py`; verl arrives for the GRPO branch**, not before.


Context and Motivation
----------------------

### Problem statement

Given a capability a small model lacks, nobody can currently say **which
intervention will fix it**. Teams try RL, find the gradient is zero, try
distillation, find the teacher never demonstrated the behaviour, and burn weeks
discovering by exhaustion what is decidable by measurement.

The decision is decidable. From `docs/OPD.md`:

- RL's gradient is `Σ A_t ∇log π_s(y_t)`. When all rollouts fail, `A_t = 0` and
  the gradient is **identically zero**. RL sharpens within the model's support;
  it cannot create support.
- OPD's gradient is `Σ (log π_s − log π_t) ∇log π_s`. It is nonzero wherever
  student and teacher disagree, **regardless of outcome**. It moves mass into
  regions the student assigns ≈0 probability.
- OPD is bounded by the *teacher's* support, not the student's.

`pass@k` measures support directly. Therefore `pass@k` decides the intervention.

### Goals

1. **Measure support**, not just difficulty — `pass@k` for k ∈ {1,4,8,16,32} per
   task per model, with an unbiased estimator.
2. **Localise failure to a step** by execution rather than LLM opinion —
   verifier-guided bisection produces the forking step.
3. **Route** each (task × capability) to exactly one of: RL, OPD, quarantine,
   or none, by a rule stated before any data is seen.
4. **Prove routing pays** — routed training beats anti-routed training at
   identical task count, method mix and compute.
5. **Produce a diagnostic report** that stands on its own — the diagnosis, the
   environment and the routing decision are useful output before any training
   runs.

### Non-goals

- **Not** building a general coding agent. Repo-diverse mining serves capability
  isolation, not generality. Competing with DeepSWE on SWE-bench Verified is
  explicitly out of scope.
- **Not** claiming mining is novel. It is a Repo2RLEnv port. Mining is the
  *transfer test*, not the corpus.
- **Not** doing cross-tokenizer distillation in v0. ULD/MinED/DSKD are real but
  unvalidated on 40-step agentic trajectories (`docs/OPD.md` row 14). v1 upgrade.
- **Not** using an adversarial discriminator (GAD). A learned reward model
  substitutes for a verifier we already have.
- **Not** running GRPO at DeepSWE scale. 100k–500k containerised rollouts is
  outside budget; the design accommodates this rather than pretending otherwise.
- **Not** shipping a model. The product is the loop.


Implementation Considerations
-----------------------------

### Constraints

| # | Constraint | Consequence |
|---|---|---|
| C1 | OPD requires `prompt_logprobs` — scoring of *supplied* tokens | Not offered by the generic OpenAI-compatible surface, so a self-hosted vLLM/SGLang teacher always works. Two vendors expose it through their own extensions — see [`docs/HOSTED_TEACHERS.md`](docs/HOSTED_TEACHERS.md). |
| C2 | Teacher and student must share a tokenizer | Same model family. Verified by check, not assumption. |
| C3 | Containerised agent rollouts cost minutes each | Environment interaction must be removed from the training loop. |
| C4 | No corpus is currently kept: 6 tasks have been mined across two `structlog` runs and neither run's output was retained. Measured yield is 2/20 merged PRs (10%), 12 of the 18 losses for `no_linked_issue` | Mining to 50 tasks needs ~500 PRs, and the output has to be persisted and version-pinned. External gyms required for per-cell power. |
| C5 | Gyms overlap SWE-bench and student pretraining | Mined held-out slice is the only clean evaluation. |
| C6 | Frontier scores are scaffold-dependent | Every number names its scaffold. |

### Design principles

- **Execution over opinion.** Where an LLM judgment and a test run disagree, the
  test run wins. Bisection exists because `diagnose.py` currently has no ground
  truth to check itself against.
- **Pre-register every threshold.** Routing rule, effect size, escalation
  schedule — all fixed before data.
- **Measure before training.** `pass@k` establishes whether the headroom an
  intervention needs is there at all. Training before that measurement teaches
  nothing you can attribute, so the measurement runs first.
- **Separable halves.** The diagnostic half stands on its own; the training half
  depends on it, not the reverse.

### Trade-offs taken

| Decision | Alternative | Why |
|---|---|---|
| ReOPD prefix replay as primary training mode | full on-policy rollouts | removes containers from the training loop; ≥4× faster; converts distributed-systems risk into data-loading |
| Qwen3-Coder-Next 80B teacher | Kimi K3 | K3 has no small sibling (cross-tokenizer), needs 8×H200 at MXFP4, and OPD queries the teacher every step |
| Kimi K3 as oracle | GPT-5 | open weights make the ceiling reproducible by a reviewer |
| Reverse KL | forward KL | 8B cannot cover an 80B distribution; mode-seeking beats hedging under capacity mismatch |
| Gyms as corpus, mining as transfer test | mining as corpus | mined volume to date cannot support the claim; and transfer is the stronger result |
| verl for GRPO, OPD written here | verl for both; vendored TRACE for both | GRPO's multi-turn rollouts and weight-syncing are hard to hand-roll and verl has them. OPD is a loss plus one `prompt_logprobs` call against a server we already run — verl's teacher *pool* solves multi-node throughput we do not have. See [`docs/PILOT.md`](docs/PILOT.md). |


High-Level Behavior
-------------------

```
[0] ENV SUPPLY      mined repos + imported gyms  →  executable tasks + verifiers
      ↓
[1] PAIRED REPLAY   teacher + student, pinned scaffold  →  trajectories
      ↓
[2] PASS@K PROFILE  n=8, escalate to n=32 on zeros  →  support classification
      ↓
[3] LOCALIZATION    verifier-guided bisection  →  forking step per failure
      ↓
[4] DIAGNOSIS       capability labels, grounded against [3]
      ↓
[5] ROUTING         (task × capability)  →  {RL, OPD, quarantine, none}
      ↓
[6] TRAINING        RL branch (GRPO)  ‖  OPD branch (reverse KL)
      ↓
[7] EVALUATION      held-out, paired, B1–B4 arms, non-regression
      ↓
[8] PROVENANCE      every number re-derivable from disk
```

### Current status

**Updated 2026-07-29**, against `main` @ `d370605` (PR #16 merged). Every stage
of the Implementation Outline now has a module and, except where noted, a test.
The blocker is no longer code — it is *execution*. Nothing below has been run at
scale, and the `pass@k` sweep has not been run.

| Stage | Module(s) | Status | Missing |
|---|---|---|---|
| 0 Env supply | `mining/`, `gym_import.py` 303 | 6 tasks mined across two `structlog` runs (4 + 2), none persisted; gym adapter coded | a corpus that is kept, and one real gym import |
| 1 Paired replay | `gap.py` 119, `rollout.py` 86 | coded | execution at scale, throughput |
| 2 pass@k | `passk.py` 505 | estimator, two-stage escalation, luck controls, per-capability aggregation — all present and unit-tested | **the sweep itself, on real tasks** |
| 3 Localization | `resume.py` 483, `intervene.py` 316 | bisection + resume coded | **the desync rate — unmeasured, and the design depends on it** |
| 4 Diagnosis | `diagnose.py`, `grounding.py` 116 | grounding coded | real forking steps to ground against (needs [3]) |
| 5 Routing | `routing.py` 421 | rule table coded, R1–R6 declared; end-to-end run on synthetic curves | R7 band still undeclared; real curves |
| 6 Training | `train.py` 360, `opd.py` 255, `dataset.py` 267, `reopd.py` 306, `modal_env.py` 17 | SFT+LoRA and OPD loss coded, unexecuted | teacher pool, container pooling, one real run |
| 7 Evaluation | `arms.py` 899, `nonregression.py` 167 | A0–A4 and B1–B4 coded | never run |
| 8 Provenance | throughout | designed | — |

~16.8k LOC in `vektori_trace/`, ~8.3k in `tests/`, 39 test files, zero
`TODO`/`FIXME`/`NotImplementedError`. The earlier reading of this table — "no
training infrastructure" — is no longer true. What is true is that none of it
has met a GPU or a real corpus.

**Routing's pre-registration is now closed, with one exception.** `routing.py`
implements R1–R6 and the decision table below declares all six — R5 (student has
support, teacher does not) and R6 (support never measured) were added during
implementation and are folded in rather than left undeclared. Both route to
QUARANTINE, so neither can move a cell into RL or OPD. The exception is R7
(`low < pass@1 < high`), which this document still does not specify and which
`routing.py` reports as `preregistered: false`. Leaving a rule undeclared while
running the experiment forfeits the pre-registration, which is the property that
makes the result worth anything — so R7 is either declared before the sweep or
excluded from the B1/B2 claim.


Support Measurement (Stage 2)
-----------------------------

**Estimator.** Sample `n` once, estimate every `k` from that sample:

```
pass@k = 1 − C(n−c, k) / C(n, k)          c = number of passing rollouts
```

Never run five separate sweeps.

**Two-stage escalation.** Stage 1 samples n=8 on all tasks, both models. Stage 2
escalates to n=32 **only for tasks at 0/8** — precisely the tasks where the
support question is live.

Cost: ~1,300 rollouts for 50 tasks against a naive 3,200.

**This is a sequential design and is pre-registered as such.** Escalating only
zeros biases naively pooled estimates upward. Report per stratum; never pool
stage 1 and stage 2 into one estimate.

**Aggregation unit is (capability, model).** `pass@k` is observed per task;
routing decisions are made per capability. Tasks are grouped by which capability
`diagnose.py` marks LACKING, and curves are aggregated within each group. `N` is
printed beside every rate.

**Luck controls.** Any task whose only passes occur at k>8 is quarantined for
review before it may set a routing decision: its passing patch is diffed against
the gold patch, and an independent second sample of n=32 must reproduce the sign
of the decision. A routing decision driven by one lucky rollout is the failure
mode that invalidates the result.


Localization (Stage 3)
----------------------

**Purpose.** Find, by execution, the step at which a failed trajectory became
unrecoverable — the *forking step*.

**Method.** Binary search over prefix length `T`:

1. Replay the student's failed trajectory to step `T`.
2. Hand the resulting environment state and history to the teacher; the teacher
   continues to termination through the pinned scaffold.
3. The mined verifier grades the result (F2P/P2P, separate container).
4. Search for the **largest** `T` at which teacher-continuation still succeeds.
   Step `T+1` is the forking step.

**Cost.** `log₂(steps)` teacher continuations per task — 5–6 for a 40-step
trajectory. Teacher access required is **text generation only**, so any model
serves, including hosted K3.

**Two outputs, both load-bearing:**

- *Training data.* Teacher continuations that pass, with the student prefix
  masked out of the loss. SAGE-OPD's finding is that selective intervention beats
  uniform; bisection selects by execution rather than by LLM turn-judgment.
- *Diagnostic ground truth.* The forking step is where the capability failed,
  established by execution. `V0_PLAN.md` names the labeller the weakest
  instrument in the system (87.8% mean, 50% floor) and sets a stop-condition on
  it. This is the first ground truth available to calibrate it against.

**Monotonicity is an assumption, and it is measured rather than assumed.**
Recoverability is usually monotone in `T` but not always — a student can wander
back into a recoverable state. Each probe point is sampled twice, and the
non-monotonic fraction is reported as a statistic.


Routing (Stage 5)
-----------------

Every (task × capability) pair receives exactly one label. The rule is fixed
before data collection.

| Rule | Candidate `pass@k` | Teacher `pass@k` | Classification | Route |
|---|---|---|---|---|
| R1 | pass@1 low, **pass@32 > 0** | high | in support, unreliable | **RL** (GRPO) |
| R2 | **pass@32 = 0** | high | outside student support | **OPD** |
| R3 | pass@32 = 0 | **pass@32 ≈ 0** | outside teacher support | **QUARANTINE** |
| R4 | pass@1 high | high | no deficit | **NONE** (excluded) |
| R5 | pass@32 > 0 | **pass@32 ≈ 0** | teacher lacks support the student has | **QUARANTINE** |
| R6 | not measured | any | support never established for this cell | **QUARANTINE**, no claim |

Thresholds, pre-registered: "pass@1 low" is ≤ 0.25; "pass@1 high" is ≥ 0.75;
"pass@32 = 0" is 0/32 passing after luck controls; "teacher pass@32 ≈ 0" is
≤ 1/32.

R5 and R6 were written during implementation and are declared here, before data
collection, so they are pre-registered on the same footing as R1–R4. Both route
to QUARANTINE: neither can move a cell into RL or OPD, so folding them in cannot
manufacture a positive result. What they do is name two cells R1–R4 left
undefined — a teacher that cannot do what the student can, and a cell whose
support was never measured — which otherwise fall through to a default nobody
declared. `routing.py` reports every decision's rule and its pre-registration
status in the manifest, so this table is checkable against what ran.

**Still not pre-registered: R7** (`low < pass@1 < high` with support → RL). The
band between the two thresholds is deliberately left undeclared here, and
`routing.py` marks R7 decisions `preregistered: false`. Exclude them from the
B1/B2 claim or declare the band before the sweep runs — not after seeing which
way it falls.

**Quarantine** is not a discard. It is the routing outcome for tasks no
intervention can fix, and it splits by cause on inspection:

- *Broken task* — impossible F2P set, environment defect, ambiguous instruction.
  This is an **environment-quality signal** and feeds back into mining filters.
- *Genuine frontier limitation* — the capability is absent from the teacher too,
  so it cannot be distilled and there is no verified trajectory to reinforce.

Both are reportable findings. The fraction of automatically-mined tasks solvable
by nobody is a number the mining literature does not publish, and it is obtained
here as a by-product.


Training (Stage 6)
------------------

**Rollout economics govern the design.** A standard GRPO configuration is 64
tasks × 8 rollouts = 512 rollouts per gradient step; at 200–1000 steps that is
100k–500k containerised agent rollouts. This budget does not exist.

Mitigations, in order of leverage:

1. **ReOPD prefix replay.** Prefixes come from pre-collected teacher
   trajectories; the student acts at one step; that step is supervised. No
   containers in the training loop, ≥4× faster per rollout.
2. **Shared environment per repo** (SWE-smith design) — one image per repo, not
   per task; ~500× storage reduction.
3. Warm container pools; never cold starts.
4. LoRA only — also the non-regression mechanism, base weights untouched.

**Loss masking.** `dataset.py` already restricts loss to `role == "assistant"
AND subagent_depth == 0`. Teacher-continuation training adds one predicate: the
student prefix is masked. Same machinery, same tests.

**Configuration:**

```
teacher:  Qwen3-Coder-30B-A3B, self-hosted vLLM on Modal    (prompt_logprobs)
          MoE, ~3.3B active — teacher latency is the loop's bottleneck,
          since OPD queries it every step. 80B is the scale-up (SCALE_*).
student:  Qwen3-8B + LoRA
oracle:   Kimi K3                                            (text generation)
loss:     reverse KL (mode-seeking; student cannot cover teacher)
trainer:  OPD here, on train.py + peft; verl for the GRPO branch
```


Evaluation (Stage 7)
--------------------

| Arm | What | Kills |
|---|---|---|
| A0 | student, untouched | — (floor) |
| A1 | student + deficit-targeted prompt | "couldn't you just prompt it?" |
| **B1** | **routed**: RL on in-support, OPD on out-of-support | the claim |
| **B2** | **anti-routed**: identical tasks, compute and method mix, assignments inverted | **the control** |
| B3 | RL on everything | "just use RL" |
| B4 | OPD on everything | "just distill" |
| A4 | oracle (Kimi K3) | — (ceiling) |

**B2 is the arm that makes this real.** It holds task count, method mix and
compute identical and permutes only the assignment, isolating the routing
decision from everything else. A random control cannot do this.

Primary metric: `pass@1` on the held-out mined slice, B1 vs B2, paired
task-by-task. **The resolvable effect size is stated before training.** SWE-Gym
moved a 7B model ~3 points; a 50-task slice cannot resolve 3 points, so the slice
is sized against the expected effect or the claim is scoped to what it resolves.

Also reported: A0 vs A4 (the gap), non-regression inside a pre-declared tolerance
(IFEval or the base model's own evals), and cost per solved task.


Error Handling
--------------

| Condition | Behavior |
|---|---|
| Infra failure during a rollout | excluded from `pass@k` denominator; never recorded as a loss (existing `V0_PLAN.md` rule) |
| Trajectory replay desync at step `T` | hard fail; task dropped from bisection; desync rate reported |
| Bisection non-monotone | recorded, not silently resolved; reported as a fraction |
| Task passes only at k>8 | quarantined pending patch-vs-gold diff and independent re-sample |
| Teacher `pass@32 ≈ 0` | QUARANTINE, split by cause on inspection |
| Routing cell underpopulated | reported as underpowered; no claim made from it |
| Tokenizer mismatch detected | hard fail at startup, before any GPU is allocated |

**Underpowered is a first-class result.** "Not enough data, N more needed" is a
valid output and is preferred to a claim asserted from four observations.


Future-Proofing
---------------

- **The gap number has a shelf life.** K2 → K2.5 → K2.6 → K3 inside a year;
  DeepSeek V3 → V4; GLM-5 → 5.2. Every figure pins its date, weights and
  scaffold. The *routing method* is invariant to which models are plugged in —
  that is the durable asset, and it is why the paper's headline is the decision
  rule rather than any point measurement.
- **Cross-tokenizer distillation (ULD/MinED/byte-level)** is the upgrade path if
  the same-family teacher proves too weak. Deferred, not rejected.
- **`verifiers` / OpenEnv interface** as an adapter over harbor — gym
  compatibility and a publication channel to the Environments Hub, without
  rewriting the container and verifier work that is already done.
- **Serving cost, not capability, is the widening gap.** K3 needs 8×H200
  minimum. The distance between what is SOTA and what an enterprise can afford
  to run on every commit is growing. That is the market.


Implementation Outline
----------------------

```
Land step6-train-arms → main
   │
   ├─► [A] Spike trajectory state resume ──────┐   2 days · informs the design
   │                                            │
   ├─► [B] Gym import adapter (R2E-Gym/SWE-smith)  unblocks volume
   │        │                                   │
   │        └─► [C] pass@k ─────────────────────┤   the support measurement
   │                    │                       │
   │                    ├─► [D] Bisection localizer ◄┘
   │                    │        └─► [E] Ground diagnosis against D
   │                    │
   │                    └─► [F] Routing rule + report
   │                              │
   └─► [G] Rollout infra ─────────┴─► [H] OPD + RL branches ─► [I] B1–B4
```

**Step 0 — verify the tokenizer.** Load `Qwen3-Coder-Next-80B` and `Qwen3-8B`
tokenizers, compare `vocab_size`, hash the merges. 30 seconds. Gates everything.

**Step A — trajectory resume spike (2 days).** Replay tool calls into a fresh
container with a hard consistency assertion: post-replay `git diff` must match
the transcript-implied diff at step `T`. Measure the desync rate. No OPD paper
hits this problem because they all use ALFWorld/WebShop/math, which reset for
free. High desync makes ReOPD mandatory rather than merely preferable — that must
be known in week 1.

**Step B — gym import.** R2E-Gym or SWE-smith behind the existing task interface.

**Step C — `passk.py`.** Estimator, two-stage escalation, per-capability
aggregation, luck controls. Then **run the sweep and look at the histogram.**

If tasks do not separate into distinct curve regimes, there is no routing
decision to make and the honest report says so — that is a result about the
task distribution, and it is worth having early. Scope G in parallel; build it
once C has produced curves.

**Step D — `intervene.py`.** Bisection driver + teacher continuation.

**Step E — grounding.** Compare `diagnose.py` labels against forking steps;
recalibrate `min_gap` from measured blur.

**Step F — `routing.py`.** Decision rule, per-cell counts, quarantine split.

**Step G — rollout infra.** ReOPD prefix replay, container pooling, teacher pool.

**Step H — training branches.** verl: GRPO and `distillation_ppo_loss`.

**Step I — arms.** B1–B4 in `arms.py`; pilot 10 tasks per arm before the full run.


Testing Approach
----------------

**Unit**

- `pass@k` estimator against the closed form for known (n, c); n=0, c=0, c=n,
  k>n boundaries.
- Two-stage escalation: pooled and per-stratum estimates diverge as expected on
  synthetic data.
- Bisection returns the planted forking step on a synthetic trajectory with a
  known break point; non-monotone case is detected, not silently resolved.
- Routing rule: every cell of the table, plus threshold boundaries.
- Prefix masking: student-prefix tokens carry `IGNORE_INDEX`; teacher
  continuation tokens do not. Extends the existing `dataset.py` suite.

**Integration**

- One mined task end to end: replay → pass@k → bisection → route → report,
  against a committed fixture, no network.
- Tokenizer compatibility check fails loudly on a deliberately mismatched pair.
- Trajectory replay consistency assertion fires on a synthetically desynced
  container.

**Manual**

- Hand-inspect 10 forking steps against transcripts. Does the located step match
  where a human reader says it went wrong? This validates the instrument that
  validates the labeller.
- Hand-inspect every quarantined task and classify broken-vs-frontier-limited.
- Pilot `run-arms` on 10 tasks: does it execute, do metrics compute, is cost sane.


Acceptance Criteria
-------------------

1. **Given** a task with `n` recorded rollouts of which `c` passed, **when**
   `pass@k` is computed, **then** it equals `1 − C(n−c,k)/C(n,k)` for every
   k ≤ n, and stage-1 and stage-2 estimates are reported separately.

2. **Given** the full corpus, **when** the sweep completes, **then** every task
   carries a support classification, and per-cell counts are printed with `N`.

3. **Given** a failed trajectory of `s` steps, **when** bisection runs, **then**
   it returns a forking step using ≤ `⌈log₂ s⌉ + 2` teacher continuations, and
   reports whether recoverability was monotone.

4. **Given** 10 hand-inspected forking steps, **when** compared to human
   judgment of where the trajectory went wrong, **then** agreement is ≥ 70% —
   the same floor `V0_PLAN.md` sets for the labeller.

5. **Given** a (task, capability) pair with measured curves for both models,
   **when** routing runs, **then** exactly one of {RL, OPD, QUARANTINE, NONE} is
   assigned, by thresholds recorded in the run manifest.

6. **Given** a quarantined task, **when** the report is produced, **then** it is
   classified broken-task or frontier-limited, with the evidence cited.

7. **Given** a task whose only passes occur at k>8, **when** it would set a
   routing decision, **then** it is held until an independent n=32 re-sample
   reproduces the decision's sign.

8. **Given** trained B1 and B2 arms, **when** evaluated on the held-out slice,
   **then** the comparison is paired task-by-task, and the resolvable effect size
   was recorded before training began.

9. **Given** any figure in the final report, **when** a reader has only the
   artefacts on disk, **then** it is re-derivable — arm, task ids and split,
   scaffold name and version, base model and adapter, seed, `pass@k` with `N`,
   paired comparison, GPU/harbor/image digest.

10. **Given** the trained student, **when** non-regression is checked, **then**
    degradation is inside a tolerance declared before training.


Stop Conditions
---------------

Inherited from `V0_PLAN.md` and extended. Any one halts the work.

- Tokenizer mismatch that no same-family substitution resolves.
- Trajectory replay desync rate too high for bisection **and** ReOPD infeasible.
- ★ **`pass@k` curves do not separate into regimes** — no routing decision exists.
- Any routing cell is too sparse to support a claim.
- A1 (prompt) closes most of the gap — this is a prompt tool with an expensive
  backend. Thesis-level finding for two eval runs.
- **B1 ≈ B2** — routing adds nothing.
- Quarantine fraction is so large the corpus is mostly broken tasks.
