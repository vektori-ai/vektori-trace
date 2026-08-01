# What the Case Study Proves — White Paper Thesis

**Date:** 2026-08-01 · companion to [`CASE_STUDY.md`](CASE_STUDY.md) (the *how*)
and [`../PLAN.md`](../PLAN.md) (the design)

This document answers one question: **when the numbers come back, what have we
established?** Written before the run, so the claim isn't reverse-engineered
from whatever the data happens to say.

---

## 1. There are two theses on the table, and they are not the same

`PLAN.md` and the stated conviction are pulling in different directions. Naming
the gap is the whole job of this document.

**Thesis A — the loop is decidable.** `PLAN.md`'s problem statement:

> *"Given a capability a small model lacks, nobody can currently say which
> intervention will fix it. Teams try RL, find the gradient is zero, try
> distillation, find the teacher never demonstrated the behaviour, and burn
> weeks discovering by exhaustion what is decidable by measurement."*

The argument is mechanical, not empirical: RL's gradient is `Σ A_t ∇log π_s`, so
when every rollout fails `A_t = 0` and the gradient is **identically zero** — RL
sharpens within support, it cannot create support. OPD's gradient is
`Σ (log π_s − log π_t) ∇log π_s`, nonzero wherever student and teacher disagree,
bounded by the *teacher's* support. `pass@k` measures support. Therefore `pass@k`
decides the intervention.

Product: **the loop.** `PLAN.md` says so explicitly — *"Not shipping a model. The
product is the loop."*

**Thesis B — small models suffice for real use cases.** The stated conviction:
SLMs will handle these tasks well in their use cases. Product: implicitly, the
economics of serving an 8B instead of a frontier model.

### They need different evidence

| | Thesis A | Thesis B |
|---|---|---|
| Claim type | **relative** — routed beats anti-routed | **absolute** — the 8B clears a usable bar |
| Key evidence | B1 vs B2 at identical compute | post-training pass@1 vs frontier, and cost |
| Fails if | routing ≈ random assignment | the 8B stays far below frontier whatever you do |
| Audience | a reviewer | a buyer |
| Risk if alone | academic; "so what" | undifferentiated; every fine-tuning shop claims it |

**Thesis B can be proven without Thesis A** — fine-tune, show a delta, done. That
is a thousand other companies. **Thesis A alone** is a methods paper nobody buys.

## 2. The synthesis — lead with this

> **Whether a small model can handle a given task is a measurable property of
> that task, knowable *before* you train. We measure it, predict which tasks are
> reachable and by which intervention, and show the prediction held.**

This converts "SLMs will handle your use case" from a hope into a **screening
procedure**. It subsumes both theses: the screening *is* the loop (A), and what
it screens *for* is small-model sufficiency (B).

Three properties make it defensible where a bare pass-rate delta is not:

**It is falsifiable in advance.** The routing rule and its thresholds are fixed
before data (`PLAN.md` §Routing: pass@1 low ≤ 0.25, high ≥ 0.75, pass@32 = 0 is
0/32 after luck controls). A prediction made under a pre-registered rule that
then holds is evidence; a delta found after the fact is not.

**Negative results are product.** If 40% of a repo's tasks are R3 — outside even
the teacher's support — then "don't try to automate these" is a real deliverable.
Every competing pitch says the model can do everything. A vendor who tells a
customer which of their work is *not* reachable is selling something different.

**It is robust to model churn.** `PLAN.md`: *"The gap number has a shelf life…
The routing method is invariant to which models are plugged in — that is the
durable asset."* A pass-rate delta on Qwen3-8B is stale in six months. A
screening procedure isn't.

---

## 3. The conviction metric, made measurable

"SLMs handle those tasks well" needs a number. The routing table already
partitions the task distribution, so use it:

| Bucket | Meaning | Reading for a buyer |
|---|---|---|
| **R4** — pass@1 ≥ 0.75 | already handled | **free**; no training needed |
| **R1** — pass@1 ≤ 0.25, pass@32 > 0 | in support, unreliable | reachable via **RL** |
| **R2** — pass@32 = 0, teacher high | outside student support | reachable via **OPD** |
| **R3/R5** — teacher pass@32 ≈ 0 | outside anyone's support | **not an SLM problem** |
| **R6** — not measured | no claim | — |

### The headline number

> **SLM-addressable fraction** = (R4 + R1 + R2) / (all tasks − R3 − R6)

Read as: *of the work where any model has support at all, what share can an 8B
reach?* Report alongside the **realized** fraction after training — how much of
the addressable set actually converted.

The sentence the paper is trying to earn:

> *On 160 real bug-fix tasks from prefect: 31% the 8B already handles, 34% are
> reachable by a named intervention, 22% are outside any model's support, and
> here is the cost per solved task at each tier.*

Every one of those numbers is a measurement, not a projection.

### The economic claim

`PLAN.md`'s metric is **cost per solved task**, and it is the right one — it
folds capability and price into a single number, so a cheaper model that solves
less doesn't look artificially good.

Report the P0's "cost per 1M tokens" as the *serving* delta of the 30B→8B swap,
and state plainly that the delta comes from the model swap, not from training —
training is what makes the swap survivable. `PLAN.md` already frames the market
this way: *"Serving cost, not capability, is the widening gap. The distance
between what is SOTA and what an enterprise can afford to run on every commit is
growing. That is the market."*

---

## 4. What each deliverable actually proves

Mapping the P0's five artifacts onto the claim chain — including what each one
does **not** establish, which is where papers usually overreach.

| Deliverable | Establishes | Does **not** establish |
|---|---|---|
| **1.** pass@k, k ∈ {1,4,8,32}, frontier vs 8B | the support profile; the R1/R2/R3/R4 partition; the addressable fraction | that any intervention works |
| **2.** Top-3 diagnosed gaps | the deficits are nameable and rank consistently | that the labels are *correct* — needs grounding against bisection (Stage 3) |
| **3.** Synthetic env for gap #1 | a deficit can be turned into an executable, verifiable task | that training on it transfers |
| **4.** Pass rate + cost before/after | the intervention converted addressable → realized | *why* it worked — that's B1 vs B2 |
| **5.** Validity proof | the generated env isolates the deficit: oracle passes, a capable agent lacking it fails | generality beyond this one env |

**Deliverables 1–3 and 5 prove the screening claim.** They need no GPU training
and are the paper's spine.

**Deliverable 4 proves the conversion claim.** It is the expensive part and the
statistically fragile one.

**Neither proves routing pays.** That is B1 vs B2 — routed assignment against
permuted assignment at identical task count, method mix and compute. If the
white paper wants Thesis A stated as *proven*, B1/B2 is not optional.
`PLAN.md`: *"B2 is the arm that makes this real… A random control cannot do
this."*

---

## 5. What kills each claim

Stated now, so a null result is a finding rather than an embarrassment.

| Claim | Falsifier | If it fires |
|---|---|---|
| Support is measurable and separates | pass@k curves don't partition — nearly all tasks in one bucket | **no routing decision exists.** Publish the support profile as a negative result about the task distribution |
| Diagnosis is real, not narrative | `ground` shows labels uncorrelated with bisection-located forking steps | the LLM labeller is describing, not diagnosing. Ship deliverable 1 only |
| Generated envs isolate deficits | validity proof fails — oracle fails, or a lacking agent passes | deliverable 3 is unsupported; the env generator needs work before it's a product |
| Intervention converts | post-training pass@1 inside noise of baseline | report the CI and the effect size the slice can resolve. `PLAN.md`: *"underpowered is a first-class result"* |
| Routing pays | B1 ≈ B2 | **the loop's central claim fails.** Publish it — it is the most useful negative result in the space |

The pre-registered thresholds and the seeded holdout split (`select --seed`,
written to the report) are what make these falsifiers real rather than
decorative.

---

## 6. What N≈160 from one repo can and cannot carry

`CASE_STUDY.md` §2 lands prefect at ~160 mined tasks. Being precise about what
that buys:

**Well-powered — the support profile.** Estimating bucket proportions at N=160
gives CIs of roughly ±7pp. "31% of tasks are R4" is a defensible claim with a
stated interval. This is the paper's headline and it is comfortably resolvable.

**Underpowered — the paired training claim.** `PLAN.md` states the constraint:
*"SWE-Gym moved a 7B model ~3 points; a 50-task slice cannot resolve 3 points."*
After removing R4 (nothing to fix) and R3 (nothing to work with), the trainable
set is maybe 60–80 tasks. For McNemar on B1 vs B2, ~10% discordant pairs at N=70
is ~7 pairs — not a test, an anecdote.

**This is consistent with `PLAN.md`, which already decided it.** Its trade-off
table reads *"Gyms as corpus, mining as transfer test — mined volume to date
cannot support the claim; and transfer is the stronger result."* The case study
doesn't overturn that. It means:

- the **screening claim** rides on mined prefect tasks (clean, contamination-free,
  one real repo — exactly the P0's framing), and
- the **routing-pays claim**, if the paper wants it, needs gym volume behind it,
  with the mined slice as the held-out transfer test.

That is a *better* paper structure than either alone: the method is validated at
volume, then shown to transfer to a real repo nobody trained on. `C5` in
`PLAN.md` says the same thing from the other side — *"Gyms overlap SWE-bench and
student pretraining; mined held-out slice is the only clean evaluation."*

---

## 7. The claim to avoid

**Do not claim the 8B beats or matches the frontier model.** It won't, on
aggregate, and the paper doesn't need it to. The claim that survives contact is
narrower and more useful:

> On the *identified subset*, the 8B reaches parity at a fraction of the serving
> cost — and we can tell you which subset in advance.

That is the whole product. "Small models are as good" is false and checkable.
"We can tell you where small models are good enough, before you spend" is true,
valuable, and nobody else is selling it.

---

## 8. Paper skeleton

1. **The decision nobody can make** — RL vs OPD, the zero-gradient argument,
   weeks burned by exhaustion (`PLAN.md` §Problem statement)
2. **Support is measurable** — pass@k as the estimator, two-stage escalation,
   luck controls
3. **The routing rule** — stated with thresholds, pre-registered, R1–R6
4. **The corpus** — prefect, post-cutoff and therefore contamination-free;
   mining filters; the skip histogram as evidence of selectivity
5. **The support profile** ← *deliverable 1*, the headline
6. **What the model lacks** ← *deliverable 2*, grounded against bisection
7. **From deficit to environment** ← *deliverables 3 and 5*
8. **Conversion** ← *deliverable 4*, with the effect size stated up front
9. **Does routing pay?** ← B1 vs B2, or an explicit statement that it is
   underpowered here and what would settle it
10. **Economics** — cost per solved task by tier; the 30B→8B serving delta
11. **What we could not establish** — the falsifiers that fired, and the ones
    that couldn't be tested at this N

§11 is not a weakness section. It is the reason anyone should believe §5–§10.

---

## 9. The one thing to decide before writing

**Is the white paper's headline the screening procedure, or the training
result?**

- **Screening** (recommended) — provable at N=160 on one repo, differentiated,
  and the negative results are product. The training result becomes supporting
  evidence that the screen predicts something real.
- **Training result** — needs gym volume for power, and lands in a crowded field
  where the reader's prior is "everyone shows +3 points."

Everything in `CASE_STUDY.md` runs the same either way up to Phase 3. The
divergence is whether Phase 4 is the climax or the corroboration.
