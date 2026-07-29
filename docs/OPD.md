On-Policy Distillation — Reference
==================================

**Date:** 2026-07-28 · Research reference for `PLAN.md`. Not a design doc.

This file characterises on-policy distillation (OPD): what it is, the design
space, every method family that exists, and which are viable for agentic coding
tasks with execution verifiers. Sources at the bottom; every external claim is
attributed.


What OPD is
-----------

All distillation minimises a mismatch between student and teacher. The question
that defines the entire field is:

> **On whose distribution do you measure the mismatch?**

**Off-policy** (SFT, SeqKD, rejection sampling) measures on the *teacher's*
distribution. Teacher outputs are collected, then the student is trained to
reproduce them. The student is only ever trained on states the teacher visits.

At inference the student visits *its own* states. It errs slightly at step 3,
landing in a state the teacher never occupied, where it has no training signal,
so it errs worse at step 4. Error compounds: behaviour cloning is `O(εT²)`,
an interactive expert is `O(εT)`. **On a 40-step agentic coding trajectory the
`T²` term dominates.** This is why rejection-sampling SFT underperforms on long
horizons, and it is not a tuning problem.

**On-policy** measures the mismatch on the *student's* distribution. The student
generates; the teacher grades what the student actually did, in the states the
student actually reaches.

> **OPD is imitation learning with a queryable expert. It is DAgger for LLMs.**

Every method below is a different answer to "how do I query the expert on the
student's own trajectory, and what does it hand back?"


The design space: three orthogonal axes
---------------------------------------

Every OPD method is a point in this cube. GKD (Agarwal et al.) frames it exactly
this way — a choice of generation source and a choice of divergence.

### Axis 1 — trajectory source

| Source | Name | On-policy? |
|---|---|---|
| Fixed dataset | SFT | no |
| Teacher samples | SeqKD | no |
| Student samples | on-policy | yes |
| Mixture, λ-interpolated | GKD "Mixed" | partial |
| Student prefix → teacher continuation | intervention / DAgger | yes, hybrid |
| Teacher prefix → student acts at one step | ReOPD prefix replay | yes, inverted |

The last row matters disproportionately: ReOPD takes the prefix entirely from a
pre-collected teacher trajectory and only the evaluated step is
student-generated. The supervised step stays on-policy while **environment
interaction during training is eliminated** — reported ≥4× faster per rollout.

### Axis 2 — teacher signal, ordered by density

| Signal | Density | Access required |
|---|---|---|
| Full logit vector | maximum | logits, shared vocab |
| logprob of the student's own sampled token | per-token | `prompt_logprobs`, shared vocab |
| Top-k logprobs | partial | API `logprobs` |
| Accept/reject on student tokens | per-token binary | speculative decoding |
| Turn-level judgment in text | per-step | text generation only |
| Continuation from the student's state | per-step | text generation only |
| Learned discriminator score | per-token, learned | text generation only |
| Scalar pass/fail per trajectory | minimum | **this is RL** |

The bottom row is the punchline: **RL is the maximally sparse member of this
family.** OPD and RL are the same policy-gradient machinery differing in how
dense a signal the supervisor returns. verl makes this literal — enabling OPD
swaps `ppo_loss` → `distillation_ppo_loss`, and `distillation_loss_coef` blends
both.

### Axis 3 — divergence

| Divergence | Behaviour | Use when |
|---|---|---|
| Forward KL | mode-**covering** — spread mass over all teacher modes | student can match teacher capacity |
| Reverse KL | mode-**seeking** — commit to one teacher mode | **student lacks capacity** |
| JSD(β) | interpolates | GKD found JSD(0.1) and reverse KL best |
| Optimal transport / Wasserstein | vocab-agnostic | cross-tokenizer (ULD) |
| Adversarial | learned | black-box teacher (GAD) |

An 8B student cannot cover an 80B teacher's distribution. Forward KL under
capacity mismatch spreads probability across modes the student cannot represent
and yields a hedging policy. Reverse KL commits. Thinking Machines, verl and
MiniLLM all use reverse KL for this reason.


Why this decides RL vs OPD
--------------------------

The theoretical core. Written as arithmetic because the argument is exact.

**RL gradient:**

```
∇L_RL  ∝  Σ_t  A_t · ∇log π_s(y_t)
```

If every rollout in the group fails, `A_t = 0` for all `t`. **The gradient is
identically zero** — not small, zero. This is why DAPO discards all-wrong
groups, and why `pass@k = 0` means RL is structurally incapable rather than
merely slow.

**OPD gradient:**

```
∇L_OPD  ∝  Σ_t  (log π_s(y_t) − log π_t(y_t)) · ∇log π_s(y_t)
```

Nonzero wherever student and teacher disagree, **independent of whether the
trajectory succeeded**. There is no failure mode where it vanishes.

> **RL sharpens within the support. OPD moves mass into regions the student
> assigns ≈0 probability. `pass@k = 0` means the region has no mass. Therefore
> RL cannot fix it and OPD can.**

The nuance that makes it non-obvious: reverse-KL OPD is *simultaneously*
mode-seeking (sharpens, like RL) and gradient-bearing on zero-probability tokens
(expands support, unlike RL). It does both jobs.

**The binding constraint is teacher support, not student support.** *Rethinking
On-Policy Distillation* finds OPD "does not add capabilities outside the
teacher's demonstrated support" — students learn best within the teacher's
competence boundary. This yields the third routing outcome for free: if the
teacher also fails at high `k`, neither method can work.


Method catalogue
----------------

| # | Method | Trajectory source | Teacher signal | Vocab | Agentic-ready | Verdict |
|---|---|---|---|---|---|---|
| 1 | SFT / SeqKD | teacher | text | any | yes | baseline |
| 2 | Rejection-sampling SFT | teacher, verifier-filtered | text | any | yes | current A3 |
| 3 | ImitKD | student | forward KL | shared | partial | dominated by GKD |
| 4 | GKD | mixed λ | any divergence | shared | partial | the general framework |
| 5 | MiniLLM | student | reverse KL via PG | shared | partial | precursor to 6 |
| 6 | **True OPD** (tinker, verl) | student | per-token reverse KL | shared | partial | **primary target** |
| 7 | **Teacher intervention / DAgger** | student prefix + teacher rollout | text | **any** | yes | **bisection lives here** |
| 8 | **ReOPD prefix replay** | teacher prefix + student step | per-token | shared | yes | **≥4× cheaper, no env in loop** |
| 9 | SAGE-OPD | student | turn labels + confidence | shared | yes | +13.3% rel. via selectivity |
| 10 | SpecKD | student | accept/reject | shared | partial | immature |
| 11 | GAD | student | discriminator | **any** | untested | skip — verifier is better |
| 12 | SODA | semi-on-policy | text | **any** | partial | fallback |
| 13 | Rubric-based OPD | student | text grades | **any** | yes | weak vs execution |
| 14 | Cross-tokenizer family | any | realigned logits | **any** | untested | see below |

**Row 14** — ULD (Wasserstein distance between sorted logits, optimal
transport), MinED (greedy vocab alignment by Levenshtein distance), DSKD
(cross-model attention to unify output spaces), byte-level interface,
approximate likelihood matching. These are real and they work, but they are
validated on summarisation, translation and short-form reasoning — **not on
40-step agentic trajectories** — and each loses fidelity at the alignment step.
Treat as a v1 upgrade, never a v0 dependency.

**Row 11 (GAD)** — Qwen2.5-14B student reaches GPT-5-Chat parity on LMSYS-Chat,
beating SeqKD. But it is demonstrated on *chat quality*, where no verifier
exists. Adversarial training buys a learned reward model precisely when you
cannot execute. With `pytest` available it is a bad trade.


Teacher access: what is actually obtainable
-------------------------------------------

The required operation is `prompt_logprobs` — **scoring of supplied tokens**,
not generation logprobs.

| Access path | Generation top-k | `prompt_logprobs` | OPD viable |
|---|---|---|---|
| OpenAI GPT-5-class, Anthropic | no | no | **no** |
| Hosted open-weight (Together, Fireworks, OpenRouter) | sometimes — ~23% of OpenRouter endpoints comply, 5–20 top-k | **no — vLLM extension, not an OpenAI-API feature** | **no** |
| **Self-hosted vLLM / SGLang (incl. Modal)** | yes | **yes** | **yes** |

**The binding constraint is hosted-vs-self-hosted, not open-vs-closed.** A
frontier open-weight model behind someone else's OpenAI-compatible endpoint
gives no more than GPT-5 does. The same weights on your own vLLM deployment give
everything.

Two further cautions:

- Hosted logprobs, where they exist, are unreliable for this purpose:
  Fireworks serves FP8, Together serves INT8 on some tiers. Quantisation noise
  inside a KL term you differentiate through.
- The **tokenizer constraint is independent of hosting**. verl: teachers "must
  share the student's tokenizer/vocab — typically satisfied by picking a teacher
  in the same model family."


Recommended configuration
-------------------------

```
teacher:  Qwen3-Coder-Next 80B     (~45 GB VRAM, 256K ctx, single H100/H200)
student:  Qwen3-8B
oracle:   Kimi K3 (open weights, reproducible ceiling — text generation only)
```

Same family, same tokenizer, teacher is frontier-class on code, fits one node.
Self-hosted on Modal → `prompt_logprobs` → true token-level OPD, verl off the
shelf. Scale the teacher to Qwen3-235B-A22B if the 80B gap proves insufficient.

**Verify before building anything:** load both tokenizers, compare `vocab_size`
and hash the merges. Qwen3-Coder-Next may have diverged from the base family.
This is a 30-second check that gates the entire plan.

Roles are separable by access requirement:

| Role | Needs | Model |
|---|---|---|
| Teacher (OPD) | `prompt_logprobs` + shared vocab | Qwen3-Coder-Next 80B, self-hosted |
| Oracle / ceiling (A4) | text generation | Kimi K3 |
| Intervention (bisection) | text generation | Kimi K3 or teacher |

Kimi K3 is unsuitable as an OPD teacher for three independent reasons, any one
fatal: no small sibling in its family (so cross-tokenizer), 1.4 TB at MXFP4
(8×H200 minimum, 64+ accelerators recommended), and OPD calls the teacher on
every training step. It is an excellent *oracle* — open weights make the ceiling
number reproducible by a reviewer, which a closed model can never be.


Licensing
---------

Training a competing model on OpenAI or Anthropic outputs violates their terms.
Kimi K3 is Modified MIT; Qwen3 is Apache 2.0. An all-open-weights pipeline has
clean provenance end to end.

This is a procurement consideration, not a footnote: customer legal asks where
the training signal came from, and "open-weight models under permissive
licences" is an answer that closes. It is also what makes an on-prem deployment
story deliverable.

Kimi K3 Modified MIT thresholds (read them directly before relying on this):
MaaS above $20M/yr requires a separate Moonshot agreement; above 100M MAU or
$20M/mo revenue the interface must display "Kimi K3". Neither binds a
pre-revenue company.


Cost reality
------------

Standard agentic RL: 64 tasks × 8 rollouts = **512 rollouts per gradient step**;
200–1000 steps = **100k–500k agent rollouts**, each a Docker container running a
multi-turn agent for minutes. DeepSWE trained on 4,500 tasks with substantial
compute.

That budget is not available, so the *design* changes rather than the ambition:

1. **ReOPD prefix replay removes containers from the training loop.** This is
   the single highest-leverage architectural decision available — it converts a
   distributed-systems problem into a data-loading problem.
2. **SWE-smith's shared-environment-per-repo design** — one image per repo, not
   per task; ~500× storage reduction versus SWE-bench's per-task images.
3. Warm container pools, never cold starts.
4. LoRA only — also the non-regression mechanism, since base weights are
   untouched.

A note on scaffolds, because the numbers are not comparable otherwise: Kimi K3
reports 67.3 on SWE-bench Verified **with the mini-SWE-agent harness**, a
deliberately minimal scaffold. Open-weight leaders on full scaffolds report
70–76% (MiniMax M2.5 75.8, GLM-5 72.8, Kimi K2.5 70.8, DeepSeek V3.2 70.0).
The gap is a property of model × scaffold. Pin the scaffold, name it in every
number.


Environment supply
------------------

Executable environments with real verifiers, available today:

| Source | Scale | Verifier provenance |
|---|---|---|
| R2E-Gym | 8.1k | procedural, back-translated from commits; hybrid verifiers |
| SWE-smith | 50k / 128 repos | LLM-injected bugs, real suites, shared env per repo |
| SWE-Gym | 2.4k | real |
| Prime Intellect Environments Hub | 2,500+ | mixed; `verifiers` library, OpenEnv standard |

DeepSWE (Qwen3-32B, pure RL on R2E-Gym) reaches 51% SWE-bench Verified;
SWE-agent-LM-32B (SFT on SWE-smith) reaches 40.2%.

Contamination warning: these overlap SWE-bench repos heavily and the student's
pretraining has seen them. A mined held-out slice is the only clean evaluation —
never train on anything sharing a repo with it.


Sources
-------

- [On-Policy Distillation — Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [verl OPD documentation](https://verl.readthedocs.io/en/latest/algo/opd.html)
- [GKD: Generalized Knowledge Distillation](https://arxiv.org/pdf/2306.13649)
- [Rethinking On-Policy Distillation](https://arxiv.org/pdf/2604.13016)
- [A Few Teacher Steps Go a Long Way](https://arxiv.org/pdf/2607.04574)
- [SAGE-OPD](https://arxiv.org/pdf/2606.19659)
- [Multi-Turn OPD with Prefix Replay (ReOPD)](https://arxiv.org/html/2607.04763)
- [Are Full Rollouts Necessary for OPD?](https://arxiv.org/pdf/2605.31490)
- [GAD: Black-Box On-Policy Distillation](https://arxiv.org/abs/2511.10643)
- [SODA: Semi On-Policy Black-Box Distillation](https://arxiv.org/pdf/2604.03873)
- [ULD: Universal Logit Distillation](https://arxiv.org/abs/2402.12030)
- [Multi-Level Optimal Transport for Cross-Tokenizer KD](https://arxiv.org/pdf/2412.14528)
- [Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching](https://arxiv.org/pdf/2503.20083)
- [R2E-Gym](https://arxiv.org/abs/2504.07164)
- [DeepSWE](https://www.together.ai/blog/deepswe)
- [SWE-smith](https://github.com/SWE-bench/SWE-smith)
- [Prime Intellect Environments Hub](https://www.primeintellect.ai/blog/environments)
- [OpenEnv](https://github.com/huggingface/OpenEnv)
- [awesome-on-policy-distillation](https://github.com/chrisliu298/awesome-on-policy-distillation)
- [Kimi K3 technical blog](https://www.kimi.com/blog/kimi-k3)
