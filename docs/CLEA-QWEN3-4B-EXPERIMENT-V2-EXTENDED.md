# CLEA Tau2 experiment V2-Extended: neutral method map and shared SFT stage

Status: parked research catalogue; no continuation method is selected here  
Date: 2026-08-24  
Target: CLEA @ NeurIPS 2026  
Scope: Tau2-bench only; Qwen3-4B student first; DeepSeek teacher traces  
Parent design: `docs/CLEA-QWEN3-4B-EXPERIMENT-V2.md`

This document records the techniques considered after V2, the data and compute
each requires, what each comparison would measure, and the evidence available
for it. It deliberately does **not** rank, recommend, or select a continuation
method. V2 remains the experiment draft; this extension is a parked decision
resource.

The shared action that does not depend on the later choice is a **30-task
Qwen3-4B Tau2 SFT pilot**. Section 2 specifies that stage. The number 30 is a
bounded pilot boundary, not a claim that 30 tasks are sufficient for a final
model. V2's 60/18/36 retail design remains a possible later scale-up.

## 1. Fixed scope and terminology

- Only Tau2 retail, telecom, and airline data may be used. No coding, generic
  instruction, chat, math, or other training dataset enters an experiment.
- Existing DeepSeek traces may be used only when their task IDs belong to the
  frozen training partition.
- Selection and test traces remain quarantined even though they have already
  been collected.
- The split unit is a Tau2 task/family, not an action, prefix, or trajectory.
- `A0` denotes frozen Qwen3-4B.
- `A_warm30` denotes the checkpoint obtained from the common 30-task SFT pilot.
- `T` denotes the frozen DeepSeek teacher.
- `R` denotes a frozen reference policy when an objective needs one. One
  definable configuration is `R = A_warm30`; this catalogue does not select it.
- A **teacher prefix** contains the conversation and observations from a stored
  DeepSeek trajectory up to an assistant decision.
- A **student action** is a newly sampled assistant completion from Qwen at a
  supplied prefix.
- **Strict on-policy** means the actions are sampled from the current parameters
  used by the update. A fixed or periodically refreshed snapshot is
  semi-on-policy/stale between refreshes.
- **Off-environment** means no sampled student action is executed to obtain its
  next Tau2 observation during training.
- **Black-box teacher** means text generation is available but teacher logits or
  likelihoods are not. DeepSeek likelihood access, if used, makes the relevant
  experiment more informative than text-only black-box distillation.

Tau2 retail, telecom, and airline normally score the final database state and
required communication. Their listed reference actions are one valid solution,
not necessarily the unique correct path. Consequently, matching a stored
teacher action and producing a valid alternative action are different notions
([Tau2 evaluation documentation](https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md)).

## 2. Shared first stage: 30-task Tau2 SFT pilot

### 2.1 Purpose

The first stage produces one shared behavioral checkpoint before choosing among
the continuation methods in this document. It measures whether Tau2-only expert
imitation can move Qwen3-4B toward valid tool use and non-floor held-out
performance.

It does not test SODA, KTO, CPO, GAD, OPD, ReOPD, Lightning OPD, or GRPO. Those
methods begin only after this common checkpoint exists.

### 2.2 Frozen data boundary

Select 30 task IDs from V2's retail-training boundary using a deterministic,
family-balanced procedure. These are `pilot-train30`. They must contain only
training tasks and must not be selected using selection/test performance.

```text
pilot-train30 is a subset of retail-train

pilot-train30 task IDs intersect retail-selection task IDs = empty
pilot-train30 task IDs intersect retail-test task IDs      = empty
```

The 30-task subset should preserve coverage of the workflow attributes already
used by the V2 family grouping: operation/mutation family, policy gates,
conditional structure, initial state, and normalized instruction template.
This is stratification, not performance-based selection.

Use the one already collected DeepSeek trajectory for each task. A second
teacher trace per task is not required to prevent leakage. If a trace fails the
eligibility checks, exclude it and report the resulting eligible count; do not
replace it with a selection/test task.

### 2.3 Eligibility and export

A trajectory enters SFT only if:

1. Tau2 completed and graded it;
2. the official reward passed;
3. messages and tool calls parse exactly;
4. no infrastructure failure explains the outcome;
5. applicable authentication, confirmation, and precondition rules pass;
6. the rendered conversation fits without silent truncation; and
7. the exact serving tokenizer/template can reconstruct it.

Export one conversational trajectory or decision-boundary rows. In both
representations:

- supervise DeepSeek assistant tokens only;
- mask system, user, simulator, and tool-observation tokens;
- keep read, mutation, clarification, confirmation, refusal/no-action, and
  recovery decisions;
- do not convert the trace into Tau2's reference action list; and
- do not truncate a prefix or target to force it into the context window.

### 2.4 Census before training

The run manifest records:

- selected and eligible task IDs and family labels;
- trace hashes and Tau2 revision;
- assistant decisions and supervised tokens per task;
- context and target length quantiles;
- overflow, parse, and zero-target counts;
- tool versus conversational actions;
- read/write/clarify/confirm/refuse/recover counts; and
- realized task and token weights.

Thirty tasks are not thirty training examples. One multi-turn trace can produce
many supervised assistant decisions and many target tokens. Both task count and
decision/token count must be reported.

### 2.5 Objective and sampling

The SFT loss is assistant-only negative log likelihood:

$$
\mathcal L_{\mathrm{SFT}}
=-\mathbb E_{(x,y_T)}\sum_t \log \pi_\theta(y_{T,t}\mid x,y_{T,<t}).
$$

The intended aggregation is task-balanced:

$$
\mathcal L_{\mathrm{task\text{-}balanced}}
=\frac{1}{|\mathcal T|}\sum_{i\in\mathcal T}
  \frac{1}{N_i}\sum_{j=1}^{N_i}\mathcal L_{ij}.
$$

If the trainer cannot implement this exact aggregation, use a task-balanced
sampler and report the realized per-task token mass.

### 2.6 Training and checkpoint record

- Start from the exact frozen Qwen3-4B post-trained revision.
- Train a LoRA with the exact chat template and tool schema used at evaluation.
- Freeze context length, precision, thinking mode, special tokens, tool parser,
  and generation format before the run.
- Run a short memory/loss/gradient canary before the paid training stage.
- Save at least the end-of-epoch checkpoint and optimizer/configuration
  provenance.
- Evaluate `A0` and the SFT checkpoint using identical held-out Tau2 selection
  tasks, seeds, simulator configuration, and generation settings.
- Do not use selection/test traces as training examples or replay prefixes.

The resulting common parent is named `A_warm30`. If the later study moves from
30 to the full 60-task V2 training boundary, the manifest must state whether the
60-task checkpoint is retrained from `A0` or obtained by continued training;
those procedures are not identical.

## 3. The three independent design axes

Several names below combine choices from different axes. Keeping the axes
separate avoids attributing an effect to the wrong component.

### 3.1 Where the optimized completion comes from

| Completion/state source | Policy relation | Environment use |
|---|---|---|
| recorded DeepSeek action at recorded DeepSeek state | off-policy teacher demonstration | none |
| frozen `A0` action | one-time student snapshot | generation only |
| frozen `A_warm30` action | one-time warmed-student snapshot | generation only |
| periodically refreshed student action | semi-on-policy | generation; environment optional |
| current student action at a stored teacher prefix | on-policy action, off-policy history | none |
| current student trajectory executed in Tau2 | fully on-policy trajectory | live Tau2 |
| interleaved teacher/student trajectory | mixed policy | depends on method |

### 3.2 What supervision is available

| Signal | Information content | Required access |
|---|---|---|
| teacher target text | one demonstrated action | teacher generation/trace |
| chosen/rejected pair | relative ordering | pair construction or pairwise judge |
| desirable/undesirable label | one binary label per completion | labeler/auditor |
| scalar outcome reward | trajectory outcome | Tau2 environment/evaluator |
| discriminator score | learned teacher-likeness | teacher and student text |
| teacher likelihood of the student action | dense sequence/token evidence | supplied-text scoring |
| full teacher distribution | maximum token-level information | teacher logits, vocabulary alignment |

### 3.3 Which optimization objective consumes it

| Objective family | Examples |
|---|---|
| maximum likelihood | SFT, SeqKD, filtered/rejection SFT |
| paired preference | DPO, CPO, SimPO, IPO, SODA's DPO stage |
| unpaired binary preference | KTO |
| policy gradient from outcomes | GRPO, PPO, REINFORCE variants |
| adversarial policy optimization | GAD |
| distribution matching | GKD, MiniLLM, standard OPD, ReOPD, Lightning OPD |

Refresh cadence is a data-collection choice, not a loss. KTO, DPO, filtered
SFT, or another compatible objective can each be trained on a static snapshot
or on periodically refreshed samples. Once two branches refresh from their own
diverged policies, their collected data distributions also differ.

## 4. High-level comparison matrix

`Teacher likelihood` below means scoring supplied student text, not merely
receiving logprobs for newly generated teacher text.

| Method | Training completion | Supervision | Pair needed | Reference policy | Teacher likelihood | Live Tau2 during training | Cross-tokenizer path | Refresh behavior |
|---|---|---|---:|---:|---:|---:|---|---|
| teacher SFT / SeqKD | DeepSeek | target text | no | no | no | no | native | static |
| continued teacher SFT | DeepSeek | target text | no | no | no | no | native | static |
| filtered/rejection SFT | teacher or student | pass/desirable filter, then NLL | no | no | no | only if outcomes label data | native | static or periodic |
| DPO | chosen + rejected | pairwise label | yes | yes | no | no | native | static or periodic |
| CPO | chosen + rejected | pairwise label + chosen NLL | yes | no | no | no | native | static or periodic |
| KTO | any labeled completion | binary label | no | yes | no | no | native | static or periodic |
| SODA | DeepSeek positive + `A0` negative | assumed pair ordering | yes | warmed student | no | no | native | one frozen `A0` snapshot in the paper |
| GAD | current student + DeepSeek text | discriminator reward | no fixed pair | implementation-dependent anchor | no | no | native | iterative/current student |
| GKD / live OPD | current student | teacher distribution | no | old/current policy for estimator | yes/logits | not necessarily | normally shared; bridge required here | every rollout/update |
| GOLD | current student | aligned teacher logits | no | old/current policy for estimator | yes/logits | not necessarily | built into method | every rollout/update |
| cross-tokenizer OPD | current student | aligned teacher likelihood | no | old/current policy for estimator | yes | not necessarily | explicit alignment | every rollout/update |
| ReOPD | current student at DeepSeek prefix | aligned teacher likelihood | no | old/current policy for estimator | yes | no | explicit alignment | current action each sampled replay state |
| Lightning OPD | frozen `A_warm` rollout | cached teacher likelihood | no | SFT reference | precomputed once | no | requires compatible scoring/alignment | frozen rollout set |
| Lightning OPD 2.0 | frozen reference rollout | residualized cached teacher likelihood | no | SFT reference | precomputed once | no | requires compatible scoring/alignment | frozen rollout set |
| GRPO | current student trajectory | group-relative Tau2 reward | no | optional KL/reference | no | yes for real outcomes | native | current rollout groups |

## 5. Maximum-likelihood methods

### 5.1 Teacher SFT / sequence-level knowledge distillation

**Data construction**

```text
stored DeepSeek prefix -> recorded DeepSeek action -> assistant-only NLL
```

SFT transfers the explicit actions present in successful traces. It does not
represent teacher uncertainty or teach directly on student-generated errors.
It is tokenizer-independent because the target text is re-tokenized by Qwen.

For this project, initial SFT creates `A_warm30`. Repeating the same operation
after `A_warm30` is continued teacher SFT. It is not a second kind of loss; it
is more optimization on the same objective and data family.

**What a comparison measures**

- `A0` versus `A_warm30`: acquisition from Tau2 expert demonstrations.
- `A_warm30` versus continued SFT: value or harm of additional expert
  likelihood training.
- continued SFT versus another matched continuation: whether that method adds
  value beyond additional demonstration training.

**Primary data risks**

- a successful trajectory may contain unnecessary or policy-questionable
  actions;
- one trace covers only one valid route;
- longer traces can dominate unless sampling/loss is task-balanced;
- exposure bias remains because all training histories are teacher histories.

### 5.2 Filtered or rejection-sampling SFT

Filtered SFT applies NLL only to completions that pass a filter. The candidates
may be teacher completions, student completions, or both.

```text
candidate completion -> verifier/auditor -> retain positives -> NLL
```

If the candidates are current/periodic student samples, the method is also
called rejection-sampling fine-tuning or filtered self-training. Negatives are
dropped and exert no direct gradient.

For Tau2, a whole completed live trajectory can receive official outcome
reward. An isolated action at a replayed prefix does not have that outcome
unless it is executed forward in the environment or an action-level labeler is
used. Policy-compliance checks can identify particular violations without
necessarily determining global action desirability.

**Controlled comparison with KTO**

On a fixed shared dataset, filtered SFT uses all retained positives while KTO
uses the same positives plus labeled negatives. This isolates the information
in negative labels more closely than two independently refreshed branches.
After branch-specific refresh, objective and collection distribution both
change.

## 6. Paired offline preference methods

### 6.1 DPO

DPO consumes `(prompt, chosen, rejected)` triplets and a frozen reference
policy:

$$
\mathcal L_{\mathrm{DPO}}=-\mathbb E\log\sigma\left(\beta
\left[
\log\frac{\pi_\theta(y^+|x)}{\pi_R(y^+|x)}-
\log\frac{\pi_\theta(y^-|x)}{\pi_R(y^-|x)}
\right]\right).
$$

It increases a relative likelihood margin; it does not independently require
the chosen likelihood to rise. Training is offline once pairs and reference
log probabilities are available. DPO was introduced and evaluated on
preference alignment for sentiment, summarization, and single-turn dialogue
([DPO](https://arxiv.org/abs/2305.18290)).

**Possible Tau2 pair sources**

1. DeepSeek action chosen, frozen Qwen action rejected.
2. Two Qwen actions with an independent pairwise ordering.
3. Successful complete trajectory chosen, failed trajectory rejected.
4. Two actions at the same state ordered by an action-level judge.

Teacher-versus-student provenance alone does not establish that every pair is
correctly ordered. This matters in Tau2 because different tool sequences can
produce the same valid outcome.

### 6.2 CPO: Contrastive Preference Optimization

CPO removes the explicit reference terms from the pairwise objective and adds
chosen-response behavior cloning:

$$
\mathcal L_{\mathrm{CPO}}=
-\log\sigma\left(\beta[\log\pi_\theta(y^+|x)-
\log\pi_\theta(y^-|x)]\right)
+\alpha[-\log\pi_\theta(y^+|x)].
$$

The NLL term provides an independent gradient toward the chosen response. The
original ALMA-R/CPO work constructed translation triplets from human
references, GPT-4, and ALMA candidates, scored candidates with translation
evaluators, and kept high/low candidates. It reports results on WMT translation
using about 22K parallel sentences, not on agentic interaction
([CPO/ALMA-R](https://proceedings.mlr.press/v235/xu24t.html)).

The CPO paper proves an upper-bound relationship under an ideal-reference
assumption. That assumption concerns the reference policy matching the true
preferred-response distribution; a large average teacher/student capability
gap is not itself that assumption.

Applying CPO to fixed DeepSeek-positive/Qwen-negative records would be CPO on
SODA-style pairs. It would not reproduce SODA's warmed-reference DPO stage.

### 6.3 SODA

SODA is a particular pipeline rather than a new pairwise loss:

1. sample a one-time response from raw base policy `q0`;
2. take the teacher response as the preferred response;
3. SFT the student on teacher responses to obtain `q_w`;
4. run one DPO stage on `(teacher response, q0 response)` pairs using `q_w` as
   the reference.

Its “semi-on-policy” property comes from negatives sampled once from the
student's pretraining policy. The negatives are not refreshed during the
reported training pipeline. The construction assumes that the teacher response
is preferable to the raw compact-model response.

The SODA paper evaluates compact Qwen2.5 and Llama-3 students on general LLM
distillation benchmarks. It reports matching or exceeding comparison methods
on 15 of 16 reported benchmark results, 10x faster training than its GAD
comparison, and 27% lower peak GPU memory. Those are paper-specific system and
benchmark measurements, not Tau2 timing estimates
([SODA](https://arxiv.org/abs/2604.03873)).

**Tau2-specific label issue**

A base action may be wrong, valid but different, or correct. Counting the
fraction of frozen base actions that are genuinely inferior measures how often
the SODA pair assumption holds. Auditor labels, final trajectory outcomes, and
teacher/student provenance provide different evidence and should not be
silently substituted for one another.

### 6.4 Other paired objectives previously mentioned

- **SimPO** removes the reference model and uses length-normalized sequence
  likelihood plus a target reward margin. It still needs chosen/rejected pairs.
- **IPO** changes DPO's logistic preference objective to an identity-preference
  formulation with a finite target margin. It still needs pairs and a
  reference-style likelihood ratio.
- **ORPO** combines supervised likelihood with a reference-free odds-ratio
  preference penalty. It still needs pairs.

These objectives modify how a fixed ordering is optimized. None independently
determines whether a Tau2 pair is correctly ordered.

## 7. Unpaired binary preference: KTO

KTO consumes one completion and one binary label per record:

```text
prompt + completion + desirable/undesirable
```

It compares the policy/reference log-likelihood ratio with a KL-derived
reference point and applies separate desirable and undesirable value
functions. It is pairing-free but normally not reference-free. The original
paper reports that KTO can match or exceed paired preference methods on its
general alignment experiments from 1B to 30B, while also stating that no one
loss is universally superior
([KTO](https://arxiv.org/abs/2402.01306)).

KTO's class weights control aggregate desirable/undesirable contributions. The
paper's imbalance condition can be written approximately as:

$$
\frac{\lambda_D n_D}{\lambda_U n_U}\in[1,4/3].
$$

This compensates for unequal retained counts. It does not correct mislabeled
examples. Current TRL exposes `desirable_weight`, `undesirable_weight`,
`beta`, and a reference model or frozen reference adapter
([TRL KTOTrainer](https://huggingface.co/docs/trl/kto_trainer)).

### 7.1 Static KTO

Sample student actions once from `A0` or `A_warm30`, label them, and train KTO
offline against a frozen reference. This permits a fixed shared-dataset
comparison with filtered SFT.

### 7.2 Periodically refreshed KTO

At one or more checkpoints:

```text
current snapshot -> sample actions at stored prefixes
                 -> label actions
                 -> add/replace training records
                 -> continue KTO
```

Refresh cadence is an extension to the data pipeline, not part of the original
KTO loss. The experiment must specify:

- snapshot cadence;
- replacement versus accumulation;
- action sampling temperature and samples per state;
- whether `R` remains `A_warm30`;
- labeler version and confidence threshold;
- treatment of old examples and label changes; and
- task/state weighting.

If KTO and filtered-SFT branches refresh from their own policies, they no
longer share the same post-refresh data distribution. Such a result compares
complete iterative algorithms rather than only the contribution of negatives.

### 7.3 Four labels to binary labels

KTO itself accepts desirable/undesirable. A four-class action auditor requires
a preregistered mapping. A generic high-precision mapping is:

| Auditor class | One possible KTO mapping |
|---|---|
| definitely desirable | desirable |
| definitely undesirable | undesirable |
| ambiguous / multiple valid paths | exclude |
| invalid or unscorable context | exclude and diagnose |

The actual class names and semantics must come from the implemented auditor.
Forcing ambiguous middle classes into binary labels changes the target dataset
and must be treated as a separate recipe.

Relevant measurements include class prevalence and label precision. For
negative-label reliability, the direct quantity is:

$$
P(\text{truly undesirable}\mid\text{auditor says undesirable}),
$$

not only the percentage of samples receiving an undesirable label.

## 8. Distribution-matching and on-policy distillation

### 8.1 GKD and standard/live OPD

Generalized Knowledge Distillation trains on a mixture of fixed and
student-generated sequences and allows multiple student/teacher divergences.
The student-generated endpoint of GKD is commonly called on-policy
distillation. GKD was evaluated on summarization, translation, arithmetic, and
instruction tuning
([GKD](https://arxiv.org/abs/2306.13649)).

For a student sample `y ~ pi_theta`, reverse-KL OPD uses teacher and student
likelihood information over the sampled sequence. Implementations use a policy
gradient/surrogate estimator because the expectation is over student samples.

Strict multi-turn live OPD executes current student actions in Tau2, obtains
the resulting observations, and queries the teacher at student-occupied
histories. It therefore requires repeated environment rollouts and teacher
scoring during training.

### 8.2 Cross-tokenizer OPD

Standard token-level OPD assumes aligned vocabularies. Qwen and DeepSeek do not
share a tokenizer. A cross-tokenizer method must define how likelihood mass on
one tokenization is transferred to the other.

The cross-tokenizer method discussed for this project synchronizes equivalent
byte/text chunks, sums student and teacher likelihoods over each chunk, and
assigns the chunk-level teacher evidence back to student tokens before the
clipped policy update. The exact action bytes are held fixed while Qwen and
DeepSeek tokenize and render their own semantically equivalent histories.

`Breaking the Tokenizer Barrier` reports a precise token-mapping method for
OPD across model families and evaluates cross-family distillation on its
benchmarks. It is the primary source for the semantic-prior/chunk assignment
used by the current local design
([cross-tokenizer OPD](https://arxiv.org/abs/2606.09456)).

Other cross-tokenizer families include Universal Logit Distillation,
optimal-transport alignment, edit-distance vocabulary mapping, byte-level
interfaces, and approximate likelihood matching. They differ in whether they
align ranked distributions, token strings, byte spans, or sequence
likelihoods. Their published benchmarks are mostly non-agentic generation;
Tau2 transfer remains an empirical question.

#### GOLD

General On-Policy Logit Distillation (GOLD) is a tokenizer-agnostic OPD method
released by Hugging Face. It extends Universal Logit Distillation from fixed
offline sequences to current student samples and addresses two distinct
alignment problems:

1. sequence alignment, including a June 2026 update using UTF-8 byte offsets;
2. vocabulary alignment, using direct mappings for shared token content and a
   sorted-distribution fallback for unmatched tokens.

The published experiments use the Countdown reasoning task. The report states
that, for a Llama-3.2-1B student and Qwen3-4B teacher, GOLD recovered 60% of the
teacher performance versus 10% for its ULD baseline. It also reports
same-tokenizer GKD and cross-tokenizer GOLD comparisons with a task-specific
GRPO baseline. These are Countdown results and do not establish Tau2 behavior
([GOLD report](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation)).

GOLD aligns distributions across vocabularies. The `Breaking the Tokenizer
Barrier` chunk/semantic-prior method aligns teacher evidence to student tokens
through a different mapping rule. They are separate cross-tokenizer
estimators, even when the surrounding student-sampling loop is the same.

### 8.3 Replayed-Prefix OPD (ReOPD)

ReOPD reuses stored teacher trajectories as state prefixes:

```text
stored DeepSeek prefix at position t
    -> current Qwen samples one action
    -> teacher scores that exact action
    -> update Qwen
    -> reset to another stored prefix
```

The current action is on-policy. The preceding history is not: it is a replayed
teacher history. The sampled action is not executed, so no new environment
observation follows it.

The ReOPD paper identifies a prefix trap: student-occupied histories are more
on-policy but may be states where teacher feedback is less reliable. Its
reported schedule samples replay positions with an early-step bias. On math
with Python and search environments, it reports preserved or improved OPD
accuracy, zero training-time tool calls, and at least 4x faster rollouts than
full OPD
([ReOPD](https://arxiv.org/abs/2607.04763)).

For this project, ReOPD additionally requires cross-tokenizer scoring. That
combination is distinct from paper-identical same-tokenizer ReOPD and must be
reported as cross-tokenizer ReOPD.

### 8.4 Lightning OPD

Lightning OPD performs a one-time preprocessing stage:

```text
SFT reference policy generates fixed rollouts
    -> teacher scores every rollout once
    -> cache teacher likelihoods
    -> train student repeatedly without a live teacher server
```

The paper calls this offline on-policy distillation because the fixed
completions come from the SFT reference rather than the teacher. They are
on-policy for the reference at collection time but become stale as training
changes the student.

The paper's teacher-consistency condition requires the teacher used to create
the SFT demonstrations to equal the teacher used for OPD scoring. With
consistency, it derives a shared optimum and bounded gradient discrepancy under
its assumptions. It reports math and code experiments, 4x higher training
efficiency than its standard-OPD system comparison, 69.9% AIME 2024 for an
SFT-initialized Qwen3-8B in 30 GPU hours, and a single 8xH100 Qwen3-30B-A3B run
([Lightning OPD](https://arxiv.org/abs/2604.13010)).

In the Tau2 setup, DeepSeek generated the initial demonstrations and would also
provide likelihood scores, so teacher identity can be consistent. The paper's
published evidence is from single-response math/code rollouts, not multi-turn
Tau2 replay states.

### 8.5 Lightning OPD 2.0

Lightning OPD 2.0 addresses cross-teacher settings. It decomposes recurring
teacher/reference disagreement into a proxy for wording, formatting, and
reasoning-cadence style, estimates that component with rollout-level
cross-fitting, and subtracts it before the token-level OPD update.

The paper reports gains over Lightning OPD in cross-teacher math and code
experiments, including 82.4% on AIME 2024 and 63.0% on LiveCodeBench v5 from
Klear-Reasoner-8B-SFT. As of its 30 July 2026 v1 paper, the abstract states that
code will be released; this document does not assume an available reference
implementation
([Lightning OPD 2.0](https://arxiv.org/abs/2607.28449)).

When the same DeepSeek teacher generates SFT data and supplies later scores,
the cross-teacher problem targeted by 2.0 is absent by construction. Other
style or tokenizer effects may still exist, but they are not automatically the
same quantity as the paper's cross-teacher recurring disagreement.

### 8.6 MiniLLM, speculative KD, DAgger/intervention, and selective OPD

These methods were discussed as adjacent trace-using alternatives:

- **MiniLLM** optimizes reverse KL using student samples and a policy-gradient
  formulation. It reports instruction-following distillation from 120M to 13B
  models ([MiniLLM](https://arxiv.org/abs/2306.08543)).
- **Speculative Knowledge Distillation** interleaves student proposals with
  teacher replacements when proposals fall outside the teacher's accepted
  region. It requires online teacher involvement and typically aligned token
  comparisons. It was evaluated on translation, summarization, math, and
  instruction following
  ([Speculative KD](https://openreview.net/forum?id=EgJhwYR2tB)).
- **DAgger/teacher intervention** executes student trajectories, queries the
  teacher on student-visited states, and aggregates corrected demonstrations.
  Text targets make it tokenizer-independent; live Tau2 interaction and
  repeated teacher generation are required.
- **Selective or confidence-weighted OPD**, including SAGE-OPD, attempts to
  train only where the teacher signal is reliable or informative. SAGE-OPD
  executes multi-turn student interaction, observes environment feedback, asks
  the teacher whether to skip or intervene at each turn, weights token-level
  distillation by teacher confidence, and normalizes the loss scale. It reports
  up to 13.3% relative improvement over standard OPD on ALFWorld unseen success;
  it is verifier-free but requires live environment feedback and teacher
  judgment ([SAGE-OPD](https://arxiv.org/abs/2606.19659)).
- **On-Policy Context Distillation (OPCD)** trains on student trajectories
  against a teacher supplied with privileged context. Its applications include
  internalizing historical solution traces and optimized system prompts; the
  paper reports math, text-game, and domain-task experiments. Using it on Tau2
  would require specifying what additional trace/context the teacher sees that
  the deployed student does not
  ([OPCD](https://arxiv.org/abs/2602.12275)).

## 9. Black-box adversarial distillation: GAD

Generative Adversarial Distillation treats Qwen as a generator and trains a
discriminator to distinguish teacher responses from current student responses.
The discriminator becomes an adaptive reward model, and student/discriminator
training forms a minimax loop.

```text
DeepSeek text + current Qwen text -> discriminator update
current Qwen text -> discriminator reward -> policy update
repeat
```

GAD needs teacher text but not teacher logits, so tokenizer differences do not
block it. It does require:

- repeated current-student generation;
- repeated discriminator training;
- policy optimization against a changing learned reward;
- balance between generator and discriminator updates; and
- controls for discriminator overfitting or reward exploitation.

The paper evaluates black-box instruction/chat distillation and reports that
GAD consistently exceeds sequence-level KD; its Qwen2.5-14B-Instruct student is
reported as comparable to GPT-5-Chat on LMSYS-Chat automatic evaluation. This
is a chat-quality result, not a Tau2 result
([GAD](https://arxiv.org/abs/2511.10643)).

For Tau2 there are two separable discriminator targets:

1. **teacher-likeness**: distinguish DeepSeek from Qwen responses;
2. **task desirability**: distinguish successful/policy-compliant from
   unsuccessful/noncompliant actions or trajectories.

The first reproduces the GAD distillation concept. The second changes the
reward-model target and requires Tau2 outcome or auditor labels.

## 10. Outcome reinforcement learning: GRPO

GRPO is a PPO-family policy-gradient method that removes the learned critic and
normalizes rewards within a group of completions sampled for the same prompt.
DeepSeekMath introduced it for mathematical reasoning
([DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300)).

For group samples `y_1,...,y_G` with rewards `r_i`, a simplified group-relative
advantage is:

$$
A_i=\frac{r_i-\operatorname{mean}(r_{1:G})}
{\operatorname{std}(r_{1:G})+\epsilon}.
$$

A clipped likelihood-ratio policy loss then raises actions with positive group
advantage and lowers actions with negative advantage. Implementations may add
KL regularization and token- or sequence-level variants.

### 10.1 Full Tau2 GRPO

```text
same Tau2 task
    -> G current-Qwen live trajectories
    -> official outcome/policy reward for each
    -> group-relative policy update
```

This requires repeated live Tau2 conversations and environment state. It uses
the benchmark's actual outcome signal but that signal is sparse and delayed.
If every rollout in a group receives the same reward, the normalized group
advantage is zero or uninformative. Reward composition also determines what is
optimized: official DB/communication success, policy gates, or a registered
combination.

### 10.2 GRPO with teacher or auditor rewards

GRPO can also consume a scalar teacher judge, discriminator, or auditor score.
That changes the reward source but not the group-relative estimator. Such a run
tests the quality and hackability of the supplied reward in addition to GRPO.

### 10.3 Relation to distillation

OPD receives dense teacher distribution information about sampled student
tokens. GRPO receives scalar relative outcome information about sampled
trajectories. They can be blended as separate loss terms, but a blend introduces
an additional coefficient and no longer isolates either signal.

## 11. “CTO” terminology note

`CTO` was named in discussion without a paper title or expansion. There is no
single standard language-model post-training objective unambiguously denoted
CTO. Recent papers use the acronym for unrelated methods, including code
translation preference optimization and collaborative tree optimization.

Therefore this document does not assign an invented CTO algorithm to the Tau2
experiment. If `CTO` meant **chain-of-thought distillation**, it is covered by
teacher SFT/SeqKD: teacher reasoning and actions are serialized as target text
and trained with NLL. If it refers to a specific paper, its full title or URL
must be added before comparison.

## 12. Relative resource profile

The following is a structural cost comparison, not a wall-clock prediction.
Actual time depends on context length, number of Tau2 turns, batch size,
hardware, teacher endpoint latency, and labeler implementation.

| Method | Student generation | Teacher generation | Teacher scoring | Tau2 environment | Extra trainable model | Structural cost drivers |
|---|---:|---:|---:|---:|---:|---|
| teacher SFT | none during training | already collected | none | none | none | Qwen forward/backward |
| continued SFT | none | already collected | none | none | none | Qwen forward/backward |
| filtered SFT, static | once | optional | none | optional | no | sampling + labeling + NLL |
| DPO/CPO, static | once for negatives | already collected | reference only for DPO | none | no | two completions per record; reference pass for DPO |
| KTO, static | once | no new teacher text | reference likelihood | none | no | labeling + policy/reference passes + KL estimate |
| KTO, refreshed | each refresh | no new teacher text | reference likelihood | none unless labels need outcomes | no | repeated sampling and labeling |
| SODA | one `A0` snapshot | already collected | warmed-reference likelihood | none | no | SFT + one DPO stage |
| GAD | repeated | repeated/fixed teacher corpus | none | none | discriminator | adversarial alternation + policy optimization |
| live OPD | every update | none if scoring only | every update | full live trajectories for agentic OPD | no | student rollout + teacher server + environment |
| GOLD | every update | none if scoring only | every update | none or live, depending on rollout source | no | full/partial logits + cross-vocabulary alignment |
| cross-tokenizer ReOPD | every update/batch | none | every sampled action | none | no | scoring latency + alignment + Qwen update |
| Lightning OPD | one reference rollout set | none | one preprocessing pass | none | no | cached likelihood storage + offline update |
| Lightning OPD 2.0 | one reference rollout set | none | one preprocessing pass | none | residual estimator | cross-fitting + cached offline update |
| full GRPO | groups every update | none | none | groups of full live episodes | no critic | rollout/environment throughput |

Paper-reported speedups are relative to each paper's hardware, implementation,
sequence lengths, and baselines. They should be cited as published results, not
converted into Tau2 hours without a local canary.

## 13. What each two-arm comparison would identify

No row below is a selection. Each row states the narrow causal question a
controlled version could answer.

| Arm 1 | Arm 2 | Question isolated when data/compute are matched | Main confound to control |
|---|---|---|---|
| `A_warm30` | continued SFT | does more expert imitation help? | update/token budget |
| continued SFT | ReOPD | do current-action teacher likelihoods add value over recorded targets? | tokenizer alignment and teacher-score cost |
| continued SFT | Lightning OPD | do cached reference-policy teacher scores add value over more NLL? | rollout set and objective budget |
| ReOPD | Lightning OPD | does continual action refresh matter relative to cached actions? | teacher-query and generation budget |
| DPO | CPO | does the chosen-response NLL/reference-free construction change results on the same pairs? | loss scaling and reference compute |
| SODA | CPO on SODA pairs | warmed-reference DPO versus reference-free contrastive plus NLL | coefficient/step matching |
| filtered SFT | static KTO | do retained negatives add information on a fixed labeled dataset? | shared positives and exact examples |
| refreshed filtered SFT | refreshed KTO | which complete iterative algorithm performs better? | branch-specific post-refresh datasets |
| SODA | refreshed KTO | fixed provenance-based pairs versus refreshed binary labels | label source and freshness both change |
| SODA | GAD | static pairwise black-box distillation versus adversarial on-policy black-box distillation | substantially different compute |
| GAD | ReOPD | learned teacher-likeness reward versus direct teacher likelihood | teacher access differs |
| ReOPD | GRPO | dense teacher matching at replay states versus sparse live outcome optimization | environment and signal density |
| continued SFT | GRPO | offline expert imitation versus live outcome learning | rollout cost and data distribution |
| Lightning OPD | Lightning OPD 2.0 | value of style residualization in a cross-teacher setting | requires genuine teacher mismatch |

## 14. Tau2 label and credit-assignment matrix

| Label source | Can certify isolated action? | Can certify final task? | Recognizes alternative valid paths? | Typical use |
|---|---:|---:|---:|---|
| DeepSeek provenance | no | teacher trace outcome only | not by itself | SODA pair assumption |
| teacher likelihood | preference density, not correctness certificate | no | can assign likelihood to alternatives | OPD/ReOPD/Lightning |
| deterministic policy audit | for encoded local rules | only included gates | only if rules permit them | filter/KTO/reward component |
| pairwise LLM judge | produces ordering | depends on supplied context | judge-dependent | DPO/CPO |
| Tau2 official reward | no isolated counterfactual action | yes for completed trajectory | yes, through equivalent end state | filtering/GRPO/evaluation |
| human adjudication | potentially | potentially | potentially | auditor calibration/small data |
| GAD discriminator | teacher-likeness, not task correctness | not unless trained on outcomes | mirrors its training target | GAD reward |

For a four-label auditor, report per-class counts, agreement with blinded human
adjudication, precision/recall for retained binary classes, and exclusions.
Class prevalence alone does not measure label validity.

## 15. Train/selection/test use by method

The same outer task boundary applies to every option:

| Partition | Allowed |
|---|---|
| `pilot-train30` | SFT targets, student sampling, pair/label construction, replay prefixes, teacher scoring, live RL if later authorized |
| retail selection | live evaluation and preregistered recipe/checkpoint decisions; no optimization examples |
| retail final test | one frozen final evaluation; no tuning |

For refreshed or online methods, all fresh actions and trajectories must still
come from training tasks. A new action sampled on a selection/test prefix is
training leakage if it influences weights, labels, hyperparameters, or
checkpoint selection beyond the registered evaluation rule.

The future continual-learning extension retains domain boundaries:

```text
retail adaptation -> evaluate retail / telecom / airline
telecom adaptation -> evaluate retail / telecom / airline
```

No telecom optimization is part of the initial 30-retail-task SFT pilot.

## 16. Measurements required before choosing a continuation

These are measurements, not method-selection thresholds:

1. `A0` and `A_warm30` Tau2 success, DB, communication, policy compliance, tool
   validity, and gradeability on the frozen selection panel.
2. Exact eligible trajectories, assistant decisions, tokens, and prefix lengths
   from `pilot-train30`.
3. DeepSeek supplied-action scoring availability, latency, token accounting,
   and score reproducibility.
4. Cross-tokenizer alignment coverage, chunk lengths, invalid mappings,
   likelihood agreement probes, and finite/nonzero gradient canary.
5. Student sampling throughput at stored prefixes.
6. If binary objectives remain options: auditor class distribution and blinded
   precision on arbitrary student actions, including valid alternatives.
7. If pairwise objectives remain options: audited fraction of DeepSeek/Qwen
   pairs in which the stated ordering holds and fraction in which both pass.
8. If GRPO remains an option: live episode throughput and distribution of
   within-task group rewards, including all-equal groups.
9. If GAD remains an option: discriminator architecture, update ratio,
   held-out discrimination calibration, and reward-hacking diagnostics.
10. If Lightning OPD remains an option: cached rollout/storage size, teacher
    consistency status, and policy drift from the rollout-generating reference.

## 17. Evidence boundary

As of 2026-08-24:

- SFT has direct precedent for agent traces, but no paper establishes that 30
  Tau2 tasks are sufficient for Qwen3-4B.
- ReOPD has multi-turn Python-tool and search evidence, but no published Tau2
  result in the cited paper.
- Cross-tokenizer OPD has cross-family benchmark evidence, but the combination
  with multi-turn Tau2 replay is not established by the cited paper.
- SODA and GAD report general LLM/chat distillation results, not Tau2 results.
- CPO reports machine-translation results, not Tau2 results.
- KTO reports general alignment results, not multi-turn Tau2 results.
- Lightning OPD and 2.0 report math/code results, not Tau2 results.
- GRPO has extensive reasoning/RL use, but a Tau2 run depends on live rollout
  throughput and useful reward variation.

Cross-paper headline scores and speedups are not directly comparable because
the models, teachers, data, objectives, hardware, and benchmarks differ.

## 18. Primary sources

- [Tau2 paper](https://arxiv.org/abs/2506.07982)
- [Tau2 evaluation semantics](https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Contrastive Preference Optimization / ALMA-R](https://proceedings.mlr.press/v235/xu24t.html)
- [KTO](https://arxiv.org/abs/2402.01306)
- [SODA](https://arxiv.org/abs/2604.03873)
- [GAD](https://arxiv.org/abs/2511.10643)
- [GKD](https://arxiv.org/abs/2306.13649)
- [MiniLLM](https://arxiv.org/abs/2306.08543)
- [ReOPD](https://arxiv.org/abs/2607.04763)
- [Cross-tokenizer OPD](https://arxiv.org/abs/2606.09456)
- [GOLD](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation)
- [Lightning OPD](https://arxiv.org/abs/2604.13010)
- [Lightning OPD 2.0](https://arxiv.org/abs/2607.28449)
- [SAGE-OPD](https://arxiv.org/abs/2606.19659)
- [On-Policy Context Distillation](https://arxiv.org/abs/2602.12275)
- [Speculative Knowledge Distillation](https://openreview.net/forum?id=EgJhwYR2tB)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [TRL KTOTrainer](https://huggingface.co/docs/trl/kto_trainer)
- [TRL CPOTrainer](https://huggingface.co/docs/trl/cpo_trainer)

## 19. Parked status

Creating this catalogue does not authorize a GPU run, teacher query, Tau2 live
rollout, or selection of a continuation method. The independent next artifact
is the frozen `pilot-train30` SFT manifest and data census. Later method research
can update this extension without changing the already frozen SFT data boundary.
