# CLEA experiment: continual capability transfer into Qwen3-4B

Status: preregistration draft  
Date: 2026-08-24  
Target: CLEA @ NeurIPS 2026

## Decision

Use the exact post-trained `Qwen3-4B` checkpoint from the audited tau2 runs as
the student. Do not substitute a pretrained-only `Qwen3-4B-Base` checkpoint.
Use DeepSeek-V4-Flash-0731 as the frozen teacher.

Qwen3-8B is not a primary arm. It may be used for one reduced replication only
after the complete 4B result exists.

## Research question

> After a shared SFT warm start, does replay-prefix on-policy distillation
> transfer DeepSeek's decision policy to Qwen3-4B better than compute-matched
> continued SFT, while causing less forgetting during sequential enterprise
> agent adaptation?

## Hypotheses

### H1: held-out capability acquisition

On unseen retail tasks, `SFT -> ReOPD` improves policy-compliant task success
over both frozen Qwen3-4B and the shared SFT warmup checkpoint.

### H2: ReOPD versus continued SFT

Given matched post-warmup update tokens, `SFT -> ReOPD` achieves higher
policy-compliant task success or equal success with lower KL drift than
`SFT -> continued SFT`.

### H3: continual-learning retention

After subsequent telecom adaptation, the ReOPD branch loses less held-out
retail capability than the continued-SFT branch.

## Experimental arms

All trained arms share one warmup checkpoint. This prevents warmup variance
from contaminating the comparison.

```text
A0  frozen Qwen3-4B
       |
       +-- SFT warmup ---------- A_warm
                                  |
                                  +-- continued SFT ---------- A_sft
                                  |
                                  +-- replay-prefix OPD ------- A_reopd
```

Required reported checkpoints:

| Arm | Meaning |
|---|---|
| `A0` | frozen post-trained Qwen3-4B |
| `A_warm` | shared SFT warmup |
| `A_sft` | warmup followed by continued SFT |
| `A_reopd` | warmup followed by replay-prefix OPD |

Optional ablation, only after the required arms: ReOPD directly from `A0`.

## Compute matching

`A_sft` and `A_reopd` must match on:

- post-warmup optimizer-update count;
- supervised student-token count, within 5%;
- LoRA rank, target modules, optimizer, scheduler, precision, and context cap;
- training task IDs;
- checkpoint-selection rule;
- evaluation tasks, seeds, user simulator, and generation settings.

Report teacher collection tokens and online teacher-scoring tokens separately.
Also report GPU-seconds, wall time, and API cost. No claim of efficiency is
allowed from parameter count alone.

## Data boundary

Pinned tau2 revision:
`f8de30c298689cbe0117d76a378e7315a17e5bd8`.

Retail contains 114 tasks. Freeze a family-aware split:

| Split | Count | Permitted use |
|---|---:|---|
| train | 60 | DeepSeek collection, SFT, replay prefixes, teacher scoring |
| development | 18 | checkpoint selection and declared hyperparameter choices |
| test | 36 | final live evaluation only |

Task 57 and task 93 must be in `test` as named diagnostics. Their historical
trajectories motivate the experiment but must not be used for training,
checkpoint selection, prompt editing, or hyperparameter tuning after the split
is frozen.

Split related or near-duplicate workflows as groups, not independently. At
minimum group by required mutation/tool family and normalized task instruction.
The emitted manifest must include task IDs, group IDs, seed, source hash, and
split hash.

Hard invariant:

```text
teacher collection IDs ∪ SFT IDs ∪ ReOPD prefix IDs ⊆ train IDs
development IDs ∩ training IDs = ∅
test IDs ∩ (training IDs ∪ development IDs) = ∅
```

Final test tasks must never be sent to DeepSeek for trajectory generation or
token scoring.

## Training signal definitions

### SFT warmup

Train on successful DeepSeek actions from retail-train trajectories. Its job is
to establish protocol validity, observation reading, authentication, and basic
tool workflow competence.

### Continued-SFT control

Continue training on held-back successful DeepSeek actions from the same
retail-train boundary. Do not reuse exact warmup examples in the continuation
unless reuse is identically applied to the ReOPD arm.

### Replay-prefix OPD

At saved DeepSeek trajectory prefixes from retail-train tasks:

1. render the exact frozen prefix;
2. sample one action from the current Qwen policy;
3. retain exact sampled bytes and student token IDs;
4. score that action with frozen DeepSeek;
5. apply the registered cross-tokenizer chunk-OPD loss;
6. update only the Qwen LoRA parameters.

This is on-policy for the evaluated action and offline for the preceding state.
It is not live multi-turn OPD and must not be described as such.

## Pilot

The pilot validates the pipeline, not the paper claim.

- 12 retail-train tasks;
- 4 development tasks;
- 12 untouched pilot-test tasks;
- 2 trials per evaluation task;
- one shared SFT warmup;
- 32 post-warmup optimizer updates per continuation arm;
- 16 distinct replay states per OPD update when memory permits;
- fixed seeds and temperature settings across arms.

Advance only if all of the following hold:

1. every evaluation trajectory is gradeable;
2. no train/development/test leakage is detected;
3. ReOPD teacher scores, alignments, losses, and gradients are finite/nonzero;
4. `A_reopd` beats `A_warm` on policy-compliant pilot-test success;
5. `A_reopd` is competitive with `A_sft` rather than clearly worse;
6. tool-call validity and policy compliance do not regress materially.

The pilot test IDs are consumed by this decision and cannot become the final
paper test set if results are used to change the method.

## Final retail evaluation

Evaluate `A0`, `A_warm`, `A_sft`, and `A_reopd` live on all 36 test tasks with
4 fixed trials per task: 144 trajectories per checkpoint.

Primary metric:

> Policy-compliant task success: tau2 DB success and all applicable hard policy
> gates pass.

Report official tau2 reward separately; do not replace or silently modify it.

Hard policy gates:

- authentication occurred before private lookup or mutation;
- complete action details were presented;
- explicit user confirmation followed those details;
- every user condition was resolved before mutation;
- tool name and arguments were correct;
- no database mutation occurred after a failed precondition.

Secondary metrics:

- official tau2 success;
- task-family success;
- task 57 and 93 diagnostic outcomes;
- tool-call validity and tool errors;
- unsupported operational claims;
- unnecessary transfer rate;
- turns and tool calls per episode;
- KL from `A0` on a frozen evaluation-prefix set;
- GPU time, teacher tokens, and cost per gained success point.

Use task-level bootstrap confidence intervals. Multiple trials from one task
are not independent tasks.

## Continual-learning extension

Run only after the retail result passes the pilot gate.

```text
T0: evaluate A_sft and A_reopd on retail-test, telecom-test, airline-test
T1: adapt each branch on retail-train; evaluate all three test sets
T2: continue each branch on telecom-train; evaluate all three test sets
```

Interpretation:

| Cell | Meaning |
|---|---|
| retail T0 -> T1 | retail capability acquisition |
| telecom/airline T0 -> T1 | zero-shot cross-domain transfer or interference |
| telecom T1 -> T2 | new-domain acquisition |
| retail T1 -> T2 | retention/forgetting |
| airline through T0/T1/T2 | untouched-domain drift |

Report average accuracy, backward transfer, forward transfer, and per-domain
forgetting. A cross-domain claim requires untouched domain test tasks; success
on retail alone is in-domain generalization, not cross-domain transfer.

## Falsification and stopping rules

The method is not supported if any of these occur:

- ReOPD fails to beat the shared warmup on final held-out success;
- ReOPD loses clearly to compute-matched continued SFT without a retention or
  cost advantage;
- apparent gains disappear under policy-compliant scoring;
- gains require test-task teacher access or test-driven tuning;
- ReOPD alignment failures, truncation, or empty samples account for the
  measured difference;
- forgetting is claimed from fewer than two sequential adaptation stages.

Null results must be reported rather than repaired through unregistered test
tuning.

## Immediate execution order

1. Implement and freeze the family-aware split manifest.
2. Add code-enforced task-ID leakage checks to collection, SFT export, and OPD.
3. Implement the deterministic policy-compliance trajectory auditor.
4. Freeze pilot IDs, seeds, budgets, and LoRA configuration.
5. Collect DeepSeek trajectories on pilot-train only.
6. Produce the shared SFT warmup checkpoint.
7. Branch into matched continued-SFT and ReOPD continuations.
8. Evaluate the frozen pilot arms.
9. Publish the pilot gate decision before scaling.
10. Run the final 60/18/36 retail experiment.
11. If successful, run the retail-to-telecom continual-learning extension.
12. Replicate the main retail comparison on Qwen3-8B only if time and budget
    remain.
