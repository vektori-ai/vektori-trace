# τ²-bench ReOPD research plan

## Purpose

This is a separate benchmark-research plan.  It does not replace or modify the
existing Harbor replay plan.  Harbor remains an engineering demonstration; this
plan tests the method on a public, stateful tool-agent benchmark.

## Research question

> On unseen τ²-bench tasks, does replayed-prefix on-policy distillation improve
> a Qwen tool agent more than no adaptation or ordinary teacher-trajectory SFT,
> at lower interaction cost than true online multi-turn OPD?

Use a pinned release of `sierra-research/tau2-bench`, initially in the `telecom`
domain.  The benchmark supplies a simulator, tool policy, task splits, and an
outcome scorer.  Record the repository revision, task JSON hash, split file
hash, model settings, agent rendering, and all seeds.

## Strict data boundary

```text
official τ² train tasks: teacher collection and all training only
official τ² test tasks: live evaluation only
```

Never collect a DeepSeek trajectory, prefix, score, or tune an adapter using a
test task.  The final scientific metric is live official-simulator performance
on `test`.  `base` may be reported separately for benchmark compatibility, but
it is not the unseen-task claim after train-split adaptation.

## Systems roles

| Component | Role |
| --- | --- |
| Qwen3-14B + LoRA | student policy; produces messages/tool calls and behavior log-probs |
| DeepSeek API | expert agent on train tasks and teacher-forced scorer of student actions |
| τ² telecom simulator | executes tool/message actions and returns observations/outcomes |
| user simulator | fixed and pinned across all arms; its LLM/API use is separately logged |
| A100 | updates LoRA parameters only; base Qwen weights remain frozen |

DeepSeek does not require a hosted GPU.  Its expert-collection calls and
teacher-scoring calls must be separate ledger categories.

## Build sequence

1. Pin τ² version and inspect its published `train`, `test`, and `base` task
   IDs.  Freeze a fixed initial task subset for the pilot, including seeds.
2. Implement a Qwen τ² agent adapter: render policy/tools/history, parse one
   valid message or tool action, execute it through τ², and save exact sampled
   action bytes, token IDs, behavior log-probs, and policy version.
3. Implement a DeepSeek τ² expert adapter with the same action semantics.
   Generate train-split trajectories, retaining successful traces and complete
   state/action/observation histories.
4. Convert successful train traces into replay prefixes.  For a prefix, Qwen
   samples exactly one new action; DeepSeek scores those exact bytes; the
   existing cross-tokenizer alignment/loss produces a LoRA update.
5. Run each final policy live through the τ² test simulator and use the
   benchmark's official outcome scorer.  Evaluation never uses teacher scores.

## Experimental arms

Every arm branches from the identical frozen `v0` adapter and receives a
comparable training token/update budget.

| Arm | Training data / rule | What it establishes |
| --- | --- | --- |
| v0 | no adaptation | baseline capability |
| SFT control | teacher actions from successful train trajectories | ordinary offline imitation |
| ReOPD | Qwen-sampled actions at saved teacher prefixes; DeepSeek likelihood supervision | proposed method |
| Online OPD, later | Qwen acts live, τ² returns next state, DeepSeek scores every visited state | costly upper comparison |

The first result needs v0, SFT, and ReOPD.  Online OPD is deliberately deferred:
it adds a live rollout controller and multiplies interaction time.

## Small, falsifiable pilot

Do not start with all tasks or 200 updates.

1. Use a predeclared subset of train task IDs to collect teacher traces.
2. Measure v0 live on a fixed test subset before training.
3. For each training arm, run 10–20 updates of 16 distinct replay states × 1
   student action.  Use a fixed action/context-token budget and one frozen
   sampling policy.
4. Run each final arm on exactly the same test task IDs, seeds, user simulator,
   and trial count.  Use 3–4 trials/task where budget permits.
5. Report success, policy/tool violations, interaction turns, tool calls,
   training wall time, and separately billed student GPU, teacher collection,
   teacher scoring, and user-simulator costs.

Advance to scale only if ReOPD improves the fixed test metric over v0 and is
competitive with SFT.  A null result is a result; do not tune on test tasks.

## What is and is not on-policy

Offline ReOPD:

```text
saved DeepSeek state -> Qwen samples one action -> DeepSeek scores -> update
```

Qwen is on-policy for the supervised action, but not for the replayed history.
No τ² environment action is executed during student training.

True online OPD:

```text
Qwen action -> τ²/user response -> next Qwen state -> DeepSeek score -> update
```

This captures recovery from Qwen-induced states but is much more expensive.

## Time and cost measurement gate

Do not project Harbor timings onto τ².  Before committing to a run, execute and
log eight DeepSeek train episodes plus eight Qwen-v0 test episodes.  Measure:

- agent turns/task and user turns/task;
- student generation seconds/turn;
- DeepSeek collection/scoring token use and latency;
- successful teacher-trace yield;
- prefix and action token-length distributions;
- simulator time and retry/error rate.

The offline ReOPD pilot should be designed as an overnight experiment.  True
online OPD cannot be estimated until this gate: its per-update time is the sum
of live student, user, environment, and teacher work over every trajectory
turn, followed by training.

## Continual-learning extension

For a CLEA-oriented result, partition the τ² train split into ordered cohorts.
After each cohort, evaluate every prior cohort and the untouched test split.

```text
cohort 1 train -> evaluate 1/test
cohort 2 train -> evaluate 1/2/test
cohort 3 train -> evaluate 1/2/3/test
```

Compare current-cohort-only training, cumulative rehearsal, and a bounded
replay-buffer condition.  Report the full accuracy matrix and per-cohort
forgetting: best prior score minus final score.  This is a follow-on experiment,
not a prerequisite for the initial τ² ReOPD result.

## Sources

- [τ²-bench repository](https://github.com/sierra-research/tau2-bench)
- [τ² Gym interface](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/gym/README.md)
- [CLI task split and evaluation documentation](https://github.com/sierra-research/tau2-bench/blob/main/docs/cli-reference.md)
