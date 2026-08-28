# Full ReOPD replay run plan

## Decision

Run a real replay-prefix experiment from the untouched `v0` adapter, with
**200 optimizer updates as the target horizon**.  The completed 32-action,
one-update run is a plumbing canary: it proves that sampling, DeepSeek scoring,
alignment, one update, and adapter reload work together.  It is not evidence of
learning and is not counted as the full experiment.

This remains an offline replay experiment.  It does not query Harbor/the
environment after a student action, and it does not claim to be live multi-turn
RL.  It answers the narrower, useful question: whether dense teacher likelihood
supervision at stored trace states improves `v0`.

## What ReOPD actually does

ReOPD starts with a fixed pool of expert trajectories.  For a selected position
`t`, it reconstructs the **teacher** history up through that position (previous
teacher actions and observations), has the student generate **one** action at
that state, and has the teacher score that exact action with teacher forcing.
There is no environment query and the next selected position is not a
continuation of the student action.

```text
stored teacher prefix at position t
  -> current student samples one action
  -> teacher scores that action
  -> student update
  -> reset to another stored teacher prefix
```

The paper creates a pool over all valid trajectory positions and samples a
position with probability proportional to `kappa^t`; its reported default is
`kappa = 0.6`.  Thus early states are deliberately favored, but later states
remain eligible.  It refreshes the student after optimization and repeats.
The implementation description uses one student turn and zero tool calls for
ReOPD; this is exactly why it is cheaper than full online OPD.  Its reference
run is a large, optimized stack (SLIME, SGLang, Megatron; eight H100s), so its
throughput and wall-clock results must not be projected onto our single-L40S /
single-A100 setup.

Sources: [ReOPD paper](https://arxiv.org/html/2607.04763v3) and its
[reference implementation](https://github.com/BaohaoLiao/ReOPD).

### What “every step” means here

It means every valid stored position is eligible in the sampling pool across
the run.  It does **not** mean advancing the environment after each student
action, nor does it require putting every position in every optimizer batch.
Sampling is the practical unbiased implementation of the paper's weighted
objective.

With `kappa=.6`, a position is 1.67x as likely as the immediately following
position.  Position 1 is about 99x as likely as position 10 (`.6^-9`), and the
first ten positions receive 99.4% of the mass of an unbounded geometric
schedule.  Therefore paper-faithful `.6` sampling is **not** a literal
"train every stored step" schedule.  It is a valid, early-step-heavy ReOPD
condition, and its sampled-step histogram must be logged.

### A trajectory step is not an optimizer step

The paper does not update the model after every action in a particular stored
trajectory.  It samples many independent replay positions, scores their
student actions, then performs an optimizer update on the batch.  One update
changes the student from `v_t` to `v_(t+1)`; it changes the action distribution
at every replay state, but it does **not** change the stored observation or
create a new next state.  In a strict run, the next batch must therefore be
sampled by `v_(t+1)`.  In the one-version async variant it may instead have
been sampled by `v_t`, which is exactly the controlled staleness trade-off
described below.

## Split and scheduler: freeze these before launch

No full-run ReOPD train/test split has been created yet.  The canary deliberately
used eight diagnostic prefixes and is neither a train split nor a test split.
Do not call an evaluation held out until it is frozen before the first paid
training action.

The recommended split unit is a **task**, never an individual trajectory or
prefix.  Otherwise another successful trace from the same Harbor task leaks
the solution structure into training.

| Partition | Proposed contents | Purpose |
| --- | --- | --- |
| Training tasks | 27 of the 34 tasks with passing teacher trajectories | source replay pool used for updates |
| Same-distribution holdout | 7 of those 34 tasks, all their trajectories excluded | clean task-level generalization test |
| Acquisition panel | 4 pre-named training tasks, evaluated but not called generalization | does the model learn the source domain? |
| External transfer/control panel | frozen tasks outside the 34-task teacher-pass pool, including a teacher-failed negative control | transfer and regression checks, never training data |

Choose the seven task IDs by a deterministic, repository-stratified seed and
write the IDs, trace counts, and corpus hash to the manifest.  Measure `v0` on
every evaluation task before training.  The older four-task OPD smoke roster is
useful as a low-cost panel, but it is not by itself an adequate 200-update
result.

There are two legitimate scheduler choices; they answer different questions:

1. **Coverage-first primary (recommended if the aim is literally every step).**
   Shuffle all valid positions from the 27 training tasks without replacement;
   take 32 distinct positions per update; start a new epoch only after every
   eligible position has appeared once.  With the current full corpus's 4,271
   fitting positions, one 32-position epoch would be at least 134 updates;
   the exact smaller training-pool count is frozen after the task split.  This
   is broader than the paper schedule and must be labeled as such.
2. **Paper-faithful ReOPD ablation.** Sample positions independently with
   `p(t) ∝ .6^t`, record realized step mass, and accept that almost all samples
   are early states.  It tests the paper's reliability weighting, not coverage
   of all steps.

Do not call 32 items “32 tasks” or “32 trajectories.”  An update has **32
state positions**, usually from many traces and tasks, with one student action
per position.  It is intentionally not a fixed set of 32 trajectories replayed
step-by-step.

## Frozen experimental recipe

| Item | Full-run choice |
| --- | --- |
| Parent policy | untouched ck75 `v0` |
| Replay corpus | freeze the task split, valid candidate snapshot, source trace IDs, tokenizer/rendering pins, and corpus hash before update 1 |
| Position distribution | choose and label either coverage-first epochs or paper-faithful `p(t) ∝ .6^t`; no hand-picked diagnostic strata |
| Diversity guard | cap a trace and a task's share per update; resample and record rejections rather than silently changing weights |
| Student action | one sampled action, stored with policy version and per-token behavior log-probabilities |
| Teacher | DeepSeek V4 Flash teacher-forced score of identical action bytes in native teacher history |
| Loss | existing cross-tokenizer aligned, clipped sampled reverse-KL loss |
| Batch budget | 32 student actions/update initially; pack to a fixed action/context-token ceiling and reduce count only on OOM |
| Horizon | 200 updates; checkpoints and held-out decisions at 0, 25, 50, 100, 150, 200 |
| Primary comparison | `v0` vs each frozen checkpoint on the same held-out replay states and fixed pass@k seed set |

The source snapshot currently has 4,271 context-fitting candidates from 117
passing traces.  That is a starting snapshot, not a promise that every record
will survive the final freeze; the manifest records the exact number.

## Three concrete batch improvements

1. **Make the default batch 32 positions × 1 action.** This is the recommended
   full-run shape.  It maximizes state coverage and makes “all steps eligible”
   meaningful.  The canary's 8 positions × 4 actions produced 314,304 repeated
   prefix tokens: 71.1% of its teacher input.  One action per selected state
   normally trades that repeated context for much broader supervision.

2. **Run a small, pre-registered composition ablation: 16 positions × 2
   actions.** It retains some within-state action variance, while doubling
   state breadth over the canary.  Compare it against 32×1 at the same 32
   actions and token budget using held-out likelihood/alignment diagnostics;
   choose one composition before the 200-update main run.  Do not mix shapes
   mid-run.

3. **Token-budgeted, full-pool sampling.** Estimate rendered-prefix and action
   length before dispatch, reject over-cap contexts, and fill batches to a
   fixed maximum token budget rather than blindly to a request count.  Keep
   task/trace caps and log both the intended `.6^t` distribution and accepted
   distribution.  This improves A100 utilization and prevents a few long,
   repeated traces from consuming the run.

The third item is not a license to change the objective: selection remains
ReOPD's position sampling.  If it materially distorts the intended step
distribution, it is a different experimental condition and must be labeled as
such.

## End-to-end controller

Use one durable run directory, for example `runs/reopd-full-<run-id>/`, with
an append-only manifest and one atomic completion marker per stage.  A restart
must resume from those markers, never infer completion from a terminal log.

### Before paid work

1. Freeze the corpus and write a manifest containing git revision, base/adaptor
   hashes, tokenizer and chat-template hashes, `kappa`, seed, batch shape,
   token caps, optimizer/loss parameters, and pricing assumptions.
2. Produce a deterministic sampled-position preview for updates 1–200, then
   apply the documented trace/task caps.  Save IDs and probabilities; do not
   re-select them interactively.
3. Add request-level accounting: request ID, dispatch/start/end timestamps,
   input/output/cached token usage returned by Fireworks, source state ID,
   action bytes hash, and score checksum.  The canary cannot yield an exact
   API invoice because these fields were not persisted.
4. Verify that the trainer saves and reloads **model, optimizer, scheduler,
   RNG, and update number**.  The one-step script starts fresh; it is not by
   itself a 200-update trainer.
5. Define budget and automatic circuit breakers: authentication failure,
   score/alignment corruption, ratio/clip anomaly, OOM, missing version, or
   evaluation regression all stop new paid dispatch and preserve artifacts.

### Per-update durable state machine

```text
PLANNED(v_t) -> SAMPLED(v_t) -> SCORED(v_t) -> TRAINED(v_t -> v_t+1)
              |                    |                 |
          actions+logprobs     scores+usage      adapter+optimizer
```

For every transition, write outputs to a temporary directory, validate counts
and hashes, then atomically mark the stage complete.  Repeating a controller
after a crash must reuse already-paid scores and never send them again.

The required checks are: action bytes match the scored bytes; every score
aligns to the student action; behavior policy version equals the declared
version; no duplicate request IDs; finite loss/gradients; checkpoint reload
matches the saved adapter; and update `n+1` cannot begin training from a
partially written `n`.

### Evaluation and decisions

Run cheap training health checks after every update: score count, token use,
alignment coverage, teacher likelihood, policy-ratio/clip fraction, effective
sample size, loss, grad norm, and batch step histogram.  Run fixed evaluation
at 0/25/50/100/150/200 with no selection changes: held-out replay scoring and
the same seeded pass@k suite for `v0` and each checkpoint.

The 25/50/100 checkpoints are decision gates, not a claim that those are the
whole experiment.  Continue toward 200 if artifacts are valid and the
pre-declared safety/regression rules allow it.  Report the full learning curve,
not only the best checkpoint.

## Stop/restart versus continuous service

These comparisons use only measured automated work from the canary, not the
human coding/debugging gaps:

| Measured component | Time |
| --- | ---: |
| L40S serving startup | 3m55s |
| 32-action sampling after stage start | 17m47s |
| DeepSeek scoring stage | 12m57s |
| A100 application total | 9m38s |
| Score + train interval | 22m35s |

At the recorded L40S price of $1.95/hour, startup costs about **$0.127**.  If
the pipeline is strictly sequential, keeping an L40S alive during the measured
score+train interval costs about **$0.734**; stopping it and booting for the
next sampling batch costs about **$0.127** and adds 3m55s.  Therefore one such
boundary saves roughly **$0.607** but delays the next batch by 3m55s.  Across
199 boundaries of a 200-update strict run, that mechanical extrapolation is
about **$121 saved** for about **13h of extra wall time**.  It excludes actual
cloud allocation overhead, evaluation, and any change in batch lengths.

### The time-first alternative: one-version asynchronous look-ahead

While DeepSeek scores and the A100 trains batch `t`, leave the L40S serving
`v_t` and sample batch `t+1`.  The measured sampling time (17m47s) fits within
the measured score+train interval (22m35s), leaving only about 4m48s of L40S
idle *if those timings recur*.  DeepSeek can score a completed next batch
immediately; it never needs to wait for the optimizer to score action bytes.

That is **not strictly on-policy**: once `v_(t+1)` exists, the queued actions
were sampled by `v_t`.  Permit at most one version of staleness, retain
behavior log-probabilities, tag every artifact with its policy hash, apply the
existing importance-ratio correction, and monitor clip fraction and ESS.  A
staleness breach, missing behavior log-probs, or a bad ratio distribution must
discard the queued batch rather than silently train on it.

Start the main experiment synchronous.  Introduce this look-ahead only as a
separately labeled async condition after a short equivalence test against the
synchronous controller; otherwise a throughput optimization changes the
scientific result at the same time as the batch recipe.

### Libraries to use rather than rebuild

- [verl fully asynchronous training](https://github.com/volcengine/verl/blob/main/docs/advance/fully_async.md)
  provides queue-based rollout/training separation, parameter synchronization,
  and staleness controls.  It is the strongest candidate if we build the async
  condition, but custom Fireworks scoring and cross-tokenizer alignment still
  need an integration layer.
- [NVIDIA NeMo RL asynchronous GRPO](https://docs.nvidia.com/nemo/rl/nightly/guides/single-controller.html)
  has an explicit rollout queue and importance-sampling handling for stale
  policy data.  It is useful as a reference/controller option, not a drop-in
  replacement for this custom likelihood loss.
- [Miles](https://github.com/radixark/miles) is an experimental fully async RL
  system with SGLang, Ray/Megatron integration and a maximum weight-staleness
  control.  Use it only for a later sustained async/live-RL migration.  Moving
  the current experiment into Miles before obtaining a synchronous ReOPD curve
  would add a large integration variable.
- The [ReOPD codebase](https://github.com/BaohaoLiao/ReOPD) itself is the
  closest recipe reference (SLIME/SGLang/Megatron), but its infrastructure is
  tailored to its homogeneous multi-GPU setup rather than our Fireworks teacher
  plus cross-tokenizer scoring path.

## Recommended execution order

1. Implement and review the durable 200-update controller, full-pool sampler,
   request accounting, persistent trainer state, and automatic cleanup.
2. Freeze the exact main configuration: `kappa=.6`, 32×1 or the winner of the
   pre-registered 32×1 versus 16×2 comparison, fixed token cap, and evaluation
   seeds.
3. Run the synchronous 200-update primary condition from `v0`, automatically
   starting only the GPU needed for each completed stage and tearing it down
   after that stage.
4. Evaluate all fixed checkpoints, publish the complete curve and exact billed
   API/GPU ledger.
5. Only then run the one-version-async condition if wall-clock throughput is
   worth the additional off-policy variable; compare it to the synchronous
   primary under the same frozen corpus and evaluation.

This gives us a full, interpretable result first.  It uses every replay step
over the run, preserves a clean `v0` comparison, and treats distributed async
execution as a controlled systems experiment instead of smuggling it into the
learning result.
