# Live cross-tokenizer OPD on Tau2 retail

Date: 2026-08-28
Status: **proposal**, not yet the plan of record — see §0
Budget ceiling: $40 provisional, unvalidated until Phase 1 measures it

## 0. Scope, and what this does and does not supersede

**Tau2, not Harbor.** Verified 2026-08-28: the two lines share no artifact.

| | Harbor line | Tau2 line (this file) |
| --- | --- | --- |
| Student | Qwen3-14B `ck75` | Qwen3-4B `a_sft_new_ck35_r2` |
| Benchmark | Terminus / corpus50_v3 | Tau2 retail |
| Artifact | `v_replay` on frozen `v0` | `A_live_opd` on `A_sft_new` |
| Plan of record | `docs/OPD-MULTITURN-PLAN.md` | `docs/CLEA-QWEN3-4B-EXPERIMENT-V2.md` |

`CLAUDE.md` names `docs/OPD-MULTITURN-PLAN.md`
authoritative for OPD and records "replay-prefix only; live multi-turn **Harbor**
OPD is out of scope." That document is the *Harbor* line — Qwen3-14B `ck75`,
Terminus, artifact `v_replay` parented on `v0`. Different student, benchmark and
artifacts. Its exclusion does not govern Tau2, and nothing here changes it.

Within Tau2 the plan of record remains
`docs/CLEA-QWEN3-4B-EXPERIMENT-V2.md` (design) and `docs/TAU2-REOPD-PLAN.md`
(operations for the replay arm). **This file does not supersede either.** It
proposes a *new* arm — live multi-turn — alongside the replay arm those
documents specify.

Promoting this to plan-of-record requires, explicitly and in writing:

1. baseline provenance resolved (§"Reuse or redo the SFT?" gate 1);
2. a preregistered evaluation design (§"Evaluation preregistration");
3. a live driver that exists and has passed Phase 1;
4. an amendment to `CLAUDE.md` naming the Tau2 live arm.

Until all four hold, this is a proposal. No GPU, endpoint or paid teacher call
is authorized by this file.

## Decision

Run a small **live multi-turn OPD pilot** with **Qwen3-4B** as student and
**DeepSeek V4 Flash** as teacher.

Start from the continued-SFT checkpoint `a_sft_new_ck35_r2` only after its
identity and training audit pass. Do not train 8B before this pilot.

The defensible statement of the baseline, pending gate 1:

> An evaluation served under the alias `Qwen3-4B-sft`
> (`tau2_eval_s16_20260826_180059_sft.json`) scored **6/8 reward, 8/8
> gradeable, zero observed loops** on tasks 57/73/75/93. **The underlying
> adapter has not been proven to be `a_sft_new_ck35_r2`** — the progress log
> records only the alias, and CK35 is an equally consistent candidate.

The entire before/after comparison depends on checkpoint identity, so no
number above may be attributed to a named adapter until gate 1 passes.

Because teacher and student use different tokenizers, this is not exact
Thinking Machines tokenwise reverse KL. Name it accurately:

> live cross-tokenizer OPD using byte-aligned chunk likelihood projection.

## Research question

> Can dense DeepSeek likelihood feedback on complete student-visited Tau2
> trajectories make a continued-SFT Qwen3-4B more likely to follow policy at
> critical decision forks, without degrading its existing retail capabilities?

The primary comparison is:

```text
A_sft_new versus the same A_sft_new checkpoint after live OPD
```

## Why OPD can change “understands correctly, acts incorrectly”

An autoregressive model does not first create a fixed conclusion and then
translate it into an action. At each token it chooses among continuations. A few
high-entropy tokens create forks into different reasoning and action paths.

For task 57, the model may correctly state that cancellation refunds must use
the original card, then reach a fork such as:

```text
Path A: "I should call cancel_pending_order ..."
Path B: "I cannot satisfy the gift-card condition, so I must not cancel ..."
```

On a Qwen-sampled trajectory, compare Qwen's behavior log probability with
DeepSeek's probability for the exact realized tokens. Conceptually, for a
same-tokenizer token:

```text
advantage_i = log p_teacher(token_i | student history)
              - log p_student_old(token_i | student history)
```

- If Qwen strongly favors a cancellation token and DeepSeek finds it unlikely,
  the advantage is negative and the update lowers that continuation.
- If Qwen samples a refusal/checking token that DeepSeek favors more, the
  advantage is positive and the update raises that continuation.
- Tokens both models already agree on receive little useful pressure.

The important supervision occurs at the **earliest decision fork**, not
necessarily at the final wrong answer. Once the history already commits to a
bad premise, the final bad answer may be predictable even to the teacher and
receive little penalty.

After the update, new live rollouts are required. If the earlier fork changes,
the student visits different downstream conversation and tool states. That is
why static SFT and stored-prefix training cannot substitute for the proposed
live experiment.

In this repository, Qwen and DeepSeek tokens are mapped through synchronized
byte chunks. For a chunk with student log-probability sum `L_S` and teacher sum
`L_T`, the implemented detached student-token advantage is:

```text
ratio = L_T / L_S
A_i = (ratio - 1) * behavior_logprob_i
```

The trainer then applies the existing clipped importance-sampling policy loss.
This is an approximation to the same-tokenizer token-local mechanism and must
be reported as such.

## Limits of the mechanism

OPD can only amplify behavior that is reachable under the student policy. It
does not guarantee that 4B can discover exhaustive search or long-horizon
planning if those strategies have effectively zero probability.

Teacher likelihood is also a preference signal, not a Tau2 correctness proof.
DeepSeek may favor a fluent but policy-wrong continuation. Therefore the pilot
must report official reward and explicit policy gates alongside the projected
teacher/student log-likelihood gap. With different tokenizers this is not an
exact tokenwise KL; report it as the projected gap, chunk-advantage statistics
and alignment coverage, never as "teacher KL".

Vanilla live OPD may become unreliable after student errors compound across
turns and move the episode far outside the teacher's familiar states. This is a
known multi-turn risk and a reason to keep the first pilot short.

## Checkpoint lineage

```text
A0: frozen Qwen3-4B
  -> SFT on W30
A_warm: CK35
  -> continued SFT on C30
A_sft_new: a_sft_new_ck35_r2
  -> proposed live OPD
A_live_opd
```

CK35 was an earliest-tied-checkpoint choice, not a measured winner over CK70.
The available CK35 and CK70 evaluation tied on tasks 73/75/93; CK35 was chosen
to reduce overfitting risk.

## Reuse or redo the SFT?

Default: **reuse `a_sft_new_ck35_r2`; do not redo SFT merely because it might
be imperfect.** OPD requires a useful initialization, not a perfect one.

The observed SFT-family evaluation is encouraging:

| Task | Result |
| --- | ---: |
| 57 | 2/2 |
| 73 | 2/2 |
| 75 | 2/2 |
| 93 | 0/2 |
| Gradeable | 8/8 |
| Observed loops | 0 |

Before reuse, pass all four gates:

1. **Provenance:** prove that served alias `Qwen3-4B-sft` in
   `tau2_eval_s16_20260826_180059_sft.json` was
   `a_sft_new_ck35_r2`, not CK35 or another adapter.
2. **Data/objective audit:** confirm the exact checkpoint used the frozen C30
   rows, correct policy/tools, assistant-only labels, stable action boundaries,
   no truncation, and the recorded rank-16 all-linear LoRA configuration.
3. **Adapter audit:** adapter-effect probe passes, weights are nonzero, the
   checkpoint reloads exactly, and training recorded finite nonzero gradients.
4. **Behavior audit:** no systematic loop/cap failure and stochastic sampling
   is not collapsed to one memorized action at C30 states.

Redo SFT only if a gate fails:

- Unknown evaluation provenance alone: rerun evaluation with the pinned
  adapter; do not retrain yet.
- Bad checkpoint metadata, labels, truncation or inactive LoRA: redo SFT.
- Good mechanics but collapsed sampling: select an earlier checkpoint or redo
  with fewer steps/lower learning rate.
- Good mechanics and diversity but weak task 93: do not redo automatically;
  that is precisely a capability the OPD pilot is meant to test.

If SFT must be redone, stay on 4B for the deadline and reuse the existing
DeepSeek trajectories. Do one clean W30 warm stage followed by one C30
continuation, with checkpoint selection based on policy compliance, validity,
length and diversity rather than training loss alone.

## Live Tau2 integration

Reuse Tau2's existing orchestrator, environments, tools, user simulator and
grader. Add an OPD-compatible capturing agent/provider; do not rebuild Tau2.

For each assistant turn, the agent must:

1. render the exact Qwen prompt;
2. sample once from the frozen behavior policy;
3. capture raw bytes, token IDs and behavior log probabilities before parsing;
4. derive reasoning/content/Hermes tool calls from that same generation;
5. return the parsed message to Tau2 for normal tool execution;
6. persist the resulting observation and environment-state hash; and
7. later render the same semantic history for DeepSeek scoring.

Persist:

```text
episode_id, task_id, seed, policy_version, turn_index,
prompt_token_ids, sampled_token_ids, behavior_logprobs,
raw_sampled_bytes, sampled-token mask, parsed message/tool call,
finish_reason, observation hash, state hash, usage and timestamps
```

Malformed tools, empty outputs and cap terminations are archived as
`FailedTurn` records **before** the parse error propagates — the generation was
paid for and its ids and behaviour logprobs exist nowhere else. They count in
the validity metrics and are **excluded from OPD training** unless their exact
execution semantics are represented. A cap termination is never trained as a
completed action: it is a fragment, which is what the 256-token cap did to the
0/13 run.

An episode cannot be resumed mid-flight — the Tau2 environment and user
simulator are stateful and cannot be rewound to turn *k*. The only honest
recovery from a crash while sampling is discard-and-resample, so `discarded` is
a terminal state and the batch-completeness rule counts discards explicitly.

## Training invariants

- The student policy is fixed for an entire episode.
- Update only after all episodes in the batch finish.
- Supervise only Qwen-sampled assistant tokens.
- Mask policy, system, tools, user, observation, header and padding tokens.
- DeepSeek scores exact sampled action bytes in the complete student-visited
  semantic history.
- Fail closed on byte mismatch, missing/non-finite score, incomplete alignment,
  truncation, stale policy version or unverified checkpoint reload.
- The next batch must be sampled from the newly reloaded checkpoint.
- Keep one global denominator over supervised student tokens.

## What the live loop still needs

Audited 2026-08-28 against `scripts/tau2_reopd_train.py`.

**The strategy is a new rollout frontend on the existing ReOPD backend, not a
second OPD stack.** That driver already runs
`PLANNED -> SAMPLED -> SCORED -> TRAINED` over `RunState`, and every stage after
sampling is indifferent to where the actions came from — `score_replay_batch`
takes `(actions, prefix_id -> canonical_messages)`, and `run_replay_chunk_opd`,
`ReOPDTrainer.step/.checkpoint` and `refresh_policy` follow unchanged. A replay
update fills those two arguments from frozen C30 prefixes; a live update fills
them from complete Tau2 episodes. **That substitution is the delta.**

Reused unchanged: `RunState` and update directories, the stage lifecycle,
atomic JSON/JSONL persistence, the paid-score cache, the frozen run manifest,
optimizer and Adam-moment resume, RNG checkpointing, adapter save/hash, reload
verification, serving refresh, policy logprob fingerprinting, the Fireworks
teacher pool, DeepSeek native rendering, byte alignment, chunk advantages, the
global supervised-token denominator, the clipped IS step, gradient and
adapter-movement checks, and the Modal wrapper and volumes. No new state
machine, checkpoint format, scoring cache, optimizer or training backend.

Built:

1. **Episode archive** — `tau2/live_episode.py`. What replay has no need for:
   an *episode*. `LiveTurn` carries the semantic history the teacher needs
   (Qwen prompt ids are not a DeepSeek context), plus observation and
   `environment.get_db_hash()` state hashes.

   `EpisodeStatus` covers the **episode** only — `sampling → sampled / failed /
   discarded`. Scoring, training, checkpointing and serving-refresh belong to
   the *update*, and `RunState`'s `PLANNED → SAMPLED → SCORED → TRAINED`
   already owns them; duplicating them per episode makes "trainable" and
   "terminal" contradictory. `failed` and `discarded` stay distinct: one is a
   measurement of the policy, the other an infrastructure event. A crash
   mid-sampling is never resumed — Tau2's environment and user simulator are
   stateful and cannot be rewound to turn *k* — so discard-and-resample is the
   only honest recovery, and both unusable outcomes are counted, never skipped.

   Observations arrive *after* the generation, so they are a separate
   `turn_observed` event merged at read time. The capture is written the moment
   it lands: waiting for the environment would risk the one thing that cannot
   be recreated.

   **Any archived capture failure makes the episode `failed`.** The agent
   re-raises every capture failure, so a turn after a failed turn cannot
   normally exist, and an episode with a hole is a partial trajectory whose
   later turns condition on a state the archive cannot describe. This also
   keeps the archive gate and the batch adapter in agreement — an episode
   accepted here would otherwise be rejected at assembly for non-contiguous
   indices.

   `batch_report` refuses a batch that is empty, lists an episode twice
   (which would double its turns in the denominator while halving diversity),
   spans policy versions, adapters or generation configs, or — given the
   update's `policy_version` — was sampled under the *previous* adapter.
   `verify_episode` checks declared turn and failure counts against what is
   archived, requires a `termination_reason` on a sampled episode, reports
   observations with no capture, and refuses a turn whose `task_id` disagrees
   with its episode — `LivePrefix.task` is read off the turn, so a mislabelled
   one skews exactly the per-task balance `max_task_share` enforces.
2. **Failure archive** — `FailedTurn` + `build_failed_turn`, wired into
   `CapturingLLMAgent` through an `on_failure` hook firing before the capture
   error propagates. The generation was paid for; its ids and behaviour
   logprobs exist nowhere else. Scope is exactly a 200 response that did not
   yield a trainable capture — malformed Hermes, a cap termination, missing
   ids, an unusable logprob. A failure with no generation to salvage (a
   non-200, a transport exception, a prompt over budget) produces no
   `FailedTurn` and is the driver's to record at episode level.
3. **Live-turn adapter** — `tau2/live_turns.py`. Small, because
   `capture_to_sampled_action` already exists: this translates an archived turn
   into the same base64 `actions.jsonl` row shape and returns the
   `(rows, rendered)` pair the existing scorer takes. `LivePrefix` supplies the
   four fields the training path reads off a prefix — `prefix_id`, `task`,
   `trace_id`, `step_index` — and `ReplayPrefix.prefix_id` is already
   `f"{trace_id}@{step_index}"`, which is the live key convention exactly. It
   adds the episode-level rules replay does not need: contiguous turn indices,
   one policy version across a whole batch, no key collisions.

   **One live-vs-replay semantic difference the driver must handle:** an
   episode's turns all share a `trace_id`, so `max_trace_share` (0.35 by
   default) rejects any live batch of few episodes. A live update must pass a
   share limit derived from its episode count rather than inherit the replay
   default.

   `stale_score_keys` closes the one gap the existing cache leaves.
   `score_replay_batch` reuses a paid score when its bytes reconstruct the
   action, which is sufficient for a frozen prefix id. A live turn key can
   recur within one update after a discard-and-resample, where identical action
   bytes may follow a *different* history; that score is stale and must be
   dropped before `already_scored` is passed along. A live score row with no
   recorded history fails closed and is rescored.

   That binding is also enforced **without driver cooperation**.
   `RunState.validate()` already compares an action row's `score_fingerprint`
   against its score row's `fingerprint` on every resume, so
   `live_score_fingerprint` writes the live binding into those same fields.
   `reopd_sample.capture_fingerprint` covers prefix, model, policy version,
   temperature, cap and prompt ids — sufficient for a frozen prefix, whose id
   names a fixed history, but not here: two live turns can share a key, an
   action and a policy version and still follow different conversations, so
   the semantic history is part of the identity. `stale_score_keys` remains the
   cheaper in-batch filter; `RunState` is the backstop.

Still to build:

4. **Live Tau2 rollout driver** — the main new component. Replay samples
   `frozen prefix -> one action -> stop`; live runs
   `state -> action -> tool/user -> action -> ... -> episode end`. Selects tasks
   and seeds, fixes one `policy_version` per batch, calls `set_episode()`,
   persists each turn as it lands.
5. **`scripts/tau2_live_opd_train.py`** — refactored from
   `tau2_reopd_train.py`, substituting live sampling at SAMPLED and blocking
   the optimizer until every required episode is COMPLETE. The shared
   orchestration should be extracted into helpers so the two drivers cannot
   drift.
6. **Two-update on-policy proof** and the canary reports.

The forward-pass cost of recomputing `log pi_current` over full Tau2 histories
is the batch-size constraint and is **unmeasured**. Measure it in Phase 1
rather than discovering it at the first real step.

## Execution plan

### Phase 0 — SFT gate

Resolve the four reuse gates above. Evaluate CK35 on task 57 with the same two
seeds if cheap; this determines whether continued SFT acquired task 57 or merely
preserved it, but it does not block reuse if provenance is otherwise proven.

### Phase 1 — capture/scoring proof

Run four complete live episodes with no training.

Exit only with exact token/logprob lengths, raw/parsed linkage, complete finite
DeepSeek scores, byte alignment, reproducible environment states and measured
GPU/teacher cost per episode and per assistant turn.

### Phase 2 — on-policy proof

Run:

```text
2 updates x 4 complete episodes = 8 episodes
```

Require finite nonzero gradients, zero loss on environment tokens, exact
checkpoint resume/reload and proof that update 1 was sampled from update 0.

### Phase 3 — signal-seeking canary

**Not an efficacy experiment.** This phase asks whether the loop moves anything
at all, and measures the quantities needed to size a real one.

If Phase 2 passes, restart from the frozen `A_sft_new` checkpoint and run:

```text
5 updates x 8 complete episodes = 40 episodes
```

Use task-balanced C30 sampling and fixed generation settings. Save updates 0,
1, 3 and 5.

Against the reference OPD recipe this repository records (256 trajectories per
update, ~150 updates, ~38,400 trajectories), 40 episodes is **32x smaller
update batches, 30x fewer updates, 960x fewer complete trajectories**. The
earlier replay run — 16 trajectories x 32 updates = 512 — was 75x short. A Tau2
episode carries multiple supervised assistant turns, so a trajectory-count
comparison against single-turn reasoning prompts is imperfect; it does not
rescue the scale. The estimated 40-80 supervised assistant actions per update
is itself an estimate, because live turn counts have not been measured.

Phase 3 exits with the numbers that determine the size of any later efficacy
pilot: gradient variance, supervised actions and tokens per update, teacher and
GPU cost per update, and measured checkpoint movement.

Success for Phase 3 means:

- reproducible finite nonzero gradients;
- measurable movement in projected teacher/student log-likelihood gap;
- stable alignment coverage across updates;
- no parser or tool-validity regression;
- some repeated behavioral movement on training diagnostics.

It does not mean "OPD improves Tau2." No efficacy claim is available from 16-24
evaluation episodes.

### Evaluation preregistration

Pin all of the following **in writing before update 0 is trained**, and do not
alter them afterwards:

- exact training task IDs (C30 subset) and exact held-out task IDs;
- paired seeds, identical across the two arms;
- generation parameters (temperature, top-p, max tokens, stop conditions);
- primary outcome (official Tau2 reward) and the policy-compliance rubric;
- handling of ungradeable episodes — excluded, and reported separately;
- task 93 is **diagnostic only**, not part of the primary endpoint.

Report paired per-episode outcomes and the count of paired disagreements, not a
difference of aggregate scores. The screen is exploratory: it can detect
catastrophic regression or a very large behavioral change, and cannot reliably
distinguish a modest improvement from sampling variation.

**Do not spend F38 on this run.** Use C30 acquisition metrics plus a
preregistered S16 exploratory screen. F38 is reserved for a properly scaled,
locked experiment sized from Phase 3's measurements.

## Success and stop rules

Call the pilot promising only if the final checkpoint:

- meets every Phase 3 exit condition above (gradients, likelihood movement,
  alignment coverage, no validity regression);
- retains task 57/73/75 capabilities on the S16 screen;
- does not materially increase invalid actions, loops or cap terminations; and
- does not achieve gains solely by longer reasoning.

"Promising" here means the pipeline works and the signal is non-null — it is a
decision to size and fund a real experiment, not a result.

Stop immediately on invalid capture/scoring, stale policy, non-finite values,
checkpoint failure, repeated loops, or projected total pilot spend above $40.
Do not fix a reasoning loop by raising the output cap.

If the pipeline is valid but 4B shows no learning signal, do not scale 4B
blindly. Repeat a support/baseline audit on Qwen3-8B, train a clean 8B SFT LoRA,
and reuse the now-validated live OPD system after the CLEA deadline.

## Repeating the experiment with Qwen3-8B

An 8B replication should reuse the same frozen task split, DeepSeek teacher,
capturing agent, scoring bridge, update counts, seeds, generation settings,
evaluation tasks and stop rules. Only the student base model, its SFT adapter
and hardware-dependent batch/accumulation settings should change.

If the 4B system has already passed Phases 1-3, the estimated additional time
for a clean 8B replication is:

| Work | Best case | Realistic |
| --- | ---: | ---: |
| 8B serving, VRAM and three-step training proof | 3-6 hours | 0.5-1 day |
| W30 warm SFT plus C30 continued SFT | 2-4 GPU hours | 0.5-1 day |
| SFT checkpoint evaluation and selection | 3-6 hours | 0.5-1 day |
| Repeat Phases 1-3 and paired evaluation | 8-16 hours | 1-2 days |
| **Total additional calendar time** | **2-3 days** | **3-5 days** |

The SFT estimate is provisional until the 8B three-step probe measures actual
memory and step time. Existing measurements establish the 4B runtime, not that
the 8B model will train at the required context length without reducing batch
size or increasing accumulation. Record measured GPU and teacher cost during
the probes before approving the complete replication; do not inherit the 4B
`$40` ceiling without recalculating it.

If work switches to 8B before the 4B live pipeline is validated, none of the
capture/scoring integration risk has been removed. The combined clean-SFT and
live-OPD effort is therefore **4-6 days best case and 6-10 days realistically**.

Eight billion parameters improve the prior probability of adequate reasoning
and action support, but do not guarantee success or a paper. Thinking Machines'
8B result used a much larger warm start and rollout budget in a same-tokenizer
setting. This experiment remains cross-tokenizer, small-data and multi-turn.
A successful 8B pilot would still be preliminary evidence; an 8B failure would
not isolate SFT as the cause because the scoring projection, rollout scale,
tool interaction and OPD configuration remain possible causes.

## Schedule

Best-case implementation and pilot: 4-6 days.

```text
Day 1: provenance/SFT gates; capturing-agent seam
Day 2: raw/parsed capture and DeepSeek-scoring proof
Day 3: two-update proof and fixes
Day 4: 40-episode canary
Day 5: paired evaluation and analysis
Day 6+: paper integration / contingency
```

Maintain the existing harness/failure experience report as the fallback. The
CLEA submission must not depend on a positive pilot result.

## Sources

- Thinking Machines Lab, [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).
- Agarwal et al., [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649).
- Wang et al., [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2506.01939).
- Thinking Machines Lab, [Tinker model-distillation cookbook](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/).
