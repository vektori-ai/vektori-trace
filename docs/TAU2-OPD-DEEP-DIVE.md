# Live cross-tokenizer OPD on Tau2 retail

Date: 2026-08-28; incident audit updated 2026-08-29
Status: **proposal plus pilot incident record**, not yet the plan of record —
see §0 and §"Pilot incident"
Budget ceiling: $40 in the original proposal; the attempted 10×8 pilot used a
$30 per-run ceiling

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
raw_sampled_bytes, raw/action hashes, reasoning byte span and student-token indices,
sampled-token mask, parsed reasoning/message/tool call,
finish_reason, observation hash, state hash, usage and timestamps
```

Live reasoning is part of the sampled action. The prompt stops at Qwen's normal
generation boundary and must not append the frozen action-only corpus's masked
empty `<think>...</think>` wrapper, which would close reasoning before sampling.
A reasoning-required run rejects any turn without a non-empty reasoning span.
Every episode transition, sampled turn, failed turn, observation and completed
Tau2 `SimulationRun` is also appended immediately to `events.jsonl`; the latter
is persisted in Tau2 `Results` form for the stock viewer.

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

4. **Live Tau2 rollout driver** — `tau2/live_rollout.py` and
   `scripts/tau2_live_rollout.py`. Replay samples
   `frozen prefix -> one action -> stop`; live runs
   `state -> action -> tool/user -> action -> ... -> episode end`. Selects tasks
   and seeds, fixes one `policy_version` per batch, persists each turn as it
   lands, and stops at SAMPLED. Tau2's complete `SimulationRun` and a viewer
   `Results` are archived per episode.
5. **The SCORED → TRAINED → reload → next-rollout bridge** — `tau2/live_train.py`
   and `scripts/tau2_live_opd_train.py`, with the post-sampling stages
   extracted into `tau2/opd_stages.py`. **Only the live driver imports it so
   far** -- `tau2_reopd_train.py` still carries its own copy, so this is a
   shared destination rather than shared execution, and migrating replay onto
   it is a separate change. The live driver
   loops `refresh → roll out → score → step → checkpoint → refresh`, which is
   what makes the arm on-policy: after an update changes the policy, the next
   update's episodes visit *different* downstream conversation and tool states.

   Three things live adds that replay does not need, each a silent failure
   otherwise:

   - **A share limit derived from the episode count.** An episode's turns all
     carry one `trace_id`, so the replay default of 0.35 rejects any batch of
     three or fewer episodes — including the Phase 1 four-episode proof. The
     limit is `min(1, (1/n_episodes) * headroom)`, recorded in the manifest.
   - **Stale-score filtering.** A live turn key can recur inside one update
     after a discard-and-resample, where identical action bytes follow a
     different history; `stale_score_keys` drops that paid score before it is
     reused, and `score_row_provenance` writes the binding `RunState.validate()`
     re-checks on every later resume.
   - **A prompt-parity proof over persisted histories**, run before any teacher
     call is paid for. A history that does not re-render to the captured prompt
     ids is one the teacher would score under a conversation that never
     happened — finite loss, clean logs, wrong number.

Still to build:

6. **Two-update on-policy proof** and the canary reports. The code path exists
   and is unit-tested against injected trainer and teacher stubs; it has not
   been run against a GPU, a live endpoint or a paid teacher call.

The forward-pass cost of recomputing `log pi_current` over full Tau2 histories
is the batch-size constraint and is **unmeasured**. Measure it in Phase 1
rather than discovering it at the first real step.

`MAX_ACTION_TOKENS` is **4096** on the live path, not the replay arm's 2048:
live actions carry reasoning, and the stored-action measurement that justified
2048 was taken over action-only rows. A cap hit is archived as a `FailedTurn`
and fails its episode rather than being trained — a fragment is not a completed
action, which is what the 256-token cap did to the 0/13 run.

## Pilot incident — consolidated audit and restart gate (2026-08-29)

This section is the single current record of what the attempted live pilot
established. Where it conflicts with estimates or operational claims elsewhere
in this document, **this section wins**. The failed volume archive must be kept
unchanged as evidence.

### Attempt and observed outcomes

Run `pilot_10x8_20260829`, plan hash `aa9251ccb6d566fa`, attempted update 0
from untouched parent adapter hash `3869b147ab7ce5d2`. Its frozen manifest
contains 10 updates × 8 episodes = 80 distinct C30 `(task, seed)` pairs. No
teacher scoring or OPD training occurred.

All eight update-0 episodes reached a terminal archive state:

| Episode | Status | Turns | Reward / reason |
| --- | --- | ---: | --- |
| task 95 / seed 1 | sampled | 10 | 1.0 |
| task 71 / seed 1 | sampled | 9 | 1.0 |
| task 53 / seed 2 | sampled | 8 | 1.0 |
| task 108 / seed 1 | sampled | 8 | 1.0 |
| task 44 / seed 1 | failed | 3 | unclosed `<think>` before tool call |
| task 68 / seed 0 | failed | 3 | unclosed `<think>` before tool call |
| task 76 / seed 0 | failed | 2 | unclosed `<think>` before tool call |
| task 109 / seed 0 | discarded | 1 | HTTP 408 infrastructure loss |

The honest denominators are different and must not be conflated:

- 4/8 planned episodes completed and were gradeable;
- 4/4 completed episodes received reward 1.0, a selected conditional result,
  **not** a 100% batch success rate;
- 3/7 non-infrastructure episode attempts encountered the parser-visible
  delimiter form (42.9% of episodes);
- the form occurred on 3 of roughly 43 generated assistant turns (about 7% of
  turns), not "40% of turns";
- one of eight planned episodes was lost to infrastructure.

Measured successful episodes averaged about 8 turns, superseding the proposal's
unmeasured "~13 turns" estimate. Consequently 10×8 is roughly 640 actions, not
about 1,040. These are correlated turns inside stateful episodes, not independent
single-turn prompts.

### Raw-generation diagnosis

The three failed generations were read directly from the Modal volume. Each:

1. contains `<think>`;
2. contains 288–1,347 tokens of coherent task-related reasoning;
3. contains one or more syntactically valid `<tool_call>...</tool_call>` blocks;
4. omits `</think>` before the first tool call; and
5. ends with `finish_reason: stop`, not `length`.

Therefore "the model dropped reasoning" and "the output was truncated" are
false descriptions. The raw model output used an unclosed-think/tool-call form.
The current client parser then misclassified it:

```text
raw: <think> reasoning ... <tool_call>...</tool_call>
                         ^ implicit reasoning boundary

current _THINK regex: requires <think>...</think> -> no match
current result: reasoning=None -> require_reasoning gate refuses the turn
```

This form is not merely a local repair hypothesis. vLLM upstream commit
[`92762ed`](https://github.com/vllm-project/vllm/commit/92762edc535c696c3b8a5f3ffee9bc1c0fac10e6),
"Treat `<tool_call>` as implicit reasoning end in Qwen3 parser", explicitly
supports it. The repository pins vLLM 0.21.0, but live capture posts
pre-tokenized ids to `/completions`; vLLM's `qwen3` reasoning parser and
`hermes` tool parser do not run on that route. `split_generation()` replaces
them client-side and has drifted behind the upstream Qwen parser contract.

There is a second, model-side contributor: SFT supervised visible actions only
after a template-provided empty `<think>\n\n</think>\n\n` wrapper. The closing
tag was context, not a supervised target. Live sampling correctly stops at
`<|im_start|>assistant\n` so reasoning can be generated, which requires the
model to emit both delimiters itself. Thus:

- parser parity drift is the immediate capture defect;
- SFT/live generation-boundary skew plausibly raises the frequency of this
  output form;
- the untouched parent produced it before OPD, so OPD did not cause these three
  failures;
- the earlier claim that one OPD update broke `</think>` remains an unproven
  causal hypothesis, not a finding.

No evidence points to `enable_thinking`, a stop string, or the action-token cap:
thinking is explicitly enabled, the completion request sends no stop sequence,
and these responses ended with `stop`.

### Required parser correction

Port the narrow upstream behavior, rather than inventing a general
"reasoning-to-end" heuristic:

1. Preserve normal closed `<think>...</think>` handling.
2. If `<think>` is unclosed and a complete, valid `<tool_call>` begins later,
   treat the first tool-call opening as the implicit end of reasoning.
3. Reasoning bytes end immediately before that tool-call opening. Tool-call
   bytes are parsed only as tools; the two spans must be disjoint.
4. Do not invent `</think>` bytes or tokens.
5. Refuse an unclosed think with no unambiguous valid tool-call boundary,
   malformed tool JSON, empty reasoning, mixed unexplained suffix text, or
   multiple ambiguous reasoning blocks.
6. Persist a recovery/parser-mode field and include the parser contract/version
   in generation and score fingerprints.

Required tests cover closed think, one and multiple implicit-boundary tool
calls, malformed/unterminated tools, empty reasoning, mixed text, multiple
think blocks, raw-byte reconstruction, disjoint token masks, Tau2 execution,
DeepSeek semantic rendering, and parity with the pinned vLLM Qwen parser.

This is a preregistration amendment because it changes capture eligibility.
It does not authorize accepting arbitrary missing-reasoning output.

### Independent numerical blocker: projected chunk advantages

The pilot must **not restart after only the parser fix**. The live projected
training path currently diverges from `chunk_opd.py` for N:1 or M:N
cross-tokenizer alignments.

The reference computes one ratio per aligned byte chunk:

```text
L_S = sum student behavior logprobs in the chunk
L_T = sum teacher logprobs in the chunk
ratio = L_T / L_S
A_i = (ratio - 1) * behavior_logprob_i
```

The live path instead divided `L_T` equally among student tokens and computed a
separate ratio against each token's `L_S`. The two functions differ whenever the
student logprobs **inside one chunk** are unequal.

Worked N:1 case, verified numerically 2026-08-29 — one teacher token against
three student tokens, teacher and student chunk likelihoods exactly equal
(`L_S = L_T = -3.0`):

```text
student logprobs   [-0.5, -1.0, -1.5]      teacher chunk total  -3.0

chunk rule     ratio = L_T/L_S = 1      -> A = [ 0.0,  0.0,  0.0]
per-token rule share = L_T/3 = -1.0     -> A = [-0.5,  0.0, +0.5]
```

Opposing gradients at exact teacher/student agreement, cancelling in aggregate
and wrong individually.

**The equal-logprob case does not expose this.** For `[-1.0, -1.0, -1.0]`
against the same `L_T`, both rules return `[0, 0, 0]`. An earlier draft of this
section used such an example and claimed a nonzero divergence; that arithmetic
was wrong, and a regression test built on it would have passed against the
defect. The distinguishing case must use **unequal** student logprobs. This is
also why one-byte-per-token and uniform-logprob fixtures hid the bug.

**Repaired 2026-08-29 in commit `4b82d09`.** `live_score` now persists whole
`ProjectedChunk` records (chunk id, kind, action-level student indices, teacher
logprobs); `live_batch.chunks_to_alignment` maps the sparse live frame into the
dense frame and delegates to `chunk_opd.assign_chunk_advantages`, so the
arithmetic exists once. The endpoint remains tokenwise advantages for a
tokenwise loss; only the ratio is chunkwise:

```text
aligned chunk -> one chunk ratio -> tokenwise advantages -> tokenwise loss
```

Chunk identity is carried through all three layers — in-memory score,
persisted score row (`score_algorithm: "chunk-v2"`), and resume
reconstruction — because a fresh run alone being correct while a resume
reverted to flat credit is the same defect with a longer fuse. Score rows
predating the fix are refused rather than reinterpreted.

19 equivalence tests cover 1:1, 1:N, N:1, M:N, mixed and sparse-index shapes
against `chunk_opd`, and pin both the defective rule and the equal-logprob
blind spot.

**The two earlier mechanism-proof checkpoints ran through the pre-repair path.**
They establish that the pipeline executes end to end — sampling, scoring,
optimizer step, reload — and they do **not** establish that the update
direction was correct, because chunk grouping had been discarded before the
advantages were formed. Their loss and gradient numbers are evidence of
execution, not of trustworthy OPD signal.

### Other confirmed correctness and reliability issues

These are restart gates unless explicitly classified otherwise:

1. **Visible content before tools can lose all teacher weight.** A DeepSeek
   token that straddles the content/DSML boundary causes the entire content
   payload to be excluded. Fifty content-class tokens were omitted in one
   archived reconciliation. Quantify and either repair boundary handling or
   preregister the omission; do not silently call it complete coverage.
2. **`thinking_mode` is absent from teacher identity.** The scorer hardcodes
   `"thinking"`, but the teacher-context hash does not bind it. Add it to the
   manifest, context hash and score fingerprint so a mode change invalidates
   cached scores.
3. **Disk score reuse is key-only.** A cached score row is reused without first
   comparing its fingerprint to the current action's `score_fingerprint`.
   Require exact equality before reuse.
4. **Stale rescoring can append duplicate score keys.** Replacing a stale row
   currently appends beside the old row; later validation then fails. Rewrite
   atomically or deduplicate before commit.
5. **Multiple think blocks and duplicate reasoning text are ambiguous.** Refuse
   them or define and test a unique byte-span rule; never supervise leftover
   markup as visible content.
6. **`rescore` and training disagree on share enforcement.** Training treats
   realized shares as telemetry, while `rescore` still defaults to enforcing
   them. Make analysis use the training contract.
7. **HTTP 408 remains an infrastructure issue.** Correlate request duration,
   client timeout and server expiry before labeling it random. It must not be
   counted as a policy failure.

### Why the failed run cannot simply resume

Stage-local recovery works for completed rollout, partial scoring, and failed
training. It does **not** recover a terminal failed/discarded episode:

1. the archive marks failed, discarded and interrupted episodes terminal;
2. on resume, `capture_live_update()` skips every episode id already present;
3. `batch_report` still requires all eight planned ids to be `sampled`;
4. therefore update 0 remains below 8/8 forever and `.SAMPLED` is never written;
5. `--start-at 0` reruns the same validation; it does not force resampling.

The existing "discard-and-resample" and generic "resume stage-locally" wording
overclaims the implementation. `--start-at N` can also skip an incomplete
earlier update and must be guarded against `pilot_status.next_update`.

Do not delete or surgically edit the failed archive. The safe restart is a new
run id with:

- the same untouched parent;
- the same exact 80 frozen `(task, seed)` pairs and plan hash;
- an amended manifest binding parser behavior/version and all scoring
  identities;
- a link to the failed run and this incident record.

This is a restart of the same planned schedule under a declared implementation
correction, not checkpoint selection or replacement sampling.

### Mandatory restart gates

No GPU, endpoint, rollout, teacher call or training step is authorized until:

1. ~~implicit Qwen tool-call reasoning boundaries match pinned vLLM behavior
   and pass raw-byte/span/execution/projection tests~~ — **met** 2026-08-29
   (`eb2a79e`). `PARSER_VERSION = "v2"`; one resolver shared by
   `split_generation` and `_reasoning_byte_span` so the parse and the scored
   byte span cannot disagree. Ambiguity still refuses: multiple openers, an
   unpaired closer, empty reasoning, an incomplete tool call.
2. ~~N:1 and M:N advantages match `chunk_opd.py` exactly~~ — **met**
   2026-08-29 (`4b82d09`); chunk identity carried through scoring, persistence
   and resume, arithmetic delegated to `chunk_opd`.
3. ~~teacher `thinking_mode` and parser contract are fingerprinted~~ — **met**
   2026-08-29 (`eb2a79e`). `live_score_fingerprint` binds `PARSER_VERSION`,
   `PROJECTION_VERSION`, `SCORE_ALGORITHM`, `thinking_mode` and
   tokenizer/teacher identity, so the parser change in gate 1 invalidates
   every score bought under the old splitter.
4. ~~score reuse checks fingerprints and stale rows cannot duplicate keys~~ —
   **met** 2026-08-29 (`eb2a79e`). Reuse requires exact fingerprint equality;
   a rescore replaces atomically and pre-existing duplicates collapse.
5. ~~content-boundary loss is measured and resolved or explicitly
   preregistered~~ — **met (preregistered)** 2026-08-29 (`eb2a79e`).
   `PROJECTION_VERSION = "v1"` supervises reasoning and visible content only;
   Qwen markup, Hermes tool JSON, `<|im_end|>` and boundary-straddling tokens
   carry zero weight and are counted by reason, with `retained_fraction`
   reported every run. **Tool calls are conditioned on but never credited** —
   Hermes JSON and DeepSeek DSML share no bytes to map through. This is a
   declared scope limit, not a defect, and it is why this arm is an
   *adaptation* of the paper's cross-tokenizer OPD rather than a reproduction.
6. ~~rollout retry semantics and `--start-at` guards match the runbook~~ —
   **met** 2026-08-29. `--start-at` now refuses to step over an update the
   volume reports as untrained (`next_update`), which was logged but never
   enforced; `--allow-gap` is the explicit, warned override. A skipped update
   leaves every later update parented on a checkpoint that was never
   produced, and nothing downstream can tell.
7. a fresh manifest freezes the unchanged 80-pair schedule and records the
   amendment;
8. ~~CPU-only tests and a no-teacher scoring dry run pass~~ — **met**
   2026-08-29. Full suite green; `scripts/tau2_offline_rehearsal.py` runs
   archived actions through the real parser, projector and chunk-advantage
   code with a deterministic fake teacher, and a mocked two-update rehearsal
   proves Adam-state continuation and update-1-parented-on-update-0 lineage.

   **What the rehearsal cannot prove.** Score rows archived before 2026-08-29
   hold only flat per-token credit whose chunk grouping is unrecoverable, so
   no genuine post-repair numerical replay can be produced from them — that
   they are refused *is* the fix. Real advantages require a paid DeepSeek
   rescore, which is the first paid step below and nothing earlier.

Gate 7 (a fresh frozen manifest under a new run id) remains open;
it is operational, not numerical, and is the last step before the paid ladder.

Then use the original cost ladder rather than jumping directly to unattended
10×8 execution: one reasoning-required episode, a two-update on-policy proof,
and only then the signal pilot. Every exit path must still verify zero Modal
ephemeral apps from both clients.

## Execution plan

### Phase 0 — SFT gate

Resolve the four reuse gates above. Evaluate CK35 on task 57 with the same two
seeds if cheap; this determines whether continued SFT acquired task 57 or merely
preserved it, but it does not block reuse if provenance is otherwise proven.

### Phase 1 — capture/scoring proof

**Run ONE episode first, not four.** `capture_live_update` catches a per-episode
failure and continues to the next plan, rejecting only at `batch_report` -- so a
four-episode update whose reasoning capture fails on episode 1 still pays the
endpoint and user simulator for episodes 2-4 before refusing the batch. The
single episode is the cheapest runtime test in the ladder and therefore comes
first; the two-update proof is the smallest run that can fail at
`checkpoint -> reload -> next rollout` specifically, which is a different and
later question.

The ladder, in cost order:

```text
preflight (CPU, $0)
  -> 1 reasoning-required episode  (+ score exactly that episode)
  -> 2 updates x 4 episodes        (the on-policy proof)
  -> 5 updates x 8 episodes        (the signal pilot)
  -> paired S16 evaluation
```

Do **not** run the diagnostic with `--allow-missing-reasoning`. The earlier
13/13 `reasoning: None` result came from the obsolete pre-closed prompt; the
corrected boundary deserves a real test, and relaxing the gate would prove
nothing about it.

Then run four complete live episodes with no training.

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

Against Thinking Machines' published numbers, restated in the unit that
matters here -- *graded student sequences*:

| Experiment | Updates | Graded sequences |
| --- | ---: | ---: |
| TML main reasoning (Qwen3-8B) | ~150 | ~308,000 (77K prompts x 4 samples) |
| TML single-prompt data-reuse demo | 20 | 5,120 |
| TML "recovered the teacher in <10 steps" | <10 | up to ~2,560 |
| **Phase 3 here** | **5** | **~520** (40 episodes x ~13 turns) |

So Phase 3 is roughly **590x** smaller than their main run, **10x** smaller
than their single-prompt demo, and **5x** smaller than their fastest recovery
result. Their main run also initialized from SFT on 400,000 prompts.

Two accounting caveats, so these numbers are not over-claimed. TML's article
reports 77K prompts while also calling `64 prompts x 4 samples = 256 rollouts`
their default batch; the former is presumably a global/distributed batch and
the article does not fully specify the relationship. And our 520 turns are
*stateful, correlated, tool-using* turns within 40 episodes, not independent
completions -- richer per sequence, but far less diverse than 520 separate
prompts. The
earlier replay run — 16 trajectories x 32 updates = 512 — was 75x short. A Tau2
episode carries multiple supervised assistant turns, so a trajectory-count
comparison against single-turn reasoning prompts is imperfect; it does not
rescue the scale. The estimated 40-80 supervised assistant actions per update
is itself an estimate, because live turn counts have not been measured.

Phase 3 exits with the numbers that determine the size of any later efficacy
pilot: gradient variance, supervised actions and tokens per update, teacher and
GPU cost per update, and measured checkpoint movement.

**The closest affordable replication target**, to be sized from those numbers
rather than committed to now:

```text
20 updates x  8 episodes = 160 episodes ~ 2,000 graded turns   (~TML's <10-step scale)
20 updates x 16 episodes = 320 episodes ~ 4,000 graded turns   (~TML's reuse demo)
```

Decision rule after Phase 3: gradients collapse or behaviour degrades -> stop;
healthy gradients but negligible movement -> 20x8; strong but noisy movement ->
20x16 if budget allows; already-repeated positive paired changes -> report as
preliminary and use the larger run as follow-up.

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
