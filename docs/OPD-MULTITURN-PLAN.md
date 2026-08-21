# Final phased plan: DeepSeek-to-Qwen replay-prefix OPD

**Student:** Stage-B `checkpoint-75` (`ck75`, Qwen3-14B)  
**Teacher:** `deepseek-ai/DeepSeek-V4-Flash-0731`, hosted by Fireworks  
**Loss:** sampled reverse-KL with the published minimal synchronized-chunk
method from *Breaking the Tokenizer Barrier*  
**Environment:** stored DeepSeek trace prefixes; ck75 samples the next action
**Starting scale:** eight replay prefixes x four ck75 actions, one optimizer update

Approval of this document does not authorize paid inference or GPU training.
Each paid phase requires a separate approval.

## 0. Revision — replay only

**Decided 2026-08-21: this experiment runs replay-prefix OPD only. Live
multi-turn Harbor OPD is out of scope.**

The original plan ran a live two-trajectory Harbor smoke first (Phase 1), then
replay (Phase 2) starting from the live-updated adapter. Two reasons that
ordering is dropped:

1. **It made the replay question unanswerable.** `v2` was defined as "`v1` plus
   one replay update", so every replay measurement was conditioned on a `v1`
   produced by two trajectories. §9's branches 2 and 3 ask whether replay is
   useful, neutral, or harmful — none of which a v1-parented `v2` can establish.
   Replay must branch from the untouched baseline.
2. **Live rollout is the expensive, most-blocked half.** It needs Harbor
   sandboxes, a rollout loop that does not yet exist, and a serving path that
   returns per-token behaviour log probabilities. Replay needs none of the
   first two: the prefixes are already on disk.

What this changes, concretely:

- there is no `v1`; the single trained artifact is **`v_replay`, trained from
  frozen `v0`**;
- Phase 1 (live smoke) is removed; the former Phase 2 becomes **the** training
  phase;
- Phase 3 compares `v0` against `v_replay` only;
- everything in Phase 0 stands unchanged — the loss, alignment, pins, and
  training proof are shared by both designs and are already complete.

Live OPD is deferred, not rejected. If replay shows value, a later plan may
revisit it; its exclusions in §2 remain in force until then.

## 1. The decision

The next experiment will test one thing:

> Can DeepSeek provide useful dense likelihood supervision on ck75's own
> actions, sampled at authentic stored trace states, when the two models use
> different tokenizers?

We will not use GOLD. Fireworks does not expose the full DeepSeek vocabulary
distribution required for faithful GOLD, and its top-5 output is not a valid
substitute.

We will not use SimpleOPD as the training loss. Masking tokenizer-mismatched
positions can discard important signal in paths, flags, numbers, JSON, and code.
Exact 1:1 span coverage will be logged only as a diagnostic.

We will use the method published in *Breaking the Tokenizer Barrier: On-Policy
Distillation across Model Families*:

1. ck75 samples an action under its own tokenizer;
2. DeepSeek evaluates the identical action bytes under its own tokenizer and
   native rendered history;
3. the two action tokenizations are divided into minimal synchronized text/byte
   chunks;
4. teacher chunk likelihood is assigned back to ck75 tokens using the paper's
   semantic-prior rule;
5. ck75 receives the standard clipped sampled reverse-KL policy update.

This is the cross-tokenizer extension of the lightweight Thinking Machines OPD
recipe. It requires the teacher log probability of every token on the realized
path, not full logits over all possible tokens.

## 2. Explicit exclusions

The following are outside this experiment:

- SFT or replay SFT;
- GOLD, ULD, forward KL, or a top-5 `other` bucket;
- SimpleOPD masking as the optimization objective;
- teacher-generated replacement actions;
- Harbor success used as a gradient reward;
- updating ck75 after each assistant turn;
- training on the stored DeepSeek action as the target;
- mixing live OPD and replay-prefix OPD before each passes separately;
- a large pilot before the one replay update pass;
- **live multi-turn Harbor OPD, and any adapter parented on a live update
  (see §0)**.

Parser validity and task success are still logged and used for evaluation and
stopping. They simply do not enter the OPD loss in this phase.

## 3. What the teacher computes

DeepSeek is a likelihood evaluator, not a scalar judge. For a ck75-generated
action, it returns the conditional log probability of every token in
DeepSeek's tokenization of that exact action.

For one 300-token Qwen action that becomes 260 DeepSeek tokens, this means:

```text
one teacher-scoring request
    -> log p_D(t_1 | DeepSeek-rendered history)
    -> log p_D(t_2 | history, t_1)
    -> ...
    -> log p_D(t_260 | history, t_<260)
```

It is not 260 requests. The complete supplied sequence is evaluated with
teacher forcing in one logical forward computation. No teacher backward or
teacher optimizer step occurs.

The teacher must process all supervised ck75 action bytes. This is the same
path-scoring requirement as lightweight Thinking Machines OPD. Cross-tokenizer
alignment adds local CPU bookkeeping, not additional teacher generations or
full-logit calls.

Economic cost is determined by processed prefix and action tokens, not HTTP
request count. Request count matters mainly for batching, latency, and rate
limits.

## 4. Context and rendering contract

Autoregressive log probabilities are meaningful only under the complete prior
context. The action must never be scored in isolation or tokenized independently
of the context in which DeepSeek scores it.

Maintain one canonical semantic message history:

```text
system instructions
user task
assistant action 1
environment observation 1
assistant action 2
environment observation 2
...
```

Render that history separately:

```text
canonical messages -> pinned Qwen/Terminus renderer -> ck75 prefix
canonical messages -> pinned DeepSeek-V4 renderer   -> teacher prefix
```

The serialized prefixes may differ because the chat templates differ. Their
semantic messages, actions, observations, compactions, and ordering must agree.
The ck75-generated action bytes appended to the two prefixes must be identical.

Teacher tokenization must be performed over `teacher_prefix + exact_action`,
not by assuming:

```text
tokenize(prefix + action) == tokenize(prefix) + tokenize(action)
```

Returned token IDs, byte values, and offsets identify the action span. A token
that crosses the prefix/action boundary must be prevented by the pinned
assistant boundary or explicitly masked; it must not be assigned using guessed
offsets.

Renderer-injected headers, role tokens, and the Stage-B empty-think wrapper are
context and receive zero loss. Only bytes actually sampled as ck75's completion
are supervised. This experiment does not inject or imitate a private DeepSeek
reasoning trace.

Compaction must also be semantic and explicit. If Harbor compacts the history,
record the exact replacement messages and render that same retained semantic
state for both models. Provider-side silent truncation is a hard failure.

## 5. Cross-tokenizer loss

Let one minimal synchronized chunk contain Qwen tokens
`s_1, ..., s_m` and DeepSeek tokens `t_1, ..., t_k`. Both sides decode to the
same bytes.

Using ck75 behavior-policy log probabilities saved during rollout:

```text
L_S = sum_i log pi_old(s_i | Qwen-rendered prefix and prior Qwen tokens)
L_T = sum_j log pi_D(t_j | DeepSeek-rendered prefix and prior DeepSeek tokens)
```

The published semantic-prior assignment is:

```text
log q_i = (L_T / L_S) * log pi_old(s_i)
A_i     = log q_i - log pi_old(s_i)
```

For a 1:1 chunk this reduces to traditional sampled reverse-KL:

```text
A_i = log pi_D(t_i) - log pi_old(s_i)
```

During the differentiable student training forward:

```text
rho_i = exp(log pi_current(s_i) - log pi_old(s_i))
```

and the implementation applies the paper's clipped importance-sampling policy
loss using detached `A_i`. Microbatch losses are summed and divided once by the
global count of supervised ck75 tokens. Gradients flow only through
`log pi_current`; all DeepSeek values, behavior values, alignment decisions, and
advantages are detached.

We will port or adapt the authors' released implementation rather than invent
new clamps or credit assignment. Any numerical guards used for very small or
very negative chunk likelihoods must match the pinned paper-code revision and
be recorded in the run manifest.

## 6. Phase 0: no-training proof

Phase 0 is local or uses only the minimum explicitly approved Fireworks probe.
It must pass before GPU training.

### 6.1 Pin every artifact

Record hashes or immutable revisions for:

- ck75 base model, adapter, tokenizer, and serving renderer;
- DeepSeek-V4 tokenizer/encoder and chat renderer;
- Fireworks model ID and serving precision;
- the *Breaking the Tokenizer Barrier* paper-code revision;
- Harbor, Terminus, task, environment image, and parser revisions;
- generation parameters, context limits, and compaction policy.

### 6.2 Alignment unit tests

Use synthetic and real Terminus actions containing JSON, shell syntax, paths,
digits, Unicode, whitespace, and special-looking strings. Prove:

- exact byte round-trip on both tokenizers;
- every action byte belongs to exactly one aligned chunk;
- chunks are ordered, non-overlapping, complete, and minimal;
- the decoded bytes of both sides of every chunk are identical;
- summed chunk log probabilities equal the sums of their member tokens;
- assigned Qwen target log probabilities sum back to `L_T` per chunk;
- 1:1 alignment exactly reproduces ordinary sampled-token OPD;
- boundary-straddling and invalid-byte cases fail closed;
- our outputs match the pinned reference implementation on frozen fixtures.

### 6.3 Fireworks scoring probe

For at least one short and one multi-turn frozen transcript, prove that a single
echo/`echo_last` scoring call returns:

- the expected DeepSeek model revision;
- the actual unsampled/teacher-forced token's `logprob`, not
  `sampling_logprob`;
- token IDs and bytes/offsets sufficient to locate the action span;
- one finite log probability per DeepSeek action token;
- no generation and no silent context truncation.

`top_logprobs` is omitted. Top-5 is unnecessary for this objective.

### 6.4 Student training proof

On a mocked two-turn batch, prove:

- rollout captures ck75 token IDs and behavior log probabilities;
- the later ck75 prefix contains the earlier Harbor observation;
- environment and template tokens have zero loss weight;
- the student recomputes current log probabilities with autograd enabled;
- a nonzero finite gradient reaches ck75 trainable parameters;
- exactly one optimizer step changes and reloads the adapter.

### 6.5 Required repository changes before training

The current repository is not yet an implementation of this published method.
In particular:

- `vektori_trace/align.py` can form byte-aligned spans and sum span
  log probabilities, but its existing “estimator B” path treats a whole span as
  one sampled unit; replace that training path with the paper's semantic-prior
  assignment to each Qwen token;
- `vektori_trace/providers/student/fireworks.py` currently requires one teacher
  log probability for every Qwen action token, which is invalid when DeepSeek
  uses a different number of tokens; change the datum to carry detached
  per-Qwen-token advantages produced after chunk alignment, plus behavior
  log probabilities for the importance ratio;
- remove any silent `min_len` truncation from the new path: a length mismatch is
  a hard error, not permission to discard suffix tokens;
- keep `vektori_trace/reopd.py`'s correct prefix/action distinction, but route
  teacher scoring and loss construction through the new chunk implementation.

Add frozen reference fixtures and tests before any paid training run. Passing
the repository's existing tests is necessary but does not prove this new loss
has been implemented.

Failure of any Phase-0 assertion stops the experiment. Do not fall back to GOLD,
SimpleOPD, isolated-action scoring, or an unreferenced byte-group loss.

**Status as of 2026-08-21:** 6.2, 6.3, 6.4 and 6.5 are complete. 6.3 passed
against the live deployment (13 gates, short and multi-turn transcripts,
`/data/opd_probe_report.json`). 6.1's schema exists but ck75's adapter, config
and tokenizer hashes are **not yet recorded** — that is the remaining Phase-0
item, and it gates the first paid call.

## 7. Live multi-turn smoke — REMOVED

Superseded by §0. This experiment does not run live Harbor rollouts, and no
adapter in it is parented on a live update.

The rollout-shape requirements that still apply to replay sampling are carried
into §8.3 rather than left here: the frozen policy version, the task-derived
per-turn token cap (**not** the previous 256-token cap), no-grad generation, and
the full record of sampled bytes, token ids, behaviour log probabilities, masks
and termination reason.

## 8. Phase 1: one replay-prefix OPD update (the only training phase)

**This is the only training phase.** It starts from frozen `v0` (§0), not from
any live-updated adapter, so its result is a measurement of replay OPD against
the untouched baseline rather than a marginal effect on top of something else.

The audited corpus is `/data/vektori-out/dsv4-corpus60` plus `-b`. Measured
2026-08-21 with `replay_corpus.corpus_report`, which is the source of truth for
these numbers:

| | |
| --- | --- |
| trajectories | **240** over **60** tasks |
| passes (`reward == 1.0`) | **117**, across **34** tasks |
| failures | 116 |
| unknown (verifier never ran) | 7 |

Earlier drafts said 238; 240 is what the loader finds. A pass is `reward == 1.0`
and nothing else — the corpus also carries fractional rewards (0.5, 0.125,
0.777…) which are partial credit, not a fixed bug.

**Trajectory lengths, and they are not what earlier drafts assumed:**

| | all 240 | passing 117 |
| --- | --- | --- |
| min steps | 11 | 11 |
| median | **59** | **39** |
| mean | 102 | 66 |
| max | **460** | 240 |

An earlier version of this plan said "14-25-turn cases". That is wrong by a wide
margin: the median passing trace is 39 steps and the longest is 240. This
matters directly for any prefix schedule — ReOPD's default `kappa=0.6` puts
99.4% of its mass in the first ten steps, so on a 39-step median it would render
roughly three quarters of every trace unreachable. See §8.3 on why this run
samples stratified instead.

Begin with valid prefixes from passing trajectories because their environment
histories and compaction boundaries are already reconstructable. Do not claim
that DeepSeek passed all 60 tasks merely because it was run on all 60 — only 34
have a pass.

### 8.1 What a trace contributes

A stored trace contributes:

- a task and concrete Harbor state;
- ordered earlier actions and environment observations;
- authentic long-horizon and post-compaction prefixes;
- task/cap selection and evaluation cases.

It does not contribute the target action for this loss. Even if the old
DeepSeek action log probabilities were saved, they score DeepSeek's old action,
not a new ck75 action.

### 8.2 Replay example

For one selected trace prefix:

```text
stored task + prior actions + prior Harbor observations
        -> render for frozen ck75 v0
        -> ck75 samples a new next action
        -> render the same semantic prefix for DeepSeek
        -> append and score the exact new ck75 action once
        -> minimal-chunk reverse-KL loss
```

The replay state is off-policy because it came from a stored DeepSeek
trajectory. The new action is on-policy with respect to the frozen current
ck75 version. This is therefore replay-prefix OPD, not fully live trajectory
OPD and not replay SFT.

### 8.3 Small replay batch

**Sampling policy: stratified diagnostic, not the ReOPD schedule.** ReOPD
(arXiv:2607.04763) draws replay prefixes under a decaying weight
`w(t, kappa) = kappa ** t`, kappa=0.6, deliberately favouring early trace steps
because those carry the least distribution shift. This run does **not** use it,
for a reason the measured lengths above make concrete: at kappa=0.6, states past
step ~10 are drawn about once per five 32-action batches, and the median passing
trace here is 39 steps. The long-horizon and post-compaction states this run
exists to stress would essentially never appear.

That is a deliberate deviation. `replay_select.reopd_step_weights` implements
the paper's schedule and is available for a later learning-oriented pilot, whose
kappa should be calibrated against the measured length distribution rather than
inherited. The run report records which policy was used
(`selection_policy`), because a stratified batch and a `kappa^t` batch are
different experiments that both report 32 actions.

- Select eight valid prefixes across distinct tasks and trace stages, including
  at least two authentic post-compaction prefixes if available.
- Sample four independent ck75 actions per prefix: 32 replay actions total.
- **Freeze `v0` for all 32 samples**, and record its pinned identity (§6.1).
- **Per-turn token cap: 9216.** Measured, not chosen — see
  `docs/action-length-measurement.md`. Across **all 7,677** assistant actions in
  the stored corpus the median is 534 tokens, p99.9 is 7,557 and the max is
  8,842; 9,216 truncates **none** of them.
  The previous run's 256-token cap cut **69.2%** of actions mid-sequence, and
  the teacher scored those fragments as completed actions; a truncated action
  still aligns and still yields a finite loss, so nothing downstream showed it.
  9216 is a loop guard rather than a length budget. The run fails closed on any
  cap hit — a truncated action is a fragment the teacher would grade as
  complete.

  One caveat carried into the run: those are *DeepSeek's* lengths. ck75's own
  `cap_hit_rate` (§10) is the only evidence the cap fits the student, and a
  non-zero rate means re-deriving it from ck75's samples.
- Generation is no-grad. Save prefix identity, exact sampled action bytes, Qwen
  token ids, behaviour log probabilities, masks, policy version and termination
  reason — the behaviour log probabilities are `log pi_old` in §5's ratio and
  cannot be recomputed later.
- Score every complete new action with DeepSeek using its full semantic prefix.
- Apply the same published chunk loss and exactly one optimizer step.
- Archive the result as **`v_replay`**, parented on `v0`. Nothing overwrites
  `v0`.

The old DeepSeek continuation must not appear in the loss labels. It may be kept
only for qualitative comparison.

### 8.4 Replay pass conditions

- every one of the 32 actions was generated by frozen `v0`;
- every supervised action byte aligns exactly once;
- all student, teacher, chunk, advantage, ratio, loss and gradient values are
  finite;
- no system, user, environment, template or injected think token contributes to
  loss;
- no context was silently truncated;
- the teacher was scored only after sampling, never inside it;
- exactly one optimizer step occurred;
- `v_replay` differs from `v0` and can be reloaded;
- the existing 45-prefix protocol/format canary does not regress;
- every replay prefix corresponds to an actually observed trace state;
- compaction reconstruction matches the historical boundary rather than a
  guessed concatenation of the whole trace;
- ck75, not DeepSeek, generated every supervised action;
- trace/task/prefix IDs make every example reproducible;
- no task or single trace dominates the global supervised-token count;
- `v_replay` reloads and does not regress the protocol/format canary.

Also produce a cost ledger: Qwen generated tokens, Qwen sampling context tokens,
DeepSeek input and scored-action tokens, repeated-prefix tokens, number of
teacher requests, student training tokens, GPU time and cache hits. Thirty-two
actions prove mechanics; they cannot prove improved capability.

## 9. Phase 2: decide from evidence

Compare `v0` against `v_replay` on the same frozen protocol canaries and a small
held-out evaluation. Both share a parent, so the difference is attributable to
the replay update and nothing else.

- `v0`: untouched ck75 baseline;
- `v_replay`: `v0` plus one 32-action replay-prefix update.

The comparison is diagnostic, not statistically conclusive — 32 actions and one
optimizer step cannot separate a real effect from noise. Its purpose is to
decide one thing:

1. If the update is mechanically sound and behaviour moves without protocol
   regression, run a larger replay pilot.
2. If replay is neutral, keep the traces for state selection and evaluation and
   reconsider whether the optimizer should see them at all.
3. If teacher likelihood pressure causes protocol, length or termination
   regression, stop and inspect the rendered histories and chunk advantages; do
   not scale the run.

A larger replay pilot must freeze its prefix selection, learning rate, renderer,
loss and compaction policy before it starts.

Live OPD and any live/replay mixture are out of scope (§0, §2). Should replay
prove useful, a later plan may add live rollouts — and it must give them their
own `v0`-parented arm rather than stacking them on `v_replay`, for exactly the
reason §0 gives.

## 10. Required artifacts and metrics

Every replay example must archive:

- task/environment revision and the source trace/prefix ID;
- policy and adapter version;
- canonical messages and separately rendered Qwen/DeepSeek prefixes;
- exact sampled action bytes;
- Qwen and DeepSeek token IDs, bytes/offsets, and per-token log probabilities;
- minimal chunk membership and `L_S`, `L_T`, assigned targets, and advantages;
- loss masks and global normalization counts;
- request IDs, model revision, latency, token usage, and truncation status;
- the stored observations reconstructed into the prefix, termination reason,
  parser validity, and outcome;
- compaction event and retained/replacement messages.

Report at minimum:

- exact 1:1 span coverage as a SimpleOPD diagnostic;
- aligned chunk counts and 1:1, 1:N, N:1, and M:N proportions;
- chunk byte/token length distributions and maximums;
- advantage sign, magnitude, clipping, and contribution by chunk type;
- supervised-token counts by task, trace and trace stage;
- response length, cap rate, parser validity and termination rate;
- input/scored tokens and actual dollar/GPU cost.

## 11. Stop conditions

Stop without scaling for any of the following:

- Fireworks cannot echo the exact realized DeepSeek action-token log probabilities;
- returned token IDs/bytes cannot be reconciled with the pinned tokenizer;
- prefix/action boundaries cannot be located exactly;
- missing, duplicated, or non-identical bytes in any aligned chunk;
- behavior log probabilities are absent or from the wrong ck75 policy version;
- non-finite or explosively clipped advantages/loss/gradients;
- environment or template tokens receive gradient;
- silent history truncation or mismatched compaction;
- the action scored by the teacher differs from the action ck75 sampled;
- stale adapter or teacher model revision;
- protocol/format canary regression;
- sharp growth in length, cap hits, repetition, or missing termination;
- projected pilot cost is not separately approved.

## 12. Sources

- Niu et al., *Breaking the Tokenizer Barrier: On-Policy Distillation across
  Model Families*: <https://arxiv.org/abs/2606.09456>
- Reference implementation: <https://github.com/ivanniu/On-Policy-Distill>
- Thinking Machines Lab, sampled reverse-KL OPD:
  <https://thinkingmachines.ai/blog/on-policy-distillation/>
- Thinking Machines/Tinker multi-turn Harbor recipe:
  <https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/#distillation-for-multi-turn-tool-use>
- Fireworks completion echo and token log probabilities:
  <https://docs.fireworks.ai/api-reference/post-completions>
- SimpleOPD, used here only to define the exact-span coverage diagnostic:
  <https://arxiv.org/abs/2608.14277>

## 13. Approval sequence

Ask separately before:

1. the minimal Fireworks scoring probe — **done, passed 2026-08-21**;
2. the 32-action replay-prefix update (sampling + teacher scoring + one GPU
   optimizer step);
3. the `v0` vs `v_replay` evaluation, if it needs GPU serving;
4. any larger replay pilot.

Live Harbor/GPU rollouts are not on this list; they are out of scope (§0).
