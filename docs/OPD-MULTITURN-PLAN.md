# Final phased plan: DeepSeek-to-Qwen multi-turn and replay OPD

**Student:** Stage-B `checkpoint-75` (`ck75`, Qwen3-14B)  
**Teacher:** `deepseek-ai/DeepSeek-V4-Flash-0731`, hosted by Fireworks  
**Loss:** sampled reverse-KL with the published minimal synchronized-chunk
method from *Breaking the Tokenizer Barrier*  
**Environment:** Harbor / Terminus, with ck75's actions actually executed  
**Starting scale:** two live multi-turn trajectories and one optimizer update

Approval of this document does not authorize paid inference, Harbor runs, or
GPU training. Each paid phase requires a separate approval.

## 1. The decision

The next experiment will test one thing:

> Can DeepSeek provide useful dense likelihood supervision on ck75's own
> multi-turn Harbor actions when the two models use different tokenizers?

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
- a large pilot before the two-trajectory smoke and one replay update pass.

Harbor outcome, parser validity, and task success are still logged and used for
evaluation and stopping. They simply do not enter the OPD loss in these phases.

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
It must pass before Harbor/GPU training.

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

### 6.5 Required repository changes before the smoke

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

## 7. Phase 1: smallest live multi-turn OPD smoke

This phase proves the real Harbor loop before traces are introduced as training
states.

### 7.1 Rollout shape

- Freeze ck75 version `v0` for the entire rollout batch.
- Start two independent fresh Harbor sandboxes.
- Require at least two assistant turns per trajectory.
- Stop at natural completion or a four-turn smoke cap.
- Use the task-derived per-turn token cap already validated for ck75; do not
  return to the previous 256-token cap.
- Generation is no-grad. Save messages, exact action bytes, Qwen token IDs,
  behavior log probabilities, observations, masks, policy version, and
  termination reason.

At every turn:

```text
ck75 samples action
    -> execute that exact action in Harbor
    -> append Harbor's exact bounded observation
    -> ck75 samples the next action from the resulting state
```

This makes the complete trajectory state distribution student-owned.

### 7.2 Teacher scoring

After both trajectories are collected, score them; do not update within an
episode.

Preferred path: render each complete trajectory once for DeepSeek, request
echoed log probabilities for the complete serialized transcript, and retain
only the ck75-generated assistant spans. This is one logical teacher request
per trajectory.

Fallback: if the endpoint cannot reliably return full-trajectory offsets, issue
one request per assistant action with that action's complete DeepSeek-rendered
prefix. Never score an action without its prefix. Record the additional repeated
prefix-token cost.

### 7.3 One student update

1. Align each action into minimal synchronized chunks.
2. Compute detached chunk likelihoods and per-Qwen-token advantages.
3. Recompute ck75 current log probabilities in training microbatches.
4. Accumulate globally token-normalized gradients over both trajectories.
5. Apply exactly one optimizer step.
6. Publish and reload `v1` only after the whole batch succeeds.
7. Re-score a frozen canary prefix under `v0` and `v1` to prove weight motion.

### 7.4 Hard pass conditions

- both trajectories were generated entirely by frozen `v0`;
- executed actions exactly match the recorded sampled bytes;
- each Harbor response appears in the next ck75 and DeepSeek semantic history;
- every supervised action byte aligns exactly once;
- all student, teacher, chunk, advantage, ratio, loss, and gradient values are
  finite;
- no system, user, environment, template, or injected think token contributes
  to loss;
- no context was silently truncated;
- the teacher was scored only after rollout collection;
- exactly one optimizer step occurred;
- `v1` differs from `v0` and can be reloaded;
- the existing 45-prefix protocol/format canary does not regress.

Also produce a cost ledger: Qwen generated tokens, Qwen rollout context tokens,
DeepSeek input and scored-action tokens, repeated-prefix tokens, number of
teacher requests, Harbor time, student training tokens, GPU time, and cache
hits. This smoke proves mechanics only; two trajectories cannot prove improved
capability.

## 8. Phase 2: one replay-prefix OPD update

Only after Phase 1 passes do we use the existing DeepSeek corpus as optimizer
data. This phase tests replay OPD separately; it is not mixed into the live
smoke.

The audited corpus contains 238 DeepSeek rollouts over 60 tasks: 117 passes and
121 failures. Thirty-four tasks have at least one pass; 26 have none. Begin with
valid prefixes from passing trajectories because their environment histories
and compaction boundaries are already reconstructable. Do not claim that
DeepSeek passed all 60 tasks merely because it was run on all 60.

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
        -> render for current ck75 version v1
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

- Select eight valid prefixes across distinct tasks and trace stages, including
  at least two authentic post-compaction prefixes if available.
- Sample four independent ck75 actions per prefix: 32 replay actions total.
- Freeze one ck75 version for all 32 samples.
- Score every complete new action with DeepSeek using its full semantic prefix.
- Apply the same published chunk loss and exactly one optimizer step.
- Archive `v2` separately; do not overwrite `v1`.

The old DeepSeek continuation must not appear in the loss labels. It may be kept
only for qualitative comparison.

### 8.4 Replay pass conditions

All Phase-1 numerical/alignment conditions still apply, plus:

- every replay prefix corresponds to an actually observed trace state;
- compaction reconstruction matches the historical boundary rather than a
  guessed concatenation of the whole trace;
- ck75, not DeepSeek, generated every supervised action;
- trace/task/prefix IDs make every example reproducible;
- no task or single trace dominates the global supervised-token count;
- `v2` reloads and does not regress the protocol/format canary.

## 9. Phase 3: choose one pilot from evidence

Do not automatically mix live and replay batches. First compare `v0`, `v1`, and
`v2` on the same frozen protocol canaries and a small held-out Harbor evaluation.

- `v0`: untouched ck75 baseline;
- `v1`: one two-trajectory live OPD update;
- `v2`: `v1` plus one 32-action replay-prefix update.

The comparison is diagnostic, not statistically conclusive. Its purpose is to
choose the next single axis:

1. If Phase 1 is mechanically sound and live behavior moves sensibly, run a
   small fresh multi-turn pilot.
2. If replay adds useful movement without protocol regression, run a replay
   pilot before introducing a live/replay mixture.
3. If replay is neutral or harmful, keep the traces for state selection and
   evaluation but do not use them in the optimizer.
4. If teacher likelihood pressure causes protocol, length, or termination
   regression, stop and inspect the rendered histories and chunk advantages;
   do not scale the run.

The first fresh pilot shape is 8 frozen task groups x 4 ck75 rollouts = 32
student-owned trajectories per optimizer batch. The turn cap must be derived
from the actual task distribution: stored successful trajectories include
14-25-turn cases, so copying a ten-turn cap is not justified. Freeze the cap,
task split, learning rate, renderer, loss, and compaction policy before the
pilot.

Only after separate live and replay pilots show value may a later plan evaluate
a mixture such as fresh student-owned trajectories plus replay prefixes. That
mixture requires explicit sampling weights and per-source metrics; it is not
part of this smallest experiment.

## 10. Required artifacts and metrics

Every live trajectory or replay example must archive:

- task/environment revision and sandbox ID where applicable;
- policy and adapter version;
- canonical messages and separately rendered Qwen/DeepSeek prefixes;
- exact sampled action bytes;
- Qwen and DeepSeek token IDs, bytes/offsets, and per-token log probabilities;
- minimal chunk membership and `L_S`, `L_T`, assigned targets, and advantages;
- loss masks and global normalization counts;
- request IDs, model revision, latency, token usage, and truncation status;
- Harbor observation, termination reason, parser validity, and outcome;
- compaction event and retained/replacement messages.

Report at minimum:

- exact 1:1 span coverage as a SimpleOPD diagnostic;
- aligned chunk counts and 1:1, 1:N, N:1, and M:N proportions;
- chunk byte/token length distributions and maximums;
- advantage sign, magnitude, clipping, and contribution by chunk type;
- supervised-token counts by task, trace, turn, and source (live/replay);
- response length, cap rate, parser validity, termination rate, and Harbor pass;
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
- executed Harbor action differs from the recorded action;
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

1. the minimal Fireworks scoring probe;
2. the two-trajectory live Harbor/GPU smoke;
3. the 32-action replay-prefix update;
4. the 32-trajectory fresh pilot;
5. any replay pilot, mixed-source experiment, or larger run.
