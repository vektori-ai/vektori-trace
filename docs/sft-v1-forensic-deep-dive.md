# V1 SFT forensic deep dive

**Date:** 2026-08-18  
**Scope:** original v1 SFT, the reason for corrective SFT, and the completed
Phase 7 checkpoint evaluation. This report distinguishes direct measurements
from interpretations. No new GPU run was performed.

## Bottom line

The corrective SFT did **not** fail in the sense of “nothing learned.” It changed
the model substantially and made native Terminus JSON reliable when at least one
native-JSON assistant action was already visible. It **did fail the deployment
gate**: a real rollout begins with no assistant history, and all 42 evaluated
turn-1 generations across the seven checkpoints failed native JSON.

The strongest supported explanation is:

1. v1 had a strong prior for the wrong `<tool_call>` protocol.
2. The corrective dataset gave 3,396 of 3,561 supervised action targets (95.4%)
   a correct native-JSON assistant action in visible history.
3. Under evaluation, the model followed the protocol shown in that history.
4. A same-prefix ablation at checkpoint 63 changed only one visible assistant
   action from native JSON to v1 format; the output flipped from native JSON to
   v1 format.

That proves a **causal dependence on visible history format** for the tested
prefix. It does not, by itself, prove that dataset composition is the only cause
of the cold-start failure. The NF4-train/BF16-serve mismatch and the much weaker
corrective optimization pressure remain unisolated contributors.

## What v1 actually was

The original adapter was a fresh rank-32 LoRA over `Qwen/Qwen3-14B`, trained:

- on 165 segments from 117 passing DeepSeek-V4 rollouts over 34 tasks;
- for five epochs, about 106.9 optimizer steps;
- at learning rate `1e-4` with a cosine schedule;
- against an NF4 base;
- to final training loss `0.6574`.

The source export preserved Harbor's post-parse ATIF representation:

- assistant turns carried structured `tool_calls`;
- observations used `role="tool"`;
- Qwen's template rendered calls as `<tool_call>...</tool_call>`.

Terminus-2 does not consume that representation. It expects assistant **text**
containing a flat object with `analysis`, `plan`, and `commands`, and returns
observations as `role="user"`.

Therefore v1 optimized the teacher's actions in the wrong wire protocol. The
training loss was real, but it measured imitation of the wrong serialization.

## Why corrective SFT was necessary

The rerun was not motivated by a disappointing loss number. It was required
because the target representation was wrong.

The corrected dataset:

- contains 165 segments and 3,561 supervised action turns;
- contains no assistant `tool_calls` and no `role="tool"` observations;
- uses literal native Terminus JSON as assistant text;
- masks 48 handoff-question turns and 18 teacher parse failures;
- retains 7,436 commands, including 146 edits and 635 test invocations;
- uses 3,528 verified raw captures and 33 canonical ATIF rebuilds;
- has 0 raw/rebuild field mismatches;
- fits all rows below 40,960 tokens.

Remote artifact checks:

- dataset SHA-256:
  `7ecfee319b75b3a02e09cc84e9f46f36603aaca0b56e6e44b3971a91910ef3bd`
- preflight: 165/165 tokenized, zero target/role/mask/TRL-agreement failures;
- 1,029,371 supervised tokens out of 3,992,567 total (25.78%);
- token lengths: 6,681 min, 24,635 median, 36,993 max.

This evidence makes a bad reconstruction or empty loss mask unlikely.

## What the corrective run actually did

It continued the existing v1 LoRA rather than stacking or creating a fresh
adapter:

- 63 optimizer steps, three epochs;
- learning rate `1e-5`, constant with five warmup steps;
- NF4 base, despite the earlier BF16 plan;
- all 560/560 LoRA tensors changed;
- loss moved from `0.6436` at step 1 to `0.4856` at step 63;
- finite, non-zero gradient norms;
- 3.05 hours, peak 63.7 GiB allocated.

This rules out “the run silently trained nothing.” Loss alone cannot establish
free-running protocol behavior because teacher forcing supplies the correct
preceding tokens.

The run also repeated v1's train/serve precision mismatch: it trained against
NF4 but Phase 7 served a BF16 base. Phase 7 correctly measured the deployable
stack, but there is no matched NF4-serving arm, so the size of this mismatch's
effect is unknown.

## Was Phase 7 a valid evaluation?

For its narrow question—“does each checkpoint emit native Terminus JSON for a
fixed next-action prefix?”—yes.

Directly verified from `/data/phase7/results.json`:

- SHA-256:
  `12b8d4397310ce2595105bf2c355703b4df15d55f20115d50e13715b49e67cd7`
- 24 fixed prefixes × 7 checkpoints = 168 results;
- zero infrastructure failures;
- all checkpoint/suite cells contain 8/8 graded prefixes;
- 160 `finish_reason=stop`, 8 `finish_reason=length`;
- distinct checkpoint outputs on 23/24 prefixes;
- no checkpoint cleared all three selection gates on all eight held-out
  generalization prefixes.

The grader's repair and driver tests were re-run locally:

```text
68 passed
```

Important limitation: Phase 7 was **not a benchmark pass@k evaluation** and did
not execute full agent rollouts. Its prefixes were purposively selected, not an
i.i.d. task sample. “79/168 = 47%” is a diagnostic completion rate, not model
pass@1.

## Raw result breakdown

Across all 168 generations:

- strict native JSON: **79**
- Harbor accepted: **79**
- v1 legacy envelope present: **71**
- parser failures: **89**
- all applicable format and behavior gates passed: **54**
- output-length truncations: **8**

Harbor acceptance and strict native JSON were identical in this run. The
grader's salvage-vs-native distinction is valid, but it did not change any
classification here.

Parser errors:

- `Missing required fields: analysis, plan, commands`: **76**
- `No valid JSON found in response`: **8**
- `Missing required fields: commands`: **5**

### Failure by turn ordinal

The cold-start result is the decisive one:

- turn 1: **0/42 native**, 41/42 legacy envelope;
- turn 2: 3/28 native, 22/28 legacy;
- turns 6 and 7: 6/7 native at each ordinal;
- turns 10, 14, 16, 24, 31, 33, and 34: **7/7 native**;
- turn 39: 9/14 native.

Every `orientation`, `first_inspection`, and `post_compaction` category was
0/21 native. `first_edit` and `test_exec`, which occur later, were each 20/21.

This is not a small aggregate deficit. It is a state-dependent protocol switch.

## Representative raw failures

### 1. Cold-start legacy envelope

Checkpoint 63, acquisition/orientation, turn 1:

```text
Analysis: We are in a clean workspace at /workspace...
Plan: First, list the contents of the workspace...
<tool_call>
{"name": "bash_command", "arguments": {"keystrokes": "ls -la\n", "duration": 0.1}}
</tool_call>
```

The intent and first command are sensible. Harbor rejects the envelope because
the top-level response is not the required object.

### 2. Correct shape, truncated before closing

Checkpoint 63, acquisition/parse-error-recovery, turn 3:

```text
{
  "analysis": "The previous response was malformed JSON...",
  "plan": "1. Inspect ... 2. Revert ... 3. Add ... 4. Run ...",
  "commands": [
    ...
```

It hit the 512-token generation cap. Harbor reported `No valid JSON found`.
Seven of the eight length truncations are this same prefix across checkpoints,
so this is primarily an evaluation-cap artifact, not evidence of a distinct
learned failure mode.

### 3. Completion object missing `commands`

Checkpoint 10, control/repo-present, turn 21:

```json
{
  "analysis": "The source patch is in place and verified...",
  "plan": "No further commands are needed. The fix has been applied and verified.",
  "task_complete": true
}
```

This is semantically understandable but invalid for Terminus, whose parser
requires `commands` even when the list is empty.

## The context ablation

Checkpoint 63 was tested twice at temperature 0 on the same turn-2 prefix.

Control, one native-JSON assistant turn visible:

```text
native_json=True, harbor_accepts=True, legacy=False
```

Ablation, that one assistant turn rewritten into v1's envelope:

```text
native_json=False, harbor_accepts=False, legacy=True
parser_error="Missing required fields: analysis, plan, commands"
```

This is strong evidence because task, instructions, terminal state, checkpoint,
turn, and decoding were held fixed. The changed history format caused the
output format to change.

Its scope must not be exaggerated:

- one checkpoint;
- one task;
- one turn-2 prefix;
- one deterministic generation per arm.

The 168-generation correlation makes the ablation credible, but replication
over several prefixes is still required before claiming a universal mechanism.

## Corrections to earlier claims

### “v1 baseline was 0/305”

Misleading. In the old `v1-shim` run, 305 was the number of parse-error
observations in one 307-step trajectory. The second had 113 parse errors in 118
agent steps. Both trials hit Harbor's 1,800-second agent timeout. They were not
305 independent samples and not completed benchmark rollouts.

The defensible baseline statement is: **original v1 entered sustained parser
loops and could not be evaluated for task-solving skill in the native harness.**

### “The repair got 47%”

Numerically true only as `79/168` strict-format diagnostic generations. It is
not pass@1, not task success, and not a population estimate.

### “The model copied instead of learning”

Supported operationally, but too absolute. The model learned enough to generate
native JSON under supportive history. The precise statement is:

**The learned behavior did not override v1's cold-start prior; visible assistant
format dominated protocol selection in the tested serving configuration.**

### “No learning rate could have changed it”

Not established. Dataset composition made the shortcut easy, but a stronger or
longer counter-update might still overwrite the old prior. That would be an
expensive and poorly targeted bet, not an impossibility.

### “Dataset composition is the sole cause”

Not proven. Other unresolved contributors are:

- NF4 training versus BF16 serving;
- corrective pressure roughly 17× smaller under the crude
  `learning_rate × optimizer_steps` comparison;
- continuation from a LoRA trained five epochs on the wrong protocol;
- only 165 first-action targets, with protocol tokens a small share of the total
  token loss.

## Why it failed, stated plainly

V1 was taught the wrong language very strongly. The corrective run then showed
the model thousands of examples where the right language was already present in
the conversation. The easiest way to reduce loss was to continue whichever
assistant format was visible. At real rollout start there is no prior assistant
message, so the old v1 behavior wins and emits `<tool_call>`. Harbor rejects it,
which prevents the agent from reaching the later states where the repair works.

That is why the result is simultaneously:

- a real improvement over v1;
- not deployable;
- not evidence that coding intent was lost;
- not a benchmark task score.

## Recommended next experiment

Do not run pass@k or OPD yet. First make cold-start format generation work.

Build a target-centered corrected set where copying visible native JSON is
impossible:

1. **Cold-start examples:** heavily weight every segment's first action.
2. **Anti-copy examples:** for later targets, render prior assistant actions in
   v1 format while retaining native JSON as the target.
3. **Short target windows:** keep the task, relevant terminal state, and enough
   history for action semantics, but avoid carrying 30k tokens merely to teach
   serialization. This may make BF16 training feasible and remove the precision
   mismatch.
4. **Matched initialization check:** compare a small continuation from ck63
   against a branch from immutable v1. Do not assume ck63 is the best
   initialization merely because it works at late turns.
5. **Pre-registered gate:** on the same frozen Phase 7 manifest, require native
   JSON on every `orientation`, `first_inspection`, and `post_compaction`
   prefix, plus replicated history-format ablations.

Only after that gate passes should one guarded no-shim Harbor rollout run. Full
pass@k is downstream of reliable cold-start parsing.

## Evidence locations

- original v1 provenance: `/data/v1-provenance/`
- corrected dataset and audits: `/data/sft-repaired/`
- Phase 7 manifest:
  `/data/phase7/manifest.json`,
  SHA-256 `735063bb0b822cc2b177509c0b2936f3949bb62a1a0e9bc5811fa119937cf5d2`
- Phase 7 raw results: `/data/phase7/results.json`
- context intervention: `/data/phase7/context_probe.json`
- local grader: `vektori_trace/evaluate/phase7.py`
- local driver: `scripts/phase7_eval.py`
- repair builder: `scripts/sft_repair_dataset.py`
- original trainer: `scripts/sft_train_modal.py`
- corrective trainer: `scripts/sft_repair_train_modal.py`
