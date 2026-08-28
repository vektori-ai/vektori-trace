# Tau2 Live OPD: verified learnings from the first two-update proof

**Investigation date:** 2026-08-29 (Asia/Kolkata)  
**Run:** `two_update_proof_20260828_182302`  
**Branch:** `feat/tau2-live-opd`  
**Diagnostic/fix commit:** `ceccb286ee93f41cf66683bfa15e8e778b0d52da`

## Simple explanation

The live loop itself worked: Qwen ran Tau2, DeepSeek scored Qwen's sampled
actions, one optimizer step changed the adapter, the new adapter was reloaded,
and another rollout was attempted.

The first training signal was not clean, however. DeepSeek was asked to assign
probabilities to Qwen-specific control strings such as `<think>` and
`<tool_call>`. Those strings are serialization instructions, not reasoning or
tool decisions, and DeepSeek uses a different native representation for them.
The resulting penalties were tens of times larger per token than the penalties
on semantic text. The update therefore mixed the desired signal (preference
over reasoning and tool content) with an invalid signal (preference over
Qwen's wire format).

The next run must prevent model-specific syntax from receiving cross-model OPD
weight while retaining reasoning text and tool semantics.

## What the run proved

Update 0 completed every stage:

- 4 Tau2 episodes and 31 captured assistant turns, with no capture failures.
- 31 DeepSeek score rows, all finite.
- One optimizer step: loss `0.731062`, gradient norm `0.730949`, 14,178
  supervised tokens, and peak memory about 12.9 GiB.
- All 504 LoRA tensors moved; maximum reported parameter delta was `1e-5`.
- The child adapter hash was `7d5ed1a75bccf482`, different from parent hash
  `3869b147ab7ce5d2`.
- Reload was real: the fixed log-probability probe changed by `0.03607`.

The next-policy rollout also genuinely ran. One episode completed, while three
episodes were refused because their raw generations opened `<think>`, emitted
non-empty coherent reasoning, and reached `<tool_call>` without first emitting
`</think>`. Thus the observed failure was an unterminated reasoning block, not
an absence of reasoning.

This proves the rollout -> score -> train -> checkpoint -> reload -> next
rollout machinery executed. It does **not** prove that the implemented scoring
objective was a valid semantic cross-tokenizer OPD objective.

## Corrections made during the investigation

### Negative action means are not evidence of collapse

All 31 action-level mean advantages were negative. That is expected for actions
sampled from the student:

`E[x~student](log p_teacher(x) - log p_student(x)) = -KL(student || teacher)`.

An action mean also hides token-level signs. The actual update contained 3,388
positive, 10,713 negative, and 77 zero token advantages. Therefore the earlier
description of the step as "uniform suppression" or having "no positive
target" was false.

### `thinking_mode="thinking"` is not a one-line repair

The live scorer inherited `thinking_mode="chat"`, whose DeepSeek prefix closes
reasoning with `</think>`. This is wrong conditioning for a raw Qwen action that
starts with `<think>`.

But changing only the flag also fails. With `thinking` mode, DeepSeek's template
opens its own reasoning section and the joint renderer closes it before
appending the raw Qwen action. On a real archived action the scored extension
became:

`</think><think>\nOkay, I need...`

instead of the original:

`<think>\nOkay, I need...`

`locate_action_span` correctly rejects this byte mismatch before a teacher API
call. Native semantic rendering is required; a mode toggle is insufficient.

### The missing `</think>` was not directly explained by a huge `</think>` penalty

Exact reconstruction shows `</think>` had mean advantage `-0.220915`, with a
range from `-3.359367` to `+1.076188`; 27 were negative and 4 positive. This is
not the dominant direct penalty.

The post-update missing delimiter may have been collateral movement caused by
large gradients elsewhere, especially because the delimiter was not directly
anchored by the SFT loss. That is a plausible hypothesis, not yet a causal
result. Advantage magnitude alone is insufficient to identify which examples
dominated the parameter gradient.

## Exact token-class evidence

The analyzer in `vektori_trace/tau2/live_token_classes.py` replays the archived
byte alignment and advantage calculation. It makes no teacher calls, allocates
no GPU, and changes no weights.

| Class | Tokens | Positive | Negative | Mean | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| markup | 80 | 4 | 76 | -24.469495 | -55.289224 | +1.076188 |
| reasoning | 11,707 | 2,523 | 9,144 | -0.599669 | -24.499999 | +3.120110 |
| content | 1,760 | 849 | 907 | -0.702808 | -27.057084 | +2.865830 |
| tool JSON | 631 | 12 | 586 | -0.276892 | -13.249996 | +0.000038 |

Markup is only 0.56% of supervised tokens, but its mean magnitude is about 35x
to 88x the semantic-class means. This establishes severe serialization
contamination. It does not, by itself, establish the exact share of the final
parameter gradient contributed by each class.

Important exact-token distributions:

| Student token | n | Mean | Minimum | Maximum | Interpretation |
|---|---:|---:|---:|---:|---|
| `<think>` | 31 | -40.644954 | -45.999950 | -35.749858 | Every opener was catastrophically penalized under the mismatched teacher context. |
| `</think>` | 31 | -0.220915 | -3.359367 | +1.076188 | Small/mixed relative to the openers. |
| `<tool_call>` | 16 | -43.158541 | -55.289224 | -0.014444 | Every occurrence was negative, but not every occurrence was near -50. |
| `</tool_call>` | 2 | -0.090502 | -0.174250 | -0.006754 | Small direct penalty in the supervised subset. |
| `<|im_end|>` | 17 | -19.665078 | -27.057084 | -0.228742 | Qwen special token incorrectly classified as content by the first analyzer. |
| `{"` | 16 | -9.151210 | -13.249996 | -0.001534 | Likely a tool-serialization boundary effect; requires semantic projection to confirm. |
| newline | 72 | -5.737275 | -24.499999 | +0.740532 | Mixed class/context; it must not be globally masked merely because some instances are extreme. |

This corrects the initial report in two ways: `<think>` is also a catastrophic
source, and the six most-negative `<tool_call>` examples are only the displayed
tail of 16 occurrences—not the complete distribution.

### How to read the negative advantages

The sign is not itself the bug. Because the actions were sampled from the
student, the expected action-level log-likelihood gap is the negative reverse
KL. Negative action means are therefore normal. The defect is the **location
and scale** of the token-level signal: model-specific serialization received
mean advantages around `-20` to `-43`, while semantic classes averaged roughly
`-0.28` to `-0.70` and contained thousands of positive tokens.

The actionable finding is consequently not "advantages were negative." It is:

> The teacher was allowed to express an enormous preference over the student's
> private wire format, even though only preferences over shared response
> meaning are transferable.

Clamping would reduce the numerical symptom without repairing that invalid
question. Content-only/native projection is the repair; clamping remains a
separate robustness decision after valid scores are measured.

## Independent confirmation from the pinned reference

The vendored implementation for *Breaking the Tokenizer Barrier* states in
`opd_reference/reward_manager_opd.py::_align_chunks`:

> Both student_ids and teacher_ids should be the response content only (no chat
> template special tokens on either side).

It also detects Qwen and DeepSeek model families and maps chat-template markers
before teacher tokenization. In particular, Qwen `<|im_end|>\n` becomes
DeepSeek `<｜end▁of▁sentence｜>`. Our proof-run scorer instead required the
teacher token bytes to reconstruct the complete raw Qwen action. That directly
violated the reference aligner's precondition and explains why Qwen template
tokens entered the loss.

The reference does **not** implement Qwen JSON tool-call to DeepSeek DSML
projection: the pinned source contains no corresponding `tool_call`, `DSML`, or
`invoke name` conversion. Its experiments therefore do not provide a reusable
solution for our tool serialization. Version one must exclude that region or
implement an independently justified semantic mapping.

The reference comment that the Qwen-to-DeepSeek replacement is "already clean"
means that the replacement leaves no trailing newline. It does not mean raw
Qwen markup is safe to score under DeepSeek.

## Semantic-projection acceptance result

Commit `52b5a41` added the minimal eligibility adapter. It reuses
`split_generation`, `encode_messages`, and `align_by_bytes`; it adds no
tokenizer, parser, or alignment algorithm. Version one retains reasoning and
visible-content payloads and excludes tool-call serialization and bytes outside
those payloads.

The free offline report over the 31 archived update-0 actions returned:

| Measurement | Result |
|---|---:|
| Actions projected | 31/31 |
| Raw archived tokens | 14,206 |
| Eligible reasoning tokens | 11,707 |
| Eligible visible-content tokens | 1,710 |
| Total eligible | 13,417 (94.45%) |
| Excluded tool serialization | 663 |
| Excluded outside any payload | 126 |
| Projection failures | 0 |
| Markup supervised | 0 |
| Tool JSON supervised | 0 |

This proves eligibility bookkeeping, not teacher scoring: the production
scoring path still needs to render the structured action natively, obtain
teacher scores for each eligible payload, and map them back to the original
student token indices.

### Token-count reconciliation gate

Commit `c24e838` added `--reconcile-only` to cross-tabulate the old chunk mask
against the new projection token by token. Both apparent discrepancies are now
fully reconciled:

`14,206 raw - 14,178 chunk-supervised - 28 sentinels = 0`.

The 28 are upstream-style zero-advantage sentinels from over-long chunks or
unaligned tails. The first classifier intentionally iterated only supervised
tokens, so it never reported them.

The projection removes 761 previously supervised tokens:

| Removed class | Tokens |
|---|---:|
| tool JSON | 631 |
| markup | 80 |
| content-classified boundary/edge tokens | 50 |
| **Total removed from prior supervision** | **761** |

Adding the 28 tokens that were already sentinels gives the projection's 789
excluded raw tokens. Thus the earlier 78-token gap is exactly 50 conservative
edge exclusions plus 28 existing sentinels.

The 50 are payload-boundary straddlers or bytes outside transferable payloads,
including block whitespace, reasoning-to-content junctions, and trailing
specials. They are not all evidence of contamination; they are deliberately
declined because a student token cannot carry a fractional advantage when only
part of its bytes belong to an eligible payload.

Final accounting is exact:

`13,417 projection-supervised + 761 newly removed + 28 existing sentinels = 14,206`.

The cross-tab also has zero tokens in `chunk=False, projection=True`: the new
projection never revives a token the prior optimizer had already refused.

## Proven implementation defects

1. Update 0 archived an empty rollout `adapter_hash`. The checkpoint's parent
   hash was correct, so the transition was genuine, but rollout provenance was
   incomplete. Commit `ceccb28` now derives the hash from the actual parent
   adapter and refuses an explicit mismatch.
2. The live path silently inherited the replay path's `thinking_mode="chat"`.
   That replay default was appropriate for action-only replay data but not for
   reasoning-inclusive live actions.
3. The raw Qwen serialization was scored as though it were native DeepSeek
   output. This gave Qwen-specific control tokens invalid teacher preferences.
4. The first token classifier omitted `<|im_end|>` from markup, causing its
   large negative scores to appear under `content`.
5. `advantage_clamp` was `None`. This is recorded configuration, not inherently
   a bug; clamping should not substitute for fixing invalid spans.
6. The proof-run scoring contract required teacher bytes to reconstruct the
   student's entire serialized action. The pinned reference instead requires
   response-content-only alignment and translates family-specific template
   markers before token alignment.

## What is not yet proven

- That syntax contamination alone caused the missing `</think>`. The temporal
  association is strong, but per-class gradient contribution or an ablation is
  needed for causality.
- That masking structural tokens alone will preserve formatting. Updates from
  semantic tokens can still move structural-token probabilities.
- That `{"` and newline extremes are all invalid. They may combine genuine
  semantic likelihood with serialization-boundary artifacts.
- That any particular advantage clamp is justified.
- That OPD improves Tau2 reward. This was a pipeline canary, not an efficacy
  experiment.

## Fastest defensible next path

1. Run and record the free token reconciliation added in `c24e838`.
2. Run the Fireworks integer-ID scoring compatibility probe against the exact
   deployed model and `/completions` route. `score_ids` sends the locally
   rendered prefix and payload as integer token IDs, so Fireworks applies no
   chat template in this path. The paid probe must verify exact echoed token-ID
   runs, ID-to-token decoding where exposed, logprob indexing, endpoint/model
   identity, and refusal on any mismatch. It must not claim to verify a
   Fireworks-rendered chat template, because no server-side chat rendering
   occurs.
3. Wire the accepted projection into the real scoring path: native DeepSeek
   rendering, per-payload scoring/alignment, and mapping back to original Qwen
   token indices. Tool serialization remains excluded in version one.
4. Dry-run the fully wired scorer over all 31 rows without teacher calls and
   require complete included/excluded accounting and zero control-token weight.
5. Re-score the same archive only after parity and the offline wiring gate pass.
6. Train exactly one step from untouched `A_sft_new`, reload it, and run fixed
   format/log-probability probes plus one fresh Tau2 episode.
7. If reasoning and tool formatting survive, complete the second update. Only
   then start the 5 updates x 8 episodes signal-seeking pilot.

Do not switch to GOLD for this canary. GOLD is a distinct distribution-alignment
method requiring a larger implementation and more teacher-logit information;
it also does not remove the need to establish semantically equivalent native
renderings. Do not introduce an arbitrary advantage clamp before invalid spans
are removed. If valid semantic advantages remain heavy-tailed afterward, a
recorded robustification rule can be evaluated separately.

## Evidence locations

- Archived run copy: `/tmp/live-opd-proof2.VOen1l/two_update_proof_20260828_182302`
- Token analyzer: `vektori_trace/tau2/live_token_classes.py`
- Analyzer entry point: `scripts/tau2_live_opd_modal.py::classify_advantages`
- Advantage implementation/statistics: `vektori_trace/chunk_opd.py`
- DeepSeek renderer: `vektori_trace/encoding_dsv4.py`
- Cross-tokenizer scoring path: `vektori_trace/providers/teacher/cross.py`
