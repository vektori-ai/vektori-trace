# SOL handoff: Tau2 live cross-tokenizer OPD

**Updated:** 2026-08-29 (Asia/Kolkata)  
**Branch:** `feat/tau2-live-opd`  
**HEAD at handoff:** `ee2b12eafef5aa39bad2589995c6ee6bf994de64`  
**Reported full suite:** 1,957 passed, 2 skipped, 0 failed (221 seconds)

This replaces the old continued-SFT handoff. `A_sft_new` already exists and is
the untouched parent for the live-OPD canary. The current task is to validate
the repaired live semantic-scoring path, then run the smallest paid OPD proof.

## Current decision

Do **not** run the ~$0.04 archive re-score yet. Three free gates remain:

1. Drive all 31 archived actions through the real production
   `train_live_update()` path with the real DeepSeek tokenizer, a fake teacher
   pool, and a fake trainer.
2. Construct a fresh canary directory from immutable `SAMPLED` artifacts only;
   never resume or mutate the failed evidence run.
3. Make the 5x8 pilot's task-share gate validate the preregistered planned
   roster rather than deriving permission only from tasks that survived.

Gate 3 is required before the pilot. Gates 1 and 2 are required before the
31-action re-score.

## What happened in the first live proof

Run `two_update_proof_20260828_182302` completed update 0:

- four Tau2 episodes, 31 captured actions, no capture failures;
- 31 finite DeepSeek scores;
- one optimizer step, loss `0.731062`, gradient norm `0.730949`;
- 14,178 supervised tokens, peak training memory about 12.9 GiB;
- all 504 LoRA tensors moved;
- child adapter hash `7d5ed1a75bccf482`, different from parent
  `3869b147ab7ce5d2`;
- reload verified by a fixed logprob-probe delta of `0.03607`.

The next-policy rollout genuinely ran. One episode completed; three opened
`<think>`, emitted non-empty reasoning, and reached a tool call without closing
`</think>`. The reasoning gate refused the next update.

This proved the rollout -> score -> train -> checkpoint -> reload -> next
rollout machinery. It did not validate the old cross-model scoring question.

## Root defect and proof

The old live path sent the complete raw Qwen action to DeepSeek scoring,
including Qwen control and tool serialization. The pinned upstream reference
requires response-content-only alignment with no chat-template special tokens
and translates markers between model families.

The archived token-level signal was mixed, not uniformly negative:

- 3,388 positive advantages;
- 10,713 negative advantages;
- 77 zero advantages.

Negative action means are expected because for student-sampled actions:

`E[log p_teacher - log p_student] = -KL(student || teacher) <= 0`.

The defect was where the extreme signal landed:

| Class | Tokens | Mean advantage | Minimum |
|---|---:|---:|---:|
| markup | 80 | -24.469495 | -55.289224 |
| reasoning | 11,707 | -0.599669 | -24.499999 |
| visible content | 1,760 | -0.702808 | -27.057084 |
| tool JSON | 631 | -0.276892 | -13.249996 |

Exact structural examples:

| Token | n | Mean | Range |
|---|---:|---:|---:|
| `<think>` | 31 | -40.644954 | -45.999950 to -35.749858 |
| `</think>` | 31 | -0.220915 | -3.359367 to +1.076188 |
| `<tool_call>` | 16 | -43.158541 | -55.289224 to -0.014444 |
| `<|im_end|>` | 17 supervised occurrences | -19.665078 | -27.057084 to -0.228742 |

Therefore the teacher was judging Qwen's private serialization as well as the
response meaning. The direct cause of the later missing `</think>` is not
proven: its own advantage was mild. Structural damage as collateral movement
is plausible but remains a hypothesis until an ablation or gradient-attribution
measurement establishes causality.

## Repair now in the branch

The repair adds no tokenizer, parser, or alignment algorithm. It reuses:

- `live_agent.split_generation()` for Qwen action parsing;
- `encoding_dsv4`/`render_teacher_prefix` for native DeepSeek rendering;
- `align.align_by_bytes()` for the existing dual-pointer byte alignment.

New components:

| Component | Purpose |
|---|---|
| `tau2/live_projection.py` | Marks transferable reasoning/content token spans and explains every exclusion. |
| `tau2/live_score.py` | Scores the structured DeepSeek-native action and maps payload credit back to Qwen token indices. |
| `tau2/live_batch.py` | Builds training advantages from already-projected per-token credit without raw-action re-alignment. |
| `tau2/live_token_classes.py` | Reconstructs archived advantages by token class. |
| `tau2_live_opd_modal.py` inspectors | Projection, reconciliation, dry-run, and historical masked-preview reports. |
| `tau2_fireworks_parity_probe.py` | Verifies Fireworks integer-ID scoring compatibility on the production `/completions` route. |

Version 1 supervises reasoning payload and visible content. It excludes Qwen
tool markup, tool JSON, template markers, and boundary-straddling tokens. Tool
semantics are intentionally deferred because Qwen JSON and DeepSeek DSML do
not share a defensible byte mapping.

Production `train_live_update()` now calls `run_projected_score_stage()` and
`run_projected_train_stage()`. It no longer calls the raw-action
`score_replay_batch()` path. A test monkeypatches that legacy scorer to raise
and drives the real live driver.

Commit `98abea3` also fixes ambiguous payload location: visible content is
located after the reasoning span, and repeated candidate matches are refused.
This covers a model previewing its eventual answer inside reasoning.

## Verified accounting and probes

The original raw batch is fully reconciled:

`14,206 raw = 13,417 projection-supervised + 761 newly excluded + 28 prior sentinels`.

The 761 newly excluded supervised tokens are:

- 631 tool-JSON tokens;
- 80 markup tokens;
- 50 conservative boundary/edge tokens.

The 28 were already zero-advantage sentinels from over-long chunks or
unaligned tails. No token is unaccounted, and the projection never revives a
token the old optimizer refused.

Offline eligibility result:

- 31/31 actions project;
- 13,417/14,206 raw tokens retained (94.45%);
- zero markup or tool JSON supervised;
- zero projection failures.

Paid Fireworks compatibility probe:

- exact model `accounts/fireworks/models/deepseek-v4-flash-0731`;
- production `POST /completions` integer-ID route;
- echoed IDs match;
- requested/scored lengths match, including the 59-token tool case;
- all returned logprobs finite;
- no duplicated thinking boundary in the locally rendered structured action.

This probe does not compare a Fireworks chat template: integer IDs are supplied
directly, so the server applies no chat template.

The historical mask preview (`mean -0.733 -> -0.580`, `p1 -9.18 -> -6.82`) is
diagnostic only. Those retained advantages were purchased under the invalid old
context. They are not post-fix values, not a lower bound, and cannot establish
that visible content was correctly discriminated. Real post-fix advantages
remain unknown.

## Exact remaining free work

### Gate A: production-path 31-action dry run

Create a test/inspector that invokes the real `train_live_update()` with:

- all 31 archived action rows and semantic histories;
- the pinned real DeepSeek tokenizer;
- a deterministic fake pool returning finite logprobs;
- a fake trainer/checkpointer;
- `score_replay_batch` monkeypatched to raise.

Require all of the following:

- 31/31 actions reach projected scoring;
- no payload is skipped;
- every action retains nonzero supervision;
- global token accounting is exact;
- zero markup/tool serialization weight in the actual `TurnAdvantages`;
- semantic score rows persist with fingerprints;
- `RunState.validate()` succeeds;
- a resume makes zero fake-teacher calls;
- no raw-action scorer is touched.

### Gate B: fresh canary construction

Create a new run ID rooted at untouched `A_sft_new`. Copy only immutable
sampling evidence needed to reproduce update 0:

- `actions.jsonl`;
- `rendered.json`;
- episode/task metadata and the valid `PLANNED`/`SAMPLED` state needed by the
  driver;
- a fresh manifest that records source-run provenance and the untouched parent
  hash.

Do not copy:

- old `scores.jsonl`;
- `SCORED` or `TRAINED` markers;
- report/checkpoint/optimizer/scheduler state;
- the damaged child adapter.

Fail if the destination already contains paid or trained artifacts. Preserve
the original failed run unchanged as evidence.

### Gate C: task-share policy

The current training code derives a 1.5x-uniform threshold from tasks present
in the assembled batch. That is acceptable for an explicitly single-task
diagnostic but cannot detect a preregistered pilot task that disappeared before
training. Before 5x8, compare the sampled batch against the planned task roster
and its intended counts. Missing tasks and excess concentration must fail.

This policy decision does not need to block the 31-action archive re-score if
that fixed archived batch contains its expected four tasks, but it must be
resolved before fresh pilot rollouts.

## Paid sequence after Gates A and B

1. Re-score the 31 archived actions through the production projected scorer
   (estimated teacher cost about $0.04).
2. Inspect and save the real post-fix advantage distribution by payload class,
   tails, finite values, supervision coverage, and cost.
3. Decide whether a clamp is justified only from these valid new scores. Do not
   introduce an arbitrary clamp to hide invalid spans.
4. Run exactly one optimizer step from untouched `A_sft_new`.
5. Save and hash the adapter; reload it; prove a fixed logprob probe changes.
6. Run strict format probes and one fresh Tau2 episode from the reloaded policy.
7. If complete reasoning and valid tool calls survive, run the second update.
8. If the two-update proof passes, resolve Gate C and run the 5 updates x 8
   episodes signal-seeking pilot.

The 5x8 pilot validates pipeline stability and seeks a learning signal. It is
not powered evidence of efficacy.

## Stop conditions

Stop before another paid request or optimizer step on any of:

- payload parsing/location ambiguity;
- byte-alignment mismatch or payload skip;
- missing/non-finite score;
- markup/tool serialization receiving weight;
- stale or mismatched score fingerprint;
- missing parent adapter hash;
- partial batch or zero supervised tokens;
- unverified checkpoint reload;
- unterminated reasoning or malformed tool call after the update;
- unexpected task-roster composition for the pilot.

## Operational context

- AWS box: EC2 `i-0a348ff3d7be9769a`; use SSM and execute repository commands
  as `ubuntu` to avoid Git ownership errors.
- Box checkout: `/data/vektori-trace`.
- Branch: `feat/tau2-live-opd`.
- Parent adapter: `/adapters/tau2/runs/a_sft_new_ck35_r2/checkpoint-32`.
- Base model: `Qwen/Qwen3-4B`; LoRA rank 16.
- Teacher and user simulator use Fireworks; no OpenAI key is required.
- Tau2 is pinned in the Modal image at commit `f8de30c`; domain data is staged
  separately with `TAU2_DATA_DIR`.
- Do not allocate a GPU or run a paid teacher call without explicit approval.
- Do not mutate the original failed proof directory.

## Evidence and companion documents

- Full learning record: `docs/TAU2-LIVE-OPD-LEARNINGS-2026-08-29.md`
- Deep-dive plan: `docs/TAU2-OPD-DEEP-DIVE.md`
- Parallel Claude handoff: `docs/CLAUDE-HANDOFF.md`
- Pinned reference implementation: `vektori_trace/opd_reference/`
- Original evidence-run copy used locally:
  `/tmp/live-opd-proof2.VOen1l/two_update_proof_20260828_182302`

## Truthful status at handoff

```text
first live loop execution              proven
old scoring defect                     proven
token accounting                       reconciled
semantic projection                    built
production score/train integration     built and unit-tested
payload ambiguity guard                built and tested
Fireworks integer-ID compatibility     paid probe passed
31-action real production dry run      OPEN
fresh untouched-parent canary dir      OPEN
real post-fix teacher advantages       UNKNOWN
post-fix optimizer step                NOT RUN
post-fix reload and Tau2 rollout       NOT RUN
5x8 pilot                              NOT RUN
```
