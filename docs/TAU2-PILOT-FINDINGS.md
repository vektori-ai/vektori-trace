# Tau2 live-OPD pilot — findings

Run `pilot_10x8_20260829d`, continued as `pilot_10x8_20260829e` after a
preregistered slot retry. Plan `aa9251ccb6d566fa`, 80 frozen (task, seed)
pairs. Contract: parser v3, projection v4, scoring chunk-v2.

**Status: exploratory continuation.** A preregistered stop rule fired at
update 2 and was amended on the evidence (§4). This document records what the
run established, what it cannot claim, and where a threshold was moved after
seeing the data that moved it.

---

## 1. What the loop established

The live cross-tokenizer OPD loop runs end to end and is verifiably on-policy:

```text
rollout -> free dry-run -> paid scoring -> optimizer step
        -> checkpoint -> reload -> next rollout on the new weights
```

| | Update 0 | Update 1 |
| --- | ---: | ---: |
| Episodes | 8/8 sampled | 8/8 sampled |
| Actions | 75 | 77 |
| Retention | 91.0% | 92.68% |
| Teacher cost | $0.1187 | ~$0.13 |
| Loss | 0.5311 | 0.5668 |
| grad_norm | 0.6398 | 0.6276 |
| Parent -> child | `3869b147…` -> `073237e3…` | `073237e3…` -> `7161d364…` |
| Tensors matched | 504/504 | 504/504 |
| `max_logit_delta_vs_base` | 2.625 | 2.703 |

Every update-1 episode carries update 0's child hash, asserted at rollout via
`--adapter-hash-expect`. That is the on-policy claim, verified rather than
assumed.

## 2. The `Okay` measurement

Preregistered before update 0 was trained: *"`Okay` frequency or confidence
declines while boundary validity stays intact."*

| | Update 0 | Update 1 |
| --- | ---: | ---: |
| Reasoning turns beginning `Okay` | 100.0% (75/75) | 100.0% (77/77) |
| Median behaviour logprob(`Okay`) | −0.01057 | −0.01067 |
| Boundary validity | 75/75 | 77/77 |
| Another opener replacing it | none | none |

**The prediction failed.** Neither frequency nor confidence declined.
Frequency was already saturated at 100%, so that half was unfalsifiable —
worth recording as a defect in how the prediction was written.

From update 1's scored advantages (47,781 supervised tokens):

```text
mean   -0.575     median -0.024     stdev 1.447
range  -22.42 … +4.34        positive/negative 14,135 / 33,646
Okay:  82 tokens = 0.17% of supervised, 3.67% of total |advantage|
       median advantage -15.52, and 10/10 of the most-negative tokens
```

So `Okay` is amplified roughly 21x relative to its token share — the
`A_i = (L_T/L_S − 1)·log p_i` form does blow up on near-deterministic
tokens — but at 3.67% it does **not** dominate the update. No clamp was
justified and none was added. The most-positive tokens (` transfer`,
` policy`, `Wait`, `Let`, ` might`) are semantically meaningful reasoning
forks, which is the healthy sign.

**Caveat:** updates 0 and 1 sample different tasks, so an unchanged logprob
distribution across different contexts is not a matched pre/post measurement.
Absolute advantage mass is also only a proxy for gradient influence.

## 3. Two reasoning-boundary findings

### 3.1 Hermes markup inside the reasoning span

The model narrates executable tool calls inside `<think>`, then repeats them
outside where they actually execute:

```text
<think>
reasoning …
<tool_call>{…}</tool_call>     <- narrated inside the thought
</think>
<tool_call>{…}</tool_call>     <- repeated, and executed
```

The episodes therefore **function correctly**. Only the reasoning payload is
unmappable: Hermes markup has no DeepSeek DSML counterpart, so student and
teacher payload bytes disagree and the payload is skipped whole.

| Update | Episodes | Actions | Reasoning bytes excluded | Retention |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1/8 | 2/75 = 2.67% | — | 91.0% |
| 1 | 2/8 | 4/77 = 5.19% | 5.81% | 92.68% |
| 2 | 3/8 | 3/78 = 3.85% | 4.31% | 93.47% |

**Breadth rose while mass fell.** The task-76 concentration that explained
updates 0–1 is dead: update 2's roster contained no task 76 and the behaviour
still appeared in three episodes (tasks 102, 4, 64). It is associated with
multi-order / parallel-tool actions.

An earlier reading of this run reported 1/75 and 1/77 ("1.3%, unchanged") by
taking the dry-run's `payload skips` counter as the incidence rate. That
counter is narrower than the number of actions carrying markup in the
reasoning span; the corrected figures are above.

### 3.2 First cap termination

Update 2 attempt 1: `u002-task98-seed0` hit the 4096-token cap at turn 3
(`finish_reason=length`). The raw generation is a **reasoning loop** — 15,600
characters, 17 loop markers (`Wait`, `Let me check`, `Looking back`), no tool
call ever emitted. The user asked for three exchanges "in the same order" when
the items span two orders; the model detected the contradiction and re-derived
it indefinitely instead of asking or calling `get_order_details`.

```text
update 0:           cap failures 0/8
update 1:           cap failures 0/8
update 2 attempt 1: cap failures 1/8
update 2 attempt 2: cap failures 0/8   (retry_slot: 7 preserved, 1 re-rolled)
```

The retry resolved the same ambiguity by calling tools and finished in ~10
turns, which favours a stochastic reading over policy degradation — on one
data point.

The cap fired as designed and **was not raised**: a truncated action is a
fragment, and training on fragments is what produced the earlier 0/13 run.

### 3.3 A plausible common mechanism

Both findings are reasoning-boundary failures, and OPD gives `</think>` zero
positive weight by construction — markup carries no credit, so there is no
positive target for the closing tag. Three updates of pressure on reasoning
content with no reinforcement of the boundary is a coherent story for drift.
It is a hypothesis, not a finding: n=3, eight episodes each, disjoint rosters.

## 4. The stop rule that fired, and the amendment

The frozen rule was *stop if interleaved-tool exclusions exceed 2/8 episodes*.
Update 2 hit 3/8 and the chain halted before any teacher call.

Amended 2026-08-30, **before update 2 was scored**: episode spread becomes a
diagnostic; the binding rules are mass-based —

```text
stop if affected actions        > 10%
stop if excluded reasoning bytes > 10%
stop if retention               < 90%
stop on any undeclared exclusion class
stop on repeated cap terminations  (never raise the cap)
```

Rationale: one affected action marks an entire episode, so the metric measures
breadth and is blind to mass — and mass fell while breadth rose.

**This threshold was still changed after seeing the data that tripped it.**
That is why this run is labelled an exploratory continuation rather than a
clean preregistered result. The conservative skip is unchanged: affected
reasoning payloads stay excluded, no Hermes is converted to DSML, and no
contaminated reasoning is sent to the teacher mid-pilot.

## 5. What this run cannot claim

No efficacy claim. Three updates of eight episodes, no matched continued-SFT
control, and batches conditioned on format validity. Reward moved 3/8 (update
0) to 1/8 (update 1) on **different tasks and seeds** — recorded, not
interpreted.

Do not size an extension from external rollout counts: correlated multi-turn
Tau2 turns are not comparable to independent single-turn rollouts.

## 6. Defects found and fixed during the run

| Defect | Consequence if unfixed |
| --- | --- |
| Manifest labelled parser v2 / projection v3 | Contract misdescribed; scores were bought under v3/v4 |
| Exclusion declared per-episode, invisible to `rescore` | Update 1 failed its own preregistration **after** the teacher was paid |
| Chain: unchecked volume downloads | A stale `/tmp` file from a previous update could be verified instead |
| Chain: dry-run exit code ignored | A crashed dry-run could pass on stale log text |
| Chain: 10% rule used the wrong denominator | Excluded *tokens* over *all* student tokens, not reasoning bytes over reasoning bytes |
| Chain: structural check `\|\|`-chained to the newline check | A clean newline satisfied missing structural evidence |
| Chain: score coverage accepted `matched > 0` | 70 scores for 77 actions would have passed |
| Chain: teardown failure only warned | The wrapper returned success while an L40S kept billing |
| Chain: `.TRAINED` skipped on the marker alone | An unverified checkpoint would be trusted |

`tests/test_pilot_chain.sh` (17 cases) drives the real scripts against a fake
Modal and found a genuine `set -u` crash that predicate tests could not.

## 7. The long-term fix for interleaved tools

Not attempted here, deliberately. A semantic event IR —
`ReasoningSegment / ToolCall / ToolResult / FinalContent` — rendered
independently through Qwen's and DeepSeek's native templates, with monotonic
span location and proof that each scored segment's teacher prefix carries the
same prior semantic events. That is a separate projection version with its own
canary and manifest.
