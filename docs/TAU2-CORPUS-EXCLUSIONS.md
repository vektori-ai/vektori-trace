# Tau2 retail corpus — named exclusions

Every task dropped from the retail training pool, with its cause and the
evidence needed to re-check the decision. Two tasks are excluded; both are
recorded here rather than summarised as tail statistics, so a reader can
disagree with either without re-running the audit.

Corpus: `/data/tau2/artifacts_16384`
Manifest: `b741bfceb1f3d027` · tools `1881b37265759ea3` · `max_length=16384`

## Task 51 — length exclusion

| | |
|---|---|
| Cause | rendered row exceeds the 16,384-token pin |
| Trace | `7fc926516a3b867f`, `flash_retail_p1.json` |
| Decisions | 9 total, **2** over the cap |
| Largest row | 81,274 tokens |
| Comparison | corpus p99 is 12,556; this is **6.5x** the p99 |

One assistant message in this trace is ~72,800 tokens — a runaway generation,
not a long conversation. The episode still passed Tau2's reward, which is why
reward alone does not establish that a trace is usable.

Excluded whole, not partially: section 5 requires the exact rendered history to
fit, and a trajectory missing two of its nine decisions is not the trajectory
the teacher produced. Raising the cap would not rescue it — at 32,768 the same
two rows still overflow.

## Task 54 — policy exclusion

| | |
|---|---|
| Cause | `policy_no_mutation_after_error` |
| Gate | a mutating tool ran while an earlier tool error was unresolved |
| Evidence | call/result/mutation ids in `eligibility_report.json` under `per_task["54"].evidence` |

The evidence records the failing call id, its error result, and the id of the
mutation that followed, so the exclusion is auditable without re-running the
audit. If manual review finds the error was in fact resolved, the gate is
wrong and should be corrected rather than the task force-included.

## Not exclusions

- **Confirmation diagnostics.** `diagnostic_confirmation` is not a gate. An
  earlier version gated on it and rejected 64/73 traces; inspection showed it
  was rejecting compliant sequences because its proposal taxonomy conflated
  overlapping mutation categories. Those tasks are in the manual-review queue,
  not excluded.
- **Tasks 57, 73, 75, 93.** Reserved to S16 as contaminated diagnostics, before
  train60 is chosen. Reserved, not excluded.

## Effect

```
73 tasks with a passing trace
 -1 task 54   policy gate
 -1 task 51   length
 = 71 fully eligible
 -4 contaminated diagnostics reserved to S16
 = 67 train60 candidates, of which 60 are used
```

Seven spare. No further DeepSeek collection is required.

## Context policy

Training rejects any row over 16,384 and drops its whole trace. Serving needs a
separate rule, because a live episode grows turn by turn:

```
serving max_model_len   16384
generation reserve       2048
effective prompt budget  14336
```

Measured final-turn prompts: retail p90 11,942 · airline p99 11,866 · telecom
p99 11,876. Generation need is p99 ≤ 344 across all three domains, so the 2,048
reserve carries roughly 6x headroom.

At 16,384 with that reserve, **0 airline and 0 telecom episodes exceed**, and
the single retail episode that does is task 51 — already excluded. 32,768 buys
no additional coverage in any domain and doubles KV cache per request.

An episode that would exceed the budget must be **stopped and recorded as
`context_exceeded`**, an infrastructure outcome, never graded as a model
failure, and never truncated mid-episode.
