# Action-length distribution, and the cap that follows from it

Measured 2026-08-21 on the box. **All 7,677 assistant actions from all 117
passing** DeepSeek trajectories in `/data/vektori-out/dsv4-corpus60{,-b}`, token
counts under `Qwen/Qwen3-14B`'s tokenizer (the student's, since the cap binds
the student). `thinking + content`, because both are sampled tokens.

## Distribution

| statistic | full (117 traces) | first pass (40 traces) |
| --- | --- | --- |
| actions | **7,677** | 2,978 |
| median | **534** | 605 |
| mean | **956** | 1,040 |
| p90 | 2,384 | 2,481 |
| p95 | 3,172 | 3,209 |
| p99 | 5,185 | 5,374 |
| p99.9 | **7,557** | 8,126 |
| max | **8,842** | 8,842 |

The full pass is slightly kinder than the 40-trace sample — p99.9 fell from
8,126 to 7,557 while the max held at 8,842, so the tail is a handful of
outliers rather than a heavy shoulder.

## Truncation by cap

Over all 7,677 actions:

| cap | actions cut mid-sequence |
| --- | --- |
| **256** (the previous run's default) | **5,238 = 68.2%** |
| 512 | 3,922 = 51.1% |
| 1,024 | 2,526 = 32.9% |
| 2,048 | 1,046 = 13.6% |
| 4,096 | 159 = 2.1% |
| 8,192 | 2 = 0.03% |
| **9,216** (chosen) | **0** |

## Why this is a finding and not a parameter note

`FireworksOPDConfig.max_new_tokens` still defaults to 256
(`providers/student/fireworks.py`). The previous OPD run — the one that
produced 0/13, unchanged from baseline — ran under it. At that cap **more than
two thirds of actions were cut mid-sequence**, and the teacher then scored those
fragments as though they were completed actions.

That failure is invisible by construction: truncated bytes still align, the
chunk loss is still finite, and every downstream metric looks normal. Nothing in
that run's logs would have shown it.

It is **one severe cause among several**, not the identified largest. The others
are independently sufficient to explain 0/13: ~200 graded sequences against the
~38,400 of the Thinking Machines recipe, batch 4 against 256, no cold-start SFT
while the student sat outside the teacher's support, and a legacy span-surrogate
loss that is not the published objective. Nothing here isolates their relative
contributions, and this measurement cannot.

This is why `chunk_opd.assert_token_cap_is_task_derived` refuses anything at or
below 256 and `validate_cross_opd_config` calls it before a rollout starts,
rather than leaving the default to be noticed by a reader.

## The cap to use

**9,216.** Clears the observed max (8,842) outright: **no** action in the
corpus would have been truncated by it. 8,192 was the earlier candidate and
still cut 2 of 7,677 — small, but the run fails closed on a cap hit, so those 2
would abort a batch if drawn. The extra ~1k tokens buys that away for nothing.

It remains a loop guard rather than a length budget: run6 recorded models
repeating identical output up to 21 times, and an uncapped degenerate sample
generates until context exhaustion.

The replay run must **fail closed** on any cap hit rather than recording one:
a truncated action is a fragment the teacher would grade as complete, which is
the failure this whole measurement exists to prevent. `--allow-truncated` exists
for diagnosis only.

Two caveats, both real:

- **These are DeepSeek's lengths, not ck75's.** ck75 may be terser or more
  verbose. The serving probe reports `cap_hit_rate`
  (`replay_sample.summarize_cap_hits`); a non-zero rate at 8,192 means this
  number is still wrong for the student and must be re-derived from its own
  samples.
- **The full 117-trace pass is done** and is what the numbers above report. The
  earlier 40-trace figures are kept only to show the two agree.

## Reproducing

CPU only, no endpoint, no spend — reads trajectories off disk and tokenizes
locally. See `scripts/measure_action_lengths.py`.
