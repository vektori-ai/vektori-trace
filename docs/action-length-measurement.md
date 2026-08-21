# Action-length distribution, and the cap that follows from it

Measured 2026-08-21 on the box. 2,978 assistant actions from 40 of the 117
passing DeepSeek trajectories in `/data/vektori-out/dsv4-corpus60{,-b}`, token
counts under `Qwen/Qwen3-14B`'s tokenizer (the student's, since the cap binds
the student). `thinking + content`, because both are sampled tokens.

## Distribution

| statistic | tokens |
| --- | --- |
| median | 605 |
| mean | 1,040 |
| p90 | 2,481 |
| p95 | 3,209 |
| p99 | 5,374 |
| p99.9 | 8,126 |
| max | 8,842 |

## Truncation by cap

| cap | actions cut mid-sequence |
| --- | --- |
| **256** (the previous run's default) | **2,062 / 2,978 = 69.2%** |
| 512 | 1,619 = 54.4% |
| 1,024 | 1,100 = 36.9% |
| 2,048 | 481 = 16.2% |
| 4,096 | 62 = 2.1% |
| 8,192 | ~0 (above p99.9) |

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

**8,192, provisionally.** Above p99.9 (8,126) but **below the observed max of
8,842** — so it can still truncate a real action, and calling it "effectively
above the max" would be wrong. It is a loop guard rather than a length budget:
run6 recorded models repeating identical output up to 21 times, and an uncapped
degenerate sample generates until context exhaustion.

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
- **40 traces, not 117.** The sample was bounded for speed. The tail is where
  the cap decision lives, so the full 117-trace pass must confirm p99.9 (and the
  max) **before** the cap is fixed for a paid run. Until then 8,192 is
  provisional.

## Reproducing

CPU only, no endpoint, no spend — reads trajectories off disk and tokenizes
locally. See `scripts/measure_action_lengths.py`.
