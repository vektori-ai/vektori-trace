# Planted-deficit recovery — first real run

**Date:** 2026-07-25 · **Model:** `gpt-5-nano` · **Command:** `vektori-trace selftest --repeats 3`
**Raw artifacts:** [`selftest-run.md`](selftest-run.md), [`selftest-run.json`](selftest-run.json).
Corpora are seeded, so every one is regenerable from `(config, seed)` — nothing
below is derived from anything not reproducible.

## The question

> Can the ranker recover a capability deficit we planted ourselves?

If not, nothing downstream matters: task selection, training, and the A3-vs-A2
comparison all assume the diagnosis names a real capability.

## The answer: yes

| | |
|---|---|
| Capability **named** (proposed at all) | **27/27 runs** — and every match was `strict` |
| **Recovered** where the ceiling allows it | **11/15 = 73%** |
| Recovered where the ceiling is 0% | 0/12 — as designed |
| Labeller accuracy vs. known ground truth | **87.8%** (min 50%, max 100%) |

9 configs × 3 repeats. 4 of the 9 configs are unrecoverable by construction at
the default thresholds, so the live rate is only meaningful on the other 5.

## Three findings worth carrying into step 5

### 1. The labeller's error is directional, and it attenuates the gap

Confusion over all 27 runs, for the planted capability:

| truth → predicted | count |
|---|---|
| NA → NA | 128 |
| PRESENT → PRESENT | 119 |
| LACKING → LACKING | 85 |
| **LACKING → PRESENT** | **29** |
| LACKING → NA | 9 |
| PRESENT → NA | 7 |

Of 123 traces that genuinely lacked the capability, only 85 (**69%**) were
labelled LACKING. The dominant error is LACKING read as PRESENT — the labeller
crediting an agent with a capability it did not exercise.

That error only ever moves one way. It lowers the incident rate in losses,
which shrinks the measured gap. A true gap of 1.0 measures around 0.69 through
this instrument. This is the attenuation the plan predicted on principle, now
with a number on it, and it is the argument for `min_gap = 0.20` being
generous rather than strict — the threshold is not the binding constraint at
this blur level, but a real deficit with a true gap of 0.3 would measure ~0.21
and sit right on the line.

### 2. Support, not gap, is what rejects things today

Every 0%-ceiling config fails the same way: the planted capability ranks
**first with a gap of 1.0** and is rejected for want of a third relevant trace.
Both floors bind independently —

- ≥3 wins that *exercise* the capability (a 3-win corpus has ~2 after the
  clean-win share), and
- ≥3 losses that *lack* it, i.e. `n_losses × prevalence ≥ 3`.

That second inequality is the concrete number v1.1 needs when it tells a
customer how many more traces to send.

### 3. Three repeats cannot distinguish 67% from 100%

Four of the five recoverable configs scored 67% (2 of 3) and one scored 100%,
including at the *lowest* prevalence — which is not a plausible ordering and is
just noise. The failures that did occur were split `outranked_by_distractor`×2
and `top_ranked_but_below_threshold`×2, i.e. two labelling misses and two
calibration misses, with no proposer misses at all.

Anything that wants to compare recovery rates between configurations needs more
repeats than this. As a pass/fail on "does the ranker work", 3 is plenty.

## What this run also caught

The first live run returned **0% recovery, `not_proposed`** — the proposer had
named seven task domains ("Cloud Object Storage Upload", "Release Branch
Push") instead of capabilities. Blinding it to outcome in step 1 had left it
with only the opening request and a turn count, and there is no behavioural
signal in that. Fixed by showing it condensed trajectories; recovery on the
same corpus and seed went 0% → 100%.

That is the self-test doing its job, and it is the reason step 2 runs before
anything that needs Docker.
