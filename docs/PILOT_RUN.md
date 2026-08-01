# Pilot run — the simple version

**Date:** 2026-08-01 · run it with [`scripts/pilot.sh`](../scripts/pilot.sh)

The long version is [`CASE_STUDY.md`](CASE_STUDY.md). This is what we actually
do first.

---

## The setup

| | |
|---|---|
| **Repo** | `prefecthq/prefect` |
| **PRs** | merged since 2025-01-01, has a linked issue, touches a test file |
| **Teacher** | DeepSeek-V4-Flash |
| **Student** | Qwen3-8B |
| **Rollouts** | 4 (smoke) then 8 (pilot). No escalation to 32. |
| **Tasks** | ~10 (smoke) then ~50 (pilot) |

**One task** = prefect at the commit before the PR, git history scrubbed, the
linked issue as the instruction, the PR's new tests hidden as the grader, the
real merged patch kept as the oracle. The agent's patch is graded in a fresh
container it never touched. Pass or fail, no partial credit.

## The run

```bash
./scripts/pilot.sh smoke     # ~100 PRs -> ~10 tasks, n=4.  Does it run at all?
./scripts/pilot.sh pilot     # ~500 PRs -> ~50 tasks, n=8.  The result.
```

Set `STUDENT_API_BASE` to the vLLM server holding Qwen3-8B, and `TEACHER_MODEL`
if the DeepSeek model string differs.

## The answer we're looking for

Three counts, and that's the whole pilot:

| bucket | meaning | what it's for |
|---|---|---|
| **already fine** | student solves it | nothing to do |
| **TRAINABLE** | student fails, teacher solves it | **everything trainable lives here** |
| **skip** | neither solves it | drop it |

`pilot.sh` prints these. If **TRAINABLE** is a healthy slice, there's a case
study. If it's near zero, we found that out for ~800 rollouts instead of a GPU
cluster.

## Two things to know, not to fix now

**n=8 can't split RL from OPD.** A task at 0/8 only tells you the true pass rate
is under ~30%, so "never passes" and "rarely passes" look identical. That split
is what chooses RL vs OPD. Fine for a pilot — just don't publish a routing claim
off n=8. It needs n=32 on the zeros.

**Cross-tokenizer OPD is PR #22, and it isn't live-tested.** DeepSeek teaching a
Qwen student normally can't work: OPD has the teacher score the exact token ids
the student sampled, and different tokenizers make those ids mean different
things. #22 implements the bridge for exactly this pair and is green offline
(722 tests). Its own body lists what's still open — **P0** live echo to V4-Flash,
**P3** provider tokenizer drift, **P4** FP8 magnitude, **P7** a real 40-step
trajectory. P7 is the one `PLAN.md` warned about when it called cross-tokenizer
distillation *"real but unvalidated on 40-step agentic trajectories."*

None of that blocks this pilot — steps above don't train anything. Run
`probe-teacher --echo` against the live endpoint before relying on #22.

## If the smoke run emits zero tasks

Read `cs/smoke/mined/mine-report.json`. The skip histogram names the reason.
The two likely ones:

- **`no_fail_to_pass` dominating** — prefect sets `filterwarnings = ["error"]`,
  so a dependency released after a base commit turns that commit's suite red for
  unrelated reasons. Loosen the test command or narrow the PR window.
- **dependency drift** — one bootstrap image at HEAD doesn't match a base commit
  from 18 months ago. Narrow the window to recent months.

## After the pilot

Only if TRAINABLE looks healthy:

1. `replay` + `diagnose` on the trainable pile → the top 3 capability gaps
2. `diagnose --prove` → the generated environment + validity proof
3. That's deliverables 1, 2, 3 and 5 — publishable with no training at all

Training is step 4 and it's a separate decision.
