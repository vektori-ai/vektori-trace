# First real mining run — `hynek/structlog`

**Date:** 2026-07-26 · harbor 0.20.0 · Docker · `--limit 40`

The v0 plan's last step-3 item: *"mine one small repo, print the skip-reason
histogram, and hand-inspect three tasks."* This is that run. Raw report:
[`mine-structlog.json`](mine-structlog.json).

## Why structlog

Pure Python, no compiled dependencies, and a 928-test suite that runs in 2.3
seconds — so one bootstrap image builds every base commit in recent history, and
the two validation runs per PR cost seconds rather than minutes.

The binding constraint was **linked issues**, not size. `require_linked_issue`
defaults on, because the linked issue is what supplies a problem statement
written *before* the fix existed; without one the PR body is the only source and
it describes the solution. Of the last 60 merged PRs:

| repo | with a linked issue |
|---|---|
| `python-jsonschema/jsonschema` | **0** |
| `python-attrs/attrs` | 7 |
| `hynek/structlog` | **14** |

jsonschema would have yielded nothing at the defaults. Worth checking before
spending a bootstrap on a repo.

## Where 40 candidate PRs went

```
emitted                       4
no_test_patch                20
non_bug_pr                    6
no_new_test_funcs             6
no_fail_to_pass               4
```

**A 10% yield, and half the loss is one filter.** `no_test_patch` (20/40) is PRs
that changed source without touching tests — unminable by construction, since
the test diff *is* the verifier. That's a property of the repo's habits, not a
bug in the filter, and it sets the arithmetic for step 4: **~400 merged PRs to
reach 50 tasks** at this rate.

`no_fail_to_pass` (4) is the interesting one — those PRs passed every filter,
built, and ran the suite twice, and the F2P set still came out empty. Those are
the ones to inspect if the yield needs raising.

## The four tasks

| task | base | F2P | P2P |
|---|---|---|---|
| `hynek__structlog-786` | `2796b22ee1b5` | 34 | 241 |
| `hynek__structlog-794` | `b2f3daf0b59c` | 1 | 92 |
| `hynek__structlog-807` | `92fd882817a8` | 3 | 121 |
| `hynek__structlog-818` | `f194271d998e` | 2 | 139 |

All four declare `verifier.environment_mode = "separate"` and
`artifacts = ["/logs/model_patch.diff"]`, and all four pass the static audit
(`vektori-trace mine` runs it automatically; see `mining/inspect.py`).

## The oracle passes on real mined code

```
$ vektori-trace prove .../hynek__structlog-794
oracle passed: True
valid: True
```

```json
{"reward": 1.0, "resolved": true,
 "f2p_total": 1, "f2p_passed": 1, "p2p_total": 92, "p2p_passed": 92,
 "regressions": [], "parse_status": "ok", "runner": "pytest", "tests_parsed": 100}
```

A 169-line model patch was collected from the agent container, carried across as
an artifact, and applied in a verifier container the agent never touched. **This
is the first evidence that `model_patch_collect(pr.base_sha)` resolves inside a
real mined container** — the base ref survives `git_history_scrub`, which was
previously only verified against the synthetic probe repo.

## Two defects the run found

### 1. `--dockerfile` could not produce a validated task

The flag's stated purpose is to skip the bootstrap agent. It did — and returned
`test_cmds=[]`, because the agent is what discovers the test command. F2P/P2P are
derived by *running the suite*, so with no command every PR failed validation and
skipped as `no_fail_to_pass`. The flag could only ever emit unverified tasks.

`--test-cmd` and `--language` now accompany it. If you own the image, you own the
statement of how to test it.

### 2. `--agent claude_code` ran nothing, ever

Harbor's `AgentName` values are hyphenated and it rejects an underscore *before
any container starts*. This CLI's own default was `claude_code`, so `harbor run`
died with `Agent name claude_code is not valid`, `HarborTraceRunner` read the
non-zero exit as an `InfraFailure`, and every task in the sweep was excluded —
leaving an empty manifest and no indication why. Underscores are normalised now,
in `HarborTraceRunner` and in `validity.run_trial` (which `--base-agent` reaches
without passing through the former).

## The first live agent run, and what it corrected

`harbor run -a terminus-2 --model openai/gpt-5-nano` against
`hynek__structlog-794`. Two things came out of it.

**A real ATIF trajectory exists now** — committed as
`tests/fixtures/atif/real-terminus2-structlog/`, closing the gap #5 flagged
against itself. `ATIF-v1.7`, 4 steps, 45KB, and our parser turns it into 9 turns
with tool calls and observations both intact. Every other ATIF fixture in the
suite is hand-built; a hand-built fixture can only contain shapes we already
thought of, and only a real file can fail the `isinstance(raw_turns, list)` check
that started all of this.

**And it exposed a flaw in the `no_model_patch` guard.** The agent scored 0.0
with `parse_status: "no_model_patch"` — but it had not been wronged. Reading its
keystrokes:

```
git checkout -b fix/monochrome-console-renderer
git show --stat --summary
git diff --name-only HEAD~2 HEAD
```

It created a branch, read **the base commit's own diff**, concluded "118 files
changed … implementing the monochrome rendering behavior", called
`mark_task_complete`, and **never edited a file.** The empty patch was correct.

That is a genuine loss, and a diagnostically rich one — *declares success without
verifying* is precisely the kind of deficit this pipeline exists to find. Routing
it to `InfraFailure` would have dropped it from the dataset, which is a selection
bias pointed the wrong way: the clearest losses would be the ones that vanish.

`[ -s ]` is false for both *absent* and *empty*, which is what conflated them:

| state | means | now |
|---|---|---|
| file **absent** | collection never ran | refuse — `no_model_patch`, task leaves the dataset |
| file **empty** | collection ran, agent changed nothing | score it — the base repo *is* the state it produced |

Re-run after the fix, same agent, same model:

```json
{"reward": 0.0, "resolved": false,
 "f2p_total": 2, "f2p_passed": 0, "p2p_total": 139, "p2p_passed": 139,
 "parse_status": "ok", "runner": "pytest", "tests_parsed": 141}
```

A clean loss: the fix tests fail, all 139 regression guards hold, and nothing
claims the run was unjudgeable.

## Still open

- **`network_mode = "allowlist"`** and deleting `env_guard.py` — the last step-3
  item. Blocked on the same question its docstring raises: an allowlist that
  blocks the fix sources also blocks whatever the suite needs at runtime.
- **The 34-F2P task** (`-786`) is worth a look before it enters a training set;
  one PR flipping 34 tests is more likely a refactor than a bug fix.
- **Yield**: 10% here. Step 4 wants ≥50 tasks, so either ~400 PRs of history or
  a second repo.
