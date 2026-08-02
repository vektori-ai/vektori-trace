# `commit_runtime`

Commit-level mining. The sibling of `pr_runtime`, not a replacement.

## Why

`pr_runtime` only sees fixes that arrived as a merged PR carrying a linked
issue. The 800-PR prefect pilot:

```
no_linked_issue    434   ← 54% of all candidates
non_bug_pr         141
already_processed  100
no_fail_to_pass     54
no_test_patch       31
no_new_test_funcs   25
emitted             12
```

54% dropped for a missing issue link. Those fixes are real and their tests are
real — the only thing absent was a GitHub cross-reference. And fixes committed
straight to main were never candidates at all.

Yield also decays with history depth: 5.9% over the first 220 PRs, **zero over
the last ~300**, because older PRs stop linking issues. Mining a repo's PRs
harder does not help; mining its commits does.

Upstream's reference dataset is 100 tasks across **13 repos** — about 8 per
repo. One repo cannot supply a corpus regardless of how well the miner works.

## What differs from `pr_runtime`

Everything upstream of validation. The F2P/P2P oracle, the graded verifier and
the emitted task shape are identical, so output from both pipelines pools into
one corpus.

| | `pr_runtime` | `commit_runtime` |
|---|---|---|
| candidates | `gh pr list` | `git log --first-parent` |
| problem statement | linked issue text | LLM-synthesized, leak-checked |
| gate | linked issue required | bugfix signal required |
| P2P | uncapped | capped (default 50) |
| task id | `owner__repo-<pr>` | `owner__repo-<short-sha>` |

## Instruction synthesis, and why it is verified rather than trusted

A commit message is written by the fixer, after the fix. It routinely names the
changed function or pastes a changelog bullet describing the solution. An agent
handed that scores 1.0 by reading, not solving — and nothing downstream can
distinguish that from a real solve, so every pass@k computed from the corpus is
inflated.

So the commit is rewritten into the symptom a user would have reported.

**The prompt alone is not sufficient.** Measured against a deliberately leaky
commit (`fix: guard against None in normalize_rrule_string`, body naming the
file, the test and the fix approach), gpt-5-nano dropped the file name, the test
name and the issue number — but kept `normalize_rrule_string` and `.strip()`,
which is most of the patch.

So synthesis output is checked against identifiers the patch *introduces*
(`+def` / `+class` / `+func` lines, plus changed-file stems):

1. synthesize
2. `leaked_identifiers()` against patch + test_patch
3. on a hit, retry once with the offending tokens named explicitly
4. still leaking → **drop the task**

Step 4 is a skip, not a fallback to raw commit text. Falling back would emit the
message that leaked in the first place. A dropped task costs one task; a leaked
one costs the credibility of every number derived from the corpus.

On the same leaky commit the retry produced *"Passing None to the recurrence
rule validator raises an exception"* — no residual leaks. Cost for the pair:
**$0.0011**.

Tokens shorter than 5 characters are ignored: `run`, `id` and `f` appear in
ordinary prose, and flagging them would drop every task.

## Token budget

`max_llm_tokens` defaults to **4096**, not upstream's 1024. Reasoning models
bill thinking against this budget and emit the visible answer from what is
left. Measured on gpt-5-nano:

| `max_tokens` | visible output |
|---|---|
| 1024 | **0/3** |
| 2048 | 3/3 |
| 4096 | 3/3 |

At 1024 every call returns empty content, which arrives as a too-thin synthesis
and drops the task — so the failure looks like "this repo has no clean
instructions" rather than a misconfigured budget.

## P2P cap

The graded reward is `f2p_rate * p2p_rate`. A whole-suite P2P of several hundred
tests scales every correct solve down on any single unrelated flake, and
lengthens every rollout. Default cap 50; `0` disables it.

The kept subset is **sorted before truncation** so two mines of the same commit
produce the same task.

## Filters

Applied before validation, because validation costs a container run per
candidate.

**Metadata** — merge commits, root commits (no parent to diff against), bot
authors, messages under `min_message_words`, non-bugfix conventional-commit
types (`chore:`/`docs:`/`feat:`/…), and anything without a positive bugfix
signal (a `fix:` prefix, a `Closes #N` trailer, or a bugfix keyword).

**Structural** — CI-only diffs, commits touching more than
`max_source_files_per_commit` source files, and commits whose test patch adds no
new test function.

## Usage

```bash
vektori-trace mine-commits --repo prefecthq/prefect --limit 200 --clone-depth 800
```

`--clone-depth` must exceed `--limit`; `git log` cannot walk past the end of a
shallow clone, and the CLI refuses the combination rather than silently
reporting a yield that looks like a property of the repo.

Notable flags: `--no-synthesis` (raw commit text — not recommended, see above),
`--max-pass-to-pass`, `--branch`, `--skip-validation`.

## Credit

Commit-level curation is R2E-Gym's SWE-GEN (Jain et al., COLM '25). This follows
the shape of Repo2RLEnv's `commit_runtime` (Apache-2.0). The leak-verification
gate and the token-budget finding are ours.
