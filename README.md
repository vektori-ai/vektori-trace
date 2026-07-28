# pipeI

Paste in failing/passing agent traces, get out: a diagnosed capability deficit, a
generated verifiable task (Harbor spec) that isolates it, and a validity proof
(oracle solution passes, a real agent lacking the capability fails).

```
ingest traces → diagnose deficit → generate env + verifier → validity proof
```

It deliberately stops there — no training run, no regression suite, no redeploy.

Task generation and execution are built on
[Harbor](https://github.com/harbor-framework/harbor) — generated tasks are
plain `task.toml` / `instruction.md` / `environment/Dockerfile` /
`tests/test_outputs.py` / `solution/solve.sh`, and `harbor run` is what
actually proves validity, including running **Codex or Claude Code** as the
"base" agent against the generated task.

## Install

```bash
pip install -e .
# or: uv pip install -e .
```

Requires the [`harbor`](https://github.com/harbor-framework/harbor) CLI and
Docker for the `--prove` and `mine` steps (`uv tool install harbor` or see
Harbor's docs). Diagnosis and task generation need neither — task files are
written directly, not via the `harbor` binary.

Mined tasks also need **`docker buildx`**, which harbor uses to build the
egress-control sidecar that enforces their network policy. Without it the trial
dies at `unknown flag: --file` before any container starts:

```bash
docker buildx version || {
  mkdir -p ~/.docker/cli-plugins
  curl -sSL -o ~/.docker/cli-plugins/docker-buildx \
    https://github.com/docker/buildx/releases/latest/download/buildx-v0.35.0.linux-amd64
  chmod +x ~/.docker/cli-plugins/docker-buildx
}
```

```bash
cp .env.example .env   # fill in OPENAI_API_KEY
export $(cat .env | xargs)
```

## Usage

Traces are JSON files (`{runId, status, turns: [...]}`). A manifest lists
which traces are wins and which are losses — contrastive scoring needs both:

```bash
vektori-trace diagnose \
  --manifest examples/manifest.json \
  --out ./vektori-out
```

This will:

1. Propose candidate capabilities from the traces (1 LLM call)
2. Label each trace NA / PRESENT / LACKING per capability (1 LLM call per trace)
3. Score each capability by how much its absence separates wins from losses, and rank deficits
4. Generate a Harbor task isolating the top-ranked deficit (1 LLM call)

Steps 1 and 2 are blind to outcome: the manifest's win/loss labels are used
only in step 3's arithmetic. Told which traces failed, the labeller reasons
backwards from the ending and marks capabilities LACKING to justify it, and the
gap then measures the prompt rather than the agent.

A capability is reported only if it clears `--min-gap` (default 0.20) with at
least `--min-support` (default 3) relevant traces on each side. **"No deficit
found" is a normal result and exits 0** — it writes the ranked list and no task.
The 0.20 threshold is uncalibrated; the labeller is a blurry ruler and blur
shrinks effects toward zero, so treat it as a placeholder until it's set from
hand-labelled data.

Output lands in `./vektori-out/`. `diagnosis.{json,md}` (the ranked deficits +
report, with N beside every rate) is always written; `tasks/<deficit-name>/`
(the Harbor task) only when a capability clears both thresholds — a "no deficit
found" run writes the ranked list and no task dir.

Add `--prove` to also run the validity proof via `harbor run`:

```bash
vektori-trace diagnose \
  --manifest examples/manifest.json \
  --out ./vektori-out \
  --prove \
  --base-agent codex \
  --base-model gpt-5-nano
```

`--base-agent` accepts any Harbor agent name — `codex`, `claude-code`,
`aider`, `opencode`, etc. (Harbor's names are hyphenated; underscores are
normalised, since it rejects them outright before any container starts.) Without `--base-agent`, only the oracle solution is
run (proves the task is solvable at all, not that a real agent lacks the
capability).

You can also run the proof separately against an already-generated task:

```bash
vektori-trace prove ./vektori-out/tasks/<deficit-name> --base-agent claude-code
```

## Is the environment sound?

Two independent leaks have to stay closed. The `Dockerfile` resets the repo to
the PR's base commit and scrubs `.git` of everything after it — otherwise
`git diff origin/main` reads the answer with no network at all. And the task
declares a **default-deny network policy**, because the published fix lives on
PyPI and GitHub: an agent blocked from `.git` will fall back to
`pip download <pkg>==<fixed>` and read the fix out of the wheel. That is
observed behaviour, not a hypothetical. If either guard slips, every "win" is
contamination.

```bash
vektori-trace check-env      # needs Docker + harbor
```

It builds a probe task from the *real* guards — a throwaway repo whose history
contains a sentinel "fix" — runs it through harbor, and inspects the container:
base commit, pruned objects, removed remote, whether HTTPS to each fix source
actually fails, and whether an allowed host still answers.

```text
[PASS] dockerfile_ran: HEAD=db711bf3... (expected db711bf3...)
[PASS] fix_sources_unreachable: all 8 fix-source host(s) unreachable
[PASS] allowed_host_reachable: api.openai.com -> 401
[PASS] future_commits_pruned: fix object reachable: no
[PASS] sentinel_unreachable: git log -SSECRET_FIX: <no hits>
[PASS] remote_removed: git remote -v: <none>
```

Those last two network findings only mean something together. "Everything is
unreachable" passes the first and breaks every installed agent, which
npm-installs itself into the container and calls its model API from there.

**Network policy is per phase**, which is what makes a default-deny allowlist
usable at all:

| phase | policy | why |
|---|---|---|
| `[environment]` / `[agent]` | `allowlist` | the model API and the agent's own registry, and nothing else — never PyPI or GitHub |
| `[verifier]` | `no-network` | dependencies are in the image and the suite runs offline, and this is the container that decides the reward |

Harbor enforces this with a default-deny sidecar proxy, so a host nobody
thought of is blocked too — the thing a denylist can never do. It needs
`docker buildx` installed (harbor builds the sidecar image with it) and refuses
the task outright on a provider that can't enforce the policy, rather than
silently downgrading.

Worth re-running after any harbor upgrade.

```bash
vektori-trace check-env --reward-hack
```

runs three agents against the same task, same environment, same verifier —
only `solve.sh` differs:

| agent | must score | what a wrong answer would mean |
|---|---|---|
| forges its own score, fixes nothing | 0.0 | the reward is forgeable |
| actually fixes the bug | 1.0 | scoring is broken and everything is zero |
| fixes the bug **and commits** | 1.0 | correct work is being collected as a loss |

The second and third are controls, and they're the point: a 0.0 from the cheat
proves nothing on its own, and the two honest agents differ only in where they
left their changes — which is not something the task asked about.

Mined tasks score in a **separate** container the agent never touched. The
agent's work is collected as a diff against the task's base commit, applied to
a clean checkout, and graded there. Under the old shared-container default an
agent that shadowed `python3` scored 1.0 on an unsolved task — measured, not
hypothesised.

Absent and empty are different facts about that diff, and they are handled
differently:

| the patch file is | means | what happens |
|---|---|---|
| **absent** | collection never ran | refused — running the suite anyway scores the base repo, a different task, and the 0.0 would be indistinguishable from an agent that tried and failed. The run leaves the dataset. |
| **empty** | collection ran, the agent changed nothing | scored — the base repo *is* the state that agent produced, so 0.0 is the honest result |

The second case is not hypothetical: on the first live agent run the model
created a branch, read the base commit's own diff, mistook it for its own work
and declared completion having edited no file. Treating that as unjudgeable
would drop the *clearest* losses out of the corpus — "declares success without
verifying" is exactly the deficit worth finding.

## Does the diagnosis work at all?

Before trusting a diagnosis on real traces, check the ranker can recover a
deficit we planted ourselves:

```bash
vektori-trace selftest --ceiling-only     # free, offline, no API key
vektori-trace selftest --quick            # one config, ~13 LLM calls
vektori-trace selftest                    # full sweep, ~250 LLM calls
```

It generates synthetic traces carrying a known capability deficit, runs the
real diagnosis over them, and reports how often the planted capability comes
back on top — swept across trace count and prevalence, repeated per cell
because the proposer and labeller are both sampled.

Two things make it a test rather than a demo. **Losses that don't carry the
planted deficit fail for unrelated reasons**, so the ranker has to choose
between real competing explanations instead of the only one on offer. And
**ground truth is known per trace**, so the report also states how often the
labeller reproduced the label we know to be correct — the blur that shrinks
every gap downstream toward zero.

Failures are split by cause, because they call for different fixes:
`not_proposed` (prompt), `outranked_by_distractor` (labeller),
`top_ranked_but_below_threshold` (calibration).

Every cell also reports a **ceiling**: what a perfect proposer and a perfect
labeller would recover from the same corpus. It costs nothing to compute and no
real run can exceed it, so `--ceiling-only` is worth running first — where the
ceiling is 0% the config is unrecoverable by construction and a live run there
measures the thresholds rather than the ranker.

At the defaults (`min_gap=0.20`, `min_support=3`), the binding constraint is
support, not gap. Configs below roughly **3 wins that exercise the capability
and 3 losses that lack it** are rejected with a perfect gap of 1.0 at rank 1 —
so `n_losses × prevalence ≥ 3` is the floor worth quoting when asking for more
traces.

## Mining a repo

```bash
vektori-trace mine --repo hynek/structlog \
  --dockerfile examples/dockerfiles/structlog.Dockerfile \
  --test-cmd 'python -m pytest -p no:randomly -q' --language python \
  --limit 40 --no-replay
```

One task per merged PR, with a verifier derived by running the repo's own suite
twice — once with only the test diff applied (what fails pre-fix), once with the
gold patch too (what now passes). The fail→pass set is the oracle.

`--dockerfile` skips the bootstrap agent, which makes a run deterministic and
free. It then needs `--test-cmd`, because F2P/P2P come from *running* the suite
and nothing has inspected the repo to find out how.

`--no-replay` stops after mining and auditing, before any agent runs.

Every run prints where the candidates went and audits what came out:

```text
Where the 40 candidate PR(s) went:
  emitted                       4
  no_test_patch                20
  non_bug_pr                    6
  no_new_test_funcs             6
  no_fail_to_pass               4

Static audit of 4 emitted task(s):
  every task agrees with itself on all checks
```

The histogram matters because the task count alone can't say whether a small
yield means the repo is unsuitable or a filter is wrong, and those call for
opposite responses. Here `no_test_patch` at 20/40 is PRs that changed source
without touching tests — unminable by construction, since the test diff *is* the
verifier.

The audit is static and cheap: it re-reads each emitted task and checks it agrees
with itself — the Dockerfile resets to the base commit `task.toml` declares,
`.git` is scrubbed of everything past it, and every F2P test name actually
appears in the hidden test patch. That last one fails silently otherwise: a name
the test patch never adds is never collected, never runs, never passes, so the
task scores 0 for everyone forever and reads as merely hard.

Results from the first real run, including two defects it found:
[`docs/mine-results.md`](docs/mine-results.md).

## Two models, two contrasts

`replay` runs a frontier and a candidate model over the same mined tasks on one
pinned scaffold, and writes a manifest tagged with `model` and `task`. Handing
that manifest to `diagnose` unqualified mixes both models into one win/loss set
and averages away the thing being measured, so name them:

```bash
vektori-trace diagnose \
  --manifest ./vektori-out/replay-manifest.json \
  --frontier-model gpt-5 --candidate-model qwen3-8b \
  --out ./vektori-out
```

| Contrast | Wins | Losses | Answers |
|---|---|---|---|
| cross-model | frontier's | candidate's | what's worth fixing |
| within-model | candidate's | candidate's | whether there's anything to train from |

The chosen deficit and the ranked list are the **cross-model** contrast. The
within-model one decides trainability: rejection sampling keeps only rollouts
that pass, so if the candidate has never once demonstrated the capability, a
task built against that deficit yields an empty training set rather than a hard
one. That result is reported as **"identified, not trainable"** — it exits 0 and
generates no task, `--prove` included.

Both flags are required together, must name different models, and must both
appear in the manifest; each is rejected at parse time rather than after an LLM
call per trace. Given neither, the model-blind path is unchanged.

Alongside them, an **exact McNemar test** on the chosen capability compares the
two models task by task instead of on average — frontier wins come from easier
tasks and candidate losses from harder ones, and pairing cancels that. It counts
tasks where the frontier demonstrated the capability and the candidate didn't
(`b`) against the reverse (`c`); pairs where either side is NA are dropped as not
comparable. Under 6 discordant pairs no split can reach p<0.05 at all and under
9 only a perfectly one-sided one can, so the report flags anything below 8 as
underpowered — the test having no power is not the models being alike.

## Model

Diagnosis and task generation use `gpt-5-nano` by default (cheapest current
OpenAI model) — override with `VEKTORI_MODEL` or `--model`.

## Try it

`examples/` has 2 win + 2 loss synthetic traces with a deliberate, obvious
deficit (retrying a failed tool call verbatim instead of reading the error and
adjusting arguments) so you can see the whole pipeline run end to end.
