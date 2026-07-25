# vektori-trace

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

Output lands in `./vektori-out/`: `tasks/<deficit-name>/` (the Harbor task) and
`diagnosis.{json,md}` (the ranked deficits + report, with N beside every rate).

Add `--prove` to also run the validity proof via `harbor run`:

```bash
vektori-trace diagnose \
  --manifest examples/manifest.json \
  --out ./vektori-out \
  --prove \
  --base-agent codex \
  --base-model gpt-5-nano
```

`--base-agent` accepts any Harbor agent name — `codex`, `claude_code`,
`aider`, `opencode`, etc. Without `--base-agent`, only the oracle solution is
run (proves the task is solvable at all, not that a real agent lacks the
capability).

You can also run the proof separately against an already-generated task:

```bash
vektori-trace prove ./vektori-out/tasks/<deficit-name> --base-agent claude_code
```

## Is the environment sound?

Mined tasks ship two files that have to both take effect: a `Dockerfile` that
resets the repo to the PR's base commit and scrubs `.git` of everything after
it, and a `docker-compose.yaml` that blackholes the hosts serving the published
fix. If only one applied, agents would read the answer out of `.git` or
`pip download` it, and every "win" would be contamination.

```bash
vektori-trace check-env      # needs Docker + harbor
```

It builds a probe task from the *real* guard functions — a throwaway repo whose
history contains a sentinel "fix" — runs it through harbor, and inspects the
container: base commit, pruned objects, removed remote, per-host resolution in
both address families, and whether HTTPS to the fix sources actually fails.

Worth re-running after any harbor upgrade: the runtime's compose-merge order is
not something we control.

```bash
vektori-trace check-env --reward-hack
```

runs two agents against the same task: one that fixes nothing and forges its
own score, and one that actually fixes the bug. The cheat must score 0.0 and
the honest fix 1.0 — the second is what distinguishes "the hack was blocked"
from "scoring is broken and everything is zero".

Mined tasks score in a **separate** container the agent never touched. The
agent's work is collected as a diff, applied to a clean checkout, and graded
there. Under the old shared-container default an agent that shadowed `python3`
scored 1.0 on an unsolved task — measured, not hypothesised.

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

## Model

Diagnosis and task generation use `gpt-5-nano` by default (cheapest current
OpenAI model) — override with `VEKTORI_MODEL` or `--model`.

## Try it

`examples/` has 2 win + 2 loss synthetic traces with a deliberate, obvious
deficit (retrying a failed tool call verbatim instead of reading the error and
adjusting arguments) so you can see the whole pipeline run end to end.
