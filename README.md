# vektori-trace

Paste in failing/passing agent traces, get out: a diagnosed capability deficit, a
generated verifiable task (Harbor spec) that isolates it, and a validity proof
(oracle solution passes, a real agent lacking the capability fails).

This is the *front half* of the Vektori loop:

```
ingest traces → diagnose deficit → generate env + verifier → validity proof
```

It deliberately stops there — no training run, no regression suite, no redeploy.
See [vektori-platform](https://github.com/vektori-ai/vektori-platform)'s
`docs/DESIGN.md` for the full methodology this implements (the ER+/ER-/Δ/Cov
formulas).

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
Docker for the `--prove` step (`uv tool install harbor` or see Harbor's docs).

```bash
cp .env.example .env   # fill in OPENAI_API_KEY
export $(cat .env | xargs)
```

## Usage

Traces are JSON files matching vektori-platform's `Run`/`Turn` schema
(`{runId, status, turns: [...]}`). A manifest lists which traces are wins and
which are losses — the contrastive formulas need both:

```bash
vektori-trace diagnose \
  --manifest examples/manifest.json \
  --out ./vektori-out
```

This will:

1. Propose candidate capabilities from the traces (1 LLM call)
2. Label each trace NA / PRESENT / LACKING per capability (1 LLM call per trace)
3. Compute ER+(c), ER-(c), Δ(c), Cov(c) per DESIGN.md and rank deficits
4. Generate a Harbor task isolating the top-ranked deficit (1 LLM call)

Output lands in `./vektori-out/`: `tasks/<deficit-name>/` (the Harbor task) and
`diagnosis.{json,md}` (the ranked deficits + report).

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

## Model

Diagnosis and task generation use `gpt-5-nano` by default (cheapest current
OpenAI model) — override with `VEKTORI_MODEL` or `--model`.

## Try it

`examples/` has 2 win + 2 loss synthetic traces with a deliberate, obvious
deficit (retrying a failed tool call verbatim instead of reading the error and
adjusting arguments) so you can see the whole pipeline run end to end.
