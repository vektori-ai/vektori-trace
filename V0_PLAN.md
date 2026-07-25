Vektori — v0 and v1 Plan
========================

**Date:** 2026-07-25 · **Tree:** `master` @ `c0f190b`

`docs/AUDIT.md` = the bug list. `docs/CLAIMS.md` = evidence for every fact and
number cited here; all code claims were reproduced by executing the code, all
external claims checked at source. This doc = what we build, in two stages.

- **v0** — mine environments from public repos ourselves, train against them, see
  if the number moves. No customer.
- **v1** — point it at customers (their traces, then their repo), then synthetic
  environments for the cases mining can't reach.

---

# v0 — Prove the loop on public repos

What v0 is
----------

**Mine environments out of public repos, train a small model against them, and see
if the number moves.** No customer, no report, no integration.

The question it answers:

> Does training a cheap model on *mined, verified* environments targeting a
> diagnosed capability beat training it on the same number of mined environments
> picked at random?

If yes, we have a product. If no, we're a report generator and we know in weeks. A
report alone can't answer it — "these capabilities cause the gap" stays an LLM's
opinion until training against it closes the gap.

Public repos, because that's the only place we get real verifiers without asking
permission, and the whole loop runs on our own machines.

Goals:

- Close the loop end to end: one repo, one candidate model, one deficit.
- A before/after number on held-out mined tasks, with an **untargeted control**.
- The real frontier-vs-candidate gap at a fixed scaffold.

Why this isn't just TRACE
-------------------------

TRACE (the method behind `diagnose.py`) is published *and fully open source* —
capability labelling, GRPO training, per-capability LoRA, and an inference routing
gate, with a SWE-bench reference. We adopt their code. Three things differ, and
they're the reason v0 is designed the way it is:

1. **Real environments, not synthesized ones.** TRACE's training environments are
   LLM-written; ours run the project's own test suite. The paper never validates
   synthetic-environment fidelity. Testable, not assumed — that's arm A3 vs A5.
2. **Cross-model contrast.** TRACE contrasts one model's wins against its own
   losses (confirmed from the paper). A capability the candidate lacks
   *consistently* is lacking in its wins too, so that gap flattens exactly where
   the commercial deficit lives.
3. **The task distribution is a specific repo's**, not a benchmark's.

Also worth knowing: the mining code is a port of HuggingFace's Repo2RLEnv, so
**mining is not our novelty** and shouldn't be pitched as such.

How it works
------------

```
repo URL
  │
  ├─[0] bootstrap ────▶ Docker image where the repo builds and tests run
  │
  ├─[1] mine ─────────▶ one task per merged PR, with real F2P/P2P verifiers
  │
  ├─[2] replay ───────▶ frontier AND candidate over the same tasks
  │                     → win/loss traces + THE GAP NUMBER
  │
  ├─[3] diagnose ─────▶ ranked capability deficits (or "none found")
  │
  ├─[4] select ───────▶ deficit-lacking tasks, in the trainable difficulty band
  │
  ├─[5] train ────────▶ rejection-sampling SFT + LoRA
  │
  └─[6] re-measure ───▶ held-out mined tasks, before vs after
```

Stages 0–1 exist and are the strongest code in the repo (~6.3k LOC). Stage 2 is
broken. Stages 3–6 need the work below.

**Environments come from two places, and only one is trustworthy:**

| Source | Verifier | Gameable? | Needs |
|---|---|---|---|
| **Mined from a repo** | the project's own tests | no | repo with tests + PR history |
| LLM-synthesized | LLM writes the checks *and* the oracle | **yes** | nothing |

v0 uses mined only. Traces carry no environment at all — which is why a
traces-only product can diagnose but never prove. Training needs an environment and
evaluation needs the same one; it's one requirement, twice.

What's broken right now
-----------------------

The pipeline has never run end to end. All four reproduced by execution
(`docs/CLAIMS.md`).

1. **Trajectories never parse.** Harbor writes ATIF — an object with `steps[]`;
   `miner.py:137` checks for a JSON array, so every trace degrades to a 4,000-char
   stdout blob. The diagnosis LLM is reading a stderr tail.
2. **Infra failure is recorded as an agent loss.** No `returncode` check, so a
   Docker OOM becomes a "loss" that diagnosis explains with an invented deficit.
3. **It can never say "no deficit found."** `cli.py:44` takes `scores[0]` with no
   threshold; inverted evidence (`gap = -1.0`) still yields a confident report.
4. **The labeller is told the answer.** `diagnose.py:118` opens with
   `"Trajectory (outcome: loss)"`.

The plan
--------

### Step 1 — Fix those four *(offline: no Docker, no harbor, no API key)*

- Infra failure ⇒ exclude the task; never write a loss trace.
- Per-task try/except; write the manifest after each task, not after the loop.
- Unique job dir per run, newest reward by mtime — today `rglob` returned a stale
  `0.0` over a fresh `1.0`.
- Threshold the diagnosis; `deficit: null` is a clean exit 0. A missing label
  counts as `NA`, not PRESENT (today it reads as *competence*).
- Remove the outcome from both labeller prompts.
- Drop the `harbor task init` shell-out — its flags don't exist. `emitter.py`
  already writes tasks correctly.
- Make the `verifier.py` ↔ `log_parsers/*` differential a real test. They're
  duplicated, they agree today, nothing enforces it, and on drift **every task
  silently scores 0.0**.
- Fix `owner_name`: `github.com/psf/requests/tree/main` → `('tree','main')` and
  `/pull/1234` → `('pull','1234')`. Every URL a human copies from a browser.

### Step 2 — Check the diagnosis works at all *(API key only)*

Generate synthetic traces with a deficit injected into losses only — `examples/`
already has one — and check the ranker finds it. Sweep trace count and prevalence.
Freeze it as a regression test.

If the ranker can't recover a deficit we planted ourselves, nothing downstream
matters. Cheapest thing that can kill the idea, so it runs early.

### Step 3 — Install harbor, verify the contract *(first step needing Docker)*

- **First, before anything else:** we emit `environment/Dockerfile` *and*
  `environment/docker-compose.yaml`, and Harbor honors one of them. If compose
  wins, the Dockerfile never runs — no base-commit reset, no git-history scrub, and
  the agent reads the fix out of `.git`. Docs say it's provider-dependent and give
  no answer for local Docker. Check in the container: `git rev-parse HEAD`,
  `git log --all`, `git remote -v`. This one check validates or invalidates every
  task the pipeline has ever emitted.
- **Real ATIF parser.** `steps[]` → turns; `message` is a string *or* a
  content-part array; `observation.results[]` become their own turns (**the
  traceback is the evidence** — without it the labeller sees what the agent tried
  and never what happened); recurse into `subagent_trajectories`, since Claude Code
  spawns subagents and skipping them drops most of the work. Import Harbor's own
  Pydantic models. **Delete the fallback and raise.** Commit a real job dir as a
  fixture — that's what keeps the bug dead.
- Move the verifier out of the agent's container (`environment_mode = "separate"`).
  Today it runs on the agent's `$PATH` with a reward fallback that means: shadow
  `python3`, exit 0, collect 1.0.
- `network_mode = "allowlist"`, delete `env_guard.py`. The default is `public`, so
  mined tasks currently have full access to github.com — and the current denylist
  only works on local Docker anyway.
- `task.toml` uses `schema_version`, not `version`; `keywords` under `[task]`.
- Timeouts: `sandbox.exec` returns 124 and nothing checks it, so a truncated test
  log produces a wrong F2P set baked into a shipped task.
- Then mine one small repo, print the skip-reason histogram, and hand-inspect three
  tasks: base commit right, `.git` scrubbed, F2P names actually in the test patch.

### Step 4 — Paired replay, and the gap number

Run frontier and candidate over the same mined tasks; `collect_traces` takes a list
of runners; manifest entries record which model produced each trace. This gives the
headline number before any diagnosis runs.

**Pin one scaffold, use it for both arms, name it in every number.** This matters
more than it sounds: in a bash-only scaffold Qwen3-Coder-30B-A3B scores 18.8% on
SWE-bench Verified while GPT-4o scores 21.2% — a 2-point gap. With a full scaffold,
32B-class open models report ~69.6%. **The gap is a property of model × scaffold,
not of the model.** So a minimal scaffold erases the gap we're studying, and the
candidate must be genuinely small (4B–8B class).

Measure the real gap on ≥50 tasks before believing any framing. Under ~10 points,
change the candidate — not the story.

### Step 5 — Diagnose, honestly

Two contrasts out of the one replay run:

| Contrast | Wins | Losses | Tells us |
|---|---|---|---|
| cross-model | frontier | candidate | what's worth fixing |
| within-model | candidate wins | candidate losses | whether there's anything to train from |

Within-model matters because if the candidate has *never* done the thing right,
rejection sampling has nothing to keep. Report a cross-model deficit with no
within-model signal as **"identified, not trainable."**

Three corrections worth making, and no more:

- **Compare the models on the same task, not on average.** Frontier wins come from
  easier tasks and candidate losses from harder ones, so averaged rates mix
  capability with difficulty. Count tasks where the frontier had the capability and
  the candidate didn't, versus the reverse, and test that (exact McNemar on
  discordant pairs). Difficulty cancels because it's the same task. This also sets
  the real support floor: under ~8 discordant pairs nothing can reach
  significance, whatever the data.
- **The 0.20 threshold is uncalibrated.** Our labeller is a blurry ruler, and blur
  always shrinks effects toward zero — a true gap of 0.5 can surface as 0.18 and be
  rejected. Hand-label ~50 traces once to measure how blurry, then adjust the
  threshold. Until then we don't know if 0.20 is strict or loose. (In v0 the
  win/loss ruler is sharp: it's execution.)
- **Print N beside every number.** Don't rank a capability measured on 3 traces
  next to one measured on 40. And when we report the top of a ranked list of 4–8
  capabilities, remember the top of a noisy list looks good even when nothing is
  there.

**Stop replaying when the ranked list stops changing** as tasks are added. If it's
still reshuffling at the budget cap, the answer is "not enough data," not a report.

### Step 6 — Train and re-measure *(the point of v0)*

- Select training tasks two ways, both required: the deficit was lacking, **and**
  the candidate's measured pass rate is in **10–40%**.
- The band isn't a hunch. Rejection sampling keeps only rollouts that pass, so at
  0% the dataset is empty. In GRPO the advantage is `(r − mean)/std`, so a group
  where every rollout scores the same gives **zero gradient** — which is why DAPO
  discards all-right and all-wrong groups. Too easy is equally useless. Hence the
  middle.
- Measure pass rate with 8–16 rollouts per task, **only on diagnosis-selected
  tasks**. Step 4's broad sweep is 1 rollout per task; 16 everywhere costs 16× and
  buys no diagnostic power.
- Carve the held-out slice *before* training. Exclude SWE-bench Verified repos and
  PRs — mining them means training on the answers, and getting caught destroys the
  number.
- Vendor TRACE's training code. v0 uses rejection-sampling SFT + LoRA only (LoRA
  also *is* the non-regression mechanism — base weights untouched). Their GRPO path
  is v1.
- Pilot on ~10 tasks before any full run: does it execute, do metrics compute, is
  the cost sane.

The experiment
--------------

| Arm | What | Answers |
|---|---|---|
| A0 | candidate, untouched | the floor |
| A1 | candidate + deficit-targeted prompt | "couldn't you just prompt it?" — free |
| **A2** | trained on **random** mined tasks, same count, same rollout budget | **the control** |
| **A3** | trained on **deficit-selected** mined tasks | the claim |
| A4 | frontier | the ceiling |

**A2 is the arm that makes this real.** Without it, "targeted training on mined
environments" is indistinguishable from TRACE with a different environment source.
A3 ≈ A2 means targeting adds nothing.

Primary metric: pass rate on the held-out mined slice, **A3 vs A2**, compared
task-by-task since both arms attempt the same tasks. **Decide the held-out size and
the smallest effect it can resolve before training** — SWE-Gym moved a 7B model
about 3 points, and a 50-task slice cannot resolve 3 points.

Also report: A0 vs A4 (the gap), a non-regression check (IFEval or the base model's
own evals — narrow training genuinely degrades instruction-following), and cost per
solved task versus frontier.

Every run writes: arm, task ids and split, scaffold name+version, base model and
adapter, seed, pass rate with N, the paired comparison, and the environment
(GPU/harbor/image digest). If a number isn't re-derivable from disk, it doesn't
count.

Stop and rethink if
-------------------

- Compose wins over the Dockerfile — every task ever emitted is invalid.
- The ranker can't find a deficit we planted (step 2).
- The gap is under ~10 points, or the candidate has almost no wins to bootstrap from.
- A prompt (A1) closes most of the gap — then we're a prompt tool with an expensive
  backend. Thesis-level finding for two eval runs.
- A3 ≈ A2 — targeting doesn't help.
- The 10–40% band is empty — nothing trainable at this model + scaffold.

Done when
---------

- Real trajectories parse into turns with tool calls and results; no fallback path
  exists; golden-file test passes.
- Infra failures never appear as losses.
- Diagnosis is blind to outcome and can return "none found" with exit 0.
- A container check confirms base commit, scrubbed `.git`, no remote; a
  shadow-`python3` reward hack scores 0.0.
- A gap number exists on ≥50 mined tasks at a pinned, named scaffold.
- A3 vs A2 measured on a held-out slice, task-by-task, with the resolvable effect
  size stated up front.
- Non-regression check inside a pre-declared tolerance.
- Every number re-derivable from artifacts on disk.

---

# v1 — Point it at customers

Each stage needs something the previous one produces. v0 is what makes v1's
instruments trustworthy: thresholds calibrated, labeller blur measured, and real
execution outcomes to check cheaper proxies against.

### v1.1 — Customer traces *(diagnosis only)*

Their real task distribution, no integration, no repo access.

**The ask is "traces plus your success/fail flag,"** not "traces." Without the flag
we'd infer outcomes with an LLM, which the whole design forbids, and every number
downstream inherits that error. Final outcome per session is enough — the
contrastive math needs one bit, not per-step labels.

Where outcome labels come from, best first: their own business signal (ticket
closed, CI green, PR merged, no retry) → explicit thumbs → LLM judge as a last
resort, and only with its agreement against a real signal measured on a subset.

What changes without an environment:

- **No re-runs.** Each trace is a different task, so per-task pass rate is
  unmeasurable and the difficulty band can't be computed. Partial recovery: cluster
  near-duplicate tasks (retries, recurring request types) and estimate a rate per
  cluster — coarser, good enough to pick a band.
- **No second model by execution.** Substitute: replay the trace state at a given
  step and ask the frontier model what it would do next, then compare to what the
  candidate actually did. Keeps the same-situation pairing, at the cost of comparing
  a judgment to an action. **Validate it in v0 first**, where both the annotation
  and the real outcome exist.
- **Two blurry rulers instead of one**, and blur multiplies: an inferred outcome
  label times a blurry labeller can halve the observable effect. Apply v0's
  calibration.
- **Count opportunities, not traces.** One trace exercises the same capability
  dozens of times — every tool call is a chance to mangle arguments. That's 10–50×
  more data points for free. But opportunities in one trace are correlated, and how
  correlated is itself the finding: always-botched means a consistent deficit (great
  training target, little info per trace); occasionally-botched means a reliability
  problem that prompting may fix.
- **"Not enough data, send N more" is a real output.** Better than a deficit
  asserted from four traces, and it tells them exactly what to send next.

No training here: traces carry no environment, so there's nothing to train against
and nothing to re-measure on.

### v1.2 — Customer repo *(the full loop on their code)*

Same machinery as v0, pointed at their repo. With traces *and* repo we get both
their distribution (which tasks matter, real outcomes) and real environments to run
both models in — so the same-model limitation disappears.

Then: self-host the miner in their VPC. It's Docker + `gh` + an LLM, so they never
hand us the repo, which removes the biggest procurement blocker. Prerequisite is
routing every LLM call through LiteLLM (`mining/llm.py` already does; `llm.py`
doesn't).

Ship the environment, don't run the training — we run it for the proof because we
need the number, but the product is the environment.

### v1.3 — Synthetic environments *(coverage where mining can't reach)*

Repos without tests, non-Python, and non-coding domains. Deferred to here on
purpose: a generator built before v0 has nothing to validate against.

Once mined tasks exist, fidelity is directly measurable — **train on synthetic,
evaluate on the mined held-out set.** Transfer *is* the fidelity number. TRACE
never did this, and being able to say "our generator is calibrated against real
verifiers" is a claim nobody else has.

Guard the known failure: one LLM writing the task, the checks, and the oracle, then
grading itself, is a policy learning to satisfy checks rather than do work.

### v1.4 — Everything else

- **The probe as a discriminator.** `taskgen.py` can separate "can't do it at all"
  from "can't do it at turn 180 under load" — isolated probe → probe + distractors →
  full task. Passes isolated, fails in situ ⇒ trainable, maybe promptable. Needs a
  frontier-passes arm too, or "the candidate failed" and "the task is broken" look
  identical. Track yield: under ~50% means we're generating broken tasks.
- **Learning curves instead of verdicts.** Never claim a model "cannot" learn
  something — no finite experiment shows that. Train at n ∈ {25, 50, 100}, plot gain
  against log n, and report the bounded form: *not learnable within R rollouts, at
  scaffold S, by method M*.
- **Multi-deficit + adapter routing** (TRACE's MoE gate), **GRPO** (v0's rollout
  adapter is the same interface, so it's a trainer swap), **multi-language** (Java
  and Ruby are realistic; Rust and Go put table-tests inside the source file, which
  makes patch/test splitting structurally impossible), **MCP surface** so a coding
  agent can drive the pipeline in one instruction.
- **Use the gold patch.** Mined tasks already carry the human's fix, F2P names, and
  changed files, and diagnosis currently sees only turns. Comparing the trajectory
  against the real fix is probably the biggest accuracy upgrade available.
- **Formal sequential stopping** — proper anytime-valid boundaries (accept / reject /
  continue) instead of v0's rank-stability heuristic. Matters when traces stream in
  continuously and we'd otherwise peek after every batch.
- **Mining robustness and yield:** per-PR error isolation with checkpointing (one
  Docker error currently aborts an hour of work), a fresh-dependency check between
  base commits (one sandbox reused across all PRs gives false F2P/P2P), and the
  yield bugs — `C#9` mangled by the leak stripper, `_path_is_test` missing
  `.go`/`.rs`/`conftest.py`, spaced diff paths dropping PRs, `effaced` matching the
  SHA regex.
- **Go-to-market:** public repos are the demo generator — mine a prospect's own
  open-source repo, run paired replay, lead with their number. No permission, no
  data handoff, no meeting needed to produce it.

Open questions
--------------

1. **Which candidate model** — 4B or 8B class. Decide from step 4's measurement,
   not from benchmark tables.
2. **Which scaffold** gets pinned, and who owns freezing it.
3. **How many held-out tasks**, and what effect size that resolves. If ~50 can't
   resolve the expected effect, mine more before training.
4. **Which repo first** — Python, real tests, issue-linked PRs, not in SWE-bench
   Verified.
5. **Training in this repo or a separate package?** Vendoring TRACE argues for
   separate.
6. **Canonical win/loss** — strict `resolved`, or `reward = f2p_rate × p2p_rate ≥
   1.0`? `grade()` computes both; pick one and say which in every report.