Vektori — measurement discipline and v1 plan
============================================

**Date:** 2026-07-28 · **Branch:** `capability-routing`

**Scope.** This doc is authoritative for **mining, verification, diagnosis and
provenance** — the measurement half. The v0 *experiment* design it originally
carried (task selection by pass-rate band, arms A2/A3, rejection-sampling SFT as
the training method) is **superseded by [`PLAN.md`](PLAN.md)**, which replaces
band-selection with support-measured routing and A2/A3 with B1/B2. Where the two
disagree about the experiment, `PLAN.md` wins.

Steps 1–3 of the original plan (the four launch bugs, the planted-deficit
selftest, the harbor contract check) are **done** — see `git log` and
`docs/selftest-results.md`, `docs/network-policy.md`, `docs/mine-results.md`.

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
> under an inverted assignment?

If yes, we have a product. If no, we're a report generator and we know in weeks. A
report alone can't answer it — "these capabilities cause the gap" stays an LLM's
opinion until training against it closes the gap.

Public repos, because that's the only place we get real verifiers without asking
permission, and the whole loop runs on our own machines.

Goals:

- Close the loop end to end: mined repos, one candidate model, one deficit.
- A before/after number on held-out mined tasks, with the control arm `PLAN.md`
  specifies (B2, anti-routed).
- The real frontier-vs-candidate gap at a fixed scaffold.

Why this isn't just TRACE
-------------------------

TRACE (the method behind `diagnose.py`) is published *and fully open source* —
capability labelling, GRPO training, per-capability LoRA, and an inference routing
gate, with a SWE-bench reference. We adopt their code. Three things differ, and
they're the reason the loop is designed the way it is:

1. **Real environments, not synthesized ones.** TRACE's training environments are
   LLM-written; ours run the project's own test suite. The paper never validates
   synthetic-environment fidelity. Testable, not assumed (v1.3).
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
  └─[4+] ─────────────▶ support measurement, routing, training — see PLAN.md
```

**Environments come from two places, and only one is trustworthy:**

| Source | Verifier | Gameable? | Needs |
|---|---|---|---|
| **Mined from a repo** | the project's own tests | no | repo with tests + PR history |
| LLM-synthesized | LLM writes the checks *and* the oracle | **yes** | nothing |

v0 uses mined only. Traces carry no environment at all — which is why a
traces-only product can diagnose but never prove. Training needs an environment and
evaluation needs the same one; it's one requirement, twice.

Paired replay, and the gap number
---------------------------------

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

Diagnose, honestly
------------------

Two contrasts out of the one replay run:

| Contrast | Wins | Losses | Tells us |
|---|---|---|---|
| cross-model | frontier | candidate | what's worth fixing |
| within-model | candidate wins | candidate losses | whether there's anything to train from |

Within-model matters because if the candidate has *never* done the thing right,
rejection sampling has nothing to keep. Report a cross-model deficit with no
within-model signal as **"identified, not trainable."** (`PLAN.md` sharpens this
into a measurement: `pass@k` decides it, rather than the within-model contrast
alone.)

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
  rejected. `PLAN.md` Step E replaces hand-labelling with execution-established
  forking steps as the calibration ground truth. Until that runs we don't know if
  0.20 is strict or loose. (In v0 the win/loss ruler is sharp: it's execution.)
- **Decide whether to fix the ruler before calibrating it.** Adjusting the
  threshold compensates for attenuation; it does not recover the *power* blur
  costs, and past some blur a real deficit is unrecoverable at any threshold.
  `selftest` already measures this against planted ground truth — it read **87.8%
  (min 50%, max 100%)** on the first sweep, and the floor is the part that matters,
  because a config that labels at chance contributes noise to the ranking no matter
  what the threshold is. **Decision rule: if per-config label accuracy has a floor
  below ~70%, fix the labeller before touching `min_gap`.**
  The cheap structural fix first, since it needs no training: `label_trace` is a
  single ungrounded pass, which is the architecture AgentV-RL (Findings of ACL
  2026) argues is unreliable, and we already store per-label `evidence` and throw
  it away. A second pass re-checking each LACKING verdict against its own cited
  evidence is one extra call per trace. Their forward/backward split is compatible
  with our outcome-blinding — their backward agent reasons backward through the
  *solution's logic*, and never sees the ground-truth verdict. Their train-free
  variant got ~2.6 points from structure alone, before any training, which is the
  order of gain to expect and the reason to try it before anything expensive.
- **Print N beside every number.** Don't rank a capability measured on 3 traces
  next to one measured on 40. And when we report the top of a ranked list of 4–8
  capabilities, remember the top of a noisy list looks good even when nothing is
  there.

**Stop replaying when the ranked list stops changing** as tasks are added. If it's
still reshuffling at the budget cap, the answer is "not enough data," not a report.

Provenance
----------

Every run writes: arm, task ids and split, scaffold name+version, base model and
adapter, seed, pass rate with N, the paired comparison, and the environment
(GPU/harbor/image digest). If a number isn't re-derivable from disk, it doesn't
count.

Stop and rethink if
-------------------

- The gap is under ~10 points, or the candidate has almost no wins to bootstrap from.
- A prompt (A1) closes most of the gap — then we're a prompt tool with an expensive
  backend. Thesis-level finding for two eval runs.
- The labeller floor sits below ~70% and no structural fix moves it.

`PLAN.md` extends this list with the routing-specific conditions (no `pass@k`
regime separation, sparse routing cells, B1 ≈ B2, quarantine fraction too large).

Done when
---------

- Real trajectories parse into turns with tool calls and results; no fallback path
  exists; golden-file test passes. ✅
- Infra failures never appear as losses. ✅
- Diagnosis is blind to outcome and can return "none found" with exit 0. ✅
- A container check confirms base commit, scrubbed `.git`, no remote; a
  shadow-`python3` reward hack scores 0.0. ✅
- A gap number exists on ≥50 mined tasks at a pinned, named scaffold.
- Labeller accuracy is stated with its per-config **floor**, not just its mean, and
  either clears ~70% or was fixed before `min_gap` was calibrated.
- The experiment's acceptance criteria — see `PLAN.md`.

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
  and the real outcome exist — `PLAN.md` Step D is that validation.
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
- **Multi-deficit + adapter routing** (TRACE's MoE gate), **multi-language** (Java
  and Ruby are realistic; Rust and Go put table-tests inside the source file, which
  makes patch/test splitting structurally impossible), **MCP surface** so a coding
  agent can drive the pipeline in one instruction.
- **Cross-tokenizer distillation** (ULD/MinED/byte-level) — the upgrade path if the
  same-family teacher proves too weak. See `docs/OPD.md`.
- **Use the gold patch.** Mined tasks already carry the human's fix, F2P names, and
  changed files, and diagnosis currently sees only turns. Comparing the trajectory
  against the real fix is probably the biggest accuracy upgrade available.
- **Formal sequential stopping** — proper anytime-valid boundaries (accept / reject /
  continue) instead of v0's rank-stability heuristic. Matters when traces stream in
  continuously and we'd otherwise peek after every batch. Also the right frame for
  `PLAN.md`'s two-stage `pass@k` escalation.
- **Mining robustness and yield:** a fresh-dependency check between base commits
  (one sandbox reused across all PRs gives false F2P/P2P), and the remaining yield
  bugs.
- **Go-to-market:** public repos are the demo generator — mine a prospect's own
  open-source repo, run paired replay, lead with their number. No permission, no
  data handoff, no meeting needed to produce it.

Open questions
--------------

1. **Which candidate model** — 4B or 8B class. Decide from the measured gap,
   not from benchmark tables.
2. **Which scaffold** gets pinned, and who owns freezing it.
3. **How many held-out tasks**, and what effect size that resolves. If ~50 can't
   resolve the expected effect, mine more before training.
4. **Canonical win/loss** — strict `resolved`, or `reward = f2p_rate × p2p_rate ≥
   1.0`? `grade()` computes both; pick one and say which in every report.
