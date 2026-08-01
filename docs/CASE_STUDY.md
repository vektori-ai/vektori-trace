# Case Study Execution Plan — DRAFT FOR REVIEW

**Date:** 2026-07-31 · against `main` @ `72adbff` (PRs #18/#19/#20 merged)

**P0, unchanged:** ship a public case study with real numbers on one repo.

1. Baseline `pass@k` for k ∈ {1,4,8,32}, frontier vs. open model
2. The top 3 diagnosed capability gaps
3. The synthetic environment generated for the #1 gap
4. Post-training results: pass rate before/after, cost per 1M tokens before/after
5. A validity-proof screenshot

Everything below serves those five artifacts. Nothing is added that doesn't.

---

## 1. Deliverable → what produces it

Every subcommand and flag here was read off `cli.py` on `main`, not invented.

| # | Deliverable | Command | Exists? |
|---|---|---|---|
| 1 | `pass@k` curves | `vektori-trace passk --tasks-dir … --agent … --model …` (twice: once per model) | coded, **never run on real tasks** |
| 2 | Top-3 gaps | `vektori-trace replay` → `vektori-trace diagnose --frontier-model … --candidate-model …` | coded, never run at scale |
| 3 | Synthetic env for gap #1 | `diagnose` writes the Harbor task automatically for the top-ranked deficit | coded, run once (`examples/`) |
| 4a | Pass rate before/after | `vektori-trace select` → `vektori-trace run-arms` (A0–A4) | coded, never run |
| 4b | Cost per 1M tokens | not a pipeline output — see §6 | needs a defined metric |
| 5 | Validity proof | `vektori-trace diagnose --prove` or `vektori-trace prove --base-agent … --base-model …` | coded, run once |

`passk` measures **one model per invocation** — there is no `--frontier-model`
flag on it. Deliverable 1 is two runs whose JSON reports are then fed to
`route --student-passk … --teacher-passk …`.

---

## 2. Picking the one repo

This is the highest-leverage decision in the whole plan, because at the observed
yield the repo determines whether deliverable 4 is resolvable at all.

### Criteria, in priority order

1. **Post-cutoff PR volume.** The candidate is Qwen3-8B (cutoff ~2024). Mining
   only PRs merged after it gives a contamination-free corpus — see §5.
2. **Linked-issue rate.** The binding constraint everywhere. `docs/mine-results.md`
   measured it directly: jsonschema **0/60**, attrs **7/60**, structlog **14/60**.
   A repo at jsonschema's rate yields nothing at the defaults.
3. **Not in SWE-bench.** Rules out astropy, django, flask, matplotlib, pylint,
   pytest, requests, scikit-learn, seaborn, sphinx, sympy, xarray. Those twelve
   are in everyone's eval sets and arguably in the weights.
4. **Pure Python, fast suite.** One bootstrap image must build every base commit,
   and validation runs the suite twice per PR. structlog's 928 tests in 2.3s is
   the shape you want.
5. **Application-level bugs.** See §3 on why this matters more than it looks.

### Shortlist

| Repo | Post-cutoff PR volume | Issue linking | Suite | Risk |
|---|---|---|---|---|
| `home-assistant/core` | very high (~1k/mo) | strong, enforced by template | per-integration, fast | bugs are device/integration-specific |
| `apache/airflow` | high | strong | heavy, slow to build | bootstrap cost per base commit |
| `prefecthq/prefect` | moderate–high | good | pure Python, moderate | smaller pool |
| `dask/dask` | moderate | good | pure Python, moderate | may not reach N alone |
| `pydantic/pydantic` | high | good | fast | `pydantic-core` Rust coupling per base commit |

**Recommendation: `home-assistant/core`.** It is the only one where post-cutoff
volume is large enough that you can filter hard — post-cutoff *and* linked issue
*and* band-resident — and still land at N. Its per-integration test isolation
also keeps P2P counts naturally small, which §3 argues is the thing that went
wrong with the current 22.

**Do not commit to it on my say-so.** Run the probe first (§4, Phase 0). It is
cheap, it is the same measurement `mine-results.md` already did for three repos,
and it decides this empirically.

---

## 3. What the current 22 tasks teach us about corpus design

Not a digression — this is the spec for deliverable 1, and getting it wrong
makes deliverables 2–4 unproducible.

The routing rule sorts on candidate `pass@k`:

| Rule | Candidate | Classification | Route |
|---|---|---|---|
| R1 | pass@1 ≤ 0.25, **pass@32 > 0** | in support, unreliable | **RL** |
| R2 | **pass@32 = 0** | outside student support | **OPD** |
| R4 | pass@1 ≥ 0.75 | no deficit | excluded |

For the case study to show a *decision*, the corpus must populate more than one
row. The 22 on your friend's machine almost certainly populate only R2/R3:

- **anyio-1121 / 1149 / 1180** — KeyboardInterrupt cancellation, Trio
  `getfixturevalue` hangs, `OutcomeException` discarding the runner task.
  Structured-concurrency internals; an 8B scores 0/32, not 1/32.
- **click-3484** (3 F2P / **595 P2P**), **click-3678** (9 / **658**),
  **click-3704** (1 / 293). Flip one test without disturbing six hundred.
- **anyio-1211** — an event-loop/root-task reference cycle causing an OOM leak.

`arms.py` already declares the target band: `passrate-min 0.10`,
`passrate-max 0.40`. These sit below it.

Two secondary problems:

**Some aren't bugs.** click-3473 adds a `help=` kwarg to `@click.argument`;
click-3704 is a deprecation; tenacity-604 adds `enabled=False`; tenacity-609 is
an explicit refactor. `non_bug_pr` passed them. An agent solving those is doing
API design, not debugging.

**The 4 audit failures probably aren't what the inventory says.**
`mining/inspect.py:94` already strips the parametrisation suffix
(`symbol.split("[", 1)[0]`), and its docstring says that was added *because* the
un-stripped check flagged 2 of 4 sound structlog tasks. So the stated cause is
already handled in code. Read the real `reason` fields in `mine-report.json`
before assuming those 4 are recoverable.

### Corpus spec that follows

- **Band-resident, measured not assumed.** `passk` stage 1 is n=8; a task with
  1–3 passes out of 8 is band-resident. That falls out of deliverable 1 for
  free — no separate screening tool needed.
- **Cheap pre-screen if the pool is large.** `--stage1-n 4` over a wide pool,
  keep the middle, then run the real n=8/32 sweep only on survivors.
- **Cap the F2P:P2P ratio.** A 1:595 task measures "don't break anything" more
  than "fix this." Report the ratio distribution even if you don't filter on it.
- **Tighten `non_bug_pr`** or hand-drop feature/refactor PRs before replay.

---

## 4. Phases

### Phase 0 — Repo probe + persistence · ~1 day · ~$0

Two things, both blocking.

**Probe the linked-issue rate** on the shortlist before spending a bootstrap:

```bash
for repo in home-assistant/core apache/airflow prefecthq/prefect dask/dask; do
  vektori-trace mine --repo "$repo" --limit 60 \
    --no-replay --skip-validation --out "./probe-$(basename $repo)"
done
```

Read the skip histogram in each `mine-report.json`. Pick on measured
`no_linked_issue` rate, not on my table above.

**Fix persistence.** `vektori-out-*/` is gitignored, which is why the structlog
corpus is gone and why the current 22 exist on exactly one machine. Before
mining at scale: commit a manifest (task ids, base commits, F2P/P2P counts,
image digests) even if the task bodies stay out of git. Nothing downstream is
reproducible — or publishable — without it.

**Gate:** a repo with a linked-issue rate ≥ structlog's 14/60, and a committed
manifest format.

---

### Phase 1 — Mine the corpus · ~3–5 days · low $

```bash
vektori-trace mine \
  --repo <chosen> \
  --dockerfile examples/dockerfiles/<repo>.Dockerfile \
  --test-cmd "<suite command>" \
  --limit 1200 \
  --no-replay \
  --out ./case-study/mined
```

**Arithmetic.** `mine-results.md` measured 10% yield on structlog and states the
implication: *"~400 merged PRs to reach 50 tasks."* Targeting ~150 emitted tasks
means screening **~1,200–1,500 merged PRs**. Restricted to the post-cutoff
window (2025-01 → 2026-07), only the high-volume end of the shortlist has that.

**Gate:** ≥120 tasks emitted and passing audit. Below that, the one-repo
constraint has failed on volume and the choice is a bigger repo or a second one.

---

### Phase 2 — `pass@k` sweep · ~1 week wall-clock · **$500–2,000 (est.)**

Deliverable 1. Two runs.

```bash
# Candidate (self-hosted vLLM — this is what #18's --api-base is for)
vektori-trace passk --tasks-dir ./case-study/mined/mined_tasks \
  --agent terminus-2 --api-base http://<ip>:8000/v1 \
  --stage1-n 8 --stage2-n 32 --max-workers 32 \
  --out ./case-study/passk-candidate

# Frontier
vektori-trace passk --tasks-dir ./case-study/mined/mined_tasks \
  --agent terminus-2 --model gpt-5 \
  --stage1-n 8 --stage2-n 32 --max-workers 32 \
  --out ./case-study/passk-frontier
```

**Rollout arithmetic (150 tasks).** Stage 1: 150 × 8 × 2 models = 2,400.
Stage 2 escalates only 0/8 tasks — if ~50% of candidate and ~10% of frontier
tasks are zeros, that's (75 × 32) + (15 × 32) = 2,880. **≈5,300 rollouts.**

At ~3 min each that is ~265 hours serial. **Container throughput, not GPU, is
what will blow this timeline** — `--max-workers 32` gets it to ~8 hours, and
that number needs to be true.

**Cost is dominated by frontier tokens, not GPU.** ~2,880 frontier rollouts at
an assumed 30–80k tokens each ≈ 90–230M tokens. At blended frontier pricing that
is roughly **$500–2,000**. Both the per-rollout token count and the blended rate
are assumptions — measure on the first 10 tasks and re-derive before committing.

**Gate — the real one.** Feed both reports to routing:

```bash
vektori-trace route \
  --student-passk ./case-study/passk-candidate/passk.json \
  --teacher-passk ./case-study/passk-frontier/passk.json \
  --out ./case-study/routing
```

Do the curves separate into R1/R2/R4 regimes? If everything lands in R2/R3, there
is no routing decision to publish. That is a real, publishable finding about the
task distribution — and it is much better to learn it here than after the GPU
spend. `PLAN.md` already names this: *"underpowered is a first-class result."*

---

### Phase 3 — Diagnosis, environment, validity proof · ~2–3 days · low $

Deliverables 2, 3 and 5.

```bash
vektori-trace replay --tasks-dir ./case-study/mined/mined_tasks \
  --agent terminus-2 --frontier-model gpt-5 \
  --candidate-model qwen3-8b --candidate-api-base http://<ip>:8000/v1 \
  --out ./case-study/replay

vektori-trace diagnose --manifest ./case-study/replay/replay-manifest.json \
  --frontier-model gpt-5 --candidate-model qwen3-8b \
  --prove --base-agent claude-code --base-model <model> \
  --out ./case-study/diagnosis
```

`diagnose` ranks deficits (deliverable 2), writes the Harbor task isolating the
top-ranked one (deliverable 3), and `--prove` runs the validity proof —
oracle solution passes, a real agent lacking the capability fails (deliverable 5).

Note `--min-gap` defaults to 0.20 and the README says plainly that the threshold
is **uncalibrated**. For a public artifact, either calibrate it against
hand-labelled traces or state in the writeup that it's a placeholder. Do not
publish a ranked list that depends on an arbitrary cut without saying so.

> **→ PUBLISH PART 1 HERE.** Deliverables 1, 2, 3, 5 are done and need no GPU
> training. "Here is where the headroom is, here is how we measured it, here is
> a generated environment that provably isolates it" is a stronger and more
> defensible artifact than most one-pagers, and it de-risks Part 2 by putting
> the method in front of critics before the expensive claim depends on it.

---

### Phase 4 — Train and measure · ~2–3 weeks · **$2,000–5,000 GPU (est.)**

Deliverable 4.

```bash
vektori-trace select --manifest … --diagnosis … --tasks-dir … --agent terminus-2 \
  --passrate-min 0.10 --passrate-max 0.40 --holdout-frac 0.3 --seed 0 \
  --out ./case-study/selection

vektori-trace run-arms --selection … --diagnosis … --tasks-dir … \
  --api-base http://<ip>:8000/v1 --out ./case-study/arms
```

A0 (base) vs A4 (frontier ceiling) gives before/after. The B1-vs-B2 arms —
routed assignment vs permuted assignment, task count and compute held identical
— are what isolate the routing decision from "training helped."

---

## 5. The claim nobody else can make

**Contamination-freshness is absent from the P0 spec and it is your strongest
asset.** Every task mined from a post-cutoff PR provably postdates the candidate
model's training data. Contamination is the standing objection to every published
coding-benchmark number, and SWE-bench-derived results cannot answer it.

Cost to add: one date filter and a merge-date column in the results table.
Recommend making it a headline of the writeup, not a footnote.

---

## 6. Two things in the P0 spec that need a definition

Neither changes the deliverable list. Both need a decision before writing.

**"Cost per 1M tokens before/after."** A LoRA on the same base model serves at
the same price — this number does not move from training. It moves when the
*model* changes, which is the actual story: a 30B teacher's capability landing on
an 8B student. Frame it as the serving-cost delta of that swap, with training as
what makes the swap survivable. `PLAN.md`'s own metric — **cost per solved
task** — captures both capability and price in one number and is the one I'd
lead with.

**Statistical power on one repo.** `PLAN.md` states the constraint directly:
*"SWE-Gym moved a 7B model ~3 points; a 50-task slice cannot resolve 3 points."*
For a paired McNemar on B1 vs B2: if routing flips ~8% of tasks and un-flips ~2%,
discordant pairs run ~10% of N. At N=150 that's ~15 discordant, p≈0.03 —
directionally right but fragile. At N=300, p≈0.001.

This is why §2's repo choice is the highest-leverage decision here: one repo can
carry deliverable 4 **if and only if** it is big enough to reach N≈150 band-
resident tasks post-cutoff. If Phase 1 lands under 120, the honest options are
to scope deliverable 4's claim to what it resolves, or add a second repo.

---

## 7. Risks

| Risk | Signal | Mitigation |
|---|---|---|
| Corpus lands entirely in R2/R3 | Phase 2 routing report has one populated row | Phase 2 gate; re-mine toward the band |
| One repo can't reach N | Phase 1 emits <120 | pick higher-volume repo, or scope the claim |
| Container throughput | Phase 2 wall-clock >> 8h at 32 workers | measure on first 10 tasks; raise workers or cut N |
| Frontier token spend overruns | measured tokens/rollout >> 80k | re-derive after 10 tasks, before the full sweep |
| `--min-gap` is uncalibrated | — | calibrate, or state it in the writeup |
| Nothing persists again | — | Phase 0 manifest, blocking |

---

## 8. Open questions for you

1. **Repo** — probe the shortlist, or do you already have a friendly customer
   repo? A customer repo is a much better narrative and I'd weight that heavily
   over my table, provided it clears the linked-issue rate.
2. **Publish in two parts, or hold everything for one drop?**
3. **Cost metric** — cost per solved task, serving cost of the 30B→8B swap, or
   both?
4. **Who runs Phase 2?** It needs the vLLM endpoint from #18 standing up, which
   has never been exercised against a real server.
