# Cross-tokenizer OPD run — DeepSeek-V4-Flash → Qwen3-14B on 2 tasks

**Status:** approved 2026-08-13, Phase 0 executing.
**Self-contained by design** — a fresh model with no chat history should be able
to execute this cold. Everything needed is below or at a named path.

**After approval, step 0: copy this file to `docs/OPD-RUN-PLAN.md`** so it lives
in the repo rather than only in `~/.claude/plans/`.

---

# 0. Phase 0 results — 2026-08-13, ALL GATES PASSED

Run on the box. Zero GPU, ~15 Fireworks calls, well under $0.01.

| Gate | Result |
|---|---|
| Offline suite | **897 passed, 4 skipped** (incl. `test_distill_cross`, `test_align`, `test_cross_kl`, `test_vocab_bridge`, `test_teacher_cross`) |
| Fireworks model exists | `accounts/fireworks/models/deepseek-v4-flash-0731`, ctx 1,048,576 — the exact id `tier_run.sh` used |
| Bridge | `/data/bridge.json` — **84,030 byte-identical pairs, matching §4's independent measurement exactly** |
| Teacher tokenizer | `deepseek-ai/DeepSeek-V4-Flash-0731` @ HF rev `7872f01b` |
| Student tokenizer | `Qwen/Qwen3-14B` @ HF rev `40c06982`, vocab 151,669 (= §4's 151,643 + 26 added) |
| Alignment (0.3) | 571 real lines from the 5 prefixes: **mean granularity 0.9250, min 0.7000**, floor is 0.5, zero alignment errors |
| **P0 echo** | **PASS** — `score_ids` 40/40 logprobs; `score_ids_topk(K=5)` row widths 5-6; exit 0 |
| P3 | PASS — ids echoed back in order, length always matches |
| P4 determinism | **5/5 runs bit-identical, max deviation 0.000e+00** — no observable fp8 noise |
| **P4 coverage** | **prob-weighted mapped mass: mean 0.994, median 1.000, min 0.825**; count fraction mean 0.991; sampled token mapped 40/40 |
| **P4 estimator split** | **39/40 spans (98%) pass the 0.9 gate → estimator A.** Not in SimCT's SimpleOPD trap. |
| P6 | PASS — `logP(teacher's own) = -95.36` vs `logP(corrupted) = -382.90`, ratio **287.5**. Alignment is not scrambled. |

**The 65.6% static bound was pessimistic by a wide margin.** `FINAL-PLAN.md` §4
guessed probability-weighted coverage "likely higher since common tokens overlap
more"; measured, it is ~99%. Estimator A carries 98% of spans, so the analytic
low-variance path is the main path as designed.

**Phase 1 done (no spend):** `/data/opd_prefixes/` holds the 5 passing rollouts,
each verified `reward == 1.0` before copying. Actual corpus:

| rollout | turns | parent assistant turns |
|---|---:|---:|
| `pallets__click-3466-0` | 51 | 25 |
| `pypa__hatch-2086-0` | 31 | 15 |
| `pypa__hatch-2086-1` | 29 | 14 |
| `pypa__hatch-2086-2` | 45 | 22 |
| `pypa__hatch-2086-3` | 43 | 21 |

**97 OPD examples** (not the ~102 estimated) — `hatch-2086` 72 (74%),
`click-3466` 25 (26%). At `--max-steps 50 --examples-per-step 4` that is
**2.06 epochs**.

Artifacts on the box: `/data/bridge.json`, `/data/opd_align_samples.txt`,
`/data/p4p6_report.json`, `/data/opd_prefixes/`, `/tmp/p0_tests.log`,
`/tmp/p0_bridge.log`.

**Next step needs GPU and therefore separate approval.** Nothing below this line
has run.

---

# 1. Context — why this run exists

## The goal

`vektori-trace` mines real GitHub bugfixes into verifiable agentic tasks, then
measures whether a small student model can solve them, then distils a large
teacher into that student. This run is the **first end-to-end exercise of the
cross-tokenizer OPD path** (`FINAL-PLAN.md`), on the two tasks where a
teacher/student capability gap has actually been measured.

## The measured gap

Two sweeps happened on 2026-08-12, both on `cs/corpus50_v3` tasks:

**Qwen3-14B, "run6"** — 6 tasks × 4 rollouts, vLLM on Modal L40S.
**Result: 0 solves in 24 rollouts.** Full forensics in
`docs/qwen14b-failure-analysis.md` (1,115 lines, untracked, local only).

**DeepSeek-V4-Flash-0731 via Fireworks, "out_trivial"** — 4 trivial-tier tasks ×
4 rollouts, `/data/tier_run.sh trivial` on the box. Clean run, zero infra
failures:

| task | DeepSeek c/n | Qwen3-14B |
|---|---|---|
| `pypa__hatch-2086` | **4/4** | 0/4 |
| `pallets__jinja-1702` | **4/4** | not run |
| `pallets__click-3466` | **1/4** | 0/4 |
| `pallets__jinja-1663` | 0/4 | not run |

**Intersection = `pypa__hatch-2086` and `pallets__click-3466`.** Those are this
run's training tasks. `hatch-2086` is the anchor: trivial tier (6 LOC gold patch,
1 file, 1 F2P), teacher solves it 4/4, student fails 4/4.

## Why the student fails (from the run6 forensics)

Not a formatting or capability-to-emit problem — a **strategy** problem:

- All 4 Qwen `hatch-2086` rollouts patched `.github/workflows/*.yml` to pin
  `hatch==1.14.2` — the bug reporter's *workaround* — instead of touching the
  resolver named by the F2P test. **None ever entered `src/`.** Two finished in
  under 250 s. Repo correctly located; four fast, confident, wrong answers.
- Across all 24 run6 rollouts: only 2 opened a source file before editing (both
  then overwrote it wholesale); literal-`\n` corruption in 12/24; tried to
  clone/download the repo in 13/24; never modified tracked source in 17/24;
  `git diff` / `git checkout -- <file>` used **0 times in 24 rollouts**.
- 10 of 24 timed out at 1800 s, every one in a verbatim loop or chasing blocked
  network. Longest identical-batch repeat: 21. **Raising the timeout buys
  nothing.**

This matters for method choice: the student's tokens are fluent and well-formed,
so the failure is **mode selection inside existing support**, which is what
reverse KL is for.

## Also established (correct the record if you find contradicting notes)

- run6's headline is **0/24, not 0/19**. `passk.json` excluded 5 rollouts as
  `infra_failures` / `fallback_exitcode`. All 5 were model-caused: three prefect
  exclusions were pytest exit 4 after the model corrupted a file conftest
  imports; `anyio-1211-1` ran `sed -i '/finally:/d'` on `_asyncio.py`;
  `click-3466-0` ran `exec /usr/bin/bash --login -c "pacman …"`, replacing the
  tmux pane's only shell and killing the server. The exclusion drops the *most
  engaged* rollouts and keeps inert ones — prefect's published pass@1 rests on a
  rollout whose patch is 0 bytes.
- **There was never a DeepSeek run over 10 trivial tasks.** `vektori-out/prove12`
  is 12 tasks at reward 1.0 with **no `config.json` and no agent output** — that
  is the *oracle prove* (gold patch applied, verifier run), not a model result.
- The corpus has **10 trivial-tier tasks** (recomputed from gold-patch shape:
  ≤10 LOC, ≤2 files, ≤2 F2P): `click-3299` (P2P 461, excluded by the
  P2P≤200 rule), `click-3466`, `click-3534`, `jinja-1614`, `jinja-1663`,
  `jinja-1702`, `jinja-1762`, `jinja-1852`, `hatch-2086`, `hatch-2266`.
  Trivial exists only in click/jinja/hatch — anyio and prefect have none.

---

# 2. Decisions already taken (do not relitigate)

| Decision | Choice | Why |
|---|---|---|
| Student | **Qwen3-14B** | Matches the measured 0/4 gap exactly, so the before-picture already exists. Qwen3-8B/14B share a tokenizer, so the bridge is valid for either. |
| Teacher | **DeepSeek-V4-Flash-0731 on Fireworks** | No teacher GPU. Scores via echo. |
| `--cross-top-k` | **5** | Do NOT set 0. See §4. |
| First action | **All zero-GPU gates first, report, then decide** | No Modal credits spent until explicitly approved. |
| Eval set | **2 trained + 2 held-out** | `click-3466`, `hatch-2086` (trained) + `jinja-1702` (DeepSeek 4/4, unseen) + `jinja-1663` (DeepSeek 0/4, negative control). |
| SFT control arm | **Skipped this round** | `train` is self-rejection-sampling SFT, not SFT-on-teacher-traces; a control needs ~30 lines of new glue. Revisit after the pipeline is proven. |
| SFT warm start | **No** | See §4. |
| Fireworks calls | **Run from the AWS box** | Key already there: `/data/.env.fw`. |

---

# 3. Hard rules

- **Never start a Modal endpoint, allocate a GPU, or launch a pass@k / rollout
  sweep without explicit per-run approval.** This plan being approved is NOT
  approval to execute its GPU steps. Ask again, per run.
- **Tearing down needs no approval** — kill idle endpoints/containers the moment
  a run finishes or is abandoned, every time.
- Golden corpus is `/data/vektori-trace/cs/corpus50_v3`, layout is **per-repo**:
  `<root>/<repo>/mined_tasks`. `<root>/mined_tasks` does not exist and silently
  finds nothing. Legacy dirs (`cs/corpus50*`, `cs/smoke`, `/data/mining/out/*`,
  `qwen-stuff-14B/`) carry `LEGACY.md` — never point tools at them.
- Box is EC2 `i-0a348ff3d7be9769a` (ap-south-1), **AWS SSM only**, no SSH:
  `aws ssm send-command --document-name AWS-RunShellScript`. Repo at
  `/data/vektori-trace`. Heredocs are unreliable over SSM and the box python3
  has no `tomllib` — to run python remotely, base64 the script and
  `echo <b64> | base64 -d > /tmp/x.py && python3 /tmp/x.py`.
- SSM caps command output at ~24 KB. Chunk anything larger.

---

# 4. The method, in enough detail to defend it

## Objective

Reverse KL, student-weighted: `Σ_e π_s(e)·(log π_s(e) − log π_t(e))`.
Mode-seeking — the student commits to one teacher mode instead of hedging.
Teacher side always `.detach()`ed.

## Two estimators, 100% span coverage

**(A) Analytic — primary, ~90% of spans.** For 1↔1 byte-identical spans. Build a
K+1 partition from the teacher's top-K: the K reported tokens plus `other`, whose
mass is `1 − Σ exp(lp_i)` — **exact arithmetic, not an assumption of zero**. Map
each teacher id to a byte-identical student id (§8 exact-byte map); unmapped
entries fall into `other`, which stays exact. Zero sampling variance.
*Coverage gate:* if mapped teacher mass < ~0.9 of reported top-K mass, demote the
span to (B).

**(B) Sampled — the ~10% (A) cannot reach.** Multi-token spans and gate failures.
`logP_s = Σ log π_s(span tokens)` (differentiable), `logP_t = Σ log π_t(...)`
(scalar, detached, from echo) → `opd.reverse_kl_surrogate` unmodified.

Normalization: **one shared global denominator per optimizer step** — total
supervised student tokens across the batch across *both* estimators. Never
per-estimator.

## Why K=5 is fine, and why `--cross-top-k 0` is wrong

KL over a coarse-graining is a **lower bound** on the true KL (data-processing
inequality), monotone in K. K=5 costs tightness, not validity. Literature runs
K=16 (thunlp/OPD) and K=20 (Phan et al., ICLR 2026) — you get a weaker bound, not
broken math.

Setting `cross_top_k=0` **disables estimator A entirely** and dumps every span on
the high-variance sampled path. The default of 5 is deliberate:
`distill.py:105` — *"Fireworks caps at 5 (FINAL-PLAN.md §2)."*

`TOP_LOGPROBS_CAP = 5` (`fireworks.py:66`) **raises rather than clamps**, so
asking for 16 fails loudly instead of silently changing the objective. The
sampled token's own logprob is always present and is **merged into A's rows
explicitly** (`fireworks.py:169-175`) because, unlike vLLM, Fireworks does not
promise the prompt's own token appears among the alternatives.

## Why no SFT warm start (this round)

SFT is MLE on a fixed corpus with one-hot targets: off-policy states, zero-entropy
targets. Its real function before OPD is **adding support** — reverse KL is
mode-seeking *within the initialization's support*, so if the student assigns ≈0
to the right tokens there is no mode to seek.

Three reasons it is not needed here:
1. Qwen3-14B is already instruct-tuned and emits fluent, well-formed tool calls.
   The failure is strategy, not support. The cold-start literature warm-starts
   from verbose *base* models.
2. The corpus is **~102 examples**. Warm-starting on the same 5 trajectories then
   running OPD on them leaves OPD nothing new.
3. This run's job is to prove the loss works end-to-end. A second training stage
   before that is a confound.

**Known risk, watch for it:** SimCT (arXiv 2605.07711), the closest published
work — cross-tokenizer OPD — reports that supervision restricted to the *exact
shared vocabulary* ("SimpleOPD") improves only marginally over its SFT warm
start, because high-probability teacher predictions often fall outside the shared
vocab even at aligned positions. Our estimator A drops unmapped mass into an
undifferentiated `other` bucket: exact, but no gradient *direction* among it.
**P4 measures whether we are in that trap.** If mapped-alternative coverage is
low, the exact-byte map is the suspect, not K.

## Monitoring obligation

Log `student_entropy` from step 0. Reverse KL collapses entropy by construction;
**monotone decline = mode collapse**, and a collapsed student makes the
downstream GRPO arm a no-op (`opd.grpo_advantage` returns zeros when group reward
std < 1e-8). Tripwire to watch, not a reason to change objective.

---

# 5. Inventory — what exists, where

## Data

| What | Path | Notes |
|---|---|---|
| Golden corpus | box `/data/vektori-trace/cs/corpus50_v3/<repo>/mined_tasks` | 60 tasks, 5 repos, tmux baked in |
| DeepSeek trivial sweep | box `/data/out_trivial/` | `passk.json`, `passk_jobs/stage1/<task>-<n>/` |
| Its launcher | box `/data/tier_run.sh`, tasks at `/data/tier_trivial/` | |
| run6 artifacts (Qwen) | local `~/vektori-out-run6-full/` | 11 MB, 24 rollouts, gpu_log.jsonl |
| run6 analysis | `docs/qwen14b-failure-analysis.md` | 1,115 lines, untracked |
| Fireworks key | box `/data/.env.fw` | `FIREWORKS_API_KEY`, `FIREWORKS_AI_API_KEY` |
| Box venv | `/data/vektori-trace/.venv` | train extra installed (torch/transformers/peft present) |

## The 5 prefix trajectories (DeepSeek passes on the 2 tasks)

Under `/data/out_trivial/passk_jobs/stage1/`:

| rollout dir | reward | steps |
|---|---|---|
| `pallets__click-3466-0` | 1.0 | 26 |
| `pypa__hatch-2086-0` | 1.0 | 16 |
| `pypa__hatch-2086-1` | 1.0 | 15 |
| `pypa__hatch-2086-2` | 1.0 | 23 |
| `pypa__hatch-2086-3` | 1.0 | 22 |

**~102 examples** (`iter_reopd_examples` yields one per parent assistant turn:
`role == "assistant"` and `subagent_depth == 0`). **80% are hatch-2086** — state
that skew next to any result.

## Code map

| Thing | Path |
|---|---|
| OPD loss, both estimators | `vektori_trace/opd.py` (`reverse_kl_surrogate` :118, `topk_reverse_kl`) |
| Training loop | `vektori_trace/distill.py` (`run_opd_training` :493, `OPDTrainConfig`) |
| Fireworks teacher | `vektori_trace/providers/teacher/fireworks.py` (`score_ids` :149, `score_ids_topk` :166) |
| Cross-tokenizer pool | `vektori_trace/providers/teacher/cross.py` |
| Bridge / byte map | `vektori_trace/vocab_bridge.py`, `align.py`, `cross_kl.py`, `encoding_dsv4.py` |
| Example generation | `vektori_trace/reopd.py` (`iter_reopd_examples` :122) |
| Trace loading | `vektori_trace/cli/commands/distill.py` (`_load_teacher_trajectories` :13) |
| SFT tokenization + masking | `vektori_trace/dataset.py` (`tokenize_sft_example`) |
| Modal wrapper to copy | `vektori_trace/train.py` (`train_lora_modal` :334) |
| Probe | `vektori_trace/cli/commands/teacher.py` (`cmd_probe_teacher` :14) |
| Reference spec | `FINAL-PLAN.md` (777 lines), `docs/OPD.md`, `docs/HOSTED_TEACHERS.md` |

## What does NOT exist yet

1. **`bridge.json`** — nowhere on box or local. Must run `build-bridge`.
2. **`distill_modal`** — `distill.py` runs in-process on whatever GPU invoked it.
   Copy `train_lora_modal` around `run_opd_training`.
3. **Teacher-prefix cache** — see Phase 3.
4. **SFT-on-teacher-traces path** — `train` is self-rejection-sampling only.

---

# 6. Execution

## Phase 0 — zero-GPU gates. Do all of this first, then STOP and report.

Run on the box (has the Fireworks key and the venv):

```bash
cd /data/vektori-trace
export PATH=/data/vektori-trace/.venv/bin:$PATH
set -a; . /data/.env.fw; set +a
```

**0.1 Offline suite** — the equivalence oracle is the critical one: the
cross-tokenizer path run with an *identical-tokenizer pair* must equal plain
`reverse_kl_surrogate` to float tolerance. It is the only test that catches
off-by-one span reindexing, which otherwise yields finite, plausible losses
forever.
```bash
uv run pytest tests/ -q
# specifically: test_distill_cross.py test_align.py test_cross_kl.py
#               test_vocab_bridge.py test_teacher_cross.py test_distill.py
uv run ruff check .
```

**0.2 Tokenizer check + bridge** (offline, downloads two tokenizers):
```bash
uv run vektori-trace check-tokenizers
uv run vektori-trace build-bridge \
  --teacher-tokenizer deepseek-ai/DeepSeek-V4-Flash-0731 \
  --student-tokenizer Qwen/Qwen3-14B \
  --thinking-mode chat \
  --out /data/bridge.json
```
Expect ≈ **84,030** byte-identical pairs (55.4% of Qwen's 151,643; 65.6% of
DeepSeek's 127,997). Both are ByteLevel BPE and all pieces round-trip with zero
failures — asserted, not assumed. Note `config.json` says `vocab_size: 129280`;
that is the padded embedding matrix, not the tokenizer — pad-range ids fall into
`other` and stay exact.

**0.3 Alignment report** (offline). Feed real content from the 5 trajectories.
Gate: granularity ≥ 0.5, reported **per content type**.
```bash
uv run vektori-trace align-report --bridge /data/bridge.json --text <samples.txt>
```
Reference from `FINAL-PLAN.md` §4: python source 0.980, trace JSON 0.986,
markdown 0.945, tool-call JSON 0.977, numeric-heavy 0.667. Numeric spans are ≤3
student tokens, far under the 8-token hard-fail. **Do not normalize numbers out
of the data** to flatter the metric.

**0.4 P0 — the echo probe. This is the gate the whole method rests on.**
```bash
uv run vektori-trace probe-teacher \
  --backend fireworks \
  --model accounts/fireworks/models/deepseek-v4-flash-0731 \
  --echo --top-k 5
```
Request shape: `prompt=[ints]`, `max_tokens=1`, `temperature=0`, `echo=true`,
`logprobs=true`, `raw_output=true`, `top_logprobs=5`. Confirm
`choices[0].logprobs.content` carries `token_id` per entry.

Exit 0 → continue. Exit 1 → **STOP.** Per `FINAL-PLAN.md` §"If P0 fails": fall
back to teacher-intervention/DAgger (`reopd.TeacherContinuationExample`) or
rejection-sampling SFT — **methods that already work, not a weakened version of
this one.** Surface it as a decision, do not auto-switch.

*Already known:* two earlier probes hit this model
(`vektori-out/teacher-dsv4*/probe.json`) and established reachability,
`max_top_logprobs: 5`, and that responses carry both `logprob` and
`sampling_logprob`. **Neither tested echo or an integer-array prompt** — they
were generation calls. Use `logprob` (pre-temperature, the quantity OPD wants),
never `sampling_logprob` (post-filter; would make the objective a KL against a
temperature-warped teacher).

**0.5 P3 / P4 / P6** (~10 Fireworks calls total, no GPU):
- **P3** — echoed token strings vs local HF tokenizer, 100 strings. Any diff =
  provider drift → stop.
- **P4** — same prompt scored 5×; record determinism / fp8 noise, and **log the
  mapped-alternative coverage distribution**. This is the SimCT check. Static
  bound is 65.6%; probability-weighted is unmeasured.
- **P6** — score a teacher continuation vs a deliberately corrupted one; the span
  log-ratio must clearly favour the teacher's own text. **The only test that
  catches a scrambled alignment.**

**Then stop.** Report: test results, bridge size, granularity by content type, P0
verdict, P4 coverage. GPU decision is the user's, made with these numbers.

## Phase 1 — prefix corpus (no spend)

`_load_teacher_trajectories` iterates `source.iterdir()` and **raises on any
subdir it cannot parse** — nothing is silently skipped. So build a dir holding
*only* the 5 passing job dirs:

```
/data/opd_prefixes/
  pallets__click-3466-0/
  pypa__hatch-2086-0/  pypa__hatch-2086-1/
  pypa__hatch-2086-2/  pypa__hatch-2086-3/
```
Copy from `/data/out_trivial/passk_jobs/stage1/`. `find_trajectory` walks
downward, so the rollout dir is the right level to hand it.

## Phase 2 — `distill_modal` glue (no GPU to write it)

Copy `train_lora_modal` (`train.py:334`) around `run_opd_training`. Reuse its
shape exactly: `modal.App`, the shared `VOLUME_NAME` volume for the adapter, the
`HF_CACHE_VOLUME_NAME` cache volume (so base weights download once for the whole
project, not once per run), `modal.Image.debian_slim(python_version="3.12")` with
`torch/transformers/peft/accelerate`, `.add_local_python_source("vektori_trace")`,
`@app.function(gpu=..., volumes=..., timeout=4*60*60)`.

Two additions over the SFT wrapper:
- **`FIREWORKS_API_KEY` into the container** (a `modal.Secret`, not baked into the
  image) plus outbound HTTPS.
- **Teacher-prefix cache**, per `FINAL-PLAN.md` §"Stage 3 detail". Key on
  `(task, step_index, encoding_dsv4 hash, thinking_mode)`. ~29 MB per 1000
  examples. Network cost is identical either way — the echo call must carry
  `prefix_ids + action_ids` regardless — so this is **for determinism, not
  speed**: re-rendering each step can silently drift the conditioning (DeepSeek's
  format has a `<｜latest_reminder｜>` role for exactly that), the loss stays
  finite, and the drift is invisible. **A cache miss on an already-seen example
  must be a hard error** — that miss *is* the bug the cache exists to surface.

## Phase 3 — smoke distill (**GPU — needs its own approval**)

~30 steps, to prove the loss moves and log the A/B span split. Est. ~35 min on
one L40S incl. cold start, ≈ $2.

```bash
vektori-trace distill \
  --teacher-traces /data/opd_prefixes \
  --teacher-backend fireworks \
  --teacher-model-id accounts/fireworks/models/deepseek-v4-flash-0731 \
  --cross-tokenizer --bridge /data/bridge.json --cross-top-k 5 \
  --thinking-mode chat \
  --student Qwen/Qwen3-14B \
  --teacher deepseek-ai/DeepSeek-V4-Flash-0731 \
  --gradient-checkpointing \
  --max-steps 30 \
  --out /data/vektori-out/opd-smoke
```

GPU sizing: `FINAL-PLAN.md` sizes 1×L40S (48 GB) / A100-40G for the **8B**; A10G
is explicitly "tight". **We are running 14B** (~28 GB bf16 weights + LoRA
optimizer state + activations + KV cache during `_sample_action`) — L40S with
gradient checkpointing is the floor; A100-80G is the safe call. Confirm headroom
before the full run.

Watch: `student_entropy` from step 0, A/B span split, granularity by content
type, cache hit/miss.

## Phase 4 — full run (**GPU — needs its own approval**)

Same command, `--max-steps 200`, all ~102 examples,
`--out /data/vektori-out/opd-trivial2`.

## Phase 5 — measure

pass@k with the LoRA student, `--no-escalate`, n=4, on **4 tasks**:

| task | role | DeepSeek | Qwen3-14B pre-OPD |
|---|---|---|---|
| `pypa__hatch-2086` | trained | 4/4 | 0/4 |
| `pallets__click-3466` | trained | 1/4 | 0/4 |
| `pallets__jinja-1702` | **held out** | 4/4 | never run |
| `pallets__jinja-1663` | **negative control** | 0/4 | never run |

`jinja-1663` is the control: DeepSeek could not solve it either, so an
"improvement" there means something is wrong.

### 5a. Pre-OPD baseline on the held-out pair — **required, GPU, separate approval**

The two trained tasks already have their before-picture (run6: 0/4 each). The two
held-out tasks **do not** — Qwen3-14B has never been run on `jinja-1702` or
`jinja-1663`. Without a baseline, a post-OPD number on them compares against
nothing and the held-out arm proves nothing.

So before Phase 3/4, run Qwen3-14B (base, no adapter) on those two tasks only:
n=4, `--no-escalate`, 8 rollouts. At run6's measured throughput (5.3 rollouts/h,
2 workers) that is **~1.5 h ≈ $4** on L40S.

Order of GPU spend, each needing its own approval:
1. **5a baseline** — 8 rollouts, ~1.5 h, ~$4 (base model)
2. **Phase 3 smoke distill** — ~30 steps, ~35 min, ~$2
3. **Phase 4 full distill** — 50 steps
4. **Phase 5 eval** — 16 rollouts across 4 tasks, ~3 h, ~$8 (LoRA student)

5a and 2 are independent; 5a can run first or be skipped only if you accept the
held-out arm being uninterpretable.

**Reading the result:** one sample of n yields every k ≤ n via
`pass@k = 1 − C(n−c,k)/C(n,k)` (`evaluate/passk.py`) — never run separate sweeps
per k. Before trusting any rate, check `passk.json` for `no_gradeable_rollouts`
and `infra_failures` and confirm per-rollout `parse_status` is not
`fallback_exitcode`; if it is, read §1 above — those are usually model-caused and
belong in the denominator. Use `--no-escalate`: escalation fires on `c == 0`
regardless of stage-1 size, turning a 4-rollout smoke test into 32 rollouts per
failing task.

**Caveat to state in any write-up:** 80% of training examples come from
`hatch-2086`, and two of the four eval tasks were trained on. The held-out pair
is what carries the claim.

---

# 7. Verification checklist

| | Check | Cost | Gate |
|---|---|---|---|
| P1 | Byte tables, ByteLevel assert, round-trip | — | DONE (§4 of FINAL-PLAN) |
| P2 | Offline alignment, granularity by content type | — | DONE |
| P0 | One echo call, exact shape | 1 call | fail → fallback, stop |
| P3 | Echoed strings vs local HF tokenizer, 100 strings | ~1 call | any diff → stop |
| P4 | Same prompt 5×; fp8 noise; mapped-alternative coverage | 5 calls | record in provenance |
| P5 | Full path incl. one LoRA step, tiny model + fixture | free | — |
| P6 | Teacher continuation vs corrupted; span log-ratio | ~2 calls | — |
| P7 | One real trajectory, per-step stats | modest | needs GPU |

---

# 8. Open questions — resolved

## 8.1 `--max-steps` — use 50, not the default 200

The loop (`distill.py:611`) is `for step in range(max_steps)`, inner
`for _ in range(examples_per_step)`, drawing from a shuffled `order` that
**reshuffles on exhaustion** (`:573-579`) — i.e. sampling without replacement,
one epoch per pass. Gradient is `(loss / examples_per_step).backward()` per
example, one optimizer step per outer step, so effective batch = 4.

```
epochs = max_steps × examples_per_step / n_examples
```

With 102 examples and the default `examples_per_step = 4`:

| max_steps | epochs over 102 examples |
|---:|---:|
| 200 (default) | **7.8** |
| 75 | 2.9 |
| **50** | **2.0** |
| 30 (smoke) | 1.2 |

**7.8 epochs over 102 examples, 80% of them one task, is memorisation.** Use
`--max-steps 50` for the full run and keep `--examples-per-step 4`. Revisit only
if the corpus grows.

## 8.2 Fireworks spend — negligible, GPU is the only real cost

One echo call per example per step, and **only one**: `score_ids_topk` merges the
sampled token's own logprob into the top-K rows, so calling `score_ids` as well
would be a redundant second echo (`distill.py:830-861`). The call carries
`prefix_ids + action_ids` — bounded by `max_prefix_tokens = 3584` +
`max_new_tokens = 256` ≈ **3,840 prompt tokens**, `max_tokens=1` out.

Fireworks lists DeepSeek-V4-Flash at **$0.14 / 1M input**:

| run | calls | prompt tokens | cost |
|---|---:|---:|---:|
| smoke, 30 steps × 4 | 120 | 0.46 M | **~$0.06** |
| full, 50 steps × 4 | 200 | 0.77 M | **~$0.11** |
| (200 steps × 4) | 800 | 3.07 M | ~$0.43 |

**Under a dollar.** No spend cap needed. What the calls do cost is *wall clock* —
~1 s round-trip each, serialized inside the training step: ~200 s of pure network
for the full run, on top of GPU time. Budget it as latency, not money.

## 8.3 `thinking_mode` — resolved, "chat" is safe

`FINAL-PLAN.md` lists this as unconfirmed and says it "changes what the teacher
scores." **For our configuration it does not.** `encoding_dsv4.py:25-40`:
`render_teacher_prefix` never passes `reasoning_effort`, and the reasoning-effort
prefix is the only chat/thinking difference at render time (`:363-367`, applied
at index 0 in thinking mode only). So both modes **render byte-identical
prefixes**.

What actually matters is **consistency**: the bridge stores `thinking_mode`
(`vocab_bridge.py:381, 410, 448`) and the §10.7 drift guard rejects a bridge
built under a different setting. So: pass `--thinking-mode chat` to *both*
`build-bridge` and `distill`, record it in provenance, done. Note also that
`ENCODING_DSV4_SHA256` pins the vendored encoder — the `0731` build is the
teacher; the older base-repo encoder is superseded and will be rejected.

## 8.4 GPU for Qwen3-14B — 1×L40S with gradient checkpointing

`OPDTrainConfig` is bf16 with no 4-bit path (`bf16: bool = True`; `load_in_4bit`
exists only on the SFT `TrainConfig`). Budget:

| item | est. |
|---|---:|
| bf16 weights (~14.8 B params) | ~29.6 GB |
| LoRA params + optimizer state | < 1 GB |
| activations, gradient checkpointing on, seq ≈ 3,840 | ~2-4 GB |
| KV cache during `_sample_action` (1 seq) | ~1-2 GB |
| **total** | **~34-36 GB** |

**1×L40S (48 GB) fits** with `--gradient-checkpointing` — same card run6 used, and
the cheaper choice. A100-40G is too tight to risk. Go to A100-80G/H100 only if
Phase 3 reports OOM or thrashing. `FINAL-PLAN.md`'s sizing note is written for the
**8B** student; this supersedes it for 14B.

## 8.5 Still genuinely open

1. **SFT — not this round. Decided, do not reopen unprompted.** No warm start and
   no SFT control arm. Revisit *only* after P4's coverage number is in hand, and
   only if the user asks. If it is ever built: ~30 lines reusing
   `_load_teacher_trajectories` + `tokenize_sft_example` + `train_lora_modal`,
   because `train` is self-rejection-sampling SFT and cannot take teacher traces.
2. **P4's mapped-alternative coverage is unmeasured** — static bound 65.6%,
   probability-weighted unknown. If it comes back low we are in SimCT's
   SimpleOPD trap and the exact-byte map, not K, is the problem. This is the one
   number in Phase 0 that could change the method.
