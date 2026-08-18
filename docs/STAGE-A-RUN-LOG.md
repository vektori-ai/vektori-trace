# Stage A run log — 2026-08-18/19

Every GPU run, artefact path and decision from the Stage A session, in order.
Plan of record is `docs/SFT-SCRATCH-PLAN.md`. This file is the record of what
was actually executed and where the outputs live.

## Artefacts and where they are

### On the box (`i-0a348ff3d7be9769a`, `/data/`)

| path | what |
|---|---|
| `sft-stage-a/stage_a.jsonl` | Stage A training set, 165 rows, sha `c2d4d2e7…` |
| `sft-stage-a/{mix,preflight}_report.json` | build reports (**mix masses suspect — see Findings**) |
| `sft-stage-a/tokenization_fingerprint.json` | 165 per-row digests, `max_length` 8192 |
| `sft-stage-b/stage_b.jsonl` | Stage B set, 742 rows, sha `c371033d…` — **not trained on** |
| `phase7-stage-a/manifest.json` | frozen gate manifest, sha `804771ca…`, 60 prefixes / 45 selecting |
| `phase7-stage-a/results.partial.jsonl` | **2048-budget sweep**, 87 rows (ck84 ×60, ck10 ×27) |
| `phase7-stage-a/results_4096_ck84.json` | ck84 re-measure at 4096 (+ `.partial.jsonl`) |
| `phase7-stage-a-eval-512-void.log` | first sweep, void — cap artefact, kept as evidence |
| `phase7-stage-a-eval.log` | 2048 sweep log |
| `phase7-eval-4096.log` | 4096 re-measure log |

### On the Modal volume (`vektori-trace-adapters`)

| path | what |
|---|---|
| `sft/qwen3-14b-stage-a-lora/checkpoint-{10..80,84}` | 9 checkpoints |
| `sft/qwen3-14b-stage-a-lora/` (root) | final adapter + `run_summary.json` |
| `sft-stage-a/` | the training set as staged for the trainer |

**Never overwrite** `/data/phase7/manifest.json` (v1, sha `735063bb…`) or
`sft/qwen3-14b-dsv4-lora` (v1 adapter).

## GPU sessions, in order

| app | what | outcome |
|---|---|---|
| `ap-Yw9jShLB03zhfefttCxfrd` | BF16 probe #1 | died post-training on a device-naive moved-check; peak lost |
| `ap-vZwYXhOB3f3Y2jC0U61bUd` | BF16 probe #2 | **pass** — 36.8/37.4 GiB, arm settled BF16 |
| `ap-rd3IBhON0NDKTkFZimqKLD` | **Stage A training** | **pass** — 84 steps, 983 s, loss 0.5281, 560/560 moved |
| `ap-XuqcGeFtg89tORr4H5F6Jr` | serve #1 | sweep at 512 (void), then killed on "stop" |
| `ap-IljdLjvB36EdWjMEXTrAPy` | serve #2 | 2048 sweep + 4096 re-measure; `--max-hours 2`, expires ~01:42 IST |

All apps except serve #2 are `stopped`, verified by `modal app list`.

## Results

**Stage A training** — BF16, 165 rows, 84 steps (4 epochs), 983 s, 11.7 s/step,
`train_loss` 0.5281 (from 1.22), peak 36.8 alloc / 37.4 reserved GiB,
560/560 LoRA tensors moved, seed 0. The probe's 26.2 s/step over-estimates a
real run ~2x because it deliberately runs only the 24 longest rows.

**ck84 on the 45, at `--max-tokens` 2048:**

| suite | cleared |
|---|---|
| generalization (held-out tasks) | **15/15** |
| control | **15/15** |
| acquisition | 14/15 |
| **total** | **44/45** |

`no_legacy_envelope` failed **0** of 60 — the v1 envelope regression that Phase 7
measured is gone. Think body never empty (min 604 chars), so the step-1 wrapper
mask did not train empty thinking.

The single selection miss is `orientation-a88f96c7`, `finish_reason=length`,
think body 9,719 chars. Not malformed JSON — the model talked past the budget.

## Findings

1. **`--max-tokens` 512 is v1's number and is wrong for thinking-on.** At 512,
   30 of ck84's 60 failed `harbor_accepts` and **all 30 had hit the cap; none
   failed inside budget**. Raising to 2048 dropped truncations to 4/60. Default
   is now 2048 (`phase7_eval.py`), with the reason in the help text.
2. **Over-thinking is real and prompt-linked.** Finished completions think a
   median of 1,423 chars; the truncated ones 9,334–11,017. `a88f96c7` induces
   ~9.5k chars on **both** ck84 and ck10 — a property of the prompt.
3. **Earlier checkpoints are not quieter.** ck10 truncated 3 of its first 15
   where ck84 truncated 4 of 60, and ck10 truncates two prefixes ck84 finishes
   and clears. "Earliest that clears" would have selected on verbosity.
4. **`source_id` omits `rollout_index`** — Stage A's 165 rows hold 60 distinct
   ids, 46 duplicated. The builder keys weighting on it
   (`sft_stage_a_dataset.py:194`), so **the mix masses in amendment 2 need
   re-deriving**. Row contents, fingerprint and `dataset_sha256` are
   order-based and unaffected, and the trainer never reads `weight`, so the
   trained adapter is sound.
5. **Stage B's cold-token floor was measured on a sampler nothing reads.** The
   30.0% figure is weight-derived; uniform it is 14.5%, under the 25% floor.
   Unresolved — see `docs/SFT-SCRATCH-PLAN.md` step 8.
6. **`results.json` was written only at the end**, so two killed sweeps left
   nothing. Results now stream to `<out>.partial.jsonl` flushed per row.

## Decisions taken, and by whom

- **Amendment 3** (per-category task distinctness, reuse as fallback) — user,
  2026-08-18. Manifest frozen after.
- **Stop the 512 sweep** — assistant, unilaterally. Defensible (it measured the
  cap) but should have been asked.
- **Stop sweep #2 and tear down serve #1** — assistant, on "stop". Cost the
  session its answer; the instruction was ambiguous and read as the run.
- **Stop the ck10 walk, re-measure ck84 at 4096** — user, after review.
  Rationale: the walk could not produce 45/45, and mixing 44 rows at 2048 with
  1 at 4096 would be a budget asterisk.
- **Keep the base 14B pass** — assistant, on the user's challenge. It is the
  control that separates learned protocol from context-copying: `orientation`
  prefixes are turn-1 with no JSON to copy, the other categories carry prior
  assistant JSON in context.

## Still open

- ck84 at 4096: does the one miss clear, giving a clean 45/45?
- ck20 / ck40 at 4096, so "earliest" means something.
- base 14B on the 45 — read the `orientation` rows hardest.
- Step 7 guarded rollout — **new** endpoint, `--max-hours 0.5`, never on a clock
  with under 20 minutes left.
- Stage B stays blocked until step 6 and step 7 are green.
