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


## Step 7 — guarded rollout (2026-08-19)

Endpoint `ap-UKoxjslnft8WKJ8X2Ty1EL`, L40S, base + ck84 on one load,
00:47:42 -> ~01:08 UTC (~21 min, ~$0.70). Torn down and verified `stopped`,
0 tasks.

Task **`pallets__click-3653`** — a training task: 6 of the 165 Stage A rows
come from it, and it is the acquisition suite's orientation task.
`--stage1-n 1 --no-escalate --timeout-sec 600`.

**Result: 6/6 turns accepted, 16 commands executed, no parser loop.** The model
oriented at `/workspace` without cloning, walked `src/click/termui.py` to
`tests/test_termui.py`, edited with `sed -i` on turn 5 and ran pytest on 5 and
6. It hit the 600 s cap mid-`pytest tests/ -q`; step 7 does not require a pass.
Per-turn latency ~100 s. Report: `/data/vektori-out/stage-a-rollout/`.

The chat template was pinned on the wire — the job's own `config.json` carries
`llm_call_kwargs.extra_body.chat_template_kwargs.enable_thinking: true`.

### Two bugs this found

1. **A trajectory is a parse, not a completion.** Harbor stores its
   decomposition: `message` is the analysis/plan prose, `tool_calls` are
   synthetic `bash_command` entries holding the extracted keystrokes. The first
   grader read `message` as raw output and scored all six correct turns as
   total format failures, with the synthetic `tool_calls` firing the
   legacy-envelope gate on top. Commands existing **is** the evidence the
   parser accepted the turn. Corrected in `scripts/rollout_gate.py`.
2. **`TimeoutExpired` carries bytes under `text=True`.** `subprocess.run`
   decodes on the success path only, so persisting harbor's output raised
   `TypeError` *after* the rollout ran and destroyed `passk.json`. A guarded
   rollout ends in a timeout by design. Fixed in `validity.py` with a test.

### Still open after step 7

- `native_json` / `required_fields` on **live** turns. A trajectory holds no raw
  text, and nothing else in the job dir does either (`harbor_stdout.txt` is 0
  bytes; the pane holds terminal output only). Only `capture-proxy` records raw
  completions, and this rollout did not run behind it.

  **Do not repeat the claim that "commands existing implies bare JSON".** It is
  false: harbor *salvages*. `TerminusJSONPlainParser.parse_response` tries the
  normal path, and on error runs auto-fixes
  (`terminus_json_plain_parser.py:39-59`). Every turn here carried a warning:

  | turn | warning | what it means |
  |---|---|---|
  | 1 | `AUTO-CORRECTED: Extracted JSON from mixed content` | the normal parse **failed**; a regex scrape recovered an object |
  | 2-6 | `Extra text detected before JSON object` | normal parse succeeded, text preceded the `{` (line 208) |

  So turn 1 is a harder failure than turns 2-6, not the same one. The normal
  extractor brace-counts from the **first `{` in the whole response**
  (lines 186-210), so a brace anywhere in the preceding text derails it. And
  the fallback `_fix_mixed_content` returns the **first regex match that
  parses** (lines 336-347) — if the pre-JSON text contains a complete
  JSON-ish object, harbor can execute a draft instead of the final action. On
  turn 1 the recovered commands were sane, but that is luck, not a guarantee,
  and a frozen-prefix Phase 7 sweep cannot see this at all.

- **What the pre-JSON text is, is unproven.** Token accounting (real Qwen3-14B
  tokenizer, against the JSON harbor kept) puts 58-94% of every completion
  outside the JSON: turn 4 billed 4078 tokens for 232 tokens of action. The
  `enable_thinking: true` pin is confirmed on the wire in the job's
  `config.json`, and `reasoning_content` is 0 on all six turns — consistent
  with no reasoning parser on the endpoint, so a think block would land in
  `message.content`. Consistent with, **not evidence of**: prose fits equally
  well. `scripts/serve_student.py --reasoning-parser qwen3` (vLLM 0.21.0)
  settles it and removes the brace hazard at the same time.
- The 600 s cap vs ~100 s/turn: Stage B needs a bigger number or it grades
  trajectories that were cut off, not finished.

## Still open

- ck20 / ck40 at 4096, so "earliest" means something. **ck84 at 4096 cleared
  45/45** — that question is closed; this one is not.
- ~~base 14B on the 45~~ — **done**: 27/41 graded (4 lost to an early
  kill-switch), `post_compaction` 1/15 vs ck84's 15/15, orientation and
  first_inspection already at base. Stage A's measured contribution is
  post-compaction, not turn-1 JSON.
- **Step 7 ran and is green** (see above). Stage B remains blocked on step 6:
  ck84 clears 45/45 at 4096, but "earliest that clears" is unmeasured — ck20 /
  ck40 at 4096 were never run.
