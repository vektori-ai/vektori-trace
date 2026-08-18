# From-scratch SFT — Stage A (format) → Stage B (trajectories)

> **AUTHORITATIVE.** This is the plan of record as of 2026-08-18. It supersedes
> `PLAN.md`, `V0_PLAN.md`, `FINAL-PLAN.md`, `docs/SFT-REPAIR-PLAN.md`,
> `docs/CL-PLAN.md` and every earlier SFT plan in this repo, which are kept for
> provenance only. Do not act on those. Deviating from this file requires an
> explicit decision recorded here — not a new document.
>
> Every number in "Measured facts" was verified against artifacts on
> 2026-08-18. Do not re-derive them from prose in other docs.

Fresh LoRA on `Qwen/Qwen3-14B`. No v1 / repaired / ck63 weights. Thinking ON
everywhere. Scope ends at Stage B.

## Measured facts this plan rests on

| Fact | Source |
|---|---|
| `chunked_nll` = `sum / num_items_in_batch` — per-token, grad-accum-correct | trl 1.10.0 `_chunked_cross_entropy_loss` |
| Qwen3 template wraps `<think>\n\n</think>\n\n` on the **last** assistant turn only; `enable_thinking` gates only `add_generation_prompt` | real tokenizer, transformers 5.5.3 |
| Hand-writing the wrapper into content is byte-identical to bare JSON | rendered both ways |
| Per-message prefix encoding overshoots labels by 4 tokens into the next user turn | `tokenize_messages` on `[u,a,u]` |
| Preflight missed it: TRL check tolerates <4% extra (`sft_repair_preflight.py:299`) | source |
| 165 segments = 117 seg0 + 35 + 12 + 1; 34 tasks; pallets 110/165 (67%) | `repair_manifest.jsonl` |
| 3561 actions; first-action mean 717 chars, later mean 1098 (ratio 1.53) | corpus scan |
| `ls -la` = 114 of ~153 first commands | `repair_audit.json` |
| One-action-per-row = **x14.37** the packed corpus (15.2M -> 218.6M chars) | prefix-sum scan |
| `harbor_accepts == native_json` on all 168 Phase 7 records | `/data/phase7/results.json` |
| turn-1 fails on the **acquisition** (trained) suite -> not a diversity problem | same |
| `EDIT_RE` is `^`-anchored, searched without MULTILINE -> only command 0 | `phase7.py:238`, `phase7_manifest.py:95` |
| NF4 @ 40960 peaked 39.6 / 37.8 GiB | `v1-provenance/run_summary.json` |

## 1. Prefix assert + wrapper mask (CPU)

A row is legal iff the **only** supervised message is the last one.

    prefix = apply_chat_template(messages[:-1], add_generation_prompt=True,  enable_thinking=True)
    full   = apply_chat_template(messages,      add_generation_prompt=False, enable_thinking=True)
    assert prefix == full[:len(prefix)]                    # else refuse the row
    assert full[len(prefix):len(prefix)+4] == [151667,271,151668,271]   # <think>\n\n</think>\n\n
    labels = [-100]*(len(prefix)+4) + full[len(prefix)+4:]

Masking the wrapper is mandatory: supervising it teaches empty thinking.
The tokens stay in `input_ids`, so the JSON is still conditioned on `</think>\n\n`.

Mirror into `sft_repair_train_modal.py`'s copy; keep the drift fingerprint.

Tests: `[u,a]` pass · `[u,a,u]` reject · packed multi-action reject ·
`[u,a_handoff,u,a_target]` pass. Preflight the existing repaired jsonl —
**every packed row must fail**. If any pass, the assert is wrong.

## 2. Serving contract (CPU)

`enable_thinking=True` in `dataset.tokenize_messages`, preflight
`TEMPLATE_KWARGS`, trainer `TEMPLATE_KWARGS`, Phase 7 `CHAT_TEMPLATE_KWARGS`,
and `served_to_harbor_kwargs` -> Harbor `extra_body.chat_template_kwargs`
(currently absent; template default is already True, pin it anyway).
No `<think>` strings in dataset content. Flip tests pinning False.

Phase 7 gate split: `<tool_call>` is the failure. An **empty** think wrapper is
stripped before the bare-JSON check. Think-with-content is logged, not a fail.
Selection stays `harbor_accepts + required_fields + no <tool_call>`.

## 3. Stage A dataset (CPU)

Source `/data/sft-repaired/` at pinned `dataset_sha256` (`7ecfee31...`). Slice,
never rebuild. Abort on sha mismatch.

- 165 firsts (117 turn-1 + 48 later-segment firsts) + 18 parse-error recovery
  actions = **183 rows**. Row = `messages[:target_index+1]`, supervise last only.
- Drop targets that fail `TerminusJSONPlainParser` (error, not warning) or carry
  `<tool_call>` / `tool_calls` / `role=tool`.
- Sampler `weight ∝ 1/n_task × repo_weight`. Mass: pallets 45%, anyio 25%,
  hatch 15%, prefect 15%; 10% floor; 3x upsample cap. Inside pallets, split
  click vs jinja proportional to their row counts. Downweight the 10-row
  tasks, do not drop them.
- Emit `preflight_report.json`, `mix_report.json`, `tokenization_fingerprint.json`.
  All three on disk and passing **before** the Stage A GPU session, exactly as
  Stage B's cold-token floor gates its own.

## 4. Gate manifest (CPU)

`orientation` / `first_inspection` / `post_compaction` x 5 x 3 suites = **45**,
one prefix per task. Other five categories n=1 as tripwires.
Fix `_ops` / `first_edit` to apply `EDIT_RE` **per command** before freezing.
`EDIT_RE` has one home: `vektori_trace.evaluate.phase7`. Delete the duplicate in
`sft_repair_dataset.py` (or bind it `= phase7.EDIT_RE`). Never a third copy.
Write the sha; never regenerate after.

## 5. Stage A train — GPU, needs approval

Fresh `LoraConfig` r=32 α=64 dropout 0.05 `all-linear`. `chunked_nll`,
liger off, `assistant_only_loss=False`, pre-tokenized labels,
`skip_prepare_dataset`, `truncate=False`, no packing.
`max_length=8192`, bs 1 x accum 8, LR 1e-4 cosine warmup 3%,
4 epochs ≈ 92 steps, save every 10, seed 0.

3-step **BF16** probe on the longest rows: finite loss, non-zero `grad_norm`,
LoRA tensors move, first supervised span decodes to `{"analysis"...` (JSON, not
the wrapper). Peak > 60 GiB or OOM -> NF4 + `prepare_model_for_kbit_training`
+ `enable_input_require_grads()`. Record the arm; no re-litigating.

Same session: score **base 14B (no adapter)** and the selected checkpoint on the
45 + tripwires. That is Stage B's baseline.

## 6. Selection

Earliest checkpoint clearing all 45 on `harbor_accepts` + `required_fields`
+ no `<tool_call>`. Exactly those three — nothing else gates.
Report `native_json` and `think_body_tokens` alongside, **logged, not gating**:
Stage A was not asked to teach think *content*, the IFT prior owns that. An
all-empty think body means the step-1 mask failed, which is a CPU bug to fix,
not a reason to reject an otherwise-correct checkpoint.

Ablation logged, not gating. Never min loss.
`first_edit` / `test_exec` logged only — Stage A wasn't asked to teach them.

## 7. Guarded rollout — GPU, separate approval

One training task, 10 min cap, `--no-escalate`, abort at turn 1 on gate failure.
Success = parseable JSON, keys execute, no parser loop. Pass not required.
Teardown verified.

## 8. Stage B — GPU, separate approval, only if 6 and 7 are green

Continue the Stage A adapter. One supervised action per row, row ends on target.

- <=4 later actions/segment: first **edit** -> first **test** -> **last** action
  -> one even **mid**. Read-only segments: 4 even ordinals incl. last. Never
  reuse Stage A's first action.
- ≈660 rows, x1.93 packed, ~1.9 h at v1's measured 1120 tok/s. Timeout 8 h.
- Cold draws >= **253** (25% supervised-token floor); target **325** (30%).
  `mix_report` must clear the floor before launch.
- No anti-copy in the first mix. LR 5e-5, 1 epoch, accum 8,
  `max_length` 40960, likely NF4.
- Judged vs **Stage A**: format on the 45 must not regress;
  `first_edit` / `test_exec` must beat the step-5 baseline.

## Decisions taken here, not settled in earlier discussion

Flagged so they are signed off explicitly rather than inherited by being written
down. Everything else in this file was agreed before it was written.

1. **The think wrapper is masked, not supervised** (step 1). Verified
   2026-08-18 on both row shapes: with `add_generation_prompt=True,
   enable_thinking=True` the prefix ends at `<|im_start|>assistant\n`, the
   wrapper falls entirely inside the target span, and the ids are exactly
   `[151667, 271, 151668, 271]`. Shipped as a runtime assert, not an assumption —
   if the template ever changes, every row fails on CPU.

   Rationale: with the wrapper *in* the loss, all 183 Stage A targets teach
   "open think, close it immediately, then JSON". That leaves
   `enable_thinking=True` set while training the behaviour out. Masking it keeps
   the JSON supervised and leaves think length to the IFT prior.

   **Needs an explicit yes.** If the answer is no, the wrapper goes back into the
   loss and step 5's probe expects `<think>\n\n</think>\n\n{...}` as the first
   supervised span instead of `{"analysis"...`.


## Fail closed — before any GPU

Source sha != pin · packed repaired jsonl still tokenizes · any non-last
supervised message · prefix equality fails on a last-only row · think wrapper not
found at the head of a target span · target unparseable or `<tool_call>` · firsts
< 165 or recoveries < 18 after drops · pallets > 45% · any of anyio/hatch/prefect
< 10% · any repo > 3x upsample · Stage B cold token share < 25% · manifest sha
changed after freeze · a Phase 7 request without `enable_thinking=True`.

## Budget

Stage A probe ~12 min · Stage A train ~35-40 min · eval (base + ~9 ck x 50) ~45
min · rollout ~20 min · Stage B train ~2.0 h · Stage B eval ~30 min.
**≈4.5 GPU-hours**, 4 approved sessions, ~$15-25. Training itself is ~2.7 h.
Throughput from v1's measured 166.7 s/step @ 8 x 24.6k tok; ±30% on Stage A.
