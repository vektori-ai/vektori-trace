# Tau2 retail SFT — handoff

Written 2026-08-25, end of the session that built the Tau2 data pipeline and
trained `A_warm`. Read `docs/CLEA-QWEN3-4B-EXPERIMENT-V2.md` first — it is the
plan of record. This file is what that document does not tell you: what was
actually built, what was measured, and which mistakes cost real time.

---

## 1. Where the work stands

**Done and verified.**

| | |
|---|---|
| Corpus | 657 rows / 71 tasks, sha `4f905c9d82e28fc4` |
| Split | 30/30/16/38 = 114, manifest `b741bfceb1f3d027` |
| Prompt parity | 7/7 rows render identically, transformers 5.5.3 vs vLLM 4.57.6 |
| Parser parity | 25/25 targets round-trip through `Hermes2ProToolParser` |
| `A_warm` | trained — 105 steps, loss 1.21 → 0.24, 3 committed checkpoints |
| Tests | 68 passing (`test_tau2_normalize`, `_split`, `_reload_gate`, `_eval`) |
| Eval harness | built 2026-08-25 (`tau2_eval_modal.py`), CPU-verified, **never run on a GPU** |

**Not measured.** Every arm. The harness exists and its refusals are tested, but
no live Tau2 episode has been run: there is still no basis for selecting among
the three `A_warm` checkpoints, and no `A0` / `B0` number. The blocker moved
from "no way to evaluate" to "evaluation not yet approved and run."

---

## 2. The trained artifact

```
volume vektori-trace-adapters : tau2/runs/a_warm_20260825_003343
  adapter_model.safetensors   132 MB   r=16 alpha=32 dropout=0.05, 7 target modules
  adapter_config.json                  base Qwen/Qwen3-4B
  tokenizer.json, tokenizer_config.json, chat_template.jinja
  checkpoint-35 / -70 / -105           11 files each, all committed=True
  train_steps.jsonl                    105 steps
  run_config.json                      hashes, versions, hyperparameters
  failure.json                         FALSE FAILURE — see 4.1
```

Measured: peak 24.3 GiB of 44 (L40S), ~25 s/step, 45 min, ~$1.40.
`grad_norm` 0.0554–0.5319, all finite.

**The loss went flat after ~step 10** (1.35 → ~0.30, then oscillating 0.19–0.45
for ninety steps). Epochs 2 and 3 did not visibly improve the objective. That is
consistent with V2 §6.4's expectation and means epoch 1 may be the right
checkpoint — but loss cannot decide it. All three stay candidates until live
evaluation exists.

---

## 3. What was built

```
vektori_trace/tau2/
  normalize.py     raw sim -> decisions; greeting gate; reasoning excluded
  eligibility.py   structural + policy gates; confirmation is NON-gating
  export.py        render, tool round-trip, census
  split.py         30/30/16/38 with asserted invariants
  taskmeta.py      task family from Tau2's own reference actions
  difficulty.py    real frontier bands from data/tau2/results/final
  runlog.py        durable per-step logging, fsync'd

scripts/
  tau2_build_corpus.py          corpus + eligibility + census
  tau2_build_split.py           freeze the manifest
  tau2_context_census.py        per-domain context measurement
  tau2_sft_preflight.py         CPU proof of every row before a GPU
  tau2_sft_train.py             the trainer (TRL SFTTrainer, pre-tokenized)
  tau2_sft_full_modal.py        full W30 launcher
  tau2_sft_probe_modal.py       3-step probe launcher
  tau2_stage_corpus_modal.py    stage corpus to volume, hash-verified
  tau2_serving_parity_modal.py  prompt + parser parity (CPU)
  tau2_adapter_diagnose_modal.py  fresh-process adapter inspection
  tau2_watchdog.sh              5-min poll, read-only
  tau2_eval_modal.py            live S16/C30 eval; multi-LoRA, one base load
```

---

## 4. Things that cost time. Read this section.

### 4.1 The gate that failed a successful run

After 45 minutes of correct training, `verify_reload` refused the artifact:
`missing ['special_tokens_map.json']`. **Qwen3 never writes that file** — its
special tokens live inside `tokenizer_config.json`. The requirement was
impossible to satisfy.

Fixed via `EITHER_FILE_OR_KEY`: accept a separate file *or* the corresponding
key. The same pattern was already correct for `chat_template.jinja` one line
above, and was not applied to the line below it.

Two lessons: a completeness check that can fail belongs **before** training, not
after it; and `failure.json` is still on the volume recording a failure that did
not happen.

### 4.2 A confirmation gate that could not fail

The first policy audit treated *any* prior assistant speech as confirmation.
Every prompt contains the scripted greeting, so it passed unconditionally — and
"73/73 policy-compliant" meant nothing.

Rewritten as a proposal → assent → matching-mutation state machine, it then
rejected 64/73 traces. Inspection showed those rejections were wrong: one agent
sentence saying "modification" matched four different mutating tools, so every
later assent looked ambiguous.

**Resolution: confirmation is a non-gating diagnostic.** `diagnostic_confirmation`
is reported, `uncertain` goes to a manual-review queue, and eligibility is
decided by structure, reward, authentication, preconditions and rendering.
Do not promote it back to a gate without a much better matcher.

### 4.3 Context: 8192 was wrong, and only measurement showed it

The first census ran **without tool schemas** and suggested 8192. Rendering the
real 15-tool retail schema adds ~3,300 tokens to every prefix. At 8192 only
20/72 traces survive. Telecom is worse — **0/114 at 8K**, because its policy is
5,204 tokens.

Measured retention by complete trace:

| cap | retail | airline | telecom |
|---|---|---|---|
| 8,192 | 20/73 | 17/22 | 0/114 |
| 12,288 | 67/73 | 22/22 | 114/114 |
| **16,384** | **72/73** | **22/22** | **114/114** |
| 32,768 | 72/73 | 22/22 | 114/114 |

**16,384 is the pin. 32K buys nothing** and doubles KV cache per request.

Serving needs its own rule: final-turn prompts reach p90 ~11,942 (retail) and
generation p99 ≤ 344 across all domains, so **16,384 cap with a 2,048 reserve**
(14,336 prompt budget). An over-budget episode must stop and record
`context_exceeded` as infrastructure, never graded as model failure, never
truncated mid-episode.

### 4.4 Train and serve cannot share an environment

vLLM 0.11.2 pins `transformers>=4.56,<5` ([vllm#30466]). This repo pins 5.5.3
everywhere it trains. **They cannot coexist in one image**, which is why parity
runs on Modal and not on the box.

This mattered enough to check: the chat template lives in transformers, and it
renders every training row. Prompt parity came back **7/7 identical**, so no
drift — but that result belongs to *that* template. It is why
`chat_template.jinja` and the tokenizer ship with the adapter.

### 4.5 Other traps

- **`p.grad` after training reads zeros.** `Trainer` zeroes gradients after each
  optimizer step. Gradient health comes from the logged `grad_norm` series.
- **Raw-logit tolerances are meaningless in bf16.** A 0.05 threshold on logits
  reaching 25 fails a model against itself. Measured ground truth: two forward
  passes of the same model are **bit-identical** (delta 0.0), and a 3-step
  adapter's true effect is **0.31 logits**.
- **TRL pins are not guessable.** `chunked_nll` does not exist in trl 0.29;
  trl 1.10 needs `datasets>=4.7`. Copy `sft_stage_a_train_modal.py`'s set:
  torch 2.13.0, transformers 5.5.3, trl 1.10.0, peft 0.19.1, accelerate 1.14.0,
  datasets 5.0.0, bitsandbytes 0.50.1.
- **`modal app logs` cannot resolve an ephemeral app by name.** `modal run`
  creates ephemeral apps; use the `ap-...` id from `modal app list`.
- **`--partition F38` was accepted** by the trainer until a test caught it. Only
  W30 and C30 are trainable, refused by name at the parser and in `load_rows`.
- **SSM transfers cap around 24 KB per response.** Use 18 KB base64 chunks, and
  prefer a Modal Volume for anything larger.

---

## 5. Invariants that must not be relaxed

```
corpus sha        4f905c9d82e28fc4
manifest          b741bfceb1f3d027
tools sha         1881b37265759ea3   (15 retail tools)
max_length        16384, truncation refused
LoRA              r=16 alpha=32 dropout=0.05 all-linear
lr                1e-4, PRECOMMITTED — not derived from any probe
seed              20260824
trainable         W30, C30 only. S16 selects; F38 is the frozen final test.
```

TRL config, asserted before the trainer is constructed:

```python
loss_type="chunked_nll"           # not "nll"
use_liger_kernel=False
assistant_only_loss=False         # our labels are authoritative
dataset_kwargs={"skip_prepare_dataset": True}
packing=False
save_only_model=False             # or checkpoints are unresumable
```

Labels are built offline and verified per row. `LabelPreservingCollator` is
required — the stock LM collator regenerates labels on pad and erases every
`-100`.

---

## 6. Data design, and why

- **Row = one assistant message**, not one tool call. 316 message / 257
  toolcall+text / 76 toolcall; 117 messages carry more than one call.
- **The scripted greeting is never a target.** Identical across all 78 traces,
  always index 0, always missing `raw_data`. It stays in every prompt because
  the simulator supplies it at serving time too. Training on it would teach the
  model to emit something it never needs to emit, on 1 row in 6.
- **`reasoning_content` never reaches the model.** DeepSeek's scratchpad is
  preserved for audit and excluded from prompt and target: putting it in the
  prompt trains `p(action | teacher's hidden reasoning)` while serving has no
  such thing.
- **All genuine decisions, no six-per-task cap.** 731 decisions across 78
  traces; capping at six would have discarded ~40% of paid-for supervision.
- **Task 51 excluded** — one 72,821-token assistant message, 6.5× the p99.
  **Task 54 excluded** — mutation after an unresolved tool error, with
  call/result ids preserved. See `docs/TAU2-CORPUS-EXCLUSIONS.md`.

---

## 7. Next steps, in order

**1. Live Tau2 eval harness — BUILT 2026-08-25, not yet run.**
   `scripts/tau2_eval_modal.py`, with `tests/test_tau2_eval.py` (18 tests).

   > **Correction.** This step previously read "`serve.py` has **no
   > `--tool-call-parser` flag** ... fix this first." That was a false lead.
   > `serve_model` never needed one: it forwards `extra_vllm_args` verbatim
   > (`serve.py:401`), and `scripts/run_tau2_smoke.py:70-74` has been passing
   > `--enable-auto-tool-choice`, `--reasoning-parser qwen3` and a per-model
   > `--tool-call-parser` through that door all along. Nothing was structurally
   > blocked. What was missing was adapter serving wired to a *partition* of the
   > frozen split — which is what the new script adds.

   One vLLM boot hosts the frozen base and all three checkpoints as LoRAs, then
   invokes tau2 once per arm. `ServedModel.adapter_models` exists for this;
   Phase 7 graded seven checkpoints on one base load. Four separate boots would
   pay the ~minutes-long base load four times for no extra signal.

   Two things it pins that the defaults get wrong, each of which would have
   produced a wrong *number* rather than an error:

   - **Prompt budget.** `serve_model`'s default clamps a 16,384 window to
     `out_cap = min(default, L//2)` = 8,192 in / 8,192 out. Retail final-turn
     prompts reach p90 ~11,942 (§4.3), so most S16 episodes would be refused for
     context — and that refusal reaches tau2 as a *model failure*. The script
     passes `model_info` explicitly: 14,336 prompt / 2,048 generation.
   - **`--max-loras`.** vLLM defaults it to 1 and refuses the second adapter
     after the GPU is allocated.

   `--dry-run` is CPU-only and runs anywhere (no tau2 install needed); a GPU
   needs `--yes`, checked before any other env check so an unapproved run is
   refused for that reason and not incidentally. `F38` is refused by name — the
   frozen final test is never reachable from a selection script.

   Still to do here: measure `B0` (frozen 8B) as the §10 gate-4 reference. It is
   a separate boot and a separate approval, and is not needed for checkpoint
   selection.

**2. Select `A_warm`** from checkpoints 35 / 70 / 105 on S16 live performance.
   V2 §6.4: never on training loss. Expect epoch 1 to be competitive.

**3. §7.1a sampling-entropy gate.** Before branching, sample k actions at C30
   prefixes and compare diversity against `A0`. If `A_warm` has collapsed to
   near-deterministic sampling, replay OPD is dead on arrival and the right move
   is an earlier checkpoint, not more replay updates. Cheap, and it prevents
   mis-attributing a data problem to the objective.

**4. Freeze the C30 prefix manifest** (§7.2). 289 rows, 30 tasks, already
   preflight-passed. Build once, read twice — both branches must consume the
   identical stream.

**5. Run the two continuation arms**, 32 updates each, from the identical
   `A_warm` with **fresh optimizer and scheduler state** and distinct recorded
   seeds. Neither may inherit momentum from warm SFT or from its sibling.

Before the next GPU run, consider a 3-step probe with `gradient_checkpointing=False`
and/or `batch_size=2`. Live VRAM was 15.4 GiB of 44, so there is real headroom —
but this is **unverified**, and turning off checkpointing could add 15–30 GB of
activations on a 13k-token row. Measure, do not assume.

---

## 8. Operational notes

```bash
# corpus / split (CPU, on the box)
PYTHONPATH=/data/tau2/src .venv/bin/python scripts/tau2_build_corpus.py --max-length 16384
PYTHONPATH=/data/tau2/src .venv/bin/python scripts/tau2_build_split.py --artifacts /data/tau2/artifacts_16384
.venv/bin/python scripts/tau2_sft_preflight.py --artifacts /data/tau2/artifacts_16384 --partition W30

# live eval — CPU dry-run first; --yes is the GPU, see CLAUDE.md
.venv/bin/python scripts/tau2_eval_modal.py --dry-run          # runs anywhere
.venv/bin/python scripts/tau2_eval_modal.py --partition S16 --yes
# results land in /data/tau2/data/simulations/tau2_eval_s16_<ts>_{a0,ck35,ck70,ck105}.json
# Grading them against the V2 §10 gates is a separate step, on purpose.

# training (Modal L40S) — needs explicit per-run approval, see CLAUDE.md
.venv/bin/modal run scripts/tau2_sft_full_modal.py
.venv/bin/modal app list                       # get the ap-... id
.venv/bin/modal app logs <ap-id> -f
scripts/tau2_watchdog.sh <ap-id> 300 25        # read-only, 5-min poll
```

Tear down the moment a run finishes — no approval needed for teardown, and it is
expected every time.

The EC2 box (`i-0a348ff3d7be9769a`, ap-south-1, SSM only) holds the corpus and
the Tau2 checkout. It has **no GPU**; it can launch Modal, which is preferable
to launching from a laptop so a disconnect cannot orphan a run.

[vllm#30466]: https://github.com/vllm-project/vllm/issues/30466
