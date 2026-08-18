# Corrective SFT — repairing v1's protocol

**Date:** 2026-08-17 · supersedes nothing · companion to
`docs/CL-PLAN.md`, memory `v1-sft-protocol-mismatch`

V2 is **not** part of this path. It appeared only as a proposed quantization
diagnostic and has been dropped. No v2 adapter, no v2 dataset. It remains
historical evidence only.

The Base/V2 × NF4/BF16 experiment is dropped. The three-step NF4-vs-BF16 probe
in Phase 5 is **not** that experiment: it uses the same v1 adapter, the same
corrected batch and the same seed, and exists only to price the continuation.

## Resolved decisions (2026-08-17)

| | |
|---|---|
| target text | **hybrid** — raw capture where it joins and verifies, canonical ATIF rebuild otherwise; per-turn `target_source` provenance |
| join key | **full semantic tuple**, not a 300-char prefix |
| handoff turns | kept as **context**, masked from loss |
| supervision | **pre-tokenized**, explicit per-message mask; not TRL `assistant_only_loss` |
| base precision | **BF16** (revised 2026-08-18 — see Phase 5); NF4 only as a measured fallback |
| dev split | **none** — train all 165 rows |
| GPU | **A100-80GB** (L40S 48 GB is serving-only; BF16 at 40k will not fit, NF4 has no safe headroom) |
| schedule | 21 optimizer steps/epoch × 3 epochs = **63 max steps** |
| checkpoints | 10, 20, 30, 40, 50, 60, 63 |
| selection | earliest checkpoint passing the native-JSON and behavior gates |

---

## 0.0 Status — 2026-08-18

Phases 1–6 are **done**. The repaired adapter exists. Phase 7 is **built and
validated offline**; the only thing left in it is one approved endpoint session.
Nothing yet demonstrates the repair works at generation time — every metric so
far is teacher-forced.

| | |
|---|---|
| Phase 1 | v1 frozen at `/data/v1-provenance/` with sha256s |
| Phase 2 | built, exit 0 — `/data/sft-repaired/`, sha `7ecfee31…` |
| Phase 3 | **PREFLIGHT PASSED** — targets, roles, masks, TRL agreement all clean |
| Phase 4 | trainer written, tests green |
| Phase 5 | **passed on NF4** — 3 steps, longest rows, 560/560 LoRA tensors moved. The BF16 arm the revised plan called for was never run; see below |
| Phase 6 | **complete** — 63 steps, 3 epochs, 3.05 h, 7 checkpoints |
| Phase 7 | **built** — corpora, frozen manifest (`735063bb…`, 24 prefixes), gates, driver, 41 tests. Needs one A100-80GB session |

### Phase 6 result

```text
63 steps · 3 epochs · 165 segments · nf4 · lr 1e-5 constant_with_warmup(5)
elapsed 10,973 s (3.05 h) · 174.2 s/step
peak VRAM 63.7 GiB allocated / 65.6 reserved of 79.25
loss  0.6436 (step 1) -> 0.4856 (step 63), mean 0.5731
grad_norms 0.621 -> ~0.016-0.020, finite and non-zero throughout
lora_tensors_changed 560/560
supervised 1,029,371 of 3,992,567 tokens
```

Checkpoints at 10/20/30/40/50/60/63 in
`sft/qwen3-14b-dsv4-lora-repaired`. v1's `adapter_model.safetensors` still
hashes to `893ca045…`, unchanged; the repaired adapter is `dda365c3…`.

**Four defects were found in Phase 4, every one a flag or default that silently
did nothing rather than failing:**

1. `enable_input_require_grads()` missing — frozen base plus checkpointing gives
   an empty backward that still logs a plausible loss.
2. `device_map` missing — `--nf4` loaded full-precision weights and reported
   success. Two "BF16 vs NF4" probes were really BF16 twice.
3. **`loss_type` lost in the port** — the actual OOM cause. TRL defaults to
   `chunked_nll`; a plain `Trainer` has no such concept and routes to
   `ForCausalLMLoss`, which materialises `[36993, 151936]` logits and upcasts
   them to fp32. That default is why v1 fit in 39.6 GiB.
4. `skip_prepare_dataset` and `--memory-history` half-wired.

Leaving TRL was never necessary: `assistant_only_loss` cannot express the mask,
but `SFTTrainer` detects a pre-tokenized dataset and skips its whole masking
pipeline (`sft_trainer.py:1397`), keeping our labels and collator. Conflating
those two cost three probes.

### The precision the run actually used — **BF16 was not run**

Phase 5 was revised on 2026-08-18 to **BF16 only**, with an explicit stop
condition: *"If BF16 does not fit safely, stop and discuss before falling back
to NF4. Do not switch silently."* The probe that ran was **NF4**
(`docs/sft-probe-monitor-gpt-5.6-sol.md`: "NF4 base: 9.1 GiB footprint, 280
bitsandbytes four-bit modules"), and so was the corrective run. The status table
above said "passed" without recording the arm change.

The fallback is right on the numbers; it is only the record that was wrong. The
probe measured **63.7 GiB peak allocated with `chunked_nll` already restored**,
against 79.25 usable. A BF16 base is 14.8e9 × 2 = 27.6 GiB where NF4 loaded at
9.25 GiB, so the same run in BF16 lands at ≈ **82 GiB** — over the card, before
allocator fragmentation.

An earlier note here read *"It failed at 71.7 GiB with the full logits path;
removing ~31 GiB of logits would likely bring it inside 79. It is a rerun
option, not a blocker."* That is withdrawn: 71.7 GiB is where an allocation
**failed**, not a peak requirement, so it cannot be used as a base to subtract
from. Anchored on the probe's own measured NF4 peak, BF16 at 36,993 tokens does
not fit an A100-80GB. It needs an H200/B200 or a shorter `max_length`.

**Consequence, and it is not cosmetic.** Repaired-v1 was fitted against NF4 and
will be served against BF16 — the *same* train/serve precision mismatch as v1,
which Phase 1 named as a defect to fix. The BF16 rationale recorded there
("step 0 of a BF16 continuation is exactly the model measured as broken", "it
absorbs v1's quantization drift") is unrealized. Phase 7 must therefore
evaluate **through the real serving stack** — vLLM, BF16 base, LoRA on top —
because loading the adapter back onto an NF4 base would certify a configuration
that will never be deployed.

### Phase 2 audit

```text
117 rollouts -> 165 segments over 34 tasks
action 3561 · handoff_question 48 · parse_error 18 · unknown 0
target source: raw_capture 3528 · atif_rebuilt 33
rebuild justification: 3528/3528 field-equivalent, 0 mismatched
join index 11226, 0 benign duplicates, 0 conflicts
commands 7436  ops {read 4343, edit 146, test 635}
```

Two independent corroborations that nothing was lost: **7,436 commands** equals
v1's known `bash_command` count, and 3561 + 48 + 18 = **3,627** equals v1's
assistant-turn count.

The 33 rebuilds are teacher responses wrapped in prose, which harbor salvaged at
runtime and which must not become targets. The join covered everything else, so
the rebuild path is proven on 3,528 turns and used on 33.

### Phase 3 preflight

```text
165 rows
tokens: min 6681  median 24635  max 36993   (cap 40960 — nothing truncates)
supervised: 1,029,371 / 3,992,567 (25.8%)
TRL cross-check: 108/108 comparable rows agree
targets ok · roles ok · masks ok · trl_agreement ok
```

Every supervised span on every row decodes to parseable Terminus JSON
containing its own target. The 57 rows excluded from the TRL comparison are the
ones carrying a handoff or parse-error turn, where the masks must differ.

v1's saved `chat_template.jinja` is **byte-identical** to the Hub's
(`a55ee1b1…`), so "renders with the exact serving template" holds literally
rather than by assumption.

## 0. The sequence

```text
preserve v1
→ reconstruct corrected raw-JSON dataset
→ audit every row and mask
→ build BF16 v1-continuation trainer
→ approved three-step BF16 probe
→ approved corrective SFT
→ select earliest passing checkpoint
→ approved native serving validation
→ one guarded no-shim rollout
→ OPD if interface and basic behavior are stable
→ support measurement
→ RL only after passes exist
```

## 0.1 The problem, restated

v1 learned the right *objective* in the wrong *protocol*. It was trained on
OpenAI-style `tool_calls` (Qwen3's template renders these as
`<tool_call>{...}</tool_call>`) with `role="tool"` observations. Harbor's
terminus-2 wants literal flat JSON —
`{analysis, plan, commands[], task_complete}` — in assistant *text*, and sends
observations as `role="user"` (`terminus_2.py:1142`). Hence 305 consecutive
`Missing required fields` in the original run.

v1 is not empty: 7,436 `bash_command` + 241 `mark_task_complete`; 148/165
segments contain an edit; first command is `ls -la` 114x / `git status` 23x.
The deficit is the envelope, not the intent. Repair by **continuing** the
adapter, not retraining it.

---

## Phase 1: Preserve provenance

Before changing anything:

- v1 stays immutable.
- Record its adapter checksum and `adapter_config.json`.
- Record the exact Qwen base revision and tokenizer.
- Preserve the original v1 dataset, logs and `run_summary.json`.
- Repaired-v1 goes to a **new** output path.
- Do not overwrite v1 or v2.

### v1 as-built (recorded)

| | |
|---|---|
| adapter | Modal volume `vektori-trace-adapters`, `sft/qwen3-14b-dsv4-lora` |
| base | `Qwen/Qwen3-14B` |
| dataset | `/data/sft/sft_train.jsonl` — 165 segments, 34 tasks |
| LoRA | r 32, alpha 64, dropout 0.05, all-linear (7 proj types), CAUSAL_LM |
| trained | 165 segments, 5 epochs, lr 1e-4, cosine, bs 1 × accum 8 → 106.9 steps |
| loss | 0.657 |
| peak VRAM | 39.6 GiB on A100-80GB |
| **base precision** | **NF4** (`load_in_4bit`, nf4, double-quant, bf16 compute) — `sft_train_modal.py:104-110`, unconditional |
| pinned image | torch 2.13.0 · transformers 5.5.3 · trl 1.10.0 · peft 0.19.1 · accelerate 1.14.0 · datasets 5.0.0 · bnb 0.50.1 |

v1 was fitted against **NF4** weights and is **served** against BF16 weights.
The corrective run uses a frozen **BF16** base to match deployment; that is a
deliberate change of base precision, recorded here so it is not mistaken for
preservation. If BF16 does not fit, **stop and reconsider NF4** — do not switch
automatically.

---

## Phase 2: Build the protocol-corrected dataset

### Sources

The same teacher knowledge v1 originally learned. No new mined tasks. Not
`/data/sft-clean` alone.

- 117 successful DeepSeek-V4 rollouts
- 165 source segments
- 34 tasks
- Both canonical teacher-run directories:
  `/data/vektori-out/dsv4-corpus60` and `/data/vektori-out/dsv4-corpus60-b`

### The two representations

Every teacher action exists twice.

**Raw capture** — what DeepSeek actually emitted, in
`<run>/captures/token_captures.jsonl` (one global file per run, written by
`CaptureProxy`; 6,351 + 5,144 = 11,495 records):

```json
{
  "analysis": "The repository is already present...",
  "plan": "Inspect the implementation...",
  "commands": [{"keystrokes": "sed -n '180,420p' src/anyio/from_thread.py\n"}]
}
```

**Harbor trajectory (ATIF)** — a logging representation created *after*
parsing that response. `terminus_2.py:1338-1348` writes
`message = f"Analysis: {analysis}\nPlan: {plan}"` and the commands as
structured `tool_calls`. The raw JSON string is **not** stored: the
`save_raw_content_in_trajectory` flag that would have stored it was off.

### The join

"Joining" means identifying which raw completion belongs to which trajectory
action. The capture file carries **no rollout identity** — it is global across
concurrent rollouts — so the join is on content and time:

1. **Timestamp:** capture `request_started_at` / `created_at` against the ATIF
   step `timestamp`.
2. **Completion order:** first action response with first action turn, and so
   on within a rollout.
3. **Prompt/call order:** distinguish normal action calls from summarization
   and other calls.
4. **Command equivalence:** parse the raw JSON with Harbor's real Terminus
   parser and confirm its commands equal the commands ATIF recorded.

```text
Raw response command: ls -la
ATIF tool call:       bash_command("ls -la")   → match
Raw response: pytest, ATIF: sed              → mismatch, join is wrong
```

An earlier naive join keyed on the first 300 chars of `analysis` reached
97.3% (3,528/3,627) with 1 collision. That key is retired.

**The key is the full semantic tuple**, because a full `Analysis/Plan` string
removes the *known* collision but does not make collisions impossible — two
responses can share analysis and plan and differ in commands:

```text
key = full rendered "Analysis: {a}\nPlan: {p}"
    + normalized commands (keystrokes, duration)
    + task_complete
    + capture/call order
```

Note the rendered string is built by *inverting* `terminus_2.py:1345` on the raw
side rather than splitting ATIF's `message` — splitting is ambiguous whenever
an analysis itself contains `"\nPlan: "`, inverting never is.

If duplicate tuples produce identical targets, the ambiguity is harmless and is
recorded. Otherwise **fail loudly**.

### Reconstruction — hybrid

ATIF is **semantically** lossless here but not byte-for-byte: the original
whitespace, key order and escaping are gone. Harmless, provided rebuilt targets
use one canonical serializer matching the dominant raw style.

For each parent-agent action:

```text
target = capture.text        if it parses AND its commands match ATIF
                                AND its task_complete matches ATIF
       = canonical rebuild    otherwise
```

1. Identify the matching raw teacher completion.
2. Parse it with Harbor's real Terminus parser.
3. Compare its commands **and `task_complete`** with ATIF's record.
4. Store the complete raw JSON as assistant **content**.
5. Store terminal output as the next `role="user"` message.
6. Record provenance on every assistant turn:
   `"target_source": "raw_capture" | "atif_rebuilt"`.

**Justify the rebuild before trusting it.** On the ~97% of turns where both
representations exist, assert

```text
parse(raw target) == parse(ATIF-rebuilt target)
```

over `analysis`, `plan`, `commands` (keystrokes *and* durations) and
`task_complete`. If equivalence holds across the joined turns, rebuilding the
remaining ~3% is justified. If it does not, the rebuild is wrong and the
mismatch is the finding.

Result: **165/165 segments, 3,627/3,627 turns**, with no guessing and no
incoherent truncation.

### Turn classification and supervision

Every assistant turn is first classified:

```text
action             -> supervised
handoff_question   -> context, masked
summarization      -> context, masked
parse_error        -> context, masked
unknown            -> FAIL THE AUDIT
```

`parse_error` was added after the first real run: 18 turns landed in `unknown`
and failed the audit as designed, and all 18 turned out to be one known shape —
a response terminus's parser **rejected**. DeepSeek emitted prose with a
```` ```bash ```` fence, or JSON cut off at the output limit; terminus recorded
the raw text with no tool calls (`terminus_2.py:1355-1370`) and asked it to try
again. **v1 was trained on those 18 as targets**, i.e. on the exact malformed
output this run repairs.

They are masked but kept, because the next action's own analysis says *"the
previous command batch was not executed due to invalid JSON in my response"* —
dropping the failure would leave that referring to nothing, and the recovery
that follows is one of the behaviours Phase 7 tests for. The error path writes
no user step, so that reply is **recomputed** from the stored response with
harbor's parser and terminus's own format string; a test pins both literals
against harbor's source so an upgrade cannot drift them silently.

`unknown` now means an agent step with no tool calls whose message nevertheless
*parses* as an action — a combination with no explanation, which is what that
bucket is for.

Train **only** `action`. There are ~48 `handoff_question` turns (one per
compaction boundary: 165 segments − 117 rollouts), carrying prose like
*"Before I continue implementing, I need the following clarifications: ..."*
from `trajectory.summarization-N-questions.json`. v1 was trained on those
(`sft_export_traces.py:75-91` pulls the head in wholesale).

They are not pure noise — they occur under a distinct handoff-question prompt,
so a model could in principle learn the conditional (action prompt → JSON,
handoff prompt → prose). But this run has one narrow objective, repair action
serialization, and 48 non-action prose targets only broaden it. Keeping them as
context preserves the genuine inference conversation structure and lets the
following actions see the real questions and answers; v1 has already seen these
examples and base Qwen can already write prose. Handoff and summarization
behavior is tested **separately, after** the repair.

Dropping them entirely (collapsing the head into one user message) was rejected:
it leaves the "Here are the answers..." user message dangling against questions
absent from the context.

Explicit mappings:

- `bash_command` → one `commands[]` entry.
- `mark_task_complete` → `"task_complete": true`.
- Never place `mark_task_complete` inside `commands`.
- Preserve raw teacher `analysis` and `plan`.
- Remove structured `tool_calls`, ids and tool roles.
- Exclude subagent turns.
- Exclude summarization completions as action targets.
- Summaries may remain as context after compaction.
- Pin `enable_thinking=False`.
- Do not put `<think>` tags into targets. DeepSeek's CoT arrives out of band in
  `reasoning_content` and is discarded.

### Missing data

The hybrid rule means an unmatched capture is **not** missing data — it falls
back to the rebuild. Truncation is reserved for an action that cannot be
represented at all (ATIF itself unparseable, or a tool call with no
recoverable keystrokes):

- Truncate immediately before that action.
- Keep the valid prefix if it contains supervised assistant responses.
- Otherwise drop the segment.

Never leave an observation responding to a removed command.

### Dataset audit report

- 165 source segments
- complete / truncated / dropped segments
- recovered assistant turns / total
- commands retained
- edit / test / read operations retained
- `task_complete` mappings
- first-command distribution
- join failures with provenance
- token lengths
- dataset checksum

Final yield may be below 165. Acceptable if every retained conversation is
coherent.

---

## Phase 3: Free preflight checks

We pre-tokenize and own the labels, so preflight must additionally prove:

- token ids render with the **exact serving chat template**
- `enable_thinking=False` is explicit
- labels are `-100` everywhere except selected action spans
- no example truncates
- every row has ≥1 supervised action
- decoding the supervised spans reproduces the raw JSON
- the custom mask **matches TRL's** mask on ordinary action-only rows

That last one is the safety net, and it is deliberately **asymmetric** rather
than exact equality. Both sides render with TRL's template — comparing counts
across two different renders would measure delimiters, not masks — and then:

- a position TRL supervises and we do not is a **hard failure**: that is a
  target we masked;
- positions we supervise beyond TRL's span are **expected and bounded**,
  because TRL's `{% generation %}` markers stop before each turn's closing
  delimiter and we include it.

Exact positional equality would fail every row for that delimiter alone, so it
would have to be switched off — a check that cannot pass is not a safety net.
Rows carrying a handoff or parse-error turn are excluded from the comparison
entirely, since there the two masks *must* differ.

Before a GPU:

1. Every assistant target parses as Terminus JSON.
2. Parsed commands match ATIF.
3. No invented tool verbs.
4. No assistant `tool_calls`.
5. No `role="tool"` observations.
6. No `<tool_call>` or `<think>` text in targets.
7. `task_complete` mapped correctly.
8. Every row renders below 40,960 Qwen tokens.
9. No silent truncation.
10. Every row has assistant supervision.
11. Masks verified across **every** row, not just batch zero.
12. Rendered with the exact chat template intended for serving.

13. `parse(raw) == parse(rebuild)` on every joined turn (the rebuild
    justification above).

**No dev split.** Train all 165 rows. This is a protocol transplant over a tiny
dataset; v1 has already seen every semantic demonstration, and holding out 10%
would only prevent correcting the protocol on those rollouts. Dev loss is a
weaker signal than generation validity for this objective — Phase 7 is the real
selection criterion.

---

## Phase 4: The v1-continuation trainer

The existing trainer starts a new adapter (`sft_train_modal.py:137` builds a
fresh `LoraConfig`) and cannot be used unchanged.

It also derives the loss mask from the chat template, which is per-**role** and
all-or-nothing — there is no way to supervise one assistant turn and not
another. So supervision moves off TRL: a standard Transformers trainer over a
**pre-tokenized** dataset with a label-preserving collator.

Benefits: selective masking of handoff turns; every label verifiable offline;
no all-or-nothing role mask; no Liger mask drop (TRL #3781); no silent mask loss
at truncation (TRL #3927); the exact supervised tokens auditable and
reproducible.

`vektori_trace/dataset.py:tokenize_sft_example` needs a small generalization —
it supports assistant-role masking and prefix masking, not arbitrary per-turn
classification. Add an explicit per-message supervision mask:

```text
False  user      handoff request
False  assistant clarification questions
False  user      answers
True   assistant JSON action
False  user      terminal output
True   assistant JSON action
```

The repaired trainer must:

1. Load `Qwen/Qwen3-14B` in BF16.
2. Freeze all base weights.
3. Load v1 through PEFT with `is_trainable=True`.
4. Confirm only existing LoRA parameters require gradients.
5. Not construct a new `LoraConfig`.
6. Not stack a second LoRA.
7. Disable model cache during training.
8. Save to a new repaired-v1 directory.
9. Checkpoint every ten optimizer steps.
10. Evaluate each checkpoint on the fixed real-prefix suite.

LoRA architecture is unchanged:

```text
rank: 32 · alpha: 64 · dropout: 0.05 · target modules: all-linear · dtype: BF16
```

---

## Phase 5: BF16 memory and correctness probe — **needs approval**

**Revised 2026-08-18: BF16 only.** The plan previously ran both arms. The NF4
arm cannot change the decision — if BF16 fits we take it on deployment-match
grounds, and if BF16 OOMs we measure NF4 *then*, as the fallback. Running it up
front buys a number discarded in the likely case. Speed cannot flip the choice
either: BF16 is taken because step 0 of a BF16 continuation is exactly the model
measured as broken and because it absorbs v1's quantization drift, and a 20%
step-time edge would not outweigh either.

(If the same-batch same-seed NF4-vs-BF16 step-time comparison is wanted as a
*paper* number, run both — that is a writeup decision, not a training one.)

**Three optimizer steps on A100-80GB**, on the **longest rows**, nothing saved.
Peak VRAM is dominated by sequence length — the logit tensor alone is
`len x 151,936` — so a probe on a shuffled sample would certify a footprint the
real run then exceeds. Selecting only long rows makes the measurement
deliberately conservative.

### Why A100-80GB, and why not L40S

|  | NF4 | BF16 |
|---|---|---|
| v1 measured peak | 39.6 GiB | — |
| estimated peak | ~40 GiB | ~59–63 GiB |
| A100-80GB headroom | ~40 GiB | ~17–21 GiB |

~17–21 GiB is the right margin for allocator fragmentation, the 37,577-token
example, and evaluation — appropriate, not excessive.

L40S 48 GB is **serving hardware, not this run**: BF16 almost certainly does
not fit at 40k context, NF4 would fit only barely with no safe headroom, and
its memory bandwidth is substantially below A100's.

### Speed expectation

BF16 is likely only **5–25%** faster than NF4, not 2×. NF4 saves memory
bandwidth but pays dequantization; BF16 gets optimized tensor-core kernels but
reads larger weights, and at 40k context attention, activations and
cross-entropy dominate enough that the gap may be small.

For 63 steps, from v1's measured 167 s/step at NF4:

- NF4: ~2h55m
- BF16: plausibly ~2h20m–2h50m
- with checkpoint evaluation: ~3–4h total

Only a same-batch probe measures this reliably.

### Both arms check

- peak VRAM
- finite loss
- finite gradient norm
- LoRA parameters change
- base parameters stay frozen
- assistant mask correct
- no accidental fresh adapter
- correct dataset loaded
- no checkpoint published as a model candidate

Tear down immediately. If BF16 does not fit safely, **stop and discuss** before
falling back to NF4. Do not switch silently.

---

## Phase 6: Corrective SFT — **needs separate approval**

```text
base precision:      per Phase 5 probe (BF16 expected), frozen
LoRA precision:      BF16
optimizer:           AdamW fused
learning rate:       1e-5
betas:               0.9, 0.999
epsilon:             1e-8
weight decay:        0
max grad norm:       1.0
schedule:            constant_with_warmup
warmup steps:        5
batch size:          1
gradient accumulation: 8
max epochs:          3
max length:          40960
packing:             false
assistant-only loss: true
label smoothing:     0
gradient checkpointing: true
seed:                0
```

All 165 rows: `ceil(165 / 8) = 21` optimizer steps/epoch × 3 epochs =
**63 max steps**. Checkpoint at **10, 20, 30, 40, 50, 60, 63**. Recompute if
fewer rows survive the audit — do not force 63.

If there is effectively no format improvement after ~20 steps: stop, return to
original v1, do not raise LR mid-run. A separately approved `2e-5` branch can
follow.

---

## Phase 7: Checkpoint evaluation — **built; needs one approved endpoint session**

At every saved checkpoint, **no shim**, and through the real serving stack:
vLLM, BF16 base, LoRA on top. Not a PEFT load in the training precision — the
run trained on NF4 (above), so anything else would certify a configuration that
will never be deployed.

Every number the corrective run produced is teacher-forced, which hands the
model a correct prefix at every position — exactly the crutch that is absent at
generation time, and generation time is where v1 failed. The probe log records
**mean token accuracy 0.81 at step 1**: v1, the model that emitted 305
consecutive `Missing required fields`, already reproduces 81% of the corrected
protocol's tokens under teacher forcing, because the supervised span is mostly
`analysis`/`plan` prose and keystrokes it already knows. The envelope is a small
minority of the 1.03M supervised tokens. Loss 0.6436 → 0.4856 is therefore
consistent with the envelope having flipped *and* with it not having flipped.
Phase 7 is the first measurement that can tell the difference.

### The fixed prefix suite

A *prompt* is the complete conversation immediately before the teacher produced
an action:

```text
Terminus instructions
Task description
Previous assistant actions
Terminal observations
```

Extracted automatically from real teacher trajectories, and read straight out of
the corpora rather than re-rendered — a second renderer is a second chance to
differ from what the adapter actually saw. Nobody invents the expected response;
the captured DeepSeek response is the reference.

"Fixed" means the corpus, line number and message index of every prefix are
frozen in a manifest with a sha256, so every checkpoint gets precisely the same
inputs.

### Three cells, not two

The unit of variation is the **conversation state**, not the task. Emitting flat
JSON has nothing to do with click versus jinja; it plausibly has everything to
do with turn 1 at 3k of context versus turn 30 at 35k after a compaction.
Sampling fifty tasks all at turn 1 would buy almost nothing over sampling five.

Training consumed only *passing* rollouts. Measured on the box: **all 26
held-out tasks have zero passing rollouts** — that is what put them outside the
34. A held-out prefix can therefore only come from a rollout that failed, which
moves two variables at once. So:

|                | passing rollout        | failing rollout    |
|----------------|------------------------|--------------------|
| trained task   | **acquisition** (117)  | **control** (18)   |
| held-out task  | *does not exist* (0)   | **generalization** (103) |

- `acquisition → control` moves only rollout outcome.
- `control → generalization` moves only task familiarity.

Without the control cell a low generalization number is unreadable: "did not
generalize" and "these are messier conversations to continue" produce the same
digit. Suite 1 leaking from training is fine and expected — prefixes are inputs,
not labels — but it cannot answer the question on its own either, because with
165 segments over 3 epochs a checkpoint can reproduce a memorised continuation.
Suite 2 next to the control is where the honest signal lives.

Suites are matched category-for-category; a category present in one and absent
from another silently changes what the two numbers mean, and the manifest
builder warns when that happens.

### Categories

`orientation` · `repo_present` · `first_inspection` · `first_edit` ·
`test_exec` · `parse_error_recovery` · `post_compaction` · `long_context`.

Prefixes are also spread across repos: the five differ in file layout, test
invocation and terminal-output shape, and drawing every held-out prefix from
click would test one repo's idiom and call it generalization.

### Two format tiers, because harbor is lenient

```text
harbor_accepts  — TerminusJSONPlainParser returns no error. The rollout proceeds.
native_json     — additionally the completion *is* the object: no ```json fence,
                  no <think> block, no prose preamble.
```

Harbor salvages a fenced object with only a warning ("Extra text detected before
JSON object"). Scoring that as a plain pass would hide a checkpoint that is
nearly repaired but still wrapping its output, and would leave
`enable_thinking=False` unverified; scoring it as a plain failure would claim a
rollout breaks when it would in fact run. Both are recorded; selection reads the
strict one.

### Greedy suite

```text
temperature 0 · max new tokens 512 · timeout 120s · enable_thinking false
```

Format gates, applied to every prefix: `harbor_accepts` · `native_json` ·
`required_fields` · `no_legacy_envelope` · `no_invented_fields` ·
`command_structure` · `eos_before_limit` · `non_repetition`.

Behavior gates, applied only where the prefix's state can test them:
`orientation` · `no_clone_when_git_exists` · `edit_emission` · `test_emission` ·
`recovery`.

One prefix per category per suite is deliberate. The format gates — the question
actually being asked — are graded on **every** prefix regardless of category, so
each checkpoint gets 16–24 samples on those and 1–3 on the secondary behavior
gates. Against a v1 baseline of 0/305 that is ample to separate "repaired" from
"not"; it is not ample to rank two working checkpoints, which is not the job.

### Sampled suite

Representative prompts at temperature 0.7, multiple seeds — catches a format
that works greedily but collapses under sampling. Run **after** a checkpoint
passes greedily, not before.

### Selection

```text
suite:  generalization           (falls back to acquisition, loudly, if absent)
gates:  native_json · required_fields · no_legacy_envelope
rule:   earliest checkpoint clearing all three on EVERY prefix of that suite
```

The **earliest**, not the final one and not the lowest training loss — loss fell
monotonically across all 63 steps, so a loss rule always returns checkpoint 63,
the one most overfit to 34 tasks, and the checkpoints exist precisely so that
choice can be made on behavior instead.

**This narrows the gate list the plan originally carried.** Orientation,
edit/test emission and no-clone were listed among the selection criteria; they
are now *reported* but not selected on. At one prefix per category they are
anecdotes — enough to notice a regression, nowhere near enough to reject a
checkpoint — and the repair being measured is the envelope, not strategy.
Strategy is Phase 9's question, and a rollout that fails strategically while
emitting valid JSON is exactly what OPD is for.

**An ungraded prefix blocks its checkpoint.** A request the endpoint dropped
produces no result at all, so a checkpoint answering 7 of 8 prefixes perfectly
and losing the eighth to an HTTP error would otherwise read as a clean sweep —
silence scoring as success. The manifest's prefix ids are passed to the selector
and a missing one fails the checkpoint outright, the same correction
`no_gradeable_rollouts` needed in the pass@k reports. Transport is retried
(4xx is not — that is the request being wrong, not the network being flaky), and
the sweep is preflighted: one 16-token request per registered adapter, checked
for reachability and for an absent `<think>` block, because
`chat_template_kwargs` is a vLLM extension and a server that ignored it would
leave `enable_thinking=False` unpinned.

### Staging, and what it costs

Selection is "earliest passing", so testing all seven checkpoints up front
computes six answers that get discarded. The driver walks them in training order
and stops at the first pass, after smoking the *last* checkpoint first — if the
most-trained checkpoint cannot emit the protocol, nothing earlier will. Not a
proof (the run could have overfit past a working middle checkpoint), which is
what `--strategy all` is for, but it aborts a dead repair for the price of one
checkpoint instead of seven.

| outcome | generations | wall clock |
|---|---|---|
| smoke on ck63 fails → stop | 16 | ~3 min |
| ck10 passes → done | 32 | ~5 min |
| passes at ck30 | 64 | ~10 min |
| nothing passes | 112–168 | ~25 min |

Plus ~15 min endpoint boot, which dominates. **A100-80GB**, not L40S: the KV
budget is 282k tokens against L40S's 93.7k, and prefixes run to ~37k.
`--max-lora-rank 32` is mandatory — vLLM defaults to 16, every adapter here is
rank 32, and the engine refuses to start *after* the GPU is allocated.

### What was built

| | |
|---|---|
| `vektori_trace/evaluate/phase7.py` | gates, summary, earliest-passing selection |
| `scripts/phase7_manifest.py` | freezes the three suites with a sha256 |
| `scripts/phase7_eval.py` | drives one endpoint, grades, reports |
| `tests/test_phase7.py` | 27 tests: teacher output as positives, v1's `<tool_call>` envelope and the 18 rejected turns as negatives |
| `scripts/sft_repair_dataset.py` | `--rollouts passing\|failing\|all`, `--tasks-in/--tasks-not-in`, `--audit-advisory` |
| `vektori_trace/runtime/serve.py` | multiple LoRAs on one base load, `--max-lora-rank`, `--max-loras` |

The grader is validated against fixtures before it is pointed at a GPU: an
untested predicate is not evidence about a checkpoint.

An endpoint that refused a request is recorded under `infra_failures`, never as
a checkpoint that failed a gate — the same correction `fallback_exitcode`
needed in the pass@k reports.

---

## Phase 8: Native serving validation — **needs an approved endpoint session**

Repaired-v1 directly, no shim.

1. Adapter appears in `/v1/models`.
2. Requests explicitly select its model id.
3. Bogus model ids return 404.
4. Base and adapter outputs differ.
5. Rank 32 supported explicitly.
6. `enable_thinking=False` honored.
7. Summarization requests produce summaries, not action JSON.
8. Client cancellation reaches vLLM.
9. Requests have output and wall-clock limits.
10. No completion can hold the GPU indefinitely.

---

## Phase 9: One guarded Harbor rollout — **needs approval**

One worker · one rollout · a known training task first · ten-minute cap ·
`--no-escalate` · no shim.

Kill after: two clone attempts · two identical command batches · two minutes
without a response · a parser loop · a runaway completion.

The rollout does **not** need to pass. Repair succeeds if: native JSON ·
commands execute · correct orientation · no parser loop · no runaway
generation · some source-level progress · edit emission when the state clearly
calls for it.

Valid JSON but strategic failure is exactly what OPD is for.

---

## Phase 10: OPD

Proceed if repaired-v1 reliably emits native JSON, runs without middleware,
does not wedge, can navigate and execute commands, and shows at least minimal
edit behavior. Verifier passes are **not** required.

Start with teacher-prefix / ReOPD:

1. Real successful teacher prefixes.
2. Student generates the next action.
3. Teacher scores the same student-generated text.
4. Cross-tokenizer alignment via the byte bridge.
5. No converter changes text between sampling and scoring.
6. Native JSON is a hard retention gate.

Small approved probe first. Longer run only if alignment checks pass, loss is
finite, gradients are nonzero, JSON format does not regress, and commands
remain executable.

---

## Phase 11: Evaluate support

Small approved pass@k sample. Check gradeability, infra failures, parser
status, no `fallback_exitcode`, no escalation.

≥1 passing rollout → RL has a usable reward difference. All fail → more OPD or
revisit data/capability. **Never run reward-only RL on all-zero groups.**
