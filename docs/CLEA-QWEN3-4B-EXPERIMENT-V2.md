# CLEA Tau2 experiment V2: SFT warm start -> replay OPD

Status: second-draft research plan, amended 2026-08-24 (see 1.1); no GPU or
paid run is authorized by this file  
Date: 2026-08-24  
Target: CLEA @ NeurIPS 2026  
Scope: Tau2-bench only; Qwen3-4B viability first, Qwen3-8B fallback/reference

This file supersedes `docs/CLEA-QWEN3-4B-EXPERIMENT.md` for the CLEA/Tau2
experiment design. It does **not** supersede the Harbor-specific operational
SFT and OPD plans named in `CLAUDE.md`; their implementation lessons remain
relevant, but their dataset sizes are not evidence for Tau2's required SFT
size.

## 1. Decision and correction from V1

Give the exact post-trained Qwen3-4B checkpoint a bounded two-day viability
experiment before committing the full run to 8B. The two-day experiment has a
pre-registered stop/continue rule. If 4B fails it, switch to 8B without changing
the split or evaluation protocol.

V2 removes two unsupported assumptions from the first discussion:

1. There is no reason to require two DeepSeek trajectories per Tau2 task when
   only one has been collected. Separate expert trajectories are not a leakage
   requirement.
2. There is no evidence that exactly 24 trajectories is the right SFT size.
   The SFT corpus is defined by a frozen task boundary and trace-quality rules,
   then its actual action/token counts are measured before choosing updates.

Telecom is not used in the two-day model-size decision. It is the later,
genuine continual-learning distribution shift.

## 1.1 Amendment 2026-08-24: warm-up/adaptation task split

V2 as first written answered a different research question than the one
intended. This section records the change so the reversal is auditable.

**What V2 said before this amendment.** Section 1 closed with:

> For the retail experiment, initial SFT, continued SFT, and replay OPD may use
> the **same eligible training tasks and stored DeepSeek trajectories**. This is
> the cleanest comparison of learning objectives under the available data:
> continued SFT repeats the recorded expert action, whereas replay OPD samples a
> current-student action at the same expert state and has DeepSeek score it.

and section 6.2 instructed:

> Use **all eligible DeepSeek trajectories from the 60 retail-train tasks**. Do
> not arbitrarily stop at 24 tasks and do not reserve half the tasks solely to
> manufacture a second SFT phase.

That produced `60 tasks -> A_warm`, then `same 60 -> continued SFT vs replay
OPD`, which answers:

> Does replay OPD outperform additional SFT on already-seen expert states?

**Why it is being changed.** The intended question is adaptation, not
re-optimization of seen states:

> After acquiring basic Tau2 behavior, which method adapts better to *new*
> retail tasks?

The V1 critique that V2 was written against correctly rejected 24 as an invented
number, the requirement of two DeepSeek traces per task, and separate traces
used merely to prevent leakage. It did **not** establish that reserving tasks
for a distinct adaptation stage is invalid. "Do not arbitrarily hold back data
when testing 4B capacity" was over-generalized into "never reserve tasks for an
adaptation stage." Reserving 30 tasks whose purpose *is* adaptation is a
research-design choice, not an arbitrary data handicap.

**What replaces it.** Retail-train 60 is split into two frozen, family-balanced
halves:

```text
retail-train 60
  |-- W: warmup 30
  |     `-- SFT -> A_warm
  `-- C: adaptation 30
        |-- continued SFT -> A_sft_new   (recorded DeepSeek action at prefix)
        `-- replay OPD    -> A_reopd     (student samples, DeepSeek scores)
```

The outer split becomes 30/30/16/38, exhausting all 114 retail tasks (see 4).
Both continuation branches start from the identical `A_warm` artifact and
consume the identical frozen C30 prefix pool.

**Consequences that must be reported, not assumed away.**

- The branches are *not* information-symmetric. `A_sft_new` receives the
  expert's recorded action at each adaptation prefix; `A_reopd` receives a
  teacher-derived score for an action the student itself sampled. These are
  different learning signals, not the same signal in different amounts. It is
  reasonable to hypothesize that the explicit target is the more direct
  imitation signal, but which is more useful in practice is precisely what the
  experiment measures — the plan must not assert the outcome in advance.
- Novelty is **not** confounded in the primary comparison. `A_sft_new` and
  `A_reopd` share the same `A_warm`, the same C30 tasks, the same prefixes and
  the same budget, so task novelty cancels between them. Novelty is confounded
  only when reading `A_reopd - A_warm` in isolation, which is not the method
  claim. A same-task replay arm (`A_reopd_same`, replay OPD over the W30
  prefixes) would isolate that difference, but it is a **later diagnostic** and
  is not part of the two-day run.
- W30 and C30 must be family-balanced against each other under the section 4
  grouping rules. An unbalanced halving measures family transfer, not method.
- Budget matching is over updates, prefix exposures, sampling order, effective
  batch size and LoRA capacity — not over token counts, which cannot be matched
  because sampled and recorded actions differ in length. Supervised tokens, GPU
  time and teacher-inference cost are reported, not forced (see 8).

Sections 6.2, 7, 8 and 9 below are amended accordingly. Where earlier text still
reads "all 60", the W30/C30 split in this section governs.

## 2. Research basis

The recipe follows these principles rather than inheriting the previous Harbor
SFT row counts:

- Tau2 is a multi-turn agent benchmark, and its telecom tasks are generated
  compositionally from atomic components. Diversity of workflows and decision
  states is therefore more meaningful than a raw trajectory count
  ([Tau2 paper](https://arxiv.org/abs/2506.07982)).
- Instruction-tuning evidence consistently identifies task diversity as an
  important driver of unseen-task generalization; increasing repetitions of a
  narrow task set is not equivalent to increasing task coverage
  ([FLAN](https://arxiv.org/abs/2109.01652),
  [Flan-PaLM](https://arxiv.org/abs/2210.11416)).
- Small, curated SFT sets can move a strong pretrained model, but LIMA's 1,000
  examples on a 65B model does not establish a universal minimum for a 4B
  tool-using agent. It supports quality over indiscriminate volume, not the
  claim that 24 Tau2 traces must suffice
  ([LIMA](https://arxiv.org/abs/2305.11206)).
- Agent-FLAN finds that agent capabilities learn at different rates, that data
  composition matters, and that action/format hallucination requires explicit
  attention. Its result that the first 25% of its own corpus produced most of
  the gain is evidence for measuring a scaling curve, not a transferable
  absolute sample count
  ([Agent-FLAN](https://arxiv.org/abs/2403.12881)).
- TRL supports conversational SFT with loss restricted to assistant tokens.
  Qwen3 templates can be patched for assistant masking, but the emitted mask
  and non-truncation must still be verified on the exact pinned tokenizer
  ([TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)).
- Replay OPD is specifically designed to reuse stored teacher trajectories as
  prefixes while sampling the current student's action. Reusing a trajectory
  first for SFT and later as a replay-state source is therefore not a test leak;
  evaluation-task reuse would be
  ([ReOPD paper](https://arxiv.org/abs/2607.04763)).

No cited paper proves the correct Tau2 SFT size. V2 therefore treats SFT size
as a measured scaling/early-stopping question inside the frozen Tau2 selection
boundary.

## 3. Research questions

### Two-day decision question

> Does Tau2-only SFT give Qwen3-4B a non-floor, policy-compliant retail policy,
> and does a small replay-OPD continuation improve that policy enough to justify
> using 4B for the full experiment?

### Main method question

> Starting from the same Tau2 SFT checkpoint and using the same expert replay
> states, does replay OPD outperform compute-matched continued SFT on unseen
> Tau2 tasks?

### Continual-learning question

> After retail acquisition, does sequential adaptation to Tau2 telecom cause
> less retail forgetting under replay OPD than under continued SFT?

## 4. Tau2-only data boundary

Retail has 114 tasks at the pinned Tau2 revision. The proposed outer split is:

| Partition | Tasks | Permitted use |
|---|---:|---|
| retail train — W30 warm-up | 30 | `A_warm` SFT only |
| retail train — C30 adaptation | 30 | continued SFT, replay prefixes, DeepSeek scoring |
| retail selection panel S16 | 16 | model/checkpoint/recipe decisions only |
| retail final test F38 | 38 | one frozen final comparison; no tuning |
| **Total** | **114** | |

> **Superseded (see 1.1).** The panel/test sizes were 18/36 in the original
> draft, which left the partitions summing to 114 only by coincidence of a
> different train count. They are now 16/38 so that the four partitions exhaust
> the benchmark exactly: 30 + 30 + 16 + 38 = 114.

Only the 60 training tasks require DeepSeek traces. S16 and F38 are evaluated by
running the student live against the Tau2 simulator, so a task with no stored
trace — or with only failing traces — is a perfectly good evaluation task. This
is what makes the arithmetic work without collecting anything new.

The W30/C30 halving is part of the frozen manifest, not a runtime choice. It
uses the same normalized family grouping and deterministic seed as the outer
split, and W30 and C30 must be balanced against each other; otherwise the
adaptation stage measures family transfer rather than method (see 1.1).

All four partitions are Tau2. “Selection panel” does not mean an outside dev
dataset.

The split unit is a normalized task family, not a DeepSeek trajectory or raw
task ID. Group at least by operation/mutation family, state, required policy
gates, conditional structure, and normalized instruction template. Allocate
groups with a deterministic seed while balancing observed public difficulty.
> **Superseded (see 1.1).** This previously read "Do not use DeepSeek
> success/failure to decide the outer split," which the current design violates
> literally: train60 is selected precisely from tasks DeepSeek passed.

DeepSeek success/failure may be used for **one purpose only** — determining
which tasks are eligible to be training tasks, since a training task without a
passing trace has nothing to imitate. It must not be used for allocation:

```text
permitted : passing trace  ->  candidate for train60
forbidden : DeepSeek outcome  ->  W30 versus C30
forbidden : DeepSeek outcome  ->  S16 versus F38
```

Within the eligible training pool, and within the remaining evaluation pool,
allocation is by normalized family and public difficulty band under a
deterministic seed. The easy-train/harder-evaluation bias this induces at the
pool level is disclosed in 4.3.

Tasks 57, 73, 75, and 93 have already influenced model and metric design. Put
them in a named diagnostic subset of the selection panel, not in the blind
final-test statistic.

> **Superseded (see 1.1).** This previously asserted "DeepSeek traces already
> exist for the entire benchmark." They do not; that line predates any count.
> The measured inventory in 4.1 is authoritative: 86 of 114 tasks traced, 73 with
> a passing trace, 28 never attempted.

Coverage is partial, which does not affect the split because only W30 and C30
require teacher traces. Whatever selection and test traces do exist must be
hashed and quarantined. The enforceable invariant is:

```text
A_warm SFT task IDs                        subset of W30
continued-SFT and replay-prefix task IDs   subset of C30
W30 intersect C30                          empty

selection/test trace content, prefixes, actions and teacher scores
    never enter optimization
```

The `W30 intersect C30 == empty` assertion is what makes the adaptation stage
an adaptation stage; it must be enforced in code, not by convention.

### 4.1 Measured trace inventory (2026-08-24)

Counted from `/data/tau2/data/simulations/flash_retail{20,20_c4,_p1,_rest}.json`
on the EC2 box. The teacher is DeepSeek-Flash at Tau2 v0.2.0, commit `f8de30c`.

| Quantity | Count |
|---|---:|
| Retail tasks in the benchmark | 114 |
| Tasks with at least one DeepSeek trace | 86 |
| Tasks with at least one `reward == 1` trace | 73 |
| Tasks traced but never passing | 13 |
| Tasks never attempted | 28 |

Non-passing traced tasks: 18, 22, 52, 56, 59, 62, 82, 91, 100, 105, 107, 110,
111.

**Structural eligibility audit result (2026-08-24,
`/data/tau2/retail_eligibility.json`).** The structural gates were run over all
73 passing candidates. **All 73 pass this first audit; structural attrition is
zero.** Final training eligibility additionally requires the policy and exact
Qwen-rendering gates in section 5; do not call all 73 fully eligible until those
checks have also passed.

| Quantity | Count |
|---|---:|
| Passing candidates audited | 73 |
| Structurally eligible | **73** |
| Structurally rejected | 0 |

Structural gates applied: reward pass, clean termination, no simulator/provider error,
structured action parsing, no orphan tool results, at least three usable
decision positions, and auditable reward components.

The files contain 78 passing trace records over the 73 tasks because five tasks
have a second passing trace. Training uses exactly one trace per task. After
final eligibility, sort each task's eligible traces by the immutable
`(source_file, simulation_index)` key and take the first. Record that key and
the trace hash; never choose a trace by inspecting which trajectory looks
easiest to imitate.

Decision-position census over the structurally eligible traces:

| Statistic | Value |
|---|---:|
| Assistant actions per trace | min 7, median 10, p90 12, max 22 |
| Passing trace records with at least six assistant actions | 78/78 (100%) |
| Tasks represented by those records | 73 |
| Six-action arithmetic ceiling after one-trace-per-task selection | 438 |

Having six assistant actions is necessary but does not by itself prove that six
hand-labelled semantic categories exist. A trace may contain no refusal, no
recovery, or no clarification, in which case its category slots cannot all be
filled from distinct categories.

**180 is an upper bound, not a prediction.** The realized count is whatever the
deterministic selector in 6.2 produces — including its fallback rule — computed
and recorded after semantic selection runs. Do not budget updates against 180
until the selector has run and its output has been counted.

The 73 structurally eligible tasks cover the 60-task train partition with 13
spare before diagnostic reservations. No further DeepSeek collection is
expected to be required, but that conclusion becomes final only after the policy
and Qwen-rendering gates. If fewer than 60 non-reserved tasks survive, collect
trials on some of the 28 untouched tasks — the 20-task retail survey cost $0.21
— rather than shrinking the panel/test partitions or promoting a frozen
evaluation task into training.

If a supposedly blind trace has already been manually inspected and affected a
decision, move its task to the selection panel and replace it before freezing
the manifest.

### 4.2 Constructing the four partitions

**Step 1 — reserve contaminated diagnostics, then choose train60.** Run the
section 5 eligibility audit over the 73 tasks that have at least one
`reward == 1` trace. Before selecting train60, remove tasks 57, 73, 75 and 93,
plus any other task whose trace was manually inspected or influenced a design
decision; preassign them to S16. Then select 60 eligible training tasks. The
surplus and preassigned diagnostics remain in the evaluation pool.

```text
73 passing-trace candidates
    -> eligibility audit
    -> reserve inspected/design-influencing tasks for S16
    -> 60 eligible training tasks
    -> surplus returns to the evaluation pool
```

The margin is pre-freeze attrition headroom. Once the split is frozen, a
training task is never replaced by a selection or test task. If fewer than 60
survive the audit, either collect another DeepSeek attempt on a **predeclared
training candidate** or report a reduced training count — never promote an
evaluation task.

**Step 2 — halve train60 into W30 and C30.** Deterministic seeded allocation,
balanced across: lookup versus mutation; authentication requirement;
confirmation requirement; clarification and conditional workflows;
refusal/no-action cases; single versus multi-condition tasks; normalized task
family; and approximate trace length and action count.

Two hard rules:

- Do not choose C30 by where `A_warm` performs poorly. The halving is frozen
  before `A_warm` exists.
- Near-duplicate task templates stay on the same side. Otherwise `A_warm` has
  effectively already trained on an "unseen" adaptation task. This is an
  assertion in the split builder, not a guideline.

**Step 3 — S16 and F38 from the remaining 54.** The leftover pool is
approximately 13 unused passing-trace tasks, 13 traced-but-never-passing tasks,
and 28 never-attempted tasks.

Stratify S16 and F38 **by normalized task family and public difficulty band**,
which is the section 4 rule already in force. Trace availability is recorded as
a reporting attribute, not used as a stratification axis. A roughly proportional
spread across trace categories is a secondary constraint, useful as a sanity
check:

| Trace category | Pool | S16 | F38 |
|---|---:|---:|---:|
| unused passing trace | ~13 | ~4 | ~9 |
| traced, no passing trace | ~13 | ~4 | ~9 |
| never attempted | 28 | ~8 | ~20 |
| **Total** | **54** | **16** | **38** |

Where family/difficulty balance and this proportional spread conflict, balance
wins. The reason is that "DeepSeek never passed it" is a difficulty signal:
allocating those 13 tasks by trace category rather than by measured difficulty
would leave S16 and F38 systematically mismatched, and S16 would stop predicting
F38.

Tasks already inspected or used in design decisions — including the named
diagnostic tasks 57, 73, 75 and 93 — go to S16, never F38.

Stored DeepSeek traces for S16 and F38 tasks are hashed and quarantined at the
code boundary. Live evaluation never needs them.

### 4.3 Selection bias that must be reported, not hidden

train60 is chosen as tasks DeepSeek passed. The evaluation pool therefore
inherits every task DeepSeek could not solve. This makes train easy-biased and
S16/F38 hard-biased **by construction**.

The bias is inherent to needing expert traces and is not a defect in the design,
but it has one consequence that must be stated wherever results appear: a gap
between `A_warm` on C30 and `A_warm` on S16 reflects difficulty composition as
well as generalization, and cannot be read as a pure generalization measure.

Report the public difficulty-band distribution of all four partitions alongside
the results so the composition is visible.

### 4.4 Statistical power at n=16

A one-task swing on S16 is 6.25 percentage points. The 5-point gates in section
10 therefore sit inside the noise floor of a single panel evaluation.

S16 gates are **directional screening**, not inference. Report task-level
bootstrap intervals and paired per-task changes, use multiple fixed trials per
task, and treat F38 as the partition where the claim actually lands.

## 5. Expert-trace eligibility

Having one trace per task is sufficient to define the experiment, but not every
trace is automatically training data.

> **Superseded (see 1.1).** This paragraph previously began "Apply eligibility
> only after the task split is frozen." That order belonged to the earlier design
> in which train60 was drawn from all 114 tasks. Under the current design train60
> is *selected from* passing-trace candidates, so eligibility must run first.

Eligibility runs **before** the split, in this order:

```text
73 passing-trace candidates
    -> full eligibility audit
    -> eligible candidates
    -> reserve inspected/design-influencing tasks for S16
    -> select 60
    -> family-balanced deterministic W30/C30 halving
    -> assign the remaining 54 to S16/F38
    -> freeze and hash everything
```

A train trace is usable for SFT/replay when all of the following hold:

1. the simulation is complete and gradeable;
2. its structured assistant messages and tool calls parse exactly;
3. the official Tau2 reward passes;
4. applicable authentication, confirmation, and user-condition gates pass;
5. no mutation occurs after a failed or unresolved precondition;
6. no simulator/provider error explains the result; and
7. the exact rendered history fits the pinned student context without silent
   truncation.

Tau2's official retail/airline/telecom reward is based on DB and communication,
and the documented reference actions are not necessarily the unique successful
path ([Tau2 evaluation documentation](https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md)).
Therefore SFT must preserve the authentic successful conversation rather than
reduce it to the listed golden action sequence. Policy-compliance diagnostics
are reported separately from the official score.

Do not replace an ineligible training trace with a selection/test trace. Either
exclude it and report the reduced eligible-task count or, if time permits,
collect another DeepSeek trial on that same retail-train task.

Successful no-action/refusal trajectories remain useful. They teach when not
to mutate; a mutation-only corpus would train action hallucination.

## 6. SFT warm start

### 6.1 Purpose

SFT is a behavioral initializer for replay OPD, not an attempt to exhaustively
distill DeepSeek. It must establish:

- valid Tau2 assistant/tool serialization;
- correct use of observations and identifiers;
- basic dialogue and tool workflow competence;
- support for safe action and safe no-action decisions; and
- enough probability on plausible actions for DeepSeek scoring to provide a
  useful replay-OPD gradient.

### 6.2 Corpus definition

> **Superseded (see 1.1).** This paragraph previously read: "Use **all eligible
> DeepSeek trajectories from the 60 retail-train tasks**. Do not arbitrarily stop
> at 24 tasks and do not reserve half the tasks solely to manufacture a second
> SFT phase." The second sentence's prohibition is withdrawn: the reservation is
> not to manufacture a phase, it is the adaptation stage the experiment measures.

Use **all eligible DeepSeek trajectories from the W30 warm-up tasks**. The
prohibition that survives is on inventing a number: do not arbitrarily stop at
24 tasks, and do not shrink W30 below the frozen split for convenience. The C30
adaptation tasks are reserved for the continuation branches and must not enter
`A_warm`.

From each W30 trace, build **exactly `min(6, n)`** action-level rows, where `n`
is the number of assistant decisions. Use a deterministic coverage selector:

1. list assistant decisions in chronological order;
2. always retain the first and last;
3. choose the other positions at evenly spaced trajectory quantiles; and
4. resolve rounding collisions by taking the nearest unused position, preferring
   the earlier one on a tie.

This selection rule is deliberately mechanical: it covers early, middle and
late states, can be reproduced without an LLM or manual judgement, and gives
W30 and C30 the same state-selection process. Tag the resulting rows afterward
as lookup, clarification, authentication, confirmation, mutation, no-action,
final or recovery for the coverage report; those descriptive labels do not
choose the rows.

```text
policy + tools + history before action  ->  recorded DeepSeek action
```

Rules:

- fewer than six rows when the trace is shorter;
- never duplicate or invent a state to reach six;
- position selection is deterministic and recorded;
- supervise only the final DeepSeek assistant action in the row;
- never truncate the prefix or the target;
- weight tasks equally, per the loss below.

The arithmetic ceiling is 30 x 6 = 180 rows. The 2026-08-24 structural audit
shows no trace is blocked by a *shortage of actions* — every candidate carries at
least seven assistant messages (min 7, median 10, max 22) — but an action
shortage is not the binding constraint. A trace contributes six rows only if six
distinct categories are actually present in it, which the audit did not test.

Realized W30 rows are therefore <= 180, counted after the selector runs. The
census, not this ceiling, is what the update budget is frozen against.

**Fallback rule.** When a category is absent from a trace, do not leave the slot
empty and do not substitute an arbitrary action. Fill remaining slots from the
un-selected assistant messages in order of decreasing prefix position, so
later-state decisions are preferred over early lookups, and record in the census
which slots were filled by fallback. This keeps the selector deterministic and
makes any category shortfall visible rather than silently absorbed.

> **Superseded (see 1.1).** V2 previously made whole-trajectory supervision the
> default representation — "one conversational trajectory with loss on all
> DeepSeek assistant tokens" — with action-level splitting as a fallback for
> traces that did not fit. That paragraph is deleted. It described a different
> experiment from the action-level rows above, and the two cannot both be the
> canonical representation.

**Action-level rows are canonical for both W30 and C30.** Loss falls on the
single recorded DeepSeek action that terminates the row, and on nothing else:
system, user, simulator and tool-observation tokens are all masked.

The representation is identical on both sides of the train split by
construction. W30 rows and C30 prefixes are produced by the same builder with
the same deterministic position rules, so the continuation branches operate on
states drawn the same way `A_warm` was trained on. Differing representations
between the halves would confound the comparison.

### 6.2.1 Minimal Tau2 trace-to-SFT conversion

The conversion unit is one decision, not one complete trajectory:

```text
everything the student would see before decision i  ->  DeepSeek decision i
```

Use this single-pass procedure:

1. Load the one frozen, fully eligible DeepSeek trace for the task.
2. Normalize it into semantic chat messages: system/policy, user messages,
   assistant messages/tool calls, and tool results. Preserve tool names,
   arguments, call/result linkage and message order. Do not preserve
   DeepSeek-specific rendered text or DeepSeek token IDs.
3. Enumerate the assistant decisions and select up to six with the deterministic
   coverage rule in 6.2.
4. For each selected decision `i`, copy every message strictly before `i` as
   visible context and append DeepSeek's decision `i` as the final message.
5. Emit an explicit message-level supervision mask: every prefix message is
   `false`; only the final assistant message is `true`.
6. Render and tokenize only with the pinned Qwen3-4B tokenizer, chat template,
   tool schema and thinking-mode settings used at serving time. Build labels by
   comparing the rendered generation prefix with the rendered full row; set
   every prefix label to `-100` and retain labels only for the final target.
7. Reject—never truncate—a row if rendering is unstable, the target is empty or
   unparsable, a tool result is orphaned, the target is not the final assistant
   message, or the full row exceeds the context limit.

Canonical semantic JSONL row:

```json
{
  "row_id": "retail:42:<trace_sha>:action_5",
  "task_id": "42",
  "trace_sha256": "<sha256>",
  "action_index": 5,
  "position_fraction": 0.50,
  "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
  "messages": [
    {"role": "system", "content": "<Tau2 policy and tool-use instruction>"},
    {"role": "user", "content": "<request>"},
    {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{...}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "<observation>"},
    {"role": "assistant", "content": "<selected DeepSeek action>"}
  ],
  "supervise": [false, false, false, false, true]
}
```

The example is schematic; the exporter must use Tau2's actual normalized
message/tool objects. Store semantic messages in the durable corpus and treat
Qwen tokenized `input_ids`, `labels` and `attention_mask` as a reproducible
derived artifact with a tokenizer/template fingerprint.

Do **not** feed this JSONL to a trainer that implements only
`assistant_only_loss=True`: such a role-derived mask also supervises earlier
assistant actions in the prefix. Use the repository's two-render
`tokenize_messages(...)` path, or an equivalent collator proven to produce:

```text
labels[prefix tokens]       = -100
labels[final target tokens] = input_ids[the same positions]
```

Preflight the complete corpus on CPU and fail unless all of these hold:

- exactly 30 W30 task IDs and six rows per selected task (180 total under the
  audited trace set);
- every row ends in exactly one supervised assistant message;
- all earlier assistant/tool/history tokens have label `-100`;
- the generation-prefix tokens are an exact prefix of the full-row tokens;
- zero truncation, zero empty targets, zero orphan tool results and zero parser
  failures;
- task IDs are a subset of W30 and disjoint from C30/S16/F38; and
- rebuilding twice produces identical row IDs, hashes and tokenization
  fingerprints.

Before GPU training, emit a census containing:

- total and eligible tasks/trajectories;
- assistant decisions per task;
- supervised assistant tokens per task;
- rendered context/target length quantiles and overflows;
- tool-action versus conversational-action counts;
- mutation, clarification, search, confirmation, refusal/no-action, and
  recovery counts; and
- task-family mass.

The training objective should weight tasks equally:

```text
L_SFT = mean over tasks(mean NLL over that task's assistant target tokens)
```

This prevents one unusually long trajectory from dominating simply because it
contains more tokens. If the trainer cannot implement task-normalized loss,
use a task-balanced sampler and report the realized per-task supervised-token
mass.

### 6.3 Masking and rendering gates

- Use the exact Qwen3-4B post-trained tokenizer and chat template used at
  serving time.
- Supervise only the final assistant action; earlier assistant actions remain
  visible context with label `-100`.
- Verify the explicit labels against decoded target spans on every row; do not
  rely on a role-derived `assistant_only_loss` mask.
- Pin thinking mode, tool schema, tool parser, special tokens, maximum context,
  base revision, and precision across train and serve.
- Fail closed on zero target tokens, target-role leakage, orphan tool results,
  template drift, or any truncation.

### 6.4 How much SFT

Do not predeclare a trajectory number as sufficient. Use nested, deterministic
subsets of the eligible W30 warm-up tasks to measure a cheap scaling curve:

```text
12 diverse tasks -> 24 diverse tasks -> all eligible W30 tasks
```

The subsets are nested and family-balanced. This is an engineering/model-size
probe, not the final reported SFT arm. Prefer independent fresh LoRAs for each
subset if time permits; progressive continuation may be used for the two-day
screen but must be labeled because it confounds task count with additional
updates.

For the actual `A_warm`, train on all eligible **W30** trajectories. Run one
epoch, evaluate on the Tau2 selection panel, and continue to epoch two only if
policy-compliant live performance is still improving. Allow at most three epochs
in the two-day gate. Select by live Tau2 metrics, never by training loss alone.

Report `A_warm` in live evaluation on the W30 training tasks alongside the
panel. This is training-task acquisition, not held-out evaluation. A large
W30-versus-S16 gap is consistent with memorization, task-family shift or the
known difficulty-composition difference; it helps diagnosis but does not by
itself uniquely distinguish those causes.

At the 180-row ceiling, three epochs at effective batch 8 is about 68 optimizer
updates, with diagnostic checkpoints near updates 20 and 40 and at the end.
Because the realized row count may fall below 180 when a trace lacks a category
(see 6.2), recompute the update count and checkpoint positions from the
selector's actual output before the run. 68 is a planning figure, not the
budget, and it also changes if the policy/rendering audit reduces the eligible
set.

Freeze the optimizer recipe only after the census and a three-step memory/loss
probe. The V2 data design does not inherit the old Harbor learning rate,
sequence length, epoch count, or quantization choice.

## 7. Experimental arms

```text
A0       frozen Qwen3-4B
  |
  +-- SFT on W30 -------------------- A_warm
                                         |
                                         +-- continued SFT on C30 -- A_sft_new
                                         |
                                         +-- replay OPD on C30 ----- A_reopd

B0       frozen Qwen3-8B reference; no training in the first gate
```

`A_reopd_same` (replay OPD over the W30 prefixes) is a later diagnostic for
separating objective from task novelty. It is deliberately **not** an arm of the
two-day run; see 1.1.

`A_sft` in earlier drafts is renamed `A_sft_new`: it trains on C30 adaptation
tasks, not on more passes over the warm-up corpus.

Evaluate `A0`, `A_warm`, `A_sft_new`, `A_reopd`, and `B0` on the same frozen
Tau2 selection tasks, trials, seeds, simulator, prompt, and generation settings.

The method claim is `A_reopd` versus `A_sft_new`. `A_warm` is the shared floor
both branches improved on, never the comparator for the method claim — reporting
`A_reopd` against `A_warm` would compare 32 extra updates against none.

### 7.1 Measure `A_warm` before adaptation begins

Evaluate both `A0` and `A_warm` on **C30 and S16** before any continuation
training runs:

- `A0 -> A_warm` on S16 is the held-out benefit of warm-up SFT;
- `A_warm` on C30 is the pre-adaptation baseline for the adaptation tasks
  themselves, and is the reference both branches are measured against.

C30 stops being held out the moment continuation training starts, so this
measurement exists only in this window. Missing it cannot be recovered later.

### 7.1a Sampling-entropy gate before branching

`A_warm` is a launchpad, not a deliverable. Section 6.1 requires it to place
"enough probability on plausible actions for DeepSeek scoring to provide a useful
replay-OPD gradient." A checkpoint can satisfy every live-performance gate and
still fail this one: if warm SFT has collapsed the action distribution onto a
single memorized continuation per state, the student samples near-deterministic
actions, DeepSeek scores nearly the same action every time, and the replay
gradient is flat. Replay OPD would then fail for a reason that has nothing to do
with the objective under test.

Before branching, sample `k` actions at a fixed subset of C30 prefixes with the
serving-time generation settings and measure:

- distinct-action rate across the `k` samples at each prefix;
- mean token-level entropy over the sampled action spans;
- the same two quantities on `A0`, as the un-collapsed reference.

`A_warm` collapsing to near-zero distinct actions relative to `A0` is a stop
condition for the replay branch, not a result about replay OPD. The remedy is an
earlier `A_warm` checkpoint, not more replay updates.

Record the measurement for whichever checkpoint is promoted, since the
checkpoint chosen on live panel performance is not necessarily the one with
usable sampling diversity, and the two can be traded off explicitly.

Capacity note: rank-32 all-linear adapters over at most 180 rows carry
substantial capacity relative to the data. That capacity is what lets the
protocol burn in, which is wanted. What must not happen is distribution collapse
on the training tasks — the same phenomenon at a different depth, and the reason
this gate exists alongside the live-evaluation early stopping in 6.4.

### 7.2 Freeze the C30 prefix manifest

Using the selected DeepSeek trace from every C30 task, select up to six decision
positions with the same deterministic rules as section 6.2, including the
fallback rule. As with W30 the count is bounded by 30 x 6 = 180, and the realized
number is whatever the selector produces. Both branches consume the same realized
pool, so an under-filled C30 shortens both arms identically and does not break
the match.

Freeze and hash: task IDs, trace hashes, prefix IDs, the exact semantic
histories, both the Qwen and DeepSeek renderings, position types, and the
sampling-order seed.

Both continuation branches consume this identical manifest. Any divergence
between the branches' prefix streams invalidates the comparison, so the manifest
is built once and read twice, never regenerated per branch.


## 8. Continued SFT versus replay OPD

> **Superseded (see 1.1).** This section previously opened: "Separate traces are
> **not crucial**. For the primary controlled comparison, both branches reuse the
> same eligible retail-train trajectories and the same frozen set of assistant
> decision prefixes," and framed the SFT control as "more SFT on the same expert
> corpus." Both branches still share prefixes, but those prefixes now come from
> the reserved C30 adaptation tasks rather than from the warm-up corpus.

Separate *traces per task* remain unnecessary: one DeepSeek trajectory per task
is sufficient. What is reserved is a distinct set of **tasks**. Both branches
start from the identical `A_warm` artifact and consume the identical frozen C30
prefix pool, so the comparison isolates the objective at matched states.

### Continued-SFT control (`A_sft_new`)

At a selected stored C30 prefix, train on the DeepSeek action that actually
follows that prefix:

```text
stored DeepSeek prefix (C30) -> recorded DeepSeek action -> SFT loss
```

This is ordinary offline imitation on new retail tasks. It is the correct
control for whether replay OPD adds value beyond spending the same optimization
budget on the expert demonstrations available for those same tasks.

Note the asymmetry recorded in 1.1: this branch sees the expert's action at
every adaptation prefix, while replay OPD sees a teacher-derived score on its
own sampled action. The former is a more direct imitation signal, but the plan
does not assume which signal produces the better policy.

### Replay OPD

At the identical stored C30 prefix:

```text
stored DeepSeek prefix (C30)
    -> current Qwen3-4B samples one exact action
    -> DeepSeek scores that exact action
    -> cross-tokenizer replay-OPD update
```

The sampled action is on-policy; the history is an offline teacher replay. Do
not describe this as a live on-policy Tau2 rollout.

### Pilot budget

After `A_warm`, run a small matched continuation:

- 32 optimizer updates per branch;
- initially 16 distinct task-balanced prefix states per update;
- one student action per replay-OPD state;
- 512 state/action exposures per branch if no batch reduction is required;
- the same eligible **C30** task IDs and the same frozen prefix pool;
- shuffling without replacement, starting a new epoch when the pool is
  exhausted;
- same LoRA capacity, context limit, effective batch size, and
  checkpoint-selection rule.

> **Superseded (see 1.1).** The list previously required "matched supervised
> student-action tokens within 5%." That is not achievable by construction:
> sampled student actions and recorded teacher actions differ in length, so
> forcing the match would mean truncating or padding one arm's actions, which
> corrupts the quantity being measured.

The preregistered match is over the countable, controllable quantities above.
Supervised token counts, generated token counts, GPU time, and teacher-inference
cost are **reported, not forced**, and any material imbalance is stated with the
result rather than engineered away.

Both branches reload the identical `A_warm` checkpoint independently. What
"independently" means is explicit:

```text
same      A_warm base weights and adapter weights
fresh     optimizer state        (no momentum from warm SFT, none from the other branch)
fresh     scheduler state
same      optimizer and scheduler configuration
distinct  branch RNG seeds, recorded before the run
```

Neither continuation arm may inherit optimizer momentum from warm SFT or from
its sibling. Carrying warm-SFT momentum into one branch and not the other would
break the match silently, with nothing in either training log showing it.

Replay checkpoints at updates 8, 16 and 32 are for diagnosis only and are never
selected on training loss.

Batch shape is provisional until the prefix/action census and memory probe. If
the token budget cannot hold 16 states, reduce both arms symmetrically and
record the new frozen budget before the run.

For the two-day gate, sample task first and position second so every task has
support and a few long traces cannot dominate. Record position histograms.
The ReOPD paper's step-decaying prefix distribution is a later preregistered
ablation; do not silently call a coverage-balanced sampler paper-identical.

## 9. Two-day 4B decision run

The 48-hour clock starts only after the Tau2 exporter, prefix builder, trainer,
and evaluator pass their CPU/no-training gates.

### Hours 0-4: calibration

1. Freeze task/family IDs and trace hashes.
2. Build the trace/action/token census.
3. Run eight live 4B Tau2 episodes.
4. Run one 16- or 32-action replay-OPD canary update.
5. Measure actual sampling, DeepSeek scoring, training, and serving time.
6. Verify sampled bytes, token IDs, teacher scores, finite loss/gradients, and
   adapter reload.
7. Record `A0` sampling diversity at the C30 prefix subset, as the reference for
   the 7.1a entropy gate.

### Hours 4-16: SFT viability

1. Evaluate `A0` and frozen 8B `B0` on the same Tau2 selection panel.
2. Run the nested 12/24/all-30-W30 SFT scaling probe as time permits.
3. Train the all-30-W30 `A_warm` with one-epoch checkpoints and live
   early-stopping.
4. Stop before replay OPD if SFT cannot move 4B without breaking policy/tool
   validity.

### Hours 16-40: matched continuation

Branch from the identical `A_warm` artifact and run 32 continued-SFT updates and
32 replay-OPD updates, both on the reserved C30 adaptation tasks. Save replay
checkpoints at updates 8, 16, and 32 for diagnosis; do not select them on
training loss.

### Hours 40-48: live evaluation and decision

Evaluate all arms on the same Tau2 selection panel, persist full trajectories,
compute task-level paired results, audit policy violations, and apply the
pre-registered stop/continue rule.

## 10. Continue/stop rule for 4B

Continue the full experiment on 4B only if all of these hold:

1. `A_warm` improves policy-compliant live success over `A0` across multiple
   task families, not only simple returns.
2. `A_reopd` is above the non-floor threshold of 30% policy-compliant success
   on the selection panel.
3. `A_reopd` beats `A_warm` by at least 5 percentage points and is not more
   than 5 points behind the budget-matched `A_sft_new` control. Parity with the
   control passes this gate; the gate asks whether replay OPD is competitive,
   not whether it dominates.
4. `A_reopd` reaches at least 80% of frozen 8B `B0` policy-compliant success on
   the identical panel.
5. Structured-action validity is at least 99%, nearly all episodes are
   gradeable, and no material authentication/confirmation/conditional-mutation
   regression appears.
6. Replay scoring/alignment/loss/gradients are finite and nonzero, and the
   saved adapter reloads exactly.
7. `A_warm` passes the 7.1a sampling-entropy gate — its action distribution at
   C30 prefixes has not collapsed relative to `A0`.

These are engineering continuation thresholds, not a claim of statistical
proof. Report task-level bootstrap intervals and the paired per-task changes.

Interpret failure before switching models:

| Observation | Likely conclusion |
|---|---|
| SFT does not move 4B while frozen 8B is stronger | 4B capacity/support problem; switch to 8B |
| SFT works but scoring/alignment/loss is broken | replay implementation problem; changing size will not fix it |
| `A_warm` passes the panel but collapses on 7.1a | over-trained warm start; promote an earlier checkpoint before judging replay |
| SFT works and continued SFT improves, but replay OPD loses by more than 5 points | replay objective/recipe problem |
| `A_reopd` and `A_sft_new` both gain, replay within 5 points | replay OPD is competitive at matched budget |
| replay OPD improves 4B but remains far below 8B | explicit cost-versus-capability decision |
| all gates pass | retain 4B for the full run; 8B becomes replication |

## 11. Final retail evaluation

After the model-size gate, freeze the selected recipe and train the full
retail arms.

Evaluation matrix:

| Arm | C30 | S16 | F38 |
|---|---|---|---|
| `A0` | baseline | baseline | final only |
| `A_warm` | pre-adaptation baseline (7.1) | pre-adaptation baseline (7.1) | final only |
| `A_sft_new` | adaptation-task acquisition | held-out screening | final |
| `A_reopd` | adaptation-task acquisition | held-out screening | final |

C30 measures acquisition on the tasks that were trained on; S16 is directional
screening (see 4.4); F38 is the single frozen comparison, run once after every
recipe is locked, with four fixed trials per task.

The primary comparison is `A_sft_new` versus `A_reopd`, read with the
information asymmetry from section 8 in view.

Primary metric:

> Tau2 official success with all applicable hard policy gates satisfied.

Report official Tau2 reward separately. Secondary metrics include DB success,
communication success, authentication, confirmation, unresolved-condition
mutation, tool validity/errors, unsupported claims, unnecessary transfer,
turns, tool calls, latency, GPU time, DeepSeek tokens, and cost. Bootstrap by
task; repeated trials of one task are not independent tasks.

## 12. Continual-learning extension

Only after retail establishes a viable method/model:

```text
T0: shared retail SFT checkpoint A_warm
    evaluate retail / telecom / airline Tau2 panels

T1: retail continuation
    continued-SFT branch versus replay-OPD branch
    evaluate retail / telecom / airline

T2: adapt both branches on Tau2 telecom training tasks
    evaluate retail / telecom / airline
```

Retail `T1 -> T2` is forgetting; telecom `T1 -> T2` is new-domain
acquisition; airline is untouched Tau2-domain drift. No outside chat, coding,
instruction-following, or safety dataset enters any stage.

The exact telecom adaptation recipe, including whether both branches receive a
common telecom SFT initialization, must be decided and frozen after the retail
result. It is not smuggled into the two-day 4B decision.

## 13. Immediate implementation order

1. Inventory all DeepSeek Tau2 traces, run structural, policy and exact-Qwen-
   rendering eligibility, deterministically choose one trace per eligible task,
   and hash every artifact.
2. Reserve every inspected/design-influencing task for S16, then build/freeze
   normalized family groups, the 30/30/16/38 manifest, and the
   family-balanced W30/C30 halving of retail-train. Assert the partitions sum to
   114 and that near-duplicate templates do not straddle W30/C30.
3. Quarantine selection/test traces at the code boundary.
4. Persist the eligibility and policy-audit report used to construct the split.
5. Implement Tau2 conversational SFT export with verified assistant masks,
   task weighting, and no truncation.
6. Implement Tau2 replay-prefix export using the exact same eligible train
   traces and rendered histories.
7. Add task-ID leakage assertions to SFT, replay sampling, and teacher scoring,
   including the `W30 intersect C30 == empty` and `A_warm subset of W30`
   invariants from section 4.
8. Run CPU token/context/data census and the no-training replay proofs.
9. Freeze the two-day pilot recipe from measured counts.
10. Ask separately for each GPU/live-evaluation launch; this plan is not run
    authorization.

## 13.1 Final structure

```text
114 retail tasks
|
+-- 60 trace-eligible training tasks   (from 73 passing candidates)
|   |
|   +-- W30 warm-up
|   |     `-- SFT -> A_warm
|   |
|   `-- C30 adaptation
|         +-- continued SFT -> A_sft_new
|         `-- ReOPD         -> A_reopd
|
+-- S16 selection panel      (directional screening; diagnostics 57/73/75/93 live here)
|
`-- F38 frozen final test    (one evaluation, after all recipes locked)
```

## 14. Falsification

The method/model claim is not supported if gains require selection/test trace
access, disappear under policy-compliant scoring, occur only on already-seen
train tasks, are explained by invalid or truncated actions, or if replay OPD
cannot beat the shared SFT warmup and is clearly worse than continued SFT at a
matched budget. A failed 4B gate triggers the predeclared 8B fallback; it does
not justify repeatedly tuning 4B on the selection panel.
