# Prompt-seed probe — running log

**Question:** can repaired ck63 be made to emit Harbor-parseable actions from
turn 1 by putting one native-JSON demonstration in its context, with no
retraining and no request proxy?

Companion to `docs/SFT-REPAIR-PLAN.md`. This is a log of what was run and what it
meant, written as it happened, including the wrong turns.

---

## Background — what Phase 7 established

ck63 emits native JSON at turn 6+ and v1's `<tool_call>` envelope at turn 1. The
instruction block is byte-identical in all 24 prefixes, so instructions cannot
explain the difference. An ablation confirmed causation: rewrite the one visible
assistant turn into v1's envelope and the output flips back.

Conclusion carried into this work: **the model copies its output protocol from
the visible assistant history.** The dose-response arm reported the transition at
0 → 1 prior assistant turns.

That last number is what this log ends up correcting.

---

## Run 1 — `20260818T072950Z` — void, my bug

Endpoint booted, rollout never ran.

```
infra_failures:        {"agronholm__anyio-1121": 1}
no_gradeable_rollouts: ["agronholm__anyio-1121"]
```

**Cause.** The wrapper requested `hosted_vllm/Qwen3-14B-ck63`.
`serve_student.py` prefixes the adapter with the base model, so the registered id
is `Qwen3-14B-Qwen3-14B-ck63`. Every request 404'd.

**Why the smoke check missed it.** `grep -q "Qwen3-14B-ck63"` matches the
double-prefixed id as a substring — the check passed on exactly the mismatch it
existed to catch. Replaced with an exact id match against the parsed
`/v1/models` list.

**Teardown also failed**, and this is the expensive part: the EXIT trap ran
`modal`, which is not on PATH under `sudo -iu ubuntu`, and `modal app stop`
aborts without `--yes` when there is no tty. Neither surfaces as a non-zero exit.
The trap printed `command not found` and an L40S billed for ~16 minutes after the
rollout had already failed. Trap now uses the absolute binary, passes `--yes`,
kills the serve pid, and **re-lists to verify** rather than assuming.

**Cost:** ~16 GPU-minutes, zero signal.

---

## Run 2 — `20260818T074059Z` — real result, negative

Endpoint fine, adapter registered, rollout ran. Eight turns, all identical:

```
turn 1-8  seed_in_context=true  native_json=false  legacy_envelope=true
          parser_error="Missing required fields: analysis, plan, commands"
          n_commands=0  has_keystrokes=false
```

Turn 1 raw output, with the native-JSON demo in context:

```
<think>

</think>

Analysis: We are in the workspace. The task is to fix a pytest issue ...
Plan: List the repository contents ...
<tool_call>
{"name": "bash_command", "arguments": {"keystrokes": "ls -la\n", "duration": 0.1}}
</tool_call>
```

### The diagnosis, from the logged context

The probe stored turn 1's messages verbatim, which is what made this readable:

```
[0] user       108   calibration
[1] assistant  317   native-JSON demo
[2] user     5,277   FULL instructions + task + terminal   <- LAST
[3] assistant  518   -> legacy envelope
```

Against Phase 7's passing turn-2 prefix (`prior_assistant_turns: 1`,
`native_json: true`):

```
[0] user       instructions + task + terminal
[1] assistant  REAL native-JSON action
[2] user       terminal observation                        <- LAST, short
```

**Same number of prior assistant turns. Opposite result.** So "one visible
native-JSON assistant turn" was never the operative variable — Phase 7's
dose-response conflated *count* with *position*.

What actually matters is what sits immediately before the generation point. In
Phase 7 that is a short observation with a native action just above it. In run 2
it is the 5,277-char instruction block — byte-identical to the unseeded turn-1
condition Phase 7 measured at 0/8 native. The demo was 5k characters upstream and
got swamped.

**This was my design error, and it was deliberate.** I placed the pair before the
rendered prompt so the fake terminal state would not contradict the live one. That
reasoning optimised conversational coherence and broke the mechanism.

### The spiral, visible in the data

Turn 2 digest: `[user 108, asst 317, user 5277, asst 518, user 246, asst 517]`.
The 246-char user message is Harbor's parse-error feedback and the 518-char
assistant turn above it is the model's own `<tool_call>`. From turn 2 the nearest
assistant turn is its own legacy output, so it copies that. Predicted before the
run; it just started at turn 1 rather than later.

### Second, independent finding

The completion opens with `<think>\n\n</think>`. Phase 7 pinned
`enable_thinking=False` via `chat_template_kwargs`; Harbor's terminus path does
not. So run 2 was not the same serving configuration Phase 7 measured.

Checked offline against Harbor's real parser:

```
bare json            ACCEPT  cmds=1  warn=''
empty think + json   ACCEPT  cmds=1  warn='Extra text detected before JSON object'
think with content   ACCEPT  cmds=1  warn='Extra text detected before JSON object'
legacy tool_call     REJECT  cmds=0
```

A leading think block is tolerated — warning only, commands extracted. So
thinking can stay enabled and a rollout genuinely proceeds. Strict `native_json`
is still recorded but no longer gates.

**Cost:** ~7 GPU-minutes. Torn down immediately on the finding; verified stopped.

---

## Run 3 — `20260818T081153Z` — split geometry — IN PROGRESS

Harbor's rendered prompt is split exactly once at `Current terminal state:`:

```
user       instructions + task + "wait for the terminal state"
assistant  calibration JSON, commands: []          <- synthetic
user       the genuine terminal state + "begin"    <- model answers this
```

Reproduces Phase 7's *action → short observation → generate* shape **without
fabricating an observation**: the terminal state is Harbor's own captured text,
relocated. Exactly three things are synthetic and preflight prints the list — the
assistant turn and the two added sentences.

`commands: []` is now the correct reply to the wait instruction rather than an
unmotivated no-op, and "begin" supersedes it, so a verbatim copy is visibly wrong
and fails the non-empty-keystroke gate.

**Changed relative to run 2:** geometry *and* temperature (now pinned to 0). This
run therefore tests "new geometry + deterministic sampling", not position in
isolation. Recorded here so the result is not over-claimed.

**Turn 1 now aborts the rollout on gate failure** rather than feeding the parse
error back — run 2 showed that loop costs GPU and teaches nothing.

Gate: `harbor_accepts` · no `<tool_call>` · ≥1 non-empty keystroke · not a seed
echo. `<think>` allowed.

### Result — FAILED at turn 1, and the failure is informative

The split applied exactly as designed. Logged geometry:

```
[0] user      5,319  instructions + task + "wait for the terminal state"
[1] assistant   376  calibration JSON, commands: []
[2] user        136  Current terminal state: root@ab3e1d50af87:/workspace#  + "begin"
[3] assistant   328  -> the model's answer
```

That is Phase 7's shape: short real observation immediately before generation,
native-JSON assistant turn directly above it. The hypothesis got a fair test.

Turn 1 output:

```
<think>

</think>

Analysis: We are in /workspace, and need to inspect the repository contents.
Plan: List the repository contents to understand what we're working with.
{"name": "bash_command", "arguments": {"keystrokes": "ls -la\n", "duration": 0.1}}
```

`harbor_accepts=false`, `Missing required fields: analysis, plan, commands`,
zero commands. Gate failed, rollout aborted at turn 1 as designed, endpoint torn
down and **verified stopped**.

### The output did move — partially

| | run 2 (pair before prompt) | run 3 (split geometry) |
|---|---|---|
| `<tool_call>` tags | present | **gone** |
| prose Analysis/Plan lines | present | present |
| JSON schema | `name` / `arguments` | `name` / `arguments` |
| harbor accepts | no | no |

So adjacency removed the **tags** and left the **schema**. The model is still
emitting v1's OpenAI tool-call object rather than Terminus's flat
`{analysis, plan, commands}`. Prompt position moved the surface form and did not
reach the schema.

That is the honest read of the negative: not "no effect", but "an effect on the
wrong layer". The deficit that matters is the schema, and it did not budge.

### A measurement bug found in the process

`has_legacy_envelope()` returned True for `<think>` OR `<tool_call>`, so this
completion was flagged `legacy_envelope=true` because of a think block we had
explicitly decided to allow. The verdict is unaffected (`harbor_accepts=false`
fails the gate on its own), but the flag would have reported "still emitting the
tool_call envelope" for a completion that had dropped the tags.

Split into three flags: `legacy_envelope` (tags only), `think_block` (allowed,
recorded), `v1_tool_schema` (the actual remaining defect). Lesson: a flag that
ORs two failure modes cannot show you when one of them moves.

**Cost:** ~11 GPU-minutes. Teardown verified by the new trap.

---

## Verdict on prompt scaffolding

Three runs, one void. The strongest remaining prompt hypothesis — Phase 7's own
working geometry, reproduced without fabricating anything — does not get turn 1
to a parseable action. Per the decision rule fixed before the run: **stop prompt
experiments.**

Corrected understanding of Phase 7's dose-response: it reported the transition at
0 -> 1 prior assistant turns, but run 2 and run 3 both had exactly one and both
failed. The variable it actually measured was neither count nor position alone —
those are now controlled — but something about the *content* of the visible
assistant turn. Phase 7's turn-2 prefixes carried a real prior action with real
commands, produced by the teacher on that task. A synthetic acknowledgement with
`commands: []` does not substitute for it, even in the identical slot.

That is a sharper statement of the Phase 7 finding than the plan currently
carries, and it makes the case for Phase 6b stronger rather than weaker: what the
model lacks cannot be supplied by context at turn 1, because at turn 1 no real
prior action exists to supply.

---

## Decision rule, fixed before the run

- **Turn 1 passes** → continue the same rollout autonomously; the scaffold is
  deployable and retraining is deferred.
- **Turn 1 fails** → stop prompt experiments. Go to a short targeted continuation
  from ck63 (5–10 steps, cold-start + anti-copy, checkpoints every 1–2 steps,
  frozen turn-1 eval after each). That is a *movement* probe, not a fix: 5–10
  steps at bs1×accum8 is 40–80 of 165 rows, under half an epoch. If turn 1 is
  unchanged after 10 steps, the v1 prior is too entrenched to correct on top of
  ck63 and the honest next step is a fresh adapter on corrected data.

## Rejected, and why

| option | why not |
|---|---|
| request proxy rewriting completions | the shim; contaminates OPD with permanent shim-dependence |
| fabricated terminal observation | invents evidence the model then reasons from |
| guided/structured decoding | not a shim (alters no text) but masks whether the protocol was learned — which is the exact signal OPD needs |
| another 3-hour broad SFT run | the finding is about supervision composition, not step count |
