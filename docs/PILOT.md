Pilot — 30B → 8B, and the order to run it in
============================================

**Date:** 2026-07-29 · **Tree:** `main` @ `d370605`

`PLAN.md` says *what* the capability-routing design is. This says *what runs
first and on which GPU.* Nothing here changes the design; it picks the smallest
configuration that can execute it and orders the steps so that each one's
result is available to the one after it.

The repository holds ~16.8k LOC and 39 test files, and **none of it has met a
GPU or a real corpus.** The blocker has not been code for some time.

**GPU choices below are written in Modal's vocabulary because that is what was
built first — not because anything depends on it.** For the same pilot on a GPU
you manage (EC2 or otherwise), see `docs/AWS.md`: `--api-base` attaches every
stage to a vLLM server you started, and Modal drops out of the run entirely.


The framework decision
----------------------

**OPD: written here. RL (GRPO): verl, later.** Not one trainer for both, which
is what an earlier draft of this argued.

The argument for verl-for-everything was that running the two branches in
different frameworks confounds B1 vs B2 — any difference could be the trainer
rather than the routing. That is true, and it is satisfied just as well by
putting both branches on the training loop already in `train.py`: same masking,
same `dataset.py`, same optimizer.

What OPD actually requires:

1. the student generates a rollout, on policy
2. the teacher scores **those exact token ids** → per-token logprobs
3. loss = reverse KL on the sampled tokens, masked to agent turns
4. backprop into the LoRA

Step 2 is one HTTP call to a vLLM server we already run — `POST /v1/completions`
with the token sequence, `prompt_logprobs`, `max_tokens: 0`. Steps 3–4 are a
loss on top of the HF + peft loop in `train.py`, and `opd.py` already carries a
local reverse-KL helper. That is a few hundred lines, not a framework.

verl's teacher resource pool exists to co-locate teacher inference with training
across many nodes at high throughput. We have one teacher and one student.

**GRPO is the opposite case.** Multi-turn agentic rollouts, advantage
computation, and vLLM weight-syncing are genuinely hard to hand-roll, and verl
has them. So verl arrives when routing actually populates the RL bucket — which
is a measurement we have not taken.

| Branch | Trainer | When |
|---|---|---|
| OPD | here, on `train.py` + peft | now |
| RL (GRPO) | verl | once `route` puts real tasks in the RL bucket |

Also considered and rejected, each for the same underlying reason — they wrap a
harness or a model we do not control, and we control ours:

- **ART** — no OPD path at all, no teacher scoring. RULER replaces a verifier we
  already have.
- **Polar** (NVIDIA) — proxies an agent harness to recover token-faithful
  trajectories. Solves a problem we do not have: the student is served from our
  own vLLM, so `return_token_ids` gets us the same thing directly.
- **SkyRL-Agent** — tool-centric; it wants to *be* the agent loop. Adopting it
  means replacing our scaffold, which is backwards.


How harbor and the trainer relate
---------------------------------

They never talk to each other. We are the bridge, and the bridge is
`dataset.py`.

```
harbor      containers, task, agent loop, TEST-SUITE REWARD   ← environment + grader
serve.py    Modal vLLM serving the student (+ LoRA)           ← inference
dataset.py  trajectory + reward → training example            ← the bridge
trainer     HF + peft (SFT / OPD)   |   verl (GRPO, later)    ← learning
```

harbor produces graded trajectories. The trainer consumes them and emits an
adapter. `serve.py` serves the adapter back. harbor runs against it again.

This layering is the reason a generic gym adapter is the wrong move. harbor's
separate-container verifier, the `.git` scrub and the default-deny network
policy are not incidental — they are the product. Handing rollouts to a trainer
that owns the environment would discard them.


The pair
--------

| | model | why |
|---|---|---|
| teacher | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | MoE: 30B total, **~3.3B active**. OPD queries the teacher at every step, so teacher *latency*, not teacher size, is the loop's bottleneck — and an MoE answers like a small model. |
| student | `Qwen/Qwen3-8B` | same family, so the tokenizer can match. verl's own documented OPD example is a 32B teacher with an 8B student, so this sits near a configuration someone else has run. |

`PLAN.md`'s stated pair (80B → 8B) stays as `SCALE_*` in `tokenizer_check.py`.
An 80B dense teacher needs several GPUs and costs more per query; it is the
scale-up after the loop closes once, not the thing to debug the loop on.

**Same family is a hypothesis, not a fact.** Step 0 verifies it, and a mismatch
means the teacher changes and every later config changes with it.


GPU requirements
----------------

**Teacher — inference only, must expose `prompt_logprobs`**

| precision | weights | fits |
|---|---|---|
| bf16 | ~61 GB | too tight on one 80 GB card once KV cache is added at long context |
| **fp8** | **~31 GB** | **1×H100-80G, comfortable** |
| bf16, TP=2 | ~61 GB split | 2×A100-80G (A100 has no native fp8) |

**Student — LoRA training.** 16 GB of bf16 weights, plus LoRA grads and
optimizer state (small), plus activations. With gradient checkpointing at 8–16K
context: **1×A100-80G or 1×H100-80G**. A full fine-tune would need ~120 GB+;
LoRA is what makes this a one-GPU job, and it is *also* the non-regression
mechanism, since base weights are untouched.

**Student — serving for rollouts.** Another ~16 GB. Either a third small GPU or
colocated with training via vLLM sleep/wake.

| Phase | GPUs | Running |
|---|---|---|
| 0 — measurement | 1× L40S **+ 1×H100** | student *and* teacher serving; the frontier arm is an API call |
| 0.5 — token capture | none | a serving flag and plumbing |
| 1 — OPD smoke | 1×H100 + 1×A100 | teacher scoring + student LoRA, one task |
| 2 — OPD real | same, longer | + student rollout serving |
| 3 — RL branch | 4–8×H100 (verl) | only if routing populates the RL bucket |

Phase 0 needs the **teacher** GPU too, not just the student: `route` consumes a
teacher `pass@k` report, and the teacher is self-hosted. An earlier draft of
this table said "student serving only", which was wrong.

Modal, per-second billing, no idle charge (checked 2026-07-29):

| GPU | $/hr | role |
|---|---|---|
| H100 80GB | $3.95 | teacher, fp8 |
| A100 80GB | $2.50 | student LoRA training |
| L40S 48GB | $1.95 | student serving |
| A100 40GB | $2.10 | — |
| A10 24GB | $1.10 | too tight for 8B + KV cache |

Volumes are $0.09/GiB/month with 1 TiB free, so the weights cache and the
adapter volume are effectively free. Phase 2 runs H100 + A100 ≈ **$6.45/hr**.

**Where Modal stops being the right answer.** Break-even against dedicated
hourly rental is roughly 30% GPU utilisation. Phase 0 is far below it — the GPU
idles while Docker containers run test suites, which is exactly what
scale-to-zero is for. Sustained training is above it: RunPod on-demand H100 is
around half Modal's rate. So Phases 0–1 stay here, and Phase 2 is worth
repricing if training hours climb past ~50. Keep *serving* on Modal regardless,
so the `serve.py` → harbor path stays intact and only the training job moves.


The order to run it in
----------------------

### Phase 0 — measurement. No trainer, no teacher, no framework decision.

```
mine → replay → diagnose → passk → route
```

Produces the diagnosis, the environment and the routing decision — three useful
outputs with no training at all. All of it is already coded.

Phase 1 onward reads the routing decision this phase produces, so this phase
runs first. If the `pass@k` curves do not separate into regimes, there is no
routing decision to act on, and that is itself a finding about the task
distribution worth reporting. One modest GPU, no verl.

### Phase 0.5 — token capture. Small, and everything after it is wrong without it.

`dataset.py` currently re-tokenizes text. For SFT that is correct and standard.
For OPD and GRPO it is not: both compare against the probability of the tokens
*actually sampled*, and a re-tokenized sequence is close but not identical. It
does not crash. It does not warn. The number simply fails to move.

vLLM added `"return_token_ids": true` on `/v1/chat/completions` for exactly this
— it returns `prompt_token_ids` and `token_ids` alongside the text, plus
`logprobs`. We own the server, so this is a flag, not a framework:

- enable it in `serve.py` (`litellm_generate` / `litellm_generate_captured`,
  `served_to_harbor_kwargs(capture_tokens=True)`)
- pass it through litellm's `hosted_vllm` provider (`extra_body`)
- store the per-request token ids and logprobs, correlated to the harbor job
  dir (`token_capture.py`, `collect_rollouts(..., capture_tokens=True)`, or
  `vektori-trace capture-proxy --upstream …`)
- `dataset.py` consumes ids instead of re-tokenizing
  (`tokenize_from_ids` / `tokenize_from_captures`)

**Status (coded).** The plumbing is shipped. A real sweep still needs a vLLM
≥0.10.2 endpoint; without `return_token_ids` the proxy records nothing and
`tokenize_rollouts_for_opd` refuses rather than silently re-tokenizing.

### Phase 1 — OPD smoke, one task.

Serve the teacher with `prompt_logprobs`; take one student rollout's token ids;
score them against the teacher; reverse-KL on the sampled tokens, masked to
assistant turns; one LoRA step; confirm the loss is finite and gradients flow.

Then the part that matters: **serve the adapter and run harbor against it.**
`train → serve → measure` has never closed. If a Modal-served LoRA cannot be
driven through litellm's `hosted_vllm` provider and graded by our own verifier,
no arm in `PLAN.md` is measurable, whichever trainer sits underneath.

### Phase 2 — OPD for real, on the tasks routing sent to the OPD bucket.

Plus non-regression inside a tolerance declared *before* the run.

**Open risk:** verl's OPD documentation never mentions LoRA, and verl's general
LoRA support has known rough edges under vLLM rollout. Our own OPD path does not
inherit that, but the question — does LoRA behave under a distillation loss —
is the same question, and it is worth answering on a small run before it is
load-bearing.

### Phase 3 — the RL branch, on verl. Only if Phase 0 populated the RL bucket.


Distilling frontier models — and why we do not
----------------------------------------------

Worth writing down, because it will be asked.

There are two ways to distill.

**White-box**, with an open-weights teacher: real logits, real token-level KL.
This is what we do. The closed frontier model (GPT-5, Claude) is never the
teacher — it is the **ceiling we measure against**.

**Black-box**, with the frontier model itself as teacher: text outputs only, no
logits. A real and active literature — SeqKD as the baseline, then GAD
(a discriminator as a co-evolving on-policy reward model; a 14B student reached
parity with GPT-5-Chat on LMSYS-Chat), OVD (trajectory-level verbal scoring),
ROPD (rubrics distilled from teacher–student contrasts), OmniOPD, DistIL.

Every one of those exists because black-box distillation **has no ground truth.**
GAD trains a discriminator precisely because there is nothing to check the
student against. ROPD invents rubrics for the same reason.

We have ground truth. It is the test suite.

`PLAN.md` already rejects GAD — *"a learned reward model substitutes for a
verifier we already have"* — and that holds for the whole family. The correct
posture is not that these methods are wrong; it is that they are what you build
when you cannot mine a verifier, and the comparison is one we should be able to
run and win.

(ROPD's "rubrics from teacher–student contrasts" is structurally what
`diagnose.py` does with frontier wins against candidate losses. Same idea,
blurrier instrument. A citation, not a threat.)


Stop conditions
---------------

Inherited from `PLAN.md`, plus what is specific to executing it:

- Tokenizers do not match and no same-family substitution fixes it.
- ★ `pass@k` curves do not separate into regimes — no routing decision exists.
- `train → serve → measure` does not close, i.e. a Modal-served LoRA cannot be
  driven by harbor. Every arm is unmeasurable until this works.
- LoRA does not behave under the distillation loss — non-regression loses its
  mechanism and the design needs rethinking before scale.
- B1 ≈ B2 — routing adds nothing.
