Hosted teachers, and a student on someone else's GPUs
=====================================================

**Date:** 2026-07-30 · Companion to `docs/OPD.md` and `docs/AWS.md`.

`docs/AWS.md` assumes two GPU instances: a teacher serving `prompt_logprobs` and
a student training against it. The teacher is the expensive half, and the half
H100 capacity keeps blocking. This document is the path where neither GPU is
rented.


What changed
------------

`docs/OPD.md` used to state that no hosted API scores tokens you supply, and
`teacher.py` was written on that basis. That claim is correct about the *generic*
OpenAI-compatible surface — `logprobs` there covers only tokens the server
sampled, which is the wrong quantity — and incorrect about two specific vendors,
each of which exposes the right quantity through its own extension.

Both now have pools implementing the same protocol as `teacher.VllmTeacherPool`
(`score_ids`, `score_ids_topk`, `generate`, `provenance`), so `distill.py`'s loop
consumes them with no changes.

| | `teacher.py` (vLLM) | `teacher_fireworks.py` | `teacher_bedrock.py` |
|---|---|---|---|
| Scores supplied tokens | `prompt_logprobs` | `echo_last` + int-array `prompt` | `prompt_logprobs` |
| Prefix as ids, no re-tokenisation | yes | yes | yes |
| Max top-K | unlimited (K=16 used) | **5** (public cap) | unlimited, per docs |
| Quantisation | yours to choose | **FP8** | yours (you imported the weights) |
| GPU to hold | yes | no | no |
| Verified firsthand | yes (`tests/test_endpoint.py`) | **no — run the probe** | **no — run the probe** |


Fireworks
---------

`POST /inference/v1/completions` accepts three things that combine into exactly
the operation OPD needs:

- `prompt` as an **array of integers** — the ids the student sampled reach the
  teacher unchanged, so no tokenisation boundary can shift between the two.
- `echo_last: N` — documented as "echo back the last N tokens of the prompt …
  useful for obtaining logprobs of the prompt suffix". `prompt_logprobs`
  restricted to a suffix, which is the only window OPD scores.
- `logprobs: true` + `top_logprobs: K` — the OpenAI-compatible response format,
  whose entries carry `token_id` on both the token and every alternative. Ids,
  not strings, so `distill.align_topk_rows` consumes them directly.

The strongest evidence this is supported rather than incidental: Fireworks built
their own on-policy distillation recipe on it
(`training.recipes.distillation_loop`, mode `sampled_reverse_kl`, trained on
`teacher_logprob - sampling_logprob`), which is the same objective as
`opd.reverse_kl_surrogate`.

Two numbers that belong in provenance, both emitted by
`FireworksTeacherPool.provenance()`:

- The field read is `logprob`, the model logprob **before** temperature and
  sampling-filter renormalisation — log π_t(a_t) under the teacher's own
  distribution. The sibling `sampling_logprob` is the post-filter one; training
  against it silently makes the objective a KL to a temperature-warped teacher.
- Fireworks serves **FP8**. Quantisation noise inside a term the student
  differentiates through. Two runs differing only in this are not comparable.

The `top_logprobs` cap of 5 means `topk_reverse_kl` cannot reach thunlp/OPD's
K=16 here. The pool **raises rather than clamping**: K is part of the objective,
not a tuning knob, so a config asking for 16 must not quietly get 5.

The pilot pair is already served — teacher
`accounts/fireworks/models/qwen3-coder-30b-a3b-instruct`, student
`accounts/fireworks/models/qwen3-8b`.


Bedrock Custom Model Import
---------------------------

CMI serves weights you import with no capacity reservation, which is the property
that matters when H100 capacity never materialises on launch. Models imported
after **2025-11-11** expose log probabilities, and AWS's schema matrix says:

| Schema | Routed by | Logprobs |
|---|---|---|
| BedrockCompletion | `max_gen_len` | output tokens only — **wrong quantity** |
| OpenAICompletion | `max_tokens` | prompt and output |
| OpenAIChatCompletion | `messages` | prompt and output |

`max_tokens` versus `max_gen_len` is the router and it is load-bearing: send the
wrong one and you get a successful response containing generated-token logprobs,
which look fine and are not what OPD asked for.

Setup is `docs/AWS.md`'s S3 + IAM + import-job path, in a CMI region
(`us-east-1`, `us-west-2`, `eu-central-1`, `ap-northeast-1` — note `ap-south-1`,
this repo's usual default, is not one; `BedrockTeacherPool` rejects it at
construction rather than letting `invoke_model` 404 in a way that reads like a
missing model).

**Two things are documented in a way that does not settle them, so the code
handles both and the probe decides:**

1. AWS demonstrates `prompt_logprobs` on `OpenAIChatCompletion` with a `messages`
   payload. It claims `OpenAICompletion` support and never shows that request,
   and it nowhere states whether `prompt` accepts an **integer array**. Both are
   load-bearing. A 400 here is a finding, and the fallback is the self-hosted
   teacher, not a workaround.
2. AWS's own CMI example shows `prompt_logprobs` as a **top-level** response
   field; vLLM's OpenAI server nests it under `choices[0]`. `_find_prompt_logprobs`
   accepts either and reports which it found.


The probe
---------

Neither vendor's documentation settles the question, so one request does:

```bash
vektori-trace probe-teacher --backend fireworks --out ./out/probe-fireworks.json
vektori-trace probe-teacher --backend bedrock --model <imported-model-arn> \
  --region us-east-1 --out ./out/probe-bedrock.json
```

Exit 0 means OPD can run against that teacher. Exit 1 means it cannot, and the
message is the reason — a result to record, not a failure to work around. Add
`--top-k 5` to probe the lower-variance objective as well; its failure is not
fatal, since `top_k=0` (`reverse_kl_surrogate`) is the declared objective and is
unaffected.

**Nothing downstream should run against a teacher that has not passed this.**


The student on Fireworks' Training API
--------------------------------------

`student_fireworks.py` is `distill.run_opd_training` with the student's
forward/backward on Fireworks GPUs instead of a local one. The loop, the data
(`reopd.ReOPDStepExample`) and the objective (`opd.reverse_kl_surrogate`,
imported, not reimplemented) are unchanged.

`forward_backward_custom(datums, loss_fn)` runs the forward pass remotely, ships
per-token logprobs back with `requires_grad=True`, calls your loss locally, then
ships `d_loss/d_logprob` back for the model backward. The student logprobs OPD
needs are exactly what it hands you, so the objective stays ordinary Python.

Two consequences that are not cosmetic:

- **`top_k > 0` cannot run here.** `topk_reverse_kl` needs student *logits* over
  the teacher's top-K set; the API returns logprobs for the datum's own tokens
  and nothing else. `run_fireworks_opd` raises instead of substituting the
  sampled-token objective. Fireworks' own recipe splits the same way — their
  `topk_forward_kl` is forward KL via multi-target `cross_entropy`, a different
  objective. The top-K arm needs a local student.
- **On-policy costs a weight sync.** The student's weights live on the trainer,
  sampling happens on a deployment, and between them sits a checkpoint. "On
  policy" is therefore true only up to `sync_every` steps. `distill.py` has no
  such gap. `sync_every=1` closes it at one checkpoint per step, and whatever was
  used lands in provenance, because a run that sampled from 8-step-stale weights
  is off-policy by 8 steps and its report should say so.

Why a custom loop rather than Fireworks' `distillation_loop` recipe: the recipe
owns the loss and the loop, which would leave `opd.py` unused and has no place
for the ReOPD prefix-replay data path (`docs/OPD.md`, Axis 1) that the whole
`reopd.py` module exists to serve.


Open questions
--------------

1. **Does Fireworks' `echo_last` return logprobs over an integer-array prompt?**
   Doc-confirmed, not probe-confirmed. `probe-teacher --backend fireworks`.
2. **Does Bedrock CMI's OpenAICompletion accept an integer-array `prompt`?**
   Undocumented either way. `probe-teacher --backend bedrock`.
3. **Is FP8 teacher noise tolerable inside the KL?** Not answerable by probing —
   it needs the same OPD run against a Fireworks teacher and a self-hosted bf16
   one, compared on the arm metrics. Until then, provenance records which was
   used so the question stays askable.
