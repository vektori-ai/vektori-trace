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
| Verified firsthand | yes (`tests/test_endpoint.py`) | request shape matches the vendor's shipped code; **not yet run** | **no — run the probe** |
| Teacher runs on | your vLLM | a **dedicated deployment**, not serverless | the imported model |


Fireworks
---------

**Settled from Fireworks' own source, not from the API reference.**
`fw-ai/cookbook`, `training/utils/distillation/sampling.py`,
`_request_teacher_echo` — the teacher-scoring call their `sampled_reverse_kl`
distillation mode runs in production:

```python
response, _metrics = await sampler.async_completions_stream(
    prompt=token_ids,        # integer array — the student's sampled ids
    max_tokens=1,
    temperature=0.0,
    logprobs=True,
    echo=True,
    raw_output=True,
    top_logprobs=K,          # only when K > 0
)
```

then reads `choices[0].logprobs.content` for one teacher logprob per sampled
response token. Their loss is `teacher_logprob - sampling_logprob`, the same
objective as `opd.reverse_kl_surrogate`. So this is not a capability inferred
from prose — it is the vendor shipping the exact operation OPD needs.

`teacher_fireworks.py` sends that request verbatim. Four details taken from the
source rather than the docs:

- **`echo: true`, not `echo_last`.** `echo_last: N` is in the public OpenAPI
  schema and is much cheaper — it ships logprobs for a suffix instead of a
  3.5k-token prefix — but the vendor's code does not use it. It is available as
  `echo_mode="last"` and is opt-in until the probe confirms it; `"full"` is the
  proven default.
- **The teacher is a dedicated inference deployment**, not a serverless model.
  The recipe auto-creates one when handed a base model
  (`teacher_deployment_id`, `teacher_deployment_shape`). Budget for it.
- **`top_logprobs` is omitted, not sent as 0**, when the sampled-token objective
  is the one running.
- **Entries are not guaranteed to carry `token_id`.** The vendor's
  `_candidate_token_id` falls back to round-tripping the token *string* through a
  local tokenizer — precisely the boundary-shift risk `teacher.py` exists to rule
  out. Our alignment verifies by id when ids are present and otherwise dispatches
  on the response *length* against the two known echo layouts, refusing anything
  that fits neither. `content[i]` scores `token[i]`, or `token[i+1]` in the
  "training-aligned" P+C−1 shape.

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

1. ~~**Does Fireworks score supplied tokens?**~~ **Resolved 2026-07-30 from
   `fw-ai/cookbook`** — their production distillation does exactly this, and our
   request is now that request. What remains is running it against a real
   deployment, and whether the cheaper `echo_mode="last"` behaves the same.
2. **Does Bedrock CMI's OpenAICompletion accept an integer-array `prompt`?**
   Undocumented either way. `probe-teacher --backend bedrock`.
3. **Is FP8 teacher noise tolerable inside the KL?** Not answerable by probing —
   it needs the same OPD run against a Fireworks teacher and a self-hosted bf16
   one, compared on the arm metrics. Until then, provenance records which was
   used so the question stays askable.
