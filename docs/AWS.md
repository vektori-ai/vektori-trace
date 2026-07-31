> **STALE — do not build from this.** Superseded by `FINAL-PLAN.md`.
> Kept for history. No longer true: Assumes two GPU instances, one serving the teacher for prompt_logprobs. That teacher instance no longer exists — teacher is Fireworks-hosted; student is local LoRA.

# Running the pilot on AWS

Historical AWS attach path for a self-hosted student endpoint. Teacher serving assumptions here are superseded by `FINAL-PLAN.md` (Fireworks teacher). Nothing in the measurement half requires Modal. The only Modal-specific
things in the repo are (a) `serve.py`, which spawns a container and owns its
lifecycle, and (b) the Volume used to hand an adapter from train to serve. Both
have replacements:

| Modal | AWS |
|---|---|
| `serve.serve_model` (spawns) | `endpoint.attach_endpoint` (attaches to yours) |
| Volume adapter handoff | local disk on the instance |
| `train_lora_modal` | `train_lora` on the instance's own GPU |
| Volume-backed `HF_HOME` | EBS, or a shared EFS mount |

The seam is `ArmsConfig.serve_cm`. `--api-base` populates it with
`endpoint.endpoint_serve_cm`, and `arms.py` never learns the difference.

> One semantic difference, deliberate: exiting the `attach_endpoint` context does
> **not** stop the server. Its lifetime belongs to the instance, not to a Python
> block. Stopping the instance is your call — and the thing that costs money, so
> `aws ec2 stop-instances` belongs in your own teardown, not in a `finally:`.

## Instances

The pilot pair is Qwen3-Coder-30B-A3B (teacher, fp8 ~31GB) → Qwen3-8B (student).

| Role | Instance | GPU | Notes |
|---|---|---|---|
| Student serve + LoRA train | `g6e.xlarge` | 1×L40S 48GB | cheapest box that holds 8B bf16 + LoRA + KV cache |
| Student serve + train (faster) | `p4d.24xlarge` slice / `p5` | A100/H100 80GB | if step time matters |
| Teacher (OPD scoring) | `p4d`/`p5`, or `g6e.12xlarge` | 80GB, or 4×L40S | fp8 30B needs ~31GB weights + KV |
| Everything else | any CPU box | — | mining, `prove`, `route`, `check-tokenizers` |

Avoid `p3` (V100) and `g4dn` (T4): pre-Ampere, so no bf16 and no fp8. `train.py`
detects this via `torch.cuda.is_bf16_supported()` and falls back to fp32, which
will then OOM an 8B — the fallback is there for CPU tests, not as a plan.

Note the mining/verification half runs in Docker on a CPU box. Do not pay for a
GPU while mining; the measured yield is 10%, so a 50-task corpus is ~500 PRs of
GitHub API and container time with no model in the loop at all
(`mine --dockerfile --test-cmd --no-replay`).

## 1. Serve the student

On the GPU instance:

```bash
pip install vllm
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1   # required for the A2/A3 arms
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 8000 \
  --served-model-name qwen3-8b \
  --enable-lora --max-lora-rank 16 \
  --enable-prefix-caching
```

`--enable-lora` must be present **at launch** — it cannot be turned on later, and
without it the A2/A3 arms fail after you have already paid for training. Keep
`--max-lora-rank` ≥ `LoraHyperparams.r` (default 16).

`--enable-prefix-caching` matters more here than on Modal: a pass@k sweep sends
hundreds of rollouts sharing a long system prompt.

Confirm from the machine that will drive the run (not from the GPU box — this is
where a security group mistake shows up):

```bash
curl -s http://<instance-ip>:8000/v1/models | jq .
```

Open port 8000 to your driver host only. A vLLM server has no auth; an
open-to-the-world `/v1/completions` is someone else's free GPU.

## 2. Point the run at it

Every command that talks to a model already takes `--api-base`:

```bash
# pass@k sweep — the support measurement behind the routing decision
vektori-trace passk --tasks-dir ./tasks \
  --model hosted_vllm/qwen3-8b \
  --api-base http://<ip>:8000/v1 \
  --max-workers 8

# full A0–A4 sweep, no Modal anywhere
vektori-trace run-arms \
  --selection ./out/selection.json --diagnosis ./out/diagnosis.json \
  --tasks-dir ./tasks --agent terminus-2 \
  --api-base http://<ip>:8000/v1 \
  --out ./out/arms
```

`--api-base` on `run-arms` implies local training, so the adapter is written to
`--out/<arm>/adapter` and registered with the running server over vLLM's
runtime-LoRA API. **Run `run-arms` on the GPU instance itself**, or put `--out` on
a mount the server can read: `load_lora_adapter` passes a *path*, and the server
reads it from its own filesystem.

`--max-workers` is worth setting. The sweep is ~1,300 containerised rollouts, and
serially that is the dominant cost of the whole pilot.

## 3. OPD teacher scoring

OPD needs `prompt_logprobs` — the teacher's logprob for tokens *we* supply, not
tokens it sampled. A teacher server you control always provides it, which is
exactly what you now have:

```bash
# on the teacher instance
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --host 0.0.0.0 --port 8001 --quantization fp8 \
  --served-model-name qwen3-coder-30b
```

Run `check-tokenizers` first — it is free, and it is the precondition for the
whole method: the student's sampled ids are sent to the teacher, so the two must
read those ids as the same strings. `run_opd_training` enforces it, but finding
out before you rent the second GPU is cheaper:

```bash
vektori-trace check-tokenizers
```

Then run the loop. Both servers up, student on the training box:

```bash
vektori-trace distill \
  --teacher-traces ./out/replay/frontier-jobs \
  --teacher-api-base http://<teacher-ip>:8001/v1 \
  --student Qwen/Qwen3-8B \
  --max-steps 200 --examples-per-step 4 \
  --out ./out/opd
```

### Not renting the teacher GPU at all

The teacher instance above is the expensive half of this page and the half that
H100 capacity keeps blocking. Two hosted endpoints score supplied tokens and can
stand in for it — Fireworks, and Bedrock Custom Model Import, which needs no
capacity reservation. `docs/HOSTED_TEACHERS.md` covers both, what they cost in
quantisation and top-K, and the one-request probe that decides whether a given
deployment can do it at all:

```bash
vektori-trace probe-teacher --backend fireworks
```

`--teacher-traces` takes harbor job dirs (what `replay` leaves behind) and/or
ATIF `.json` traces. Each parent assistant turn becomes one ReOPD step-example:
the turns before it are the frozen prefix, and the student acts there.

Per step, per example: one student sample, one teacher round-trip, one student
forward. Watch `out/opd/opd_log.jsonl` — `mean_log_ratio` is the monitoring
scalar and should trend toward 0 as the student's distribution approaches the
teacher's. `loss` alone is not readable as progress (the surrogate is not a
divergence); the ratio is.

If `skipped_empty_samples` is a large fraction, the student is emitting immediate
EOS at those prefixes and the run is measuring nothing — check the chat template
and `--max-new-tokens` before spending more.

### `--top-k`: the lower-variance objective

Default (`--top-k 0`) is the objective PLAN.md declares: a reverse-KL surrogate
over the tokens the student sampled. `--top-k 16` switches to an analytic KL over
the teacher's top-K set at each position — **the same teacher request**, so it
costs nothing extra, and thunlp/OPD reports 97–99% of the mass at
student-visited states sitting in a small shared token set, so K=16 is nearly the
whole distribution rather than a slice of it. Much lower gradient variance.

It is a *different objective*, not a tuning knob: pre-register the choice, and
note the run records which one it used in `opd.json`'s `provenance.loss`.

Because OPD queries the teacher *every step*, teacher latency dominates the run.
That is why the pair is an MoE teacher (~3.3B active). Put the teacher and the
student in the **same AZ**: per-step scoring makes cross-AZ round-trips a real
cost, and cross-AZ transfer is billed.

## Prior art, and what it says about this design

**thunlp/OPD** (the paper's code) is the closest published thing to this branch,
and two of its findings bear directly on the plan:

1. *"OPD works when the teacher offers genuinely new capabilities beyond the
   student's training data."* This is the routing rule, arrived at
   independently. `passk` + `route` decide RL-vs-OPD by whether the capability
   is inside the student's support — which is exactly the condition thunlp found
   determines whether OPD helps at all. Their result is the strongest external
   evidence that the measurement is the right one, and it means a run that skips
   it can train for a long time and learn nothing.
2. *Top-K, not single-token* (`LOG_PROB_TOP_K=16`). Implemented here as
   `--top-k`; see above.

Their caveat is worth carrying too: *"OPD's apparent free lunch of dense
token-level reward comes at a cost, raising the question of whether OPD can scale
to long-horizon distillation."* Agent trajectories are long-horizon. ReOPD's
one-step-per-prefix structure is a partial answer (each example is a single
action, not a whole episode), but it is not evidence, and the pilot should not
assume the dense signal survives horizon.

**verl** is where the RL branch goes, unchanged from `PILOT.md`. Note that
thunlp implements OPD *inside* verl as `adv_estimator=token_reward_direct`, which
is prior art for OPD-on-verl — but their setup is 8×A800 80GB, and adopting verl
here would mean taking its whole rollout and resharding stack to run one teacher
against one student. verl earns its complexity on multi-turn GRPO (3D-HybridEngine
weight resharding, which is genuinely hard to hand-roll); OPD needs none of it,
because the teacher is frozen and the student samples in-process. The split stands.

## Still open on this path

- **`dataset.py` re-tokenizes text.** Correct for SFT, and now moot for OPD:
  `distill.py` holds prefix and sampled tokens as ids end to end and
  `teacher.score_ids` never round-trips them through text. The SFT path is
  unaffected either way.
- **The loop is single-example-at-a-time.** `examples_per_step` accumulates
  gradients but each example is a separate sample→score→forward round-trip, so
  teacher latency is not amortised. Batching the teacher calls across an
  accumulation group is the obvious next win and needs no new infrastructure.
- **Untested against a real vLLM.** The wire contract is covered against a stub
  (`tests/test_endpoint.py`), including `prompt_logprobs` shape and alignment.
  First contact should confirm the entry format on the real server before a long
  run — a mismatch raises rather than corrupts, which is the point of the guards.
- **Spot instances** will kill a run mid-sweep. `resume.py` exists for mining;
  neither the arms sweep nor the OPD loop checkpoints, so use on-demand or accept
  restarts. Mid-run OPD checkpointing is not built.
