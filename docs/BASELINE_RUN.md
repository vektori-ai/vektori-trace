# Baseline run — Qwen3-8B on one mined prefect task

Handoff runbook. Everything a fresh session needs to take this from nothing to a
measured number. Written 2026-08-02, against `main` at `aea5f59`.

---

## 1. What this is

**Question:** can Qwen3-8B solve any of the mined prefect tasks at all?

`FINAL-PLAN.md` flags that the mined tasks have never been measured against the
student — *"if the student already solves them there is no gap to distil into."*
Every downstream stage (OPD, routing, the arms matrix) is worthless until that
number exists. This run produces it, at the smallest scale that produces it.

**Scope, decided deliberately:**

| | |
|---|---|
| Tasks | **1** (of 48 available) |
| Rollouts | **n = 4** |
| Report | **pass@1 = c/4** |
| Token capture | **on** |
| Teacher / Fireworks / OPD | **no** |
| Training | **no** |
| wandb | **no** |

### Why pass@1 and not pass@4

`pass@k = 1 − C(n−c, k) / C(n, k)`, where **n** = attempts run, **c** = attempts
that passed, **k** = the hypothetical budget being reported.

At n=4, k=4 there is exactly one way to choose 4 from 4, so the formula collapses
to 0 if c=0 and 1 otherwise. A task where 1 of 4 passed scores identically to one
where 4 of 4 passed. **k must be meaningfully smaller than n.** The same 4
rollouts reported as pass@1 = c/4 are a real measurement; pass@4 from n=4 is not.
If a genuine pass@4 is ever wanted, n≥8 is required.

---

## 2. Architecture

```
  EC2 t3.large (ap-south-1)                    Modal
  ┌──────────────────────────────┐            ┌──────────────────┐
  │ serve_student.py ────────────┼── holds ──▶│ L40S, vLLM       │
  │   └─ GPU sampler thread ─────┼── NVML ───▶│   Qwen3-8B bf16  │
  │                              │            └──────────────────┘
  │ capture-proxy :port ─────────┼── forwards ──────▲
  │   └─ token_captures.jsonl    │                  │
  │                              │                  │
  │ vektori-trace passk          │                  │
  │   └─ harbor ─ terminus-2 ─ litellm ─────────────┘
  │        └─ Docker verifier containers (prefect test suite)
  └──────────────────────────────┘
```

GPU serves tokens; the box runs the agent loop and the Docker verifier. **The box
is the bottleneck, not the GPU** — 2 vCPU running a prefect test suite, while the
L40S bills $1.95/h waiting. Cost here is set by container wall-clock, not tokens.

---

## 3. Access

**AWS box** — `i-0a348ff3d7be9769a`, `t3.large` (2 vCPU / 7 GB), **ap-south-1**,
account `478499050241` (alias `vektori`). **SSM Session Manager only** — no SSH,
no inbound ports, no `.pem`. Region must be passed explicitly; the default CLI
region is wrong for this account.

```bash
aws ssm start-session --target i-0a348ff3d7be9769a --region ap-south-1
sudo su - ubuntu
cd /data/vektori-trace
```

For non-interactive commands use `aws ssm send-command --document-name
AWS-RunShellScript --parameters file://params.json` — **inline `--parameters`
JSON breaks on nested quotes**, write a params file. Then
`aws ssm get-command-invocation` for output.

Run everything as **`ubuntu`**, not root. Python is `/data/vektori-trace/.venv/bin/python`
(`uv` is not on root's PATH).

**Modal** — token at `~/.modal.toml` on both the box and the laptop.

---

## 4. Verified state (all checked 2026-08-02 — nothing below is assumed)

**Box toolchain:**

```
docker        29.1.3, prefect bootstrap image built (local/r2e-bootstrap/prefecthq__prefect)
harbor        0.20.0
transformers  5.5.3    tokenizers 0.22.2    torch 2.13.0+cu130
litellm       1.93.0   (hosted_vllm provider present — this is how harbor reaches vLLM)
modal         1.5.3    openai 2.48.0
HF cache      17M — Qwen/Qwen3-8B tokenizer downloaded, encode→decode round-trip verified
disk          88G free on /data          RAM 7G total, ~6G free
git           synced to aea5f59, clean tree, credential.helper=store
```

**Modal side:**

```
auth          works from the box, account `laxmansrivast` (~/.modal.toml)
volumes       hf-model-cache          ✓ exists (2026-07-29)
              vektori-trace-adapters  ✓ exists (2026-07-29)
apps running  NONE — nothing is billing
```

`git config --global --add safe.directory /data/vektori-trace` has been applied
**for `ubuntu`**. If git complains about dubious ownership, it is being run as
root — that is a different global config.

**The one genuine unknown:** whether Qwen3-8B's ~16 GB of weights are already in
the `hf-model-cache` Volume. That is the difference between a ~2 minute start and
a ~15 minute one. Step A1 below checks it for free.

---

## 5. The corpus

48 task dirs / **46 unique ids**, all `prefecthq/prefect`, produced by the
`commit_runtime` pipeline (#26/#27). Untracked, and present **only** on that EBS
volume — there is no second copy anywhere.

```
/data/vektori-trace/cs/smoke/cmined/mined_tasks/         28 dirs   ← use this one
/data/vektori-trace/cs/smoke/mined/mined_tasks/          18 dirs
/data/vektori-trace/cs/smoke/cmined-smoke/mined_tasks/    2 dirs   (dupes of cmined)
```

Each dir is a harbor task: `task.toml` (schema 1.3), `instruction.md`,
`environment/`, `solution/`, `tests/`.

> `docs/mined-tasks-inventory.md` is **stale** — it describes an older
> anyio/click/tenacity corpus and is gitignored. Ignore it.

### Picking the task

All 28 are `difficulty = "medium"`, `category = "bugfix"` — that metadata carries
no signal. Pick on test counts instead:

| task | F2P | P2P | note |
|---|---|---|---|
| **`prefecthq__prefect-65ea05bef8d9`** | **1** | **4** | **recommended** — smallest verifier surface, fastest wall clock, cheapest first run |
| `prefecthq__prefect-59ec4f7f6dd4` | 1 | 8 | next smallest |
| `prefecthq__prefect-27c8ee6110b5` | 1 | 21 | |
| `prefecthq__prefect-fb3d166c11e5` | 12 | 100 | heaviest — avoid for a first run |
| `prefecthq__prefect-5c70e74e25fd` | 8 | **0** | **avoid** — no P2P means no regression guard, so a patch that breaks everything else still scores 1.0 |

The first run is about proving the pipeline, so minimise verifier wall-clock.

---

## 6. Files that matter

| file | why |
|---|---|
| `scripts/serve_student.py` | Holds the Modal endpoint open. Refuses impossible `--max-model-len` **before** allocating a GPU. Owns the NVML sampler → `gpu_log.jsonl`. Traps SIGHUP. |
| `scripts/verify_run_logs.py` | Proves the four logs are complete and join. `--preflight` before spending rollouts. |
| `scripts/vllm_monitor.py` | vLLM `/metrics` — KV %, queue depth, throughput. `--log` persists samples. |
| `vektori_trace/serve.py` | `serve_model` context manager, the pinned vLLM image, `VllmServer.gpu_stats` (NVML), the KV arithmetic in the docstring. |
| `vektori_trace/token_capture.py` | `CaptureProxy`, `CapturedCompletion` (ids **+ text** + latency). |
| `vektori_trace/passk.py` | `two_stage_sweep`, `measure_passk_stage`, `pass_at_k`, `passk_log.jsonl`. |
| `vektori_trace/validity.py` | `run_trial` — shells out to `harbor run`, parses the reward, times it. |
| `vektori_trace/cli.py` | `passk` and `capture-proxy` subcommands. Logging config at `main()`. |
| `FINAL-PLAN.md` | The authoritative plan. §1 glossary, §2 config, §3 the existing loop. |
| `docs/AWS.md` | **Banner-marked STALE.** Historical. Do not build from it. |

---

## 7. How the model gets onto Modal

There is **nothing to install or provision**. `serve_model` (`vektori_trace/serve.py`)
declares the whole thing, and Modal builds it on first call. Understanding what it
declares matters mainly for reading failures.

**Image — the version pins are load-bearing and must move together:**

```
nvidia/cuda:12.9.0-devel-ubuntu22.04      ← -devel, NOT -runtime: carries nvcc
  + vllm==0.21.0
  + huggingface_hub
  + nvidia-ml-py                           ← backs VllmServer.gpu_stats
```

Unpinned `pip install vllm` resolves to the newest release, which pulls **cu13**
wheels onto a CUDA 12.x base; and the original `debian_slim` image had no CUDA
toolchain at all, so vLLM's V1 engine died on *"Could not find nvcc"*. Both
failures land **after** Modal has allocated a GPU and pulled ~16 GB. This is the
pair Modal publishes in its own vLLM example, so it is a tested combination.

**Volumes:**

| mount | volume | role |
|---|---|---|
| `/root/.cache/huggingface` | `hf-model-cache` | `HF_HOME`. Weights survive scale-to-zero. The env var *alone* only relocated the cache inside an ephemeral container, so every cold start re-downloaded the model. |
| `/adapters` | `vektori-trace-adapters` | LoRA handoff. **Unused this run** — no adapter. |

**Container:** `gpu="L40S"`, `serialized=True`, `scaledown_window=600s`,
`timeout=3600`.

`serialized=True` is required because `VllmServer` is defined *inside*
`serve_model` so it can close over the model name and flags. Modal normally
re-imports classes by module path; without this, every call raised
`LocalFunctionError` before reaching a GPU.

**Env:** `HF_HOME`, `HF_XET_HIGH_PERFORMANCE=1` (faster Hub pulls),
`VLLM_LOG_STATS_INTERVAL=1` (so `/metrics` is live — `vllm_monitor.py` reads
exactly these).

**What vLLM is actually launched with:**

```
python -m vllm.entrypoints.openai.api_server
  --model Qwen/Qwen3-8B --host 0.0.0.0 --port 8000
  --served-model-name Qwen3-8B
  --max-model-len 40960 --gpu-memory-utilization 0.90
```

No `--enable-lora` this run. If a LoRA is ever served, that flag **must** be
present at launch — it cannot be turned on later, and its absence surfaces only
after training has already been paid for.

---

## 8. Runbook

Four tmux windows on the box. **Start the GPU monitor before the sweep** so the
idle baseline is on record.

### Step 0 — sync and sanity

```bash
cd /data/vektori-trace && git pull      # idempotent; was at aea5f59 as of writing
uv run pytest tests/ -q                 # expect 827 passed, 1 skipped
uv run modal app list                   # expect EMPTY. Anything here is billing.
```

### Step A1 — is the model already cached? (free, no GPU)

```bash
uv run modal volume ls hf-model-cache /hub
```

Look for `models--Qwen--Qwen3-8B`. Present → fast start. Absent → the first start
pulls ~16 GB into the Volume, once, and every later start is fast.

### Step 1 — serve (window `serve`)

```bash
tmux new -s serve
cd /data/vektori-trace
uv run python scripts/serve_student.py \
  --gpu L40S \
  --max-model-len 40960 \
  --max-hours 2 \
  --gpu-log /data/vektori-trace/vektori-out/baseline/gpu_log.jsonl \
  --gpu-log-interval 5 \
  --write-env /data/vektori-trace/.env.run
```

It prints the KV budget first, then starts. Expect:

```
kv budget   26.60 GiB  =  193,695 tokens total
max-model-len 40,960
→ concurrent seqs 4  at full length
```

First cold start downloads ~16 GB into the `hf-model-cache` Modal Volume — slow
once, fast after. The script runs a **real one-token completion** before
reporting success; "UP" without that only means a socket is listening.

`Ctrl-B D` to detach. **The endpoint lives exactly as long as this process.**

### Step 2 — capture proxy (window `proxy`)

```bash
cd /data/vektori-trace
source .env.run                       # sets STUDENT_API_BASE
uv run vektori-trace capture-proxy \
  --upstream "$STUDENT_API_BASE" \
  --out /data/vektori-trace/vektori-out/baseline/captures
```

Prints a local `/v1` URL. **Point harbor at the proxy, not at Modal** — that is
what "capture on" means. The proxy forwards each request, injects vLLM's
`return_token_ids`, records ids + text + latency, and passes the response
through unchanged.

### Step 3 — vLLM metrics (window `gpu`)

```bash
uv run python scripts/vllm_monitor.py \
  --api-base "$STUDENT_API_BASE" \
  --kv-total-tokens 193695 \
  --interval 5 \
  --log /data/vektori-trace/vektori-out/baseline/vllm_metrics.jsonl
```

### Step 4 — preflight (~30 seconds of GPU, do not skip)

Send one throwaway completion **through the proxy**, then:

```bash
uv run python scripts/verify_run_logs.py \
  --out /data/vektori-trace/vektori-out/baseline \
  --preflight \
  --tokenizer Qwen/Qwen3-8B
```

Must show PASS for: captures exist · every capture carries emitted text · every
capture carries proxy timing · no empty `token_ids` · **decoding `token_ids`
reproduces text** · gpu log exists with real samples and plausible memory.

That decode check is the important one. Ids and text that silently disagree are
worse than no capture at all, because they look fine. If anything fails, fix it
before spending rollouts.

### Step 5 — the sweep (window `sweep`)

`passk` takes a **directory of tasks** and runs every task in it, so stage the one
task into a directory of its own first:

```bash
cd /data/vektori-trace
TASK=prefecthq__prefect-65ea05bef8d9
mkdir -p /tmp/one && cp -r "cs/smoke/cmined/mined_tasks/$TASK" /tmp/one/
ls /tmp/one          # must show exactly one task dir

uv run vektori-trace passk \
  --tasks-dir /tmp/one \
  --agent terminus-2 \
  --model hosted_vllm/Qwen3-8B \
  --api-base "$PROXY_API_BASE" \
  --model-info @model_info.json \
  --stage1-n 4 \
  --no-escalate \
  --max-workers 1 \
  --out /data/vektori-trace/vektori-out/baseline
```

`$PROXY_API_BASE` is the **capture proxy's** URL from Step 2, not Modal's. Point
it at Modal and the run works but records no tokens.

`--model-info` is required by harbor for `hosted_vllm/` models — it carries token
limits and costs, and harbor refuses to start without it. `serve_student.py`
prints the correct JSON at startup under `harbor kwargs:`; save the `model_info`
value to `model_info.json`.

**`--no-escalate` is not optional.** Without it, a task that fails all 4 rollouts
escalates to `--stage2-n` (default **32**) more, because escalation triggers on
`c == 0` with no regard for how small stage 1 was. Failing is the likely outcome
for an untested model.

### Step 6 — verify and read

```bash
uv run python scripts/verify_run_logs.py \
  --out /data/vektori-trace/vektori-out/baseline \
  --expect-rollouts 4 \
  --tokenizer Qwen/Qwen3-8B
```

Then **tear the endpoint down** (`Ctrl-C` in the `serve` window) and confirm:

```bash
uv run modal app list      # nothing should still be running
```

---

## 9. What the run produces

| artifact | contents |
|---|---|
| `vektori-out/baseline/passk.json` | pass@k curves, `support`, `infra_failures`, `no_gradeable_rollouts`, and a `rollouts` list naming each job dir |
| `vektori-out/baseline/passk_log.jsonl` | 4 lines, one per rollout, flushed as it completes: pass/fail, reward, elapsed, started_at, turn/tool-call counts, job dir |
| `vektori-out/baseline/passk_jobs/stage1/<task>-{0..3}/` | 4 **distinct** dirs, each with harbor's ATIF trajectory; failures also get `harbor_stdout.txt` / `harbor_stderr.txt` |
| `.../token_captures.jsonl` | one line per model call: `prompt_token_ids`, `token_ids`, `text`, `logprobs`, `request_started_at`, `latency_ms` |
| `vektori-out/baseline/gpu_log.jsonl` | NVML every 5s: util %, memory, temperature, power, SM clock |
| `vektori-out/baseline/vllm_metrics.jsonl` | KV %, running/waiting counts, tokens/sec |

All carry **wall-clock UTC** so they join. `verify_run_logs.py` asserts the join.

### Reading the result

**pass@1 = c/4.** Report it with the scaffold — `terminus-2` — because pass@k is a
property of (model + scaffold), not the model alone.

**Check `infra_failures` before believing a 0.** On a 2-vCPU box some failures
will be harbor timeouts, not the model failing. Those are excluded from the
denominator by design; counting them as losses is the easiest way to publish a
wrongly-pessimistic baseline. `no_gradeable_rollouts` names tasks that produced
nothing gradeable at all.

**Record per-rollout wall time.** It is the number that sizes every future sweep,
and right now it is completely unknown — that uncertainty dominates the GPU bill
far more than any choice of n.

---

## 10. Failure modes to expect

| symptom | cause |
|---|---|
| Modal rejects the container before it starts | GPU string. Modal uses `"A10"`, not `"A10G"`. `"L40S"` is correct. |
| vLLM refuses to start on `--max-model-len` | Exceeds the KV budget. On L40S at 40960 it fits; the guard in `serve_student.py` catches this before a GPU is allocated. |
| "engine core initialization failed" | The real traceback is ~150 lines earlier in `modal app logs`. `serve.py` polls `Popen.returncode` and fails fast naming this. |
| 36 rollouts instead of 4 | `--no-escalate` was omitted. |
| Rollouts share one job dir | Fixed in #31, but if a sweep predates it, trajectories overwrite each other and only the last survives. |
| Empty `token_captures.jsonl` | Harbor was pointed at Modal directly rather than at the proxy. |
| GPU still billing after the session died | `serve_student.py` traps SIGHUP for exactly this, and `--max-hours` is a backstop. **Always verify with `modal app list`.** A killed tmux session previously left a GPU running with nothing attached. |

---

## 11. When to stop

The GPU bills on wall-clock, so the expensive mistake is retrying into a failure
rather than reading it. Hard stops:

| condition | action |
|---|---|
| Cold start fails **twice** | **Stop.** Tear down, read `modal app logs`, diagnose offline. Do not retry a third time — each attempt is a fresh GPU allocation and a possible 16 GB pull. |
| Any preflight check fails | **Fix before rollouts.** Rollouts recorded against broken logging are money spent for nothing, and the whole point of the preflight is that it costs ~30 seconds instead of 4 rollouts. |
| A single rollout exceeds ~40 min | Kill it. `run_trial`'s timeout is 1800 s, but a prefect suite that slow makes 4 serial rollouts a multi-hour bill on 2 vCPU. Reconsider the task choice. |
| `modal app list` non-empty after teardown | `modal app stop <id>` **immediately** — that is a GPU billing with nothing attached. This has happened before (#24, bug 4). |
| Anything unexpected costs more than ~$10 | Stop and report rather than pushing through. The whole run is scoped to ~$4. |

**Teardown is not optional and is not automatic if the process is killed
uncleanly.** `serve_student.py` traps SIGINT/SIGTERM/SIGHUP and has a
`--max-hours` backstop, but always confirm with `modal app list`.

---

## 12. Decisions already made — do not re-litigate

A fresh session will be tempted to reopen these. They were settled deliberately:

| decision | why |
|---|---|
| **n=4, report pass@1** | pass@4 from n=4 is degenerate (§1). n=8 doubles the bill for a curve nobody needs yet. |
| **`--no-escalate`** | Escalation fires on `c == 0` regardless of stage-1 size. Without it, 4 rollouts becomes 36. |
| **1 task, not 4** | This run proves the pipeline. Breadth costs linearly and buys nothing until one rollout is known to work end to end. |
| **Token capture on** | Rollouts are expensive and unrepeatable. Capture costs ~nothing now and cannot be recovered afterwards. |
| **GPU on Modal, driver on EC2** | G/VT quota in ap-south-1 is **0**, on-demand and spot, with prior increase requests `CASE_CLOSED`. An L40S on EC2 is not available; this is not a preference. |
| **`terminus-2`** | Model-agnostic scaffold. `claude-code`/`codex` are vendor scaffolds and would confound a Qwen baseline. |
| **No wandb** | Inference + verification only. Four JSONL logs plus harbor job dirs cover it. |
| **No teacher / Fireworks / OPD** | Nothing to distil into until the gap is measured. |

---

## 13. Cost

L40S = **$0.000542/sec = $1.95/h**, billed on wall-clock while the endpoint is up.

```
cost ≈ (rollouts / workers) × per_rollout_minutes × $1.95/h
```

4 rollouts serially. **Per-rollout time is unknown** — a prefect test suite in
Docker on 2 vCPU could be 5 minutes or 40. That is why `--max-hours 2` is set,
and why the whole point of this run is to measure it.

---

## 14. Explicitly out of scope

No teacher, no Fireworks, no OPD, no `distill.py` changes, no training, no
multi-task sweep, no wandb.

**Known gaps left open, deliberately:**

- `distill.py` has **no Modal dispatch at all**. `train.py` has `train_lora_modal`
  (SFT only); `run_opd_training` runs wherever it is invoked, and the box has no
  GPU. Blocks training-on-Modal until it exists.
- `teacher_fireworks.py` has no retry/backoff/429 handling, and `distill.py` saves
  the adapter only after the full loop (`distill.py:1122`) — one hiccup at step
  137/200 loses the run.
- `FINAL-PLAN.md`'s P0 — one echo call confirming the Fireworks teacher returns
  `token_id` per logprob entry — has **never run**. It gates all OPD work.
- The `~1.3 GiB` CUDA-overhead constant in the KV arithmetic is an **estimate**,
  not measured. Ample slack at 40960 on L40S, but it is the first term to
  distrust if a start ever fails near the boundary.
- The 48 task dirs exist on one EBS volume and nowhere else. No backup.
