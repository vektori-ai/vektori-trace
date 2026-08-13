# Modal serving concurrency — why one GPU became three

**Incident: 2026-08-13. Cost: ~$3.20 in 27 minutes at 3× the intended rate.
Every GPU sweep this repo has ever run is affected, run6 included.**

## What happened

The held-out baseline sweep was launched with three processes pointed at one
vLLM endpoint:

1. rollout worker 1 (`--max-workers 2`)
2. rollout worker 2
3. `scripts/vllm_monitor.py`, polling `/metrics` every 5 s

`modal container list` showed **three containers**, each holding its own L40S
and its own 27.6 GB copy of Qwen3-14B.

## Why

`serve.py` declares the server with `@app.cls(gpu=gpu, ...)` and set no
concurrency limit. **Modal's unit of scaling is the in-flight request, and its
default is one request per container.** A second request arriving while the
first is open does not queue — it boots another container.

An agent rollout is not one request. It is a loop: one HTTP call per turn, each
taking seconds to generate. Rollout `jinja-1663-1` made 23 calls over 712 s. So
two rollouts keep two requests open essentially continuously, and the metrics
poller makes a third:

```
t=0.0s  worker 1 POST /v1/chat/completions  → container A (limit 1)
t=0.4s  worker 2 POST /v1/chat/completions  → A busy → BOOT container B
t=5.0s  monitor  GET  /metrics              → A, B busy → BOOT container C
```

The telemetry call got its own GPU. It then reported `running: 0`, because it
was measuring the idle GPU its own existence had created.

## Why this was worse than 3× cost

vLLM's continuous batching lives *inside* a container. Modal routes *above* it.
With one request per container, no engine ever received a second request, so
there was never anything to batch — three engines each ran a single sequence.

This is the explanation for a number that had been misread for weeks: run6
recorded **GPU utilisation of 1.0% mean, idle 99% of samples**, and that was
rationalised as "agentic evals are inherently latency-bound." Some of it is.
Most of it was this. The startup banner even advertised the unused capacity:

```
kv budget           14.30 GiB  =  93,716 tokens total
→ concurrent seqs   2  at full length
```

KV reserved for concurrent sequences that never arrived, on every container.

## The fix

```python
@app.cls(
    gpu=gpu,
    ...,
    max_containers=1,
)
@modal.concurrent(max_inputs=32, target_inputs=8)
class VllmServer:
```

- **`max_containers=1`** — never answer load by cloning a GPU.
- **`@modal.concurrent`** — required, and applied **to the class**, per Modal's
  docs: all methods share the container, so this covers `web_server` (the
  OpenAI API) and the `method()` calls (`health`, `gpu_stats`) alike.
- **`max_inputs=32`** — hard ceiling on requests entering one engine. Set above
  KV capacity deliberately so Modal never queues ahead of vLLM; vLLM's
  scheduler is the one that should decide what runs.
- **`target_inputs=8`** — autoscaler advice, inert while `max_containers=1`.
  It only matters if that cap is ever raised.

`max_containers=1` alone would be a regression: one container accepting one
input at a time is a queue, not a batch. Both parts are required.

### API notes (Modal 1.5.3)

- `allow_concurrent_inputs` was removed in Modal 1.0. It is **not** a valid
  `App.cls` parameter — passing it raises rather than configuring anything.
- `App.cls` also accepts a `max_inputs`, which is unrelated: it retires a
  container after N lifetime inputs. Do not confuse it with
  `modal.concurrent(max_inputs=)`.

## Where requests queue now

```
N rollout workers
      ↓ HTTP
[Modal input queue]     queues by container slot; admits up to max_inputs=32
      ↓
[vLLM scheduler]        queues by KV cache: `waiting` → `running` each pass
      ↓
    GPU
```

`vllm_metrics.jsonl`'s `running` / `waiting` fields are the check. During the
incident both read `0` for the entire run. With 8 workers they should read
`running: 8, waiting: 0`.

## `--max-model-len`, sized from measured traffic

vLLM plans how many sequences fit against this value, so it directly sets
concurrency. Across run6's 24 rollouts:

| | tokens |
|---|---:|
| median prompt | 8,310 |
| p90 | 9,408 |
| **max observed** | **11,304** |

- `8192` (the old default) sits **below the observed maximum** — real requests
  would be rejected mid-sweep.
- `40960` (used by hand in run6, and repeated when launching this baseline)
  makes the scheduler plan for 41k-token sequences and admit ~2.
- **`16384`** clears the observed max with margin and admits ~10.

## What was checked and found *not* to be a problem

- **Prefix caching.** Claimed to be off based on `total_cached_tokens: 0` in the
  agent-side metrics. That was wrong. `serve.py` pins `vllm==0.21.0`, the V1
  engine, where prefix caching is **on by default** — and removing
  `--enable-prefix-caching` does not disable it; only
  `--no-enable-prefix-caching` does. The zero is a reporting gap in litellm's
  usage field.
- **`--gpu-memory-utilization`.** Already `0.90`. Raising it would buy ~2.4 GB
  of KV (~16k tokens) and risk OOM during CUDA-graph capture *after* a
  four-minute weight load. KV is a fixed pool allocated at startup — it cannot
  grow and blow up mid-run; when it fills, vLLM preempts and recomputes. Not
  worth the trade. Left alone.
- **Quantisation.** Would double throughput by halving the weights, and changes
  the model numerically. These runs are baselines; the number has to be real
  Qwen3-14B. Rejected.

## Cost, before and after

| | before | after |
|---|---|---|
| containers for a 2-worker sweep | 3 | 1 |
| GPU rate | ~$5.85/h | $1.95/h |
| concurrent sequences per engine | 1 | ~10 |
| run6 (24 rollouts, 4.53 h) | ~$33 actual vs ~$11 recorded | — |
| projected per rollout | ~$1.40 | ~$0.10 |

`boxinfo.sh` computes cost as `uptime × one L40S rate`, so **every cost figure
this project has recorded is low by the container count.** Anything quoted
before this date should be treated as a lower bound.

## Guardrails added

`tests/test_serve_concurrency.py` asserts, at source level, that the serving
class is pinned to one container, declares input concurrency, applies the
decorator to the class rather than a method, does not use the removed
`allow_concurrent_inputs`, and keeps `--max-model-len` above the largest
observed prompt. They are source assertions because the failure is invisible
locally — it shows up only as a bill.
