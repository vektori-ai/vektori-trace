#!/bin/bash
# One guarded rollout of agronholm__anyio-1121 against repaired ck63, with a
# single native-JSON demonstration seeded ahead of turn 1.
#
# This is a launch wrapper, not a shim: it stages a task, boots an endpoint, runs
# harbor, grades the log and tears the endpoint down. Nothing here sits between
# harbor and vLLM, and no request or completion is touched.
#
#   ./scripts/run_prompt_seed_probe.sh
#
# Costs one L40S for the duration. Teardown is an EXIT trap, so it fires on a
# failed sweep and on a boot that never yields a URL, not only on success.
set -euo pipefail

cd /data/vektori-trace
[ -f .env ] && set -a && . ./.env && set +a

TASK=agronholm__anyio-1121
# Per-repo layout. `cs/corpus50_v3/anyio` does not exist and would silently
# stage nothing, which is the failure mode CLAUDE.md calls out by name.
CORPUS=/data/vektori-trace/cs/corpus50_v3/agronholm_anyio/mined_tasks
CKPT=/adapters/sft/qwen3-14b-dsv4-lora-repaired/checkpoint-63
SERVED=Qwen3-14B-ck63

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
OUT=/data/prompt-seed-probe/$RUN_ID
PROBE_LOG=$OUT/probe.jsonl
STAGE=/tmp/seedprobe-$RUN_ID

# Agent cap 12 min, outer cap 15. A clean timeout is inconclusive about strategy,
# not a model failure -- passk records timed_out and excludes it from the
# denominator rather than scoring it as a loss.
AGENT_TIMEOUT_SEC=${AGENT_TIMEOUT_SEC:-720}
OUTER_TIMEOUT_SEC=${OUTER_TIMEOUT_SEC:-900}

mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "== run $RUN_ID =="
echo "task      $TASK"
echo "checkpoint $CKPT"
echo "out       $OUT"

# --- 1. preflight: never boot a GPU on an unverified prompt geometry ----------
echo
echo "== preflight =="
uv run python scripts/prompt_seed_probe.py --preflight --out-dir "$OUT"
echo "preflight OK"

# --- 2. stage the task -------------------------------------------------------
echo
echo "== staging =="
if [ ! -d "$CORPUS/$TASK" ]; then
  echo "STOP: $CORPUS/$TASK does not exist" >&2
  exit 2
fi
rm -rf "$STAGE" && mkdir -p "$STAGE"
cp -r "$CORPUS/$TASK" "$STAGE/"
grep -q tmux "$STAGE/$TASK/environment/Dockerfile" \
  || { echo "STOP: no tmux in the task Dockerfile; terminus-2 cannot act" >&2; exit 2; }
echo "staged $(ls "$STAGE")"

# --- 3. serve ck63 -----------------------------------------------------------
# L40S is serving hardware and this is serving: BF16 14.8B is ~27.6 GiB of 48,
# and at concurrency 1 the ~93.7k-token KV budget comfortably holds one 40k
# sequence. --max-lora-rank 32 is mandatory; vLLM defaults to 16 and refuses to
# start *after* the GPU is allocated.
echo
echo "== serving =="
ENV_FILE=$OUT/endpoint.env
teardown() {
  echo
  echo "== teardown =="
  modal app stop vektori-trace-serve-student 2>&1 | tail -5 || true
  modal app list 2>&1 | grep -i "serve-student" || echo "no serve-student app listed"
}
trap teardown EXIT

uv run python scripts/serve_student.py \
  --gpu L40S \
  --base-model Qwen/Qwen3-14B \
  --adapter "$SERVED=$CKPT" \
  --max-lora-rank 32 \
  --max-loras 1 \
  --max-model-len 40960 \
  --max-hours 0.5 \
  --write-env "$ENV_FILE" \
  --gpu-log "$OUT/gpu_log.jsonl" &
SERVE_PID=$!

echo "waiting for endpoint (pid $SERVE_PID) ..."
for _ in $(seq 1 90); do
  [ -f "$ENV_FILE" ] && break
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "STOP: serve died before writing $ENV_FILE" >&2; exit 3; }
  sleep 10
done
[ -f "$ENV_FILE" ] || { echo "STOP: no endpoint after 15 min" >&2; exit 3; }
set -a && . "$ENV_FILE" && set +a
API_BASE="${STUDENT_API_BASE:?no STUDENT_API_BASE in $ENV_FILE}"
echo "endpoint $API_BASE"

# --- 4. smoke: the adapter is actually registered and selectable --------------
echo
echo "== smoke =="
curl -sS "$API_BASE/models" | tee "$OUT/models.json" | head -c 800
echo
grep -q "$SERVED" "$OUT/models.json" \
  || { echo "STOP: $SERVED not in /v1/models -- LoRA registration failed" >&2; exit 4; }
echo "adapter $SERVED registered"

# --- 5. one guarded rollout --------------------------------------------------
# The import path must reach harbor verbatim; validity.py only hyphenates bare
# agent names, and a colon exempts this form.
echo
echo "== rollout =="
export PROMPT_SEED_PROBE_LOG="$PROBE_LOG"
export PYTHONPATH=/data/vektori-trace:${PYTHONPATH:-}
set +e
timeout "$OUTER_TIMEOUT_SEC" uv run vektori-trace passk \
  --tasks-dir "$STAGE" \
  --agent scripts.prompt_seed_probe:Terminus2PromptSeed \
  --model "hosted_vllm/$SERVED" \
  --api-base "$API_BASE" \
  --model-info @/data/vektori-trace/model_info.json \
  --stage1-n 1 \
  --no-escalate \
  --max-workers 1 \
  --timeout-sec "$AGENT_TIMEOUT_SEC" \
  --out "$OUT"
SWEEP_EXIT=$?
set -e
echo "SWEEP_EXIT=$SWEEP_EXIT"

# --- 6. grade ----------------------------------------------------------------
echo
echo "== protocol gate =="
if [ -f "$PROBE_LOG" ]; then
  set +e
  uv run python scripts/prompt_seed_probe.py --summarize "$PROBE_LOG"
  GATE_EXIT=$?
  set -e
  echo "GATE_EXIT=$GATE_EXIT"
else
  echo "no probe log at $PROBE_LOG -- the agent never produced a completion"
fi

echo
echo "== verifier =="
# passk's own accounting. Check no_gradeable_rollouts / infra_failures and the
# per-rollout parse_status before trusting any rate; a verifier that crashed
# before collecting tests is not a model failure.
[ -f "$OUT/passk.json" ] && head -c 2000 "$OUT/passk.json" || echo "no passk.json"

echo
echo "== done: $OUT =="
