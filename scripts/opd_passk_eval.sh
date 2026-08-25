#!/bin/bash
# Two-arm pass@k for v0 vs v_replay. Owns one serve_student child and one
# Modal app, and stops both the moment the sweeps end -- however they end.
#
# The 2026-08-22 run cost ~$8 of idle L40S because the sweep exited at 12:35
# and nothing tore the endpoint down until 16:36. Teardown was manual and
# gated on a human noticing. This wrapper removes the human from that path:
# the trap fires on normal exit, on error, and on every signal, so the GPU
# cannot outlive the work by more than the few seconds cleanup takes.
set -Eeuo pipefail

cd /data/vektori-trace
export PATH=/data/vektori-trace/.venv/bin:$PATH   # harbor lives here
export PYTHONUNBUFFERED=1

OUT=/data/opd-passk
TASKS="$OUT/tasks"
MODEL_INFO="$OUT/model_info.json"
MODAL=/data/vektori-trace/.venv/bin/modal
ENV_FILE="$OUT/endpoint.env"
APP_ID_FILE="$OUT/app_id.txt"
SERVE_PID=""
CLEANED=0

# Settings the last run proved necessary:
#   workers 1  -- vLLM serialises anyway (Running:1/Waiting:1); a second
#                 worker only burns the queued rollout's wall clock.
#   3600 s     -- at ~20 tok/s with thinking on, 1800 killed 5 of 6 rollouts
#                 before the model had a chance to finish.
#   max-hours 2 -- a dead-man switch belongs just above the estimate, not 4x it.
N=${N:-2}
WORKERS=${WORKERS:-1}
TIMEOUT_SEC=${TIMEOUT_SEC:-3600}
MAX_HOURS=${MAX_HOURS:-2}

mkdir -p "$OUT"
rm -f "$ENV_FILE" "$APP_ID_FILE"

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  [ "$CLEANED" -eq 1 ] && exit "$rc"
  CLEANED=1
  echo "=== teardown (exit $rc) $(date -Is) ==="

  if [ -n "$SERVE_PID" ] && kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "stopping serve pid $SERVE_PID"
    kill -TERM "$SERVE_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVE_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
  fi

  # Only ever stop the app this process created. Never diff app-list.
  local app_id=""
  [ -s "$APP_ID_FILE" ] && IFS= read -r app_id < "$APP_ID_FILE"
  if [[ "$app_id" =~ ^ap-[A-Za-z0-9]+$ ]]; then
    echo "stopping owned Modal app: $app_id"
    "$MODAL" app stop "$app_id" --yes 2>&1 | tail -2 || true
  else
    echo "no app ID recorded"
  fi

  # NOTE: this deliberately does not stop containers.
  #
  # It used to run `docker ps -q | xargs -r docker stop`, three lines below a
  # comment about *owned* resources -- which stops every container on the host,
  # including ones this script never created and other people's work on a shared
  # box. It ran from an EXIT trap, so it fired on every exit path including a
  # clean one. Since the script never records the container IDs it owns, there
  # is nothing it can safely stop; killing by ownership would require capturing
  # those IDs at creation time, which happens inside harbor, not here.
  sleep 3
  echo "--- containers (must be empty) ---"
  "$MODAL" container list 2>&1 | cut -c1-60 || true
  echo "DONE rc=$rc $(date -Is)"
  exit "$rc"
}
on_signal() { exit "$((128 + $1))"; }
trap cleanup EXIT
trap 'on_signal 2' INT
trap 'on_signal 15' TERM
trap 'on_signal 1' HUP

echo "=== preflight $(date -Is) ==="
git -C /data/vektori-trace log --oneline -1 || true
[ -d "$TASKS" ] || { echo "FATAL: $TASKS missing" >&2; exit 2; }
[ -s "$MODEL_INFO" ] || { echo "FATAL: $MODEL_INFO missing" >&2; exit 2; }
command -v harbor >/dev/null || { echo "FATAL: harbor not on PATH" >&2; exit 2; }
for d in "$TASKS"/*/; do
  grep -qi tmux "$d/environment/Dockerfile" || {
    echo "FATAL: no tmux in $(basename "$d")" >&2; exit 2; }
done
echo "tasks: $(ls -1 "$TASKS" | tr '\n' ' ')"
echo "n=$N workers=$WORKERS timeout=${TIMEOUT_SEC}s max-hours=$MAX_HOURS"

echo "=== serving $(date -Is) ==="
.venv/bin/python scripts/serve_student.py \
  --gpu L40S --base-model Qwen/Qwen3-14B \
  --adapter v0=/adapters/sft/qwen3-14b-stage-b-lora/checkpoint-75 \
  --adapter v_replay=/adapters/opd/v_replay \
  --max-lora-rank 32 --max-loras 2 --max-model-len 40960 \
  --reasoning-parser qwen3 --max-hours "$MAX_HOURS" \
  --write-env "$ENV_FILE" --write-app-id "$APP_ID_FILE" \
  --gpu-log "$OUT/gpu_log.jsonl" \
  > "$OUT/serve.log" 2>&1 &
SERVE_PID=$!
echo "serve pid: $SERVE_PID"

echo "waiting for endpoint (~6 min expected, 15 min ceiling)..."
for _ in $(seq 1 90); do
  grep -q STUDENT_API_BASE "$ENV_FILE" 2>/dev/null && break
  grep -q "refusing to report an endpoint" "$OUT/serve.log" 2>/dev/null && {
    echo "SERVE FAILED SMOKE TEST" >&2; exit 3; }
  kill -0 "$SERVE_PID" 2>/dev/null || {
    echo "serve died:" >&2; tail -20 "$OUT/serve.log" >&2; exit 6; }
  sleep 10
done
grep -q STUDENT_API_BASE "$ENV_FILE" 2>/dev/null || {
  echo "endpoint never came up" >&2; tail -30 "$OUT/serve.log" >&2; exit 4; }
set -a; . "$ENV_FILE"; set +a
echo "=== endpoint up: $STUDENT_API_BASE ==="

# Both adapters must be addressable, or an arm silently hits the wrong weights.
for m in Qwen3-14B-v0 Qwen3-14B-v_replay; do
  curl -s --max-time 30 "$STUDENT_API_BASE/models" | grep -q "\"$m\"" || {
    echo "FATAL: $m not registered on the endpoint" >&2; exit 7; }
  echo "registered: $m"
done

run_arm() {
  local label="$1"
  echo "=== arm $label $(date -Is) ==="
  set +e
  .venv/bin/vektori-trace passk \
    --tasks-dir "$TASKS" --agent terminus-2 \
    --model "hosted_vllm/Qwen3-14B-$label" \
    --api-base "$STUDENT_API_BASE" \
    --model-info "@$MODEL_INFO" \
    --stage1-n "$N" --no-escalate \
    --max-workers "$WORKERS" --timeout-sec "$TIMEOUT_SEC" \
    --out "$OUT/$label" 2>&1 | tee "$OUT/$label.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "=== arm $label finished rc=$rc $(date -Is) ==="
  # Record the failure instead of swallowing it. Returning 0 here -- so that a
  # failed arm could not skip the other arm's teardown -- also meant BOTH arms
  # could fail and the wrapper still exit 0, reporting success for a run that
  # produced nothing.
  if [[ "$rc" -ne 0 ]]; then
    ARM_FAILURES+=("$label(rc=$rc)")
  fi
  return 0
}

# v_replay first, then v0 -- both on this one endpoint, one session, so the
# comparison is attributable to the adapter and nothing else.
ARM_FAILURES=()
run_arm v_replay
run_arm v0

if [[ ${#ARM_FAILURES[@]} -gt 0 ]]; then
  echo "FATAL: ${#ARM_FAILURES[@]} arm(s) failed: ${ARM_FAILURES[*]}" >&2
  echo "The comparison needs both arms; a partial result is not a result." >&2
  exit 8
fi

echo "=== both arms done $(date -Is) ==="
for a in v_replay v0; do
  echo "--- $a ---"
  grep -E "trial task=|exceeded timeout" "$OUT/$a.log" 2>/dev/null | tail -20 || true
done
# cleanup runs here via the EXIT trap -- the GPU stops with the work.
