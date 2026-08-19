#!/bin/bash
# Stage B eval driver. Owns one serve_student child and one exact Modal app.
set -Eeuo pipefail

cd /data/vektori-trace
export PATH=/data/vektori-trace/.venv/bin:$PATH
export PYTHONUNBUFFERED=1

OUT=/data/phase7-stage-b
CK=/adapters/sft/qwen3-14b-stage-b-lora
MODAL=/data/vektori-trace/.venv/bin/modal
ENV_FILE="$OUT/serve.env"
APP_ID_FILE="$OUT/app_id.txt"
SERVE_PID=""
CLEANED=0

mkdir -p "$OUT"
rm -f "$ENV_FILE" "$APP_ID_FILE"

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [ "$CLEANED" -eq 1 ]; then
    exit "$rc"
  fi
  CLEANED=1

  echo "=== teardown (exit $rc) $(date -Is) ==="

  # Stop only the child started by this wrapper. SIGTERM lets serve_student
  # unwind app.run() normally; SIGKILL is only a bounded fallback.
  if [ -n "$SERVE_PID" ] && kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "stopping serve pid $SERVE_PID"
    kill -TERM "$SERVE_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVE_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$SERVE_PID" 2>/dev/null; then
      echo "serve pid $SERVE_PID did not exit in 30s; killing it" >&2
      kill -KILL "$SERVE_PID" 2>/dev/null || true
    fi
    wait "$SERVE_PID" 2>/dev/null || true
  fi

  # Belt-and-braces cleanup targets only the app ID recorded by this process.
  # Never infer ownership from an account-wide before/after app-list diff.
  local app_id=""
  if [ -s "$APP_ID_FILE" ]; then
    IFS= read -r app_id < "$APP_ID_FILE"
  fi
  if [[ "$app_id" =~ ^ap-[A-Za-z0-9]+$ ]]; then
    echo "stopping owned Modal app: $app_id"
    "$MODAL" app stop "$app_id" --yes 2>&1 | tail -2 || true
  elif [ -n "$app_id" ]; then
    echo "WARNING: refusing malformed app id: $app_id" >&2
  else
    echo "no app ID was created/recorded"
  fi

  sleep 5
  echo "--- final Modal app state ---"
  "$MODAL" app list 2>&1 | tail -12 || true
  echo "DONE rc=$rc $(date -Is)"
  exit "$rc"
}

on_signal() {
  local signal_number=$1
  exit "$((128 + signal_number))"
}

trap cleanup EXIT
trap 'on_signal 2' INT
trap 'on_signal 15' TERM
trap 'on_signal 1' HUP

echo "=== preflight $(date -Is) ==="
git -C /data/vektori-trace log --oneline -1
.venv/bin/python scripts/phase7_eval.py --help 2>&1 | grep -q "selection-policy" || {
  echo "FATAL: this checkout has no --selection-policy" >&2
  exit 2
}
.venv/bin/python scripts/serve_student.py --help 2>&1 | grep -q "write-app-id" || {
  echo "FATAL: serve_student.py has no --write-app-id" >&2
  exit 2
}
echo "selection policy and exact app ownership support present"

echo "=== serving $(date -Is) ==="
.venv/bin/python scripts/serve_student.py \
  --base-model Qwen/Qwen3-14B \
  --adapter ckB25="$CK/checkpoint-25" \
  --adapter ckB50="$CK/checkpoint-50" \
  --adapter ckB75="$CK/checkpoint-75" \
  --adapter ckB93="$CK/checkpoint-93" \
  --max-lora-rank 32 --gpu L40S --max-model-len 40960 --max-hours 3 \
  --write-env "$ENV_FILE" --write-app-id "$APP_ID_FILE" \
  > "$OUT/serve.log" 2>&1 &
SERVE_PID=$!
echo "serve pid: $SERVE_PID"

echo "waiting for endpoint (cache is warm; ~4 min expected, 15 min ceiling)..."
for _ in $(seq 1 90); do
  grep -q STUDENT_API_BASE "$ENV_FILE" 2>/dev/null && break
  grep -q "refusing to report an endpoint" "$OUT/serve.log" 2>/dev/null && {
    echo "SERVE FAILED SMOKE TEST" >&2
    exit 3
  }
  kill -0 "$SERVE_PID" 2>/dev/null || {
    echo "serve process died; last log:" >&2
    tail -20 "$OUT/serve.log" >&2
    exit 6
  }
  sleep 10
done
grep -q STUDENT_API_BASE "$ENV_FILE" 2>/dev/null || {
  echo "endpoint never came up" >&2
  tail -30 "$OUT/serve.log" >&2
  exit 4
}

set -a
. "$ENV_FILE"
set +a
MODEL="${STUDENT_MODEL:-}"
[ -n "$MODEL" ] || {
  echo "no STUDENT_MODEL in serve.env" >&2
  exit 5
}
echo "=== endpoint up: $STUDENT_API_BASE  model=$MODEL ==="

echo "=== eval $(date -Is) ==="
set +e
.venv/bin/python scripts/phase7_eval.py \
  --manifest /data/phase7-stage-a/manifest.json \
  --api-base "$STUDENT_API_BASE" \
  --checkpoints ckB25=Qwen3-14B-ckB25 ckB50=Qwen3-14B-ckB50 ckB75=Qwen3-14B-ckB75 ckB93=Qwen3-14B-ckB93 \
  --selection-policy stage-b --strategy staged \
  --max-tokens 4096 \
  --out "$OUT/results.json" 2>&1 | tee "$OUT/eval.log"
EVAL_RC=${PIPESTATUS[0]}
set -e
echo "EVAL_EXIT=$EVAL_RC $(date -Is)"
exit "$EVAL_RC"
