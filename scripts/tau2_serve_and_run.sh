#!/bin/bash
# Serve a student endpoint, run a command against it, then ALWAYS tear down
# exactly that endpoint.
#
#   bash scripts/tau2_serve_and_run.sh --adapter <volume-path> [options] -- <cmd...>
#
# Why this exists: the rollout app is ephemeral and dies with its entrypoint,
# but the SERVING app does not -- it is a separate app in its own tmux session
# with no idea the rollout finished. Its only stop condition is --max-hours, so
# a forgotten teardown bills for hours. On 2026-08-29 that gap cost ~20 minutes
# of idle L40S between a rollout landing and a manual teardown.
#
# It stops ONLY the app id this script created, read back from --write-app-id.
# It never enumerates ephemeral apps: a blanket `modal app stop` over everything
# ephemeral would kill a concurrent unrelated job, which is a worse failure than
# the one being fixed.
#
# --max-hours stays as a second, independent backstop for the case where this
# script itself is killed hard enough to skip its own trap (SIGKILL, box reboot).
set -uo pipefail

REPO=/data/vektori-trace
BASE_MODEL=Qwen/Qwen3-4B
GPU=L40S
MAX_MODEL_LEN=24576
MAX_LORA_RANK=16
MAX_HOURS=1
ADAPTER=""
TAG="serve$$"
STATE=/data/tau2

while [ $# -gt 0 ]; do
  case "$1" in
    --adapter)        ADAPTER=$2; shift 2 ;;
    --base-model)     BASE_MODEL=$2; shift 2 ;;
    --gpu)            GPU=$2; shift 2 ;;
    --max-model-len)  MAX_MODEL_LEN=$2; shift 2 ;;
    --max-lora-rank)  MAX_LORA_RANK=$2; shift 2 ;;
    --max-hours)      MAX_HOURS=$2; shift 2 ;;
    --tag)            TAG=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -n "$ADAPTER" ] || { echo "--adapter is required" >&2; exit 2; }
[ $# -gt 0 ]      || { echo "no command given after --" >&2; exit 2; }

ENV_FILE=$STATE/${TAG}_student.env
APP_FILE=$STATE/${TAG}_app_id.txt
SERVE_LOG=$STATE/${TAG}_serve.log
GPU_LOG=$STATE/${TAG}_gpu.jsonl
mkdir -p "$STATE"
rm -f "$ENV_FILE" "$APP_FILE"

cd "$REPO" || exit 1
set -a; . ./.env; set +a
MODAL=$REPO/.venv/bin/modal

SERVE_PID=""
TORN_DOWN=0

cleanup() {
  # Runs on success, failure, and interrupt. Idempotent: EXIT fires after
  # INT/TERM, so guard against stopping twice.
  [ "$TORN_DOWN" = "1" ] && return
  TORN_DOWN=1
  local rc=${1:-$?}
  echo ""
  echo "[teardown] cleaning up (exit $rc)"

  if [ -n "$SERVE_PID" ] && kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "[teardown] killing local serve pid $SERVE_PID"
    kill "$SERVE_PID" 2>/dev/null
    for _ in 1 2 3 4 5; do
      kill -0 "$SERVE_PID" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$SERVE_PID" 2>/dev/null
  fi

  # Stop ONLY our own app id. Never enumerate ephemeral apps.
  local app_id=""
  [ -f "$APP_FILE" ] && app_id=$(grep -oE 'ap-[A-Za-z0-9]+' "$APP_FILE" | head -1)
  if [ -z "$app_id" ] && [ -f "$SERVE_LOG" ]; then
    app_id=$(grep -oE 'ap-[A-Za-z0-9]+' "$SERVE_LOG" | head -1)
  fi

  if [ -n "$app_id" ]; then
    echo "[teardown] stopping serving app $app_id"
    "$MODAL" app stop -y "$app_id" 2>&1 | tail -2
    if "$MODAL" app list 2>/dev/null | grep -q "$app_id.*ephemeral"; then
      echo "[teardown] WARNING: $app_id still listed ephemeral -- stop it by hand:"
      echo "           $MODAL app stop -y $app_id"
    else
      echo "[teardown] $app_id stopped"
    fi
  else
    echo "[teardown] WARNING: no app id found in $APP_FILE or $SERVE_LOG."
    echo "           If an endpoint started, stop it by hand:"
    echo "           $MODAL app list   # then: $MODAL app stop -y <ap-...>"
  fi
  echo "[teardown] remaining ephemeral apps (FYI, not touched):"
  "$MODAL" app list 2>/dev/null | grep -c ephemeral
}
trap 'cleanup $?' EXIT
trap 'echo "[teardown] interrupted"; cleanup 130; exit 130' INT TERM

echo "[serve] adapter   $ADAPTER"
echo "[serve] gpu       $GPU   max-hours $MAX_HOURS (backstop)"
echo "[serve] log       $SERVE_LOG"
"$REPO/.venv/bin/python" scripts/serve_student.py \
  --base-model "$BASE_MODEL" \
  --adapter "sft=$ADAPTER" \
  --gpu "$GPU" --max-model-len "$MAX_MODEL_LEN" --max-lora-rank "$MAX_LORA_RANK" \
  --reasoning-parser qwen3 \
  --write-env "$ENV_FILE" \
  --write-app-id "$APP_FILE" \
  --gpu-log "$GPU_LOG" --max-hours "$MAX_HOURS" \
  > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!

echo -n "[serve] waiting for endpoint"
READY=0
for _ in $(seq 1 180); do   # up to ~15 min; a cold model pull is ~5-10
  if grep -q STUDENT_API_BASE "$ENV_FILE" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo ""; echo "[serve] FAILED -- process exited before writing $ENV_FILE"
    tail -20 "$SERVE_LOG"
    exit 1
  fi
  echo -n "."
  sleep 5
done
echo ""
[ "$READY" = "1" ] || { echo "[serve] TIMEOUT waiting for endpoint"; tail -20 "$SERVE_LOG"; exit 1; }

set -a; . "$ENV_FILE"; set +a
echo "[serve] ready: $STUDENT_API_BASE"
echo "[run] $*"
echo "----------------------------------------"
"$@"
RC=$?
echo "----------------------------------------"
echo "[run] exit $RC"
exit $RC
