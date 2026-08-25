#!/usr/bin/env bash
# Poll a running Modal app and report progress. Read-only: it never kills.
#
# `modal app logs <app>` STREAMS while the app is running, so calling it inside
# a command substitution would block forever and the loop would never tick.
# Every log read here is wrapped in `timeout`, which bounds the stream and gives
# us whatever arrived in that window.
#
# It does not terminate anything on its own. One optimizer step is 8 sequential
# microbatches over rows up to 13k tokens, and a slow step is not a hang -- the
# judgement stays with a human.
#
#   scripts/tau2_watchdog.sh <APP_NAME_OR_ID> [INTERVAL_SEC] [LOG_WINDOW_SEC]
set -uo pipefail
APP="${1:?usage: tau2_watchdog.sh <APP_NAME_OR_ID> [interval] [window]}"
INTERVAL="${2:-300}"
WINDOW="${3:-20}"
MODAL="${MODAL:-.venv/bin/modal}"

# Measured on the probe: 17.6 s/optimizer step. Five minutes is ~17 missed
# steps, which is worth a human look rather than an automatic kill.
STEP_SECONDS=17.6
LAST=""
STALE=0

# `modal run` creates an EPHEMERAL app, which `modal app logs` cannot resolve
# by name -- only by id. Fail loudly here rather than reporting "no step line
# yet" forever while training runs perfectly well.
if timeout 15 $MODAL app logs "$APP" 2>&1 | grep -q "No App with name"; then
  echo "ERROR: '$APP' is not resolvable by name (ephemeral apps need their id)."
  echo "       find it with: $MODAL app list | grep ephemeral"
  exit 3
fi

echo "watchdog: app=$APP interval=${INTERVAL}s window=${WINDOW}s"
echo "  (read-only; ~$(echo "$INTERVAL/$STEP_SECONDS" | bc) steps expected per interval)"

while true; do
  TS=$(date +%H:%M:%S)
  LOG=$(timeout "$WINDOW" $MODAL app logs "$APP" 2>/dev/null | tail -300)
  # Strip the progress bar's ANSI codes, then read each field separately.
  # Step, progress fraction and loss arrive on DIFFERENT lines -- TRL's bar
  # writes "N/105", our flushed line writes "step N", the metrics dict writes
  # "'loss': 'X'". A pattern requiring them on one line matches nothing, which
  # is how this reported "no step line yet" through 25 healthy steps.
  CLEAN=$(echo "$LOG" | sed 's/\x1b\[[0-9;]*[A-Za-z]//g')
  PROG=$(echo "$CLEAN" | grep -oE '[0-9]+/[0-9]+ \[' | tail -1 | tr -d ' [')
  LOSS=$(echo "$CLEAN" | grep -oE "'loss': '[0-9.]+'" | tail -1 | grep -oE '[0-9.]+$|[0-9]+\.[0-9]+')
  STEP="${PROG:-}${LOSS:+ loss ${LOSS}}"
  VRAM=$(echo "$CLEAN" | grep -oE 'vram [0-9.]+G' | tail -1)
  CKPT=$(echo "$CLEAN" | grep -cE 'checkpoint step' || true)

  if echo "$CLEAN" | grep -qE "App completed|Runner terminated|Stopping app"; then
    echo "[$TS] app finished"
    echo "$CLEAN" | tail -20
    exit 0
  fi
  if echo "$CLEAN" | grep -qE "GATE FAILED|FAILURE recorded|CUDA out of memory|Traceback"; then
    echo "[$TS] *** FAILURE SIGNAL IN LOG ***"
    echo "$CLEAN" | grep -E "GATE FAILED|FAILURE recorded|out of memory|Error" | tail -8
    echo "[$TS] logs and checkpoints are on the volume; not terminating from here"
    exit 2
  fi

  if [ -n "$STEP" ] && [ "$STEP" = "$LAST" ]; then
    STALE=$((STALE + 1))
    MIN=$((STALE * INTERVAL / 60))
    echo "[$TS] no new step for ${MIN} min | last: ${STEP} ${VRAM} | ckpts: ${CKPT}"
    [ "$STALE" -ge 2 ] && echo "[$TS] *** ${MIN} min without progress — worth a look ***"
  elif [ -z "$STEP" ]; then
    echo "[$TS] no step line yet (loading model / building?) ${VRAM}"
  else
    STALE=0
    echo "[$TS] ${STEP} ${VRAM} | ckpts: ${CKPT}"
  fi
  LAST="$STEP"
  sleep "$INTERVAL"
done
