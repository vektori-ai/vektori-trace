#!/usr/bin/env bash
# Read-only pilot watcher. Observes; never stops, kills, or writes anything.
#
# Deliberately has no `modal app stop`, no `kill`, no writes outside /tmp:
# a watcher that can act is a watcher that can act WRONGLY at 3am. Teardown
# belongs to the orchestrator, which owns the app ids it started.
#
# usage: ./scripts/pilot_watch.sh [interval_seconds]   (default 60)

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODAL="$REPO/.venv/bin/modal"
S=/tmp/claude-1000/-home-alex-hunterz-Desktop-projects-vektori-trace/41ba1a3b-1b30-4b10-841c-d6fdea0bb9af/scratchpad
INTERVAL="${1:-60}"
RUN_ID=pilot_10x8_20260829

while true; do
  echo "════════ $(date '+%F %T') ════════"

  # GPUs actually billing right now
  echo "── running modal apps ──"
  "$MODAL" app list 2>/dev/null | awk '/ephemeral/ {print "  " $2, $6, "tasks="$8}' \
    || echo "  (modal unreachable)"
  n=$("$MODAL" app list 2>/dev/null | grep -c ephemeral)
  echo "  GPUs/containers up: ${n:-?}"

  # Endpoint state
  if [ -s "$S/pilot_env.sh" ]; then
    api=$(grep STUDENT_API_BASE "$S/pilot_env.sh" | cut -d= -f2-)
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 "$api/models" 2>/dev/null)
    echo "── endpoint: HTTP $code ──"
  else
    echo "── endpoint: still loading ──"
  fi

  # Budget ledger, if the orchestrator has started one
  if [ -f "$S/pilotstate/ledger.json" ]; then
    echo "── budget ──"
    python3 -c "
import json;d=json.load(open('$S/pilotstate/ledger.json'))
print(f\"  {d.get('hours',0):.2f}h  ~\${d.get('estimate_usd',0):.2f} of \${d.get('max_usd')}  updates={d.get('updates_done')}\")" 2>/dev/null
  fi

  # Stage progress
  if [ -f "$S/pilot_run.log" ]; then
    echo "── last stages ──"
    grep -E "=== update|rollout took|scored [0-9]+ new|trained in|endpoint now serves|complete|STOPPING" \
      "$S/pilot_run.log" 2>/dev/null | tail -4 | sed 's/^/  /'
  fi

  # Anything that looks wrong
  if [ -f "$S/pilot_run.log" ]; then
    bad=$(grep -cE "Traceback|Error|FAILED|STOPPING|!!" "$S/pilot_run.log" 2>/dev/null)
    [ "${bad:-0}" -gt 0 ] && echo "  ⚠️  $bad error line(s) in pilot_run.log"
  fi

  echo
  sleep "$INTERVAL"
done
