#!/usr/bin/env bash
# Phase 0.5 — token capture. See docs/PILOT.md.
#
# Does not start a GPU or deploy a model. It documents / exercises the plumbing
# that must sit in front of any OPD or GRPO run:
#
#   1. litellm / harbor request `return_token_ids: true`
#   2. a capture proxy injects the flag for agents we do not control
#   3. sampled ids land next to the harbor job dir as token_captures.jsonl
#   4. dataset.tokenize_from_ids consumes those ids (no re-tokenization)
#
# Usage:
#   UPSTREAM=http://127.0.0.1:8000/v1 ./scripts/phase05.sh   # start the proxy
#   DRY=1 ./scripts/phase05.sh                               # print only

set -euo pipefail

OUT="${OUT:-./vektori-out/token-captures}"
UPSTREAM="${UPSTREAM:-http://127.0.0.1:8000/v1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-0}"
DRY="${DRY:-0}"

run() {
  echo
  echo "--- $*"
  if [ "$DRY" = "1" ]; then return 0; fi
  "$@"
}

echo "Phase 0.5 — token capture"
echo "  upstream: $UPSTREAM"
echo "  captures: $OUT"
echo
echo "Point harbor's api_base at the printed proxy URL, then collect rollouts"
echo "with --capture-tokens (run-arms) or collect_rollouts(..., capture_tokens=True)."
echo "OPD must call tokenize_rollouts_for_opd — it refuses if captures are missing."
echo

run uv run vektori-trace capture-proxy \
  --upstream "$UPSTREAM" \
  --out "$OUT" \
  --host "$HOST" \
  --port "$PORT"
