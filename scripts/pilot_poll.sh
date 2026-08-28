#!/usr/bin/env bash
# Observe-only 60s status line. Never stops, kills, or writes the state dir.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  exec python3 "$HERE/pilot_monitor.py" poll --interval "$1" "${@:2}"
fi
exec python3 "$HERE/pilot_monitor.py" poll "$@"
