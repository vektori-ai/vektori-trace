#!/usr/bin/env bash
# Observe-only one-shot health check. Exit 0 healthy, 1 warn, 2 bad.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/pilot_monitor.py" health "$@"
