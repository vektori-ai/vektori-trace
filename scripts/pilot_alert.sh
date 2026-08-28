#!/usr/bin/env bash
# Observe-only failure tail (serve.log + pilot_run.log). Never update-NNN.log.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/pilot_monitor.py" alert "$@"
