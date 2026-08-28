#!/usr/bin/env bash
# The previous watcher hardcoded a local Cursor scratchpad path that does not
# exist on the box. It is not a watcher. Use the observe-only scripts:
#   scripts/pilot_poll.sh     60s compact status
#   scripts/pilot_alert.sh    loud failure tail (serve.log + pilot_run.log)
#   scripts/pilot_health.sh   one-shot
echo "scripts/pilot_watch.sh is retired (scratchpad path). On the box:" >&2
echo "  python3 scripts/pilot_monitor.py poll     # or scripts/pilot_poll.sh" >&2
echo "  python3 scripts/pilot_monitor.py alert    # or scripts/pilot_alert.sh" >&2
echo "  python3 scripts/pilot_monitor.py health   # or scripts/pilot_health.sh" >&2
exit 2
