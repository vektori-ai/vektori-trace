#!/bin/bash
# Watch a prompt-seed probe rollout as it happens.
#
#   ./scripts/watch_prompt_seed_probe.sh            # newest run
#   ./scripts/watch_prompt_seed_probe.sh <RUN_ID>   # a specific one
#
# Two views, because they answer different questions:
#
#   pane   terminus's mirror of the agent's tmux session -- the literal terminal,
#          keystrokes going in and output coming back. This is "what is it doing".
#   probe  one JSON line per model turn -- native_json, legacy_envelope, parser
#          result, intended keystrokes. This is "is the protocol holding".
#
# The pane is the honest view of execution: text only appears there because a
# command actually ran, which the probe log cannot show (it is written between
# parsing and execution).
set -uo pipefail

ROOT=/data/prompt-seed-probe
RUN=${1:-$(ls -t "$ROOT" 2>/dev/null | grep -E '^[0-9]{8}T' | head -1)}
OUT=$ROOT/$RUN
[ -d "$OUT" ] || { echo "no run at $OUT" >&2; exit 2; }
echo "run $RUN"

# The pane appears only once harbor has built the container and started the
# agent, which is a minute or two after launch.
echo "waiting for the agent's tmux pane ..."
PANE=""
for _ in $(seq 1 120); do
  PANE=$(find "$OUT" -name "terminus_2.pane" 2>/dev/null | head -1)
  [ -n "$PANE" ] && break
  sleep 5
done

if [ -z "$PANE" ]; then
  echo "no pane yet; showing the run log instead"
  exec tail -F "$OUT/run.log"
fi

echo "pane  $PANE"
PROBE=$OUT/probe.jsonl
echo "probe $PROBE"
echo

# Per-turn verdicts alongside the terminal, prefixed so the two streams stay
# distinguishable in one scrollback.
if [ -f "$PROBE" ] || : ; then
  (
    tail -F "$PROBE" 2>/dev/null | while read -r line; do
      echo "$line" | python3 -c '
import json, sys
for raw in sys.stdin:
    try:
        r = json.loads(raw)
    except Exception:
        continue
    print(
        f"[PROBE] ep{r.get(\"episode\")} turn {r.get(\"turn\")}  "
        f"native_json={r.get(\"native_json\")}  legacy={r.get(\"legacy_envelope\")}  "
        f"cmds={r.get(\"n_commands\")}  keys={r.get(\"parsed_keystrokes\")}  "
        f"err={r.get(\"parser_error\")}",
        flush=True,
    )
'
    done
  ) &
  PROBE_TAIL=$!
  trap 'kill $PROBE_TAIL 2>/dev/null' EXIT
fi

tail -F "$PANE"
