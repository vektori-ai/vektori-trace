#!/usr/bin/env bash
# Poll a mining run on the EC2 box and print only the lines worth acting on.
#
#   scripts/watch-mine.sh                        # commit mine (default log)
#   scripts/watch-mine.sh /data/mine-pilot.log   # PR mine
#   INSTANCE=i-0abc... INTERVAL=60 scripts/watch-mine.sh
#
# The mine runs in tmux on the box and survives everything; this watcher does
# not — it dies with the shell that started it, so re-arm it after a restart.
# Losing it costs visibility, never the run.
#
# Emits at every Nth task, on any error signature, and on completion. Silence
# is not success: a run that dies without a completion marker still reports.
#
# Reads the log directly rather than depending on a helper script living on the
# box, so it works against a fresh instance.

set -uo pipefail

INSTANCE="${INSTANCE:-i-0a348ff3d7be9769a}"
LOG="${1:-/data/cmine.log}"
INTERVAL="${INTERVAL:-120}"
STEP="${STEP:-10}"

command -v aws >/dev/null 2>&1 || {
  echo "watch-mine: aws CLI not on PATH (try PATH=\$HOME/.local/bin:\$PATH)" >&2
  exit 1
}

# One SSM round trip. Returns empty on failure so the caller retries rather
# than reading a flaky API call as a dead run.
remote() {
  local cid
  cid=$(aws ssm send-command --instance-ids "$INSTANCE" \
        --document-name AWS-RunShellScript \
        --parameters "commands=[\"$1\"]" \
        --query 'Command.CommandId' --output text 2>/dev/null) || return 1
  sleep 10
  aws ssm get-command-invocation --command-id "$cid" --instance-id "$INSTANCE" \
    --query 'StandardOutputContent' --output text 2>/dev/null
}

echo "watching $LOG on $INSTANCE (poll ${INTERVAL}s, milestone every $STEP tasks)"
last=0

while true; do
  # Both pipelines log "emitted task" once per emitted task, so one probe
  # covers pr_runtime and commit_runtime.
  out=$(remote "n=\$(grep -ac 'emitted task' $LOG 2>/dev/null); echo COUNT=\$n; grep -aE 'EXIT=|Traceback|rror:|dropped as leaky|task\\(s\\) mined' $LOG 2>/dev/null | tail -5")
  if [ -z "$out" ]; then sleep "$INTERVAL"; continue; fi

  n=$(printf '%s' "$out" | sed -n 's/^COUNT=//p' | head -1)
  [ -z "$n" ] && n=0

  if [ "$n" -ge $((last + STEP)) ]; then
    echo "milestone: $n tasks mined"
    last=$n
  fi

  # Surface errors as they appear, not only at the end — a run that is going to
  # fail should not do so quietly for another hour.
  if printf '%s' "$out" | grep -qaE 'Traceback|rror:'; then
    printf '%s\n' "$out" | grep -aE 'Traceback|rror:' | tail -2
  fi

  if printf '%s' "$out" | grep -qa 'EXIT='; then
    echo "MINE COMPLETE at $n tasks"
    # The histogram is the actual result. A small yield can mean the repo is
    # unsuitable or that a filter is wrong, and those call for opposite
    # responses — the count alone cannot tell them apart.
    remote "sed -n '/Where the/,\$p' $LOG | head -30"
    exit 0
  fi

  sleep "$INTERVAL"
done
