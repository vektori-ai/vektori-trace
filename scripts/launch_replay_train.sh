#!/bin/bash
# Launch the replay OPD optimizer step, refusing to spend on stale code.
#
# The previous run cost an A100 and produced no adapter because a `git pull`
# was silenced inside a one-liner and the box stayed on the commit with the
# broken persist path. The pull is visible here and a mismatched HEAD is fatal.
set -euo pipefail

EXPECTED="${1:?usage: launch_replay_train.sh <expected-commit-sha> [max_trace_share]}"
SHARE="${2:-0.45}"
REPO=/data/vektori-trace

sudo -u ubuntu -H git -C "$REPO" fetch origin opd-multiturn
sudo -u ubuntu -H git -C "$REPO" merge --ff-only origin/opd-multiturn

HEAD_SHA=$(git -C "$REPO" rev-parse HEAD)
echo "HEAD=$HEAD_SHA"
if [[ "$HEAD_SHA" != "$EXPECTED"* ]]; then
  echo "REFUSING: box is at $HEAD_SHA, expected $EXPECTED" >&2
  exit 1
fi

# Prove the persist fixes are present rather than trusting the sha alone.
grep -q "output_dir=dest_dir" "$REPO/scripts/replay_train_modal.py" \
  || { echo "REFUSING: adapter does not save to the volume" >&2; exit 1; }
grep -q "MIN_ADAPTER_BYTES" "$REPO/scripts/replay_train_modal.py" \
  || { echo "REFUSING: no weights-size floor" >&2; exit 1; }
! grep -q "200 \* 1024 \* 1024" "$REPO/scripts/replay_train_modal.py" \
  || { echo "REFUSING: the 200MB filter is still present" >&2; exit 1; }
echo "persist gates present"

cd "$REPO"
sudo -u ubuntu -H setsid nohup .venv/bin/modal run scripts/replay_train_modal.py \
  --max-trace-share "$SHARE" > /data/replay_train.log 2>&1 < /dev/null &
sleep 6
pgrep -f replay_train_modal >/dev/null \
  && echo "LAUNCHED (share=$SHARE)" \
  || { echo "FAILED TO START — see /data/replay_train.log" >&2; tail -20 /data/replay_train.log >&2; exit 1; }
