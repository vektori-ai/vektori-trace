#!/bin/bash
# Progress for pilot_10x8_20260829c. Read-only, no spend.
#   ssm 'bash /data/vektori-trace/scripts/pilotc_status.sh'
R=pilot_10x8_20260829c
S=/data/tau2/pilotc-state
cd /data/vektori-trace 2>/dev/null || exit 1

echo "=== GPU / billing ==="
su - ubuntu -c "cd /data/vektori-trace && set -a && . ./.env && set +a && .venv/bin/modal app list 2>/dev/null | grep -c ephemeral" | sed 's/^/ephemeral apps: /'
su - ubuntu -c "tmux ls" 2>&1 | sed 's/^/tmux: /'

echo
echo "=== episodes (status, turns, reward) ==="
for u in 0 1 2 3 4 5 6 7 8 9; do
  f=$(printf "%s/u%d_episodes.jsonl" "$S" "$u")
  su - ubuntu -c "cd /data/vektori-trace && set -a && . ./.env && set +a && .venv/bin/modal volume get vektori-trace-adapters tau2/live-opd/$R/update-$(printf %03d $u)/live_archive/episodes.jsonl $f --force" >/dev/null 2>&1 || continue
  [ -s "$f" ] || continue
  echo "-- update $u --"
  python3 - "$f" <<'PY'
import json,sys
seen={}
for line in open(sys.argv[1]):
    r=json.loads(line); seen[r["episode_id"]]=r
for e,r in seen.items():
    print(f'  {e:26s} {r["status"]:10s} turns={r["num_turns"]:<3} reward={r.get("reward")}')
done=[r for r in seen.values() if r["status"]=="sampled"]
print(f'  -> {len(done)}/{len(seen)} sampled')
PY
done

echo
echo "=== stage markers ==="
su - ubuntu -c "cd /data/vektori-trace && set -a && . ./.env && set +a && .venv/bin/modal volume ls vektori-trace-adapters tau2/live-opd/$R 2>/dev/null" | tail -12

echo
echo "=== latest log ==="
ls -t $S/*.log 2>/dev/null | head -1 | xargs -r tail -3
