#!/bin/bash
# Read-only progress for a staged live-OPD pilot. No spend, no GPU.
#
# Run AS ubuntu on the box (it needs ~/.modal.toml and the venv):
#   bash scripts/pilotc_status.sh [run_id]
# There is no `su` here on purpose -- an earlier version called `su - ubuntu`
# unconditionally, which prompts for a password when you are already ubuntu.
R="${1:-pilot_10x8_20260829c}"
S=/data/tau2/pilotc-state
cd /data/vektori-trace || exit 1
set -a; . ./.env 2>/dev/null; set +a
M=.venv/bin/modal

echo "=== billing ==="
echo "ephemeral apps: $($M app list 2>/dev/null | grep -c ephemeral)"
tmux ls 2>&1 | sed 's/^/tmux: /'

echo
echo "=== run $R ==="
mkdir -p "$S"
for u in 0 1 2 3 4 5 6 7 8 9; do
  uu=$(printf "%03d" "$u")
  f="$S/u${u}_episodes.jsonl"
  $M volume get vektori-trace-adapters \
     "tau2/live-opd/$R/update-$uu/live_archive/episodes.jsonl" "$f" --force \
     >/dev/null 2>&1 || continue
  [ -s "$f" ] || continue
  echo "-- update $u --"
  .venv/bin/python - "$f" <<'PY'
import json, sys
seen = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    seen[r["episode_id"]] = r          # last row per episode wins
for e, r in sorted(seen.items()):
    print(f'  {e:26s} {r["status"]:10s} turns={r["num_turns"]:<3} '
          f'reward={r.get("reward")}')
ok  = sum(1 for r in seen.values() if r["status"] == "sampled")
fmt = sum(1 for r in seen.values() if r["status"] == "failed")
inf = sum(1 for r in seen.values() if r["status"] == "discarded")
print(f'  -> {ok}/{len(seen)} sampled   format-failed={fmt}   infra-discarded={inf}')
PY
done

echo
echo "=== stage markers ==="
$M volume ls vektori-trace-adapters "tau2/live-opd/$R" 2>/dev/null | tail -12

echo
echo "=== latest log ==="
ls -t "$S"/*.log 2>/dev/null | head -1 | xargs -r tail -3
