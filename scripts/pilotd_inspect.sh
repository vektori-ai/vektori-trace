#!/bin/bash
# Read-only inspection of a live-OPD run's lineage and next-update plan.
# No spend, no GPU. Run as ubuntu (needs ~/.modal.toml and the venv).
#   bash scripts/pilotd_inspect.sh [run_id] [update]
R="${1:-pilot_10x8_20260829d}"
U="${2:-1}"
UU=$(printf "%03d" "$U")
PREV=$(printf "%03d" $((U-1)))
cd /data/vektori-trace || exit 1
set -a; . ./.env 2>/dev/null; set +a
M=.venv/bin/modal
T=$(mktemp -d)

echo "=== ephemeral apps ==="
$M app list 2>/dev/null | grep -c ephemeral

echo "=== update-$PREV checkpoint state ==="
$M volume get vektori-trace-adapters \
  "tau2/live-opd/$R/update-$PREV/checkpoint/state.json" "$T/state.json" --force >/dev/null 2>&1 \
  && .venv/bin/python -m json.tool "$T/state.json" | head -40 \
  || echo "MISSING"

echo "=== update-$PREV report (lineage) ==="
$M volume get vektori-trace-adapters \
  "tau2/live-opd/$R/update-$PREV/report.json" "$T/report.json" --force >/dev/null 2>&1 \
  && .venv/bin/python -c '
import json,sys
r=json.load(open(sys.argv[1]))
for k in ("parent_adapter_hash","child_adapter_hash","adapter_hash","loss","grad_norm",
          "weights_moved","reload_verified","n_examples","supervised_tokens",
          "retained_fraction","policy_version"):
    if k in r: print(f"  {k} = {r[k]}")
' "$T/report.json" \
  || echo "MISSING"

echo "=== update-$UU planned roster ==="
$M volume get vektori-trace-adapters \
  "tau2/live-opd/$R/update-$UU/plan.json" "$T/plan.json" --force >/dev/null 2>&1 \
  && .venv/bin/python -m json.tool "$T/plan.json" | head -60 \
  || echo "no update-$UU/plan.json (not yet PLANNED)"

echo "=== manifest: schedule for update $U ==="
$M volume get vektori-trace-adapters "tau2/live-opd/$R/manifest.json" "$T/mf.json" --force >/dev/null 2>&1 \
  && .venv/bin/python -c '
import json,sys
m=json.load(open(sys.argv[1])); u=int(sys.argv[2])
print("  plan_hash:", m.get("plan_hash"))
print("  top-level keys:", sorted(m.keys()))
sched = m.get("schedule") or m.get("updates") or m.get("plan")
if isinstance(sched, list) and u < len(sched):
    print(f"  update {u} pairs:", json.dumps(sched[u])[:600])
else:
    print("  schedule shape:", type(sched).__name__)
' "$T/mf.json" "$U" \
  || echo "MISSING"

echo "=== stage markers ==="
$M volume ls vektori-trace-adapters "tau2/live-opd/$R" 2>/dev/null | tail -15
rm -rf "$T"
