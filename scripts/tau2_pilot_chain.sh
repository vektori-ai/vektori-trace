#!/bin/bash
# Run updates N..M of a live-OPD pilot end to end, halting on any stop rule.
#
#   bash scripts/tau2_pilot_chain.sh <run_id> <first_update> <last_update>
#
# Per update:
#   serve(prev checkpoint) -> rollout -> teardown -> dry-run -> verify
#     -> score -> train -> verify checkpoint -> next
#
# Every rollout samples from the checkpoint the PREVIOUS update produced, which
# is what keeps the arm on-policy. The child hash is read off the parent's
# state.json and asserted with --adapter-hash-expect, so a wrong-parent rollout
# fails loudly instead of silently producing off-policy data.
#
# The serving endpoint is torn down after EVERY rollout via
# tau2_serve_and_run.sh's EXIT trap -- it stops only the app id it created.
set -uo pipefail

RUN=${1:?run_id}
FIRST=${2:?first update}
LAST=${3:?last update}

REPO=/data/vektori-trace
STATE=/data/tau2/pilotd-state
VOL=vektori-trace-adapters
RUNS=tau2/live-opd
LR=1e-5

cd "$REPO" || exit 1
set -a; . ./.env; set +a
M=$REPO/.venv/bin/modal
PY=$REPO/.venv/bin/python
mkdir -p "$STATE"

halt() { echo ""; echo "!!!! HALT at update $1: $2"; echo "     see $3"; exit 1; }

vget() { # vget <volume-path> <local>
  $M volume get "$VOL" "$1" "$2" --force >/dev/null 2>&1
}

for U in $(seq "$FIRST" "$LAST"); do
  P=$((U-1))
  UU=$(printf "%03d" "$U"); PP=$(printf "%03d" "$P")
  echo ""
  echo "================ UPDATE $U ================"
  date -u +"start %H:%M:%SZ"

  # --- parent checkpoint hash: what this rollout must sample from ----------
  vget "$RUNS/$RUN/update-$PP/checkpoint/state.json" /tmp/parent_state.json \
    || halt "$U" "parent update-$PP has no checkpoint" "volume"
  PHASH=$($PY -c 'import json,sys; print(json.load(open("/tmp/parent_state.json"))["adapter_hash"])')
  [ -n "$PHASH" ] || halt "$U" "could not read parent adapter hash" /tmp/parent_state.json
  echo "parent checkpoint: update-$PP  hash $PHASH"

  # --- rollout, with the endpoint torn down on every exit path ------------
  RLOG=$STATE/u${U}_rollout.log
  bash scripts/tau2_serve_and_run.sh \
      --adapter "/adapters/$RUNS/$RUN/update-$PP/checkpoint" \
      --tag "chain_u$U" --max-hours 1 \
      -- $M run scripts/tau2_live_opd_modal.py::rollout_only \
           --run-id "$RUN" --update "$U" \
           --api-base "\$STUDENT_API_BASE" --student-model Qwen3-4B-sft \
           --adapter-hash-expect "$PHASH" \
      > "$RLOG" 2>&1
  RC=$?
  grep -aE "rollout took|episodes" "$RLOG" | tail -2
  [ $RC -eq 0 ] || halt "$U" "rollout failed (exit $RC)" "$RLOG"

  # --- free dry-run: retention, accounting, structural weight --------------
  DLOG=$STATE/u${U}_dryrun.log
  timeout 900 $M run scripts/tau2_live_opd_modal.py::scoring_dryrun \
      --run-id "$RUN" --update "$U" > "$DLOG" 2>&1
  RET=$(grep -aoE 'retained: [0-9.]+' "$DLOG" | tail -1 | grep -oE '[0-9.]+')
  echo "retention: $RET"
  [ -n "$RET" ] || halt "$U" "dry-run produced no retention figure" "$DLOG"
  $PY -c "import sys; sys.exit(0 if float('$RET') >= 0.90 else 1)" \
    || halt "$U" "retention $RET < 0.90" "$DLOG"
  grep -aq "no index-1 boundary newline : True" "$DLOG" \
    || halt "$U" "boundary-newline weight present" "$DLOG"
  grep -aq "token accounting complete   : True" "$DLOG" \
    || halt "$U" "token accounting incomplete" "$DLOG"

  # --- exclusions must match the DECLARED policy, not merely be few -------
  vget "$RUNS/$RUN/update-$UU/actions.jsonl" /tmp/chain_actions.jsonl
  vget "$RUNS/$RUN/manifest.json" /tmp/chain_manifest.json
  ELOG=$STATE/u${U}_exclusions.log
  $PY scripts/pilotd_verify_exclusions.py /tmp/chain_manifest.json \
      /tmp/chain_actions.jsonl > "$ELOG" 2>&1
  grep -aE "affected (actions|episodes)|undeclared classes" "$ELOG"
  grep -aq "VERDICT: all skips covered by declared policy" "$ELOG" \
    || halt "$U" "undeclared exclusion class" "$ELOG"
  EPHIT=$(grep -aoE 'affected episodes  : [0-9]+' "$ELOG" | grep -oE '[0-9]+$')
  if [ -n "$EPHIT" ] && [ "$EPHIT" -ge 3 ]; then
    halt "$U" "hermes markup in $EPHIT/8 episodes (rule: stop at >=3/8)" "$ELOG"
  fi

  # --- paid scoring --------------------------------------------------------
  SLOG=$STATE/u${U}_score.log
  timeout 2400 $M run scripts/tau2_live_opd_modal.py::rescore \
      --run-id "$RUN" --update "$U" > "$SLOG" 2>&1
  SRC=$?
  grep -aE "teacher tokens|cost|declared skips" "$SLOG" | tail -3
  if [ $SRC -ne 0 ]; then
    grep -aE "UNDECLARED|RuntimeError" "$SLOG" | tail -3
    halt "$U" "scoring stage failed (exit $SRC) -- scores may still be on disk" "$SLOG"
  fi

  # --- one optimizer step from the cached scores ---------------------------
  TLOG=$STATE/u${U}_train.log
  timeout 2400 $M run scripts/tau2_live_opd_modal.py::one_step \
      --run-id "$RUN" --update "$U" --learning-rate "$LR" > "$TLOG" 2>&1
  TRC=$?
  grep -aE "loss |grad_norm|child hash|weights moved|reload_verified" "$TLOG" | tail -6
  [ $TRC -eq 0 ] || halt "$U" "training failed (exit $TRC)" "$TLOG"

  # --- checkpoint verification: lineage, movement, reload ------------------
  vget "$RUNS/$RUN/update-$UU/checkpoint/state.json" /tmp/child_state.json \
    || halt "$U" "no checkpoint written" "$TLOG"
  $PY - "$PHASH" <<'PYV' || halt "$U" "checkpoint verification failed" /tmp/child_state.json
import json, sys, math
s = json.load(open("/tmp/child_state.json"))
parent = sys.argv[1]
ok = True
if s.get("parent_policy_hash") != parent:
    print("  !! parent mismatch: %s != %s" % (s.get("parent_policy_hash"), parent)); ok = False
if not s.get("reload_verified"):  print("  !! reload not verified"); ok = False
if not s.get("bytes_verified"):   print("  !! bytes not verified"); ok = False
r = s.get("reload_report") or {}
if r.get("n_matched") != r.get("n_tensors"):
    print("  !! tensors %s/%s" % (r.get("n_matched"), r.get("n_tensors"))); ok = False
drift = (s.get("bytes_report") or {}).get("max_drift_from_parent", 0)
if not drift or drift <= 0 or math.isnan(drift):
    print("  !! weights did not move (drift=%s)" % drift); ok = False
print("  child %s  parent %s  tensors %s/%s  drift %.3g"
      % (s.get("adapter_hash"), s.get("parent_policy_hash"),
         r.get("n_matched"), r.get("n_tensors"), drift))
sys.exit(0 if ok else 1)
PYV

  # loss / grad must be finite
  $PY - "$TLOG" <<'PYL' || halt "$U" "non-finite loss or grad_norm" "$TLOG"
import re, sys, math
t = open(sys.argv[1], errors="replace").read()
def grab(name):
    m = re.findall(name + r"\s*:?=?\s*([0-9eE.+-]+)", t)
    return float(m[-1]) if m else None
for n in ("loss", "grad_norm"):
    v = grab(n)
    if v is None or not math.isfinite(v):
        print("  !! %s = %s" % (n, v)); sys.exit(1)
    print("  %s = %g" % (n, v))
PYL

  date -u +"done  %H:%M:%SZ"
  echo "---- update $U OK ----"
done

echo ""
echo "======== chain complete: updates $FIRST..$LAST ========"
$M app list 2>/dev/null | grep -c ephemeral | sed 's/^/ephemeral apps remaining: /'
