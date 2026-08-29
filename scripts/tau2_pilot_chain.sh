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

# Download and PROVE it landed. An unchecked `volume get` that fails leaves a
# stale file from a previous update in place, and the verifier then passes on
# the wrong update's data -- a fail-open path, not a missing feature.
vget() { # vget <volume-path> <local>
  rm -f "$2"
  if ! $M volume get "$VOL" "$1" "$2" >/dev/null 2>&1; then
    return 1
  fi
  [ -s "$2" ] || return 1
  return 0
}

for U in $(seq "$FIRST" "$LAST"); do
  P=$((U-1))
  UU=$(printf "%03d" "$U"); PP=$(printf "%03d" "$P")
  # Fresh scratch per update: nothing can leak across iterations.
  TMP=$(mktemp -d "/tmp/chain_${RUN}_u${U}_XXXXXX") || halt "$U" "mktemp failed" -
  trap 'rm -rf "$TMP"' RETURN 2>/dev/null || true
  echo ""
  echo "================ UPDATE $U ================"
  date -u +"start %H:%M:%SZ"

  # --- parent checkpoint hash: what this rollout must sample from ----------
  # Resolved BEFORE the marker checks, because verifying an existing
  # .TRAINED checkpoint compares its parent_policy_hash against this.
  vget "$RUNS/$RUN/update-$PP/checkpoint/state.json" "$TMP/parent_state.json" \
    || halt "$U" "parent update-$PP has no checkpoint" "volume"
  PHASH=$($PY -c "import json;print(json.load(open('$TMP/parent_state.json'))['adapter_hash'])")
  [ -n "$PHASH" ] || halt "$U" "could not read parent adapter hash" "$TMP/parent_state.json"
  echo "parent checkpoint: update-$PP  hash $PHASH"

  # Resume from stage markers: a halt must never redo a rollout or, worse,
  # re-pay the teacher for scores already on the volume.
  MK=$($M volume ls "$VOL" "$RUNS/$RUN/update-$UU" 2>/dev/null)
  HAVE_SAMPLED=0; HAVE_SCORED=0; HAVE_TRAINED=0
  echo "$MK" | grep -q '\.SAMPLED' && HAVE_SAMPLED=1
  echo "$MK" | grep -q '\.SCORED'  && HAVE_SCORED=1
  echo "$MK" | grep -q '\.TRAINED' && HAVE_TRAINED=1
  # Markers must be consistent: a later stage without its predecessor means
  # the volume is in a state this script did not create, so stop for a human.
  if [ $HAVE_SCORED -eq 1 ] && [ $HAVE_SAMPLED -eq 0 ]; then
    halt "$U" ".SCORED present without .SAMPLED -- inconsistent markers" "volume"
  fi
  if [ $HAVE_TRAINED -eq 1 ] && [ $HAVE_SCORED -eq 0 ]; then
    halt "$U" ".TRAINED present without .SCORED -- inconsistent markers" "volume"
  fi

  if [ $HAVE_TRAINED -eq 1 ]; then
    # Never skip on a marker alone: verify the checkpoint the marker claims.
    vget "$RUNS/$RUN/update-$UU/checkpoint/state.json" "$TMP/done_state.json" \
      || halt "$U" ".TRAINED but checkpoint state.json is missing" "volume"
    $PY - "$PHASH" "$TMP/done_state.json" <<'PYD' || halt "$U" ".TRAINED checkpoint failed verification" "$TMP/done_state.json"
import json, sys
s = json.load(open(sys.argv[2])); parent = sys.argv[1]; ok = True
if s.get("parent_policy_hash") != parent:
    print("  !! parent %s != expected %s" % (s.get("parent_policy_hash"), parent)); ok = False
if not s.get("reload_verified") or not s.get("bytes_verified"):
    print("  !! reload/bytes not verified"); ok = False
r = s.get("reload_report") or {}
if r.get("n_matched") != r.get("n_tensors"):
    print("  !! tensors %s/%s" % (r.get("n_matched"), r.get("n_tensors"))); ok = False
print("  verified existing checkpoint %s (tensors %s/%s)"
      % (s.get("adapter_hash"), r.get("n_matched"), r.get("n_tensors")))
sys.exit(0 if ok else 1)
PYD
    echo "already TRAINED and verified -- skipping"; rm -rf "$TMP"; continue
  fi
  [ $HAVE_SAMPLED -eq 1 ] && echo "resume: .SAMPLED present, skipping rollout"
  [ $HAVE_SCORED  -eq 1 ] && echo "resume: .SCORED present, reusing paid scores"


  # --- rollout, with the endpoint torn down on every exit path ------------
  if [ $HAVE_SAMPLED -eq 1 ]; then
   echo "skip rollout (already sampled)"
  else
  RLOG=$STATE/u${U}_rollout.log
  # The endpoint URL is not known until serve_student.py writes it, so the
  # rollout is invoked through `bash -c`: the wrapper sources the env file
  # before running this, and the single quotes defer $STUDENT_API_BASE to
  # that moment. Passing it directly through "$@" would hand the rollout the
  # literal string '$STUDENT_API_BASE' -- verified, it does not expand.
  bash scripts/tau2_serve_and_run.sh \
      --adapter "/adapters/$RUNS/$RUN/update-$PP/checkpoint" \
      --tag "chain_u$U" --max-hours 1 \
      -- bash -c "$M run scripts/tau2_live_opd_modal.py::rollout_only \
           --run-id '$RUN' --update '$U' \
           --api-base \"\$STUDENT_API_BASE\" --student-model Qwen3-4B-sft \
           --adapter-hash-expect '$PHASH'" \
      > "$RLOG" 2>&1
  RC=$?
  grep -aE "rollout took|episodes" "$RLOG" | tail -2
  # 90/91 are the wrapper's teardown-failure codes: an endpoint may still be
  # billing, so the chain must stop rather than start another update.
  if grep -aq "TEARDOWN_FAILED" "$RLOG"; then
    grep -a "TEARDOWN_FAILED" "$RLOG" | tail -2
    halt "$U" "ENDPOINT TEARDOWN FAILED -- a GPU may still be billing" "$RLOG"
  fi
  [ $RC -eq 0 ] || halt "$U" "rollout failed (exit $RC)" "$RLOG"
  fi

  # --- free dry-run: retention, accounting, structural weight --------------
  DLOG=$STATE/u${U}_dryrun.log
  timeout 900 $M run scripts/tau2_live_opd_modal.py::scoring_dryrun \
      --run-id "$RUN" --update "$U" > "$DLOG" 2>&1
  DRC=$?
  # An ignored exit code lets a crashed dry-run pass on stale matching text.
  [ $DRC -eq 0 ] || halt "$U" "dry-run failed (exit $DRC)" "$DLOG"
  vget "$RUNS/$RUN/update-$UU/actions.jsonl" "$TMP/actions.jsonl" \
    || halt "$U" "could not download actions.jsonl" "volume"
  vget "$RUNS/$RUN/manifest.json" "$TMP/manifest.json" \
    || halt "$U" "could not download manifest.json" "volume"
  RET=$(grep -aoE 'retained: [0-9.]+' "$DLOG" | tail -1 | grep -oE '[0-9.]+')
  echo "retention: $RET"
  [ -n "$RET" ] || halt "$U" "dry-run produced no retention figure" "$DLOG"
  $PY -c "import sys; sys.exit(0 if float('$RET') >= 0.90 else 1)" \
    || halt "$U" "retention $RET < 0.90" "$DLOG"
  grep -aq "no index-1 boundary newline : True" "$DLOG" \
    || halt "$U" "boundary-newline weight present" "$DLOG"
  grep -aq "token accounting complete   : True" "$DLOG" \
    || halt "$U" "token accounting incomplete" "$DLOG"
  # Two independent requirements. An `||` chain between them lets a passing
  # newline check satisfy a MISSING structural-zero result, which is a
  # fail-open: markup weight would go unnoticed whenever the newline is clean.
  $PY - "$DLOG" <<'PYS' || halt "$U" "structural/markup tokens may carry weight" "$DLOG"
import re, sys
t = open(sys.argv[1], errors="replace").read()
# every exclusion class that must contribute exactly zero supervised weight
struct = re.search(r"structural weight (\w+)", t)
if struct:
    if struct.group(1) != "zero":
        print("  !! structural weight is %r, not zero" % struct.group(1)); sys.exit(1)
    print("  structural weight zero: confirmed"); sys.exit(0)
# no explicit line: require the dry-run's own structural acceptance instead
m = re.search(r"no index-1 boundary newline\s*:\s*(\w+)", t)
if not m:
    print("  !! dry-run reported no structural-weight evidence at all"); sys.exit(1)
if m.group(1) != "True":
    print("  !! boundary newline carries weight"); sys.exit(1)
print("  structural evidence: boundary-newline True (no explicit structural line)")
PYS

  # Declared stop rule: >10% of reasoning bytes excluded. Derived from the
  # dry-run's own exclusion counters rather than assumed from retention.
  $PY - "$DLOG" "$TMP/actions.jsonl" <<'PYB' || halt "$U" "excluded reasoning exceeds 10%" "$DLOG"
import base64, json, re, sys
sys.path.insert(0, "/data/vektori-trace")
from vektori_trace.tau2.live_agent import split_generation

# The preregistered rule is *reasoning bytes excluded / total reasoning bytes*.
# Dividing excluded TOKENS by ALL student tokens is a different quantity with a
# different denominator and silently understates the ratio.
log = open(sys.argv[1], errors="replace").read()
total_bytes = 0
excluded_bytes = 0
skipped = set()
m = re.search(r"payload skips\s*:\s*(\{.*?\})", log)
n_skips = 0
if m:
    try:
        n_skips = sum(json.loads(m.group(1)).values())
    except Exception:
        n_skips = 0

for line in open(sys.argv[2]):
    line = line.strip()
    if not line:
        continue
    a = json.loads(line)
    raw = base64.b64decode(a["action_bytes_b64"]).decode("utf-8", "replace")
    try:
        reasoning, _c, _t = split_generation(raw)
    except Exception:
        continue
    if not reasoning:
        continue
    nb = len(reasoning.encode("utf-8"))
    total_bytes += nb
    # a payload skip drops the WHOLE reasoning payload for that action
    if any(mk in reasoning for mk in ("<tool_call>", "</tool_call>")):
        excluded_bytes += nb

frac = (excluded_bytes / total_bytes) if total_bytes else 1.0
print("  excluded reasoning: %d / %d bytes = %.2f%% (rule: stop above 10%%)"
      % (excluded_bytes, total_bytes, 100 * frac))
sys.exit(0 if frac <= 0.10 else 1)
PYB

  # --- exclusions must match the DECLARED policy, not merely be few -------
  ELOG=$STATE/u${U}_exclusions.log
  $PY scripts/pilotd_verify_exclusions.py "$TMP/manifest.json" \
      "$TMP/actions.jsonl" > "$ELOG" 2>&1 \
    || halt "$U" "exclusion verifier crashed" "$ELOG"
  grep -aE "affected (actions|episodes)|undeclared classes" "$ELOG"
  grep -aq "VERDICT: all skips covered by declared policy" "$ELOG" \
    || halt "$U" "undeclared exclusion class" "$ELOG"
  EPHIT=$(grep -aoE 'affected episodes  : [0-9]+' "$ELOG" | grep -oE '[0-9]+$')
  if [ -n "$EPHIT" ] && [ "$EPHIT" -ge 3 ]; then
    halt "$U" "hermes markup in $EPHIT/8 episodes (rule: stop at >=3/8)" "$ELOG"
  fi

  # --- paid scoring --------------------------------------------------------
  if [ $HAVE_SCORED -eq 1 ]; then
   echo "skip scoring (.SCORED present, scores reused)"
  else
  SLOG=$STATE/u${U}_score.log
  timeout 2400 $M run scripts/tau2_live_opd_modal.py::rescore \
      --run-id "$RUN" --update "$U" > "$SLOG" 2>&1
  SRC=$?
  grep -aE "teacher tokens|cost|declared skips" "$SLOG" | tail -3
  if [ $SRC -ne 0 ]; then
    grep -aE "UNDECLARED|RuntimeError" "$SLOG" | tail -3
    halt "$U" "scoring stage failed (exit $SRC) -- scores may still be on disk" "$SLOG"
  fi
  fi

  # --- post-score validation: the marker alone is not evidence ------------
  MK2=$($M volume ls "$VOL" "$RUNS/$RUN/update-$UU" 2>/dev/null)
  echo "$MK2" | grep -q '\.SCORED' || halt "$U" ".SCORED missing after scoring" "$SLOG"
  vget "$RUNS/$RUN/update-$UU/scores.jsonl" "$TMP/scores.jsonl" \
    || halt "$U" "scores.jsonl missing after scoring" "volume"
  $PY scripts/pilotd_scorerow.py "$TMP/scores.jsonl" "$TMP/actions.jsonl" \
      > "$STATE/u${U}_scorecheck.log" 2>&1 \
    || halt "$U" "score-row check crashed" "$STATE/u${U}_scorecheck.log"
  grep -aE "matched [0-9]+   mismatched" "$STATE/u${U}_scorecheck.log"
  $PY - "$TMP/scores.jsonl" "$TMP/actions.jsonl" <<'PYF' || halt "$U" "score coverage incomplete" "$STATE/u${U}_scorecheck.log"
import json, sys

# `matched > 0 and mismatched == 0` accepts 70 scores for 77 actions. The
# requirement is that every scoreable action HAS a fingerprint-bound score.
def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

scores = {r["key"]: r for r in load(sys.argv[1])}
actions = {r["key"]: r for r in load(sys.argv[2])}
missing = sorted(set(actions) - set(scores))
orphan = sorted(set(scores) - set(actions))
bad = [k for k in set(scores) & set(actions)
       if scores[k].get("fingerprint") != actions[k].get("score_fingerprint")]
print("  actions %d  scores %d  missing %d  orphan %d  fingerprint-mismatch %d"
      % (len(actions), len(scores), len(missing), len(orphan), len(bad)))
if missing:
    print("  !! actions with no score: %s" % missing[:5])
if orphan:
    print("  !! scores with no action: %s" % orphan[:5])
if bad:
    print("  !! fingerprint mismatch: %s" % bad[:5])
sys.exit(0 if not (missing or orphan or bad) else 1)
PYF

  # --- one optimizer step from the cached scores ---------------------------
  TLOG=$STATE/u${U}_train.log
  timeout 2400 $M run scripts/tau2_live_opd_modal.py::one_step \
      --run-id "$RUN" --update "$U" --learning-rate "$LR" > "$TLOG" 2>&1
  TRC=$?
  grep -aE "loss |grad_norm|child hash|weights moved|reload_verified" "$TLOG" | tail -6
  [ $TRC -eq 0 ] || halt "$U" "training failed (exit $TRC)" "$TLOG"

  # --- checkpoint verification: lineage, movement, reload ------------------
  vget "$RUNS/$RUN/update-$UU/checkpoint/state.json" "$TMP/child_state.json" \
    || halt "$U" "no checkpoint written" "$TLOG"
  $PY - "$PHASH" "$TMP/child_state.json" <<'PYV' || halt "$U" "checkpoint verification failed" "$TMP/child_state.json"
import json, sys, math
s = json.load(open(sys.argv[2]))
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
  rm -rf "$TMP"
  echo "---- update $U OK ----"
done

echo ""
echo "======== chain complete: updates $FIRST..$LAST ========"
$M app list 2>/dev/null | grep -c ephemeral | sed 's/^/ephemeral apps remaining: /'
