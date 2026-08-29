#!/bin/bash
# Behavioural tests for tau2_pilot_chain.sh and tau2_serve_and_run.sh.
#
# These drive the REAL scripts against a fake `modal` and fake stage markers,
# so they exercise orchestration paths -- resume, stale files, failed
# downloads, failed teardown, partial scoring, inconsistent markers -- rather
# than individual predicates. No GPU, no network, no spend.
#
#   bash tests/test_pilot_chain.sh
set -u
REPO_SRC=$(cd "$(dirname "$0")/.." && pwd)
PASS=0; FAIL=0
ok()   { echo "  ok   $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL $1"; echo "$2" | sed 's/^/       /' | head -12; FAIL=$((FAIL+1)); }

# Build a sandbox whose fake `modal` is driven by files in $CTL.
setup() {
  T=$(mktemp -d); CTL=$T/ctl; mkdir -p "$T/scripts" "$T/.venv/bin" "$CTL" "$T/state"
  cp "$REPO_SRC/scripts/tau2_pilot_chain.sh" "$REPO_SRC/scripts/tau2_serve_and_run.sh" \
     "$REPO_SRC/scripts/pilotd_verify_exclusions.py" \
     "$REPO_SRC/scripts/pilotd_scorerow.py" "$T/scripts/" 2>/dev/null
  sed -i "s#^REPO=.*#REPO=$T#; s#^STATE=.*#STATE=$T/state#" "$T/scripts/tau2_pilot_chain.sh"
  sed -i "s#^REPO=.*#REPO=$T#; s#^STATE=.*#STATE=$T/state#" "$T/scripts/tau2_serve_and_run.sh"
  echo "" > "$T/.env"; ln -sf "$(command -v python3)" "$T/.venv/bin/python"
  : > "$CTL/markers"; echo 0 > "$CTL/stop_fails"; echo 0 > "$CTL/get_fails"

  cat > "$T/.venv/bin/modal" <<FAKE
#!/bin/bash
CTL=$CTL
case "\$1 \$2" in
  "volume ls")  cat "\$CTL/markers" ;;
  "volume get")
     if [ "\$(cat \$CTL/get_fails)" = "1" ]; then exit 1; fi
     # positional args differ with/without --force; find them by shape
     src=""; dst=""
     for a in "\$@"; do
       case "\$a" in
         volume|get|--force|vektori-trace-adapters) ;;
         */*) if [ -z "\$src" ]; then src=\$a; else dst=\$a; fi ;;
         *) if [ -n "\$src" ] && [ -z "\$dst" ]; then dst=\$a; fi ;;
       esac
     done
     case "\$src" in
       *update-001/checkpoint/state.json) cp "\$CTL/parent_state.json" "\$dst" 2>/dev/null || exit 1 ;;
       *update-002/checkpoint/state.json) cp "\$CTL/child_state.json"  "\$dst" 2>/dev/null || exit 1 ;;
       *actions.jsonl) cp "\$CTL/actions.jsonl" "\$dst" 2>/dev/null || exit 1 ;;
       *scores.jsonl)  cp "\$CTL/scores.jsonl"  "\$dst" 2>/dev/null || exit 1 ;;
       *manifest.json) cp "\$CTL/manifest.json" "\$dst" 2>/dev/null || exit 1 ;;
       *) exit 1 ;;
     esac ;;
  "app list")
     if [ "\$(cat \$CTL/stop_fails)" = "1" ]; then
       echo "| ap-STUCK | x | ephemeral | 1 |"
     else echo "no apps"; fi ;;
  "app stop") echo "stop requested \$3" ;;
  "run "*|"run") echo "FAKE MODAL RUN: \$*" ;;
esac
exit 0
FAKE
  chmod +x "$T/.venv/bin/modal"
  cat > "$CTL/parent_state.json" <<'J'
{"adapter_hash":"PARENT01","parent_policy_hash":"GRAND01","reload_verified":true,
 "bytes_verified":true,"reload_report":{"n_matched":504,"n_tensors":504},
 "bytes_report":{"max_drift_from_parent":1e-5}}
J
  cat > "$CTL/child_state.json" <<'J'
{"adapter_hash":"CHILD01","parent_policy_hash":"PARENT01","reload_verified":true,
 "bytes_verified":true,"reload_report":{"n_matched":504,"n_tensors":504},
 "bytes_report":{"max_drift_from_parent":1e-5}}
J
  echo '{}' > "$CTL/manifest.json"; : > "$CTL/actions.jsonl"; : > "$CTL/scores.jsonl"
}
teardown() { rm -rf "$T"; }

run_chain() { (cd "$T" && bash scripts/tau2_pilot_chain.sh testrun 2 2 2>&1); }

echo "== chain orchestration =="

# 1. .TRAINED with a good checkpoint -> skip, and re-verify it
setup
printf 'update-002/.SAMPLED\nupdate-002/.SCORED\nupdate-002/.TRAINED\n' > "$CTL/markers"
out=$(run_chain)
echo "$out" | grep -q "already TRAINED and verified -- skipping" \
  && ok "TRAINED skips after re-verifying checkpoint" \
  || bad "TRAINED skip did not re-verify" "$out"
teardown

# 2. .TRAINED whose checkpoint has the WRONG parent -> must halt, not skip
setup
printf 'update-002/.SAMPLED\nupdate-002/.SCORED\nupdate-002/.TRAINED\n' > "$CTL/markers"
sed -i 's/"PARENT01"/"WRONGPARENT"/' "$CTL/child_state.json"
out=$(run_chain)
echo "$out" | grep -q "HALT" \
  && ok "TRAINED with wrong-parent checkpoint halts" \
  || bad "wrong-parent checkpoint was skipped as done" "$out"
teardown

# 3. .SCORED without .SAMPLED -> inconsistent markers
setup
printf 'update-002/.SCORED\n' > "$CTL/markers"
out=$(run_chain)
echo "$out" | grep -q "inconsistent markers" \
  && ok "SCORED without SAMPLED halts" \
  || bad "inconsistent markers accepted" "$out"
teardown

# 4. .TRAINED without .SCORED -> inconsistent markers
setup
printf 'update-002/.SAMPLED\nupdate-002/.TRAINED\n' > "$CTL/markers"
out=$(run_chain)
echo "$out" | grep -q "inconsistent markers" \
  && ok "TRAINED without SCORED halts" \
  || bad "inconsistent markers accepted" "$out"
teardown

# 5. failed download must halt, never verify a stale file
setup
printf 'update-002/.SAMPLED\n' > "$CTL/markers"
echo 1 > "$CTL/get_fails"
out=$(run_chain)
echo "$out" | grep -qE "HALT.*(download|parent)" \
  && ok "failed download halts instead of using stale data" \
  || bad "failed download did not halt" "$out"
teardown

echo "== endpoint teardown =="

# 6. teardown failure must be a non-zero exit and a machine-readable marker
setup
echo 1 > "$CTL/stop_fails"
cat > "$T/scripts/serve_student.py" <<'S'
import sys, time
a = sys.argv; v = lambda f: a[a.index(f)+1]
open(v("--write-app-id"), "w").write("MODAL_APP_ID=ap-STUCK\n")
open(v("--write-env"), "w").write("STUDENT_API_BASE=http://x/v1\n")
time.sleep(120)
S
out=$( (cd "$T" && bash scripts/tau2_serve_and_run.sh --adapter /f --tag t -- true 2>&1) )
rc=$?
if echo "$out" | grep -q "TEARDOWN_FAILED" && [ $rc -ne 0 ]; then
  ok "stuck endpoint -> TEARDOWN_FAILED and non-zero exit ($rc)"
else
  bad "stuck endpoint did not fail loudly (rc=$rc)" "$out"
fi
teardown

# 7. clean teardown still returns the command's own status
setup
cat > "$T/scripts/serve_student.py" <<'S'
import sys, time
a = sys.argv; v = lambda f: a[a.index(f)+1]
open(v("--write-app-id"), "w").write("MODAL_APP_ID=ap-OK\n")
open(v("--write-env"), "w").write("STUDENT_API_BASE=http://x/v1\n")
time.sleep(120)
S
(cd "$T" && bash scripts/tau2_serve_and_run.sh --adapter /f --tag t -- bash -c 'exit 3' >/dev/null 2>&1)
[ $? -eq 3 ] && ok "clean teardown preserves command exit code" \
             || bad "exit code not preserved" ""
teardown


# ---------------------------------------------------------------------------
# Semantic gate tests. Each asserts the CORRECT behaviour and is written so it
# would FAIL against the earlier implementation it replaces.
# ---------------------------------------------------------------------------
echo "== semantic gates =="
PY=$(command -v python3)

# G1. post-training verifier must read argv, not an unexpanded "$TMP" literal.
#     Old form: json.load(open(""$TMP/child_state.json"")) -> NameError/IOError.
cat > /tmp/gt_child.json <<'J'
{"adapter_hash":"C1","parent_policy_hash":"P1","reload_verified":true,
 "bytes_verified":true,"reload_report":{"n_matched":504,"n_tensors":504},
 "bytes_report":{"max_drift_from_parent":1e-5}}
J
out=$(awk '/<<.PYV./{f=1;next} f&&/^PYV$/{exit} f' scripts/tau2_pilot_chain.sh \
      | $PY - P1 /tmp/gt_child.json 2>&1)
if echo "$out" | grep -q "child C1"; then ok "G1 checkpoint verifier reads argv"
else bad "G1 verifier broken (would die after paid training)" "$out"; fi

# G2. structural-weight must be independently required: a log with a clean
#     newline line but NO structural evidence must still fail.
mkg2() { printf '%s\n' "$@" > /tmp/gt_dry.log; }
g2() { awk '/<<.PYS./{f=1;next} f&&/^PYS$/{exit} f' scripts/tau2_pilot_chain.sh \
       | $PY - /tmp/gt_dry.log >/dev/null 2>&1; }
mkg2 "structural weight zero" "  no index-1 boundary newline : True"; g2 && r1=PASS || r1=HALT
mkg2 "structural weight nonzero" "  no index-1 boundary newline : True"; g2 && r2=PASS || r2=HALT
[ "$r1" = PASS ] && [ "$r2" = HALT ] \
  && ok "G2 structural weight is required independently ($r1/$r2)" \
  || bad "G2 structural gate fail-open ($r1/$r2)" ""

# G4. coverage must require EVERY action to have a bound score: 2 scores for
#     3 actions must fail, where the old `matched>0 && mismatched==0` passed.
printf '%s\n' '{"key":"a","fingerprint":"f1"}' '{"key":"b","fingerprint":"f2"}' > /tmp/gt_sc.jsonl
printf '%s\n' '{"key":"a","score_fingerprint":"f1"}' '{"key":"b","score_fingerprint":"f2"}' \
              '{"key":"c","score_fingerprint":"f3"}' > /tmp/gt_ac.jsonl
# anchor on the PYF block specifically: several heredocs start "import json, sys"
covg() { awk '/<<.PYF./{f=1;next} f&&/^PYF$/{exit} f' scripts/tau2_pilot_chain.sh \
         | $PY - "$1" "$2" 2>&1; }
out=$(covg /tmp/gt_sc.jsonl /tmp/gt_ac.jsonl); rc=$?
[ $rc -ne 0 ] && ok "G4 incomplete coverage (2 of 3) halts" \
              || bad "G4 accepted incomplete coverage" "$out"
printf '%s\n' '{"key":"a","fingerprint":"f1"}' '{"key":"b","fingerprint":"f2"}' \
              '{"key":"c","fingerprint":"f3"}' > /tmp/gt_sc.jsonl
out=$(covg /tmp/gt_sc.jsonl /tmp/gt_ac.jsonl); rc=$?
[ $rc -eq 0 ] && ok "G4 complete bound coverage passes" \
              || bad "G4 rejected a complete set" "$out"
printf '%s\n' '{"key":"a","fingerprint":"WRONG"}' '{"key":"b","fingerprint":"f2"}' \
              '{"key":"c","fingerprint":"f3"}' > /tmp/gt_sc.jsonl
out=$(covg /tmp/gt_sc.jsonl /tmp/gt_ac.jsonl); rc=$?
[ $rc -ne 0 ] && ok "G4 fingerprint mismatch halts" \
              || bad "G4 accepted a mismatched fingerprint" "$out"

rm -f /tmp/gt_child.json /tmp/gt_dry.log /tmp/gt_sc.jsonl /tmp/gt_ac.jsonl
echo ""
echo "TOTAL passed $PASS, failed $FAIL"
[ $FAIL -eq 0 ]
