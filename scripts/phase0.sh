#!/usr/bin/env bash
# Phase 0 — measurement. See docs/PILOT.md.
#
# Produces the three things that are a customer deliverable with no training at
# all: a diagnosis, an environment, and a routing decision. No trainer, no
# teacher GPU, no framework decision. All of it is already coded and none of it
# has ever run.
#
#   0  check-tokenizers   free    teacher/student vocab. gates OPD entirely.
#   1  mine               ~mins   tasks + real verifiers on disk
#   2  replay             $$      both models, one scaffold -> THE GAP NUMBER
#   3  diagnose           $       ranked deficits, or "none found" (exit 0)
#   4  passk (student)    $$$     support measurement, n=8 escalating to n=32
#   5  passk (teacher)    $$$     same, for the teacher
#   6  route              free    (task x capability) -> RL | OPD | QUARANTINE | NONE
#
# GATE at step 6: if the pass@k curves do not separate into regimes, there is no
# routing decision, and everything after Phase 0 is pointless. That is the
# cheapest way this thesis can die and it is why nothing expensive precedes it.
#
# Resumable: each step skips if its output exists. Delete the output to re-run.
#
#   ./scripts/phase0.sh            # run it
#   STEP=2 ./scripts/phase0.sh     # start at step 2
#   DRY=1 ./scripts/phase0.sh      # print the commands, run nothing

set -euo pipefail

OUT="${OUT:-./vektori-out/phase0}"
REPO="${REPO:-hynek/structlog}"
DOCKERFILE="${DOCKERFILE:-examples/dockerfiles/structlog.Dockerfile}"
TEST_CMD="${TEST_CMD:-python -m pytest -p no:randomly -q}"
LANGUAGE="${LANGUAGE:-python}"
LIMIT="${LIMIT:-40}"

# One scaffold, named in every number, shared by every arm. The gap is a
# property of model x scaffold, not of the model.
AGENT="${AGENT:-claude-code}"

# Defaults track vektori_trace/tokenizer_check.py (PILOT_*).
TEACHER="${TEACHER:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
STUDENT="${STUDENT:-Qwen/Qwen3-8B}"

# The frontier arm is an API model and is never Modal-served. It is the ceiling
# we are measuring against, not the distillation teacher.
FRONTIER="${FRONTIER:-gpt-5}"

STAGE1_N="${STAGE1_N:-8}"
STAGE2_N="${STAGE2_N:-32}"
STEP="${STEP:-0}"
DRY="${DRY:-0}"

TASKS_DIR="$OUT/tasks"
MANIFEST="$OUT/manifest.json"
DIAGNOSIS="$OUT/diagnosis.json"
PASSK_STUDENT="$OUT/passk-student"
PASSK_TEACHER="$OUT/passk-teacher"

run() {
  echo
  echo "--- $*"
  if [ "$DRY" = "1" ]; then return 0; fi
  "$@"
}

skip_if() {  # skip_if <path> <label>
  if [ -e "$1" ]; then echo "[skip] $2 — $1 exists"; return 0; fi
  return 1
}

mkdir -p "$OUT"

# --- 0. tokenizer check ------------------------------------------------------
# 30 seconds, free, gates everything: OPD requires teacher and student to share
# a vocabulary. A mismatch means the teacher changes and every later config
# changes with it. Cheapest possible kill.
if [ "$STEP" -le 0 ]; then
  run uv run vektori-trace check-tokenizers --teacher "$TEACHER" --student "$STUDENT"
fi

# --- 1. mine -----------------------------------------------------------------
# --dockerfile skips the bootstrap agent, making this deterministic and free,
# but then --test-cmd is required: F2P/P2P come from running the suite.
# --no-replay stops after mining and auditing, before any agent runs.
if [ "$STEP" -le 1 ]; then
  skip_if "$TASKS_DIR" "mine" || run uv run vektori-trace mine \
    --repo "$REPO" --dockerfile "$DOCKERFILE" --test-cmd "$TEST_CMD" \
    --language "$LANGUAGE" --limit "$LIMIT" --no-replay --out "$OUT"
fi

# --- 2. replay ---------------------------------------------------------------
# The headline number, before any diagnosis. Both models, same tasks, same
# pinned scaffold.
#
# STOP: under ~10 points, change the candidate, not the story. A minimal
# scaffold erases the very gap being studied.
if [ "$STEP" -le 2 ]; then
  skip_if "$MANIFEST" "replay" || run uv run vektori-trace replay \
    --tasks-dir "$TASKS_DIR" --agent "$AGENT" \
    --frontier-model "$FRONTIER" --candidate-model "$STUDENT" --out "$OUT"
fi

# --- 3. diagnose -------------------------------------------------------------
# "No deficit found" is a normal result and exits 0. If that is the answer, the
# chain stops here and what is needed is more traces, not a bigger model.
if [ "$STEP" -le 3 ]; then
  skip_if "$DIAGNOSIS" "diagnose" || run uv run vektori-trace diagnose \
    --manifest "$MANIFEST" --out "$OUT"
fi

# --- 4/5. pass@k, per model --------------------------------------------------
# Sample n once, estimate every k from that sample. Escalate to n=32 only on
# tasks at 0/8 — precisely where the support question is live.
#
# Separate --out per model on purpose: `passk` always writes <out>/passk.json,
# so a shared directory would have the teacher overwrite the student.
#
# The strata are never pooled. Escalating only the zeros biases a naively pooled
# estimate upward, and this is pre-registered as a sequential design.
if [ "$STEP" -le 4 ]; then
  skip_if "$PASSK_STUDENT/passk.json" "passk (student)" || run uv run vektori-trace passk \
    --tasks-dir "$TASKS_DIR" --agent "$AGENT" --model "$STUDENT" \
    --diagnosis "$DIAGNOSIS" --manifest "$MANIFEST" \
    --stage1-n "$STAGE1_N" --stage2-n "$STAGE2_N" --out "$PASSK_STUDENT"
fi

if [ "$STEP" -le 5 ]; then
  skip_if "$PASSK_TEACHER/passk.json" "passk (teacher)" || run uv run vektori-trace passk \
    --tasks-dir "$TASKS_DIR" --agent "$AGENT" --model "$TEACHER" \
    --diagnosis "$DIAGNOSIS" --manifest "$MANIFEST" \
    --stage1-n "$STAGE1_N" --stage2-n "$STAGE2_N" --out "$PASSK_TEACHER"
fi

# --- 6. route ----------------------------------------------------------------
# ★ THE GATE. Every (task x capability) gets exactly one of RL, OPD, QUARANTINE
# or NONE, by thresholds fixed before the data was seen.
#
# Read the per-cell counts, not just the assignment. If the curves do not
# separate into distinct regimes, there is no routing decision to make and the
# thesis dies here — for the price of Phase 0 and nothing else.
if [ "$STEP" -le 6 ]; then
  run uv run vektori-trace route \
    --student-passk "$PASSK_STUDENT/passk.json" \
    --teacher-passk "$PASSK_TEACHER/passk.json" \
    --diagnosis "$DIAGNOSIS" --manifest "$MANIFEST" --out "$OUT"
fi

echo
echo "Phase 0 complete — artifacts in $OUT"
echo
echo "Read routing.json before spending anything on Phase 1:"
echo "  - do the pass@k curves separate into regimes, or is everything one blob?"
echo "  - how many tasks landed in the OPD bucket? under a handful is underpowered."
echo "  - what fraction is QUARANTINE, and is it broken tasks or frontier limits?"
echo
echo "Every number here should be re-derivable from what is on disk."
