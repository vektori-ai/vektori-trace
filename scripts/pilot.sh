#!/usr/bin/env bash
# Pilot run: mine prefect, measure both models, print the three counts.
#
#   ./scripts/pilot.sh smoke     ~100 PRs  -> ~10 tasks, n=4   (day 1: does it run?)
#   ./scripts/pilot.sh pilot     ~500 PRs  -> ~50 tasks, n=8   (week 1: the result)
#
# Needs: docker, a GitHub token (gh auth), and STUDENT_API_BASE pointing at a
# vLLM server holding the student. TEACHER_MODEL goes through harbor's normal
# model routing.
set -euo pipefail

# Prefer the repo's venv so this runs from a fresh clone without activating it.
if [ -x ".venv/bin/vektori-trace" ]; then
  vektori-trace() { .venv/bin/vektori-trace "$@"; }
elif ! command -v vektori-trace >/dev/null 2>&1; then
  echo "vektori-trace not found. Run 'uv pip install -e .' or activate the venv." >&2
  exit 2
fi

MODE="${1:-smoke}"
case "$MODE" in
  smoke) LIMIT=100; N=4  ;;
  pilot) LIMIT=500; N=8  ;;
  *) echo "usage: $0 [smoke|pilot]" >&2; exit 2 ;;
esac

OUT="./cs/$MODE"
AGENT="${AGENT:-terminus-2}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-flash}"
STUDENT_API_BASE="${STUDENT_API_BASE:-http://127.0.0.1:8000/v1}"
TASKS="$OUT/mined/mined_tasks"

echo "=== [1/4] mine prefect ($LIMIT PRs) ==="
vektori-trace mine \
  --repo prefecthq/prefect \
  --dockerfile examples/dockerfiles/prefect.Dockerfile \
  --test-cmd 'python -m pytest -m "not service" -p no:randomly -q' \
  --language python \
  --limit "$LIMIT" \
  --no-replay \
  --out "$OUT/mined"

n_tasks=$(find "$TASKS" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
echo "    -> $n_tasks tasks emitted"
if [ "$n_tasks" -eq 0 ]; then
  echo "    !! nothing emitted. Read $OUT/mined/mine-report.json — the skip"
  echo "       histogram says why. Most likely prefect's filterwarnings=error"
  echo "       or dependency drift between bootstrap HEAD and the PR base."
  exit 1
fi

# --stage2-n == --stage1-n turns off escalation to 32. At n=8 a 0/8 task only
# tells you the true pass rate is under ~30%, so this pilot cannot separate
# "never passes" from "rarely passes". That is the RL-vs-OPD split — fine to
# skip while checking the machinery, not fine to publish a routing claim from.
echo "=== [2/4] teacher pass@k ($TEACHER_MODEL, n=$N) ==="
vektori-trace passk --tasks-dir "$TASKS" --agent "$AGENT" \
  --model "$TEACHER_MODEL" --stage1-n "$N" --stage2-n "$N" \
  --out "$OUT/passk-teacher"

echo "=== [3/4] student pass@k (n=$N) ==="
vektori-trace passk --tasks-dir "$TASKS" --agent "$AGENT" \
  --api-base "$STUDENT_API_BASE" --stage1-n "$N" --stage2-n "$N" \
  --out "$OUT/passk-student"

echo "=== [4/4] the three counts ==="
python3 - "$OUT/passk-teacher/passk.json" "$OUT/passk-student/passk.json" <<'PY'
import json, sys

teacher = json.load(open(sys.argv[1]))["stage1"]
student = json.load(open(sys.argv[2]))["stage1"]

fine = trainable = skip = 0
rows = []
for task in sorted(set(teacher) & set(student)):
    t, s = teacher[task], student[task]
    if not t["n"] or not s["n"]:
        continue  # no gradeable rollout; never counted as a loss
    t_pass, s_pass = t["c"] > 0, s["c"] > 0
    if s_pass:
        bucket = "already fine"; fine += 1
    elif t_pass:
        bucket = "TRAINABLE"; trainable += 1
    else:
        bucket = "skip"; skip += 1
    rows.append((task, f"{s['c']}/{s['n']}", f"{t['c']}/{t['n']}", bucket))

print(f"\n{'task':44} {'student':>8} {'teacher':>8}  bucket")
print("-" * 76)
for task, sc, tc, b in rows:
    print(f"{task[:44]:44} {sc:>8} {tc:>8}  {b}")

total = fine + trainable + skip
print(f"\n  already fine  {fine:>4}   student solves it, nothing to do")
print(f"  TRAINABLE     {trainable:>4}   student fails, teacher solves it  <-- the pile that matters")
print(f"  skip          {skip:>4}   neither solves it")
print(f"  total         {total:>4}")
if total:
    print(f"\n  trainable fraction: {trainable/total:.0%}")
if trainable == 0:
    print("\n  Nothing trainable. Either the tasks are too hard for both, or the")
    print("  student is closer to the teacher than expected. Check a few traces")
    print("  before spending anything on a GPU.")
PY
