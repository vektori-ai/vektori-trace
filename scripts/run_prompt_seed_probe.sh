#!/bin/bash
# One guarded rollout of agronholm__anyio-1121 against repaired ck63, with a
# single native-JSON demonstration seeded ahead of turn 1.
#
# This is a launch wrapper, not a shim: it stages a task, boots an endpoint, runs
# harbor, grades the log and tears the endpoint down. Nothing here sits between
# harbor and vLLM, and no request or completion is touched.
#
#   ./scripts/run_prompt_seed_probe.sh
#
# Costs one L40S for the duration. Teardown is an EXIT trap, so it fires on a
# failed sweep and on a boot that never yields a URL, not only on success.
set -euo pipefail

cd /data/vektori-trace
[ -f .env ] && set -a && . ./.env && set +a

TASK=agronholm__anyio-1121
# Per-repo layout. `cs/corpus50_v3/anyio` does not exist and would silently
# stage nothing, which is the failure mode CLAUDE.md calls out by name.
CORPUS=/data/vektori-trace/cs/corpus50_v3/agronholm_anyio/mined_tasks
CKPT=/adapters/sft/qwen3-14b-dsv4-lora-repaired/checkpoint-63
SERVED=Qwen3-14B-ck63

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
OUT=/data/prompt-seed-probe/$RUN_ID
PROBE_LOG=$OUT/probe.jsonl
STAGE=/tmp/seedprobe-$RUN_ID

# Agent cap 12 min, outer cap 15. A clean timeout is inconclusive about strategy,
# not a model failure -- passk records timed_out and excludes it from the
# denominator rather than scoring it as a loss.
AGENT_TIMEOUT_SEC=${AGENT_TIMEOUT_SEC:-720}
OUTER_TIMEOUT_SEC=${OUTER_TIMEOUT_SEC:-900}

mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "== run $RUN_ID =="
echo "task      $TASK"
echo "checkpoint $CKPT"
echo "out       $OUT"

# --- 1. preflight: never boot a GPU on an unverified prompt geometry ----------
echo
echo "== preflight =="
uv run python scripts/prompt_seed_probe.py --preflight --out-dir "$OUT"
echo "preflight OK"

# --- 2. stage the task -------------------------------------------------------
echo
echo "== staging =="
if [ ! -d "$CORPUS/$TASK" ]; then
  echo "STOP: $CORPUS/$TASK does not exist" >&2
  exit 2
fi
rm -rf "$STAGE" && mkdir -p "$STAGE"
cp -r "$CORPUS/$TASK" "$STAGE/"
grep -q tmux "$STAGE/$TASK/environment/Dockerfile" \
  || { echo "STOP: no tmux in the task Dockerfile; terminus-2 cannot act" >&2; exit 2; }
echo "staged $(ls "$STAGE")"

# --- 3. serve ck63 -----------------------------------------------------------
# L40S is serving hardware and this is serving: BF16 14.8B is ~27.6 GiB of 48,
# and at concurrency 1 the ~93.7k-token KV budget comfortably holds one 40k
# sequence. --max-lora-rank 32 is mandatory; vLLM defaults to 16 and refuses to
# start *after* the GPU is allocated.
echo
echo "== serving =="
ENV_FILE=$OUT/endpoint.env
# Teardown must never be able to silently no-op. The first run's trap fired,
# printed `modal: command not found` and left an L40S billing: `modal` is not on
# PATH under `sudo -iu ubuntu`, and `modal app stop` aborts without --yes when
# there is no tty. Both are fixed here, and the trap now *verifies* the result
# rather than assuming it, because a teardown you do not check is not a teardown.
MODAL=/data/vektori-trace/.venv/bin/modal
teardown() {
  echo
  echo "== teardown =="
  kill "${SERVE_PID:-0}" 2>/dev/null && echo "killed serve pid ${SERVE_PID:-}"
  for app in $("$MODAL" app list 2>/dev/null | grep -oE 'ap-[A-Za-z0-9]+'); do
    echo "stopping $app"
    "$MODAL" app stop "$app" --yes 2>&1 | tail -2 || true
  done
  sleep 5
  echo "-- post-teardown app list --"
  "$MODAL" app list 2>&1 | head -20
  if "$MODAL" app list 2>/dev/null | grep -qE 'ephemeral|deployed'; then
    echo "!!! WARNING: a Modal app is STILL RUNNING -- kill it by hand:" >&2
    echo "!!!   $MODAL app list; $MODAL app stop <APP_ID> --yes" >&2
  else
    echo "all Modal apps stopped"
  fi
}
trap teardown EXIT INT TERM

uv run python scripts/serve_student.py \
  --gpu L40S \
  --base-model Qwen/Qwen3-14B \
  --adapter "$SERVED=$CKPT" \
  --max-lora-rank 32 \
  --max-loras 1 \
  --max-model-len 40960 \
  --max-hours 0.5 \
  --write-env "$ENV_FILE" \
  --gpu-log "$OUT/gpu_log.jsonl" &
SERVE_PID=$!

echo "waiting for endpoint (pid $SERVE_PID) ..."
for _ in $(seq 1 90); do
  [ -f "$ENV_FILE" ] && break
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "STOP: serve died before writing $ENV_FILE" >&2; exit 3; }
  sleep 10
done
[ -f "$ENV_FILE" ] || { echo "STOP: no endpoint after 15 min" >&2; exit 3; }
set -a && . "$ENV_FILE" && set +a
API_BASE="${STUDENT_API_BASE:?no STUDENT_API_BASE in $ENV_FILE}"
# serve_student.py prefixes the adapter name with the base model, so the id it
# actually registers is `Qwen3-14B-<name>`, not `<name>`. Take the served id from
# the env file it writes rather than reconstructing it here -- a reconstruction
# that drifts produces a 404 only once the GPU is already running.
SERVED_ID="${STUDENT_MODEL:?no STUDENT_MODEL in $ENV_FILE}"
HARBOR_MODEL="${STUDENT_HARBOR_MODEL:?no STUDENT_HARBOR_MODEL in $ENV_FILE}"
echo "endpoint     $API_BASE"
echo "served id    $SERVED_ID"
echo "harbor model $HARBOR_MODEL"

# --- 4. smoke: the adapter is actually registered and selectable --------------
echo
echo "== smoke =="
curl -sS "$API_BASE/models" > "$OUT/models.json"
# Exact id match, not a substring grep: `Qwen3-14B-ck63` is a substring of
# `Qwen3-14B-Qwen3-14B-ck63`, so a grep passes on precisely the mismatch that
# then 404s every request.
uv run python - "$OUT/models.json" "$SERVED_ID" <<'PYEOF'
import json, sys
ids = [m["id"] for m in json.load(open(sys.argv[1]))["data"]]
print("served ids:", ids)
if sys.argv[2] not in ids:
    sys.exit(f"STOP: {sys.argv[2]!r} not in /v1/models -- LoRA registration failed")
print(f"adapter {sys.argv[2]} registered")
PYEOF

# --- 5. one guarded rollout --------------------------------------------------
# The import path must reach harbor verbatim; validity.py only hyphenates bare
# agent names, and a colon exempts this form.
echo
echo "== rollout =="
echo "watch live:  ./scripts/watch_prompt_seed_probe.sh $RUN_ID"
echo "  pane   $OUT/**/agent/terminus_2.pane   (the agent's actual terminal)"
echo "  probe  $PROBE_LOG                      (per-turn protocol verdicts)"
export PROMPT_SEED_PROBE_LOG="$PROBE_LOG"
export PYTHONPATH=/data/vektori-trace:${PYTHONPATH:-}
set +e
timeout "$OUTER_TIMEOUT_SEC" uv run vektori-trace passk \
  --tasks-dir "$STAGE" \
  --agent scripts.prompt_seed_probe:Terminus2PromptSeed \
  --model "$HARBOR_MODEL" \
  --api-base "$API_BASE" \
  --model-info @/data/vektori-trace/model_info_14b.json \
  --stage1-n 1 \
  --no-escalate \
  --max-workers 1 \
  --timeout-sec "$AGENT_TIMEOUT_SEC" \
  --out "$OUT"
SWEEP_EXIT=$?
set -e
echo "SWEEP_EXIT=$SWEEP_EXIT"

# --- 6. grade ----------------------------------------------------------------
echo
echo "== protocol gate =="
if [ -f "$PROBE_LOG" ]; then
  set +e
  uv run python scripts/prompt_seed_probe.py --summarize "$PROBE_LOG"
  GATE_EXIT=$?
  set -e
  echo "GATE_EXIT=$GATE_EXIT"
else
  echo "no probe log at $PROBE_LOG -- the agent never produced a completion"
fi

# --- 6b. execution evidence --------------------------------------------------
# The protocol gate proves the model *intended* keystrokes; it sits between
# parsing and _execute_commands and has no execution evidence. Real execution is
# only visible in harbor's trajectory, where terminus writes terminal output back
# as `user` steps -- a user step whose content is terminal output is proof a
# command ran.
echo
echo "== execution evidence (trajectory) =="
TRAJ=$(find "$OUT" -name "*trajectory*.json" -not -name "*summarization*" 2>/dev/null | head -1)
if [ -n "$TRAJ" ]; then
  echo "trajectory: $TRAJ"
  cp "$TRAJ" "$OUT/trajectory.json"
  uv run python - "$TRAJ" <<'PYEOF'
import json, sys
steps = json.load(open(sys.argv[1])).get("steps", [])
print(f"{len(steps)} steps")
user_steps = [s for s in steps if s.get("source") == "user"]
# Step 1 is the initial prompt; any later user step is terminal output returned
# after commands actually ran.
executed = user_steps[1:]
print(f"user steps after the initial prompt (= command executions): {len(executed)}")
for s in steps[:8]:
    msg = (s.get("message") or "")[:400]
    print(f"\n--- step {s.get('step_id')} source={s.get('source')} ---")
    print(msg)
PYEOF
else
  echo "no trajectory found under $OUT -- cannot confirm execution"
fi

echo
echo "== verifier =="
# passk's own accounting. Check no_gradeable_rollouts / infra_failures and the
# per-rollout parse_status before trusting any rate; a verifier that crashed
# before collecting tests is not a model failure.
[ -f "$OUT/passk.json" ] && head -c 2000 "$OUT/passk.json" || echo "no passk.json"

echo
echo "== done: $OUT =="
