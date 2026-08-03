#!/bin/bash
# Solve the same task with DeepSeek-V4-Flash-0731 on Fireworks, capturing top-K
# logprobs for every generated token.
#
# Same shape as run_sweep.sh -- same task, same agent, same pass@k harness -- so
# the two runs are comparable. Only the model behind the proxy changes.
#
#   TOP_LOGPROBS=10 ./run_teacher_task.sh
#
# Requires FIREWORKS_API_KEY in the environment (or .env).
set -euo pipefail

cd /data/vektori-trace
[ -f .env ] && set -a && . ./.env && set +a

TASK=prefecthq__prefect-65ea05bef8d9
MODEL_PATH=accounts/fireworks/models/deepseek-v4-flash-0731
TOP_LOGPROBS=${TOP_LOGPROBS:-5}
OUT=/data/vektori-trace/vektori-out/teacher-dsv4
# The stage1 timeout that turned every Qwen3-8B rollout into an "infra failure"
# rather than a graded fail. 30 min was not enough for a 40-step agentic
# trajectory; a real fail and a timeout must not be the same record.
TIMEOUT_SEC=${TIMEOUT_SEC:-5400}

if [ -z "${FIREWORKS_API_KEY:-}" ]; then
  echo "FIREWORKS_API_KEY is not set -- put it in /data/vektori-trace/.env" >&2
  exit 2
fi

mkdir -p "$OUT"

# --- preflight: never start a multi-hour run on an unverified K --------------
echo "== preflight: probing $MODEL_PATH =="
uv run python scripts/probe_fireworks_logprobs.py \
  --model "$MODEL_PATH" --max-k "$TOP_LOGPROBS" --out "$OUT/probe.json" \
  | tee "$OUT/probe.log"

MAX_K=$(python3 -c "import json;print(json.load(open('$OUT/probe.json'))['max_top_logprobs'])")
if [ "$MAX_K" -lt "$TOP_LOGPROBS" ]; then
  echo >&2
  echo "STOP: this deployment accepts top_logprobs<=$MAX_K, run asked for $TOP_LOGPROBS." >&2
  echo "K is part of the objective -- not silently lowering it. Either rerun with" >&2
  echo "TOP_LOGPROBS=$MAX_K, or stand up a dedicated deployment with a higher" >&2
  echo "--max-logprobs and point MODEL_PATH at accounts/<acct>/deployments/<id>." >&2
  exit 3
fi
echo "== preflight OK: K=$TOP_LOGPROBS accepted =="

# --- stage the task (identical to run_sweep.sh, incl. the tmux bake-in) ------
rm -rf /tmp/one && mkdir -p /tmp/one
cp -r "cs/smoke/cmined/mined_tasks/$TASK" /tmp/one/
DOCKERFILE="/tmp/one/$TASK/environment/Dockerfile"
if ! grep -q "tmux" "$DOCKERFILE"; then
  python3 - "$DOCKERFILE" <<'PYEOF'
import sys
path = sys.argv[1]
content = open(path).read()
marker = "# Position the working tree at the PR's base commit"
patch = '''# terminus-2 acts only through a tmux pane; the runtime network allowlist
# blocks installing tmux later, so bake it in at build time.
RUN command -v tmux >/dev/null 2>&1 || \\
    (apt-get update && apt-get install -y --no-install-recommends tmux \\
     && rm -rf /var/lib/apt/lists/*) || \\
    apk add --no-cache tmux || true
'''
assert marker in content, "marker not found"
open(path, "w").write(content.replace(marker, patch + marker))
print("tmux patch applied")
PYEOF
fi
echo "STAGED: $(ls /tmp/one)"

# --- capture proxy in front of Fireworks ------------------------------------
# The proxy is what makes the run recoverable: it injects return_token_ids +
# logprobs + top_logprobs on every call and writes one JSONL line per completion.
# --upstream-api-key-env replaces whatever key harbor hands litellm with the
# real Fireworks one.
tmux kill-session -t dsv4-proxy 2>/dev/null || true
tmux new-session -d -s dsv4-proxy \
  "cd /data/vektori-trace && set -a && . ./.env && set +a && \
   PYTHONUNBUFFERED=1 uv run vektori-trace capture-proxy \
     --upstream https://api.fireworks.ai/inference/v1 \
     --top-logprobs $TOP_LOGPROBS \
     --upstream-api-key-env FIREWORKS_API_KEY \
     --host 127.0.0.1 --port 37370 \
     --out $OUT/captures 2>&1 | tee $OUT/proxy.log"

PROXY_API_BASE="http://127.0.0.1:37370/v1"
for i in $(seq 1 40); do
  if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(s.connect_ex(('127.0.0.1',37370)))" 2>/dev/null; then
    break
  fi
  sleep 1
done
python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(s.connect_ex(('127.0.0.1',37370)))" \
  || { echo "proxy did not bind 127.0.0.1:37370" >&2; tmux kill-session -t dsv4-proxy 2>/dev/null; exit 4; }
echo "== proxy up at $PROXY_API_BASE =="

# Deliberately the *same* window as the Qwen3-8B run (40448), not DeepSeek's real
# 1,048,576. Two reasons: the runs are only comparable if the agent hits the same
# summarization pressure, and a 1M window on a model that bills per token turns
# one rollout into an open-ended spend.
cat > "$OUT/model_info.json" <<'JSON'
{
  "max_input_tokens": 40448,
  "max_output_tokens": 8192,
  "input_cost_per_token": 0.0,
  "output_cost_per_token": 0.0
}
JSON

# --- the run ----------------------------------------------------------------
# `openai/` prefix: litellm passes the rest of the model string through to
# api_base untouched, which is what a Fireworks resource path needs.
uv run vektori-trace passk \
  --tasks-dir /tmp/one \
  --agent terminus-2 \
  --model "openai/$MODEL_PATH" \
  --api-base "$PROXY_API_BASE" \
  --model-info "@$OUT/model_info.json" \
  --stage1-n ${STAGE1_N:-1} \
  --no-escalate \
  --max-workers 1 \
  --timeout-sec "$TIMEOUT_SEC" \
  --out "$OUT" 2>&1 | tee "$OUT/sweep.log"
echo "SWEEP_EXIT=$?"

# --- reconcile: captures vs failures ----------------------------------------
echo "== capture summary =="
uv run python scripts/summarize_captures.py "$OUT/captures" | tee "$OUT/capture_summary.txt"
