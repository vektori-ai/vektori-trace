#!/bin/bash
set -x
cd /data/vektori-trace
PROXY_URL="http://127.0.0.1:37369/v1"
curl -sS "$PROXY_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"say OK"}],"max_tokens":8}'
echo
echo "---PREFLIGHT---"
uv run python scripts/verify_run_logs.py \
  --out /data/vektori-trace/vektori-out/baseline \
  --preflight \
  --tokenizer Qwen/Qwen3-8B
