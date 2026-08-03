#!/bin/bash
cd /data/vektori-trace
source .env.run
export PYTHONUNBUFFERED=1
uv run python scripts/vllm_monitor.py \
  --api-base "$STUDENT_API_BASE" \
  --kv-total-tokens 193695 \
  --interval 5 \
  --log /data/vektori-trace/vektori-out/baseline/vllm_metrics.jsonl 2>&1 | tee /data/vektori-trace/vektori-out/baseline/gpu_metrics.log
