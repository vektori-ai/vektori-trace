#!/bin/bash
cd /data/vektori-trace
source .venv/bin/activate
source .env.run
export PYTHONUNBUFFERED=1
MODEL_INFO='{"max_input_tokens": 40448, "max_output_tokens": 8192, "input_cost_per_token": 0.0, "output_cost_per_token": 0.0}'
harbor run \
  -p /tmp/one/prefecthq__prefect-65ea05bef8d9 \
  -a terminus-2 \
  --env docker \
  --yes \
  -o /data/vektori-trace/vektori-out/baseline/repro_job \
  --model hosted_vllm/Qwen3-8B \
  --ak api_base=http://127.0.0.1:37369/v1 \
  --ak "model_info=$MODEL_INFO" \
  --no-delete \
  --n-attempts 1 \
  2>&1 | tee /data/vektori-trace/vektori-out/baseline/repro.log
