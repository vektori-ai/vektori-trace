#!/bin/bash
cd /data/vektori-trace
source .env.run
echo "UPSTREAM=$STUDENT_API_BASE"
export PYTHONUNBUFFERED=1
uv run vektori-trace capture-proxy --upstream "$STUDENT_API_BASE" --out /data/vektori-trace/vektori-out/baseline/captures 2>&1 | tee /data/vektori-trace/vektori-out/baseline/proxy.log
