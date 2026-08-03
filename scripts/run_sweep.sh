#!/bin/bash
cd /data/vektori-trace
source .env.run
export PYTHONUNBUFFERED=1
TASK=prefecthq__prefect-65ea05bef8d9
rm -rf /tmp/one
mkdir -p /tmp/one
cp -r "cs/smoke/cmined/mined_tasks/$TASK" /tmp/one/
# terminus-2's only interaction mechanism is a tmux pane; the mined task's
# bootstrap image doesn't ship tmux and the runtime network allowlist blocks
# installing it later, so bake it in at image-build time here, every restage.
DOCKERFILE="/tmp/one/$TASK/environment/Dockerfile"
if ! grep -q "tmux" "$DOCKERFILE"; then
  python3 - "$DOCKERFILE" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    content = f.read()
marker = "# Position the working tree at the PR's base commit"
patch = '''# Defensive: terminus-2's only interaction mechanism is a tmux pane
# (send keystrokes, read pane back) -- no tmux means the agent has no
# way to act at all. The task's runtime network allowlist blocks
# installing it later, so it must be baked in at build time here.
RUN command -v tmux >/dev/null 2>&1 || \\
    (apt-get update && apt-get install -y --no-install-recommends tmux \\
     && rm -rf /var/lib/apt/lists/*) || \\
    apk add --no-cache tmux || true
'''
assert marker in content, "marker not found"
content = content.replace(marker, patch + marker)
with open(path, "w") as f:
    f.write(content)
print("tmux patch applied")
PYEOF
fi
echo "STAGED:"
ls /tmp/one
PROXY_API_BASE="http://127.0.0.1:37369/v1"
uv run vektori-trace passk \
  --tasks-dir /tmp/one \
  --agent terminus-2 \
  --model hosted_vllm/Qwen3-8B \
  --api-base "$PROXY_API_BASE" \
  --model-info @/data/vektori-trace/model_info.json \
  --stage1-n 4 \
  --no-escalate \
  --max-workers 1 \
  --out /data/vektori-trace/vektori-out/baseline 2>&1 | tee /data/vektori-trace/vektori-out/baseline/sweep.log
echo "SWEEP_EXIT=$?"
