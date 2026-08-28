#!/bin/bash
# One-shot status of the replay OPD run on the box.
CID=$(aws ssm send-command --instance-ids i-0a348ff3d7be9769a --region ap-south-1 \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["echo \"time: $(date -Is)\"; echo \"scoring: $(pgrep -f run_replay_opd >/dev/null && echo RUNNING || echo stopped)\"; echo \"GPU serving: $(pgrep -f serve_student >/dev/null && echo ALIVE_BILLING || echo none)\"; echo \"captures: $(wc -l < /data/replay-v1/captures.jsonl 2>/dev/null || echo 0)/32\"; echo \"scores: $(wc -l < /data/replay-v1/teacher_scores.jsonl 2>/dev/null || echo 0)/32\"; echo; echo \"--- last log lines ---\"; tail -12 /data/replay_score.log"]' \
  --query Command.CommandId --output text)
sleep 6
aws ssm get-command-invocation --command-id "$CID" --instance-id i-0a348ff3d7be9769a \
  --region ap-south-1 --query 'StandardOutputContent' --output text 2>/dev/null \
  || echo "(still fetching — rerun in a few seconds)"
