#!/usr/bin/env bash
# Status check for DeepSeek on dgx-spark-llm.
set -euo pipefail

PORT="${LLM_PORT:-8888}"
PIDF="${DEEPSEEK_PIDF:-$HOME/deepseek-v4/llama-deepseek.pid}"

echo "===== dgx-spark-llm DeepSeek Status ====="
echo "Time: $(date -Is)"
free -h
echo

if [[ -f "$PIDF" ]]; then
  PID="$(cat "$PIDF")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "Process: RUNNING (PID $PID)"
  else
    echo "Process: STOPPED (stale PID file)"
  fi
else
  echo "Process: NOT RUNNING"
fi

echo "GPU compute applications:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
echo

if curl -fsS -m 3 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "API: ONLINE on http://localhost:$PORT"
  curl -fsS -m 5 "http://localhost:${PORT}/v1/models"
  echo
else
  echo "API: OFFLINE on http://localhost:$PORT"
fi
