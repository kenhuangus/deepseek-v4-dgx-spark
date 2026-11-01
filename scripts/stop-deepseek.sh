#!/usr/bin/env bash
# Stop DeepSeek llama-server and release port 8888 on dgx-spark-llm.
set -euo pipefail

PIDF="${DEEPSEEK_PIDF:-$HOME/deepseek-v4/llama-deepseek.pid}"
PORT="${LLM_PORT:-8888}"

if [[ -f "$PIDF" ]]; then
  PID="$(cat "$PIDF")"
  echo "Stopping DeepSeek process $PID on dgx-spark-llm..."
  kill "$PID" 2>/dev/null || true
  rm -f "$PIDF"
fi

fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 1

echo "DeepSeek stopped on dgx-spark-llm."
free -h
