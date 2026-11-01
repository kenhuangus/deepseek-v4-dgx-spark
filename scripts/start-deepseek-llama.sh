#!/usr/bin/env bash
# Launch DeepSeek-V4.1-Flash-1017 on DGX Spark (dgx-spark-llm) via CUDA llama-server on port 8888.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_BIN="${LLAMA_RUN_BIN:-$HOME/llama-deepseek41-run/bin}"
SRC_BIN="${LLAMA_SRC_BIN:-$HOME/llama-deepseek41-bin/bin}"
PORT="${LLM_PORT:-8888}"
LOG="${DEEPSEEK_LOG:-$HOME/deepseek-v4/llama-deepseek.log}"
PIDF="${DEEPSEEK_PIDF:-$HOME/deepseek-v4/llama-deepseek.pid}"
CTX="${DEEPSEEK_CTX:-65536}"

# 1. Prepare runtime environment if needed
if [[ ! -x "$RUN_BIN/llama-server" ]]; then
  echo "Preparing GB10 runtime environment on dgx-spark-llm..."
  bash "$ROOT/scripts/prepare-llama-run-env.sh" "$SRC_BIN" "$RUN_BIN"
fi

# 2. Locate model shard 1
SHARD1=""
for candidate in \
  "$HOME/deepseek-v4/UD-IQ1_S/UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf" \
  "$HOME/deepseek-v4/UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf"; do
  if [[ -f "$candidate" ]]; then
    SHARD1="$candidate"
    break
  fi
done

if [[ -z "$SHARD1" ]]; then
  echo "ERROR: Could not find DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf on dgx-spark-llm"
  echo "Run ./scripts/download-deepseek.sh first"
  exit 1
fi

echo "Using model: $SHARD1"

# 3. Clean up prior process on port 8888
if [[ -f "$PIDF" ]]; then
  kill "$(cat "$PIDF")" 2>/dev/null || true
  rm -f "$PIDF"
fi
fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 1

export LD_LIBRARY_PATH="$RUN_BIN:${LD_LIBRARY_PATH:-}"

echo "Starting llama-server on localhost:$PORT (Context: $CTX, Reasoning effort: 75)..."
nohup "$RUN_BIN/llama-server" \
  --model "$SHARD1" \
  --load-mode mmap \
  --lazy-mode on \
  --override-tensor "per_layer_token_embd.weight=CPU" \
  --alias deepseek-v4.1-flash,deepseek,flint \
  -ngl 99 \
  -c "$CTX" \
  -np 1 \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  --jinja \
  --temp 1.0 \
  --top-p 1.0 \
  --top-k 0 \
  --min-p 0.0 \
  --chat-template-kwargs '{"reasoning_effort":75}' \
  --host 0.0.0.0 \
  --port "$PORT" \
  > "$LOG" 2>&1 &

echo $! > "$PIDF"
echo "DeepSeek llama-server started with PID $(cat "$PIDF") on dgx-spark-llm"

# 4. Wait for readiness
echo "Waiting for readiness on http://localhost:${PORT}/v1/models ..."
for i in $(seq 1 180); do
  if curl -fsS -m 2 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "SUCCESS: DeepSeek is READY on dgx-spark-llm!"
    curl -fsS -m 5 "http://localhost:${PORT}/v1/models"
    exit 0
  fi
  if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "ERROR: llama-server exited unexpectedly on dgx-spark-llm. Log:"
    tail -80 "$LOG"
    exit 1
  fi
  sleep 2
done

echo "TIMEOUT: Server did not respond within 360s on dgx-spark-llm. Log:"
tail -80 "$LOG"
exit 1
