#!/usr/bin/env bash
# Free unified memory on DGX Spark (dgx-spark-llm) by stopping non-essential background services.
set -euo pipefail

echo "Reclaiming unified memory on DGX Spark (dgx-spark-llm)..."

# 1. Stop desktop display manager / GUI if running (frees ~1.5 - 2.0 GB RAM)
for dm in gdm3 gdm lightdm sddm; do
  if systemctl is-active --quiet "$dm" 2>/dev/null; then
    echo "Stopping $dm display manager..."
    sudo systemctl stop "$dm" 2>/dev/null || true
  fi
done

# Set default boot target to headless multi-user mode
if [[ "$(systemctl get-default 2>/dev/null || true)" != "multi-user.target" ]]; then
  echo "Setting system default target to headless multi-user mode..."
  sudo systemctl set-default multi-user.target 2>/dev/null || true
fi

# 2. Stop non-essential background Docker containers (e.g. TTS, embedding models)
if command -v docker >/dev/null 2>&1; then
  CONTAINERS=$(docker ps -q 2>/dev/null || true)
  if [[ -n "$CONTAINERS" ]]; then
    echo "Stopping background Docker containers: $CONTAINERS"
    docker stop $CONTAINERS 2>/dev/null || true
  fi
fi

# 3. Stop Ollama daemon if running
if systemctl is-active --quiet ollama 2>/dev/null; then
  echo "Stopping Ollama daemon..."
  sudo systemctl stop ollama 2>/dev/null || true
fi

# 4. Flush dirty filesystem buffers and drop inactive OS page cache
echo "Flushing dirty buffers and dropping inactive page cache..."
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

echo "Memory status after reclamation:"
free -h
