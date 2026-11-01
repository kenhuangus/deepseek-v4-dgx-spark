#!/usr/bin/env bash
# Prepare aarch64 GB10 runtime environment for llama.cpp on dgx-spark-llm.
set -euo pipefail

SRC_BIN="${1:-$HOME/llama-deepseek41-bin/bin}"
RUN_BIN="${2:-$HOME/llama-deepseek41-run/bin}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:13.0.1-devel-ubuntu24.04}"

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

if [[ ! -x "$SRC_BIN/llama-server" ]]; then
  log "ERROR: missing $SRC_BIN/llama-server"
  exit 1
fi

mkdir -p "$RUN_BIN"
cp -a "$SRC_BIN"/. "$RUN_BIN/"
cd "$RUN_BIN"

# Create SONAME symlinks
for f in lib*.so.[0-9]*.[0-9]*; do
  [[ -f "$f" && ! -L "$f" ]] || continue
  maj="$(echo "$f" | sed -E 's/(\.so\.[0-9]+).*/\1/')"
  base="$(echo "$f" | sed -E 's/\.so\.[0-9].*/.so/')"
  ln -sfn "$f" "$maj"
  ln -sfn "$maj" "$base"
done

# Pull libnccl from CUDA devel container if missing on host
if [[ ! -e libnccl.so.2 ]]; then
  log "Fetching libnccl from ${CUDA_IMAGE}..."
  docker run --rm -v "$RUN_BIN:/out" "$CUDA_IMAGE" \
    bash -lc 'cp -a /usr/lib/aarch64-linux-gnu/libnccl.so* /out/'
  if [[ -f libnccl.so.2.28.3 ]]; then
    ln -sfn libnccl.so.2.28.3 libnccl.so.2
    ln -sfn libnccl.so.2 libnccl.so
  elif compgen -G 'libnccl.so.2.*' >/dev/null; then
    ver="$(ls -1 libnccl.so.2.* | head -1)"
    ln -sfn "$ver" libnccl.so.2
    ln -sfn libnccl.so.2 libnccl.so
  fi
fi

export LD_LIBRARY_PATH="$RUN_BIN:${LD_LIBRARY_PATH:-}"
if ! ldd ./llama-server | grep -q 'libnccl.so.2 =>'; then
  log "ERROR: libnccl.so.2 unresolved"
  exit 1
fi

log "Runtime environment validated successfully at $RUN_BIN on dgx-spark-llm"
