#!/usr/bin/env bash
# Build llama.cpp with CUDA Blackwell sm_121a support for DeepSeek-V4.1 on NVIDIA DGX Spark (dgx-spark-llm).
set -euo pipefail

ROOT="${LLAMA_SRC_DIR:-$HOME/llama.cpp}"
OUT="${LLAMA_OUT_DIR:-$HOME/llama-deepseek41-bin}"
IMAGE="nvidia/cuda:13.0.1-devel-ubuntu24.04"

echo "=== Building llama.cpp for dgx-spark-llm ==="

if [[ ! -d "$ROOT" ]]; then
  git clone https://github.com/ggml-org/llama.cpp "$ROOT"
fi

cd "$ROOT"
git checkout -f master 2>/dev/null || true
git fetch origin pull/30817/head:deepseek41
git checkout deepseek41
echo "Checked out deepseek41 at commit $(git rev-parse --short HEAD)"

mkdir -p "$OUT"

docker run --rm --gpus all \
  -v "${ROOT}:/src:ro" \
  -v "${OUT}:/out" \
  -w /tmp/build \
  "${IMAGE}" \
  bash -lc '
set -euo pipefail
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential curl libcurl4-openssl-dev >/dev/null
mkdir -p /tmp/llama
tar -C /src --exclude="./build*" --exclude="./.git" -cf - . | tar -C /tmp/llama -xf -
cd /tmp/llama
cmake -B build-spark \
  -DGGML_NATIVE=OFF \
  -DGGML_CUDA=ON \
  -DGGML_CURL=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121a-real
cmake --build build-spark --config Release --target llama-server llama-cli llama-gguf-split -j$(nproc)
mkdir -p /out/bin
find build-spark/bin -maxdepth 1 -type f -exec cp -a {} /out/bin/ \;
chmod +x /out/bin/llama-*
'

echo "Build completed successfully into ${OUT}/bin on dgx-spark-llm"
