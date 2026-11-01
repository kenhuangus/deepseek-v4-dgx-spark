#!/usr/bin/env bash
# Resumable multi-shard downloader for DeepSeek-V4.1-Flash-1017 UD-IQ1_S (1-bit GGUF).
set -euo pipefail

DEST_DIR="${DEEPSEEK_DIR:-$HOME/deepseek-v4/UD-IQ1_S}"
LOG_FILE="${DEEPSEEK_LOG:-$HOME/deepseek-v4/download.log}"
export HF_HOME="${HF_HOME:-$HOME/deepseek-v4/.hf_cache}"

mkdir -p "$DEST_DIR" "$HF_HOME"

echo "=== DeepSeek-V4.1-Flash-1017 UD-IQ1_S Download on dgx-spark-llm ===" | tee -a "$LOG_FILE"
echo "Start time: $(date -Is)" | tee -a "$LOG_FILE"

python3 - << 'PY' 2>&1 | tee -a "$LOG_FILE"
import sys
from huggingface_hub import hf_hub_download

repo_id = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
local_dir = "/home/kengpu/deepseek-v4/UD-IQ1_S"

shards = [
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf",
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00002-of-00005.gguf",
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00003-of-00005.gguf",
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00004-of-00005.gguf",
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00005-of-00005.gguf",
]

for i, shard in enumerate(shards, 1):
    subpath = f"UD-IQ1_S/{shard}"
    print(f"[{i}/5] Downloading {shard} ...", flush=True)
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=subpath,
            local_dir=local_dir,
            resume_download=True,
        )
        print(f"[{i}/5] Completed {shard} -> {path}", flush=True)
    except Exception as e:
        print(f"ERROR downloading {shard}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

print("All 5 shards downloaded successfully on dgx-spark-llm!", flush=True)
PY

echo "End time: $(date -Is)" | tee -a "$LOG_FILE"
ls -lh "$DEST_DIR" | tee -a "$LOG_FILE"
