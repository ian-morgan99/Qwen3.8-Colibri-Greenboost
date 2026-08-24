#!/usr/bin/env bash
# T3 proof-of-life: run the 2.4T Q1_0 GGUF through llama.cpp with mmap (pseudo-T3).
# Read-only w.r.t. FreeToken; uses LM Studio's bundled llama-server + vendor CUDA libs.
# Usage: t3_proof_of_life.sh [prompt] [max_tokens]
set -euo pipefail

MODEL_DIR=/home/ian/Documents/VSCodeProjects/Qwen3.8/models-download/Qwen3.8-2.4T-A95B-UD-Q1_0
BACKEND=/home/ian/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.30.0
VENDOR=/home/ian/.lmstudio/extensions/backends/vendor/linux-llama-cuda12-vendor-v1
PORT=8012
PROMPT="${1:-Write a haiku about storage tiers.}"
NTOK="${2:-64}"

# sanity: all 10 shards present at full size
for i in $(seq -w 1 10); do
  f="$MODEL_DIR/Qwen3.8-2.4T-A95B-UD-Q1_0-0000${i}-of-00010.gguf"
  [[ -f "$f" ]] || { echo "MISSING $f"; exit 1; }
done
echo "all shards present"

export LD_LIBRARY_PATH="$VENDOR:$BACKEND"
# --no-warmup: don't pre-fault 397GB into page cache; let mmap page in on demand (the T3 behavior)
# -ngl 999: offload what fits in VRAM (dense+attention), experts stream from NVMe via mmap
exec "$BACKEND/llama-server" \
  -m "$MODEL_DIR/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf" \
  --host 127.0.0.1 --port "$PORT" \
  -ngl 99 -c 4096 --no-warmup -fa on \
  --mlock off --no-mmap off
