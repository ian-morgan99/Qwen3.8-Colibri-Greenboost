#!/usr/bin/env bash
# T3 proof-of-life: run the 2.4T Q1_0 GGUF through llama.cpp with mmap (pseudo-T3).
# Uses the unslothai/llama.cpp iq1-narrow build (required for GGML_TYPE_IQ1_XXXS = type 66).
# Usage: t3_proof_of_life.sh [prompt] [max_tokens]
set -euo pipefail

# Canonical location: LM Studio models dir (moved 2026-08-25); fall back to the old download dir if present.
MODEL_DIR=/home/ian/.lmstudio/models/unsloth/Qwen3.8-2.4T-A95B-GGUF/UD-Q1_0
[[ -d "$MODEL_DIR" ]] || MODEL_DIR=/home/ian/Documents/VSCodeProjects/Qwen3.8/models-download/Qwen3.8-2.4T-A95B-UD-Q1_0
BACKEND=/home/ian/Documents/VSCodeProjects/Qwen3.8/llama.cpp-iq1narrow/build/bin
PORT=1234
PROMPT="${1:-Write a haiku about storage tiers.}"
NTOK="${2:-64}"

# sanity: all 10 shards present at full size
for i in $(seq 1 10); do
  f="$MODEL_DIR/Qwen3.8-2.4T-A95B-UD-Q1_0-$(printf '%05d' $i)-of-00010.gguf"
  [[ -f "$f" ]] || { echo "MISSING $f"; exit 1; }
done
echo "all shards present"

# --no-warmup: don't pre-fault 397GB into page cache; let mmap page in on demand (the T3 behavior)
# -ngl 99 + --cpu-moe: attention/dense on GPU, all expert weights stay in CPU RAM (mmap from NVMe)
exec "$BACKEND/llama-server" \
  -m "$MODEL_DIR/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf" \
  --host 127.0.0.1 --port "$PORT" \
  -ngl 99 --cpu-moe -c 4096 --no-warmup -fa on \
  --load-mode mmap
