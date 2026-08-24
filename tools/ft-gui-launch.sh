#!/usr/bin/env bash
# GUI launcher: pick options, then start/stop ft serve.
set -euo pipefail

PROJECT_DIR="/home/ian/Documents/VSCodeProjects/Qwen3.8"
MODEL_DIR="$PROJECT_DIR/models-download/Huihui-Qwen3.8-27B-abliterated-NVFP4"
VENV="$PROJECT_DIR/freetoken-env/.venv"
LOG_FILE="$PROJECT_DIR/models-download/ftserve_8001.log"
PORT=8001
URL="http://127.0.0.1:$PORT"

Z="zenity --width 520"

server_pids() { pgrep -f "ft serve --model-path $MODEL_DIR" || true; }

if [[ -n "$(server_pids)" ]]; then
  if $Z --question --title="FreeToken Huihui" \
       --text="Server is RUNNING on port $PORT.\n\nStop it?" --ok-label="Stop server" --cancel-label="Leave running"; then
    for pid in $(server_pids); do kill "$pid" 2>/dev/null || true; done
    sleep 3
    for pid in $(server_pids); do kill -9 "$pid" 2>/dev/null || true; done
    $Z --info --title="FreeToken Huihui" --text="Server stopped." || true
  fi
  exit 0
fi

# --- option picker ---
CHOICE=$($Z --forms --title="Start FreeToken — Huihui Qwen3.8 27B NVFP4" --text="Server options" \
  --add-combo="NVFP4 backend" --combo-values="triton|flashinfer|marlin|auto" \
  --add-combo="Attention backend" --combo-values="auto (fi)|fi|fa3|triton" \
  --add-entry="Port (default 8001)" \
  --add-combo="Context length" --combo-values="default (262k)|8192|16384|32768|65536|131072" \
  --add-combo="Max concurrent requests" --combo-values="4 (default)|1|2|8" \
  --separator="|" ) || exit 0

IFS='|' read -r NVFP4 ATTN PORT CTX CONC <<<"$CHOICE"
[[ -n "$PORT" ]] || PORT=8001
[[ "$PORT" =~ ^[0-9]+$ ]] || PORT=8001

ATTN_FLAG="${ATTN%% *}"   # strip " (fi)" style suffixes
[[ "$ATTN_FLAG" == "auto"* ]] && ATTN_ARG=() || ATTN_ARG=(--attention-backend "$ATTN_FLAG")

CTX_ARG=()
case "$CTX" in
  default*) : ;;
  *) CTX_ARG=(--max-seq-len-override "$CTX") ;;
esac

LOG_FILE="$PROJECT_DIR/models-download/ftserve_${PORT}.log"

# VRAM guard
if command -v nvidia-smi >/dev/null; then
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib < 30000 )); then
    $Z --error --title="FreeToken Huihui" \
      --text="Not enough free VRAM (${free_mib} MiB; need ~30000 MiB).\nUnload LM Studio / other GPU models first." || true
    exit 1
  fi
fi

: > "$LOG_FILE"
nohup "$VENV/bin/ft" serve --model-path "$MODEL_DIR" \
  --host 0.0.0.0 --port "$PORT" \
  --nvfp4-backend "$NVFP4" "${ATTN_ARG[@]}" "${CTX_ARG[@]}" \
  --max-running-requests "${CONC%% *}" \
  >>"$LOG_FILE" 2>&1 &
SRV_PID=$!

for _ in {1..240}; do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    $Z --error --title="FreeToken Huihui" --text="Server failed to start.\nSee $LOG_FILE" || true
    exit 1
  fi
  if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=2)" >/dev/null 2>&1; then
    $Z --info --title="FreeToken Huihui" \
      --text="Server ready.\n\nLocal:   http://127.0.0.1:$PORT/v1\nLAN:     http://$(hostname -I | awk '{print $1}'):$PORT/v1\nBackend: $NVFP4" || true
    exit 0
  fi
  sleep 1
done

$Z --warning --title="FreeToken Huihui" --text="Still starting after 4 min; see $LOG_FILE" || true
