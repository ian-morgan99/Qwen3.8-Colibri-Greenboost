#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/ian/Documents/VSCodeProjects/Qwen3.8"
MODEL_DIR="$PROJECT_DIR/models-download/Huihui-Qwen3.8-27B-abliterated-NVFP4"
VENV="$PROJECT_DIR/freetoken-env/.venv"
LOG_FILE="$PROJECT_DIR/models-download/ftserve_8001.log"
URL="http://127.0.0.1:8001"
PORT=8001
PID_FILE="/tmp/ftserve_${PORT}.pid"

notify() {
  command -v notify-send >/dev/null && notify-send "FreeToken / Huihui Qwen3.8" "$1"
}

server_pids() {
  pgrep -f "ft serve --model-path $MODEL_DIR" || true
}

# Stop if running
if [[ -n "$(server_pids)" ]]; then
  for pid in $(server_pids); do kill "$pid" 2>/dev/null || true; done
  sleep 3
  for pid in $(server_pids); do kill -9 "$pid" 2>/dev/null || true; done
  notify "Stopped"
  exit 0
fi

if command -v nvidia-smi >/dev/null; then
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  required_mib=30000
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib < required_mib )); then
    notify "Not enough free VRAM (${free_mib} MiB; need ~${required_mib} MiB). Unload another GPU model first."
    exit 1
  fi
fi

: > "$LOG_FILE"
nohup "$VENV/bin/ft" serve --model-path "$MODEL_DIR" \
  --host 0.0.0.0 --port "$PORT" >>"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

for _ in {1..240}; do
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    notify "Failed to start; see $LOG_FILE"
    exit 1
  fi
  if python3 -c "import urllib.request; urllib.request.urlopen('$URL/health', timeout=2)" >/dev/null 2>&1; then
    notify "Started on $URL/v1"
    exit 0
  fi
  sleep 1
done

notify "Still starting; see $LOG_FILE"
