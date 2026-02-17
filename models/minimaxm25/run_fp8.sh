#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "This script takes no arguments."
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing .venv. Run ./setup_env.sh first."
  exit 1
fi

. .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HOST="${SGLANG_HOST:-0.0.0.0}"
PORT="${SGLANG_PORT:-8000}"

if ss -ltn | awk -v p="$PORT" '$4 ~ (":" p "$") {found=1} END {exit !found}'; then
  echo "Port $PORT is already in use. Stop the existing service and retry."
  exit 1
fi

echo "Starting FP8 server on ${HOST}:${PORT} ..."
echo "Use Ctrl+C to stop."

exec python -m sglang.launch_server \
  --model-path catid/MiniMax-M2.5-catid \
  --tp-size 4 \
  --quantization fp8 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --fp8-gemm-backend cutlass \
  --cuda-graph-bs 1 2 4 8 16 32 64 128 \
  --cuda-graph-max-bs 128 \
  --host "$HOST" \
  --port "$PORT"
