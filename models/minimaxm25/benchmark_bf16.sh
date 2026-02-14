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

mkdir -p logs results
RESULT_FILE="results/bench_bf16.jsonl"
SERVER_LOG="logs/server_bf16.log"
PORT=8000
rm -f "$RESULT_FILE"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if ss -ltn | awk -v p="$PORT" '$4 ~ (":" p "$") {found=1} END {exit !found}'; then
  echo "Port $PORT is already in use. Stop the existing service and retry."
  exit 1
fi

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python -m sglang.launch_server \
  --model-path MiniMaxAI/MiniMax-M2.5 \
  --tp-size 4 \
  --dtype bfloat16 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --fp8-gemm-backend cutlass \
  --cuda-graph-bs 1 2 4 8 16 32 64 128 \
  --cuda-graph-max-bs 128 \
  --host 127.0.0.1 \
  --port "$PORT" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

ready=0
for _ in {1..180}; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server failed to start. See $SERVER_LOG"
    tail -n 120 "$SERVER_LOG"
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done

if [[ "$ready" -ne 1 ]]; then
  echo "Timed out waiting for server readiness. See $SERVER_LOG"
  exit 1
fi

python -m sglang.bench_one_batch_server \
  --model-path MiniMaxAI/MiniMax-M2.5 \
  --trust-remote-code \
  --base-url "http://127.0.0.1:${PORT}" \
  --batch-size 1 2 4 8 16 32 64 128 \
  --input-len 1000 \
  --output-len 1000 \
  --dataset-name random \
  --skip-warmup \
  --result-filename "$RESULT_FILE"

echo "Saved $RESULT_FILE"
