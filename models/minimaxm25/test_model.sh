#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "This script takes no arguments."
  exit 2
fi

BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:8000}"
MODEL_ID="${SGLANG_MODEL_ID:-catid/MiniMax-M2.5-catid}"

echo "Checking model listing at ${BASE_URL}/v1/models ..."
curl -fsS "${BASE_URL}/v1/models" > /dev/null
echo "Model endpoint is reachable."

echo "Running chat completion smoke test ..."
curl -fsS "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "${MODEL_ID}",
  "messages": [
    {
      "role": "user",
      "content": "Reply with exactly: server is ready"
    }
  ],
  "max_tokens": 64,
  "temperature": 0
}
EOF

echo
