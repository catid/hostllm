#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "This script takes no arguments."
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not found in PATH."
  exit 1
fi

if [[ ! -f requirements.lock.txt ]]; then
  echo "requirements.lock.txt is missing."
  exit 1
fi

rm -rf .venv
uv venv --python 3.12.11 .venv

. .venv/bin/activate

uv pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  -r requirements.lock.txt

python - <<'PY'
import sys
import torch
import sglang

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"sglang={sglang.__version__}")
PY

echo "Environment setup complete."
