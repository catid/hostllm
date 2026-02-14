# MiniMax M2.5 (SGLang)

All scripts in this folder take no positional arguments.

## 1) Setup

```bash
./setup_env.sh
```

This creates `.venv` with Python `3.12.11` and installs pinned packages from `requirements.lock.txt`.

## 2) Run server (remote-accessible)

Run one of:

```bash
./run_bf16.sh
./run_fp8.sh
```

Both launch SGLang on 4 GPUs (`--tp-size 4`) and bind to `0.0.0.0:8000` by default, so other machines can connect.

Optional environment variables:
- `SGLANG_HOST` (default: `0.0.0.0`)
- `SGLANG_PORT` (default: `8000`)

## 3) Test model

From the server machine:

```bash
./test_model.sh
```

From another machine on the same network:

```bash
SGLANG_BASE_URL=http://<SERVER_IP>:8000 ./test_model.sh
```

Manual API test from another machine:

```bash
curl -sS http://<SERVER_IP>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-M2.5",
    "messages": [{"role": "user", "content": "Reply with exactly: server is ready"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

## 4) Benchmarking

Benchmarking is separate from server run scripts:

```bash
./benchmark_bf16.sh
./benchmark_fp8.sh
```

These scripts launch a temporary local server (`127.0.0.1:8000`), run the fixed benchmark sweep, then stop the server.

Outputs:
- `results/bench_bf16.jsonl`
- `results/bench_fp8.jsonl`
- `logs/server_bf16.log`
- `logs/server_fp8.log`
