# Reproducible MiniMax-M2.5 + SGLang Setup

Scripts in this repo are no-argument and target an identical 4-GPU host.

## 1) Setup environment

```bash
./setup_env.sh
```

This creates `.venv` with Python `3.12.11` and installs pinned packages from `requirements.lock.txt`.

## 2) Run BF16 benchmark

```bash
./run_bf16.sh
```

Outputs:
- `results/bench_bf16.jsonl`
- `logs/server_bf16.log`

## 3) Run FP8 benchmark

```bash
./run_fp8.sh
```

Outputs:
- `results/bench_fp8.jsonl`
- `logs/server_fp8.log`

## Notes

- Both run scripts force `CUDA_VISIBLE_DEVICES=0,1,2,3` and launch SGLang with `--tp-size 4`.
- Benchmark workload is fixed to:
  - input length: `1000`
  - output length: `1000`
  - batch sizes: `1 2 4 8 16 32 64 128`
