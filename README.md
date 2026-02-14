# hostllm

Reproducible local-hosted LLM scripts organized by model profile.

## Layout

- `models/<model-id>/setup_env.sh`
- `models/<model-id>/run_*.sh`
- `models/<model-id>/requirements.lock.txt`

Each model directory is self-contained so you can add more models without touching existing ones.

## Current model profiles

- `models/minimaxm25`
- `models/TEMPLATE.md` (starter checklist for new profiles)

## Add a new model

1. Create `models/<new-model-id>/`.
2. Add a no-argument `setup_env.sh`.
3. Add no-argument run scripts (for example `run_bf16.sh`, `run_fp8.sh`).
4. Add a pinned `requirements.lock.txt`.
5. Add a short model-specific `README.md`.

## MiniMax M2.5 quick start

```bash
cd models/minimaxm25
./setup_env.sh
./run_bf16.sh
./run_fp8.sh
```
