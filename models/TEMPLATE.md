# Model Profile Template

Create a new model profile as `models/<model-id>/` with:

- `setup_env.sh` (no arguments)
- one or more no-arg server scripts, e.g. `run_bf16.sh`, `run_fp8.sh`
- optional no-arg benchmark scripts, e.g. `benchmark_bf16.sh`, `benchmark_fp8.sh`
- `test_model.sh` (smoke test against an OpenAI-compatible endpoint)
- `requirements.lock.txt`
- `README.md`

Keep each profile self-contained so model-specific dependencies and run flags do not affect other models.
