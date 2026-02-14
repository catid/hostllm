# Model Profile Template

Create a new model profile as `models/<model-id>/` with:

- `setup_env.sh` (no arguments)
- one or more no-arg run scripts, e.g. `run_bf16.sh`, `run_fp8.sh`
- `requirements.lock.txt`
- `README.md`

Keep each profile self-contained so model-specific dependencies and run flags do not affect other models.
