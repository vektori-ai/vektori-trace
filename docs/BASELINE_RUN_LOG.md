# Baseline run — execution log

Errors hit while executing `docs/BASELINE_RUN.md` on the box (i-0a348ff3d7be9769a), and the one-line fix for each.

- **capture-proxy printed nothing / looked hung** — stdout fully buffered because piped through `tee` (not a tty). Fix: `export PYTHONUNBUFFERED=1` before `uv run vektori-trace capture-proxy`.
- **harbor: `ValueError: hosted_vllm model_info missing 'max_input_tokens'`** — `model_info.json` held the whole `harbor kwargs:` block instead of just its inner `model_info` value. Fix: `model_info.json` must be exactly `{"max_input_tokens":..., "max_output_tokens":..., "input_cost_per_token":..., "output_cost_per_token":...}`.
- **harbor: `docker compose ... down` → `unknown flag: --project-name` / `docker: unknown command: docker compose`** — the box's `docker.io` package ships no compose plugin at all (checked both `ubuntu` and `root`, `/usr/libexec/docker/cli-plugins/` had only `docker-trust`). Fix: `sudo apt-get install -y docker-compose-v2` (this box's Ubuntu jammy package name for the v2 compose plugin; `docker-compose-plugin` is not in these apt sources).
