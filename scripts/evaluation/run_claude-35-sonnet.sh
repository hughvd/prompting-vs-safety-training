#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/experiments/claude-35-sonnet.yaml}
shift || true

uv run python -m src.scripts.run_experiment "$CONFIG" "$@"
