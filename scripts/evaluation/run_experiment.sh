#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/experiments/test.yaml}
shift || true

uv run python -m src.scripts.run_experiment "$CONFIG" "$@"
