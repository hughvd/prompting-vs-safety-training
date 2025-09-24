#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/experiments/deepseekR1.yaml}
shift || true

uv run python -m src.scripts.run_experiment "$CONFIG" "$@"
