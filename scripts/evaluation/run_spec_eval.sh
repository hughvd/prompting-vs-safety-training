#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/evaluation/spec_adherence.yaml}

uv run python -m src.scripts.run_evaluation "$CONFIG"
