from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation import run_from_config
from src.utils import load_env_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the spec adherence evaluation pipeline.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/evaluation/spec_adherence.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    results_path = run_from_config(args.config)
    print(f"Results saved to {Path(results_path)}")


if __name__ == "__main__":
    main()
