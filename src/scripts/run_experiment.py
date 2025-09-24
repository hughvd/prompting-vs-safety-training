from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from src.evaluations.capability import run_capability_evaluation
from src.evaluations.refusal import run_refusal_evaluation
from src.model_client import ChatModelClient
from src.utils import (
    ensure_output_dirs,
    load_env_file,
    load_spec_texts,
    load_yaml_config,
    resolve_output_path,
)


LOGGER = logging.getLogger("experiment_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation experiment(s) for a model.")
    parser.add_argument(
        "experiment",
        help="Path to the experiment YAML configuration.",
    )
    parser.add_argument(
        "--skip-refusal",
        action="store_true",
        help="Skip refusal evaluations even if enabled in the config.",
    )
    parser.add_argument(
        "--skip-capability",
        action="store_true",
        help="Skip capability evaluations even if enabled in the config.",
    )
    return parser.parse_args()


def _is_enabled(section: Dict[str, Any], override_skip: bool) -> bool:
    if override_skip:
        return False
    return bool(section.get("enabled", True))


def main() -> None:
    load_env_file()
    args = parse_args()

    experiment_cfg = load_yaml_config(args.experiment)

    log_level = experiment_cfg.get("logging", {}).get("level", "WARNING")
    logging.basicConfig(level=getattr(logging, str(log_level).upper(), logging.INFO))
    for noisy in ("openai", "openai._logs", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
        logging.getLogger(noisy).propagate = False

    output_cfg = experiment_cfg.get("output")
    if output_cfg is None:
        raise ValueError("Experiment config must define an 'output' section")

    base_output = resolve_output_path(output_cfg)
    ensure_output_dirs(base_output, create_standard_layout=False)

    model_cfg = experiment_cfg.get("model")
    if not model_cfg:
        raise ValueError("Experiment config must include model configuration")

    request_timeout = float(experiment_cfg.get("http", {}).get("timeout", 60.0))
    dry_run = bool(experiment_cfg.get("run", {}).get("dry_run", False))
    generation_cfg = experiment_cfg.get("generation", {})

    model = ChatModelClient(
        model_cfg,
        request_timeout=request_timeout,
        dry_run=dry_run,
    )

    specs_cfg = experiment_cfg.get("specs")
    if not specs_cfg:
        raise ValueError("Experiment config must include a specs section")

    spec_texts = load_spec_texts(specs_cfg["directory"], specs_cfg["files"])

    evaluations_cfg = experiment_cfg.get("evaluations", {})

    refusal_cfg_entry = evaluations_cfg.get("refusal", {})
    capability_cfg_entry = evaluations_cfg.get("capability", {})

    refusal_enabled = _is_enabled(refusal_cfg_entry, args.skip_refusal)
    capability_enabled = _is_enabled(capability_cfg_entry, args.skip_capability)

    if not refusal_enabled and not capability_enabled:
        LOGGER.warning("No evaluations enabled. Nothing to do.")
        return

    if refusal_enabled:
        refusal_cfg_path = refusal_cfg_entry.get("config")
        if not refusal_cfg_path:
            raise ValueError("Refusal evaluation enabled but no config path provided")
        refusal_cfg = load_yaml_config(refusal_cfg_path)
    else:
        refusal_cfg = None

    if capability_enabled:
        capability_cfg_path = capability_cfg_entry.get("config")
        if not capability_cfg_path:
            raise ValueError("Capability evaluation enabled but no config path provided")
        capability_cfg = load_yaml_config(capability_cfg_path)
    else:
        capability_cfg = None

    summaries = {}

    for spec_entry in specs_cfg["files"]:
        spec_id = spec_entry["id"]
        spec_text = spec_texts[spec_id]
        spec_output_base = Path(base_output) / spec_id
        ensure_output_dirs(spec_output_base, create_standard_layout=False)

        spec_summary = {}

        if refusal_enabled and refusal_cfg is not None:
            refusal_output_subdir = refusal_cfg_entry.get("output_subdir", "refusal")
            refusal_output_base = spec_output_base / refusal_output_subdir
            result = run_refusal_evaluation(
                spec_id=spec_id,
                spec_text=spec_text,
                datasets_cfg=refusal_cfg,
                model=model,
                generation_cfg=generation_cfg,
                output_base=refusal_output_base,
            )
            spec_summary["refusal"] = result

        if capability_enabled and capability_cfg is not None:
            capability_output_subdir = capability_cfg_entry.get("output_subdir", "capability")
            capability_output_base = spec_output_base / capability_output_subdir
            dataset_cfg = capability_cfg.get("dataset")
            if not dataset_cfg:
                raise ValueError("Capability config must include a 'dataset' section")
            result = run_capability_evaluation(
                spec_id=spec_id,
                spec_text=spec_text,
                dataset_cfg=dataset_cfg,
                model=model,
                generation_cfg=generation_cfg,
                output_base=capability_output_base,
            )
            spec_summary["capability"] = result

        summaries[spec_id] = spec_summary

    summary_path = Path(base_output) / "experiment_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)

    LOGGER.debug("Experiment summary written to %s", summary_path)


if __name__ == "__main__":  # pragma: no cover
    main()
