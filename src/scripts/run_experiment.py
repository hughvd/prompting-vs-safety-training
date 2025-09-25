from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from datasets import load_dataset

from src.evaluations.capability import evaluate_capability_chunk
from src.evaluations.refusal import evaluate_refusal_chunk
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
    parser.add_argument("experiment", help="Path to the experiment YAML configuration.")
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


def _chunk_list(items: Sequence[Dict[str, Any]], chunk_size: int | None) -> List[List[Dict[str, Any]]]:
    if not items:
        return []
    if not chunk_size or chunk_size <= 0:
        return [list(items)]
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


def _derive_seed(base_seed: int | None, fallback: int) -> int:
    if base_seed is None:
        return fallback
    return int(base_seed) + fallback


def _load_refusal_samples(dataset_cfg: Dict[str, Any], run_seed: int) -> List[Dict[str, Any]]:
    dataset = load_dataset(
        dataset_cfg["hf_dataset"],
        dataset_cfg["hf_config"],
        split=dataset_cfg.get("split", "train"),
    )

    if dataset_cfg.get("shuffle"):
        shuffle_seed = _derive_seed(dataset_cfg.get("seed"), run_seed)
        dataset = dataset.shuffle(seed=shuffle_seed)

    limit = dataset_cfg.get("limit")
    if limit is not None:
        limit = min(int(limit), len(dataset))
        dataset = dataset.select(range(limit))

    records: List[Dict[str, Any]] = []
    for index, sample in enumerate(dataset):
        records.append(
            {
                "index": index,
                "prompt": sample.get("prompt", ""),
                "category": sample.get("category"),
            }
        )
    return records


def _load_capability_samples(dataset_cfg: Dict[str, Any], run_seed: int) -> List[Dict[str, Any]]:
    dataset = load_dataset(
        dataset_cfg["hf_dataset"],
        split=dataset_cfg.get("split", "validation"),
    )

    if dataset_cfg.get("shuffle"):
        shuffle_seed = _derive_seed(dataset_cfg.get("seed"), run_seed)
        dataset = dataset.shuffle(seed=shuffle_seed)

    limit = dataset_cfg.get("limit")
    if limit is not None:
        limit = min(int(limit), len(dataset))
        dataset = dataset.select(range(limit))

    records: List[Dict[str, Any]] = []
    for index, sample in enumerate(dataset):
        records.append(
            {
                "index": index,
                "question_id": sample.get("question_id"),
                "question": sample.get("question", ""),
                "options": sample.get("options", []),
                "answer": sample.get("answer"),
                "category": sample.get("category"),
            }
        )
    return records


def _ensure_spec_dirs(spec_dir: Path, sections: Iterable[str]) -> None:
    for section in sections:
        (spec_dir / section).mkdir(parents=True, exist_ok=True)


def _write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _execute_run(
    run_dir: Path,
    run_seed: int,
    *,
    spec_entries: List[Dict[str, Any]],
    spec_texts: Dict[str, str],
    generation_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    request_timeout: float,
    dry_run: bool,
    refusal_cfg_entry: Dict[str, Any],
    capability_cfg_entry: Dict[str, Any],
    refusal_cfg: Dict[str, Any] | None,
    capability_cfg: Dict[str, Any] | None,
    refusal_enabled: bool,
    capability_enabled: bool,
    max_workers: int,
    max_prompts_per_worker: int | None,
    reasoning: bool,
) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}

    tasks = []
    spec_dirs: Dict[str, Path] = {}

    for spec_entry in spec_entries:
        spec_id = spec_entry["id"]
        spec_text = spec_texts[spec_id]
        spec_dir = run_dir / spec_id
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_dirs[spec_id] = spec_dir

        sections = []
        if refusal_enabled:
            sections.append(refusal_cfg_entry.get("output_subdir", "refusal"))
        if capability_enabled:
            sections.append(capability_cfg_entry.get("output_subdir", "capability"))
        _ensure_spec_dirs(spec_dir, sections)

        if refusal_enabled and refusal_cfg is not None:
            refusal_dataset_cfg = refusal_cfg.get("datasets", {})
            for dataset_name, dataset_cfg in refusal_dataset_cfg.items():
                samples = _load_refusal_samples(dataset_cfg, run_seed)
                if not samples:
                    continue
                chunks = _chunk_list(samples, max_prompts_per_worker)
                for chunk_idx, chunk in enumerate(chunks):
                    tasks.append(
                        (
                            "refusal",
                            spec_id,
                            spec_text,
                            dataset_name,
                            chunk,
                            chunk_idx,
                        )
                    )

        if capability_enabled and capability_cfg is not None:
            dataset_cfg = capability_cfg.get("dataset")
            if dataset_cfg is None:
                raise ValueError("Capability config must include a 'dataset' section")
            samples = _load_capability_samples(dataset_cfg, run_seed)
            if samples:
                dataset_name = dataset_cfg.get("name", "mmlu_pro")
                chunks = _chunk_list(samples, max_prompts_per_worker)
                for chunk_idx, chunk in enumerate(chunks):
                    tasks.append(
                        (
                            "capability",
                            spec_id,
                            spec_text,
                            dataset_name,
                            chunk,
                            chunk_idx,
                        )
                    )

    if not tasks:
        LOGGER.warning("No evaluation tasks generated.")
        return summaries

    total_tasks = len(tasks)
    LOGGER.info("Submitting %d evaluation chunks with %d worker(s)", total_tasks, max_workers)
    progress_interval = max(1, total_tasks // 10)
    completed_chunks = 0

    refusal_results: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    capability_results: Dict[str, List[Dict[str, Any]]] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {}
        for task in tasks:
            task_type = task[0]
            if task_type == "refusal":
                _, spec_id, spec_text, dataset_name, chunk, _ = task
                future = executor.submit(
                    evaluate_refusal_chunk,
                    spec_id,
                    spec_text,
                    dataset_name,
                    chunk,
                    model_cfg,
                    generation_cfg,
                    request_timeout=request_timeout,
                    dry_run=dry_run,
                    reasoning=reasoning,
                )
            else:
                _, spec_id, spec_text, dataset_name, chunk, _ = task
                future = executor.submit(
                    evaluate_capability_chunk,
                    spec_id,
                    spec_text,
                    chunk,
                    model_cfg,
                    generation_cfg,
                    request_timeout=request_timeout,
                    dry_run=dry_run,
                    reasoning=reasoning,
                )
            future_to_task[future] = task

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            task_type = task[0]
            try:
                chunk_results = future.result()
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Task %s failed: %s", task, exc)
                raise

            if task_type == "refusal":
                _, spec_id, _, dataset_name, _, _ = task
                key = (spec_id, dataset_name)
                refusal_results.setdefault(key, []).extend(chunk_results)
            else:
                _, spec_id, _, dataset_name, _, _ = task
                key = spec_id
                capability_results.setdefault(key, []).extend(chunk_results)

            completed_chunks += 1
            if completed_chunks % progress_interval == 0 or completed_chunks == total_tasks:
                message = f"Progress: {completed_chunks}/{total_tasks} chunks completed"
                LOGGER.info(message)
                print(message, flush=True)

    for spec_id, spec_dir in spec_dirs.items():
        spec_summary: Dict[str, Any] = {}

        if refusal_enabled:
            dataset_summary = []
            refusal_output_subdir = refusal_cfg_entry.get("output_subdir", "refusal")
            refusal_dir = spec_dir / refusal_output_subdir

            for (result_spec, dataset_name), records in refusal_results.items():
                if result_spec != spec_id:
                    continue
                records.sort(key=lambda r: r["index"])
                output_path = refusal_dir / f"{dataset_name}.jsonl"
                _write_jsonl(records, output_path)
                dataset_summary.append(
                    {
                        "dataset": dataset_name,
                        "samples": len(records),
                        "output_path": str(output_path),
                    }
                )

            if dataset_summary:
                summary_path = refusal_dir / "summary.json"
                summary_path.write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding="utf-8")
                spec_summary["refusal"] = {
                    "datasets": dataset_summary,
                    "summary_path": str(summary_path),
                }

        if capability_enabled:
            capability_output_subdir = capability_cfg_entry.get("output_subdir", "capability")
            capability_dir = spec_dir / capability_output_subdir
            records = capability_results.get(spec_id, [])
            if records:
                records.sort(key=lambda r: r["index"])
                dataset_name = capability_cfg.get("dataset", {}).get("name", "mmlu_pro") if capability_cfg else "mmlu_pro"
                output_path = capability_dir / f"{dataset_name}.jsonl"
                _write_jsonl(records, output_path)
                correct = sum(1 for r in records if r.get("is_correct"))
                accuracy = correct / len(records)
                metrics = {
                    "spec_id": spec_id,
                    "total": len(records),
                    "correct": correct,
                    "accuracy": accuracy,
                    "output_path": str(output_path),
                }
                metrics_path = capability_dir / "metrics.json"
                metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                spec_summary["capability"] = {
                    "metrics_path": str(metrics_path),
                    "accuracy": accuracy,
                    "total": metrics["total"],
                    "correct": metrics["correct"],
                }

        summaries[spec_id] = spec_summary

    return summaries


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
    reasoning_flag = bool(model_cfg.get("reasoning", False))

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

    parallel_cfg = experiment_cfg.get("parallel", {})
    max_workers = max(1, int(parallel_cfg.get("max_workers", 1)))
    max_prompts_per_worker = parallel_cfg.get("max_prompts_per_worker")
    if max_prompts_per_worker is not None:
        max_prompts_per_worker = int(max_prompts_per_worker)

    runs_cfg = experiment_cfg.get("runs", {})
    run_count = max(1, int(runs_cfg.get("count", 1)))
    base_seed = runs_cfg.get("base_seed")

    summaries = {}

    for run_index in range(run_count):
        run_seed = (base_seed or 0) + run_index
        run_dir = base_output if run_count == 1 else base_output / f"run_{run_index + 1:03d}"
        ensure_output_dirs(run_dir, create_standard_layout=False)

        run_summary = _execute_run(
            run_dir,
            run_seed,
            spec_entries=specs_cfg["files"],
            spec_texts=spec_texts,
            generation_cfg=generation_cfg,
            model_cfg=model_cfg,
            request_timeout=request_timeout,
            dry_run=dry_run,
            refusal_cfg_entry=refusal_cfg_entry,
            capability_cfg_entry=capability_cfg_entry,
            refusal_cfg=refusal_cfg,
            capability_cfg=capability_cfg,
            refusal_enabled=refusal_enabled,
            capability_enabled=capability_enabled,
            max_workers=max_workers,
            max_prompts_per_worker=max_prompts_per_worker,
            reasoning=reasoning_flag,
        )

        summary_path = run_dir / "experiment_summary.json"
        summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries[summary_path.parent.name if run_count > 1 else "run"] = run_summary

    aggregate_path = base_output / "experiment_runs_summary.json"
    aggregate_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Experiment runs complete. Summary written to %s", aggregate_path)


if __name__ == "__main__":  # pragma: no cover
    main()
