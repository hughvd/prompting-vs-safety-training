from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict
import re

from datasets import load_dataset

from ..model_client import ChatModelClient
from ..utils import ensure_output_dirs

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

LOGGER = logging.getLogger(__name__)

def run_refusal_evaluation(
    *,
    spec_id: str,
    spec_text: str,
    datasets_cfg: Dict[str, Any],
    model: ChatModelClient,
    generation_cfg: Dict[str, Any],
    output_base: Path,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run refusal benchmarks for a single spec and return summary statistics."""

    ensure_output_dirs(output_base, create_standard_layout=False)

    temperature = generation_cfg.get("temperature")
    max_tokens = generation_cfg.get("max_tokens")

    summaries = []

    for dataset_id, dataset_cfg in datasets_cfg.get("datasets", {}).items():
        output_path = output_base / f"{dataset_id}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        split = dataset_cfg.get("split", "train")
        dataset = load_dataset(
            dataset_cfg["hf_dataset"],
            dataset_cfg["hf_config"],
            split=split,
        )

        if dataset_cfg.get("shuffle"):
            dataset = dataset.shuffle(seed=dataset_cfg.get("seed"))

        limit = dataset_cfg.get("limit")
        if limit is not None:
            limit = min(int(limit), len(dataset))
            dataset = dataset.select(range(limit))

        num_samples = len(dataset)

        progress = None
        if show_progress and tqdm is not None:
            progress = tqdm(desc=f"{spec_id}:{dataset_id}", total=num_samples, unit="prompt")

        try:
            with output_path.open("w", encoding="utf-8") as handle:
                for index, sample in enumerate(dataset):
                    prompt = sample["prompt"]
                    category = sample.get("category")
                    messages = [
                        {"role": "system", "content": spec_text},
                        {"role": "user", "content": prompt},
                    ]
                    completion_payload = model.complete(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        return_metadata=True,
                    )

                    if isinstance(completion_payload, dict):
                        full_response = completion_payload.get("text", "")
                        reasoning_segment = completion_payload.get("reasoning_text") or ""
                        answer_text = completion_payload.get("answer_text") or full_response
                        if reasoning_segment and answer_text.startswith(reasoning_segment):
                            answer_text = answer_text[len(reasoning_segment) :].strip()
                    else:
                        full_response = str(completion_payload)
                        answer_text = full_response

                    answer_text = re.sub(
                        r"<think>.*?</think>", "", answer_text, flags=re.IGNORECASE | re.DOTALL
                    ).strip()

                    record = {
                        "spec_id": spec_id,
                        "dataset": dataset_id,
                        "index": index,
                        "category": category,
                        "prompt": prompt,
                        "response": full_response,
                        "answer_text": answer_text,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False))
                    handle.write("\n")
                    if progress is not None:
                        progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        summaries.append(
            {
                "dataset": dataset_id,
                "samples": num_samples,
                "output_path": str(output_path),
            }
        )
        LOGGER.debug(
            "Refusal dataset %s (spec=%s): wrote %d samples to %s",
            dataset_id,
            spec_id,
            num_samples,
            output_path,
        )

    summary_path = output_base / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)

    return {
        "spec_id": spec_id,
        "datasets": summaries,
        "summary_path": str(summary_path),
    }


__all__ = ["run_refusal_evaluation"]
