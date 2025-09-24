from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from datasets import load_dataset

from ..model_client import ChatModelClient
from ..utils import ensure_output_dirs

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

LOGGER = logging.getLogger(__name__)

def _format_prompt(question: str, options: Iterable[str]) -> str:
    lines = [question.strip(), "", "Options:"]
    for idx, option in enumerate(options):
        label = chr(ord("A") + idx)
        lines.append(f"({label}) {option}")
    lines.append("")
    lines.append(
        "After reasoning, output a final line formatted exactly as 'Answer: <letter>'."
    )
    return "\n".join(lines)


def _extract_answer(text: str, valid_labels: Iterable[str]) -> Optional[str]:
    if not text:
        return None

    valid = {label.upper() for label in valid_labels}

    final_line_pattern = re.compile(r"(?:final answer|answer)\s*[:\-]\s*([A-J])", re.IGNORECASE)
    matches = final_line_pattern.findall(text)
    for candidate in reversed(matches):
        upper = candidate.upper()
        if upper in valid:
            return upper

    letter_pattern = re.compile(r"\b([A-J])\b")
    tokens = letter_pattern.findall(text.upper())
    for token in reversed(tokens):
        if token in valid:
            return token

    return None


def run_capability_evaluation(
    *,
    spec_id: str,
    spec_text: str,
    dataset_cfg: Dict[str, Any],
    model: ChatModelClient,
    generation_cfg: Dict[str, Any],
    output_base: Path,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run the MMLU-Pro capability evaluation for a single spec."""

    ensure_output_dirs(output_base, create_standard_layout=False)

    temperature = generation_cfg.get("temperature")
    max_tokens = generation_cfg.get("max_tokens")

    split = dataset_cfg.get("split", "validation")
    dataset = load_dataset(
        dataset_cfg["hf_dataset"],
        split=split,
    )

    if dataset_cfg.get("shuffle"):
        dataset = dataset.shuffle(seed=dataset_cfg.get("seed"))

    limit = dataset_cfg.get("limit")
    if limit is not None:
        limit = min(int(limit), len(dataset))
        dataset = dataset.select(range(limit))

    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(desc=f"{spec_id}:MMLU-Pro", total=len(dataset), unit="question")

    output_path = output_base / "mmlu_pro.jsonl"
    metrics_path = output_base / "metrics.json"

    total = len(dataset)
    correct = 0

    with output_path.open("w", encoding="utf-8") as handle:
        try:
            for index, sample in enumerate(dataset):
                options = sample["options"]
                prompt = _format_prompt(sample["question"], options)
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
                    completion = completion_payload.get("text", "")
                    reasoning_segment = completion_payload.get("reasoning_text") or ""
                    answer_source = completion_payload.get("answer_text") or completion
                    if reasoning_segment and answer_source.startswith(reasoning_segment):
                        answer_source = answer_source[len(reasoning_segment) :].strip()
                else:
                    completion = str(completion_payload)
                    answer_source = completion

                answer_source = re.sub(r"<think>.*?</think>", "", answer_source, flags=re.IGNORECASE | re.DOTALL)

                valid_labels = [chr(ord("A") + idx) for idx in range(len(options))]
                predicted = _extract_answer(answer_source, valid_labels)
                correct_label = str(sample.get("answer", "")).strip().upper()
                is_correct = predicted == correct_label
                if is_correct:
                    correct += 1

                record = {
                    "spec_id": spec_id,
                    "question_id": sample.get("question_id"),
                    "category": sample.get("category"),
                    "question": sample["question"],
                    "options": options,
                    "correct_answer": correct_label,
                    "model_answer": predicted,
                    "is_correct": is_correct,
                    "response": completion,
                    "answer_text": answer_source,
                }
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()

    accuracy = correct / total if total else 0.0
    metrics = {
        "spec_id": spec_id,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "split": split,
        "limit": limit,
        "output_path": str(output_path),
    }

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    LOGGER.debug(
        "MMLU-Pro (spec=%s): accuracy %.2f%% (%d/%d)",
        spec_id,
        accuracy * 100,
        correct,
        total,
    )

    return {
        "spec_id": spec_id,
        "metrics_path": str(metrics_path),
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
    }


__all__ = ["run_capability_evaluation"]
