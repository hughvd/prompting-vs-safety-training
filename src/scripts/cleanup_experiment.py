from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable, List


LOGGER = logging.getLogger("cleanup_experiment")

THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_TAG_PATTERN = re.compile(r"</?think>", re.IGNORECASE)
FINAL_LINE_PATTERN = re.compile(r"(?:final answer|answer)\s*[:\-]\s*([A-J])", re.IGNORECASE)
LETTER_PATTERN = re.compile(r"\b([A-J])\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean <think> tokens out of experiment outputs.")
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to the experiment directory to clean (e.g., data/experiments/deepseekR1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files and report changes without modifying them.",
    )
    return parser.parse_args()


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = THINK_BLOCK_PATTERN.sub("", value)
    text = THINK_TAG_PATTERN.sub("", text)
    return text.strip()


def clean_record(record: dict) -> dict:
    if "response" in record:
        record["response"] = clean_text(record["response"])
    if "answer_text" in record:
        record["answer_text"] = clean_text(record["answer_text"])
    return record


def extract_answer(text: str, options: List[str]) -> str | None:
    if not text:
        return None
    valid_labels = [chr(ord("A") + idx) for idx in range(len(options))]
    valid = {label.upper() for label in valid_labels}

    matches = FINAL_LINE_PATTERN.findall(text)
    for candidate in reversed(matches):
        upper = candidate.upper()
        if upper in valid:
            return upper

    tokens = LETTER_PATTERN.findall(text.upper())
    for token in reversed(tokens):
        if token in valid:
            return token
    return None


def iter_jsonl_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.jsonl"):
        if path.is_file():
            yield path


def clean_jsonl(path: Path, dry_run: bool) -> bool:
    changed = False
    lines = []
    is_capability = "capability" in path.parts
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                lines.append(line)
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                lines.append(line)
                continue

            original_response = record.get("response")
            original_answer = record.get("answer_text")
            original_model_answer = record.get("model_answer")
            original_is_correct = record.get("is_correct")

            record = clean_record(record)

            if is_capability and "options" in record:
                options = record.get("options") or []
                answer_text = record.get("answer_text") or ""
                predicted = extract_answer(answer_text, options)
                correct_label = str(record.get("correct_answer", "")).strip().upper()
                record["model_answer"] = predicted
                record["is_correct"] = bool(predicted and predicted == correct_label)

            if record.get("response") != original_response or record.get("answer_text") != original_answer:
                changed = True
            if is_capability and (
                record.get("model_answer") != original_model_answer
                or record.get("is_correct") != original_is_correct
            ):
                changed = True
            lines.append(json.dumps(record, ensure_ascii=False) + "\n")

    if changed and not dry_run:
        with path.open("w", encoding="utf-8") as handle:
            handle.writelines(lines)

    return changed


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    experiment_dir = args.experiment_dir.resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    jsonl_files = list(iter_jsonl_files(experiment_dir))
    if not jsonl_files:
        LOGGER.info("No .jsonl files found under %s", experiment_dir)
        return

    LOGGER.info("Scanning %d jsonl files under %s", len(jsonl_files), experiment_dir)
    modified = 0

    for path in jsonl_files:
        if clean_jsonl(path, args.dry_run):
            modified += 1
            LOGGER.info("Cleaned %s", path)

    if args.dry_run:
        LOGGER.info("Dry run complete. %d files would be modified.", modified)
    else:
        LOGGER.info("Cleanup complete. %d files updated.", modified)


if __name__ == "__main__":  # pragma: no cover
    main()
