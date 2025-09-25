from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from ..model_client import ChatModelClient

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

## SHITTY FIX
is_reasoning = True

def _format_prompt(question: str, options: Iterable[str]) -> str:
    lines = [question.strip(), "", "Options:"]
    for idx, option in enumerate(options):
        label = chr(ord("A") + idx)
        lines.append(f"({label}) {option}")
    lines.append("")
    lines.append("After reasoning, output a final line formatted exactly as 'Answer: <letter>'.")
    return "\n".join(lines)


def _extract_answer(text: str, valid_labels: Iterable[str]) -> str | None:
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


def _clean_answer_text(
    answer_text: str,
    reasoning_segment: str | None = None,
    *,
    expects_reasoning: bool = False,
) -> str:
    # Shitty fix for non reasoning models
    if not is_reasoning:
        return answer_text.strip()
    text = answer_text or ""
    if reasoning_segment and text.startswith(reasoning_segment):
        text = text[len(reasoning_segment) :]
    elif expects_reasoning:
        closing = re.search(r"</think>", text, re.IGNORECASE)
        if closing:
            text = text[closing.end() :]
        elif re.search(r"<think", text, re.IGNORECASE):
            return ""
        else:
            return text.strip()
    else:
        return text.strip()
    text = THINK_PATTERN.sub("", text)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def evaluate_capability_chunk(
    spec_id: str,
    spec_text: str,
    samples: Iterable[Dict[str, Any]],
    model_cfg: Dict[str, Any],
    generation_cfg: Dict[str, Any],
    *,
    request_timeout: float,
    dry_run: bool,
    reasoning: bool,
) -> List[Dict[str, Any]]:
    client = ChatModelClient(model_cfg, request_timeout=request_timeout, dry_run=dry_run)

    temperature = generation_cfg.get("temperature")
    max_tokens = generation_cfg.get("max_tokens")

    results: List[Dict[str, Any]] = []
    for sample in samples:
        options = sample.get("options", [])
        prompt = _format_prompt(sample.get("question", ""), options)
        messages = [
            {"role": "system", "content": spec_text},
            {"role": "user", "content": prompt},
        ]

        completion_payload = client.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            return_metadata=True,
        )

        if isinstance(completion_payload, dict):
            full_response = completion_payload.get("text", "")
            reasoning_segment = completion_payload.get("reasoning_text") or ""
            answer_text = completion_payload.get("answer_text") or full_response
        else:
            full_response = str(completion_payload)
            reasoning_segment = ""
            answer_text = full_response

        cleaned_answer = _clean_answer_text(
            answer_text,
            reasoning_segment,
            expects_reasoning=reasoning,
        )

        valid_labels = [chr(ord("A") + idx) for idx in range(len(options))]
        predicted = _extract_answer(cleaned_answer, valid_labels)
        correct_label = str(sample.get("answer", "")).strip().upper()

        results.append(
            {
                "spec_id": spec_id,
                "index": sample.get("index", 0),
                "question_id": sample.get("question_id"),
                "category": sample.get("category"),
                "question": sample.get("question"),
                "options": options,
                "correct_answer": correct_label,
                "model_answer": predicted,
                "is_correct": predicted == correct_label,
                "response": full_response,
                "answer_text": cleaned_answer,
            }
        )

    return results


__all__ = ["evaluate_capability_chunk"]
