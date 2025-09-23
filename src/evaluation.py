from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover
    InferenceClient = None  # type: ignore

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

from .utils import append_jsonl, ensure_output_dirs, load_spec_texts, read_jsonl


class SpecEvaluator:
    """Run spec adherence experiments using configurable model endpoints."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._configure_logging()
        self.logger = logging.getLogger("spec_evaluator")

        self.output_dirs = ensure_output_dirs(config["output_dir"])
        self.dataset = list(read_jsonl(config["dataset"]["path"]))
        self.spec_entries = config["specs"]["files"]
        self.spec_texts = load_spec_texts(config["specs"]["directory"], self.spec_entries)

        self.run_config = config.get("run", {})
        self.http_config = config.get("http", {})
        self.request_timeout = float(self.http_config.get("timeout", 60.0))
        self.generation = config.get("generation", {})
        self.judging = config.get("judging", {})

        self.task_model = self._prepare_model_config(config["task_model"], "task")
        self.judge_model = self._prepare_model_config(config["judge_model"], "judge")
        self.show_progress = bool(self.run_config.get("show_progress", True))
        if self.show_progress and tqdm is None:
            self.logger.warning("tqdm is not installed; progress bar disabled.")
            self.show_progress = False

        self.logger.info(
            "Loaded %d prompts across %d specs",
            len(self.dataset),
            len(self.spec_entries),
        )

    def _configure_logging(self) -> None:
        level_name = self.config.get("logging", {}).get("level", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
        for noisy_logger in ("openai", "openai._logs", "httpx"):
            logger = logging.getLogger(noisy_logger)
            logger.setLevel(logging.WARNING)
            logger.propagate = False

    def run(self) -> Path:
        results_path = self.output_dirs["results"] / self.run_config.get("output_filename", "results.jsonl")
        if results_path.exists():
            results_path.unlink()

        dry_run = bool(self.run_config.get("dry_run", False))
        self.logger.info("Starting evaluation loop (dry_run=%s)", dry_run)

        total_pairs = len(self.dataset) * len(self.spec_entries)
        progress = None
        if self.show_progress and total_pairs and tqdm is not None:
            progress = tqdm(total=total_pairs, desc="Evaluating", unit="pair")

        try:
            for spec_entry in self.spec_entries:
                spec_id = spec_entry["id"]
                spec_text = self.spec_texts[spec_id]
                for prompt in self.dataset:
                    record = self._evaluate_prompt(spec_id, spec_text, prompt, dry_run)
                    append_jsonl([record], results_path)
                    if progress is not None:
                        progress.update(1)
                        progress.set_postfix(
                            {
                                "spec": spec_id,
                                "prompt": prompt.get("id", "-"),
                            },
                            refresh=False,
                        )
        finally:
            if progress is not None:
                progress.close()

        self.logger.info("Results written to %s", results_path)
        return results_path

    def _evaluate_prompt(
        self,
        spec_id: str,
        spec_text: str,
        prompt: Dict[str, Any],
        dry_run: bool,
    ) -> Dict[str, Any]:
        response_text = self._generate_response(spec_text, prompt["prompt"], dry_run)
        judgment = self._judge_response(spec_text, prompt["prompt"], response_text, dry_run)
        return {
            "spec_id": spec_id,
            "prompt_id": prompt.get("id"),
            "domain": prompt.get("domain"),
            "safety": prompt.get("safety"),
            "response": response_text,
            "judgment": judgment,
        }

    def _generate_response(self, spec_text: str, user_prompt: str, dry_run: bool) -> str:
        if dry_run:
            return (
                f"[dry-run] Would query model with spec snippet '{spec_text[:30]}...' "
                f"for prompt '{user_prompt[:40]}...'"
            )

        payload = {
            "model": self.task_model["name"],
            "messages": [
                {"role": "system", "content": spec_text},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.generation.get("temperature", 0.0),
            "max_tokens": self.generation.get("max_tokens", 512),
        }
        return self._post_chat_completion(self.task_model, payload)

    def _judge_response(
        self,
        spec_text: str,
        user_prompt: str,
        assistant_response: str,
        dry_run: bool,
    ) -> Dict[str, Any]:
        if dry_run:
            return {
                "adherence": True,
                "over_refusal": False,
                "under_refusal": False,
                "rationale": "Dry-run mode: no judgment performed.",
            }

        user_template = self.judging.get("user_template")
        if not user_template:
            raise ValueError("judging.user_template must be provided in the config")

        user_message = user_template.format(
            spec=spec_text,
            user_prompt=user_prompt,
            assistant_response=assistant_response,
        )
        payload = {
            "model": self.judge_model["name"],
            "messages": [
                {"role": "system", "content": self.judging.get("system_prompt", "")},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": self.judging.get("max_tokens", 256),
        }
        format_instructions = self.judging.get("format_instructions")
        if format_instructions:
            payload["messages"].append({"role": "user", "content": format_instructions})

        raw_output = self._post_chat_completion(self.judge_model, payload)
        return self._parse_judgment(raw_output)

    def _post_chat_completion(self, model_cfg: Dict[str, Any], payload: Dict[str, Any]) -> str:
        client = model_cfg.get("client")
        if client is None:
            raise RuntimeError(
                "Model client is not initialized. Ensure API keys are configured and dry_run is disabled only when ready."
            )

        provider = model_cfg.get("provider", "openai")

        call_kwargs = {
            "messages": payload["messages"],
        }
        if payload.get("model") is not None:
            call_kwargs["model"] = payload["model"]
        if payload.get("temperature") is not None:
            call_kwargs["temperature"] = payload.get("temperature")
        if payload.get("max_tokens") is not None:
            call_kwargs["max_tokens"] = payload.get("max_tokens")

        extra_body = model_cfg.get("extra_body", {})
        if extra_body:
            call_kwargs["extra_body"] = extra_body

        if provider == "openai":
            headers = model_cfg.get("headers")
            if headers:
                call_kwargs["extra_headers"] = headers
            call_kwargs["timeout"] = self.request_timeout
        elif provider == "huggingface":
            # timeouts are controlled when instantiating InferenceClient
            pass
        else:
            raise ValueError(f"Unsupported provider '{provider}'")

        response = client.chat.completions.create(**call_kwargs)

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError) as exc:
            raise ValueError(
                f"Unexpected response format from provider '{provider}': {response}"
            ) from exc

        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        return content or ""

    def _prepare_model_config(self, model_cfg: Dict[str, Any], label: str) -> Dict[str, Any]:
        result = dict(model_cfg)
        headers = dict(model_cfg.get("headers", {}))
        extra_body = dict(model_cfg.get("extra_body", {}))
        provider = str(model_cfg.get("provider", "openai")).lower()

        api_key = model_cfg.get("api_key")
        api_key_env = model_cfg.get("api_key_env")
        if api_key_env:
            api_key = os.getenv(api_key_env) or api_key

        client = None
        dry_run = self.run_config.get("dry_run", False)

        if not dry_run:
            if provider == "openai":
                if not api_key:
                    raise RuntimeError(
                        f"API key required for {label} model '{model_cfg.get('name', 'unknown')}'. Set {api_key_env or 'api_key'} before running."
                    )
                if OpenAI is None:
                    raise RuntimeError(
                        "The 'openai' package is required. Run 'uv sync' to install project dependencies."
                    )
                client = OpenAI(base_url=model_cfg["base_url"], api_key=api_key)
            elif provider == "huggingface":
                if not api_key:
                    raise RuntimeError(
                        f"Hugging Face token required for {label} model '{model_cfg.get('name', 'unknown')}'. Set {api_key_env or 'api_key'} before running."
                    )
                if InferenceClient is None:
                    raise RuntimeError(
                        "The 'huggingface_hub' package is required. Run 'uv sync' to install project dependencies."
                    )
                client_kwargs = dict(model_cfg.get("client_kwargs", {}))
                client = InferenceClient(
                    model=model_cfg.get("name"),
                    token=api_key,
                    timeout=self.request_timeout,
                    base_url=model_cfg.get("base_url"),
                    headers=headers or None,
                    **client_kwargs,
                )
            else:
                raise ValueError(f"Unsupported provider '{provider}' for {label} model")

        result["provider"] = provider
        result["headers"] = headers or None
        result["extra_body"] = extra_body or {}
        result["client"] = client
        return result

    @staticmethod
    def _parse_judgment(raw_output: str) -> Dict[str, Any]:
        if raw_output is None:
            return {"raw": raw_output}

        text = str(raw_output).strip()

        if text.startswith("```") and "\n" in text:
            lines = text.splitlines()
            # Remove opening fence like ```json or ```
            lines = lines[1:]
            # Drop closing fence if present
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Handle cases where the model adds leading/trailing prose
        trimmed = text
        if trimmed.startswith("{") and trimmed.endswith("}"):
            candidate = trimmed
        else:
            # Attempt to extract the first JSON object substring
            start = trimmed.find("{")
            end = trimmed.rfind("}")
            candidate = trimmed[start : end + 1] if start != -1 and end != -1 and end > start else trimmed

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return {"raw": raw_output}


def load_config(config_path: str | Path) -> Dict[str, Any]:
    from .utils import load_yaml_config

    return load_yaml_config(config_path)


def run_from_config(config_path: str | Path) -> Path:
    evaluator = SpecEvaluator(load_config(config_path))
    return evaluator.run()


__all__ = ["SpecEvaluator", "run_from_config", "load_config"]
