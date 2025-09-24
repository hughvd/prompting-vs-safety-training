from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover
    InferenceClient = None  # type: ignore


class ChatModelClient:
    """Helper around OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        request_timeout: float = 60.0,
        dry_run: bool = False,
    ) -> None:
        self.config = dict(config)
        self.request_timeout = float(request_timeout)
        self.dry_run = bool(dry_run)
        self.provider = str(self.config.get("provider", "openai")).lower()
        self.headers = dict(self.config.get("headers", {}))
        self.extra_body = dict(self.config.get("extra_body", {}))
        self.client_kwargs = dict(self.config.get("client_kwargs", {}))

        api_key = self.config.get("api_key")
        api_key_env = self.config.get("api_key_env")
        if api_key_env:
            api_key = os.getenv(api_key_env) or api_key
        self.api_key = api_key

        self.client = None
        if not self.dry_run:
            self.client = self._create_client()

    def _create_client(self):  # type: ignore[override]
        if self.provider == "openai":
            return self._create_openai_client()
        if self.provider == "huggingface":
            return self._create_hf_client()
        raise ValueError(f"Unsupported provider '{self.provider}'")

    def _create_openai_client(self):
        if OpenAI is None:  # pragma: no cover
            raise RuntimeError(
                "The 'openai' package is required. Run 'uv sync' to install project dependencies."
            )
        if not self.api_key:
            raise RuntimeError(
                f"API key required for model '{self.config.get('name', 'unknown')}'. Set {self.config.get('api_key_env') or 'api_key'} before running."
            )
        base_url = self.config.get("base_url")
        if not base_url:
            raise ValueError("base_url must be provided for OpenAI-compatible endpoints")
        return OpenAI(base_url=base_url, api_key=self.api_key)

    def _create_hf_client(self):
        if InferenceClient is None:  # pragma: no cover
            raise RuntimeError(
                "The 'huggingface_hub' package is required. Run 'uv sync' to install project dependencies."
            )
        if not self.api_key:
            raise RuntimeError(
                f"Hugging Face token required for model '{self.config.get('name', 'unknown')}'. Set {self.config.get('api_key_env') or 'api_key'} before running."
            )
        client_kwargs: Dict[str, Any] = {
            "token": self.api_key,
            "timeout": self.request_timeout,
            "headers": self.headers or None,
            **self.client_kwargs,
        }

        base_url = self.config.get("base_url")
        model_name = self.config.get("name")

        if base_url:
            client_kwargs["base_url"] = base_url
        elif model_name:
            client_kwargs["model"] = model_name
        else:
            raise ValueError("Hugging Face client requires either 'name' or 'base_url' in the config")

        return InferenceClient(**client_kwargs)

    def _object_to_dict(self, obj: Any) -> Optional[Dict[str, Any]]:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        for attr in ("to_dict", "model_dump", "dict"):
            extractor = getattr(obj, attr, None)
            if callable(extractor):
                try:
                    value = extractor()
                except TypeError:  # pragma: no cover - signature mismatch
                    continue
                if isinstance(value, dict):
                    return value
        return None

    @staticmethod
    def _normalize_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    part = item.get("text", "")
                else:
                    part = str(item)
                parts.append(part)
            return "".join(parts).strip()
        if isinstance(value, dict):
            for key in ("text", "content", "value"):
                if key in value and value[key]:
                    return str(value[key]).strip()
            return ""
        return str(value).strip()

    def complete(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        return_metadata: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        if self.dry_run:
            head = messages[-1]["content"] if messages else ""
            return f"[dry-run] Would query model '{model or self.config.get('name')}' for '{head[:40]}...'"

        if self.client is None:
            raise RuntimeError("Model client is not initialized")

        call_kwargs: Dict[str, Any] = {
            "messages": messages,
        }
        model_name = model or self.config.get("name")
        if model_name:
            call_kwargs["model"] = model_name
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens

        merged_extra_body = dict(self.extra_body)
        if extra_body:
            merged_extra_body.update(extra_body)
        if merged_extra_body:
            call_kwargs["extra_body"] = merged_extra_body

        if self.provider == "openai":
            merged_headers = dict(self.headers)
            if extra_headers:
                merged_headers.update(extra_headers)
            if merged_headers:
                call_kwargs["extra_headers"] = merged_headers
            call_kwargs["timeout"] = self.request_timeout

        response = self.client.chat.completions.create(**call_kwargs)

        try:
            choice = response.choices[0]
            message = choice.message
        except (AttributeError, IndexError, KeyError) as exc:  # pragma: no cover
            raise ValueError(
                f"Unexpected response format from provider '{self.provider}': {response}"
            ) from exc

        message_dict = self._object_to_dict(message) or {}
        choice_dict = self._object_to_dict(choice) or {}
        response_dict = self._object_to_dict(response) or {}

        reasoning_text = self._normalize_text(message_dict.get("reasoning"))
        content_text = self._normalize_text(getattr(message, "content", None))
        if not content_text:
            content_text = self._normalize_text(message_dict.get("content"))
        output_text = self._normalize_text(message_dict.get("output_text"))
        if not output_text:
            output_text = self._normalize_text(choice_dict.get("message", {}).get("output_text"))

        answer_text = output_text or content_text

        segments = []
        if reasoning_text:
            segments.append(reasoning_text)
        if answer_text and (not segments or answer_text not in segments[-1]):
            segments.append(answer_text)

        combined_text = "\n\n".join(segment for segment in segments if segment)
        if not combined_text:
            combined_text = answer_text or content_text or ""

        result_payload = {
            "text": combined_text,
            "reasoning_text": reasoning_text or "",
            "answer_text": answer_text or combined_text,
            "message": message_dict,
            "choice": choice_dict,
            "response": response_dict,
        }

        if return_metadata:
            return result_payload

        return result_payload["text"]


__all__ = ["ChatModelClient"]
