from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from google import genai

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, UpstreamServiceError


logger = logging.getLogger(__name__)


class GeminiGateway:
    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise ConfigurationError(
                "Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running ingestion or querying."
            )
        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def generate_text(self, prompt: str) -> str:
        response = self._call_with_retry(
            action="generate_text",
            func=lambda: self.client.models.generate_content(
                model=self.settings.generation_model,
                contents=prompt,
            ),
        )
        return (response.text or "").strip()

    def embed_document(self, title: str, text: str) -> list[float]:
        content = f"title: {title or 'none'} | text: {text}"
        response = self._call_with_retry(
            action="embed_document",
            func=lambda: self.client.models.embed_content(
                model=self.settings.embedding_model,
                contents=content,
            ),
        )
        return list(response.embeddings[0].values)

    def embed_query(self, query: str) -> list[float]:
        content = f"task: question answering | query: {query}"
        response = self._call_with_retry(
            action="embed_query",
            func=lambda: self.client.models.embed_content(
                model=self.settings.embedding_model,
                contents=content,
            ),
        )
        return list(response.embeddings[0].values)

    def generate_json(self, prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
        text = self.generate_text(prompt)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return fallback
        return fallback

    def _call_with_retry(self, *, action: str, func):
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return func()
            except Exception as exc:  # pragma: no cover - network error handling
                last_error = exc
                message = str(exc)
                if not self._is_retryable(message) or attempt == 2:
                    raise UpstreamServiceError(
                        f"Gemini {action} failed: {message}"
                    ) from exc
                delay = 2 ** attempt
                logger.info(
                    "Retrying Gemini %s after transient error on attempt %s: %s",
                    action,
                    attempt + 1,
                    message,
                )
                time.sleep(delay)
        raise UpstreamServiceError(f"Gemini {action} failed: {last_error}")

    @staticmethod
    def _is_retryable(message: str) -> bool:
        normalized = message.upper()
        return any(
            token in normalized
            for token in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL"]
        )
