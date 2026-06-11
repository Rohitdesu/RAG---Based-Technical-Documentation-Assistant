from __future__ import annotations

import json
import re
from typing import Any

from google import genai

from app.core.config import Settings
from app.core.exceptions import ConfigurationError


class GeminiGateway:
    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise ConfigurationError(
                "Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running ingestion or querying."
            )
        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def generate_text(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.settings.generation_model,
            contents=prompt,
        )
        return (response.text or "").strip()

    def embed_document(self, title: str, text: str) -> list[float]:
        content = f"title: {title or 'none'} | text: {text}"
        response = self.client.models.embed_content(
            model=self.settings.embedding_model,
            contents=content,
        )
        return list(response.embeddings[0].values)

    def embed_query(self, query: str) -> list[float]:
        content = f"task: question answering | query: {query}"
        response = self.client.models.embed_content(
            model=self.settings.embedding_model,
            contents=content,
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
