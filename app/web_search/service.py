from __future__ import annotations

import hashlib
import logging

import httpx

from app.core.config import Settings
from app.core.exceptions import WebSearchError
from app.core.models import RetrievedChunk


logger = logging.getLogger(__name__)


class TavilyWebSearchService:
    """Call Tavily search and normalize results into RetrievedChunk objects."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def search(self, query: str) -> list[RetrievedChunk]:
        logger.info("Running Tavily fallback search for query: %s", query)
        response = self._request_search(query, use_auth=bool(self.settings.tavily_api_key))
        if response.status_code in {401, 403}:
            logger.info("Tavily key was rejected. Retrying with keyless access.")
            response = self._request_search(query, use_auth=False)
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise WebSearchError(f"Tavily search failed: {exc}") from exc

        payload = response.json()
        results = payload.get("results") or []
        normalized: list[RetrievedChunk] = []
        for index, item in enumerate(results):
            snippet = (item.get("content") or item.get("raw_content") or "").strip()
            url = str(item.get("url") or "")
            title = str(item.get("title") or url or f"Web Result {index + 1}")
            if not snippet or not url:
                continue
            normalized.append(
                RetrievedChunk(
                    chunk_id=f"web-{index}-{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}",
                    source_id=url,
                    document_name=title,
                    source_type="url",
                    source_kind="web_search",
                    location=url,
                    chunk_index=index,
                    text=snippet,
                    similarity_score=round(float(item.get("score") or 0.0), 4),
                    metadata={"published_date": item.get("published_date")},
                )
            )

        if not normalized:
            raise WebSearchError("Tavily did not return any usable web results.")
        return normalized

    def _request_search(self, query: str, *, use_auth: bool) -> httpx.Response:
        headers = {"Content-Type": "application/json"}
        if use_auth and self.settings.tavily_api_key:
            headers["Authorization"] = f"Bearer {self.settings.tavily_api_key}"
        return httpx.post(
            "https://api.tavily.com/search",
            headers=headers,
            json={
                "query": query,
                "search_depth": self.settings.tavily_search_depth,
                "max_results": self.settings.web_search_max_results,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
                "safe_search": True,
            },
            timeout=45.0,
        )
