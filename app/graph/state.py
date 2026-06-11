from __future__ import annotations

from typing import TypedDict

from app.core.models import RetrievedChunk, SourceReference


class GraphState(TypedDict):
    user_query: str
    rewritten_query: str
    query_type: str
    retrieved_chunks: list[RetrievedChunk]
    relevant_chunks: list[RetrievedChunk]
    answer: str
    sources: list[SourceReference]
    retry_count: int
    max_retries: int
