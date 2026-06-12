from __future__ import annotations

from typing import TypedDict

from app.core.models import ChatMessage, HallucinationCheckResult, RetrievedChunk, SourceReference


class GraphState(TypedDict):
    session_id: str
    chat_history: list[ChatMessage]
    user_query: str
    rewritten_query: str
    query_type: str
    retrieved_chunks: list[RetrievedChunk]
    relevant_chunks: list[RetrievedChunk]
    web_chunks: list[RetrievedChunk]
    answer: str
    sources: list[SourceReference]
    retry_count: int
    max_retries: int
    used_web_search: bool
    hallucination_retry_count: int
    max_hallucination_retries: int
    hallucination_check: HallucinationCheckResult
    warning: str | None
