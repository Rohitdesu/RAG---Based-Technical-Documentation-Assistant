from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question to ask.")


class SourceReference(BaseModel):
    document_name: str
    source_id: str
    source_type: Literal["file", "url"]
    location: str
    chunk_id: str
    chunk_index: int
    similarity_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    query_type: str
    rewritten_query: str
    retry_count: int


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)
    question: str | None = Field(default=None, max_length=1000)
    answer: str | None = Field(default=None, max_length=4000)


class FeedbackResponse(BaseModel):
    message: str


class DocumentRecord(BaseModel):
    source_id: str
    document_name: str
    source_type: Literal["file", "url"]
    location: str
    chunks_count: int
    chunk_ids: list[str]
    ingested_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]


class IngestResponse(BaseModel):
    message: str
    documents: list[DocumentRecord]


class UrlIngestItem(BaseModel):
    title: str | None = None
    url: HttpUrl


class LoadedDocument(BaseModel):
    source_id: str
    document_name: str
    source_type: Literal["file", "url"]
    location: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    chunk_id: str
    source_id: str
    document_name: str
    source_type: Literal["file", "url"]
    location: str
    chunk_index: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_id: str
    document_name: str
    source_type: Literal["file", "url"]
    location: str
    chunk_index: int
    text: str
    similarity_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
