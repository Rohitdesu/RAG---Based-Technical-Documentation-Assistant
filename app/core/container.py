from __future__ import annotations

import json
from functools import lru_cache

from app.core.config import get_settings
from app.core.gemini import GeminiGateway
from app.core.registry import DocumentRegistry
from app.grading.hallucination import HallucinationChecker
from app.generation.service import AnswerGenerator
from app.grading.service import DocumentGrader
from app.graph.workflow import QueryAnalyzer, QueryWorkflow
from app.ingestion.chunker import TextChunker
from app.ingestion.service import IngestionService
from app.memory.session_store import SessionStore
from app.retrieval.vector_store import ChromaVectorStore
from app.web_search.service import TavilyWebSearchService


class AppContainer:
    def __init__(self):
        self.settings = get_settings()
        self.registry = DocumentRegistry(self.settings.registry_path)
        self.sessions = SessionStore(self.settings.sessions_dir)
        self.chunker = TextChunker(self.settings.chunk_size, self.settings.chunk_overlap)
        self._gemini: GeminiGateway | None = None
        self._vector_store: ChromaVectorStore | None = None
        self._ingestion: IngestionService | None = None
        self._query_workflow: QueryWorkflow | None = None
        self._web_search: TavilyWebSearchService | None = None

    @property
    def gemini(self) -> GeminiGateway:
        if self._gemini is None:
            self._gemini = GeminiGateway(self.settings)
        return self._gemini

    @property
    def vector_store(self) -> ChromaVectorStore:
        if self._vector_store is None:
            self._vector_store = ChromaVectorStore(self.settings, self.gemini)
        return self._vector_store

    @property
    def ingestion(self) -> IngestionService:
        if self._ingestion is None:
            self._ingestion = IngestionService(
                chunker=self.chunker,
                vector_store=self.vector_store,
                registry=self.registry,
            )
        return self._ingestion

    @property
    def query_workflow(self) -> QueryWorkflow:
        if self._query_workflow is None:
            self._query_workflow = QueryWorkflow(
                settings=self.settings,
                vector_store=self.vector_store,
                query_analyzer=QueryAnalyzer(self.gemini),
                grader=DocumentGrader(self.gemini),
                generator=AnswerGenerator(self.gemini),
                hallucination_checker=HallucinationChecker(self.gemini),
                web_search=self.web_search,
            )
        return self._query_workflow

    @property
    def web_search(self) -> TavilyWebSearchService:
        if self._web_search is None:
            self._web_search = TavilyWebSearchService(self.settings)
        return self._web_search

    def append_feedback(self, payload: dict) -> None:
        self.settings.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self.settings.feedback_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload) + "\n")


@lru_cache(maxsize=1)
def get_container() -> AppContainer:
    return AppContainer()
