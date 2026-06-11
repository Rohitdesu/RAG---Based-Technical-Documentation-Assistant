from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.core.config import Settings
from app.core.gemini import GeminiGateway
from app.core.models import QueryResponse
from app.generation.service import AnswerGenerator
from app.grading.service import DocumentGrader
from app.graph.state import GraphState
from app.retrieval.vector_store import ChromaVectorStore


class QueryAnalyzer:
    def __init__(self, gemini: GeminiGateway):
        self.gemini = gemini

    def analyze(self, question: str, retry_count: int) -> tuple[str, str]:
        prompt = f"""
You are preparing a search query for a retrieval-augmented technical documentation assistant.

Return strict JSON with this schema:
{{
  "rewritten_query": "improved retrieval query",
  "query_type": "Conceptual | How-to | Troubleshooting | API Reference"
}}

Instructions:
- Rewrite the query for better semantic retrieval.
- Expand with useful technical synonyms if needed.
- Resolve ambiguity when possible.
- Keep the rewritten query focused and under 40 words.
- The query type must be one of: Conceptual, How-to, Troubleshooting, API Reference.

Retry count: {retry_count}
User question: {question}
"""
        fallback = {
            "rewritten_query": question,
            "query_type": self._heuristic_type(question),
        }
        result = self.gemini.generate_json(prompt, fallback=fallback)
        rewritten_query = str(result.get("rewritten_query") or question).strip()
        query_type = str(result.get("query_type") or fallback["query_type"]).strip()
        if query_type not in {"Conceptual", "How-to", "Troubleshooting", "API Reference"}:
            query_type = fallback["query_type"]
        return rewritten_query, query_type

    @staticmethod
    def _heuristic_type(question: str) -> str:
        lowered = question.lower()
        if any(token in lowered for token in ["error", "issue", "fail", "exception", "bug"]):
            return "Troubleshooting"
        if any(token in lowered for token in ["how ", "steps", "create", "build", "use"]):
            return "How-to"
        if any(token in lowered for token in ["parameter", "argument", "class", "method", "api"]):
            return "API Reference"
        return "Conceptual"


class QueryWorkflow:
    def __init__(
        self,
        *,
        settings: Settings,
        vector_store: ChromaVectorStore,
        query_analyzer: QueryAnalyzer,
        grader: DocumentGrader,
        generator: AnswerGenerator,
    ):
        self.settings = settings
        self.vector_store = vector_store
        self.query_analyzer = query_analyzer
        self.grader = grader
        self.generator = generator
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("query_analysis", self.query_analysis_node)
        builder.add_node("retrieval", self.retrieval_node)
        builder.add_node("document_grading", self.document_grading_node)
        builder.add_node("increment_retry", self.increment_retry_node)
        builder.add_node("generation", self.generation_node)
        builder.add_node("fallback_response", self.fallback_response_node)

        builder.add_edge(START, "query_analysis")
        builder.add_edge("query_analysis", "retrieval")
        builder.add_edge("retrieval", "document_grading")
        builder.add_edge("increment_retry", "query_analysis")
        builder.add_edge("generation", END)
        builder.add_edge("fallback_response", END)
        builder.add_conditional_edges(
            "document_grading",
            self.route_after_grading,
            {
                "generation": "generation",
                "retry": "increment_retry",
                "fallback": "fallback_response",
            },
        )
        return builder.compile()

    def invoke(self, question: str) -> QueryResponse:
        result = self.graph.invoke(
            {
                "user_query": question,
                "rewritten_query": "",
                "query_type": "",
                "retrieved_chunks": [],
                "relevant_chunks": [],
                "answer": "",
                "sources": [],
                "retry_count": 0,
                "max_retries": self.settings.max_retries,
            }
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            query_type=result["query_type"],
            rewritten_query=result["rewritten_query"],
            retry_count=result["retry_count"],
        )

    def query_analysis_node(self, state: GraphState):
        rewritten_query, query_type = self.query_analyzer.analyze(
            state["user_query"],
            state["retry_count"],
        )
        return {"rewritten_query": rewritten_query, "query_type": query_type}

    def retrieval_node(self, state: GraphState):
        retrieved = self.vector_store.similarity_search(
            state["rewritten_query"] or state["user_query"],
            self.settings.top_k,
        )
        return {"retrieved_chunks": retrieved}

    def document_grading_node(self, state: GraphState):
        relevant = [
            chunk
            for chunk in state["retrieved_chunks"]
            if self.grader.grade(state["user_query"], state["rewritten_query"], chunk)
        ]
        return {"relevant_chunks": relevant}

    def increment_retry_node(self, state: GraphState):
        return {"retry_count": state["retry_count"] + 1}

    def generation_node(self, state: GraphState):
        answer, sources = self.generator.generate(state["user_query"], state["relevant_chunks"])
        return {"answer": answer, "sources": sources}

    def fallback_response_node(self, state: GraphState):
        return {
            "answer": "I don't know based on the available documentation.",
            "sources": [],
        }

    @staticmethod
    def route_after_grading(state: GraphState) -> Literal["generation", "retry", "fallback"]:
        if state["relevant_chunks"]:
            return "generation"
        if state["retry_count"] < state["max_retries"]:
            return "retry"
        return "fallback"
