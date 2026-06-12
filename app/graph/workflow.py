from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.core.config import Settings
from app.core.gemini import GeminiGateway
from app.core.models import ChatMessage, HallucinationCheckResult, QueryResponse, RetrievedChunk
from app.grading.hallucination import HallucinationChecker
from app.generation.service import AnswerGenerator
from app.grading.service import DocumentGrader
from app.graph.state import GraphState
from app.retrieval.vector_store import ChromaVectorStore
from app.web_search.service import TavilyWebSearchService


logger = logging.getLogger(__name__)


class QueryAnalyzer:
    def __init__(self, gemini: GeminiGateway):
        self.gemini = gemini

    def analyze(
        self,
        question: str,
        retry_count: int,
        chat_history: list[ChatMessage],
    ) -> tuple[str, str]:
        history_text = "\n".join(
            f"{message.role.title()}: {message.content}" for message in chat_history[-6:]
        )
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
- If the question is a follow-up, rewrite it into a standalone query using the conversation history.
- Keep the rewritten query focused and under 40 words.
- The query type must be one of: Conceptual, How-to, Troubleshooting, API Reference.

Retry count: {retry_count}
Conversation history:
{history_text or "No previous messages."}
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
        hallucination_checker: HallucinationChecker,
        web_search: TavilyWebSearchService,
    ):
        self.settings = settings
        self.vector_store = vector_store
        self.query_analyzer = query_analyzer
        self.grader = grader
        self.generator = generator
        self.hallucination_checker = hallucination_checker
        self.web_search = web_search
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("query_analysis", self.query_analysis_node)
        builder.add_node("retrieval", self.retrieval_node)
        builder.add_node("document_grading", self.document_grading_node)
        builder.add_node("increment_retry", self.increment_retry_node)
        builder.add_node("web_search", self.web_search_node)
        builder.add_node("generation", self.generation_node)
        builder.add_node("hallucination_check", self.hallucination_check_node)
        builder.add_node("increment_hallucination_retry", self.increment_hallucination_retry_node)
        builder.add_node("unsupported_warning", self.unsupported_warning_node)
        builder.add_node("web_failure_response", self.web_failure_response_node)

        builder.add_edge(START, "query_analysis")
        builder.add_edge("query_analysis", "retrieval")
        builder.add_edge("retrieval", "document_grading")
        builder.add_edge("increment_retry", "query_analysis")
        builder.add_edge("generation", "hallucination_check")
        builder.add_edge("increment_hallucination_retry", "generation")
        builder.add_edge("unsupported_warning", END)
        builder.add_edge("web_failure_response", END)
        builder.add_conditional_edges(
            "document_grading",
            self.route_after_grading,
            {
                "generation": "generation",
                "retry": "increment_retry",
                "web_search": "web_search",
            },
        )
        builder.add_conditional_edges(
            "web_search",
            self.route_after_web_search,
            {
                "generation": "generation",
                "web_failure": "web_failure_response",
            },
        )
        builder.add_conditional_edges(
            "hallucination_check",
            self.route_after_hallucination_check,
            {
                "end": END,
                "regenerate": "increment_hallucination_retry",
                "warning": "unsupported_warning",
            },
        )
        return builder.compile()

    def invoke(
        self,
        *,
        question: str,
        session_id: str,
        chat_history: list[ChatMessage],
    ) -> QueryResponse:
        result = self.graph.invoke(
            {
                "session_id": session_id,
                "chat_history": chat_history,
                "user_query": question,
                "rewritten_query": "",
                "query_type": "",
                "retrieved_chunks": [],
                "relevant_chunks": [],
                "web_chunks": [],
                "answer": "",
                "sources": [],
                "retry_count": 0,
                "max_retries": self.settings.max_retries,
                "used_web_search": False,
                "hallucination_retry_count": 0,
                "max_hallucination_retries": self.settings.max_hallucination_retries,
                "hallucination_check": HallucinationCheckResult(
                    grounded=False,
                    confidence_score=0.0,
                    explanation="Answer has not been checked yet.",
                ),
                "warning": None,
            }
        )
        return QueryResponse(
            session_id=result["session_id"],
            answer=result["answer"],
            sources=result["sources"],
            query_type=result["query_type"],
            rewritten_query=result["rewritten_query"],
            retry_count=result["retry_count"],
            used_web_search=result["used_web_search"],
            hallucination_check=result["hallucination_check"],
            warning=result["warning"],
        )

    def query_analysis_node(self, state: GraphState):
        rewritten_query, query_type = self.query_analyzer.analyze(
            state["user_query"],
            state["retry_count"],
            state["chat_history"],
        )
        logger.info(
            "Query analysis completed for session %s. Type=%s Retry=%s",
            state["session_id"],
            query_type,
            state["retry_count"],
        )
        return {"rewritten_query": rewritten_query, "query_type": query_type}

    def retrieval_node(self, state: GraphState):
        retrieved = self.vector_store.similarity_search(
            state["rewritten_query"] or state["user_query"],
            self.settings.top_k,
        )
        logger.info(
            "Retrieved %s local chunks for session %s",
            len(retrieved),
            state["session_id"],
        )
        return {"retrieved_chunks": retrieved}

    def document_grading_node(self, state: GraphState):
        relevant = self.grader.grade_many(
            state["user_query"],
            state["rewritten_query"],
            state["retrieved_chunks"],
        )
        logger.info(
            "Document grading kept %s/%s chunks for session %s",
            len(relevant),
            len(state["retrieved_chunks"]),
            state["session_id"],
        )
        return {"relevant_chunks": relevant}

    def increment_retry_node(self, state: GraphState):
        logger.info(
            "Retrying local retrieval for session %s. Retry %s -> %s",
            state["session_id"],
            state["retry_count"],
            state["retry_count"] + 1,
        )
        return {"retry_count": state["retry_count"] + 1}

    def web_search_node(self, state: GraphState):
        try:
            web_chunks = self.web_search.search(state["rewritten_query"] or state["user_query"])
            logger.info(
                "Web search returned %s results for session %s",
                len(web_chunks),
                state["session_id"],
            )
            return {"web_chunks": web_chunks, "used_web_search": True}
        except Exception as exc:
            logger.info(
                "Web search failed for session %s: %s",
                state["session_id"],
                exc,
            )
            return {"web_chunks": [], "used_web_search": True}

    def generation_node(self, state: GraphState):
        context_chunks = self._active_context(state)
        if not context_chunks:
            return {
                "answer": "I couldn't find reliable information in either the documentation corpus or web search.",
                "sources": [],
            }
        answer, sources = self.generator.generate(
            state["user_query"],
            context_chunks,
            chat_history=state["chat_history"],
            regeneration_attempt=state["hallucination_retry_count"],
            previous_answer=state["answer"],
        )
        return {"answer": answer, "sources": sources}

    def hallucination_check_node(self, state: GraphState):
        context_chunks = self._active_context(state)
        if not context_chunks or not state["answer"]:
            result = HallucinationCheckResult(
                grounded=False,
                confidence_score=0.0,
                explanation="No context or answer was available for grounding validation.",
                regeneration_attempted=state["hallucination_retry_count"] > 0,
            )
        else:
            result = self.hallucination_checker.evaluate(state["answer"], context_chunks)
            result.regeneration_attempted = state["hallucination_retry_count"] > 0
        logger.info(
            "Hallucination check for session %s grounded=%s confidence=%.2f",
            state["session_id"],
            result.grounded,
            result.confidence_score,
        )
        return {"hallucination_check": result}

    def increment_hallucination_retry_node(self, state: GraphState):
        logger.info(
            "Regenerating answer after failed grounding for session %s",
            state["session_id"],
        )
        return {"hallucination_retry_count": state["hallucination_retry_count"] + 1}

    def unsupported_warning_node(self, state: GraphState):
        warning = "Some parts of this answer may not be fully supported by the available documentation."
        hallucination_check = state["hallucination_check"].model_copy(
            update={"warning": warning, "regeneration_attempted": True}
        )
        answer = state["answer"]
        if warning not in answer:
            answer = f"{answer}\n\nWarning: {warning}"
        return {
            "answer": answer,
            "warning": warning,
            "hallucination_check": hallucination_check,
        }

    def web_failure_response_node(self, state: GraphState):
        return {
            "answer": "I couldn't find reliable information in either the documentation corpus or web search.",
            "sources": [],
            "warning": None,
            "hallucination_check": HallucinationCheckResult(
                grounded=False,
                confidence_score=0.0,
                explanation="No reliable local or web context was available.",
            ),
        }

    @staticmethod
    def route_after_grading(state: GraphState) -> Literal["generation", "retry", "web_search"]:
        if state["relevant_chunks"]:
            logger.info("Routing to generation using local documents.")
            return "generation"
        if state["retry_count"] < state["max_retries"]:
            logger.info("No relevant local documents found. Retrying local retrieval.")
            return "retry"
        logger.info("Local retrieval exhausted. Routing to web search.")
        return "web_search"

    @staticmethod
    def route_after_hallucination_check(state: GraphState) -> Literal["end", "regenerate", "warning"]:
        if state["hallucination_check"].grounded:
            logger.info("Grounded answer accepted.")
            return "end"
        if not state["sources"]:
            logger.info("No sources available after hallucination check. Ending with warning.")
            return "warning"
        if state["hallucination_retry_count"] < state["max_hallucination_retries"]:
            logger.info("Answer not grounded. Regenerating once.")
            return "regenerate"
        logger.info("Answer still not grounded after regeneration. Returning warning.")
        return "warning"

    @staticmethod
    def route_after_web_search(state: GraphState) -> Literal["generation", "web_failure"]:
        if state["web_chunks"]:
            logger.info("Web search produced usable context. Routing to generation.")
            return "generation"
        logger.info("Web search produced no usable context. Returning web failure response.")
        return "web_failure"

    @staticmethod
    def _active_context(state: GraphState) -> list[RetrievedChunk]:
        return state["relevant_chunks"] or state["web_chunks"]
