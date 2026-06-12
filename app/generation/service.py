from __future__ import annotations

from collections.abc import Sequence

from app.core.gemini import GeminiGateway
from app.core.models import ChatMessage, RetrievedChunk, SourceReference


class AnswerGenerator:
    def __init__(self, gemini: GeminiGateway):
        self.gemini = gemini

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        *,
        chat_history: Sequence[ChatMessage],
        regeneration_attempt: int = 0,
        previous_answer: str = "",
    ) -> tuple[str, list[SourceReference]]:
        history_lines = [
            f"{message.role.title()}: {message.content}"
            for message in list(chat_history)[-6:]
        ]
        ordered_sources = [
            SourceReference(
                document_name=chunk.document_name,
                source_id=chunk.source_id,
                source_type=chunk.source_type,
                source_kind=chunk.source_kind,
                location=chunk.location,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                similarity_score=chunk.similarity_score,
                snippet=chunk.text[:240],
            )
            for chunk in chunks
        ]
        context_blocks = []
        for index, chunk in enumerate(chunks, start=1):
            context_blocks.append(
                "\n".join(
                    [
                        f"[{index}] Document: {chunk.document_name}",
                        f"[{index}] Location: {chunk.location}",
                        f"[{index}] Source Kind: {chunk.source_kind}",
                        f"[{index}] Chunk Index: {chunk.chunk_index}",
                        f"[{index}] Content: {chunk.text}",
                    ]
                )
            )
        regeneration_instruction = ""
        if regeneration_attempt > 0:
            regeneration_instruction = f"""
Previous draft that failed grounding review:
{previous_answer}

Regenerate a more conservative answer that removes any unsupported claims.
"""
        prompt = f"""
You are a technical documentation assistant.
Answer the user question using only the provided context.

Rules:
- If the context is insufficient, say you do not know based on the available documentation.
- Do not invent APIs, steps, or parameters.
- Cite claims inline using bracketed citations like [1] or [2].
- Prefer concise, accurate explanations.
- Treat the conversation history only as user intent context. Ground factual claims only in the supplied retrieval context.

User question:
{question}

Conversation history:
{chr(10).join(history_lines) if history_lines else "No previous messages."}

Context:
{chr(10).join(context_blocks)}

{regeneration_instruction}
"""
        answer = self.gemini.generate_text(prompt)
        return answer, ordered_sources
