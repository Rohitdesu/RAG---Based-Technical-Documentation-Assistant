from __future__ import annotations

from app.core.gemini import GeminiGateway
from app.core.models import RetrievedChunk, SourceReference


class AnswerGenerator:
    def __init__(self, gemini: GeminiGateway):
        self.gemini = gemini

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> tuple[str, list[SourceReference]]:
        ordered_sources = [
            SourceReference(
                document_name=chunk.document_name,
                source_id=chunk.source_id,
                source_type=chunk.source_type,
                location=chunk.location,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                similarity_score=chunk.similarity_score,
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
                        f"[{index}] Chunk Index: {chunk.chunk_index}",
                        f"[{index}] Content: {chunk.text}",
                    ]
                )
            )
        prompt = f"""
You are a technical documentation assistant.
Answer the user question using only the provided context.

Rules:
- If the context is insufficient, say you do not know based on the available documentation.
- Do not invent APIs, steps, or parameters.
- Cite claims inline using bracketed citations like [1] or [2].
- Prefer concise, accurate explanations.

User question:
{question}

Context:
{chr(10).join(context_blocks)}
"""
        answer = self.gemini.generate_text(prompt)
        return answer, ordered_sources
