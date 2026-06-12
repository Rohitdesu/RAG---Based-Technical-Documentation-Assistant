from __future__ import annotations

from app.core.gemini import GeminiGateway
from app.core.models import RetrievedChunk


class DocumentGrader:
    def __init__(self, gemini: GeminiGateway):
        self.gemini = gemini

    def grade_many(
        self,
        question: str,
        rewritten_query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        chunk_descriptions = []
        for index, chunk in enumerate(chunks, start=1):
            chunk_descriptions.append(
                "\n".join(
                    [
                        f"Chunk ID: {chunk.chunk_id}",
                        f"Document: {chunk.document_name}",
                        f"Location: {chunk.location}",
                        f"Content: {chunk.text}",
                    ]
                )
            )
        prompt = f"""
You are grading whether retrieved documentation chunks are useful for answering a user question.

Return strict JSON with this schema:
{{
  "relevant_chunk_ids": ["chunk-id-1"],
  "reason": "short explanation"
}}

Rules:
- Only include chunk IDs that are truly useful for answering the question.
- If none are useful, return an empty array.
- Be conservative and prefer precision over recall.

Original question:
{question}

Rewritten retrieval query:
{rewritten_query}

Chunks:
\"\"\"
{chr(10).join(chunk_descriptions)}
\"\"\"
"""
        result = self.gemini.generate_json(
            prompt,
            fallback={"relevant_chunk_ids": [], "reason": "Could not parse grading response."},
        )
        relevant_ids = {
            str(chunk_id).strip() for chunk_id in result.get("relevant_chunk_ids", []) if chunk_id
        }
        return [chunk for chunk in chunks if chunk.chunk_id in relevant_ids]
