from __future__ import annotations

from app.core.gemini import GeminiGateway
from app.core.models import RetrievedChunk


class DocumentGrader:
    def __init__(self, gemini: GeminiGateway):
        self.gemini = gemini

    def grade(self, question: str, rewritten_query: str, chunk: RetrievedChunk) -> bool:
        prompt = f"""
You are grading whether a retrieved documentation chunk is useful for answering a user question.

Return strict JSON with this schema:
{{
  "decision": "relevant" | "irrelevant",
  "reason": "short explanation"
}}

Original question:
{question}

Rewritten retrieval query:
{rewritten_query}

Document name:
{chunk.document_name}

Chunk:
\"\"\"
{chunk.text}
\"\"\"
"""
        result = self.gemini.generate_json(
            prompt,
            fallback={"decision": "irrelevant", "reason": "Could not parse grading response."},
        )
        return str(result.get("decision", "")).strip().lower() == "relevant"
