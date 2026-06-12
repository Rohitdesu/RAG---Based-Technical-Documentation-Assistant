from __future__ import annotations

from app.core.gemini import GeminiGateway
from app.core.models import HallucinationCheckResult, RetrievedChunk


class HallucinationChecker:
    """Verify that an answer is grounded in the provided context."""

    def __init__(self, gemini: GeminiGateway):
        self.gemini = gemini

    def evaluate(self, answer: str, chunks: list[RetrievedChunk]) -> HallucinationCheckResult:
        context = "\n\n".join(
            [
                "\n".join(
                    [
                        f"Document: {chunk.document_name}",
                        f"Location: {chunk.location}",
                        f"Source Kind: {chunk.source_kind}",
                        f"Content: {chunk.text}",
                    ]
                )
                for chunk in chunks
            ]
        )
        prompt = f"""
You are checking whether an answer is fully supported by the provided context.

Return strict JSON with this schema:
{{
  "grounded": true,
  "confidence_score": 0.0,
  "explanation": "short explanation"
}}

Rules:
- grounded must be false if the answer contains unsupported claims.
- confidence_score must be a number between 0 and 1.
- Be conservative.

Answer:
\"\"\"
{answer}
\"\"\"

Context:
\"\"\"
{context}
\"\"\"
"""
        result = self.gemini.generate_json(
            prompt,
            fallback={
                "grounded": False,
                "confidence_score": 0.0,
                "explanation": "Could not reliably validate grounding.",
            },
        )
        confidence = result.get("confidence_score", 0.0)
        try:
            confidence_value = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence_value = 0.0
        return HallucinationCheckResult(
            grounded=bool(result.get("grounded", False)),
            confidence_score=confidence_value,
            explanation=str(result.get("explanation", "No explanation provided.")).strip(),
            regeneration_attempted=False,
        )
