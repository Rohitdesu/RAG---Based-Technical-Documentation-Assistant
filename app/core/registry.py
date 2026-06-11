from __future__ import annotations

import json

from app.core.models import DocumentRecord


class DocumentRegistry:
    def __init__(self, path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def list_documents(self) -> list[DocumentRecord]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return [DocumentRecord.model_validate(item) for item in data]

    def get_document(self, source_id: str) -> DocumentRecord | None:
        for document in self.list_documents():
            if document.source_id == source_id:
                return document
        return None

    def upsert_documents(self, documents: list[DocumentRecord]) -> None:
        existing = {document.source_id: document for document in self.list_documents()}
        for document in documents:
            existing[document.source_id] = document
        serialized = [
            {
                **record.model_dump(mode="json"),
                "ingested_at": record.ingested_at.isoformat(),
            }
            for record in sorted(existing.values(), key=lambda item: item.document_name.lower())
        ]
        self.path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    def remove_document(self, source_id: str) -> None:
        remaining = [doc for doc in self.list_documents() if doc.source_id != source_id]
        serialized = [
            {
                **record.model_dump(mode="json"),
                "ingested_at": record.ingested_at.isoformat(),
            }
            for record in remaining
        ]
        self.path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    @staticmethod
    def build_record(
        *,
        source_id: str,
        document_name: str,
        source_type: str,
        location: str,
        chunk_ids: list[str],
    ) -> DocumentRecord:
        return DocumentRecord(
            source_id=source_id,
            document_name=document_name,
            source_type=source_type,
            location=location,
            chunks_count=len(chunk_ids),
            chunk_ids=chunk_ids,
            ingested_at=datetime.utcnow(),
        )
