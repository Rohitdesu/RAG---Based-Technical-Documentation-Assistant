from __future__ import annotations

import hashlib
from pathlib import Path

from app.core.exceptions import IngestionError
from app.core.models import ChunkRecord, DocumentRecord, LoadedDocument
from app.core.registry import DocumentRegistry
from app.ingestion.chunker import TextChunker
from app.ingestion.loaders import load_file_document, load_url_document
from app.retrieval.vector_store import ChromaVectorStore


class IngestionService:
    def __init__(
        self,
        *,
        chunker: TextChunker,
        vector_store: ChromaVectorStore,
        registry: DocumentRegistry,
    ):
        self.chunker = chunker
        self.vector_store = vector_store
        self.registry = registry

    def ingest_files(self, files: list[tuple[str, bytes]]) -> list[DocumentRecord]:
        loaded_documents = [load_file_document(Path(name), content) for name, content in files]
        return self._ingest_loaded_documents(loaded_documents)

    def ingest_paths(self, paths: list[Path]) -> list[DocumentRecord]:
        loaded_documents = [load_file_document(path) for path in paths]
        return self._ingest_loaded_documents(loaded_documents)

    def ingest_urls(self, urls: list[tuple[str | None, str]]) -> list[DocumentRecord]:
        loaded_documents = [load_url_document(url, title=title) for title, url in urls]
        return self._ingest_loaded_documents(loaded_documents)

    def _ingest_loaded_documents(self, loaded_documents: list[LoadedDocument]) -> list[DocumentRecord]:
        records: list[DocumentRecord] = []
        for document in loaded_documents:
            existing = self.registry.get_document(document.source_id)
            if existing:
                self.vector_store.delete_chunks(existing.chunk_ids)
                self.registry.remove_document(document.source_id)

            chunks = self._build_chunks(document)
            self.vector_store.add_chunks(chunks)
            record = self.registry.build_record(
                source_id=document.source_id,
                document_name=document.document_name,
                source_type=document.source_type,
                location=document.location,
                chunk_ids=[chunk.chunk_id for chunk in chunks],
            )
            records.append(record)

        self.registry.upsert_documents(records)
        return records

    def _build_chunks(self, document: LoadedDocument) -> list[ChunkRecord]:
        pieces = self.chunker.split_text(document.text)
        if not pieces:
            raise IngestionError(
                f"No chunks were produced for document '{document.document_name}'."
            )
        chunks: list[ChunkRecord] = []
        for index, piece in enumerate(pieces):
            digest = hashlib.sha256(
                f"{document.source_id}:{index}:{piece}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(
                ChunkRecord(
                    chunk_id=f"chunk-{digest}",
                    source_id=document.source_id,
                    document_name=document.document_name,
                    source_type=document.source_type,
                    location=document.location,
                    chunk_index=index,
                    text=piece,
                    metadata=document.metadata,
                )
            )
        return chunks
