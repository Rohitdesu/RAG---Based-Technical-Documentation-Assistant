from __future__ import annotations

from typing import Iterable

import chromadb

from app.core.config import Settings
from app.core.gemini import GeminiGateway
from app.core.models import ChunkRecord, RetrievedChunk


class ChromaVectorStore:
    def __init__(self, settings: Settings, gemini: GeminiGateway):
        self.settings = settings
        self.gemini = gemini
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_or_create_collection(name=settings.collection_name)

    def count(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        embeddings = [
            self.gemini.embed_document(chunk.document_name, chunk.text) for chunk in chunks
        ]
        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "source_id": chunk.source_id,
                    "document_name": chunk.document_name,
                    "source_type": chunk.source_type,
                    "location": chunk.location,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                }
                for chunk in chunks
            ],
        )

    def delete_chunks(self, chunk_ids: Iterable[str]) -> None:
        chunk_ids = list(chunk_ids)
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)

    def similarity_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_embedding = self.gemini.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        retrieved: list[RetrievedChunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            similarity = 1 / (1 + float(distance))
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    source_id=str(metadata["source_id"]),
                    document_name=str(metadata["document_name"]),
                    source_type=str(metadata["source_type"]),
                    location=str(metadata["location"]),
                    chunk_index=int(metadata["chunk_index"]),
                    text=document,
                    similarity_score=round(similarity, 4),
                    metadata=metadata,
                )
            )
        return retrieved
