from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    app_name: str = "RAG-Based Technical Documentation Assistant"
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    generation_model: str = os.getenv("GENERATION_MODEL", "gemini-3.5-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
    chroma_path: Path = Path(os.getenv("CHROMA_PATH", ROOT_DIR / "chroma_db"))
    collection_name: str = os.getenv("CHROMA_COLLECTION", "technical_documentation")
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    data_dir: Path = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
    documents_dir: Path = Path(os.getenv("DOCUMENTS_DIR", ROOT_DIR / "documents"))

    @property
    def feedback_path(self) -> Path:
        return self.data_dir / "feedback.jsonl"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "indexed_documents.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    return settings
