from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.core.exceptions import IngestionError
from app.core.models import LoadedDocument


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm"}


def _normalize_text(value: str) -> str:
    lines = [line.strip() for line in value.splitlines()]
    compact = "\n".join(line for line in lines if line)
    return compact.strip()


def _extract_html_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled HTML Document"
    text = soup.get_text("\n", strip=True)
    return title, _normalize_text(text)


def _source_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def load_file_document(path: Path, content: bytes | None = None) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file type '{suffix}'. Supported types are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    raw_text = content.decode("utf-8", errors="ignore") if content is not None else path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        document_name, text = _extract_html_text(raw_text)
    else:
        document_name = path.stem.replace("_", " ").strip() or path.name
        text = _normalize_text(raw_text)

    if not text:
        raise IngestionError(f"No indexable text found in file '{path.name}'.")

    return LoadedDocument(
        source_id=_source_id("file", str(path.resolve())),
        document_name=document_name,
        source_type="file",
        location=str(path.resolve()),
        text=text,
        metadata={"filename": path.name},
    )


def load_url_document(url: str, title: str | None = None) -> LoadedDocument:
    try:
        response = httpx.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise IngestionError(f"Failed to fetch URL '{url}': {exc}") from exc
    document_name, text = _extract_html_text(response.text)
    parsed = urlparse(str(response.url))
    final_title = title or document_name or parsed.netloc
    if not text:
        raise IngestionError(f"No indexable text found at URL '{url}'.")

    return LoadedDocument(
        source_id=_source_id("url", str(response.url)),
        document_name=final_title,
        source_type="url",
        location=str(response.url),
        text=text,
        metadata={"host": parsed.netloc},
    )
