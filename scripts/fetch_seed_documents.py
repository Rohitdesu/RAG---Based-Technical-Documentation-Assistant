from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.container import get_container
from app.ingestion.loaders import load_url_document


def main() -> None:
    container = get_container()
    sources_path = ROOT_DIR / "documents" / "seed_sources.json"
    seed_dir = ROOT_DIR / "documents" / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)

    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    url_payloads: list[tuple[str | None, str]] = []

    for item in sources:
        title = item.get("title")
        url = item["url"]
        loaded = load_url_document(url, title=title)
        slug = re.sub(r"[^a-z0-9_]+", "_", loaded.document_name.lower().replace(" ", "_")).strip("_")
        snapshot_path = seed_dir / f"{slug[:80] or 'document'}.txt"
        snapshot_path.write_text(loaded.text, encoding="utf-8")
        url_payloads.append((title, url))

    records = container.ingestion.ingest_urls(url_payloads)
    print(f"Ingested {len(records)} seed documents into ChromaDB.")


if __name__ == "__main__":
    main()
