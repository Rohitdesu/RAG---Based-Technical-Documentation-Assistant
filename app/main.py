from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings


settings = get_settings()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="A LangGraph-powered RAG assistant for technical documentation.",
)
app.include_router(router)


@app.get("/")
def root():
    return {
        "message": settings.app_name,
        "docs": "/docs",
    }
