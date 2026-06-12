from __future__ import annotations

from datetime import datetime
import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.container import AppContainer, get_container
from app.core.exceptions import (
    ConfigurationError,
    IngestionError,
    SessionNotFoundError,
    UpstreamServiceError,
)
from app.core.models import (
    DocumentListResponse,
    FeedbackRequest,
    FeedbackResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SessionCreateResponse,
    SessionDeleteResponse,
    SessionRecord,
)


router = APIRouter()


def _parse_url_form(urls: str | None) -> list[tuple[str | None, str]]:
    if not urls:
        return []
    try:
        parsed = json.loads(urls)
        if isinstance(parsed, list):
            return [
                (item.get("title"), item["url"]) if isinstance(item, dict) else (None, str(item))
                for item in parsed
            ]
    except json.JSONDecodeError:
        pass
    return [(None, line.strip()) for line in urls.splitlines() if line.strip()]


@router.post("/query", response_model=QueryResponse)
def query_docs(
    request: QueryRequest,
    container: AppContainer = Depends(get_container),
):
    if not request.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question is required.")
    try:
        if container.vector_store.count() == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No documents are indexed yet. Use POST /ingest first.",
            )
        if request.session_id:
            session = container.sessions.get_session(request.session_id)
        else:
            session = container.sessions.create_session()
        response = container.query_workflow.invoke(
            question=request.question.strip(),
            session_id=session.session_id,
            chat_history=session.chat_history,
        )
        container.sessions.append_messages(
            session.session_id,
            user_message=request.question.strip(),
            assistant_message=response.answer,
        )
        return response
    except ConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except UpstreamServiceError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive API guard
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected query error: {exc}",
        ) from exc


@router.post("/ingest", response_model=IngestResponse)
async def ingest_docs(
    files: list[UploadFile] | None = File(default=None),
    urls: str | None = Form(default=None),
    container: AppContainer = Depends(get_container),
):
    parsed_urls = _parse_url_form(urls)
    if not files and not parsed_urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one uploaded file or one URL.",
        )

    uploaded_files: list[tuple[str, bytes]] = []
    if files:
        for upload in files:
            data = await upload.read()
            if not data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Uploaded file '{upload.filename}' is empty.",
                )
            uploaded_files.append((upload.filename or "uploaded_document.txt", data))

    try:
        records = []
        if uploaded_files:
            records.extend(container.ingestion.ingest_files(uploaded_files))
        if parsed_urls:
            records.extend(container.ingestion.ingest_urls(parsed_urls))
    except IngestionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except UpstreamServiceError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API guard
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected ingestion error: {exc}",
        ) from exc

    return IngestResponse(message="Documents ingested successfully.", documents=records)


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(container: AppContainer = Depends(get_container)):
    return DocumentListResponse(documents=container.registry.list_documents())


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    request: FeedbackRequest,
    container: AppContainer = Depends(get_container),
):
    container.append_feedback(
        {
            **request.model_dump(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
    return FeedbackResponse(message="Feedback recorded successfully.")


@router.post("/session/create", response_model=SessionCreateResponse)
def create_session(container: AppContainer = Depends(get_container)):
    session = container.sessions.create_session()
    return SessionCreateResponse(
        session_id=session.session_id,
        created_at=session.created_at,
    )


@router.get("/session/{session_id}", response_model=SessionRecord)
def get_session(session_id: str, container: AppContainer = Depends(get_container)):
    try:
        return container.sessions.get_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.delete("/session/{session_id}", response_model=SessionDeleteResponse)
def delete_session(session_id: str, container: AppContainer = Depends(get_container)):
    try:
        container.sessions.delete_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return SessionDeleteResponse(message="Session cleared successfully.")
