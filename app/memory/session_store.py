from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from app.core.exceptions import SessionNotFoundError
from app.core.models import ChatMessage, SessionRecord


class SessionStore:
    """Persist chat history per session using one JSON file per session."""

    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> SessionRecord:
        now = datetime.utcnow()
        session = SessionRecord(
            session_id=str(uuid.uuid4()),
            chat_history=[],
            created_at=now,
            updated_at=now,
        )
        self._write_session(session)
        return session

    def get_session(self, session_id: str) -> SessionRecord:
        path = self._session_path(session_id)
        if not path.exists():
            raise SessionNotFoundError(f"Session '{session_id}' was not found.")
        data = json.loads(path.read_text(encoding="utf-8"))
        return SessionRecord.model_validate(data)

    def append_messages(
        self,
        session_id: str,
        *,
        user_message: str,
        assistant_message: str,
    ) -> SessionRecord:
        session = self.get_session(session_id)
        now = datetime.utcnow()
        session.chat_history.extend(
            [
                ChatMessage(role="user", content=user_message, timestamp=now),
                ChatMessage(role="assistant", content=assistant_message, timestamp=now),
            ]
        )
        session.updated_at = now
        self._write_session(session)
        return session

    def delete_session(self, session_id: str) -> None:
        path = self._session_path(session_id)
        if not path.exists():
            raise SessionNotFoundError(f"Session '{session_id}' was not found.")
        path.unlink()

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def _write_session(self, session: SessionRecord) -> None:
        self._session_path(session.session_id).write_text(
            json.dumps(session.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
