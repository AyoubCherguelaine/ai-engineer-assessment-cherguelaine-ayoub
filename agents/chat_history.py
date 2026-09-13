"""Small SQLite store for API chat history, separate from the IMDb index."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .base import AskResponse


class ChatHistoryStore:
    """Persist one completed question/answer exchange per row."""

    def __init__(self, database: str | Path):
        self.database = Path(database)

    def save(self, question: str, response: AskResponse, model_used: str) -> None:
        """Store only chat metadata; IMDb's database is never opened here."""
        if not response.session_id:
            raise ValueError("A session_id is required to save chat history.")

        self.database.parent.mkdir(parents=True, exist_ok=True)
        sources = [source.model_dump(mode="json") for source in response.sources]
        source_used = list(dict.fromkeys(source["kind"] for source in sources))
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    source_used TEXT NOT NULL,
                    source_details TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id "
                "ON chat_messages(session_id)"
            )
            connection.execute(
                """
                INSERT INTO chat_messages
                    (session_id, question, answer, model_used, source_used, source_details)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    response.session_id,
                    question,
                    response.answer,
                    model_used,
                    json.dumps(source_used),
                    json.dumps(sources),
                ),
            )
