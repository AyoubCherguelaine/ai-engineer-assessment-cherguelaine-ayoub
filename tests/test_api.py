"""API behavior tests without a live ADK model call."""

import json
import sqlite3

from fastapi.testclient import TestClient

from agents.base import AppError, AskResponse, Source, SourceKind
from agents.chat_history import ChatHistoryStore
from main import create_app


class ScenarioCoordinator:
    async def ask(self, question, session_id=None):
        if question == "trigger failure":
            raise AppError("The Superhero API is temporarily unavailable.", 502)
        return AskResponse(
            answer="Grounded answer verified against retrieved evidence.",
            sources=[
                Source(
                    kind=SourceKind.IMDB,
                    label="IMDb Non-Commercial Datasets",
                    detail="The Dark Knight (2008)",
                ),
                Source(
                    kind=SourceKind.SUPERHERO_API,
                    label="Superhero API",
                    detail="Search result: Batman",
                ),
            ],
        )


def test_api_returns_the_answer_and_sources_from_the_coordinator():
    client = TestClient(create_app(ScenarioCoordinator()))
    response = client.post("/ask", json={"question": "Is Batman in The Dark Knight?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Grounded answer verified against retrieved evidence."
    assert {source["kind"] for source in response.json()["sources"]} == {
        "imdb",
        "superhero_api",
    }
    assert response.json()["session_id"]
    assert response.json()["model_used"] == "unknown"


def test_api_rejects_invalid_payloads_without_calling_the_coordinator():
    client = TestClient(create_app(ScenarioCoordinator()))
    assert client.post("/ask", json={}).status_code == 422
    assert client.post("/ask", json={"question": "  "}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 1_001}).status_code == 422


def test_expected_upstream_failure_is_exposed():
    client = TestClient(create_app(ScenarioCoordinator()))
    response = client.post("/ask", json={"question": "trigger failure"})

    assert response.status_code == 502
    assert response.json()["detail"] == "The Superhero API is temporarily unavailable."


def test_chat_history_uses_a_separate_sqlite_database(tmp_path):
    database = tmp_path / "chat_history.db"
    response = AskResponse(
        answer="Batman is a DC superhero.",
        sources=[
            Source(
                kind=SourceKind.SUPERHERO_API,
                label="Superhero API",
                detail="Search result: Batman",
            )
        ],
        session_id="session-123",
        model_used="cerebras/gpt-oss-120b",
    )

    ChatHistoryStore(database).save(
        "Who is Batman?", response, response.model_used or "unknown"
    )

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT session_id, question, answer, model_used, source_used, source_details "
            "FROM chat_messages"
        ).fetchone()
    assert row[:4] == (
        "session-123",
        "Who is Batman?",
        "Batman is a DC superhero.",
        "cerebras/gpt-oss-120b",
    )
    assert json.loads(row[4]) == ["superhero_api"]
    assert json.loads(row[5]) == [
        {
            "kind": "superhero_api",
            "label": "Superhero API",
            "detail": "Search result: Batman",
        }
    ]
