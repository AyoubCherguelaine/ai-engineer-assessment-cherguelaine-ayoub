import sqlite3

from fastapi.testclient import TestClient

from agents.base import AskResponse, Source, SourceKind
from agents.coordinator import CerebrasLiteLLMClient, Coordinator
from agents.imdb import IMDbAgent
from google.genai import types
from main import create_app


class StubCoordinator:
    async def ask(self, question):
        return AskResponse(
            answer="Inception was directed by Christopher Nolan.",
            sources=[
                Source(
                    kind=SourceKind.IMDB,
                    label="IMDb Non-Commercial Datasets",
                    detail="Inception (2010)",
                )
            ],
        )


def test_adk_agent_uses_litellm_and_exposes_the_two_tools():
    coordinator = Coordinator(model="cerebras/gpt-oss-120b", superhero=None)
    agent = coordinator._build_agent([], [], [])

    assert type(agent.model).__name__ == "LiteLlm"
    assert [tool.__name__ for tool in agent.tools] == ["search_imdb", "search_superhero"]


def test_cerebras_client_removes_replayed_reasoning_content():
    messages = [
        {"role": "user", "content": "Who directed Inception?"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Internal reasoning that Cerebras rejects on replay.",
        },
    ]

    sanitized = CerebrasLiteLLMClient._without_reasoning_content(messages)

    assert "reasoning_content" not in sanitized[1]
    assert "reasoning_content" in messages[1]


def test_visible_text_excludes_adk_thought_parts():
    parts = [
        types.Part(text="Internal reasoning.", thought=True),
        types.Part(text="Batman is a DC superhero."),
    ]

    assert Coordinator._visible_text(parts) == "Batman is a DC superhero."


def test_imdb_agent_searches_a_prepared_sqlite_index(tmp_path):
    database = tmp_path / "imdb.db"
    with sqlite3.connect(database) as connection:
        connection.execute("""
            CREATE TABLE imdb_titles (
                tconst TEXT PRIMARY KEY, primary_title TEXT, original_title TEXT,
                normalized_title TEXT, normalized_search TEXT, start_year INTEGER, genres TEXT, directors TEXT,
                average_rating REAL, num_votes INTEGER
            )
        """)
        connection.execute(
            "INSERT INTO imdb_titles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("tt1375666", "Inception", "Inception", "inception", "inceptioninceptionscifithrillerchristophernolan", 2010, "Action,Sci-Fi,Thriller", "Christopher Nolan", 8.8, 2_600_000),
        )

    result = IMDbAgent(database).search("Who directed Inception?")
    assert result[0]["primary_title"] == "Inception"
    assert result[0]["average_rating"] == 8.8


def test_question_validation_and_openapi_endpoint():
    client = TestClient(create_app(StubCoordinator()))

    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.get("/openapi.json").status_code == 200


def test_missing_cerebras_key_is_a_clear_service_error(monkeypatch):
    monkeypatch.setenv("CEREBRAS_API_KEY", "")
    monkeypatch.delenv("LITELLM_MODEL", raising=False)
    with TestClient(create_app()) as client:
        response = client.post("/ask", json={"question": "Who directed Inception?"})
    assert response.status_code == 503
    assert response.json()["detail"] == "CEREBRAS_API_KEY is not configured."


def test_stubbed_api_returns_its_grounded_source():
    client = TestClient(create_app(StubCoordinator()))
    response = client.post("/ask", json={"question": "Who directed Inception?"})

    assert response.status_code == 200
    assert response.json()["sources"][0]["kind"] == "imdb"
