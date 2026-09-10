"""API behavior tests without a live ADK model call."""

from fastapi.testclient import TestClient

from agents.base import AppError, AskResponse, Source, SourceKind
from main import create_app


class ScenarioCoordinator:
    async def ask(self, question):
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
