"""FastAPI entrypoint. Run with: uvicorn main:app --reload."""

import os
import sqlite3
import warnings
from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from agents import Coordinator
from agents.base import AppError, AskRequest, AskResponse, SourceKind
from agents.chat_history import ChatHistoryStore
from agents.superhero import SuperheroAgent

# ADK currently labels its function-declaration JSON schema as experimental.
# It is required for our typed function tools and the one-time warning is not
# actionable for API operators, so keep it out of normal server logs.
warnings.filterwarnings(
    "ignore",
    message=r"\[EXPERIMENTAL\] feature FeatureName\.JSON_SCHEMA_FOR_FUNC_DECL is enabled\.",
    category=UserWarning,
    module=r"google\.adk\.models\.llm_request",
)

# Preserve the original public imports after moving implementation into agents/.
__all__ = ["AskRequest", "Coordinator", "SourceKind", "create_app", "app"]


def configured_coordinator() -> Coordinator:
    timeout = float(os.getenv("HTTP_TIMEOUT_SECONDS", "15"))
    superhero_timeout = float(os.getenv("SUPERHERO_TIMEOUT_SECONDS", "30"))
    if not 1 <= timeout <= 60:
        raise AppError("HTTP_TIMEOUT_SECONDS must be between 1 and 60.")
    if not 1 <= superhero_timeout <= 60:
        raise AppError("SUPERHERO_TIMEOUT_SECONDS must be between 1 and 60.")
    model = os.getenv(
        "LITELLM_MODEL",
        f"cerebras/{os.getenv('CEREBRAS_MODEL', 'gpt-oss-120b')}",
    )
    if model.startswith("cerebras/") and not os.getenv("CEREBRAS_API_KEY"):
        raise AppError("CEREBRAS_API_KEY is not configured.")
    superhero = SuperheroAgent(os.getenv("SUPERHERO_API_TOKEN", ""), superhero_timeout)
    return Coordinator(model=model, superhero=superhero, timeout=timeout)


def create_app(coordinator: Coordinator | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        load_dotenv()
        yield

    app = FastAPI(
        title="Cinema Chatbot API",
        version="1.0.0",
        lifespan=lifespan,
        description="Google ADK + LiteLLM answers grounded in IMDb data and/or Superhero API data.",
    )
    app.state.coordinator = coordinator
    app.state.chat_history = ChatHistoryStore(
        os.getenv("CHAT_HISTORY_DB", "data/chat_history.db")
    )

    @app.post("/ask", response_model=AskResponse, summary="Ask a grounded chatbot question")
    async def ask(payload: AskRequest, request: Request) -> AskResponse:
        try:
            active = getattr(request.app.state, "coordinator", None) or configured_coordinator()
            response = await active.ask(payload.question, session_id=payload.session_id)
            # Keep history persistence outside the agent and in its own SQLite file.
            # This database is deliberately unrelated to data/imdb.db.
            if not response.session_id:
                response.session_id = payload.session_id
            if not response.session_id:
                response.session_id = str(uuid4())
            model_used = response.model_used or getattr(active, "model", "unknown")
            response.model_used = model_used
            try:
                request.app.state.chat_history.save(payload.question, response, model_used)
            except (OSError, sqlite3.Error, ValueError) as exc:
                raise AppError("Chat history could not be saved.") from exc
            return response
        except AppError as exc:
            raise HTTPException(status_code=exc.http_status, detail=exc.message) from exc

    return app


app = create_app()
