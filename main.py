"""FastAPI entrypoint. Run with: uvicorn main:app --reload."""

import os
import warnings
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from agents import Coordinator
from agents.base import AppError, AskRequest, AskResponse, SourceKind
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

    @app.post("/ask", response_model=AskResponse, summary="Ask a grounded chatbot question")
    async def ask(payload: AskRequest, request: Request) -> AskResponse:
        try:
            active = getattr(request.app.state, "coordinator", None) or configured_coordinator()
            return await active.ask(payload.question)
        except AppError as exc:
            raise HTTPException(status_code=exc.http_status, detail=exc.message) from exc

    return app


app = create_app()
