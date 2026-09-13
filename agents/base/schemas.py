from enum import Enum

from fastapi import status
from pydantic import BaseModel, Field, field_validator


class SourceKind(str, Enum):
    IMDB = "imdb"
    # Compatibility alias for callers that imported the old generic dataset name.
    DATASET = "imdb"
    SUPERHERO_API = "superhero_api"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1_000, description="Natural-language question to answer.")
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional ID used to group chat messages. A UUID is generated when omitted.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        return value

    @field_validator("session_id")
    @classmethod
    def session_id_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("session_id must not be blank")
        return value


class Source(BaseModel):
    kind: SourceKind
    label: str
    detail: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    session_id: str | None = None
    model_used: str | None = None


class AppError(Exception):
    """Expected upstream or configuration failure safe to return to an API caller."""

    def __init__(self, message: str, http_status: int = status.HTTP_503_SERVICE_UNAVAILABLE):
        self.message = message
        self.http_status = http_status
