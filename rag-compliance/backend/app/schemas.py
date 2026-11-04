"""
schemas.py — Pydantic request / response models.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="User query")
    historical: bool = Field(
        default=False,
        description="If True, include superseded documents in retrieval.",
    )


class FeedbackRequest(BaseModel):
    query_id: int = Field(..., description="query_log.id from the /ask response header")
    rating: int = Field(..., ge=-1, le=1, description="-1 = thumbs down, 1 = thumbs up")
    comment: str | None = Field(default=None, max_length=1000)


# ── Response ──────────────────────────────────────────────────────────────────

class Source(BaseModel):
    chunk_id: int
    document_id: str
    circular_no: str | None
    title: str | None
    section: str | None
    issue_date: date | None
    status: str
    similarity: float
    snippet: str = Field(..., description="First 300 chars of the chunk")


class AskResponse(BaseModel):
    query_id: int
    question: str
    answer: str
    refused: bool
    sources: list[Source]
    retrieval_ms: int
    ttft_ms: int | None
    total_ms: int


class SSEEvent(BaseModel):
    """Shape of each SSE data payload."""
    type: str  # "token" | "sources" | "done" | "error"
    data: Any
