"""
routers/ask.py — POST /ask (SSE) and POST /feedback.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from app.core.db import get_pool
from app.schemas import AskRequest, FeedbackRequest
from app.services.rag import rag_stream

router = APIRouter(prefix="/api", tags=["ask"])

_FEEDBACK_INSERT = """
INSERT INTO feedback (query_id, rating, comment) VALUES ($1, $2, $3) RETURNING id
"""


async def _get_pool(request: Request):
    return request.app.state.pool


async def _event_generator(
    pool, question: str, historical: bool
) -> AsyncIterator[str]:
    """Convert rag_stream dicts into SSE-formatted strings."""
    async for event in rag_stream(pool, question, historical=historical):
        yield f"data: {json.dumps(event)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/ask")
async def ask(body: AskRequest, request: Request):
    """
    SSE endpoint — streams tokens, sources, and final metadata.

    Response format (each line):
        data: {"type": "sources", "data": [...]}
        data: {"type": "token",   "data": "word"}
        data: {"type": "done",    "data": {...}}
        data: [DONE]
    """
    pool = request.app.state.pool
    logger.info(f"[/ask] q={body.question[:80]!r} historical={body.historical}")

    return StreamingResponse(
        _event_generator(pool, body.question, body.historical),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/feedback", status_code=201)
async def feedback(body: FeedbackRequest, request: Request):
    """Store user rating for a completed query."""
    pool = request.app.state.pool
    async with pool.acquire() as conn:
        # Check query exists
        exists = await conn.fetchval(
            "SELECT id FROM query_log WHERE id = $1", body.query_id
        )
        if not exists:
            raise HTTPException(status_code=404, detail="query_id not found")
        row = await conn.fetchrow(
            _FEEDBACK_INSERT, body.query_id, body.rating, body.comment
        )
    return {"feedback_id": row["id"], "status": "recorded"}


@router.get("/health")
async def health():
    return {"status": "ok"}
