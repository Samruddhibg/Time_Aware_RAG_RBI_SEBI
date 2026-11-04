"""
services/rag.py — Orchestration: retrieve → build context → stream LLM → log.
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import asyncpg
from loguru import logger

from app.core.config import settings
from app.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, build_context
from app.services.retrieval import retrieve
from app.services.llm import stream_completion

_LOG_INSERT = """
INSERT INTO query_log (question, historical, chunk_ids, best_sim, refused,
                       retrieval_ms, ttft_ms, total_ms, answer)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
RETURNING id
"""

_REFUSAL_MSG = (
    "I could not find a definitive answer in the available RBI/SEBI circulars. "
    "Please consult the original regulator documents."
)


async def rag_stream(
    pool: asyncpg.Pool,
    question: str,
    historical: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """
    Async generator — yields dicts:
        {"type": "sources", "data": [...]}    — emitted first
        {"type": "token",   "data": "word"}   — one per LLM token
        {"type": "done",    "data": {...}}     — final metadata
        {"type": "error",   "data": "msg"}    — on failure
    """
    t_start = time.perf_counter()

    try:
        # ── 1. Retrieve ───────────────────────────────────────────────────────
        chunks, best_sim, retrieval_ms = await retrieve(
            pool, question, historical=historical
        )

        # ── 2. Refusal check ──────────────────────────────────────────────────
        refused = not chunks or best_sim < settings.min_sim_threshold

        # Always emit sources so frontend can display them
        yield {"type": "sources", "data": chunks}

        if refused:
            logger.info(f"Refusing: best_sim={best_sim:.3f} chunks={len(chunks)}")
            # Log and return
            query_id = await _log(
                pool,
                question=question,
                historical=historical,
                chunk_ids=[c["chunk_id"] for c in chunks],
                best_sim=best_sim,
                refused=True,
                retrieval_ms=retrieval_ms,
                ttft_ms=None,
                total_ms=int((time.perf_counter() - t_start) * 1000),
                answer=_REFUSAL_MSG,
            )
            yield {"type": "token", "data": _REFUSAL_MSG}
            yield {
                "type": "done",
                "data": {
                    "query_id": query_id,
                    "refused": True,
                    "retrieval_ms": retrieval_ms,
                    "total_ms": int((time.perf_counter() - t_start) * 1000),
                },
            }
            return

        # ── 3. Build prompt ───────────────────────────────────────────────────
        context = build_context(chunks)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context, question=question
        )

        # ── 4. Stream LLM ─────────────────────────────────────────────────────
        tokens: list[str] = []
        ttft_ms: int | None = None

        async for token in stream_completion(SYSTEM_PROMPT, user_prompt):
            if ttft_ms is None:
                ttft_ms = int((time.perf_counter() - t_start) * 1000)
            tokens.append(token)
            yield {"type": "token", "data": token}

        answer = "".join(tokens)
        total_ms = int((time.perf_counter() - t_start) * 1000)

        # ── 5. Log ────────────────────────────────────────────────────────────
        query_id = await _log(
            pool,
            question=question,
            historical=historical,
            chunk_ids=[c["chunk_id"] for c in chunks],
            best_sim=best_sim,
            refused=False,
            retrieval_ms=retrieval_ms,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            answer=answer,
        )

        yield {
            "type": "done",
            "data": {
                "query_id": query_id,
                "refused": False,
                "retrieval_ms": retrieval_ms,
                "ttft_ms": ttft_ms,
                "total_ms": total_ms,
            },
        }

    except Exception as exc:
        logger.exception("rag_stream error")
        yield {"type": "error", "data": str(exc)}


async def _log(
    pool: asyncpg.Pool,
    *,
    question: str,
    historical: bool,
    chunk_ids: list[int],
    best_sim: float,
    refused: bool,
    retrieval_ms: int,
    ttft_ms: int | None,
    total_ms: int,
    answer: str,
) -> int:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            _LOG_INSERT,
            question,
            historical,
            chunk_ids,
            best_sim,
            refused,
            retrieval_ms,
            ttft_ms,
            total_ms,
            answer,
        )
    return row["id"]
