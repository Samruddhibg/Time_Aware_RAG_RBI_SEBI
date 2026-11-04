"""
core/db.py — asyncpg connection pool with pgvector type registration.
"""
from __future__ import annotations

import asyncpg
from loguru import logger

from app.core.config import settings

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Return the shared connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
            init=_init_connection,
        )
        logger.info("asyncpg pool created (min=2, max=10)")
    return _pool


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Called on every new connection — register pgvector codec."""
    await conn.execute("SET search_path TO public")
    # Register the vector type as a text codec (pgvector returns arrays)
    await conn.set_type_codec(
        "vector",
        encoder=_encode_vector,
        decoder=_decode_vector,
        schema="public",
        format="text",
    )


def _encode_vector(value: list[float]) -> str:
    return "[" + ",".join(str(v) for v in value) + "]"


def _decode_vector(value: str) -> list[float]:
    return [float(v) for v in value.strip("[]").split(",")]


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("asyncpg pool closed")
