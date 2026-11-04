"""
conftest.py — pytest configuration and shared fixtures.

Fixtures:
  - db_pool    : asyncpg connection pool (module-scoped, requires live DB)
  - api_client : httpx AsyncClient pointed at the FastAPI app (no DB needed)
"""
from __future__ import annotations

import asyncio
import os
from typing import AsyncGenerator

import pytest


# ── pytest-asyncio mode ───────────────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring a live database connection",
    )


# ── Shared event loop (module-scoped so fixtures don't fight) ─────────────────

@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop so async fixtures live as long as their tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Database pool ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
async def db_pool():
    """
    asyncpg connection pool connected to the test database.

    Skips automatically if DATABASE_URL isn't reachable.
    Set DATABASE_URL in your environment or .env before running integration tests.
    """
    import asyncpg
    from app.core.config import settings

    try:
        pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=1,
            max_size=3,
            timeout=5,
        )
    except Exception as exc:
        pytest.skip(f"Database not reachable ({exc}). "
                    "Start postgres and re-run with a live DB.")
        return  # unreachable, but satisfies type checker

    yield pool
    await pool.close()


# ── FastAPI test client ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def api_client():
    """
    httpx TestClient for the FastAPI app.
    Does NOT require a database — use for unit/smoke tests only.
    """
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
