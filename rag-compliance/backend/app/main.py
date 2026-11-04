"""
main.py — FastAPI app factory with lifespan, CORS, and router registration.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import settings
from app.core.db import get_pool, close_pool
from app.routers.ask import router as ask_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info(f"Starting RAG-Compliance API (env={settings.log_level})")
    pool = await get_pool()
    app.state.pool = pool
    # Eagerly load embedder (avoids cold-start on first request)
    from app.services.embedder import embedder
    embedder._load()
    logger.info("Startup complete")
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────
    await close_pool()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG-Compliance: RBI/SEBI Circular Assistant",
        description=(
            "Time-aware Retrieval-Augmented Generation system for Indian regulatory "
            "circulars. Answers compliance questions citing active circulars only."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(ask_router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "RAG-Compliance API",
            "version": "1.0.0",
            "docs": "/docs",
        }

    return app


app = create_app()
