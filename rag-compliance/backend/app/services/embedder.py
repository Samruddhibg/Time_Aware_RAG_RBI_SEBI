"""
services/embedder.py — Sentence-Transformers wrapper with async offload.

Usage:
    from app.services.embedder import embedder
    vec = await embedder.aencode("What is the KYC limit?")
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Sequence

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import settings

# BGE models expect an instruction prefix for queries (not for passages)
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class AsyncEmbedder:
    """Thread-pool backed async wrapper around SentenceTransformer."""

    def __init__(self, model_name: str, dim: int, batch_size: int) -> None:
        self._model_name = model_name
        self._dim = dim
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedder: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedder loaded")
        return self._model

    # ── Sync (for ingest scripts) ─────────────────────────────────────────────

    def encode_query(self, text: str) -> list[float]:
        model = self._load()
        vec = model.encode(
            _QUERY_PREFIX + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.tolist()

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        return model.encode(
            list(texts),
            normalize_embeddings=True,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

    # ── Async (for FastAPI) ───────────────────────────────────────────────────

    async def aencode_query(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.encode_query, text)

    async def aencode_passages(self, texts: Sequence[str]) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.encode_passages, texts)


@lru_cache(maxsize=1)
def _get_embedder() -> AsyncEmbedder:
    return AsyncEmbedder(
        model_name=settings.embed_model,
        dim=settings.embed_dim,
        batch_size=settings.embed_batch,
    )


# Module-level singleton
embedder = _get_embedder()
