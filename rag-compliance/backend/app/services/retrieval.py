"""
services/retrieval.py — Dense + lexical retrieval with RRF fusion and temporal filtering.
"""
from __future__ import annotations

import time
from typing import Any

import asyncpg
from loguru import logger

from app.core.config import settings
from app.services.embedder import embedder

# ── SQL queries ───────────────────────────────────────────────────────────────

_DENSE_SQL_ACTIVE = """
SELECT
    c.id,
    c.document_id,
    c.chunk_index,
    c.section,
    c.text,
    c.token_count,
    d.circular_no,
    d.title,
    d.issue_date,
    d.status,
    1 - (c.embedding <=> $1::vector) AS similarity
FROM chunks c
JOIN documents d ON d.document_id = c.document_id
WHERE d.status = 'active'
ORDER BY c.embedding <=> $1::vector
LIMIT $2
"""

_DENSE_SQL_ALL = """
SELECT
    c.id,
    c.document_id,
    c.chunk_index,
    c.section,
    c.text,
    c.token_count,
    d.circular_no,
    d.title,
    d.issue_date,
    d.status,
    1 - (c.embedding <=> $1::vector) AS similarity
FROM chunks c
JOIN documents d ON d.document_id = c.document_id
ORDER BY c.embedding <=> $1::vector
LIMIT $2
"""

_LEXICAL_SQL_ACTIVE = """
SELECT
    c.id,
    c.document_id,
    c.chunk_index,
    c.section,
    c.text,
    c.token_count,
    d.circular_no,
    d.title,
    d.issue_date,
    d.status,
    ts_rank(c.tsv, plainto_tsquery('english', $1)) AS similarity
FROM chunks c
JOIN documents d ON d.document_id = c.document_id
WHERE d.status = 'active'
  AND c.tsv @@ plainto_tsquery('english', $1)
ORDER BY similarity DESC
LIMIT $2
"""

_LEXICAL_SQL_ALL = """
SELECT
    c.id,
    c.document_id,
    c.chunk_index,
    c.section,
    c.text,
    c.token_count,
    d.circular_no,
    d.title,
    d.issue_date,
    d.status,
    ts_rank(c.tsv, plainto_tsquery('english', $1)) AS similarity
FROM chunks c
JOIN documents d ON d.document_id = c.document_id
WHERE c.tsv @@ plainto_tsquery('english', $1)
ORDER BY similarity DESC
LIMIT $2
"""

_FETCH_CHUNKS_SQL = """
SELECT
    c.id,
    c.document_id,
    c.chunk_index,
    c.section,
    c.text,
    c.token_count,
    d.circular_no,
    d.title,
    d.issue_date,
    d.status
FROM chunks c
JOIN documents d ON d.document_id = c.document_id
WHERE c.id = ANY($1::bigint[])
"""


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _rrf(*ranked_lists: list[int], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


# ── Main retrieval function ───────────────────────────────────────────────────

async def retrieve(
    pool: asyncpg.Pool,
    question: str,
    historical: bool = False,
    top_k_dense: int | None = None,
    top_k_lexical: int | None = None,
    top_k_rerank: int | None = None,
) -> tuple[list[dict[str, Any]], float, int]:
    """
    Returns:
        (chunks, best_similarity, retrieval_ms)
        chunks — list of dicts ordered by RRF score (top top_k_rerank)
    """
    kd = top_k_dense or settings.top_k_dense
    kl = top_k_lexical or settings.top_k_lexical
    kr = top_k_rerank or settings.top_k_rerank

    t0 = time.perf_counter()

    # 1. Embed query
    qvec = await embedder.aencode_query(question)
    vec_str = "[" + ",".join(str(v) for v in qvec) + "]"

    dense_sql = _DENSE_SQL_ALL if historical else _DENSE_SQL_ACTIVE
    lexical_sql = _LEXICAL_SQL_ALL if historical else _LEXICAL_SQL_ACTIVE

    async with pool.acquire() as conn:
        # 2. Dense retrieval (HNSW)
        dense_rows = await conn.fetch(dense_sql, vec_str, kd)
        # 3. Lexical retrieval (FTS)
        lexical_rows = await conn.fetch(lexical_sql, question, kl)

    dense_ids = [r["id"] for r in dense_rows]
    lexical_ids = [r["id"] for r in lexical_rows]

    # 4. RRF fusion
    fused_ids = _rrf(dense_ids, lexical_ids)[:kr]

    # Build id→row lookup from both result sets
    row_lookup: dict[int, asyncpg.Record] = {}
    for r in dense_rows:
        row_lookup[r["id"]] = r
    for r in lexical_rows:
        if r["id"] not in row_lookup:
            row_lookup[r["id"]] = r

    # 5. Collect final chunks preserving RRF order
    best_sim = 0.0
    chunks: list[dict[str, Any]] = []
    for cid in fused_ids:
        if cid not in row_lookup:
            continue
        r = row_lookup[cid]
        sim = float(r.get("similarity") or 0.0)
        if sim > best_sim:
            best_sim = sim
        issue_date = r["issue_date"]
        chunks.append(
            {
                "chunk_id": r["id"],
                "document_id": r["document_id"],
                "circular_no": r.get("circular_no"),
                "title": r.get("title"),
                "section": r.get("section"),
                "text": r["text"],
                "token_count": r.get("token_count"),
                "issue_date": str(issue_date) if issue_date else None,
                "status": r.get("status", "active"),
                "similarity": round(sim, 4),
                "snippet": r["text"][:300],
            }
        )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        f"Retrieval done: dense={len(dense_rows)} lexical={len(lexical_rows)} "
        f"fused={len(chunks)} best_sim={best_sim:.3f} {elapsed_ms}ms"
    )
    return chunks, best_sim, elapsed_ms
