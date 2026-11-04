"""
training/mine_negatives.py — Hard negative mining for contrastive fine-tuning.

For each (query, positive) pair, finds the hardest non-positive passage
that ranks in positions [lo, hi] (skipping same-document chunks).

Usage:
    python -m training.mine_negatives \
        --pairs data/training/synthetic_pairs.jsonl \
        --db-url postgresql://... \
        --out data/training/pairs_with_negatives.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import asyncpg
import numpy as np
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.core.config import settings
from app.services.embedder import embedder


async def load_corpus(pool: asyncpg.Pool, train_doc_ids: list[str]) -> tuple:
    """Load all training-split chunks for building the negative index."""
    rows = await pool.fetch(
        """
        SELECT c.id, c.document_id, c.text, c.embedding
        FROM chunks c
        WHERE c.document_id = ANY($1::text[])
        ORDER BY c.id
        """,
        train_doc_ids,
    )
    ids = [r["id"] for r in rows]
    doc_ids = [r["document_id"] for r in rows]
    texts = [r["text"] for r in rows]
    # embeddings stored as list[float] already via codec
    embeddings = np.array([r["embedding"] for r in rows], dtype=np.float32)
    return ids, doc_ids, texts, embeddings


def mine_negatives_batch(
    queries: list[str],
    pos_chunk_ids: list[int],
    pos_doc_ids: list[str],
    corpus_ids: list[int],
    corpus_doc_ids: list[str],
    corpus_texts: list[str],
    corpus_embeddings: np.ndarray,
    lo: int = 10,
    hi: int = 40,
) -> list[str | None]:
    """Vectorised hard-negative mining."""
    Q = embedder.encode_passages(queries)  # (N, D)
    C = corpus_embeddings  # (M, D)
    # Cosine similarity (both already normalised)
    sims = Q @ C.T  # (N, M)
    ranked = np.argsort(-sims, axis=1)[:, :hi]  # top-hi per query

    negatives: list[str | None] = []
    for qi, row in enumerate(ranked):
        pos_id = pos_chunk_ids[qi]
        pos_doc = pos_doc_ids[qi]
        # Skip top-lo (too easy / might be true positives), skip same doc
        candidates = [
            j for j in row[lo:hi]
            if corpus_ids[j] != pos_id and corpus_doc_ids[j] != pos_doc
        ]
        if candidates:
            negatives.append(corpus_texts[candidates[0]])
        else:
            negatives.append(None)
    return negatives


async def main(
    pairs_path: Path,
    db_url: str,
    out_path: Path,
    lo: int,
    hi: int,
    batch_size: int,
) -> None:
    pairs = [json.loads(l) for l in pairs_path.read_text().splitlines() if l.strip()]
    train_doc_ids = list({p["document_id"] for p in pairs})
    logger.info(f"Loaded {len(pairs)} pairs from {len(train_doc_ids)} documents")

    pool = await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=3)
    corpus_ids, corpus_doc_ids, corpus_texts, corpus_embeddings = await load_corpus(
        pool, train_doc_ids
    )
    await pool.close()

    logger.info(f"Corpus: {len(corpus_ids)} chunks, embedding shape {corpus_embeddings.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, 0

    with open(out_path, "w") as f:
        for i in tqdm(range(0, len(pairs), batch_size), desc="Mining negatives"):
            batch = pairs[i : i + batch_size]
            queries = [p["anchor"] for p in batch]
            pos_ids = [p["chunk_id"] for p in batch]
            pos_docs = [p["document_id"] for p in batch]

            negs = mine_negatives_batch(
                queries, pos_ids, pos_docs,
                corpus_ids, corpus_doc_ids, corpus_texts, corpus_embeddings,
                lo=lo, hi=hi,
            )
            for pair, neg in zip(batch, negs):
                if neg is None:
                    skipped += 1
                    continue
                record = {
                    "anchor": pair["anchor"],
                    "positive": pair["positive"],
                    "negative": neg,
                    "chunk_id": pair["chunk_id"],
                    "document_id": pair["document_id"],
                }
                f.write(json.dumps(record) + "\n")
                written += 1

    logger.success(f"Wrote {written} triplets, skipped {skipped} (no hard neg found)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--out", default="data/training/pairs_with_negatives.jsonl")
    parser.add_argument("--lo", type=int, default=10)
    parser.add_argument("--hi", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    asyncio.run(main(Path(args.pairs), args.db_url, Path(args.out), args.lo, args.hi, args.batch_size))
