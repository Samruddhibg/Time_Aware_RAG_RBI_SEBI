"""
training/split_by_doc.py — Create document-disjoint train/test splits.

CRITICAL: Must be run before generating any synthetic data or fine-tuning.
The test set documents must NEVER appear in training.

Usage:
    python -m training.split_by_doc --db-url postgresql://... --out data/splits/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import asyncpg
from loguru import logger


async def split_documents(
    pool: asyncpg.Pool,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Return (train_doc_ids, test_doc_ids) with no overlap."""
    random.seed(seed)
    rows = await pool.fetch(
        "SELECT document_id, regulator FROM documents ORDER BY document_id"
    )
    doc_ids = [r["document_id"] for r in rows]
    random.shuffle(doc_ids)
    n_test = max(1, int(len(doc_ids) * test_fraction))
    test_ids = doc_ids[:n_test]
    train_ids = doc_ids[n_test:]
    return train_ids, test_ids


async def main(db_url: str, out_dir: Path, test_fraction: float, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=3)

    train_ids, test_ids = await split_documents(pool, test_fraction, seed)
    await pool.close()

    splits = {"train": train_ids, "test": test_ids, "seed": seed}
    split_path = out_dir / "doc_splits.json"
    split_path.write_text(json.dumps(splits, indent=2))

    logger.info(f"Train: {len(train_ids)} docs | Test: {len(test_ids)} docs")
    logger.info(f"Saved to {split_path}")
    logger.warning(
        "IMPORTANT: Never generate synthetic data for test documents. "
        "Test set must remain unseen during training."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--out", default="data/splits")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main(args.db_url, Path(args.out), args.test_fraction, args.seed))
