"""
training/gen_synthetic.py — Generate (query, positive_chunk) pairs using an LLM.

These pairs are used ONLY for training (train split documents only).
Test documents must be excluded.

Usage:
    python -m training.gen_synthetic \
        --splits data/splits/doc_splits.json \
        --db-url postgresql://... \
        --out data/training/synthetic_pairs.jsonl \
        --pairs-per-chunk 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import asyncpg
import httpx
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.core.config import settings

_SYSTEM = (
    "You are a compliance analyst. Given a text passage from an RBI or SEBI regulatory "
    "document, generate realistic questions that a bank compliance officer would ask. "
    "Questions should be specific, require reading this exact passage to answer, and "
    "include at least one question asking about dates, limits, or circular numbers."
)

_USER_TEMPLATE = """Passage (from {circular_no}, {issue_date}):
---
{text}
---
Generate exactly {n} different, natural-language questions whose answer lies in the passage above.
Return ONLY a JSON array of strings, no other text.
Example: ["What is the KYC limit for...", "When did the RBI circular ..."]"""


async def _call_llm(
    client: httpx.AsyncClient,
    system: str,
    user: str,
) -> list[str]:
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False,
    }
    r = await client.post(
        f"{settings.llm_base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        timeout=30,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    # Parse JSON array
    try:
        questions = json.loads(content)
        return [q for q in questions if isinstance(q, str)]
    except json.JSONDecodeError:
        # Try to extract JSON array from free-form text
        import re
        m = re.search(r"\[.*?\]", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return []


async def generate_pairs(
    pool: asyncpg.Pool,
    train_doc_ids: list[str],
    pairs_per_chunk: int,
    max_chunks: int,
    out_path: Path,
) -> None:
    # Fetch train chunks
    rows = await pool.fetch(
        """
        SELECT c.id, c.text, c.section, d.circular_no, d.issue_date, d.document_id
        FROM chunks c JOIN documents d ON d.document_id = c.document_id
        WHERE c.document_id = ANY($1::text[])
        ORDER BY random()
        LIMIT $2
        """,
        train_doc_ids,
        max_chunks,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    async with httpx.AsyncClient() as client:
        with open(out_path, "w") as f:
            for row in tqdm(rows, desc="Generating queries"):
                user_msg = _USER_TEMPLATE.format(
                    circular_no=row["circular_no"] or row["document_id"],
                    issue_date=row["issue_date"] or "unknown",
                    text=row["text"][:1000],
                    n=pairs_per_chunk,
                )
                try:
                    questions = await _call_llm(client, _SYSTEM, user_msg)
                    for q in questions[:pairs_per_chunk]:
                        pair = {
                            "anchor": q,
                            "positive": row["text"],
                            "chunk_id": row["id"],
                            "document_id": row["document_id"],
                        }
                        f.write(json.dumps(pair) + "\n")
                        written += 1
                except Exception as e:
                    logger.warning(f"LLM error for chunk {row['id']}: {e}")

    logger.success(f"Generated {written} pairs → {out_path}")


async def main(
    splits_path: Path,
    db_url: str,
    out_path: Path,
    pairs_per_chunk: int,
    max_chunks: int,
) -> None:
    splits = json.loads(splits_path.read_text())
    train_ids = splits["train"]
    logger.info(f"Using {len(train_ids)} train documents")

    pool = await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=3)
    await generate_pairs(pool, train_ids, pairs_per_chunk, max_chunks, out_path)
    await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", required=True)
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--out", default="data/training/synthetic_pairs.jsonl")
    parser.add_argument("--pairs-per-chunk", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=500)
    args = parser.parse_args()
    asyncio.run(
        main(
            Path(args.splits),
            args.db_url,
            Path(args.out),
            args.pairs_per_chunk,
            args.max_chunks,
        )
    )
