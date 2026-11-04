"""
scripts/annotate_chunks.py — Auto-annotate test_queries.jsonl with relevant_chunk_ids.

For each query, runs retrieval and asks you (or an LLM) to confirm which
retrieved chunks are genuinely relevant. Writes the annotated file back.

Modes:
  --mode auto   : top-1 chunk from retrieval is assumed relevant (fast bootstrap)
  --mode llm    : use the configured LLM to judge relevance (more accurate)
  --mode manual : print each chunk and prompt you to confirm in the terminal

Usage:
    # Fast bootstrap (assume top hit is correct)
    python -m scripts.annotate_chunks \
        --queries eval/test_queries.jsonl \
        --db-url postgresql://raguser:ragpass@localhost:5432/ragdb \
        --mode auto \
        --out eval/test_queries_annotated.jsonl

    # LLM-judged (best quality)
    python -m scripts.annotate_chunks \
        --queries eval/test_queries.jsonl \
        --db-url postgresql://raguser:ragpass@localhost:5432/ragdb \
        --mode llm \
        --out eval/test_queries_annotated.jsonl
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.core.config import settings
from app.services.retrieval import retrieve

_JUDGE_SYSTEM = (
    "You are a relevance judge for a regulatory information retrieval system. "
    "Given a compliance question and a text passage, respond with exactly one word: "
    "'relevant' if the passage contains information that helps answer the question, "
    "or 'irrelevant' if it does not."
)

_JUDGE_USER = """Question: {question}

Passage:
{text}

Is this passage relevant to the question? Answer with one word only."""


async def _llm_judge(
    client: httpx.AsyncClient,
    question: str,
    text: str,
) -> bool:
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": _JUDGE_USER.format(
                question=question, text=text[:800]
            )},
        ],
        "temperature": 0.0,
        "max_tokens": 10,
        "stream": False,
    }
    try:
        r = await client.post(
            f"{settings.llm_base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=20,
        )
        r.raise_for_status()
        verdict = r.json()["choices"][0]["message"]["content"].strip().lower()
        return "relevant" in verdict
    except Exception as e:
        logger.warning(f"LLM judge error: {e} — defaulting to relevant")
        return True


def _manual_judge(question: str, chunk: dict) -> bool:
    print(f"\n{'=' * 60}")
    print(f"QUESTION: {question}")
    print(f"CHUNK ID: {chunk['chunk_id']}")
    print(f"CIRCULAR: {chunk.get('circular_no', 'N/A')} | {chunk.get('issue_date', 'N/A')}")
    print(f"SECTION : {chunk.get('section', 'N/A')}")
    print(f"TEXT    :\n{chunk['text'][:600]}")
    print(f"{'=' * 60}")
    while True:
        ans = input("Relevant? [y/n/s(kip)]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        if ans in ("s", "skip"):
            raise KeyboardInterrupt("Skipped")


async def annotate(
    queries_path: Path,
    db_url: str,
    out_path: Path,
    mode: str,
    top_k: int,
) -> None:
    queries = [
        json.loads(l) for l in queries_path.read_text().splitlines() if l.strip()
    ]
    pool = await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=3)

    annotated: list[dict] = []

    async with httpx.AsyncClient() as http_client:
        for i, q in enumerate(queries):
            question = q["question"]
            logger.info(f"[{i+1}/{len(queries)}] {question[:70]}")

            chunks, best_sim, _ = await retrieve(
                pool, question, historical=False,
                top_k_dense=top_k, top_k_rerank=top_k,
            )

            if not chunks:
                logger.warning(f"No chunks retrieved — query will be unannotated")
                annotated.append(q)
                continue

            relevant_ids: list[int] = []

            for chunk in chunks:
                try:
                    if mode == "auto":
                        # Top-1 is considered relevant; rest are not
                        if chunk == chunks[0]:
                            relevant_ids.append(chunk["chunk_id"])
                            break
                    elif mode == "llm":
                        is_rel = await _llm_judge(http_client, question, chunk["text"])
                        if is_rel:
                            relevant_ids.append(chunk["chunk_id"])
                    elif mode == "manual":
                        is_rel = _manual_judge(question, chunk)
                        if is_rel:
                            relevant_ids.append(chunk["chunk_id"])
                except KeyboardInterrupt:
                    break

            updated_q = {**q, "relevant_chunk_ids": relevant_ids}
            annotated.append(updated_q)
            logger.info(f"  → {len(relevant_ids)} relevant chunk(s) found")

    await pool.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for q in annotated:
            f.write(json.dumps(q) + "\n")

    n_annotated = sum(1 for q in annotated if q.get("relevant_chunk_ids"))
    logger.success(
        f"Annotated {n_annotated}/{len(annotated)} queries → {out_path}\n"
        f"Now update test_queries.jsonl with this file to enable the CI gate."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="eval/test_queries.jsonl")
    parser.add_argument("--db-url", required=True)
    parser.add_argument(
        "--mode",
        choices=["auto", "llm", "manual"],
        default="auto",
        help=(
            "auto=top-1 chunk assumed relevant; "
            "llm=LLM judges each chunk; "
            "manual=you confirm in terminal"
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", default="eval/test_queries_annotated.jsonl")
    args = parser.parse_args()
    asyncio.run(
        annotate(
            Path(args.queries),
            args.db_url,
            Path(args.out),
            args.mode,
            args.top_k,
        )
    )
