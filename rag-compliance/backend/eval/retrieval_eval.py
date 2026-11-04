"""
eval/retrieval_eval.py — Run retrieval evaluation against the test set.

Usage:
    python -m eval.retrieval_eval \
        --queries eval/test_queries.jsonl \
        --db-url postgresql://... \
        --out eval/retrieval_results.json

Requires that test queries have been annotated with relevant chunk IDs.
If chunk_ids are not annotated, this script will print a labelling prompt.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import asyncpg
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.core.config import settings
from app.services.retrieval import retrieve
from eval.metrics import aggregate_retrieval_metrics, stale_answer_rate


async def run_eval(
    queries_path: Path,
    db_url: str,
    out_path: Path,
    historical: bool,
    k_values: list[int],
) -> None:
    queries = [json.loads(l) for l in queries_path.read_text().splitlines() if l.strip()]
    pool = await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=5)

    retrieval_results = []
    stale_results = []
    latencies = []

    for q in queries:
        question = q["question"]
        relevant_chunk_ids = q.get("relevant_chunk_ids", [])

        t0 = time.perf_counter()
        chunks, best_sim, retrieval_ms = await retrieve(
            pool, question, historical=historical
        )
        latencies.append(retrieval_ms)

        retrieved_ids = [c["chunk_id"] for c in chunks]

        if relevant_chunk_ids:
            retrieval_results.append(
                {
                    "question": question,
                    "retrieved_ids": retrieved_ids,
                    "relevant_ids": relevant_chunk_ids,
                    "best_sim": best_sim,
                    "retrieval_ms": retrieval_ms,
                }
            )

        # Stale check
        cited_statuses = [c["status"] for c in chunks]
        stale_results.append(
            {"question": question, "cited_status": cited_statuses}
        )

    await pool.close()

    # Compute metrics
    metrics = aggregate_retrieval_metrics(retrieval_results, k_values=k_values)
    metrics["stale_answer_rate"] = stale_answer_rate(stale_results)
    metrics["median_latency_ms"] = sorted(latencies)[len(latencies) // 2]
    metrics["p95_latency_ms"] = sorted(latencies)[int(len(latencies) * 0.95)]
    metrics["n_queries"] = len(queries)
    metrics["n_annotated"] = len(retrieval_results)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))

    logger.info("=== Retrieval Evaluation Results ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    if not retrieval_results:
        logger.warning(
            "No annotated queries found. "
            "Add 'relevant_chunk_ids' to your test_queries.jsonl by "
            "running a retrieval pass and labelling correct chunks."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="eval/test_queries.jsonl")
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--out", default="eval/retrieval_results.json")
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args()
    asyncio.run(
        run_eval(
            Path(args.queries),
            args.db_url,
            Path(args.out),
            args.historical,
            args.k,
        )
    )
