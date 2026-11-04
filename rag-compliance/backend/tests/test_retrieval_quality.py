"""
tests/test_retrieval_quality.py — CI gate: fails if retrieval quality drops below baseline.

Run with: pytest tests/test_retrieval_quality.py -v
"""
import json
import asyncio
import pytest
import asyncpg
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.services.retrieval import retrieve
from eval.metrics import aggregate_retrieval_metrics, stale_answer_rate

# ── Thresholds (set from your measured baseline) ──────────────────────────────
RECALL_5_MIN = 0.40   # Update after measuring baseline
MRR_10_MIN = 0.30     # Update after measuring baseline
STALE_RATE_MAX = 0.10  # Max 10% stale answers in active mode

# ── Test data (subset of eval queries with annotated chunk IDs) ───────────────
_QUERIES_PATH = Path(__file__).parent.parent / "eval" / "test_queries.jsonl"


def _load_annotated_queries():
    if not _QUERIES_PATH.exists():
        return []
    queries = [json.loads(l) for l in _QUERIES_PATH.read_text().splitlines() if l.strip()]
    return [q for q in queries if q.get("relevant_chunk_ids")]


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def pool():
    p = await asyncpg.create_pool(dsn=settings.database_url, min_size=1, max_size=3)
    yield p
    await p.close()


@pytest.mark.asyncio
async def test_recall_at_5(pool):
    """CI gate: Recall@5 must meet minimum threshold."""
    queries = _load_annotated_queries()
    if not queries:
        pytest.skip("No annotated queries (add relevant_chunk_ids to test_queries.jsonl)")

    results = []
    for q in queries:
        chunks, best_sim, _ = await retrieve(pool, q["question"], historical=False)
        results.append(
            {
                "retrieved_ids": [c["chunk_id"] for c in chunks],
                "relevant_ids": q["relevant_chunk_ids"],
            }
        )

    metrics = aggregate_retrieval_metrics(results, k_values=[5, 10])
    recall5 = metrics.get("recall@5", 0.0)
    assert recall5 >= RECALL_5_MIN, (
        f"Recall@5 = {recall5:.4f} < threshold {RECALL_5_MIN}. "
        f"Retrieval quality regression detected!"
    )


@pytest.mark.asyncio
async def test_mrr_at_10(pool):
    """CI gate: MRR@10 must meet minimum threshold."""
    queries = _load_annotated_queries()
    if not queries:
        pytest.skip("No annotated queries")

    results = []
    for q in queries:
        chunks, _, _ = await retrieve(pool, q["question"], historical=False)
        results.append(
            {
                "retrieved_ids": [c["chunk_id"] for c in chunks],
                "relevant_ids": q["relevant_chunk_ids"],
            }
        )

    metrics = aggregate_retrieval_metrics(results, k_values=[10])
    mrr = metrics.get("mrr@10", 0.0)
    assert mrr >= MRR_10_MIN, (
        f"MRR@10 = {mrr:.4f} < threshold {MRR_10_MIN}. Regression detected!"
    )


@pytest.mark.asyncio
async def test_stale_answer_rate(pool):
    """CI gate: stale-answer rate in active mode must be below threshold."""
    # Use all queries (no annotation needed)
    if not _QUERIES_PATH.exists():
        pytest.skip("No query file")

    queries = [json.loads(l) for l in _QUERIES_PATH.read_text().splitlines() if l.strip()]
    stale_results = []
    for q in queries[:10]:  # quick check on first 10
        chunks, _, _ = await retrieve(pool, q["question"], historical=False)
        stale_results.append({"cited_status": [c["status"] for c in chunks]})

    rate = stale_answer_rate(stale_results)
    assert rate <= STALE_RATE_MAX, (
        f"Stale-answer rate = {rate:.1%} > max {STALE_RATE_MAX:.1%}. "
        "Time-aware filter not working!"
    )
