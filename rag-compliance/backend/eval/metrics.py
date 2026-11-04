"""
eval/metrics.py — Retrieval and generation metrics.

Metrics implemented:
- Recall@k
- MRR@k
- nDCG@k
- Stale-answer rate
- Refusal precision and recall
"""
from __future__ import annotations

import math
from typing import Any


def recall_at_k(
    retrieved_ids: list[int], relevant_ids: set[int], k: int
) -> float:
    """Fraction of relevant docs found in top-k retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for i in top_k if i in relevant_ids)
    return hits / len(relevant_ids)


def mrr_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Mean Reciprocal Rank — returns 1/rank of first relevant result, or 0."""
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain (binary relevance)."""
    def dcg(ids: list[int], rel: set[int], k: int) -> float:
        return sum(
            (1.0 if ids[i] in rel else 0.0) / math.log2(i + 2)
            for i in range(min(k, len(ids)))
        )

    ideal_ids = list(relevant_ids)[:k]
    idcg = dcg(ideal_ids, relevant_ids, k)
    if idcg == 0:
        return 0.0
    return dcg(retrieved_ids, relevant_ids, k) / idcg


def aggregate_retrieval_metrics(
    results: list[dict[str, Any]],
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """
    Aggregate metrics over a list of per-query result dicts.

    Each result dict:
        {"retrieved_ids": [...], "relevant_ids": [...]}
    """
    if not results:
        return {}

    agg: dict[str, list[float]] = {
        f"recall@{k}": [] for k in k_values
    }
    agg.update({f"mrr@{k}": [] for k in k_values})
    agg.update({f"ndcg@{k}": [] for k in k_values})

    for r in results:
        retrieved = r["retrieved_ids"]
        relevant = set(r["relevant_ids"])
        for k in k_values:
            agg[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            agg[f"mrr@{k}"].append(mrr_at_k(retrieved, relevant, k))
            agg[f"ndcg@{k}"].append(ndcg_at_k(retrieved, relevant, k))

    return {key: round(sum(vals) / len(vals), 4) for key, vals in agg.items()}


def stale_answer_rate(answers: list[dict[str, Any]]) -> float:
    """
    Fraction of answers that cite a superseded document.

    Each answer dict: {"cited_status": list[str]}  where status in ('active','superseded','withdrawn')
    """
    if not answers:
        return 0.0
    stale = sum(
        1 for a in answers
        if any(s != "active" for s in a.get("cited_status", []))
    )
    return round(stale / len(answers), 4)


def refusal_metrics(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Precision and recall for the refusal path.

    Each result dict:
        {"should_refuse": bool, "refused": bool}
    """
    tp = sum(1 for r in results if r["should_refuse"] and r["refused"])
    fp = sum(1 for r in results if not r["should_refuse"] and r["refused"])
    fn = sum(1 for r in results if r["should_refuse"] and not r["refused"])
    tn = sum(1 for r in results if not r["should_refuse"] and not r["refused"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "refusal_precision": round(precision, 4),
        "refusal_recall": round(recall, 4),
        "refusal_f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }
