"""
eval/generation_eval.py — Evaluate answer quality: faithfulness, correctness, refusal.

Usage:
    python -m eval.generation_eval \
        --queries eval/test_queries.jsonl \
        --unanswerable eval/unanswerable.jsonl \
        --api-url http://localhost:8000 \
        --out eval/generation_results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx
from loguru import logger

from eval.metrics import refusal_metrics


async def ask_api(
    client: httpx.AsyncClient,
    api_url: str,
    question: str,
    historical: bool = False,
) -> dict:
    """Call /api/ask and collect full SSE response."""
    sources = []
    tokens = []
    query_id = None
    refused = False

    async with client.stream(
        "POST",
        f"{api_url}/api/ask",
        json={"question": question, "historical": historical},
        timeout=60,
    ) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                break
            try:
                event = json.loads(raw)
                if event["type"] == "sources":
                    sources = event["data"]
                elif event["type"] == "token":
                    tokens.append(event["data"])
                elif event["type"] == "done":
                    query_id = event["data"].get("query_id")
                    refused = event["data"].get("refused", False)
            except (json.JSONDecodeError, KeyError):
                continue

    return {
        "answer": "".join(tokens),
        "sources": sources,
        "query_id": query_id,
        "refused": refused,
    }


def check_citation_quality(answer: str, sources: list[dict]) -> dict:
    """Check if answer contains citation numbers matching sources."""
    citations_in_answer = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))
    valid_refs = set(range(1, len(sources) + 1))
    cited = citations_in_answer & valid_refs
    return {
        "has_citations": len(citations_in_answer) > 0,
        "all_valid": citations_in_answer.issubset(valid_refs),
        "citation_count": len(cited),
        "dangling_citations": len(citations_in_answer - valid_refs),
    }


def check_answer_contains(answer: str, keywords: list[str]) -> bool:
    """Check if answer contains any of the expected keywords."""
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)


async def evaluate_answerable(
    client: httpx.AsyncClient,
    api_url: str,
    queries: list[dict],
) -> dict:
    results = []
    for q in queries:
        result = await ask_api(client, api_url, q["question"])
        answer = result["answer"]
        keywords = q.get("answer_contains", [])
        contains_answer = check_answer_contains(answer, keywords) if keywords else None
        citation_quality = check_citation_quality(answer, result["sources"])

        results.append(
            {
                "question": q["question"],
                "refused": result["refused"],
                "answer_preview": answer[:200],
                "contains_expected": contains_answer,
                "citation_quality": citation_quality,
                "n_sources": len(result["sources"]),
            }
        )

    n = len(results)
    answer_quality = [r for r in results if r["contains_expected"] is not None]
    return {
        "n_answerable": n,
        "refusal_rate": sum(1 for r in results if r["refused"]) / n if n else 0,
        "answer_correctness": (
            sum(1 for r in answer_quality if r["contains_expected"]) / len(answer_quality)
            if answer_quality else None
        ),
        "citation_rate": sum(1 for r in results if r["citation_quality"]["has_citations"]) / n if n else 0,
        "per_query": results,
    }


async def evaluate_unanswerable(
    client: httpx.AsyncClient,
    api_url: str,
    queries: list[dict],
) -> dict:
    refusal_data = []
    for q in queries:
        result = await ask_api(client, api_url, q["question"])
        refusal_data.append(
            {
                "question": q["question"],
                "should_refuse": q.get("expected_refusal", True),
                "refused": result["refused"],
            }
        )
    metrics = refusal_metrics(refusal_data)
    return {"n_unanswerable": len(queries), **metrics, "per_query": refusal_data}


async def main(
    queries_path: Path,
    unanswerable_path: Path,
    api_url: str,
    out_path: Path,
) -> None:
    queries = [json.loads(l) for l in queries_path.read_text().splitlines() if l.strip()]
    unanswerable = [json.loads(l) for l in unanswerable_path.read_text().splitlines() if l.strip()]

    async with httpx.AsyncClient() as client:
        logger.info(f"Evaluating {len(queries)} answerable questions...")
        answerable_results = await evaluate_answerable(client, api_url, queries)

        logger.info(f"Evaluating {len(unanswerable)} unanswerable questions...")
        refusal_results = await evaluate_unanswerable(client, api_url, unanswerable)

    out = {
        "answerable": answerable_results,
        "unanswerable": refusal_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    logger.info("=== Generation Evaluation ===")
    logger.info(f"  Answerable refusal rate: {answerable_results['refusal_rate']:.1%}")
    logger.info(f"  Answer correctness: {answerable_results['answer_correctness']}")
    logger.info(f"  Citation rate: {answerable_results['citation_rate']:.1%}")
    logger.info(f"  Refusal precision: {refusal_results['refusal_precision']:.1%}")
    logger.info(f"  Refusal recall: {refusal_results['refusal_recall']:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="eval/test_queries.jsonl")
    parser.add_argument("--unanswerable", default="eval/unanswerable.jsonl")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--out", default="eval/generation_results.json")
    args = parser.parse_args()
    asyncio.run(main(Path(args.queries), Path(args.unanswerable), args.api_url, Path(args.out)))
