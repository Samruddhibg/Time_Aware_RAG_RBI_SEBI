"""
eval/time_traps_eval.py — Evaluate time-trap queries: tests the system correctly
returns the CURRENT regulation, not an outdated superseded one.

A "time trap" is a query where an old circular contains a plausible-but-wrong
answer (e.g., KYC limit was Rs 10,000 before 2016, now Rs 50,000).

Usage:
    python -m eval.time_traps_eval \
        --traps eval/time_traps.jsonl \
        --api-url http://localhost:8000 \
        --out eval/time_traps_results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx
from loguru import logger


async def ask_api(
    client: httpx.AsyncClient,
    api_url: str,
    question: str,
    historical: bool = False,
) -> dict:
    """Call /api/ask and collect full SSE response."""
    sources: list[dict] = []
    tokens: list[str] = []
    refused = False
    query_id = None

    async with client.stream(
        "POST",
        f"{api_url}/api/ask",
        json={"question": question, "historical": historical},
        timeout=60,
    ) as resp:
        resp.raise_for_status()
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
                    refused = event["data"].get("refused", False)
                    query_id = event["data"].get("query_id")
            except (json.JSONDecodeError, KeyError):
                continue

    return {
        "answer": "".join(tokens),
        "sources": sources,
        "refused": refused,
        "query_id": query_id,
    }


def _check_stale_answer(answer: str, trap: dict) -> bool:
    """Return True if the answer contains the known-stale (wrong) value."""
    stale = trap.get("stale_answer", "")
    if not stale:
        return False
    return stale.lower() in answer.lower()


def _check_correct_answer(answer: str, trap: dict) -> bool | None:
    """Return True if the answer contains the correct value, None if no ground truth."""
    correct = trap.get("correct_answer", "")
    if not correct:
        return None
    return correct.lower() in answer.lower()


def _check_active_sources_only(sources: list[dict]) -> bool:
    """Return True if ALL cited sources are active (not superseded/withdrawn)."""
    if not sources:
        return False
    return all(s.get("status", "active") == "active" for s in sources)


def _check_historical_query(trap: dict, result: dict) -> dict:
    """For historical queries, the system should answer with historical=True context."""
    # Historical queries are answered correctly if they don't refuse and return an answer
    return {
        "answered": not result["refused"],
        "has_answer": bool(result["answer"].strip()),
    }


async def evaluate_time_traps(
    traps_path: Path,
    api_url: str,
    out_path: Path,
) -> None:
    traps = [json.loads(l) for l in traps_path.read_text().splitlines() if l.strip()]
    logger.info(f"Evaluating {len(traps)} time-trap queries against {api_url}")

    per_query = []
    metrics = {
        "total": 0,
        "stale_answer_count": 0,        # answered with the wrong (old) value
        "correct_answer_count": 0,       # answered with the right (new) value
        "active_sources_only_count": 0,  # all cited sources are active
        "refused_count": 0,
        "by_trap_type": {},
    }

    async with httpx.AsyncClient() as client:
        for i, trap in enumerate(traps):
            question = trap["question"]
            trap_type = trap.get("trap_type", "unknown")
            is_historical = trap_type == "historical_query"

            logger.info(f"[{i+1}/{len(traps)}] [{trap_type}] {question[:70]}")

            try:
                result = await ask_api(client, api_url, question, historical=is_historical)
            except Exception as e:
                logger.error(f"  API error: {e}")
                per_query.append({
                    "question": question,
                    "trap_type": trap_type,
                    "error": str(e),
                })
                continue

            answer = result["answer"]
            sources = result["sources"]
            refused = result["refused"]

            is_stale = _check_stale_answer(answer, trap)
            is_correct = _check_correct_answer(answer, trap)
            active_only = _check_active_sources_only(sources)

            per_query.append({
                "question": question,
                "trap_type": trap_type,
                "is_historical": is_historical,
                "refused": refused,
                "stale_answer_given": is_stale,
                "correct_answer_given": is_correct,
                "active_sources_only": active_only,
                "n_sources": len(sources),
                "source_statuses": [s.get("status") for s in sources],
                "answer_preview": answer[:300],
                "correct_answer": trap.get("correct_answer"),
                "stale_answer": trap.get("stale_answer"),
                "correct_circular": trap.get("correct_answer_circular"),
            })

            metrics["total"] += 1
            if is_stale:
                metrics["stale_answer_count"] += 1
            if is_correct:
                metrics["correct_answer_count"] += 1
            if active_only:
                metrics["active_sources_only_count"] += 1
            if refused:
                metrics["refused_count"] += 1

            # Per-trap-type breakdown
            if trap_type not in metrics["by_trap_type"]:
                metrics["by_trap_type"][trap_type] = {
                    "total": 0, "stale": 0, "correct": 0, "active_only": 0
                }
            metrics["by_trap_type"][trap_type]["total"] += 1
            if is_stale:
                metrics["by_trap_type"][trap_type]["stale"] += 1
            if is_correct:
                metrics["by_trap_type"][trap_type]["correct"] += 1
            if active_only:
                metrics["by_trap_type"][trap_type]["active_only"] += 1

    # Aggregate
    n = metrics["total"]
    if n > 0:
        metrics["stale_rate"] = round(metrics["stale_answer_count"] / n, 4)
        metrics["correct_rate"] = round(metrics["correct_answer_count"] / n, 4)
        metrics["active_source_rate"] = round(metrics["active_sources_only_count"] / n, 4)
        metrics["refusal_rate"] = round(metrics["refused_count"] / n, 4)
    else:
        metrics["stale_rate"] = 0.0
        metrics["correct_rate"] = 0.0
        metrics["active_source_rate"] = 0.0
        metrics["refusal_rate"] = 0.0

    output = {"summary": metrics, "per_query": per_query}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    logger.info("=" * 50)
    logger.info("TIME TRAP EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"  Total queries      : {n}")
    logger.info(f"  Stale answer rate  : {metrics['stale_rate']:.1%}  (lower is better)")
    logger.info(f"  Correct answer rate: {metrics['correct_rate']:.1%}  (higher is better)")
    logger.info(f"  Active sources only: {metrics['active_source_rate']:.1%}  (higher is better)")
    logger.info(f"  Refusal rate       : {metrics['refusal_rate']:.1%}")
    logger.info("")
    logger.info("By trap type:")
    for ttype, stats in metrics["by_trap_type"].items():
        t = stats["total"]
        logger.info(
            f"  {ttype:30s} | {t} queries | "
            f"stale={stats['stale']}/{t} | "
            f"correct={stats['correct']}/{t} | "
            f"active_src={stats['active_only']}/{t}"
        )
    logger.info(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traps", default="eval/time_traps.jsonl")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--out", default="eval/time_traps_results.json")
    args = parser.parse_args()
    asyncio.run(evaluate_time_traps(Path(args.traps), args.api_url, Path(args.out)))
