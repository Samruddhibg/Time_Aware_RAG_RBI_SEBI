"""
scripts/run_pipeline.py — End-to-end pipeline orchestrator.

Runs each stage in the correct order with dependency checks and
clear progress reporting. Designed to be run once from scratch to
produce a fully trained, evaluated, and ready-to-serve system.

Stages
------
  1. fetch      — Download RBI + SEBI PDFs from manifests
  2. ingest     — Parse PDFs → chunk → embed → upsert into PostgreSQL
  3. split      — Create document-disjoint train/test split
  4. synthetic  — Generate synthetic (query, passage) pairs via LLM
  5. negatives  — Mine hard negatives for each training pair
  6. finetune   — Fine-tune the embedding model
  7. rebuild    — Re-embed all chunks with the fine-tuned model
  8. annotate   — Auto-annotate test queries with relevant_chunk_ids
  9. eval       — Run retrieval + generation + time-trap evaluation

Usage
-----
    # Full pipeline from scratch
    python -m scripts.run_pipeline --db-url postgresql://raguser:ragpass@localhost:5432/ragdb

    # Skip stages you've already completed
    python -m scripts.run_pipeline --db-url ... --start-from ingest

    # Only run evaluation (assumes ingest is done)
    python -m scripts.run_pipeline --db-url ... --only eval

    # Dry-run (print what would run, don't execute)
    python -m scripts.run_pipeline --db-url ... --dry-run

Available --start-from / --only values:
    fetch, ingest, split, synthetic, negatives, finetune, rebuild, annotate, eval
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent  # backend/
_DATA = _ROOT.parent / "data"         # rag-compliance/data/
_MODELS = _ROOT.parent / "models"     # rag-compliance/models/


# ── Stage runner ──────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path = _ROOT, check: bool = True) -> int:
    """Run a subprocess command; stream output live."""
    display = " ".join(str(c) for c in cmd)
    logger.info(f"$ {display}")
    result = subprocess.run(cmd, cwd=str(cwd))
    if check and result.returncode != 0:
        logger.error(f"Command failed (exit {result.returncode}): {display}")
        sys.exit(result.returncode)
    return result.returncode


def _python(*args: str, cwd: Path = _ROOT, check: bool = True) -> int:
    return _run([sys.executable, *args], cwd=cwd, check=check)


def _banner(text: str) -> None:
    line = "─" * 60
    logger.info(f"\n{line}\n  {text}\n{line}")


# ── Individual stages ─────────────────────────────────────────────────────────

def stage_fetch(args: argparse.Namespace) -> None:
    _banner("STAGE 1 / 9 — Fetching RBI + SEBI PDFs")
    rbi_manifest = _DATA / "raw" / "rbi_manifest.json"
    sebi_manifest = _DATA / "raw" / "sebi_manifest.json"
    rbi_out = _DATA / "raw" / "rbi"
    sebi_out = _DATA / "raw" / "sebi"

    _python("-m", "ingest.fetch_docs",
            "--regulator", "RBI",
            "--manifest", str(rbi_manifest),
            "--out", str(rbi_out),
            check=False)  # non-fatal — some docs may need manual download

    _python("-m", "ingest.fetch_docs",
            "--regulator", "SEBI",
            "--manifest", str(sebi_manifest),
            "--out", str(sebi_out),
            check=False)

    # Count how many PDFs are present
    rbi_pdfs = list(rbi_out.glob("*.pdf"))
    sebi_pdfs = list(sebi_out.glob("*.pdf"))
    logger.info(f"RBI PDFs present : {len(rbi_pdfs)}/10")
    logger.info(f"SEBI PDFs present: {len(sebi_pdfs)}/10")

    if len(rbi_pdfs) + len(sebi_pdfs) == 0:
        logger.error(
            "No PDFs were downloaded. "
            "Check the manual download guide printed above, place PDFs in "
            f"{rbi_out}/ and {sebi_out}/, then re-run with --start-from ingest."
        )
        sys.exit(1)
    elif len(rbi_pdfs) + len(sebi_pdfs) < 20:
        logger.warning(
            f"Only {len(rbi_pdfs) + len(sebi_pdfs)}/20 PDFs present. "
            "Pipeline will continue with available documents. "
            "More documents → better retrieval and training quality."
        )


def stage_ingest(args: argparse.Namespace) -> None:
    _banner("STAGE 2 / 9 — Ingesting PDFs → PostgreSQL")

    rbi_dir = _DATA / "raw" / "rbi"
    sebi_dir = _DATA / "raw" / "sebi"
    rebuild_flag = ["--rebuild"] if getattr(args, "rebuild", False) else []

    if list(rbi_dir.glob("*.pdf")):
        _python("-m", "ingest.build_index",
                "--dir", str(rbi_dir),
                "--regulator", "RBI",
                "--manifest-out", str(_DATA / "raw" / "rbi_manifest.json"),
                *rebuild_flag)
    else:
        logger.warning(f"No RBI PDFs found in {rbi_dir} — skipping RBI ingest")

    if list(sebi_dir.glob("*.pdf")):
        _python("-m", "ingest.build_index",
                "--dir", str(sebi_dir),
                "--regulator", "SEBI",
                "--manifest-out", str(_DATA / "raw" / "sebi_manifest.json"),
                *rebuild_flag)
    else:
        logger.warning(f"No SEBI PDFs found in {sebi_dir} — skipping SEBI ingest")


def stage_split(args: argparse.Namespace) -> None:
    _banner("STAGE 3 / 9 — Creating train/test split")
    splits_dir = _DATA / "splits"
    _python("-m", "training.split_by_doc",
            "--db-url", args.db_url,
            "--out", str(splits_dir),
            "--test-fraction", str(args.test_fraction),
            "--seed", "42")


def stage_synthetic(args: argparse.Namespace) -> None:
    _banner("STAGE 4 / 9 — Generating synthetic training pairs")
    splits_file = _DATA / "splits" / "doc_splits.json"
    if not splits_file.exists():
        logger.error(f"Splits file not found: {splits_file}. Run stage 'split' first.")
        sys.exit(1)

    out_path = _DATA / "training" / "synthetic_pairs.jsonl"
    _python("-m", "training.gen_synthetic",
            "--splits", str(splits_file),
            "--db-url", args.db_url,
            "--out", str(out_path),
            "--pairs-per-chunk", str(args.pairs_per_chunk),
            "--max-chunks", str(args.max_chunks))

    count = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    logger.info(f"Synthetic pairs generated: {count}")
    if count < 100:
        logger.warning(
            f"Only {count} pairs generated. For quality fine-tuning, aim for ≥500. "
            "Increase --max-chunks or --pairs-per-chunk."
        )


def stage_negatives(args: argparse.Namespace) -> None:
    _banner("STAGE 5 / 9 — Mining hard negatives")
    pairs_path = _DATA / "training" / "synthetic_pairs.jsonl"
    if not pairs_path.exists():
        logger.error(f"Synthetic pairs not found: {pairs_path}. Run stage 'synthetic' first.")
        sys.exit(1)

    out_path = _DATA / "training" / "pairs_with_negatives.jsonl"
    _python("-m", "training.mine_negatives",
            "--pairs", str(pairs_path),
            "--db-url", args.db_url,
            "--out", str(out_path),
            "--lo", "10",
            "--hi", "40",
            "--batch-size", "64")

    count = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    logger.info(f"Training triplets with hard negatives: {count}")


def stage_finetune(args: argparse.Namespace) -> None:
    _banner("STAGE 6 / 9 — Fine-tuning embedding model")
    triplets_path = _DATA / "training" / "pairs_with_negatives.jsonl"
    if not triplets_path.exists():
        logger.error(f"Training triplets not found: {triplets_path}. Run stage 'negatives' first.")
        sys.exit(1)

    count = sum(1 for _ in open(triplets_path))
    if count < 50:
        logger.error(
            f"Only {count} training triplets — too few to fine-tune reliably. "
            "Need at least 50; ideally 500+. Check your synthetic pair generation."
        )
        sys.exit(1)

    output_dir = _MODELS / "bge-small-ft"
    _python("-m", "training.finetune",
            "--data", str(triplets_path),
            "--base-model", args.base_model,
            "--output", str(output_dir),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr))

    logger.success(f"Fine-tuned model saved to: {output_dir}")
    logger.info(
        "Next: update EMBED_MODEL in your .env to point to this directory, "
        "then the 'rebuild' stage will re-embed all chunks."
    )


def stage_rebuild(args: argparse.Namespace) -> None:
    _banner("STAGE 7 / 9 — Re-embedding with fine-tuned model")
    ft_model = _MODELS / "bge-small-ft"
    if not ft_model.exists():
        logger.warning(
            f"Fine-tuned model not found at {ft_model}. "
            "Using base model for rebuild. Run 'finetune' stage first for best results."
        )

    rbi_dir = _DATA / "raw" / "rbi"
    sebi_dir = _DATA / "raw" / "sebi"

    if list(rbi_dir.glob("*.pdf")):
        _python("-m", "ingest.build_index",
                "--dir", str(rbi_dir),
                "--regulator", "RBI",
                "--rebuild")

    if list(sebi_dir.glob("*.pdf")):
        _python("-m", "ingest.build_index",
                "--dir", str(sebi_dir),
                "--regulator", "SEBI",
                "--rebuild")


def stage_annotate(args: argparse.Namespace) -> None:
    _banner("STAGE 8 / 9 — Annotating test queries with relevant chunk IDs")
    queries_path = _ROOT / "eval" / "test_queries.jsonl"
    out_path = _ROOT / "eval" / "test_queries_annotated.jsonl"

    _python("-m", "scripts.annotate_chunks",
            "--queries", str(queries_path),
            "--db-url", args.db_url,
            "--mode", args.annotate_mode,
            "--out", str(out_path))

    # Copy annotated file over the original so tests pick it up
    if out_path.exists():
        import shutil
        shutil.copy(out_path, queries_path)
        logger.success(f"Updated {queries_path} with relevant_chunk_ids")


def stage_eval(args: argparse.Namespace) -> None:
    _banner("STAGE 9 / 9 — Evaluation")
    results_dir = _ROOT / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    db_flag = ["--db-url", args.db_url]
    api_flag = ["--api-url", args.api_url]

    logger.info("Running retrieval evaluation...")
    _python("-m", "eval.retrieval_eval",
            "--queries", str(_ROOT / "eval" / "test_queries.jsonl"),
            *db_flag,
            "--out", str(results_dir / "retrieval_results.json"),
            check=False)

    logger.info("Running generation evaluation...")
    _python("-m", "eval.generation_eval",
            "--queries", str(_ROOT / "eval" / "test_queries.jsonl"),
            "--unanswerable", str(_ROOT / "eval" / "unanswerable.jsonl"),
            *api_flag,
            "--out", str(results_dir / "generation_results.json"),
            check=False)

    logger.info("Running time-trap evaluation...")
    _python("-m", "eval.time_traps_eval",
            "--traps", str(_ROOT / "eval" / "time_traps.jsonl"),
            *api_flag,
            "--out", str(results_dir / "time_traps_results.json"),
            check=False)

    # Print a consolidated summary
    _print_eval_summary(results_dir)


def _print_eval_summary(results_dir: Path) -> None:
    import json

    summary_lines = ["\n" + "=" * 60, "EVALUATION SUMMARY", "=" * 60]

    retrieval_path = results_dir / "retrieval_results.json"
    if retrieval_path.exists():
        r = json.loads(retrieval_path.read_text())
        summary_lines += [
            "RETRIEVAL",
            f"  Recall@5          : {r.get('recall@5', 'N/A')}",
            f"  MRR@10            : {r.get('mrr@10', 'N/A')}",
            f"  nDCG@10           : {r.get('ndcg@10', 'N/A')}",
            f"  Stale answer rate : {r.get('stale_answer_rate', 'N/A')}",
            f"  Median latency ms : {r.get('median_latency_ms', 'N/A')}",
            f"  p95 latency ms    : {r.get('p95_latency_ms', 'N/A')}",
        ]

    gen_path = results_dir / "generation_results.json"
    if gen_path.exists():
        g = json.loads(gen_path.read_text())
        ans = g.get("answerable", {})
        unans = g.get("unanswerable", {})
        summary_lines += [
            "",
            "GENERATION",
            f"  Answer correctness: {ans.get('answer_correctness', 'N/A')}",
            f"  Citation rate     : {ans.get('citation_rate', 'N/A')}",
            f"  Refusal precision : {unans.get('refusal_precision', 'N/A')}",
            f"  Refusal recall    : {unans.get('refusal_recall', 'N/A')}",
            f"  Refusal F1        : {unans.get('refusal_f1', 'N/A')}",
        ]

    traps_path = results_dir / "time_traps_results.json"
    if traps_path.exists():
        t = json.loads(traps_path.read_text())["summary"]
        summary_lines += [
            "",
            "TIME-AWARENESS",
            f"  Correct answer rate: {t.get('correct_rate', 'N/A')}",
            f"  Stale answer rate  : {t.get('stale_rate', 'N/A')}  (target: 0.0)",
            f"  Active sources only: {t.get('active_source_rate', 'N/A')}  (target: 1.0)",
        ]

    summary_lines.append("=" * 60)
    for line in summary_lines:
        logger.info(line)

    # Write consolidated baseline
    import json
    baseline = {}
    if retrieval_path.exists():
        baseline.update(json.loads(retrieval_path.read_text()))
    if gen_path.exists():
        g = json.loads(gen_path.read_text())
        baseline.update(g.get("answerable", {}))
        baseline.update(g.get("unanswerable", {}))
    if traps_path.exists():
        baseline.update(json.loads(traps_path.read_text())["summary"])

    baseline_path = _ROOT / "eval" / "baseline.json"
    baseline_path.write_text(json.dumps(baseline, indent=2, default=str))
    logger.info(f"Baseline metrics saved to: {baseline_path}")


# ── Stage registry ────────────────────────────────────────────────────────────

STAGES: list[tuple[str, Callable]] = [
    ("fetch",     stage_fetch),
    ("ingest",    stage_ingest),
    ("split",     stage_split),
    ("synthetic", stage_synthetic),
    ("negatives", stage_negatives),
    ("finetune",  stage_finetune),
    ("rebuild",   stage_rebuild),
    ("annotate",  stage_annotate),
    ("eval",      stage_eval),
]

STAGE_NAMES = [name for name, _ in STAGES]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline for Time-Aware RAG (RBI/SEBI).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--db-url",
        default="postgresql://raguser:ragpass@localhost:5432/ragdb",
        help="asyncpg-compatible PostgreSQL connection string",
    )

    # Stage selection
    parser.add_argument(
        "--start-from",
        choices=STAGE_NAMES,
        default="fetch",
        metavar="STAGE",
        help=f"Skip all stages before this one. Choices: {', '.join(STAGE_NAMES)}",
    )
    parser.add_argument(
        "--only",
        choices=STAGE_NAMES,
        metavar="STAGE",
        help="Run only this single stage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which stages would run without executing them.",
    )

    # Stage-specific options
    parser.add_argument("--rebuild", action="store_true",
                        help="Force re-ingest of already-indexed PDFs (stage: ingest/rebuild)")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="Fraction of docs held out for testing (stage: split)")
    parser.add_argument("--pairs-per-chunk", type=int, default=2,
                        help="Synthetic queries generated per chunk (stage: synthetic)")
    parser.add_argument("--max-chunks", type=int, default=500,
                        help="Max chunks to generate synthetic pairs for (stage: synthetic)")
    parser.add_argument("--base-model", default="BAAI/bge-small-en-v1.5",
                        help="HuggingFace model ID for fine-tuning base (stage: finetune)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs (stage: finetune)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size (stage: finetune)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (stage: finetune)")
    parser.add_argument("--annotate-mode", choices=["auto", "llm", "manual"], default="auto",
                        help="How to annotate test queries with relevant chunk IDs (stage: annotate)")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="Running API URL for generation eval (stage: eval)")

    args = parser.parse_args()

    # Determine which stages to run
    if args.only:
        stages_to_run = [(n, fn) for n, fn in STAGES if n == args.only]
    else:
        start_idx = STAGE_NAMES.index(args.start_from)
        stages_to_run = STAGES[start_idx:]

    if args.dry_run:
        logger.info("DRY RUN — stages that would execute:")
        for name, _ in stages_to_run:
            logger.info(f"  • {name}")
        return

    t_total = time.perf_counter()
    logger.info(f"Running {len(stages_to_run)} stage(s): "
                f"{', '.join(n for n, _ in stages_to_run)}")

    for name, fn in stages_to_run:
        t0 = time.perf_counter()
        fn(args)
        elapsed = time.perf_counter() - t0
        logger.info(f"Stage '{name}' completed in {elapsed:.1f}s")

    total_elapsed = time.perf_counter() - t_total
    logger.success(
        f"\nAll stages complete in {total_elapsed/60:.1f} min. "
        "Your system is trained and evaluated."
    )


if __name__ == "__main__":
    main()
