"""
ingest/build_index.py — End-to-end ingest: PDF → parse → chunk → embed → upsert DB.

Usage:
    python -m ingest.build_index --dir data/raw/rbi --regulator RBI
    python -m ingest.build_index --dir data/raw/sebi --regulator SEBI --rebuild
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import asyncpg
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.services.embedder import embedder
from ingest.parse_pdf import parse_pdf, pages_to_document_text
from ingest.chunk import chunk_document
from ingest.extract_metadata import extract_metadata

_DOC_UPSERT = """
INSERT INTO documents
    (document_id, regulator, circular_no, title, doc_type, issue_date,
     status, source_url, content_hash, parse_ok, meta_confidence)
VALUES ($1,$2,$3,$4,$5,$6,'active',$7,$8,$9,$10)
ON CONFLICT (document_id) DO UPDATE
    SET circular_no=EXCLUDED.circular_no,
        title=EXCLUDED.title,
        issue_date=EXCLUDED.issue_date,
        content_hash=EXCLUDED.content_hash,
        meta_confidence=EXCLUDED.meta_confidence,
        retrieved_at=now()
"""

_CHUNK_UPSERT = """
INSERT INTO chunks
    (document_id, chunk_index, section, text, token_count, embed_model, embedding)
VALUES ($1,$2,$3,$4,$5,$6,$7::vector)
ON CONFLICT (document_id, chunk_index) DO UPDATE
    SET section=EXCLUDED.section,
        text=EXCLUDED.text,
        token_count=EXCLUDED.token_count,
        embed_model=EXCLUDED.embed_model,
        embedding=EXCLUDED.embedding
"""

_SUPERSESSION_UPSERT = """
INSERT INTO supersessions (new_doc_id, old_doc_id, relation, evidence_text, confidence)
VALUES ($1, $2, $3, $4, 0.7)
ON CONFLICT (new_doc_id, old_doc_id, relation) DO NOTHING
"""

_FIND_DOC_BY_CIRCULAR = """
SELECT document_id FROM documents WHERE circular_no ILIKE $1 LIMIT 1
"""


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


async def ingest_file(
    pool: asyncpg.Pool,
    pdf_path: Path,
    regulator: str,
    force: bool = False,
    embed_model: str = settings.embed_model,
) -> dict:
    """Ingest a single PDF. Returns a summary dict."""
    result = {"path": str(pdf_path), "ok": False, "chunks": 0, "doc_id": None}

    # 1. Hash check
    content_hash = _file_hash(pdf_path)
    async with pool.acquire() as conn:
        existing_hash = await conn.fetchval(
            "SELECT content_hash FROM documents WHERE content_hash = $1", content_hash
        )
    if existing_hash and not force:
        logger.debug(f"Skipping unchanged: {pdf_path.name}")
        result["ok"] = True
        result["skipped"] = True
        return result

    # 2. Parse
    pages = parse_pdf(pdf_path)
    text = pages_to_document_text(pages)
    parse_ok = any(p["ok"] for p in pages)
    if not text:
        logger.error(f"No text extracted from {pdf_path.name}")
        result["error"] = "no text"
        return result

    # 3. Extract metadata
    meta = extract_metadata(text, pdf_path, regulator)
    doc_id = meta["document_id"]
    result["doc_id"] = doc_id

    # 4. Upsert document
    async with pool.acquire() as conn:
        await conn.execute(
            _DOC_UPSERT,
            doc_id,
            regulator,
            meta.get("circular_no"),
            meta.get("title"),
            meta.get("doc_type"),
            meta.get("issue_date"),
            meta.get("source_url"),
            content_hash,
            parse_ok,
            meta.get("meta_confidence", 0.5),
        )

    # 5. Chunk
    chunks = chunk_document(text, doc_id)
    if not chunks:
        logger.warning(f"No chunks produced for {pdf_path.name}")
        return result

    # 6. Embed in batch
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode_passages(texts)

    # 7. Upsert chunks
    async with pool.acquire() as conn:
        for c, emb in zip(chunks, embeddings):
            vec_str = "[" + ",".join(str(v) for v in emb.tolist()) + "]"
            await conn.execute(
                _CHUNK_UPSERT,
                doc_id,
                c["chunk_index"],
                c.get("section"),
                c["text"],
                c.get("token_count", 0),
                embed_model,
                vec_str,
            )

    # 8. Upsert supersessions
    async with pool.acquire() as conn:
        for sup in meta.get("supersessions", []):
            old_ref = sup["old_ref"]
            old_doc_id = await conn.fetchval(_FIND_DOC_BY_CIRCULAR, f"%{old_ref}%")
            if old_doc_id and old_doc_id != doc_id:
                try:
                    await conn.execute(
                        _SUPERSESSION_UPSERT,
                        doc_id,
                        old_doc_id,
                        sup["relation"],
                        sup["evidence_text"],
                    )
                except Exception as e:
                    logger.warning(f"Supersession insert failed: {e}")

        # Recompute statuses
        await conn.execute("SELECT recompute_status()")

    result["ok"] = True
    result["chunks"] = len(chunks)
    logger.success(f"Ingested {pdf_path.name}: {len(chunks)} chunks, doc_id={doc_id}")
    return result


async def run_ingest(
    directory: Path,
    regulator: str,
    force: bool = False,
    manifest_out: Path | None = None,
) -> None:
    """Ingest all PDFs in a directory."""
    pdfs = list(directory.rglob("*.pdf"))
    if not pdfs:
        logger.warning(f"No PDFs found in {directory}")
        return

    logger.info(f"Found {len(pdfs)} PDFs in {directory} (regulator={regulator})")
    pool = await asyncpg.create_pool(dsn=settings.database_url, min_size=2, max_size=5)

    results = []
    for pdf in tqdm(pdfs, desc="Ingesting"):
        r = await ingest_file(pool, pdf, regulator, force=force)
        results.append(r)

    await pool.close()

    ok = sum(1 for r in results if r.get("ok"))
    skip = sum(1 for r in results if r.get("skipped"))
    total_chunks = sum(r.get("chunks", 0) for r in results)
    logger.info(f"Ingest complete: {ok}/{len(results)} ok, {skip} skipped, {total_chunks} chunks")

    if manifest_out:
        manifest_out.write_text(json.dumps(results, indent=2, default=str))
        logger.info(f"Manifest saved to {manifest_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vector index from PDFs")
    parser.add_argument("--dir", required=True, help="Directory of PDFs")
    parser.add_argument("--regulator", choices=["RBI", "SEBI"], required=True)
    parser.add_argument("--rebuild", action="store_true", help="Force re-embed all")
    parser.add_argument("--manifest-out", help="Save ingest summary JSON")
    args = parser.parse_args()

    asyncio.run(
        run_ingest(
            Path(args.dir),
            args.regulator,
            force=args.rebuild,
            manifest_out=Path(args.manifest_out) if args.manifest_out else None,
        )
    )
