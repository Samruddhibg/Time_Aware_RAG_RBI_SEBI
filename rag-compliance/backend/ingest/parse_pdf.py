"""
ingest/parse_pdf.py — PDF → structured text using pdfplumber.

Handles:
- Normal text extraction
- Table extraction (converted to pipe-separated markdown)
- Per-page failure logging
- Returns a list of page dicts

Usage:
    from ingest.parse_pdf import parse_pdf
    pages = parse_pdf(Path("data/raw/rbi/RBI-2024-25-001.pdf"))
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore

try:
    import fitz  # PyMuPDF fallback
except ImportError:
    fitz = None  # type: ignore


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Convert a pdfplumber table to a simple pipe-delimited markdown string."""
    rows = []
    for i, row in enumerate(table):
        cells = [str(c or "").strip().replace("\n", " ") for c in row]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(rows)


def _clean_text(text: str) -> str:
    """Remove header/footer noise, collapse whitespace."""
    # Remove page numbers like "Page 1 of 12"
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix hyphenation
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()


def _parse_with_pdfplumber(path: Path) -> list[dict[str, Any]]:
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                # Extract tables first (remove them from text area)
                tables = page.extract_tables()
                table_md = "\n\n".join(_table_to_markdown(t) for t in tables if t)

                # Extract text (without table bounding boxes)
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                text = _clean_text(text)

                full = text
                if table_md:
                    full = text + "\n\n" + table_md

                pages.append(
                    {"page": i + 1, "text": full, "has_tables": bool(tables), "ok": True}
                )
            except Exception as exc:
                logger.warning(f"{path.name} page {i+1} parse error: {exc}")
                pages.append({"page": i + 1, "text": "", "ok": False, "error": str(exc)})
    return pages


def _parse_with_pymupdf(path: Path) -> list[dict[str, Any]]:
    pages = []
    doc = fitz.open(str(path))
    for i, page in enumerate(doc):
        try:
            text = page.get_text("text")
            text = _clean_text(text)
            pages.append({"page": i + 1, "text": text, "has_tables": False, "ok": True})
        except Exception as exc:
            logger.warning(f"{path.name} page {i+1} PyMuPDF error: {exc}")
            pages.append({"page": i + 1, "text": "", "ok": False, "error": str(exc)})
    doc.close()
    return pages


def parse_pdf(path: Path) -> list[dict[str, Any]]:
    """
    Parse a PDF file into a list of page dicts.
    Falls back from pdfplumber → PyMuPDF → empty result.

    Returns:
        [{"page": int, "text": str, "has_tables": bool, "ok": bool}, ...]
    """
    if pdfplumber is not None:
        try:
            return _parse_with_pdfplumber(path)
        except Exception as exc:
            logger.warning(f"pdfplumber failed for {path.name}: {exc}. Trying PyMuPDF.")

    if fitz is not None:
        try:
            return _parse_with_pymupdf(path)
        except Exception as exc:
            logger.error(f"PyMuPDF also failed for {path.name}: {exc}")

    logger.error(f"All parsers failed for {path.name}")
    return [{"page": 1, "text": "", "ok": False, "error": "all parsers failed"}]


def pages_to_document_text(pages: list[dict[str, Any]]) -> str:
    """Concatenate parsed pages into a single document string."""
    parts = []
    for p in pages:
        if p.get("ok") and p.get("text"):
            parts.append(p["text"])
    return "\n\n".join(parts)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ingest.parse_pdf <path_to_pdf>")
        sys.exit(1)
    result = parse_pdf(Path(sys.argv[1]))
    for p in result:
        ok = "OK" if p["ok"] else "FAIL"
        tables = " [tables]" if p.get("has_tables") else ""
        preview = p["text"][:200].replace("\n", " ")
        print(f"Page {p['page']:3d} [{ok}]{tables}: {preview}")
