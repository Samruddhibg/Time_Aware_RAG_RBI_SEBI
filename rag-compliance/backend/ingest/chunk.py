"""
ingest/chunk.py — Structure-aware chunking with section path tracking.

Usage:
    from ingest.chunk import chunk_document
    chunks = chunk_document(text, document_id="RBI-2024-25-001", max_words=250, overlap=40)
"""
from __future__ import annotations

import re
from typing import Any

# ── Heading patterns for RBI/SEBI documents ───────────────────────────────────
_HEADING_RE = re.compile(
    r"(?m)^(?:"
    r"(?P<md_h>#{1,4})\s+.+"              # Markdown headings # ## ### ####
    r"|(?P<num>\d+(?:\.\d+)*)\.\s+[A-Z]"  # 1. Title / 1.2 Section
    r"|(?P<roman>[IVX]+)\.\s+[A-Z]"       # Roman numeral sections
    r"|Part\s+[IVXABC\d]"                 # Part I, Part A, Part 1
    r")"
)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split document text into (heading, body) pairs.
    Returns [(heading_path, text), ...].
    """
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    current_text = ""
    current_path = "Introduction"

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            # Save previous section
            if current_text.strip():
                sections.append((" > ".join(path) if path else current_path, current_text.strip()))
            # Update path
            if m.group("md_h"):
                level = len(m.group("md_h"))
                heading = line.lstrip("#").strip()
                path = path[: level - 1] + [heading]
            else:
                heading = line.strip()
                path = [heading]
            current_path = " > ".join(path)
            current_text = ""
        else:
            current_text += line + "\n"

    # Last section
    if current_text.strip():
        sections.append((current_path, current_text.strip()))

    return sections


def _sliding_window(
    words: list[str], max_words: int, overlap: int
) -> list[list[str]]:
    """Split a word list into overlapping windows."""
    step = max(1, max_words - overlap)
    windows = []
    for i in range(0, max(len(words), 1), step):
        window = words[i : i + max_words]
        if len(window) >= 30:  # skip tiny fragments
            windows.append(window)
        if i + max_words >= len(words):
            break
    return windows


def chunk_document(
    text: str,
    document_id: str,
    max_words: int = 250,
    overlap: int = 40,
    min_words: int = 30,
) -> list[dict[str, Any]]:
    """
    Produce overlapping chunks with section metadata.

    Returns:
        [{"document_id": str, "chunk_index": int, "section": str, "text": str,
          "token_count": int}, ...]
    """
    sections = _split_into_sections(text)
    chunks: list[dict[str, Any]] = []
    idx = 0

    for section_path, section_text in sections:
        words = section_text.split()
        if len(words) < min_words:
            # Keep tiny sections as single chunk
            combined = f"{section_path}\n{section_text}"
            chunks.append(
                {
                    "document_id": document_id,
                    "chunk_index": idx,
                    "section": section_path,
                    "text": combined.strip(),
                    "token_count": len(words),
                }
            )
            idx += 1
            continue

        for window in _sliding_window(words, max_words, overlap):
            piece = " ".join(window)
            combined = f"{section_path}\n{piece}"
            chunks.append(
                {
                    "document_id": document_id,
                    "chunk_index": idx,
                    "section": section_path,
                    "text": combined.strip(),
                    "token_count": len(window),
                }
            )
            idx += 1

    return chunks


if __name__ == "__main__":
    import sys, json
    from pathlib import Path
    from ingest.parse_pdf import parse_pdf, pages_to_document_text

    if len(sys.argv) < 2:
        print("Usage: python -m ingest.chunk <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    pages = parse_pdf(pdf_path)
    text = pages_to_document_text(pages)
    chunks = chunk_document(text, document_id=pdf_path.stem)
    print(f"Produced {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"  [{c['chunk_index']}] {c['section']}: {c['text'][:100]!r}")
