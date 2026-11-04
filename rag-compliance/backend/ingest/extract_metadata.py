"""
ingest/extract_metadata.py — Regex-based metadata extractor for RBI/SEBI circulars.

Extracts: document_id, regulator, circular_no, title, issue_date, doc_type,
          supersedes references, confidence score.
"""
from __future__ import annotations

import re
import hashlib
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger

# ── Patterns ──────────────────────────────────────────────────────────────────
_RBI_CIRCULAR_NO = re.compile(
    r"RBI\s*/\s*(\d{4}-\d{2,4})\s*/\s*(\d+)", re.IGNORECASE
)
_SEBI_CIRCULAR_NO = re.compile(
    r"SEBI\s*/\s*([\w/\-]+/CIR/\d{4}/\d+)", re.IGNORECASE
)
_DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
               r"September|October|November|December),?\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"),
]
_SUPERSEDES_PATTERNS = [
    re.compile(r"superced(?:es|ing)\s+(?:circular|direction|notification)?\s*"
               r"(RBI[/\s\d-]+|SEBI[/\s\w-]+)", re.IGNORECASE),
    re.compile(r"withdraw[ns]?\s+(?:circular|direction)?\s*(RBI[/\s\d-]+|SEBI[/\s\w-]+)",
               re.IGNORECASE),
    re.compile(r"amend(?:s|ing)?\s+(?:the\s+)?(?:Master\s+Direction|circular)\s+"
               r"(RBI[/\s\d-]+|SEBI[/\s\w-]+)", re.IGNORECASE),
]
_TITLE_PATTERN = re.compile(
    r"(?:Subject|Re|Dear\s+Sir/Madam)[:\s]+([^\n]{10,200})", re.IGNORECASE
)
_DOC_TYPE_KEYWORDS = {
    "master_direction": ["master direction", "master directions"],
    "master_circular": ["master circular"],
    "circular": ["circular", "notification"],
    "notification": ["notification", "press release"],
}

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _parse_date(text: str) -> date | None:
    # "12 January 2024"
    m = _DATE_PATTERNS[0].search(text)
    if m:
        try:
            return date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
        except (ValueError, KeyError):
            pass
    # ISO "2024-01-12"
    m = _DATE_PATTERNS[1].search(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _detect_supersessions(
    text: str, new_doc_id: str
) -> list[dict[str, str]]:
    results = []
    for pat in _SUPERSEDES_PATTERNS:
        for m in pat.finditer(text):
            ref = m.group(1).strip()
            relation = "supersedes"
            if "withdraw" in m.group(0).lower():
                relation = "withdraws"
            elif "amend" in m.group(0).lower():
                relation = "amends"
            results.append(
                {
                    "new_doc_id": new_doc_id,
                    "old_ref": ref,
                    "relation": relation,
                    "evidence_text": m.group(0)[:300],
                }
            )
    return results


def _compute_confidence(meta: dict[str, Any]) -> float:
    score = 0.0
    if meta.get("circular_no"):
        score += 0.3
    if meta.get("issue_date"):
        score += 0.3
    if meta.get("title"):
        score += 0.2
    if meta.get("doc_type"):
        score += 0.2
    return round(score, 2)


def extract_metadata(
    text: str,
    source_path: Path,
    regulator: str,
    document_id: str | None = None,
) -> dict[str, Any]:
    """
    Extract metadata from the first ~3000 chars of a document.

    Returns a dict compatible with the documents table.
    """
    head = text[:3000]

    # Circular number
    circular_no = None
    if regulator == "RBI":
        m = _RBI_CIRCULAR_NO.search(head)
        if m:
            circular_no = f"RBI/{m.group(1)}/{m.group(2)}"
    else:
        m = _SEBI_CIRCULAR_NO.search(head)
        if m:
            circular_no = m.group(1)

    # Auto document_id
    if not document_id:
        if circular_no:
            slug = circular_no.replace("/", "-").replace(" ", "")
            document_id = f"{regulator}-{slug}"
        else:
            h = hashlib.md5(text[:500].encode()).hexdigest()[:8]
            document_id = f"{regulator}-{h}"

    # Date
    issue_date = _parse_date(head)

    # Title
    title = None
    m = _TITLE_PATTERN.search(head)
    if m:
        title = m.group(1).strip()[:200]
    if not title:
        # Take first non-empty line after stripping
        for line in head.splitlines():
            line = line.strip()
            if len(line) > 15:
                title = line[:200]
                break

    # Doc type
    doc_type = "circular"
    head_lower = head.lower()
    for dtype, keywords in _DOC_TYPE_KEYWORDS.items():
        if any(kw in head_lower for kw in keywords):
            doc_type = dtype
            break

    # Supersessions
    supersessions = _detect_supersessions(text, document_id)

    meta = {
        "document_id": document_id,
        "regulator": regulator,
        "circular_no": circular_no,
        "title": title,
        "doc_type": doc_type,
        "issue_date": str(issue_date) if issue_date else None,
        "status": "active",
        "source_url": str(source_path),
        "supersessions": supersessions,
    }
    meta["meta_confidence"] = _compute_confidence(meta)
    return meta


if __name__ == "__main__":
    import sys, json
    from ingest.parse_pdf import parse_pdf, pages_to_document_text

    path = Path(sys.argv[1])
    regulator = sys.argv[2] if len(sys.argv) > 2 else "RBI"
    pages = parse_pdf(path)
    text = pages_to_document_text(pages)
    meta = extract_metadata(text, path, regulator)
    print(json.dumps(meta, indent=2, default=str))
