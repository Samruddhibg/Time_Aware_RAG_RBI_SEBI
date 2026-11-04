"""
ingest/fetch_docs.py — Download RBI and SEBI circular PDFs.

RBI and SEBI both serve HTML index pages, not direct PDF links.
This script:
  1. Reads the manifest JSON for each regulator.
  2. For each entry, tries the URL as a direct PDF first.
  3. If the response is HTML, scrapes the page for the first PDF link and
     downloads that instead.
  4. Falls back to a BeautifulSoup link scan if no obvious PDF link is found.

Usage:
    python -m ingest.fetch_docs --regulator RBI \
        --manifest data/raw/rbi_manifest.json \
        --out data/raw/rbi

    python -m ingest.fetch_docs --regulator SEBI \
        --manifest data/raw/sebi_manifest.json \
        --out data/raw/sebi

    # Both at once
    python -m ingest.fetch_docs --regulator ALL
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from loguru import logger

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    logger.warning(
        "beautifulsoup4 not installed — HTML scraping disabled. "
        "Run: pip install beautifulsoup4 lxml"
    )

# ── Constants ─────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
}

# Known direct PDF base URLs for each regulator portal
_RBI_PDF_BASE = "https://www.rbi.org.in"
_SEBI_PDF_BASE = "https://www.sebi.gov.in"

# Regex patterns to recognise PDF hrefs
_PDF_HREF_RE = re.compile(r"\.pdf(\?.*)?$", re.IGNORECASE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _is_pdf_content_type(headers: httpx.Headers) -> bool:
    ct = headers.get("content-type", "").lower()
    return "pdf" in ct or "octet-stream" in ct


def _scrape_pdf_url(html: str, base_url: str) -> str | None:
    """Parse HTML and return the first PDF href found, resolved to absolute URL."""
    if not _BS4_AVAILABLE:
        # Fallback: naive regex scan
        matches = re.findall(r'href=["\']([^"\']*\.pdf[^"\']*)["\']', html, re.IGNORECASE)
        if matches:
            href = matches[0]
            return href if href.startswith("http") else urljoin(base_url, href)
        return None

    soup = BeautifulSoup(html, "lxml")
    # 1. Direct <a> tags with .pdf href
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if _PDF_HREF_RE.search(href):
            return href if href.startswith("http") else urljoin(base_url, href)

    # 2. RBI-specific: look for links to /rdocs/ or /upload/
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(segment in href for segment in ["/rdocs/", "/upload/", "/content/"]):
            full = href if href.startswith("http") else urljoin(base_url, href)
            return full

    # 3. SEBI-specific: look for download links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "download" in href.lower() or "attachment" in href.lower():
            full = href if href.startswith("http") else urljoin(base_url, href)
            return full

    return None


def _get_rbi_pdf_url(html: str, page_url: str) -> str | None:
    """
    RBI pages embed the PDF in an <object> or <iframe> tag,
    or link to /rdocs/Content/PDFs/*.pdf
    """
    if not _BS4_AVAILABLE:
        m = re.search(r'(?:src|href)=["\']([^"\']*(?:rdocs|upload)[^"\']*\.pdf)["\']',
                      html, re.IGNORECASE)
        if m:
            href = m.group(1)
            return href if href.startswith("http") else urljoin(_RBI_PDF_BASE, href)
        return _scrape_pdf_url(html, page_url)

    soup = BeautifulSoup(html, "lxml")

    # Check <object data="...pdf">
    obj = soup.find("object", attrs={"data": _PDF_HREF_RE})
    if obj:
        href = obj["data"]
        return href if href.startswith("http") else urljoin(_RBI_PDF_BASE, href)

    # Check <iframe src="...pdf">
    frame = soup.find("iframe", src=_PDF_HREF_RE)
    if frame:
        href = frame["src"]
        return href if href.startswith("http") else urljoin(_RBI_PDF_BASE, href)

    # Check <embed src="...pdf">
    embed = soup.find("embed", src=_PDF_HREF_RE)
    if embed:
        href = embed["src"]
        return href if href.startswith("http") else urljoin(_RBI_PDF_BASE, href)

    return _scrape_pdf_url(html, page_url)


def _get_sebi_pdf_url(html: str, page_url: str) -> str | None:
    """SEBI pages use <a class="fileLink"> or direct PDF links."""
    if not _BS4_AVAILABLE:
        return _scrape_pdf_url(html, page_url)

    soup = BeautifulSoup(html, "lxml")

    # SEBI-specific: <a class="fileLink">
    link = soup.find("a", class_=re.compile(r"fileLink|download-link|pdf", re.I))
    if link and link.get("href"):
        href = link["href"]
        return href if href.startswith("http") else urljoin(_SEBI_PDF_BASE, href)

    return _scrape_pdf_url(html, page_url)


# ── Core download ─────────────────────────────────────────────────────────────

def download_pdf(
    url: str,
    dest: Path,
    regulator: str = "RBI",
    retries: int = 3,
    delay: float = 2.0,
) -> bool:
    """
    Download a PDF from url → dest.
    Handles the case where url is an HTML page embedding the PDF.
    Returns True on success.
    """
    if dest.exists() and dest.stat().st_size > 1024:
        logger.debug(f"Already exists: {dest.name}")
        return True

    for attempt in range(retries):
        try:
            with httpx.Client(
                headers=HEADERS,
                follow_redirects=True,
                timeout=60,
                verify=False,  # Some govt portals have cert issues
            ) as client:
                r = client.get(url)
                r.raise_for_status()

                # Direct PDF response
                if _is_pdf_content_type(r.headers) or r.content[:4] == b"%PDF":
                    dest.write_bytes(r.content)
                    logger.success(
                        f"Downloaded {dest.name} ({len(r.content) // 1024} KB)"
                    )
                    return True

                # HTML response — scrape for the actual PDF URL
                html = r.text
                if regulator == "RBI":
                    pdf_url = _get_rbi_pdf_url(html, url)
                else:
                    pdf_url = _get_sebi_pdf_url(html, url)

                if not pdf_url:
                    logger.warning(
                        f"No PDF link found on page: {url}\n"
                        f"  → Manually download and place at: {dest}"
                    )
                    return False

                logger.info(f"Resolved PDF URL: {pdf_url}")
                pdf_resp = client.get(pdf_url)
                pdf_resp.raise_for_status()

                if not (
                    _is_pdf_content_type(pdf_resp.headers)
                    or pdf_resp.content[:4] == b"%PDF"
                ):
                    logger.warning(f"Resolved URL is not a PDF: {pdf_url}")
                    return False

                dest.write_bytes(pdf_resp.content)
                logger.success(
                    f"Downloaded {dest.name} via scrape ({len(pdf_resp.content) // 1024} KB)"
                )
                return True

        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP {e.response.status_code} on attempt {attempt + 1}: {url}")
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")

        if attempt < retries - 1:
            time.sleep(delay * (attempt + 1))

    logger.error(f"All {retries} attempts failed for {url}")
    return False


# ── Manifest-based download ───────────────────────────────────────────────────

def fetch_from_manifest(
    manifest_path: Path,
    out_dir: Path,
    regulator: str,
    force: bool = False,
) -> tuple[int, int]:
    """
    Download PDFs listed in a JSON manifest.

    Manifest format (array of objects):
    [
      {
        "document_id": "RBI-2024-25-001",
        "regulator": "RBI",
        "url": "https://...",
        ...
      }
    ]

    Returns:
        (n_ok, n_failed)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return 0, 0

    manifest = json.loads(manifest_path.read_text())
    ok, fail = 0, 0
    updated = False

    for entry in manifest:
        doc_id = entry["document_id"]
        url = entry.get("url", "")
        dest = out_dir / f"{doc_id}.pdf"

        if not url:
            logger.warning(f"No URL for {doc_id} — skipping")
            fail += 1
            continue

        if dest.exists() and not force:
            logger.debug(f"Skipping (exists): {dest.name}")
            entry.setdefault("local_path", str(dest))
            entry.setdefault("content_hash", _sha256(dest))
            ok += 1
            continue

        reg = entry.get("regulator", regulator)
        # Try direct URL first; if it fails and url_page exists, try scraping that
        success = download_pdf(url, dest, regulator=reg)
        if not success and entry.get("url_page"):
            logger.info(f"Direct URL failed, trying page scrape: {entry['url_page']}")
            success = download_pdf(entry["url_page"], dest, regulator=reg)

        if success:
            entry["local_path"] = str(dest)
            entry["content_hash"] = _sha256(dest)
            entry["download_ok"] = True
            ok += 1
            updated = True
        else:
            entry["download_ok"] = False
            entry["local_path"] = str(dest)  # expected path even if missing
            fail += 1
            updated = True

        # Polite delay between requests
        time.sleep(1.5)

    # Write updated manifest (with local_path and content_hash filled in)
    if updated:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(f"[{regulator}] Done: {ok} downloaded/skipped, {fail} failed")
    return ok, fail


def print_manual_download_guide(manifest_path: Path, out_dir: Path) -> None:
    """Print a guide for any files that still need manual download."""
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    missing = [
        e for e in manifest
        if not Path(e.get("local_path", "")).exists()
    ]
    if not missing:
        logger.info("All files downloaded successfully — no manual steps needed.")
        return

    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 70)
    print(f"The following {len(missing)} file(s) could not be downloaded automatically.")
    print("Visit each URL in your browser, download the PDF, and save it to the")
    print(f"indicated path.\n")
    for e in missing:
        expected_path = out_dir / f"{e['document_id']}.pdf"
        print(f"  Document : {e['document_id']}")
        print(f"  Title    : {e.get('title', 'N/A')}")
        print(f"  URL      : {e['url']}")
        print(f"  Save to  : {expected_path}")
        print()
    print("=" * 70 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download RBI/SEBI regulatory PDFs from manifests."
    )
    parser.add_argument(
        "--regulator",
        choices=["RBI", "SEBI", "ALL"],
        default="ALL",
        help="Which regulator's documents to fetch (default: ALL)",
    )
    parser.add_argument(
        "--manifest",
        help="Path to a specific manifest JSON file (overrides --regulator auto-path)",
    )
    parser.add_argument(
        "--out",
        default="data/raw",
        help="Output root directory (default: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local file exists",
    )
    args = parser.parse_args()

    out_root = Path(args.out)
    regs = ["RBI", "SEBI"] if args.regulator == "ALL" else [args.regulator]

    total_ok, total_fail = 0, 0
    for reg in regs:
        if args.manifest:
            manifest_path = Path(args.manifest)
            # When --manifest is given alongside --out, use --out directly as
            # the destination folder (don't append reg.lower() again).
            reg_dir = out_root
        else:
            # Default: manifests live at data/raw/{reg}_manifest.json
            # PDFs go to  data/raw/{reg}/
            manifest_path = out_root / f"{reg.lower()}_manifest.json"
            reg_dir = out_root / reg.lower()

        reg_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Fetching {reg} documents from {manifest_path} → {reg_dir}")

        n_ok, n_fail = fetch_from_manifest(
            manifest_path, reg_dir, reg, force=args.force
        )
        total_ok += n_ok
        total_fail += n_fail

        print_manual_download_guide(manifest_path, reg_dir)

    if total_fail > 0:
        logger.warning(
            f"{total_fail} file(s) failed. See the manual download guide above."
        )
        sys.exit(1)
    else:
        logger.success(f"All {total_ok} documents ready.")
