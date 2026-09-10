"""
ingestion/loader.py
-------------------
PDF text extraction using PyMuPDF (imported as `fitz`).

WHY PyMuPDF (fitz)?
  Several Python PDF libraries exist — pdfplumber, pypdf, pdfminer.
  PyMuPDF is the fastest and most robust for raw text extraction, handles
  many edge cases (rotated pages, multi-column layouts) and gives us
  page-level access, which is essential for the metadata we attach to
  every chunk. The package name on PyPI is `pymupdf` but it imports as
  `fitz` (a historical naming artifact from the MuPDF C library).

DESIGN DECISION — Page-Level Extraction:
  We return a list of (page_number, text) tuples rather than one big
  concatenated string. This preserves page boundaries so that downstream
  chunking can record which page each chunk came from — a key feature for
  showing citations in the UI ("Answer sourced from page 3 of report.pdf").

FAILURE MODES HANDLED:
  1. File not found / wrong path          → DocumentLoadError
  2. File exists but is not a valid PDF   → DocumentLoadError
  3. PDF is password-protected            → DocumentLoadError
  4. PDF opens but has no text layer      → EmptyDocumentError
     (scanned / image-only PDF)
"""

from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF — pip install pymupdf

from utils.exceptions import DocumentLoadError, EmptyDocumentError
from utils.logger import get_logger

log = get_logger(__name__)


def load_pdf(file_path: str | Path) -> List[Tuple[int, str]]:
    """
    Extract text from a PDF, returning one entry per page.

    Args:
        file_path: Path to the PDF file (string or Path object).

    Returns:
        A list of (page_number, page_text) tuples.
        page_number is 1-indexed (matching how humans label pages).
        Pages with no text are silently skipped (blank separators, etc.).

    Raises:
        DocumentLoadError: If the file cannot be opened or parsed.
        EmptyDocumentError: If the PDF opens but contains no extractable text.
    """
    path = Path(file_path)
    log.info("Loading PDF: %s", path.name)

    # ── 1. Verify the file exists before handing it to PyMuPDF ───────────────
    # PyMuPDF raises a generic RuntimeError for missing files; catching it
    # here lets us raise a more specific, informative exception.
    if not path.exists():
        raise DocumentLoadError(
            f"PDF file not found: '{path}'. "
            "Check that the path is correct and the file has not been moved."
        )

    if not path.is_file():
        raise DocumentLoadError(
            f"'{path}' exists but is not a file (maybe it's a directory?)."
        )

    # ── 2. Open the PDF with PyMuPDF ─────────────────────────────────────────
    try:
        doc = fitz.open(str(path))
    except fitz.FileDataError as exc:
        # FileDataError covers corrupted files and wrong file formats.
        raise DocumentLoadError(
            f"Could not parse '{path.name}' as a PDF. "
            "The file may be corrupted or not a valid PDF."
        ) from exc
    except Exception as exc:
        # Catch-all for unexpected PyMuPDF errors (e.g., internal C error).
        # We re-raise as DocumentLoadError so the caller never sees a raw
        # fitz exception — keeping our exception hierarchy clean.
        raise DocumentLoadError(
            f"Unexpected error opening '{path.name}': {exc}"
        ) from exc

    # ── 3. Handle encrypted (password-protected) PDFs ────────────────────────
    # fitz.open() succeeds on encrypted PDFs but the document is locked.
    # Attempting to read page text from a locked doc raises RuntimeError.
    # We detect this early and give the user an actionable message.
    if doc.is_encrypted:
        doc.close()
        raise DocumentLoadError(
            f"'{path.name}' is password-protected. "
            "Please decrypt the PDF before uploading."
        )

    # ── 4. Extract text page by page ─────────────────────────────────────────
    pages: List[Tuple[int, str]] = []
    total_pages = len(doc)  # save BEFORE closing — len(doc) returns '?' after close

    try:
        for page_index in range(total_pages):
            page = doc.load_page(page_index)

            # get_text("text") extracts the text layer only (no images).
            # "blocks" or "dict" would give richer layout info but we don't
            # need layout for a RAG pipeline — plain text is sufficient and
            # faster to process downstream.
            text = page.get_text("text").strip()

            if text:
                # Page numbers are 1-indexed for human readability in citations.
                pages.append((page_index + 1, text))
            else:
                log.debug(
                    "Page %d of '%s' has no text layer — skipping.",
                    page_index + 1,
                    path.name,
                )
    finally:
        # Always close the file handle, even if extraction fails partway
        # through — prevents resource leaks in long-running Streamlit sessions.
        doc.close()

    # ── 5. Guard against image-only PDFs ─────────────────────────────────────
    if not pages:
        raise EmptyDocumentError(
            f"'{path.name}' contains no extractable text. "
            "It may be a scanned document (image-only PDF). "
            "Try running it through an OCR tool first."
        )

    log.info(
        "Loaded '%s': %d pages with text (out of %d total).",
        path.name,
        len(pages),
        total_pages,
    )
    return pages
