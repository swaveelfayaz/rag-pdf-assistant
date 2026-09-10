"""
tests/test_ingestion.py
-----------------------
Test suite for ingestion/loader.py and ingestion/chunker.py.

TEST STRATEGY:
  We test at the unit level — each test is small, isolated, and tests
  exactly one behavior. We use pytest fixtures to create synthetic test
  artifacts (real PDFs, not mocks) so tests are realistic but controlled.

  WHY REAL PDFs, NOT MOCKS?
    Mocking PyMuPDF internals would be brittle — the tests would pass
    even if PyMuPDF breaks. Real PDFs let us test the actual code path.
    We create minimal PDFs programmatically using PyMuPDF itself, which
    avoids needing binary test fixtures committed to git.

RUNNING THE TESTS:
    cd rag-pdf-assistant
    pytest tests/ -v

  All 8 tests should pass on a clean install.
"""

from pathlib import Path
import os
import pytest
import fitz  # PyMuPDF — used here only to create test PDFs

# ── Make sure the project root is on sys.path ────────────────────────────────
# pytest usually handles this, but add conftest.py or set PYTHONPATH if needed.

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_documents, Chunk
from utils.exceptions import (
    DocumentLoadError,
    EmptyDocumentError,
    ChunkingError,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def normal_pdf(tmp_path: Path) -> Path:
    """
    Create a minimal valid PDF with two pages of text.
    Returns the path to the created PDF file.
    """
    pdf_path = tmp_path / "test_normal.pdf"
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_text(
        fitz.Point(50, 100),
        "This is the first page of the test document. "
        "It contains some sample text about machine learning and AI.",
    )

    # Page 2
    page2 = doc.new_page()
    page2.insert_text(
        fitz.Point(50, 100),
        "This is the second page. "
        "It discusses retrieval-augmented generation and vector databases.",
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    """
    Create a PDF with no text layer (image-only simulation).
    All pages have no insertable text — simulates a scanned document.
    """
    pdf_path = tmp_path / "test_empty.pdf"
    doc = fitz.open()
    doc.new_page()  # blank page — no text inserted
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: loader.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadPdf:
    """Unit tests for ingestion/loader.py::load_pdf()"""

    def test_loads_normal_pdf_successfully(self, normal_pdf: Path) -> None:
        """
        Happy path: a valid PDF with text should return a list of
        (page_number, text) tuples.
        """
        pages = load_pdf(normal_pdf)

        assert isinstance(pages, list)
        assert len(pages) == 2, "Expected 2 pages with text"

        # Verify tuple structure
        page_num, text = pages[0]
        assert isinstance(page_num, int)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_page_numbers_are_one_indexed(self, normal_pdf: Path) -> None:
        """
        Page numbers must start at 1 (human-readable), not 0 (zero-indexed).
        This is important for citations — users expect 'page 1', not 'page 0'.
        """
        pages = load_pdf(normal_pdf)
        page_numbers = [p[0] for p in pages]
        assert page_numbers[0] == 1, "First page should be page 1, not page 0"
        assert page_numbers[1] == 2

    def test_text_content_is_correct(self, normal_pdf: Path) -> None:
        """
        The extracted text should contain the content we inserted.
        This verifies that PyMuPDF is actually reading our test data.
        """
        pages = load_pdf(normal_pdf)
        all_text = " ".join(text for _, text in pages)

        assert "machine learning" in all_text.lower()
        assert "retrieval-augmented generation" in all_text.lower()

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        """
        Requesting a file that doesn't exist must raise DocumentLoadError,
        not a generic FileNotFoundError or RuntimeError from PyMuPDF.
        """
        missing_path = tmp_path / "does_not_exist.pdf"

        with pytest.raises(DocumentLoadError, match="not found"):
            load_pdf(missing_path)

    def test_raises_for_non_pdf_file(self, tmp_path: Path) -> None:
        """
        Passing a non-PDF file (e.g., a text file with .pdf extension)
        must raise DocumentLoadError.
        """
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_text("this is not a PDF", encoding="utf-8")

        with pytest.raises(DocumentLoadError):
            load_pdf(fake_pdf)

    def test_raises_for_empty_pdf(self, empty_pdf: Path) -> None:
        """
        A PDF that opens successfully but has no text layer must raise
        EmptyDocumentError (a subclass of DocumentLoadError).
        """
        with pytest.raises(EmptyDocumentError):
            load_pdf(empty_pdf)

    def test_accepts_path_object_and_string(self, normal_pdf: Path) -> None:
        """
        load_pdf() must accept both pathlib.Path objects and plain strings.
        This ensures compatibility with Streamlit's UploadedFile path handling.
        """
        # Path object
        pages_from_path = load_pdf(normal_pdf)
        # String
        pages_from_str = load_pdf(str(normal_pdf))

        assert len(pages_from_path) == len(pages_from_str)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: chunker.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkDocuments:
    """Unit tests for ingestion/chunker.py::chunk_documents()"""

    def test_chunks_normal_document(self, normal_pdf: Path) -> None:
        """
        Happy path: a valid document should be split into at least one chunk.
        """
        pages = load_pdf(normal_pdf)
        chunks = chunk_documents(pages, source_file="test_normal.pdf")

        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_metadata_is_correct(self, normal_pdf: Path) -> None:
        """
        Every chunk must carry the correct source_file and page_number.
        This metadata is what powers citations in the final answer.
        """
        pages = load_pdf(normal_pdf)
        chunks = chunk_documents(pages, source_file="my_report.pdf")

        for chunk in chunks:
            assert chunk.source_file == "my_report.pdf"
            assert isinstance(chunk.page_number, int)
            assert chunk.page_number >= 1
            assert chunk.chunk_id.startswith("my_report.pdf::")
            assert len(chunk.text) > 0

    def test_chunk_ids_are_unique(self, normal_pdf: Path) -> None:
        """
        All chunk IDs must be unique within a document.
        Duplicate IDs would cause silent overwrites in the vector store.
        """
        pages = load_pdf(normal_pdf)
        chunks = chunk_documents(pages, source_file="test.pdf")
        chunk_ids = [c.chunk_id for c in chunks]

        assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs must be unique"

    def test_raises_for_empty_pages_list(self) -> None:
        """
        Passing an empty list to chunk_documents must raise ChunkingError,
        not silently return an empty list (which would hide the bug upstream).
        """
        with pytest.raises(ChunkingError, match="empty document"):
            chunk_documents([], source_file="empty.pdf")

    def test_raises_for_invalid_chunk_config(self, normal_pdf: Path) -> None:
        """
        If chunk_overlap >= chunk_size, chunks would have no unique content.
        The chunker must raise ValueError immediately, not produce garbage output.
        """
        pages = load_pdf(normal_pdf)

        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_documents(
                pages,
                source_file="test.pdf",
                chunk_size=100,
                chunk_overlap=100,  # overlap equals size — invalid
            )
