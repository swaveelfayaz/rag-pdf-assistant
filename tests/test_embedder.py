"""
tests/test_embedder.py
----------------------
Tests for ingestion/embedder.py.

KEY TEST — "EMBED AND RETRIEVE":
  The most important test for a RAG pipeline is: if I embed a piece of
  text, can I retrieve it by querying with a semantically similar (or
  identical) query? This is the foundation the whole pipeline rests on.
  If this test fails, retrieval is broken regardless of everything else.

ISOLATION:
  Each test that writes to ChromaDB calls reset_collection() in its
  setup so tests don't interfere with each other. We use a test-specific
  ChromaDB path so test runs never corrupt the real application data.

PERFORMANCE NOTE:
  The first test in this file loads the sentence-transformer model
  (~80 MB). Subsequent tests reuse the cached module-level singleton.
  Expect ~3-5 seconds on first run, <1 second for subsequent tests.
"""

import os
import pytest

# ── Override ChromaDB path BEFORE importing embedder ─────────────────────────
# The embedder creates a PersistentClient pointing at CHROMA_PERSIST_DIR.
# We redirect it to a temp directory so tests don't touch real app data.
# IMPORTANT: This must happen before `from ingestion.embedder import ...`
# because the module reads config at import time via the singleton pattern.
os.environ["CHROMA_PERSIST_DIR"] = "./test_chroma_db"
os.environ["CHROMA_COLLECTION_NAME"] = "test_collection"

from ingestion.embedder import (
    embed_and_store,
    query_similar,
    get_collection_count,
    reset_collection,
)
from ingestion.chunker import Chunk
from utils.exceptions import EmbeddingError, VectorStoreError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunk(text: str, page: int = 1, idx: int = 0) -> Chunk:
    """Create a Chunk with predictable test data."""
    return Chunk(
        text=text,
        source_file="test_doc.pdf",
        page_number=page,
        chunk_id=f"test_doc.pdf::p{page}::c{idx}",
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_collection():
    """
    Reset ChromaDB before every test in this file.

    `autouse=True` means this fixture runs automatically — you don't need
    to include it in each test's parameter list. This guarantees test
    isolation: a chunk inserted in test A cannot affect test B.
    """
    reset_collection()
    yield
    # Teardown: reset again after the test so the last test also cleans up.
    reset_collection()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEmbedAndStore:
    """Tests for embed_and_store()."""

    def test_embeds_and_stores_chunks(self) -> None:
        """
        Happy path: embedding a list of chunks should increase collection count.
        This verifies the full encode → upsert pipeline works end to end.
        """
        chunks = [
            _make_chunk("Machine learning is a subset of artificial intelligence.", idx=0),
            _make_chunk("ChromaDB is a vector database for AI applications.", idx=1),
        ]

        count = embed_and_store(chunks)

        assert count == 2, "Should report 2 chunks upserted"
        assert get_collection_count() == 2, "ChromaDB should contain 2 documents"

    def test_returns_zero_for_empty_list(self) -> None:
        """
        Passing an empty list should return 0 without raising an exception.
        Defensive programming: the caller might pass an empty list if a PDF
        had no text — this should be a no-op, not a crash.
        """
        count = embed_and_store([])
        assert count == 0

    def test_upsert_is_idempotent(self) -> None:
        """
        Upserting the same chunks twice should NOT increase the count.
        This verifies we're using upsert (overwrite) not add (duplicate).
        If this test fails, re-uploading a PDF creates duplicate chunks
        that corrupt retrieval results.
        """
        chunk = _make_chunk("The quick brown fox jumps over the lazy dog.", idx=0)

        embed_and_store([chunk])
        embed_and_store([chunk])  # same chunk_id — should overwrite, not duplicate

        assert get_collection_count() == 1, (
            "Upserting the same chunk twice must not create duplicates"
        )


class TestQuerySimilar:
    """Tests for query_similar() — the dense retrieval function."""

    def test_retrieves_embedded_chunk_with_identical_query(self) -> None:
        """
        THE CORE RAG TEST: embed a chunk, then query with the same text.
        The embedded chunk must be the top result.

        Why this matters: if the most similar document to an exact query is
        NOT that document itself, the embedding model or the similarity metric
        is broken and the entire pipeline is unreliable.
        """
        target_text = "Retrieval-augmented generation combines search with language models."
        chunk = _make_chunk(target_text, idx=0)

        embed_and_store([chunk])
        results = query_similar(target_text, top_k=1)

        # ChromaDB results are nested lists: results['documents'][0] is the
        # list of documents for the first query (we only send one query).
        returned_docs = results["documents"][0]
        assert len(returned_docs) == 1
        assert returned_docs[0] == target_text, (
            "An identical query must return the embedded chunk as the top result"
        )

    def test_retrieves_semantically_similar_chunk(self) -> None:
        """
        Query with DIFFERENT words but the SAME meaning — the correct chunk
        must still rank #1. This is the whole point of semantic search.

        If this fails, we have embedding-level semantic understanding missing.
        """
        chunks = [
            _make_chunk("Cats are domestic animals that are kept as pets.", idx=0),
            _make_chunk("The stock market saw significant gains today.", idx=1),
            _make_chunk("Python is a popular programming language for data science.", idx=2),
        ]
        embed_and_store(chunks)

        # Query with related words but not identical text.
        results = query_similar("felines are common household companions", top_k=1)
        top_doc = results["documents"][0][0]

        # The cat chunk should rank higher than stock markets or Python.
        assert "cats" in top_doc.lower() or "domestic" in top_doc.lower(), (
            f"Semantic search failed: got '{top_doc}' for a query about felines"
        )

    def test_metadata_is_returned_with_results(self) -> None:
        """
        Every retrieved chunk must carry source_file and page_number metadata.
        This metadata is what powers the citation feature in the UI.
        """
        chunk = _make_chunk("Vector databases enable efficient similarity search.", page=7, idx=0)
        embed_and_store([chunk])

        results = query_similar("similarity search", top_k=1)
        metadata = results["metadatas"][0][0]

        assert metadata["source_file"] == "test_doc.pdf"
        assert metadata["page_number"] == 7

    def test_raises_when_collection_is_empty(self) -> None:
        """
        Querying an empty collection must raise VectorStoreError.
        Allowing a query on an empty store would silently return zero results,
        which is harder to debug than a clear error message.
        """
        # clean_collection fixture already reset — store is empty here.
        with pytest.raises(VectorStoreError, match="empty"):
            query_similar("anything", top_k=1)

    def test_top_k_limits_results(self) -> None:
        """
        Requesting top_k=2 from a collection of 5 chunks should return ≤2 results.
        """
        chunks = [_make_chunk(f"Document about topic number {i}.", idx=i) for i in range(5)]
        embed_and_store(chunks)

        results = query_similar("topic", top_k=2)
        assert len(results["documents"][0]) <= 2
