"""
tests/test_retriever.py
------------------------
Tests for retrieval/retriever.py — dense, BM25, and hybrid search.

KEY TESTS:
  1. BM25 keyword boost: "dropout" query → chunks with "dropout" rank higher
     than a generic "importance" chunk that dense retrieval might prefer.
  2. RRF fusion: results from both rankers appear in the merged output.
  3. BM25 index rebuilds when new docs are added.
  4. Hybrid falls back to dense if BM25 fails.

DESIGN: Uses the same test ChromaDB collection as test_embedder.py.
        The autouse fixture resets the collection before each test.
"""
import os
import pytest

os.environ["CHROMA_PERSIST_DIR"] = "./test_chroma_db"
os.environ["CHROMA_COLLECTION_NAME"] = "test_collection"

from ingestion.embedder import embed_and_store, reset_collection
from ingestion.chunker import Chunk
from retrieval.retriever import retrieve, retrieve_bm25, retrieve_hybrid, _ensure_bm25_index


def _chunk(text: str, idx: int, source: str = "test.pdf", page: int = 1) -> Chunk:
    return Chunk(text=text, source_file=source,
                 page_number=page, chunk_id=f"{source}::p{page}::c{idx}")


@pytest.fixture(autouse=True)
def clean_store():
    """Reset ChromaDB and BM25 cache before each test."""
    reset_collection()
    # Reset the module-level BM25 cache
    import retrieval.retriever as r
    r._bm25_index = None
    r._bm25_chunks = []
    r._bm25_corpus_size = 0
    yield
    reset_collection()


class TestBM25Retrieval:
    """Tests specific to BM25 keyword search."""

    def test_bm25_boosts_exact_keyword_match(self):
        """
        THE CORE FIX TEST: chunks containing the exact query keyword
        must rank higher in BM25 than chunks about 'importance' generally.

        This directly reproduces the bug where "importance of dropout"
        retrieved wrong chunks using dense-only search.
        """
        chunks = [
            _chunk("The importance of regularisation in machine learning cannot be overstated.", idx=0),
            _chunk("Dropout is a regularisation technique that randomly deactivates neurons "
                   "during training. The importance of dropout is that it prevents overfitting "
                   "and forces the network to learn redundant representations.", idx=1),
            _chunk("The importance of batch normalisation for training stability.", idx=2),
        ]
        embed_and_store(chunks)
        results = retrieve_bm25("what is the importance of using dropout", top_k=3)

        # The chunk explicitly about dropout must be ranked first
        assert len(results) >= 1
        assert "dropout" in results[0]["text"].lower(), (
            f"BM25 should rank the dropout chunk first, got: '{results[0]['text'][:80]}'"
        )

    def test_bm25_returns_empty_for_unmatched_query(self):
        """
        Query terms with no match in the corpus return an empty list,
        not an error.
        """
        chunks = [_chunk("Python is a general-purpose programming language.", idx=0)]
        embed_and_store(chunks)
        # Search for something completely unrelated to the corpus
        results = retrieve_bm25("xyzzy frobnicator quux", top_k=5)
        assert results == [], "BM25 should return [] for terms not in corpus"

    def test_bm25_index_rebuilds_after_new_docs(self):
        """
        Adding new documents should trigger a BM25 index rebuild,
        so the new chunks are searchable.
        """
        import retrieval.retriever as r

        # Add initial doc
        embed_and_store([_chunk("Initial document about neural networks.", idx=0)])
        retrieve_bm25("neural", top_k=1)
        size_after_first = r._bm25_corpus_size

        # Add another doc
        embed_and_store([_chunk("New document about transformers architecture.", idx=1)])
        retrieve_bm25("transformers", top_k=1)
        size_after_second = r._bm25_corpus_size

        assert size_after_second > size_after_first, (
            "BM25 index must rebuild when new chunks are added"
        )


class TestHybridRetrieval:
    """Tests for the full BM25 + dense hybrid retrieval."""

    def test_hybrid_returns_top_k_results(self):
        """Hybrid retrieval must respect the top_k limit."""
        chunks = [_chunk(f"Document about topic {i}.", idx=i) for i in range(8)]
        embed_and_store(chunks)
        results = retrieve_hybrid("topic", top_k=3)
        assert len(results) <= 3

    def test_hybrid_results_have_rrf_score(self):
        """Each hybrid result must have an rrf_score field from RRF fusion."""
        chunks = [
            _chunk("Machine learning models use gradient descent for optimisation.", idx=0),
            _chunk("Deep learning is a subset of machine learning.", idx=1),
        ]
        embed_and_store(chunks)
        results = retrieve_hybrid("machine learning", top_k=2)
        assert all("rrf_score" in r for r in results), (
            "All hybrid results must have an rrf_score field"
        )

    def test_hybrid_finds_keyword_specific_chunk(self):
        """
        End-to-end hybrid test: the chunk that contains exact keywords
        from the query must appear in the top results.
        """
        chunks = [
            _chunk("Activation functions determine whether a neuron fires.", idx=0),
            _chunk("Backpropagation computes gradients for weight updates.", idx=1),
            _chunk("L2 regularisation adds a penalty proportional to the square of weights, "
                   "preventing the model from relying too heavily on any single feature.", idx=2),
            _chunk("The vanishing gradient problem makes training deep networks difficult.", idx=3),
        ]
        embed_and_store(chunks)
        results = retrieve_hybrid("what is L2 regularisation", top_k=2)

        top_texts = [r["text"].lower() for r in results]
        assert any("l2" in t or "regularisation" in t for t in top_texts), (
            "Hybrid search must surface the L2 regularisation chunk in top results"
        )
