"""
tests/test_reranker.py
-----------------------
Tests for retrieval/reranker.py (cross-encoder reranking).

IMPORTANT BEHAVIOUR BEING TESTED:
  1. Reranker correctly promotes the most relevant chunk to rank #1
     even when it wasn't rank #1 in the input candidates.
  2. Reranker respects top_n — returns at most n results.
  3. Empty input is handled gracefully (no crash).
  4. Every result has a rerank_score field.
  5. retrieve_reranked() orchestrates hybrid + reranking end-to-end.

PERFORMANCE NOTE:
  The cross-encoder model (~85 MB) is downloaded on the first test run.
  Subsequent runs are fast because the model is cached by sentence-transformers.
  Expect ~60-90 seconds on first run, ~5-10 seconds on subsequent runs.
"""
import os
import pytest

os.environ["CHROMA_PERSIST_DIR"] = "./test_chroma_db"
os.environ["CHROMA_COLLECTION_NAME"] = "test_collection"

from ingestion.embedder import embed_and_store, reset_collection
from ingestion.chunker import Chunk
from retrieval.reranker import rerank
from retrieval.retriever import retrieve_reranked


def _chunk(text: str, idx: int, source: str = "test.pdf", page: int = 1) -> Chunk:
    return Chunk(
        text=text,
        source_file=source,
        page_number=page,
        chunk_id=f"{source}::p{page}::c{idx}",
    )


def _candidate(text: str, chunk_id: str, score: float = 0.5) -> dict:
    """Create a mock candidate dict as retrieve_hybrid() would return."""
    return {
        "text": text,
        "source_file": "test.pdf",
        "page_number": 1,
        "chunk_id": chunk_id,
        "score": score,
        "rrf_score": 0.01,
    }


@pytest.fixture(autouse=True)
def clean_store():
    reset_collection()
    import retrieval.retriever as r
    r._bm25_index = None
    r._bm25_chunks = []
    r._bm25_corpus_size = 0
    yield
    reset_collection()


class TestRerank:
    """Unit tests for the rerank() function directly."""

    def test_rerank_promotes_most_relevant_chunk(self):
        """
        THE CORE RERANKER TEST:
        Given candidates where the most relevant chunk is NOT at position 0,
        the cross-encoder must promote it to rank #1.

        This tests the entire value proposition of reranking.
        """
        query = "what is dropout regularisation?"

        # Candidates in a deliberately wrong order:
        # The true answer is candidate C (index 2) but it's last in the input list.
        candidates = [
            _candidate(
                "Neural networks are computational models inspired by the brain.",
                chunk_id="c0",
                score=0.9,  # high cosine score but semantically vague
            ),
            _candidate(
                "Regularisation techniques prevent models from memorising training data.",
                chunk_id="c1",
                score=0.7,
            ),
            _candidate(
                "Dropout randomly deactivates neurons during training, forcing the "
                "network to learn redundant representations and preventing overfitting. "
                "It is one of the most effective regularisation methods for deep neural networks.",
                chunk_id="c2",
                score=0.5,  # low cosine score despite being the best answer
            ),
        ]

        reranked = rerank(query, candidates, top_n=3)

        # The dropout chunk must be ranked #1 by the cross-encoder
        assert reranked[0]["chunk_id"] == "c2", (
            f"Cross-encoder must promote the dropout chunk to rank #1, "
            f"but got '{reranked[0]['text'][:60]}'"
        )

    def test_rerank_adds_rerank_score_to_all_results(self):
        """Every result must have a rerank_score field (a float)."""
        candidates = [
            _candidate("Machine learning is a field of AI.", chunk_id="c0"),
            _candidate("Deep learning uses neural networks.", chunk_id="c1"),
        ]
        reranked = rerank("what is machine learning", candidates, top_n=2)

        assert all("rerank_score" in r for r in reranked), (
            "Every reranked result must have a rerank_score field"
        )
        assert all(isinstance(r["rerank_score"], float) for r in reranked)

    def test_rerank_respects_top_n(self):
        """Requesting top_n=1 from 3 candidates must return exactly 1 result."""
        candidates = [
            _candidate("Topic A explanation.", chunk_id="c0"),
            _candidate("Topic B explanation.", chunk_id="c1"),
            _candidate("Topic C explanation.", chunk_id="c2"),
        ]
        reranked = rerank("explain topic A", candidates, top_n=1)
        assert len(reranked) == 1

    def test_rerank_handles_empty_candidates(self):
        """Empty input must return an empty list, not raise an exception."""
        result = rerank("any query", [], top_n=5)
        assert result == []

    def test_rerank_scores_are_sorted_descending(self):
        """Results must be sorted from highest to lowest rerank_score."""
        candidates = [
            _candidate("Backpropagation algorithm explanation.", chunk_id="c0"),
            _candidate("Dropout prevents overfitting in deep learning.", chunk_id="c1"),
            _candidate("The history of artificial intelligence.", chunk_id="c2"),
        ]
        reranked = rerank("how does dropout work", candidates, top_n=3)

        scores = [r["rerank_score"] for r in reranked]
        assert scores == sorted(scores, reverse=True), (
            "Reranked results must be sorted by rerank_score descending"
        )


class TestRetrieveReranked:
    """End-to-end tests for the full retrieve_reranked() pipeline."""

    def test_retrieve_reranked_returns_top_n_results(self):
        """Full pipeline must return at most top_n results."""
        chunks = [
            _chunk("Gradient descent minimises the loss function.", idx=0),
            _chunk("Backpropagation computes gradients via chain rule.", idx=1),
            _chunk("Learning rate controls the step size in gradient descent.", idx=2),
            _chunk("Momentum accelerates gradient descent in relevant directions.", idx=3),
        ]
        embed_and_store(chunks)

        results = retrieve_reranked("how does gradient descent work", top_n=2)
        assert len(results) <= 2

    def test_retrieve_reranked_results_have_rerank_score(self):
        """Full pipeline results must include rerank_score from the cross-encoder."""
        chunks = [
            _chunk("Attention mechanisms allow models to focus on relevant input parts.", idx=0),
            _chunk("Transformers use self-attention and positional encodings.", idx=1),
        ]
        embed_and_store(chunks)

        results = retrieve_reranked("what is self-attention", top_n=2)
        if results:  # might get less if ChromaDB has few chunks
            assert all("rerank_score" in r for r in results), (
                "retrieve_reranked() results must all have a rerank_score"
            )
