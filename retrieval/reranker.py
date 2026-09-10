"""
retrieval/reranker.py
---------------------
Cross-encoder reranking — the precision layer of the RAG pipeline.

WHERE THIS FITS:
  Full corpus (1000s of chunks)
    ↓  Step 3: embed_and_store()      — index everything
    ↓  Step 5: retrieve_hybrid()      — fast BM25+dense, high RECALL
    ↓  Step 6: rerank()               — slow but accurate, high PRECISION ← THIS FILE
    ↓  Step 7: generate_answer()      — LLM generation

─────────────────────────────────────────────────────────────────────────────
BI-ENCODER vs CROSS-ENCODER — THE KEY DISTINCTION

  BI-ENCODER (all-MiniLM-L6-v2, used in Step 3):
    • Encodes query → vector Q
    • Encodes each document → vector D  (pre-computed, stored in ChromaDB)
    • Similarity = cosine(Q, D)
    • Query and document NEVER attend to each other during encoding
    • Fast: O(1) at query time (just vector lookup)
    • Limitation: loses fine-grained cross-attention signals

  CROSS-ENCODER (ms-marco-MiniLM-L-6-v2, used here):
    • Input: concatenated [CLS] query [SEP] document [SEP]
    • Every token in the query attends to every token in the document
      via full transformer self-attention (12 layers × 12 heads)
    • Output: single relevance score (logit, can be any real number)
    • Slow: O(k) at query time — must run for each (query, chunk) pair
    • But much more accurate: cross-attention captures query-document
      interactions that bi-encoders completely miss

  EXAMPLE of what cross-attention catches:
    Query: "what causes gradient to vanish in deep networks?"
    Chunk A: "Vanishing gradients occur when gradients become very small
              during backpropagation, preventing early layers from learning."
    Chunk B: "Gradients are used to update weights in neural networks.
              Deep networks have many layers."

    Bi-encoder: both chunks have "gradient" + "deep network" — similar scores
    Cross-encoder: sees that "vanish" in the query matches "become very small"
                   in chunk A (via attention), not chunk B — correct ranking.

MODEL CHOICE — cross-encoder/ms-marco-MiniLM-L-6-v2:
  • Trained on MS MARCO: 500k (query, passage, label) triplets
  • 22M parameters — same size as our bi-encoder
  • ~85 MB download, ~30-50ms per pair on CPU
  • NDCG@10 on BEIR benchmark: beats MiniLM bi-encoder by ~8 points
  • Larger option: ms-marco-electra-base (440 MB, better, slower)

WHY NOT ALWAYS USE CROSS-ENCODER FOR RETRIEVAL?
  With 10,000 chunks: 10,000 pairs × 50ms = 500 seconds per query.
  That's why we first narrow down to 15 candidates (< 1 second each),
  then cross-encode only those 15 (< 1 second total).
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import List

from sentence_transformers import CrossEncoder

from utils.config import config
from utils.exceptions import RerankingError
from utils.logger import get_logger

log = get_logger(__name__)

# ── Module-level singleton ────────────────────────────────────────────────────
# Same lazy-loading pattern as embedder.py. CrossEncoder loads a BERT-style
# transformer from disk (~85 MB) — expensive to create, cheap to reuse.
_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    """
    Lazily load the cross-encoder model on first call.

    WHY max_length=512?
      BERT-based models have a hard limit of 512 tokens. The input is:
      [CLS] query tokens [SEP] chunk tokens [SEP]
      Query ≈ 20-50 tokens. Chunk ≈ 150-200 tokens. Total ≈ 200-250 tokens.
      We're well within the limit, but setting it explicitly prevents
      silent truncation if someone uses very long queries or large chunks.

    Returns:
        Loaded CrossEncoder, ready to score (query, passage) pairs.

    Raises:
        RerankingError: If the model cannot be loaded.
    """
    global _cross_encoder
    if _cross_encoder is None:
        log.info(
            "Loading cross-encoder model '%s' (first call)...",
            config.rerank_model,
        )
        try:
            _cross_encoder = CrossEncoder(
                config.rerank_model,
                max_length=512,
            )
            log.info("Cross-encoder loaded.")
        except Exception as exc:
            raise RerankingError(
                f"Failed to load cross-encoder '{config.rerank_model}'. "
                "Check your internet connection (first run downloads ~85 MB). "
                f"Error: {exc}"
            ) from exc
    return _cross_encoder


def rerank(
    query: str,
    candidates: List[dict],
    top_n: int | None = None,
) -> List[dict]:
    """
    Re-score a list of candidate chunks with a cross-encoder and return top_n.

    PROCESS:
      1. Build (query, chunk_text) pairs for the cross-encoder.
      2. Run a single batched forward pass — all pairs scored simultaneously.
      3. Sort candidates by descending cross-encoder score.
      4. Return the top_n most relevant chunks.

    WHY BATCHED?
      CrossEncoder.predict(pairs) processes all pairs in one GPU/CPU batch.
      This is more efficient than iterating one pair at a time because the
      transformer can parallelise across the batch dimension.

    THE SCORE:
      CrossEncoder outputs a raw logit (unbounded real number).
      Higher = more relevant to the query.
      We add it as 'rerank_score' alongside the existing 'score' (cosine)
      so you can compare both in the UI and understand the difference.

    Args:
        query:      The user's question.
        candidates: List of result dicts from retrieve_hybrid().
                    Each must have at least 'text' and 'chunk_id'.
        top_n:      Number of results to return after reranking.
                    Defaults to config.rerank_top_n (3).

    Returns:
        Reranked list of result dicts, sorted by rerank_score descending.
        Each dict gains a 'rerank_score' field.

    Raises:
        RerankingError: If the cross-encoder fails.
    """
    n = top_n if top_n is not None else config.rerank_top_n

    if not candidates:
        log.warning("rerank() called with empty candidates — returning [].")
        return []

    # Cap n to the number of available candidates
    n = min(n, len(candidates))

    encoder = _get_cross_encoder()

    # Build pairs: [(query, chunk1_text), (query, chunk2_text), ...]
    # The cross-encoder concatenates them internally as:
    #   [CLS] query [SEP] chunk_text [SEP]
    pairs = [(query, chunk["text"]) for chunk in candidates]

    log.info(
        "Reranking %d candidates with cross-encoder (returning top %d)...",
        len(candidates), n,
    )

    try:
        # predict() returns a numpy array of floats, one per pair
        scores = encoder.predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except Exception as exc:
        raise RerankingError(
            f"Cross-encoder scoring failed: {exc}"
        ) from exc

    # Attach scores to candidates and sort
    scored = []
    for chunk, score in zip(candidates, scores):
        entry = chunk.copy()
        entry["rerank_score"] = float(score)  # convert numpy float → Python float
        scored.append(entry)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    log.info(
        "Reranking complete. Top score: %.3f | Bottom score: %.3f",
        scored[0]["rerank_score"],
        scored[-1]["rerank_score"],
    )

    return scored[:n]
