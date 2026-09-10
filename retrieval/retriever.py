"""
retrieval/retriever.py
-----------------------
Dense (semantic) retrieval, BM25 (keyword) retrieval, and
Reciprocal Rank Fusion to combine both into a single ranked list.

THE RETRIEVAL ROADMAP:
  Step 4  → retrieve()          ← dense only (still available for baseline)
  Step 5  → retrieve_hybrid()   ← BM25 + dense + RRF  (THIS STEP)
  Step 6  → retrieve_reranked() ← hybrid + cross-encoder reranking (next step)

─────────────────────────────────────────────────────────────────────────────
WHY DOES PURE DENSE RETRIEVAL FAIL FOR SPECIFIC TERMS?

  Query: "what is the importance of using dropout"
  Dense embedding ≈ "significance of a regularisation technique"

  ChromaDB returns chunks about ANY kind of "importance" — because
  the 384-dim vector captures the gestalt, not individual keywords.
  The specific word "dropout" gets diluted by the phrase.

WHY DOES BM25 HELP?

  BM25 scores each chunk with:
    score(chunk, query) = Σ IDF(t) × [TF(t,chunk) × (k1+1)] / [TF(t,chunk) + k1×(1-b+b×|chunk|/avgdl)]

  Where:
    IDF(t)  = log((N - df + 0.5) / (df + 0.5))
               N  = total chunks in corpus
               df = chunks containing term t
    TF      = term frequency in this chunk
    k1=1.5  = term frequency saturation (diminishing returns for repeated terms)
    b=0.75  = length normalisation (penalises very long chunks slightly)
    avgdl   = average chunk length across corpus

  For "dropout": chunks that literally contain "dropout" many times
  score very high, regardless of what the surrounding context is about.
  This directly fixes the bug you observed.

WHY RECIPROCAL RANK FUSION (RRF)?

  We have two ranked lists — dense and BM25. How do we merge them?

  Simple approach: normalise and add scores. Problem: scores from
  different systems are on incompatible scales (cosine vs BM25).

  RRF (Cormack et al. 2009) solution:
    rrf_score(chunk) = Σ_ranker  1 / (k + rank_of_chunk_in_ranker)
    where k=60 is a smoothing constant (prevents top-1 results from
    dominating if one ranker is very confident).

  RRF is rank-based, not score-based. It's robust to scale differences
  and consistently outperforms most learned fusion methods on IR benchmarks.
  (This is why it's used in BEIR, TREC, and most production RAG systems.)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
from typing import List

from rank_bm25 import BM25Okapi

from ingestion.embedder import query_similar, get_all_chunks
from utils.config import config
from utils.exceptions import RetrievalError, RerankingError
from utils.logger import get_logger

log = get_logger(__name__)

# RRF smoothing constant — standard value from the 2009 paper.
# Higher k reduces the advantage of being ranked #1 vs #2.
_RRF_K = 60

# ── BM25 index cache ──────────────────────────────────────────────────────────
# We rebuild the BM25 index only when the corpus size changes.
# Rebuilding is O(n × avg_token_count) — fast enough to be unnoticeable
# (< 100ms for a 10,000-chunk corpus on CPU).
_bm25_index: BM25Okapi | None = None
_bm25_chunks: List[dict] = []      # parallel list — chunk at index i ↔ bm25_index row i
_bm25_corpus_size: int = 0         # chunk count at last build


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    """
    Lightweight tokeniser: lowercase, remove punctuation, split on whitespace.

    WHY NOT USE NLTK/spaCy?
      They require downloads and add latency. For BM25, simple whitespace
      tokenisation captures >95% of the benefit while remaining dependency-free.
      The main loss is stemming ("runs" ≠ "run"), which BM25Okapi doesn't
      do either, so the playing field is level.

    Args:
        text: Raw chunk or query text.

    Returns:
        List of lowercase tokens with punctuation stripped.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove punctuation
    return text.split()                         # split on any whitespace


# ── BM25 index management ─────────────────────────────────────────────────────

def _ensure_bm25_index() -> None:
    """
    Build (or rebuild) the BM25 index if the corpus has changed.

    Called at the start of every BM25 or hybrid search. The check is O(1)
    (just compare two integers) so it adds no meaningful latency.

    Side effects:
        Updates module-level _bm25_index, _bm25_chunks, _bm25_corpus_size.

    Raises:
        RetrievalError: If ChromaDB cannot be read.
    """
    global _bm25_index, _bm25_chunks, _bm25_corpus_size

    try:
        all_chunks = get_all_chunks()
    except Exception as exc:
        raise RetrievalError(f"Could not fetch chunks for BM25 index: {exc}") from exc

    if len(all_chunks) == _bm25_corpus_size and _bm25_index is not None:
        # Corpus hasn't changed — reuse cached index.
        return

    if not all_chunks:
        log.warning("BM25 index requested but corpus is empty.")
        _bm25_index = None
        _bm25_chunks = []
        _bm25_corpus_size = 0
        return

    log.info("Building BM25 index over %d chunks...", len(all_chunks))
    tokenised_corpus = [_tokenise(chunk["text"]) for chunk in all_chunks]
    _bm25_index = BM25Okapi(tokenised_corpus)
    _bm25_chunks = all_chunks
    _bm25_corpus_size = len(all_chunks)
    log.info("BM25 index built.")


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf_fuse(
    dense_results: List[dict],
    bm25_results: List[dict],
    top_k: int,
) -> List[dict]:
    """
    Merge two ranked result lists using Reciprocal Rank Fusion.

    ALGORITHM:
      1. For each chunk, compute rrf = Σ  1 / (k + rank_i)
         where rank_i is the 1-based rank in ranker i.
      2. Chunks that appear in only one ranker still get a contribution
         from that ranker (the other ranker contributes 0).
      3. Sort by descending RRF score and return top_k.

    Args:
        dense_results: Ordered list from dense (cosine) retrieval.
        bm25_results:  Ordered list from BM25 retrieval.
        top_k:         Number of final results to return.

    Returns:
        Merged, re-ranked list of result dicts with an added 'rrf_score' field.
    """
    scores: dict[str, float] = {}  # chunk_id → accumulated RRF score
    meta: dict[str, dict] = {}     # chunk_id → result dict

    for rank, result in enumerate(dense_results, start=1):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
        meta[cid] = result

    for rank, result in enumerate(bm25_results, start=1):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
        if cid not in meta:
            meta[cid] = result

    # Sort by RRF score descending, take top_k
    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]

    fused = []
    for cid in ranked_ids:
        entry = meta[cid].copy()
        entry["rrf_score"] = round(scores[cid], 6)
        fused.append(entry)

    return fused


# ── Public retrieval functions ────────────────────────────────────────────────

def retrieve(query: str, top_k: int | None = None) -> List[dict]:
    """
    Dense-only retrieval (cosine similarity via ChromaDB).

    Kept for:
      - Backward compatibility with Step 4 code.
      - Baseline comparison in Step 8 (RAGAS evaluation).

    Args:
        query:  The user's question.
        top_k:  Number of results. Defaults to config.top_k.

    Returns:
        List of result dicts sorted by cosine similarity (score: 0→1).

    Raises:
        RetrievalError: On any retrieval failure.
    """
    k = top_k if top_k is not None else config.top_k
    log.info("Dense retrieval: query='%s...', top_k=%d", query[:60], k)

    try:
        raw = query_similar(query, top_k=k)
    except Exception as exc:
        raise RetrievalError(f"Dense retrieval failed: {exc}") from exc

    results = _parse_chroma_results(raw)
    log.info("Dense retrieved %d chunks.", len(results))
    return results


def retrieve_bm25(query: str, top_k: int | None = None) -> List[dict]:
    """
    BM25 keyword retrieval against all stored chunks.

    Useful for debugging: call this alone to see what pure keyword
    search returns, before hybrid fusion.

    Args:
        query:  The user's question.
        top_k:  Number of results. Defaults to config.top_k.

    Returns:
        List of result dicts sorted by BM25 score (score: 0→∞, normalised).

    Raises:
        RetrievalError: If the BM25 index cannot be built.
    """
    k = top_k if top_k is not None else config.top_k
    log.info("BM25 retrieval: query='%s...', top_k=%d", query[:60], k)

    _ensure_bm25_index()
    if _bm25_index is None:
        raise RetrievalError("BM25 index is empty — no documents in the knowledge base.")

    query_tokens = _tokenise(query)
    raw_scores = _bm25_index.get_scores(query_tokens)  # ndarray, one score per chunk

    # Get top_k indices sorted by descending score.
    import numpy as np
    top_indices = np.argsort(raw_scores)[::-1][:k]
    max_score = float(raw_scores.max()) if raw_scores.max() > 0 else 1.0

    results = []
    for idx in top_indices:
        score = float(raw_scores[idx])
        if score <= 0:
            break  # BM25 score of 0 means no query term matched — stop
        chunk = _bm25_chunks[idx]
        results.append(
            {
                "text": chunk["text"],
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "score": round(score / max_score, 4),  # normalise to 0→1
            }
        )

    log.info("BM25 retrieved %d chunks.", len(results))
    return results


def retrieve_hybrid(query: str, top_k: int | None = None) -> List[dict]:
    """
    Hybrid retrieval: BM25 + dense search fused with Reciprocal Rank Fusion.

    This is the primary retrieval function used by the app from Step 5 onward.

    STRATEGY:
      1. Run dense retrieval for top_k × 2 candidates (wider net).
      2. Run BM25 retrieval for top_k × 2 candidates.
      3. Fuse with RRF and return top_k.

    WHY × 2 FOR CANDIDATES?
      RRF re-ranks after fusion. Fetching more candidates from each source
      gives the fusion step more material to work with, catching cases where
      a relevant chunk ranked #6 in one system but #1 in the other.

    Args:
        query:  The user's question.
        top_k:  Final number of results to return. Defaults to config.top_k.

    Returns:
        List of result dicts with both 'score' (dense) and 'rrf_score' fields,
        sorted by RRF score descending.

    Raises:
        RetrievalError: On any retrieval failure.
    """
    k = top_k if top_k is not None else config.top_k
    candidate_k = k * 2  # gather more candidates before fusion
    log.info("Hybrid retrieval: query='%s...', top_k=%d", query[:60], k)

    # Run both retrievers — failures in BM25 fall back to dense-only.
    dense_results = retrieve(query, top_k=candidate_k)

    try:
        bm25_results = retrieve_bm25(query, top_k=candidate_k)
    except RetrievalError as exc:
        log.warning("BM25 failed, falling back to dense-only: %s", exc)
        return dense_results[:k]

    fused = _rrf_fuse(dense_results, bm25_results, top_k=k)
    log.info("Hybrid retrieved %d chunks (RRF fused).", len(fused))
    return fused

def retrieve_reranked(
    query: str,
    top_k: int | None = None,
    top_n: int | None = None,
) -> List[dict]:
    """
    Full 3-stage retrieval: hybrid (BM25+dense) → cross-encoder reranking.

    This is the highest-quality retrieval function. Use it in production.
    For RAGAS baseline evaluation, compare against retrieve() (dense only).

    STAGE BREAKDOWN:
      Stage 1 — Hybrid BM25+dense with RRF: retrieves top_k × 3 candidates
                (wider net than hybrid alone so the reranker has more to work with)
      Stage 2 — Cross-encoder reranking: re-scores all candidates and
                returns only the top_n most relevant

    GRACEFUL DEGRADATION:
      If the reranker fails (network error, OOM, etc.), the function logs
      a warning and falls back to the hybrid results. The user still gets
      an answer — just with slightly less precise ranking.

    Args:
        query:  The user's question.
        top_k:  Candidates to fetch from hybrid retrieval.
                Defaults to config.top_k × 3 for a wider net.
        top_n:  Final results after reranking.
                Defaults to config.rerank_top_n (3).

    Returns:
        List of top_n result dicts, sorted by rerank_score descending.
        Each dict has: text, source_file, page_number, chunk_id,
                       score (cosine), rrf_score, rerank_score.

    Raises:
        RetrievalError: If the hybrid stage fails (no documents ingested).
    """
    from retrieval.reranker import rerank  # local import avoids circular deps

    k = top_k if top_k is not None else (config.top_k * 3)
    n = top_n if top_n is not None else config.rerank_top_n

    log.info(
        "Reranked retrieval: query='%s...', candidates=%d, top_n=%d",
        query[:60], k, n,
    )

    # Stage 1: Hybrid retrieval — gather candidates
    candidates = retrieve_hybrid(query, top_k=k)
    if not candidates:
        return []

    # Stage 2: Cross-encoder reranking — precision pass
    try:
        reranked = rerank(query, candidates, top_n=n)
    except RerankingError as exc:
        # Graceful fallback: reranking failed but hybrid results are still good
        log.warning(
            "Cross-encoder reranking failed, falling back to hybrid results: %s", exc
        )
        return candidates[:n]

    return reranked


def _parse_chroma_results(raw: dict) -> List[dict]:
    """
    Convert ChromaDB's nested-list result format into a flat list of dicts.

    ChromaDB returns { 'documents': [[...]], 'distances': [[...]], ... }
    where the outer list index = query index (we always send one query).

    distance → score: ChromaDB cosine distance = 1 - cosine_similarity.
    So score = 1 - distance (higher score = more similar).
    """
    documents = raw.get("documents", [[]])[0]
    metadatas  = raw.get("metadatas",  [[]])[0]
    distances  = raw.get("distances",  [[]])[0]

    results: List[dict] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        results.append(
            {
                "text":        doc,
                "source_file": meta.get("source_file", "unknown"),
                "page_number": meta.get("page_number", 0),
                "chunk_id":    meta.get("chunk_id", ""),
                "score":       round(max(0.0, 1.0 - dist), 4),
            }
        )
    return results
