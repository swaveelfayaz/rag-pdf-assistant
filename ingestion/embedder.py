from __future__ import annotations  # defers annotation evaluation — fixes chromadb.PersistentClient | None

"""
ingestion/embedder.py
---------------------
Embed text chunks and store them in a ChromaDB vector store.

THE PIPELINE THIS MODULE SITS IN:
  PDF → loader.py → [pages] → chunker.py → [Chunks] → embedder.py → ChromaDB

WHAT IS AN EMBEDDING?
  An embedding is a fixed-length list of floating-point numbers that
  encodes the *meaning* of a piece of text. Texts with similar meanings
  produce vectors that are close together in the 384-dimensional space
  that all-MiniLM-L6-v2 was trained on.

  Example (simplified to 3 dimensions for illustration):
    "The cat sat on the mat"     → [0.12, -0.45, 0.83]
    "A feline rested on the rug" → [0.11, -0.44, 0.81]  ← very similar
    "The stock market crashed"   → [-0.72, 0.31, -0.19] ← very different

WHY COSINE SIMILARITY?
  We measure similarity as the cosine of the angle between two vectors.
  Cosine similarity = (A · B) / (||A|| × ||B||)
  It ranges from -1 (opposite) to 1 (identical direction).

  The key advantage over Euclidean distance: cosine similarity is
  scale-invariant. A short chunk and a long chunk about the same topic
  will both point in the same "direction" and score high, even if their
  magnitudes differ. Euclidean distance would unfairly penalise the
  shorter chunk for having a smaller magnitude.

  ChromaDB uses cosine similarity by default — we don't need to configure
  this explicitly, but it's important to know for interviews.

MODEL CHOICE — all-MiniLM-L6-v2:
  - 22 million parameters (vs. 110M for all-mpnet-base-v2)
  - 384-dimensional embeddings (vs. 768 for larger models)
  - ~80 MB on disk (vs. ~420 MB)
  - Runs a 500-chunk document in ~5 seconds on CPU
  - Quality is ~2-5% lower than all-mpnet on MTEB benchmark
  TRADEOFF: We accept a small quality loss for a 5× speed gain.
  For a demo/learning project, this is exactly the right call.

BATCH PROCESSING:
  We embed in batches of `batch_size` chunks (default 32) rather than
  one at a time. Sentence-transformers can vectorise a batch in one
  forward pass through the model — roughly the same time as one chunk.
  Batching gives us ~10× throughput improvement for free.

UPSERT vs INSERT:
  We use ChromaDB's `upsert` (not `add`) so that re-uploading the same
  PDF is safe — it overwrites existing chunks instead of creating
  duplicates. Without upsert, uploading twice would double all chunks
  and corrupt retrieval results.
"""

from pathlib import Path
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from ingestion.chunker import Chunk
from utils.config import config
from utils.exceptions import EmbeddingError, VectorStoreError
from utils.logger import get_logger

log = get_logger(__name__)

# ── Module-level singletons ───────────────────────────────────────────────────
# We keep one model instance and one ChromaDB client alive for the lifetime
# of the process. Creating these objects is expensive (model loading, disk
# I/O) — recreating them per call would be ~2-5 seconds of overhead each time.
#
# WHY MODULE-LEVEL? In Streamlit, the app script re-runs on every user
# interaction. Module-level objects are NOT recreated on re-runs — Python
# caches imported modules. This is the correct pattern for heavy singletons.

_model: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    """
    Lazily load the sentence-transformer model on first call.

    Lazy loading (instead of loading at import time) means:
      - Tests that don't test embedding don't pay the 2-second load cost.
      - The app shows its UI immediately; the model loads when first needed.

    Returns:
        The loaded SentenceTransformer model, ready to encode text.

    Raises:
        EmbeddingError: If the model cannot be loaded.
    """
    global _model
    if _model is None:
        log.info("Loading embedding model '%s' (first call)...", config.embedding_model)
        try:
            _model = SentenceTransformer(config.embedding_model)
            log.info("Embedding model loaded.")
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{config.embedding_model}'. "
                "Check your internet connection (first run downloads ~80 MB) "
                f"or set a different EMBEDDING_MODEL in .env. Error: {exc}"
            ) from exc
    return _model


def _get_collection() -> chromadb.Collection:
    """
    Lazily initialise the ChromaDB client and collection on first call.

    WHY PersistentClient?
      We use `chromadb.PersistentClient(path=...)` instead of the in-memory
      `chromadb.Client()`. The persistent client writes the vector store to
      disk at CHROMA_PERSIST_DIR so that:
        1. Documents survive app restarts (you don't re-embed on every run).
        2. We can embed once and query thousands of times for free.

    WHY get_or_create_collection?
      `get_or_create_collection` is idempotent — safe to call whether the
      collection exists or not. Using `create_collection` would crash if
      the collection already exists (e.g., after re-uploading a document).

    Returns:
        A ChromaDB Collection object, ready for upsert and query.

    Raises:
        VectorStoreError: If ChromaDB cannot be initialised.
    """
    global _chroma_client, _collection
    if _collection is None:
        persist_dir = Path(config.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        log.info("Connecting to ChromaDB at '%s'.", persist_dir)
        try:
            _chroma_client = chromadb.PersistentClient(path=str(persist_dir))
            _collection = _chroma_client.get_or_create_collection(
                name=config.chroma_collection_name,
                # Cosine similarity is the correct metric for sentence-transformer
                # embeddings. ChromaDB normalises vectors internally when you
                # use cosine, so query results are ordered from most to least similar.
                metadata={"hnsw:space": "cosine"},
            )
            log.info(
                "Collection '%s' ready (%d documents already stored).",
                config.chroma_collection_name,
                _collection.count(),
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to initialise ChromaDB at '{persist_dir}': {exc}"
            ) from exc
    return _collection


def embed_and_store(chunks: List[Chunk], batch_size: int = 32) -> int:
    """
    Convert chunks to embeddings and upsert them into ChromaDB.

    PROCESS:
      1. Extract text from each Chunk.
      2. Batch encode with the sentence-transformer model.
      3. Upsert (text, embedding, metadata) into ChromaDB.

    WHY UPSERT?
      `collection.upsert()` inserts new items OR updates existing ones
      (matched by chunk_id). This makes re-uploading the same PDF safe —
      it overwrites rather than duplicates. `collection.add()` would
      raise an error on duplicate IDs.

    Args:
        chunks:     List of Chunk objects from chunk_documents().
        batch_size: Number of chunks to embed in one model forward pass.
                    32 is a safe default for CPU — large enough for
                    throughput, small enough to avoid OOM errors.

    Returns:
        Number of chunks successfully upserted.

    Raises:
        EmbeddingError:   If the model fails to encode any batch.
        VectorStoreError: If ChromaDB fails to store any batch.
    """
    if not chunks:
        log.warning("embed_and_store called with empty chunk list — nothing to do.")
        return 0

    model = _get_model()
    collection = _get_collection()

    log.info("Embedding %d chunks in batches of %d...", len(chunks), batch_size)
    total_upserted = 0

    # Process chunks in batches so we don't load all embeddings into RAM at once.
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        # ── 1. Encode the batch ───────────────────────────────────────────────
        texts = [chunk.text for chunk in batch]
        try:
            # show_progress_bar=False: suppress tqdm output in Streamlit logs.
            # convert_to_list=True: returns Python floats, not numpy arrays —
            # ChromaDB's Python client expects plain Python lists.
            embeddings = model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,  # numpy → list conversion below
            ).tolist()
        except Exception as exc:
            raise EmbeddingError(
                f"Model failed to encode batch starting at index {batch_start}: {exc}"
            ) from exc

        # ── 2. Build metadata dicts for ChromaDB ─────────────────────────────
        # ChromaDB's metadata must be a dict of {str: str|int|float|bool}.
        # We store source_file and page_number so retrieval can return them
        # alongside the text, enabling source citations in the UI.
        metadatas = [
            {
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in batch
        ]

        # ── 3. Upsert into ChromaDB ───────────────────────────────────────────
        try:
            collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=embeddings,
                documents=texts,       # stores raw text alongside the vector
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"ChromaDB upsert failed for batch at index {batch_start}: {exc}"
            ) from exc

        total_upserted += len(batch)
        log.debug("Upserted batch %d/%d.", batch_start // batch_size + 1,
                  (len(chunks) + batch_size - 1) // batch_size)

    log.info("Done — %d chunks embedded and stored.", total_upserted)
    return total_upserted


def query_similar(query_text: str, top_k: int | None = None) -> dict:
    """
    Embed a query and retrieve the top-k most similar chunks from ChromaDB.

    This is the "dense retrieval" step — we convert the user's question
    into a vector and find the stored chunks whose vectors are most similar.

    Args:
        query_text: The user's question or search query.
        top_k:      Number of results to return (defaults to config.top_k).

    Returns:
        A ChromaDB query result dict with keys:
          'ids'        : List of chunk IDs (outer list = one per query)
          'documents'  : List of chunk texts
          'metadatas'  : List of metadata dicts (source_file, page_number)
          'distances'  : List of cosine distances (0=identical, 2=opposite)

    Raises:
        VectorStoreError: If ChromaDB is empty or the query fails.
        EmbeddingError:   If the query text cannot be embedded.
    """
    k = top_k if top_k is not None else config.top_k
    collection = _get_collection()

    if collection.count() == 0:
        raise VectorStoreError(
            "The vector store is empty. Please upload and process at least "
            "one PDF before searching."
        )

    # Embed the query using the same model used for documents.
    # This is critical — using a different model for queries vs. documents
    # would produce incompatible vector spaces and return garbage results.
    try:
        query_embedding = _get_model().encode(
            query_text, show_progress_bar=False, convert_to_numpy=True
        ).tolist()
    except Exception as exc:
        raise EmbeddingError(f"Failed to embed query text: {exc}") from exc

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection.count()),  # can't ask for more than we have
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise VectorStoreError(f"ChromaDB query failed: {exc}") from exc

    return results


def get_collection_count() -> int:
    """
    Return the number of chunks currently stored in ChromaDB.

    Useful for health checks in the UI and tests.

    Returns:
        Integer count of stored chunks.

    Raises:
        VectorStoreError: If ChromaDB cannot be reached.
    """
    try:
        return _get_collection().count()
    except Exception as exc:
        raise VectorStoreError(f"Could not read collection count: {exc}") from exc


def reset_collection() -> None:
    """
    Delete all documents from the ChromaDB collection.

    Used in tests to guarantee a clean state. In production, this
    would be exposed via a UI button ("Clear all documents").

    WARNING: This is irreversible — all embeddings are deleted from disk.
    Re-embedding requires re-uploading all PDFs.

    Raises:
        VectorStoreError: If the reset operation fails.
    """
    global _collection
    if _chroma_client is None:
        # Nothing initialised yet — nothing to reset.
        return
    try:
        _chroma_client.delete_collection(config.chroma_collection_name)
        # Re-create so _collection is not None going forward.
        _collection = _chroma_client.get_or_create_collection(
            name=config.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("Collection '%s' reset.", config.chroma_collection_name)
    except Exception as exc:
        raise VectorStoreError(f"Failed to reset collection: {exc}") from exc


def get_all_chunks() -> list[dict]:
    """
    Fetch every chunk stored in ChromaDB as a flat list of dicts.

    WHY THIS IS NEEDED FOR BM25:
      BM25 is a corpus-level algorithm — it needs to see ALL documents
      to compute IDF (Inverse Document Frequency) scores correctly.
      IDF = log(N / df) where N is total docs and df is how many docs
      contain the term. Without all documents, IDF scores are wrong.

      This is called once when the BM25 index is first built, then
      cached in retriever.py. The index is rebuilt only when the chunk
      count changes (i.e., new PDFs were uploaded).

    Returns:
        List of dicts, each with: text, source_file, page_number, chunk_id.
        Empty list if the collection is empty.

    Raises:
        VectorStoreError: If ChromaDB cannot be read.
    """
    try:
        collection = _get_collection()
        count = collection.count()
        if count == 0:
            return []

        # ChromaDB's .get() returns all items (up to limit).
        # We don't need embeddings here — just text and metadata.
        results = collection.get(
            limit=count,
            include=["documents", "metadatas"],
        )

        chunks = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            chunks.append(
                {
                    "text": doc,
                    "source_file": meta.get("source_file", ""),
                    "page_number": meta.get("page_number", 0),
                    "chunk_id": meta.get("chunk_id", ""),
                }
            )
        log.debug("Fetched %d chunks from ChromaDB for BM25 indexing.", len(chunks))
        return chunks

    except Exception as exc:
        raise VectorStoreError(f"Failed to fetch all chunks: {exc}") from exc

