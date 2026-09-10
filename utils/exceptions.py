"""
utils/exceptions.py
-------------------
Custom exception hierarchy for the RAG pipeline.

WHY A CUSTOM HIERARCHY?
  Python's built-in exceptions (ValueError, IOError, etc.) are generic.
  When something goes wrong deep inside a pipeline, you lose context about
  *which stage* failed. By defining our own tree of exceptions:

    RagBaseError
      ├── DocumentLoadError       ← problem reading/parsing the PDF
      │     └── EmptyDocumentError  ← PDF parsed but had no text
      ├── ChunkingError           ← problem splitting text into chunks
      ├── EmbeddingError          ← problem converting text to vectors
      ├── VectorStoreError        ← problem persisting or querying ChromaDB
      ├── RetrievalError          ← problem during search / ranking
      └── GenerationError         ← problem calling the LLM

  Callers can catch at ANY level of specificity:
      except DocumentLoadError:   # only PDF problems
      except RagBaseError:        # any pipeline problem, nothing else

  This lets the Streamlit UI show the user exactly what went wrong
  ("Could not read that PDF — is it encrypted?") instead of a traceback.
"""


class RagBaseError(Exception):
    """
    Root of the custom exception tree.

    Every domain-specific exception in this project inherits from here
    so that a single broad catch (`except RagBaseError`) is still safe —
    it will never accidentally swallow Python's own SystemExit or
    KeyboardInterrupt.
    """


# ── Document Ingestion Layer ─────────────────────────────────────────────────

class DocumentLoadError(RagBaseError):
    """
    Raised when a PDF cannot be opened or parsed.

    Typical causes:
      - File path does not exist
      - File is corrupted or not a valid PDF
      - File is encrypted / password-protected
      - PyMuPDF raises an internal error
    """


class EmptyDocumentError(DocumentLoadError):
    """
    Raised when a PDF opens successfully but yields zero text.

    This is a *subclass* of DocumentLoadError so callers that catch
    DocumentLoadError will also catch this, but callers that want to
    handle "scanned / image-only PDF" as a special case can catch
    EmptyDocumentError specifically.

    Typical causes:
      - Scanned PDF (image-only, no embedded text layer)
      - PDF with only form fields or vector graphics
    """


class ChunkingError(RagBaseError):
    """
    Raised when text splitting fails unexpectedly.

    Practically this should be rare — the chunker has very few failure
    modes — but we define it so that tests and callers can assert on it
    rather than catching a generic exception.
    """


# ── Embedding & Vector Store Layer ──────────────────────────────────────────

class EmbeddingError(RagBaseError):
    """
    Raised when the sentence-transformer model fails to encode text.

    Typical causes:
      - Model files not downloaded (network error during first run)
      - Input text exceeds model's maximum token length
      - Out-of-memory error on constrained hardware
    """


class VectorStoreError(RagBaseError):
    """
    Raised when ChromaDB operations fail.

    Covers both write operations (adding documents) and read operations
    (querying the collection), because from the caller's perspective
    either failure means "the vector store is unavailable".

    Typical causes:
      - Disk full (cannot persist new embeddings)
      - Collection does not exist yet at query time
      - ChromaDB version mismatch / corrupt persistence directory
    """


# ── Retrieval Layer ──────────────────────────────────────────────────────────

class RetrievalError(RagBaseError):
    """
    Raised when the retrieval pipeline (dense search, BM25, reranking)
    fails to return results.

    Typical causes:
      - Vector store is empty (no documents ingested)
      - BM25 index not initialized before querying
      - Reranker model unavailable
    """


class RerankingError(RagBaseError):
    """
    Raised when the cross-encoder reranker fails.

    Kept separate from RetrievalError so the UI can tell the user:
    "Search worked, but the precision ranking step failed" — allowing
    a graceful fallback to the un-reranked hybrid results.

    Typical causes:
      - Cross-encoder model download failed (first run, no internet)
      - Input pair exceeds max_length and truncation caused issues
      - Out-of-memory error on very constrained hardware
    """


# ── Generation Layer ─────────────────────────────────────────────────────────

class GenerationError(RagBaseError):
    """
    Raised when the LLM call fails.

    Typical causes:
      - Groq API key missing or invalid
      - Groq rate limit exceeded (free tier)
      - Network timeout
      - Unexpected response schema from the API
    """
