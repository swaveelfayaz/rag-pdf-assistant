"""
utils/config.py
---------------
Centralized, validated configuration for the RAG pipeline.

DESIGN PATTERN — "Fail Fast":
  All configuration is loaded and validated *at import time*. If a
  required value is missing or invalid, we raise a clear ConfigError
  immediately — before any pipeline code runs.

  The alternative (lazy validation) means your app starts fine but
  crashes 10 minutes in when the user tries to submit a query. Fail-fast
  is strictly better for developer experience and user experience.

HOW IT WORKS:
  1. `python-dotenv` reads .env from the project root into os.environ.
  2. The Config dataclass reads each value with a sensible default.
  3. Config.__post_init__ validates the values and raises early if
     anything is wrong.
  4. A module-level singleton `config` is exported — every other module
     does `from utils.config import config` and reads attributes.

ADDING NEW CONFIG:
  1. Add a field to the Config dataclass with its default.
  2. Add validation logic in __post_init__ if needed.
  3. Add the key to .env.example with a comment.
  Never hardcode values anywhere else in the codebase.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ── Locate and load .env ─────────────────────────────────────────────────────

# Walk up from this file (utils/) to the project root, then load .env.
# `override=False` means environment variables already set in the shell
# take precedence — important for CI/CD and Docker deployments.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


# ── Config Dataclass ─────────────────────────────────────────────────────────

@dataclass
class Config:
    """
    Single source of truth for all tuneable parameters.

    Attributes are grouped by pipeline stage so it's easy to find the
    knobs relevant to whichever part of the code you're working on.
    """

    # ── LLM / API ────────────────────────────────────────────────────────────
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    # Model must be on Groq's free tier. As of 2025, llama-3.3-70b-versatile
    # is the best freely available model on Groq. Check groq.com/docs for
    # current availability — free-tier models change frequently.
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    # ollama_model is the fallback for fully offline usage.
    # Run `ollama pull llama3.2` locally to enable this.
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2")
    )
    # Which backend to use: "groq" or "ollama"
    llm_backend: str = field(
        default_factory=lambda: os.getenv("LLM_BACKEND", "groq")
    )

    # ── Embedding Model ───────────────────────────────────────────────────────
    # all-MiniLM-L6-v2: 22M params, 384-dim vectors, ~80 MB on disk.
    # TRADEOFF: Much faster than larger models (e.g. all-mpnet-base-v2)
    # but slightly lower quality. For a learning project the speed win
    # matters more than marginal quality gains.
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
    )

    # ── Chunking Parameters ───────────────────────────────────────────────────
    # chunk_size=800 chars ≈ 150-200 tokens. Large enough for context but
    # small enough that a single chunk doesn't dominate an LLM's context.
    # chunk_overlap=150 prevents answers from being cut off at chunk
    # boundaries — the same sentence might appear at the end of chunk N
    # and the start of chunk N+1, so retrieval still finds it.
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150"))
    )

    # ── Vector Store ──────────────────────────────────────────────────────────
    # ChromaDB persists to disk at this path. Using a relative path here
    # means the DB lives next to the project, which is fine for dev/demo.
    # In production you'd point this at a mounted volume.
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv(
            "CHROMA_PERSIST_DIR",
            str(_PROJECT_ROOT / "chroma_db")
        )
    )
    chroma_collection_name: str = field(
        default_factory=lambda: os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
    )

    # ── Retrieval Parameters ──────────────────────────────────────────────────
    # top_k: number of chunks returned by dense search.
    # WHY 5? Empirically, 3 is too few (misses nuanced queries) and 10 is
    # too many (dilutes the context, LLM loses focus). 5 is the standard
    # starting point in most RAG papers.
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "5"))
    )
    # bm25_top_k: how many candidates BM25 retrieves before fusion.
    # Deliberately larger than top_k so we have room to rerank.
    bm25_top_k: int = field(
        default_factory=lambda: int(os.getenv("BM25_TOP_K", "10"))
    )
    # rerank_top_n: final number of chunks after reranking.
    rerank_top_n: int = field(
        default_factory=lambda: int(os.getenv("RERANK_TOP_N", "3"))
    )
    # Cross-encoder model for reranking — trained on MS MARCO passage ranking.
    # ms-marco-MiniLM-L-6-v2: ~85 MB, runs on CPU, excellent quality.
    # Larger option: cross-encoder/ms-marco-electra-base (~440 MB, slower, better)
    rerank_model: str = field(
        default_factory=lambda: os.getenv(
            "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )

    # ── Generation ────────────────────────────────────────────────────────────
    # Temperature=0 means fully deterministic — same question always gets
    # the same answer. This is what you want for a RAG assistant where
    # consistency and faithfulness matter more than creativity.
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS", "1024"))
    )

    def __post_init__(self) -> None:
        """
        Validate all config values immediately after construction.

        We raise ConfigError (not ValueError) so that callers can catch
        it specifically without accidentally catching unrelated errors.
        """
        valid_backends = {"groq", "ollama"}
        if self.llm_backend not in valid_backends:
            raise ConfigError(
                f"LLM_BACKEND must be one of {valid_backends}, "
                f"got '{self.llm_backend}'"
            )

        # Only require the API key if we're actually using Groq.
        # This lets the project run fully offline with Ollama without
        # needing to set a dummy key.
        if self.llm_backend == "groq" and not self.groq_api_key:
            raise ConfigError(
                "GROQ_API_KEY is required when LLM_BACKEND='groq'. "
                "Get a free key at https://console.groq.com, then add it "
                "to your .env file."
            )

        if self.chunk_size <= 0:
            raise ConfigError(
                f"CHUNK_SIZE must be a positive integer, got {self.chunk_size}"
            )

        if self.chunk_overlap < 0:
            raise ConfigError(
                f"CHUNK_OVERLAP must be non-negative, got {self.chunk_overlap}"
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be less than "
                f"CHUNK_SIZE ({self.chunk_size}) — otherwise chunks have "
                f"no unique content."
            )

        if self.top_k <= 0:
            raise ConfigError(
                f"TOP_K must be a positive integer, got {self.top_k}"
            )

        if self.rerank_top_n <= 0:
            raise ConfigError(
                f"RERANK_TOP_N must be a positive integer, got {self.rerank_top_n}"
            )


# ── Module-level singleton ────────────────────────────────────────────────────
# Instantiated once at import time. Any import error means the config is
# broken, which is exactly what we want (fail fast).
# Tests that need to override values should monkeypatch os.environ BEFORE
# importing this module, or use the Config() constructor directly.
try:
    config = Config()
except ConfigError:
    # Re-raise so the import fails with the original message.
    raise
