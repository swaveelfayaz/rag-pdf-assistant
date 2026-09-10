"""
ingestion/chunker.py
--------------------
Text splitting into overlapping chunks with metadata preservation.

WHY CHUNK AT ALL?
  A 200-page research paper might be 500,000 characters. The LLM's
  context window is 4,000–32,000 tokens. We cannot send the whole
  document. Instead, we:
    1. Split the document into small, manageable chunks.
    2. Embed each chunk as a vector.
    3. At query time, retrieve only the top-K most relevant chunks.
    4. Send only those chunks to the LLM.
  This is the fundamental trick that makes RAG scalable.

WHY RECURSIVE CHARACTER SPLITTING?
  The algorithm tries a cascade of separators in order:
  ["\n\n", "\n", " ", ""] — i.e., it prefers to split at paragraph
  breaks, then at newlines, then at spaces, and only as a last resort
  splits mid-word. This preserves semantic coherence much better than
  naive fixed-width splitting.

  DEPENDENCY NOTE: We implement this directly in pure Python rather than
  using langchain-text-splitters, because langchain-core triggers
  LangSmith telemetry initialization on import, which can cause network
  hangs in offline/restricted environments. Our implementation is ~60
  lines and matches langchain's algorithm exactly.

WHY CHUNK_SIZE=800 chars / CHUNK_OVERLAP=150 chars?
  800 chars ≈ 150–200 tokens (rough rule: 1 token ≈ 4 chars).
  - Large enough to capture a complete thought or paragraph.
  - Small enough that a chunk doesn't dominate the LLM context.
  - 150-char overlap ensures answers at chunk boundaries aren't lost.
  These are defaults from config.py and can be tuned via .env.

METADATA ATTACHED TO EVERY CHUNK:
  source_file : str   — original PDF filename
  page_number : int   — which page the chunk originated from
  chunk_id    : str   — globally unique ID: "{filename}::p{page}::c{n}"
  
  This metadata travels with the chunk through embedding and retrieval,
  so the final answer can say "Source: report.pdf, page 12".
"""

from dataclasses import dataclass
from typing import List

from utils.config import config
from utils.exceptions import ChunkingError
from utils.logger import get_logger

log = get_logger(__name__)

# Separator cascade (priority order):
# Try paragraph breaks first, then line breaks, then spaces, then chars.
_SEPARATORS = ["\n\n", "\n", " ", ""]


@dataclass
class Chunk:
    """
    A single piece of text with its origin metadata.

    Using a dataclass (instead of a plain dict) gives us:
      - Type hints on every field
      - A readable __repr__ for debugging
      - Dot-attribute access (chunk.text, not chunk["text"])
    """
    text: str
    source_file: str
    page_number: int
    chunk_id: str


def _merge_splits(splits: List[str], separator: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Merge a list of small text pieces into chunks of at most `chunk_size`
    characters, repeating `chunk_overlap` characters between consecutive
    chunks so that context at boundaries is not lost.

    Args:
        splits:        Small text pieces to merge.
        separator:     The separator that was used to produce these splits
                       (needed to re-join them correctly).
        chunk_size:    Maximum character length per output chunk.
        chunk_overlap: Characters to repeat at the start of each new chunk.

    Returns:
        List of merged text chunks.
    """
    chunks: List[str] = []
    current_pieces: List[str] = []
    current_len = 0
    sep_len = len(separator)

    for piece in splits:
        piece_len = len(piece)
        # +sep_len accounts for the separator we'd insert between pieces.
        addition = sep_len + piece_len if current_pieces else piece_len

        if current_len + addition > chunk_size and current_pieces:
            # Flush the current buffer as a chunk.
            chunk_text = separator.join(current_pieces).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Build overlap: keep trailing pieces that fit within chunk_overlap.
            overlap_pieces: List[str] = []
            overlap_len = 0
            for p in reversed(current_pieces):
                p_addition = sep_len + len(p) if overlap_pieces else len(p)
                if overlap_len + p_addition <= chunk_overlap:
                    overlap_pieces.insert(0, p)
                    overlap_len += p_addition
                else:
                    break

            current_pieces = overlap_pieces
            current_len = overlap_len

        current_pieces.append(piece)
        current_len += sep_len + piece_len if len(current_pieces) > 1 else piece_len

    # Flush any remaining pieces.
    if current_pieces:
        chunk_text = separator.join(current_pieces).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def _recursive_split(text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Pure-Python recursive character text splitter.

    Mirrors the algorithm used by LangChain's RecursiveCharacterTextSplitter:
      1. Pick the first separator that appears in the text.
      2. Split the text on that separator into pieces.
      3. Recursively split any piece that is still too large, using the
         next separator in the cascade.
      4. Merge pieces into final chunks with overlap.

    Args:
        text:         The text to split.
        separators:   Ordered list of separator strings to try.
        chunk_size:   Maximum character length of each chunk.
        chunk_overlap: Characters to repeat at chunk boundaries.

    Returns:
        List of text chunks, each <= chunk_size characters.
    """
    if not text.strip():
        return []

    # Base case: already fits.
    if len(text) <= chunk_size:
        return [text.strip()]

    # Find the first separator that appears in the text.
    chosen_sep = ""
    remaining_seps: List[str] = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            chosen_sep = sep
            remaining_seps = separators[i + 1:]
            break

    # Split on the chosen separator.
    if chosen_sep:
        raw_splits = text.split(chosen_sep)
    else:
        # Character-level fallback: slice directly.
        result = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            result.append(text[i:i + chunk_size])
        return [s.strip() for s in result if s.strip()]

    # Recursively handle pieces that are still too large, collect good ones.
    good_splits: List[str] = []
    all_chunks: List[str] = []

    for piece in raw_splits:
        if not piece.strip():
            continue
        if len(piece) <= chunk_size:
            good_splits.append(piece)
        else:
            # Flush accumulated good splits first.
            if good_splits:
                all_chunks.extend(_merge_splits(good_splits, chosen_sep, chunk_size, chunk_overlap))
                good_splits = []
            # Recurse with the next separator.
            if remaining_seps:
                all_chunks.extend(_recursive_split(piece, remaining_seps, chunk_size, chunk_overlap))
            else:
                all_chunks.append(piece[:chunk_size].strip())

    if good_splits:
        all_chunks.extend(_merge_splits(good_splits, chosen_sep, chunk_size, chunk_overlap))

    return [c for c in all_chunks if c.strip()]


def chunk_documents(
    pages: List[tuple[int, str]],
    source_file: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Chunk]:
    """
    Split page texts into overlapping chunks and attach metadata.

    Args:
        pages:         List of (page_number, text) tuples from loader.py.
        source_file:   The original PDF filename (used in citations).
        chunk_size:    Override the default from config (useful for tests).
        chunk_overlap: Override the default from config.

    Returns:
        A flat list of Chunk objects, ordered by page then position.

    Raises:
        ChunkingError: If the splitter fails or `pages` is empty.
        ValueError:    If chunk_overlap >= chunk_size.
    """
    if not pages:
        raise ChunkingError(
            "Cannot chunk an empty document. "
            "Make sure load_pdf() returned at least one page with text."
        )

    # Use config defaults unless explicit overrides are provided.
    # This pattern (param or config.default) lets tests pass small values
    # without modifying the global config object.
    size = chunk_size if chunk_size is not None else config.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap

    if overlap >= size:
        raise ValueError(
            f"chunk_overlap ({overlap}) must be less than chunk_size ({size}). "
            "Otherwise chunks have no unique content — the overlap would cover "
            "the entire chunk."
        )

    log.info(
        "Chunking '%s': %d pages, chunk_size=%d, chunk_overlap=%d",
        source_file,
        len(pages),
        size,
        overlap,
    )

    # ── Process each page ─────────────────────────────────────────────────────
    all_chunks: List[Chunk] = []
    chunk_counter = 0  # global counter across all pages for unique chunk IDs

    for page_number, page_text in pages:
        try:
            raw_chunks = _recursive_split(page_text, _SEPARATORS, size, overlap)
        except Exception as exc:
            raise ChunkingError(
                f"Failed to split text on page {page_number} of '{source_file}': {exc}"
            ) from exc

        for raw_chunk in raw_chunks:
            stripped = raw_chunk.strip()
            if not stripped:
                # Skip chunks that are only whitespace — can happen when a
                # page has long runs of blank lines between sections.
                continue

            # Chunk ID format: "report.pdf::p3::c17"
            # Using "::" as separator avoids conflicts with typical filenames.
            chunk_id = f"{source_file}::p{page_number}::c{chunk_counter}"
            chunk_counter += 1

            all_chunks.append(
                Chunk(
                    text=stripped,
                    source_file=source_file,
                    page_number=page_number,
                    chunk_id=chunk_id,
                )
            )

    log.info(
        "Chunked '%s' into %d chunks.",
        source_file,
        len(all_chunks),
    )

    if not all_chunks:
        raise ChunkingError(
            f"Chunking produced zero chunks for '{source_file}'. "
            "The document may contain only whitespace."
        )

    return all_chunks
