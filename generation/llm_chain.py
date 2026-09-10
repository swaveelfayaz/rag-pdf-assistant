"""
generation/llm_chain.py
------------------------
Grounded LLM generation with structured citation output.

WHAT "GROUNDED" MEANS:
  A grounded answer is constrained to only use information explicitly
  present in the provided document excerpts. The LLM cannot add facts
  from its training data ("hallucinate"). This is enforced through:
    1. A strict system prompt with clear, numbered rules.
    2. Numbered excerpt headers so the model can cite precisely.
    3. Citation parsing to verify which excerpts were actually used.
    4. An is_grounded flag that detects "I don't know" responses.

WHY STRUCTURED OUTPUT (GroundedAnswer dataclass)?
  Returning a plain string loses information:
    - Which specific excerpts did the model actually cite?
    - Did the model answer or say "I don't know"?
    - Which source files and pages should be highlighted in the UI?

  GroundedAnswer packages all this so the UI can render:
    - The answer text
    - Expandable citation cards (exact chunk + source + page)
    - A "Grounded" / "Not Found" status badge

CITATION FORMAT USED:
  Excerpts are numbered [1], [2], ... in the context.
  The model is instructed to cite as (Excerpt 1), (Excerpt 2), etc.
  We parse these with a regex and map back to the source chunk dicts.

  Why numbered references instead of inline source names?
  Because "According to DL-UNIT-2.pdf page 47" is harder for the model
  to generate consistently than "According to Excerpt 3". The numbered
  system produces more reliable citations that our regex can parse.

GROQ FREE TIER NOTE:
  All generation uses llama-3.3-70b-versatile on Groq's free tier.
  Rate limits (as of 2025): 6,000 tokens/minute, 500 requests/day.
  If you hit rate limits, wait 60s or switch to a smaller model like
  llama-3.1-8b-instant in your .env file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from groq import Groq

from utils.config import config
from utils.exceptions import GenerationError
from utils.logger import get_logger

log = get_logger(__name__)


# ── Output Type ───────────────────────────────────────────────────────────────

@dataclass
class GroundedAnswer:
    """
    Structured output from generate_answer().

    Using a dataclass instead of a plain string or dict gives:
      - Type safety (mypy/pyright can catch misuse)
      - Self-documenting fields
      - Clean attribute access in app.py (ans.answer, ans.citations)

    Attributes:
        answer:      The LLM's generated answer text (may be "I don't know...").
        citations:   List of chunk dicts that were explicitly cited in the answer.
                     Each dict has: text, source_file, page_number, chunk_id.
                     Empty if the model said "I don't know".
        is_grounded: True if answer is based on context. False if the model
                     indicated the information was not found in the documents.
        raw_context: All context chunks that were sent to the model.
                     Kept for debugging and RAGAS evaluation (Step 8).
    """
    answer: str
    citations: List[dict] = field(default_factory=list)
    is_grounded: bool = True
    raw_context: List[dict] = field(default_factory=list)


# ── Client singleton ──────────────────────────────────────────────────────────

_client: Groq | None = None


def _get_client() -> Groq:
    """Lazily initialise the Groq client."""
    global _client
    if _client is None:
        if not config.groq_api_key:
            raise GenerationError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com and add it to your .env file."
            )
        _client = Groq(api_key=config.groq_api_key)
        log.info("Groq client initialised (model: %s).", config.groq_model)
    return _client


# ── Prompt Templates ──────────────────────────────────────────────────────────

# DESIGN DECISIONS IN THIS PROMPT:
#
# Rule 1 — "ONLY the provided excerpts":
#   Prevents the model from mixing in training knowledge. Without this,
#   a model like Llama-70B will confidently add textbook facts even when
#   they're not in your document.
#
# Rule 2 — Exact "I don't know" trigger phrase:
#   Using an exact phrase lets our code detect non-answers programmatically.
#   If we let the model paraphrase, detection becomes fuzzy NLP classification.
#   The exact phrase is: "I don't know based on the provided documents."
#
# Rule 3 — "(Excerpt N)" citation format:
#   Numbered because: (a) shorter than repeating the full filename, (b) our
#   regex `\(Excerpt\s+\d+\)` reliably extracts them, (c) models are trained
#   on numbered citation formats in academic text.
#
# Rule 4 — No speculation:
#   "The document says X, so probably Y" is speculation. We prohibit it
#   to keep answers faithful (high precision over high recall).
#
# Rule 5 — Conciseness:
#   LLMs tend to pad answers with caveats and repetition. Explicit instruction
#   to be concise reduces token usage and improves readability.

_SYSTEM_PROMPT = """You are a precise, citation-focused research assistant.
Your ONLY job is to answer questions based on the document excerpts provided below.

STRICT RULES (follow all of them):
1. Base every claim on information explicitly stated in the provided excerpts.
2. Cite your sources using the format (Excerpt N) after each claim. Example: "Dropout reduces overfitting (Excerpt 2)."
3. If the answer cannot be found in ANY of the provided excerpts, respond with EXACTLY this sentence and nothing else:
   "I don't know based on the provided documents."
4. Do NOT add knowledge from outside the excerpts — no training data, no general knowledge.
5. Do NOT speculate or infer beyond what is explicitly stated.
6. FORMAT your answer for easy reading:
   - Start with one short sentence that directly answers the question.
   - Use **bold** to highlight the key term or concept being defined.
   - Use bullet points (- item) whenever listing 3 or more things.
   - Use numbered steps (1. 2. 3.) for sequential processes or procedures.
   - Keep each paragraph to 2-3 sentences maximum.
   - Do NOT use section headers (##) — keep the answer conversational."""

_CONTEXT_TEMPLATE = """Document Excerpts:
{context}

---
Question: {question}

Cite the specific excerpts that support your answer using (Excerpt N) notation."""

# Exact phrase the model must use when it cannot answer from context.
# This is what we detect to set is_grounded=False.
_IDONTKNOW_PHRASE = "i don't know based on the provided documents"


# ── Context Building ──────────────────────────────────────────────────────────

def _build_numbered_context(context_chunks: List[dict]) -> str:
    """
    Format chunks as numbered excerpts for citation.

    Format:
      [Excerpt 1 — DL-Unit2.pdf, Page 15]
      <chunk text>

      [Excerpt 2 — DL-Unit2.pdf, Page 16]
      <chunk text>

    The number in [Excerpt N] directly maps to (Excerpt N) in the model's
    answer, making citation parsing a simple index lookup.

    Args:
        context_chunks: List of chunk dicts from retrieve_reranked().

    Returns:
        Formatted multi-excerpt string ready to insert into the prompt.
    """
    parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        header = (
            f"[Excerpt {i} — {chunk['source_file']}, "
            f"Page {chunk['page_number']}]"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


# ── Citation Parsing ──────────────────────────────────────────────────────────

def _parse_cited_excerpts(answer: str, context_chunks: List[dict]) -> List[dict]:
    """
    Extract the chunks that the model explicitly cited in its answer.

    MECHANISM:
      1. Find all "(Excerpt N)" occurrences in the answer via regex.
      2. Convert 1-based excerpt numbers → 0-based list indices.
      3. Return the corresponding chunk dicts (deduplicated, in citation order).

    WHY DEDUPLICATE?
      The model might cite the same excerpt twice: "X (Excerpt 2) and Y
      (Excerpt 2)". We deduplicate but preserve insertion order so the
      citations appear in the order they were first mentioned.

    FALLBACK:
      If the model cited no excerpts (e.g., it forgot the format), we
      fall back to returning the top 3 context chunks — the most likely
      sources — rather than returning empty citations.

    Args:
        answer:         The model's generated answer text.
        context_chunks: The full list of chunks sent to the model.

    Returns:
        Deduplicated list of cited chunk dicts.
    """
    # Pattern: matches "(Excerpt 1)", "(Excerpt 12)", "(excerpt 3)" etc.
    pattern = re.compile(r'\(Excerpt\s+(\d+)\)', re.IGNORECASE)
    cited_indices_ordered: list[int] = []
    seen: set[int] = set()

    for match in pattern.finditer(answer):
        # Excerpt numbers are 1-based in the prompt, 0-based in the list
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(context_chunks) and idx not in seen:
            cited_indices_ordered.append(idx)
            seen.add(idx)

    if cited_indices_ordered:
        return [context_chunks[i] for i in cited_indices_ordered]

    # Fallback: model answered but didn't use citation format
    log.warning(
        "Model did not use (Excerpt N) citation format. "
        "Falling back to top-%d context chunks as implicit citations.",
        min(3, len(context_chunks)),
    )
    return context_chunks[:3]


# ── Main Generation Function ──────────────────────────────────────────────────

def generate_answer(question: str, context_chunks: List[dict]) -> GroundedAnswer:
    """
    Generate a grounded, cited answer using the Groq LLM.

    PROCESS:
      1. Build numbered context string from chunks.
      2. Construct a two-turn conversation: system rules + user (context + question).
      3. Call Groq API with temperature=0 (deterministic).
      4. Detect if the model said "I don't know".
      5. Parse which excerpts were cited.
      6. Return GroundedAnswer with all structured metadata.

    TEMPERATURE = 0:
      We use temperature=0 for maximum determinism. Same question + same
      context always produces the same answer. This matters for:
        - Reproducible RAGAS evaluation (Step 8)
        - User trust (consistent answers build confidence)
        - Debugging (can reproduce issues exactly)

    Args:
        question:       The user's question.
        context_chunks: Ordered list of reranked chunk dicts from retrieve_reranked().

    Returns:
        GroundedAnswer with answer text, cited chunks, and grounding flag.

    Raises:
        GenerationError: If the Groq API call fails for any reason.
    """
    # Handle the edge case of no context gracefully
    if not context_chunks:
        return GroundedAnswer(
            answer=(
                "I don't know based on the provided documents. "
                "No relevant excerpts were found in your uploaded documents."
            ),
            citations=[],
            is_grounded=False,
            raw_context=[],
        )

    client = _get_client()
    context_str = _build_numbered_context(context_chunks)
    user_message = _CONTEXT_TEMPLATE.format(
        context=context_str,
        question=question,
    )

    log.info(
        "Calling Groq (%s) — %d excerpts, ~%d chars of context...",
        config.groq_model,
        len(context_chunks),
        len(context_str),
    )

    try:
        response = client.chat.completions.create(
            model=config.groq_model,
            messages=[
                {"role": "system",  "content": _SYSTEM_PROMPT},
                {"role": "user",    "content": user_message},
            ],
            temperature=config.llm_temperature,
            max_tokens=config.max_tokens,
        )
        answer_text = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens if response.usage else 0
        log.info("Generation complete — %d tokens used.", tokens_used)

    except Exception as exc:
        raise GenerationError(
            f"Groq API call failed: {exc}. "
            "Check your GROQ_API_KEY in .env and your network connection."
        ) from exc

    # Detect whether the model answered or said "I don't know"
    is_grounded = _IDONTKNOW_PHRASE not in answer_text.lower()

    # Parse which excerpts were explicitly cited
    citations = (
        _parse_cited_excerpts(answer_text, context_chunks)
        if is_grounded
        else []
    )

    return GroundedAnswer(
        answer=answer_text,
        citations=citations,
        is_grounded=is_grounded,
        raw_context=context_chunks,
    )
