"""
evaluation/metrics.py
----------------------
LLM-as-Judge evaluation metrics for RAG pipelines.

WHY "LLM-AS-JUDGE"?
  Traditional NLP metrics (BLEU, ROUGE) measure surface similarity to a
  reference answer. They fail for RAG because:
    - BLEU: "The dropout technique prevents overfitting" vs
            "Dropout regularises neural networks" → low BLEU, same meaning
    - ROUGE: Rewards long, repetitive answers

  LLM-as-Judge uses a strong LLM (same Groq/Llama70B we use for generation)
  to evaluate quality the way a human would — semantically, not lexically.
  This is the approach used by RAGAS, MT-Bench, and Chatbot Arena.

THE 3 METRICS (RAGAS-inspired):

  1. FAITHFULNESS (anti-hallucination):
     "Does every claim in the answer come from the context?"
     - Decompose the answer into atomic statements
     - For each statement, ask: "Is this supported by the context?"
     - Score = supported_statements / total_statements
     - LOW score → model is hallucinating
     - HIGH score → model is grounded

  2. ANSWER RELEVANCY:
     "Does the answer actually address the question?"
     - Ask the LLM: rate 1-5 how well the answer addresses the question
     - Normalise to 0-1
     - LOW score → answer is off-topic or too vague
     - HIGH score → answer directly addresses the question

  3. CONTEXT RELEVANCY (precision):
     "Are the retrieved chunks actually useful?"
     - Ask the LLM: for each chunk, is it relevant to the question? (yes/no)
     - Score = relevant_chunks / total_chunks
     - LOW score → retrieval is noisy (many irrelevant chunks)
     - HIGH score → retrieval is precise

USAGE:
  from evaluation.metrics import score_sample
  result = score_sample(
      question="What is dropout?",
      answer="Dropout randomly deactivates neurons...",
      context_chunks=[{"text": "Dropout is a...", "source_file": "..."}],
  )
  print(result)  # {'faithfulness': 0.9, 'answer_relevancy': 0.85, 'context_relevancy': 1.0}
"""

from __future__ import annotations

import re
from typing import List

from groq import Groq

from utils.config import config
from utils.logger import get_logger

log = get_logger(__name__)

# Module-level Groq client (reuse connection)
_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.groq_api_key)
    return _client


def _ask_score(prompt: str) -> float:
    """
    Ask the LLM to return a score between 0 and 1. Returns 0.5 on failure.

    WHY A SEPARATE HELPER?
      All three metrics send a prompt and expect a float. Centralising
      the API call + error handling avoids repeating try/except in every
      metric function.

    The prompt must end with: "Return ONLY a decimal number between 0 and 1."

    Args:
        prompt: Full evaluation prompt including all context.

    Returns:
        Float between 0.0 and 1.0, or 0.5 if parsing fails.
    """
    try:
        resp = _get_client().chat.completions.create(
            model=config.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,  # We only need a single number
        )
        raw = resp.choices[0].message.content.strip()
        # Extract the first decimal/integer number from the response
        match = re.search(r"\d+\.?\d*", raw)
        if match:
            val = float(match.group())
            return round(min(max(val, 0.0), 1.0), 3)  # clamp to [0, 1]
    except Exception as exc:
        log.warning("LLM scoring call failed: %s", exc)
    return 0.5  # neutral fallback


def faithfulness(
    answer: str,
    context_chunks: List[dict],
) -> float:
    """
    Measure how grounded the answer is in the retrieved context.

    ALGORITHM:
      Ask the LLM: "Given these context passages, what fraction of the
      claims in the answer are explicitly supported by the context?
      Return a decimal between 0 (nothing supported) and 1 (fully supported)."

    WHY THIS MATTERS:
      If faithfulness = 0.4, the model is making up 60% of its content.
      This is the most important metric for a research assistant —
      a hallucinating assistant is worse than no assistant.

    Args:
        answer:         The generated answer string.
        context_chunks: The context chunks given to the generation model.

    Returns:
        Float 0-1. Higher = more faithful / less hallucination.
    """
    context_text = "\n\n".join(
        f"[Passage {i+1}]: {c['text']}"
        for i, c in enumerate(context_chunks)
    )
    prompt = f"""You are an expert evaluator for AI systems.

Context passages:
{context_text}

Answer to evaluate:
{answer}

Task: What fraction of the claims in the answer are explicitly supported by the context passages?
A claim is "supported" if it can be directly inferred from the passage text.
A claim is "unsupported" if it introduces information not present in any passage.

Return ONLY a decimal number between 0 and 1 (e.g., 0.75 means 75% of claims are supported)."""

    score = _ask_score(prompt)
    log.debug("Faithfulness score: %.3f", score)
    return score


def answer_relevancy(
    question: str,
    answer: str,
) -> float:
    """
    Measure how well the answer addresses the original question.

    ALGORITHM:
      Ask the LLM: "Given this question and answer, how directly and
      completely does the answer address what was asked?"

    WHY THIS MATTERS:
      A perfectly faithful answer that doesn't address the question is still
      a bad answer. Example: Q: "What is dropout?" A: "Overfitting occurs
      when a model memorises training data." — faithful but not relevant.

    Args:
        question: The user's original question.
        answer:   The generated answer.

    Returns:
        Float 0-1. Higher = more relevant / directly addresses the question.
    """
    prompt = f"""You are an expert evaluator for AI question-answering systems.

Question: {question}

Answer: {answer}

Task: How directly and completely does the answer address the question?
- 1.0: The answer directly and completely addresses the question
- 0.5: The answer is partially relevant or addresses it indirectly
- 0.0: The answer does not address the question at all

Return ONLY a decimal number between 0 and 1."""

    score = _ask_score(prompt)
    log.debug("Answer relevancy score: %.3f", score)
    return score


def context_relevancy(
    question: str,
    context_chunks: List[dict],
) -> float:
    """
    Measure retrieval precision: what fraction of retrieved chunks are useful?

    ALGORITHM:
      For each chunk, ask the LLM: "Is this chunk relevant to the question?"
      Score = count of relevant chunks / total chunks.

    WHY THIS MATTERS (and why it differs from faithfulness):
      - Faithfulness measures the answer relative to the context.
      - Context relevancy measures the context relative to the question.
      - A pipeline could have high faithfulness but low context relevancy:
        the model faithfully cited the retrieved chunks, but those chunks
        were irrelevant noise retrieved by a bad ranker.

    Args:
        question:       The user's original question.
        context_chunks: The retrieved chunks (before they're given to the LLM).

    Returns:
        Float 0-1. Higher = better retrieval precision.
    """
    if not context_chunks:
        return 0.0

    relevant_count = 0
    for chunk in context_chunks:
        prompt = f"""You are evaluating a retrieval system for a Q&A assistant.

Question: {question}

Retrieved passage:
{chunk['text'][:600]}

Task: Is this passage relevant to answering the question?
A passage is relevant if it contains information that would help answer the question.

Return ONLY a decimal number between 0 and 1 (1 = highly relevant, 0 = completely irrelevant)."""

        relevant_count += _ask_score(prompt)

    score = round(relevant_count / len(context_chunks), 3)
    log.debug("Context relevancy score: %.3f", score)
    return score


def score_sample(
    question: str,
    answer: str,
    context_chunks: List[dict],
) -> dict:
    """
    Run all three metrics on a single (question, answer, context) triple.

    This is the main entry point for evaluating one Q&A pair.

    Args:
        question:       The user's question.
        answer:         The generated answer text.
        context_chunks: The context chunks used to generate the answer.

    Returns:
        Dict with keys: faithfulness, answer_relevancy, context_relevancy.
    """
    log.info("Scoring sample — question: '%s...'", question[:50])
    return {
        "faithfulness":       faithfulness(answer, context_chunks),
        "answer_relevancy":   answer_relevancy(question, answer),
        "context_relevancy":  context_relevancy(question, context_chunks),
    }
