"""
evaluation/run_eval.py
-----------------------
CLI script to compare the three retrieval strategies on our test dataset.

WHAT THIS SCRIPT DOES:
  1. Reads the evaluation questions from dataset.py.
  2. For each question, runs THREE retrieval + generation pipelines:
       A. Dense-only  (retrieve)
       B. Hybrid      (retrieve_hybrid)
       C. Full        (retrieve_reranked) ← our best pipeline
  3. Scores each (question, answer, context) triple using LLM-as-Judge.
  4. Prints a comparison table and saves results to eval_results.json.

HOW TO RUN:
  Make sure your PDF is already processed (open the Streamlit app and click
  "Process Documents" at least once to populate the vector store).
  
  Then run:
    cd rag-pdf-assistant
    python -m evaluation.run_eval

WHY COMPARE ALL THREE?
  This is the standard ablation study pattern in ML research:
    1. Establish a baseline (dense-only).
    2. Add one component (hybrid search).
    3. Add the final component (reranking).
  If step 3 doesn't improve over step 2, the reranker isn't helping and
  you should investigate why (maybe the corpus is too small, or the
  reranker model is inappropriate for your domain).

EXPECTED OUTPUT (example — your numbers will vary by PDF):
  ┌─────────────────────┬─────────────┬──────────────────┬───────────────────┐
  │ Method              │ Faithfulness│ Answer Relevancy │ Context Relevancy │
  ├─────────────────────┼─────────────┼──────────────────┼───────────────────┤
  │ Dense only          │    0.72     │      0.78        │       0.61        │
  │ Hybrid (BM25+Dense) │    0.79     │      0.81        │       0.74        │
  │ Hybrid + Reranked   │    0.85     │      0.86        │       0.82        │
  └─────────────────────┴─────────────┴──────────────────┴───────────────────┘

FOR YOUR RESUME / INTERVIEW:
  Run this script and screenshot the table. You can say:
  "I implemented a 3-stage RAG pipeline and evaluated it with LLM-as-judge
   metrics inspired by RAGAS. Hybrid search improved context relevancy from
   0.61 to 0.74, and cross-encoder reranking further improved faithfulness
   from 0.79 to 0.85."
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ── Path setup: allow running as `python -m evaluation.run_eval` ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")
os.environ.setdefault("CHROMA_COLLECTION_NAME", "rag_documents")

from retrieval.retriever import retrieve, retrieve_hybrid, retrieve_reranked
from generation.llm_chain import generate_answer
from evaluation.metrics import score_sample
from evaluation.dataset import EVAL_QUESTIONS
from utils.logger import get_logger

log = get_logger(__name__)

# Output file for results (saved next to this script)
RESULTS_FILE = Path(__file__).parent / "eval_results.json"


def _run_pipeline(
    question: str,
    retriever_fn,
    retriever_name: str,
    top_k: int = 5,
) -> dict:
    """
    Run one retrieval + generation pipeline for one question.

    Returns a dict with question, retriever, answer, context_chunks, and scores.
    """
    log.info("[%s] Retrieving for: '%s...'", retriever_name, question[:50])

    try:
        # Different retrievers have different parameter names
        if retriever_name == "Hybrid + Reranked":
            chunks = retriever_fn(question, top_n=top_k)
        else:
            chunks = retriever_fn(question, top_k=top_k)
    except Exception as exc:
        log.error("[%s] Retrieval failed: %s", retriever_name, exc)
        return {
            "question": question,
            "retriever": retriever_name,
            "answer": "RETRIEVAL_FAILED",
            "context_chunks": [],
            "scores": {"faithfulness": 0, "answer_relevancy": 0, "context_relevancy": 0},
        }

    try:
        grounded = generate_answer(question, chunks)
        answer = grounded.answer
    except Exception as exc:
        log.error("[%s] Generation failed: %s", retriever_name, exc)
        answer = "GENERATION_FAILED"

    # Score the (question, answer, context) triple
    log.info("[%s] Scoring answer...", retriever_name)
    scores = score_sample(question, answer, chunks)

    return {
        "question": question,
        "retriever": retriever_name,
        "answer": answer,
        "context_chunks": [
            {"source_file": c["source_file"], "page_number": c["page_number"]}
            for c in chunks
        ],
        "scores": scores,
    }


def run_evaluation(
    questions: list[dict] | None = None,
    top_k: int = 3,
) -> list[dict]:
    """
    Run the full evaluation across all retrievers and all questions.

    Args:
        questions: List of question dicts from dataset.py.
                   Defaults to EVAL_QUESTIONS.
        top_k:     Number of chunks to retrieve per question.
                   Smaller = faster evaluation, larger = more representative.

    Returns:
        List of result dicts, one per (question, retriever) combination.
    """
    questions = questions or EVAL_QUESTIONS

    # The three pipelines to compare
    pipelines = [
        (retrieve,          "Dense only"),
        (retrieve_hybrid,   "Hybrid (BM25+Dense)"),
        (retrieve_reranked, "Hybrid + Reranked"),
    ]

    all_results = []

    for i, q_entry in enumerate(questions, 1):
        question = q_entry["question"]
        print(f"\n[{i}/{len(questions)}] Question: {question}")
        print("-" * 60)

        for retriever_fn, name in pipelines:
            result = _run_pipeline(question, retriever_fn, name, top_k=top_k)
            all_results.append(result)

            s = result["scores"]
            print(
                f"  {name:<28} | "
                f"Faithfulness: {s['faithfulness']:.2f} | "
                f"Relevancy: {s['answer_relevancy']:.2f} | "
                f"Context: {s['context_relevancy']:.2f}"
            )

    return all_results


def _compute_averages(results: list[dict]) -> dict:
    """Aggregate results into per-retriever average scores."""
    from collections import defaultdict
    sums: dict = defaultdict(lambda: {"faithfulness": 0, "answer_relevancy": 0, "context_relevancy": 0, "count": 0})

    for r in results:
        name = r["retriever"]
        s = r["scores"]
        sums[name]["faithfulness"]     += s["faithfulness"]
        sums[name]["answer_relevancy"] += s["answer_relevancy"]
        sums[name]["context_relevancy"]+= s["context_relevancy"]
        sums[name]["count"]            += 1

    averages = {}
    for name, totals in sums.items():
        n = totals["count"]
        averages[name] = {
            "faithfulness":       round(totals["faithfulness"] / n, 3),
            "answer_relevancy":   round(totals["answer_relevancy"] / n, 3),
            "context_relevancy":  round(totals["context_relevancy"] / n, 3),
            "n_questions":        n,
        }
    return averages


def print_summary_table(averages: dict) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY")
    print("=" * 75)
    print(f"{'Method':<28} | {'Faithfulness':>12} | {'Ans.Relevancy':>13} | {'Ctx.Relevancy':>13}")
    print("-" * 75)

    # Print in a fixed order so the table reads as baseline → improvement
    order = ["Dense only", "Hybrid (BM25+Dense)", "Hybrid + Reranked"]
    for name in order:
        if name in averages:
            a = averages[name]
            print(
                f"{name:<28} | "
                f"{a['faithfulness']:>12.3f} | "
                f"{a['answer_relevancy']:>13.3f} | "
                f"{a['context_relevancy']:>13.3f}"
            )
    print("=" * 75)
    print("Higher scores are better (scale: 0.0 → 1.0)")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    print("=" * 75)
    print("RAG Pipeline Evaluation — LLM-as-Judge (RAGAS-inspired)")
    print("=" * 75)
    print(f"Evaluating {len(EVAL_QUESTIONS)} questions × 3 retrieval methods")
    print("This will make ~", len(EVAL_QUESTIONS) * 3 * 4, "Groq API calls")
    print("Estimated time: 3-8 minutes (Groq free tier)")
    print("-" * 75)

    all_results = run_evaluation()
    averages = _compute_averages(all_results)

    print_summary_table(averages)

    # Save full results to JSON for further analysis
    output = {
        "summary": averages,
        "detailed": all_results,
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nDetailed results saved to {RESULTS_FILE}")
