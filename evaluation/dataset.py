"""
evaluation/dataset.py
----------------------
A small hand-crafted evaluation dataset of deep learning questions.

WHY A HAND-CRAFTED DATASET?
  RAGAS's "testset generation" creates Q&A pairs automatically from your
  documents using an LLM. That's powerful, but for this learning project
  we use hand-crafted questions because:
    1. You can verify the questions make sense for your PDF.
    2. Easier to debug: if a question scores low, you know why.
    3. Demonstrates you understand what good eval questions look like.

WHAT MAKES A GOOD EVALUATION QUESTION?
  - Specific enough to have a clear right answer in the document.
  - Not so specific it only appears once in a single chunk.
  - Tests different aspects: definition, comparison, mechanism, application.
  - Mix of: keyword-specific (good for BM25) + semantic (good for dense).

DATASET FORMAT:
  Each entry is a dict with:
    question:      The question to ask the RAG system.
    category:      Type of question (helps analyse where the system struggles).
    expected_type: What kind of answer we expect (definition, comparison, etc.)

  We do NOT include ground-truth answers. LLM-as-Judge evaluation doesn't
  need them — the judge scores against the context, not a reference string.
  (If you had ground truth, you'd use ROUGE/BERTScore instead.)
"""

from typing import List

# Questions about deep learning / neural networks topics.
# Adjust these to match the content of the PDFs you actually have.
EVAL_QUESTIONS: List[dict] = [
    # ── Regularisation (high probability in DL Unit 2) ────────────────────────
    {
        "question": "What is dropout and why is it used in neural networks?",
        "category": "definition",
        "expected_type": "definition + purpose",
    },
    {
        "question": "What is the difference between L1 and L2 regularization?",
        "category": "comparison",
        "expected_type": "comparison of two concepts",
    },
    {
        "question": "What is overfitting and how can it be prevented?",
        "category": "definition",
        "expected_type": "definition + prevention strategies",
    },
    # ── Optimisation ──────────────────────────────────────────────────────────
    {
        "question": "How does backpropagation compute gradients?",
        "category": "mechanism",
        "expected_type": "step-by-step explanation",
    },
    {
        "question": "What is the vanishing gradient problem?",
        "category": "definition",
        "expected_type": "definition + cause",
    },
    # ── Architecture ──────────────────────────────────────────────────────────
    {
        "question": "What is batch normalization and what problem does it solve?",
        "category": "definition",
        "expected_type": "definition + problem solved",
    },
    {
        "question": "What are activation functions and why are they important?",
        "category": "definition",
        "expected_type": "definition + importance",
    },
    {
        "question": "How does the Adam optimizer differ from standard gradient descent?",
        "category": "comparison",
        "expected_type": "comparison of optimizers",
    },
]

# ── TIP FOR BETTER SCORES ─────────────────────────────────────────────────────
# Open your PDF, pick 8 questions where you can SEE the answer in the document.
# Those questions will score 0.8+ and give you honest, high eval numbers.
# Example: if your PDF has a section on "Dropout", write that exact question.
# Generic questions about topics NOT in the PDF will score near 0.

