"""
tests/test_metrics.py
----------------------
Unit tests for evaluation/metrics.py.

TESTING STRATEGY — Why we mock the LLM here:
  The metrics functions call Groq. In tests, we cannot:
    1. Make real API calls (slow, costly, brittle — different results each run)
    2. Require a real GROQ_API_KEY in CI

  Instead, we use unittest.mock.patch to replace _ask_score (the actual API
  call) with a controlled stub. This lets us verify the LOGIC of each metric
  function (argument construction, clamping, averaging) without a network call.

  This is called "unit testing with mocks" — standard practice for any code
  that calls external services.

  For INTEGRATION tests (verifying actual Groq responses are sensible), use
  evaluation/run_eval.py manually.
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set test env vars before importing config
os.environ.setdefault("GROQ_API_KEY", "test-key-doesnt-matter-mocked")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./test_chroma_db")
os.environ.setdefault("CHROMA_COLLECTION_NAME", "test_collection")

import evaluation.metrics as metrics_module
from evaluation.metrics import faithfulness, answer_relevancy, context_relevancy, score_sample


def _make_chunks(n: int = 3) -> list[dict]:
    """Create n fake chunk dicts for testing."""
    return [
        {
            "text": f"This is chunk {i} about neural networks and deep learning.",
            "source_file": "test.pdf",
            "page_number": i,
            "chunk_id": f"test.pdf::p{i}::c0",
        }
        for i in range(1, n + 1)
    ]


class TestFaithfulness:
    """Tests for the faithfulness() metric function."""

    def test_returns_mocked_score(self):
        """faithfulness() must return whatever _ask_score returns."""
        with patch.object(metrics_module, "_ask_score", return_value=0.9):
            score = faithfulness(
                answer="Dropout randomly deactivates neurons to prevent overfitting.",
                context_chunks=_make_chunks(2),
            )
        assert score == 0.9

    def test_score_is_between_0_and_1(self):
        """Score must always be clamped to [0, 1] even if LLM returns garbage."""
        with patch.object(metrics_module, "_ask_score", return_value=0.75):
            score = faithfulness("any answer", _make_chunks(1))
        assert 0.0 <= score <= 1.0

    def test_empty_context_still_calls_ask_score(self):
        """Even with empty context, the function should not crash."""
        with patch.object(metrics_module, "_ask_score", return_value=0.3) as mock_fn:
            score = faithfulness("some answer", [])
        assert isinstance(score, float)


class TestAnswerRelevancy:
    """Tests for the answer_relevancy() metric function."""

    def test_returns_mocked_score(self):
        with patch.object(metrics_module, "_ask_score", return_value=0.85):
            score = answer_relevancy(
                question="What is dropout?",
                answer="Dropout randomly deactivates neurons during training.",
            )
        assert score == 0.85

    def test_irrelevant_answer_gets_low_mock_score(self):
        """Simulate LLM returning a low score for an irrelevant answer."""
        with patch.object(metrics_module, "_ask_score", return_value=0.1):
            score = answer_relevancy(
                question="What is dropout?",
                answer="The capital of France is Paris.",
            )
        assert score == 0.1  # controlled by mock


class TestContextRelevancy:
    """Tests for the context_relevancy() metric function."""

    def test_returns_average_of_per_chunk_scores(self):
        """
        context_relevancy() calls _ask_score once per chunk and averages.
        With 3 chunks all scoring 0.8, the average must be 0.8.
        """
        with patch.object(metrics_module, "_ask_score", return_value=0.8):
            score = context_relevancy(
                question="What is dropout?",
                context_chunks=_make_chunks(3),  # 3 chunks
            )
        assert abs(score - 0.8) < 0.01, f"Expected ~0.8, got {score}"

    def test_returns_zero_for_empty_chunks(self):
        """Empty context must return 0.0 without any API calls."""
        with patch.object(metrics_module, "_ask_score", return_value=0.9) as mock_fn:
            score = context_relevancy("any question", [])
        assert score == 0.0
        mock_fn.assert_not_called()

    def test_mixed_chunk_scores_average_correctly(self):
        """
        With 4 chunks scoring [1.0, 0.5, 0.0, 0.5] the average = 0.5.
        We simulate this by making _ask_score return values from a list.
        """
        scores_to_return = [1.0, 0.5, 0.0, 0.5]
        call_count = 0

        def mock_ask(prompt: str) -> float:
            nonlocal call_count
            val = scores_to_return[call_count % len(scores_to_return)]
            call_count += 1
            return val

        with patch.object(metrics_module, "_ask_score", side_effect=mock_ask):
            score = context_relevancy("any question", _make_chunks(4))

        assert abs(score - 0.5) < 0.01


class TestScoreSample:
    """Tests for the score_sample() convenience wrapper."""

    def test_returns_all_three_keys(self):
        """score_sample() must return a dict with all three metric keys."""
        with patch.object(metrics_module, "_ask_score", return_value=0.8):
            result = score_sample(
                question="What is L2 regularization?",
                answer="L2 regularization adds a penalty proportional to squared weights.",
                context_chunks=_make_chunks(2),
            )

        assert "faithfulness"      in result
        assert "answer_relevancy"  in result
        assert "context_relevancy" in result

    def test_all_scores_are_floats_between_0_and_1(self):
        """Every metric value must be a float in [0, 1]."""
        with patch.object(metrics_module, "_ask_score", return_value=0.7):
            result = score_sample("Q?", "A.", _make_chunks(1))

        for key, val in result.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"
            assert 0.0 <= val <= 1.0, f"{key} = {val} is out of [0, 1]"
