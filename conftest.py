"""
conftest.py
-----------
Pytest configuration for the rag-pdf-assistant project.

This file is automatically loaded by pytest before any tests run.
Its primary job here is to ensure the project root is on sys.path,
so that `from utils.config import config` works in tests without
needing to install the package in editable mode.

WHY IS THIS NEEDED?
  When pytest collects tests, it starts from the test file's directory.
  Without adding the project root to sys.path, Python can't find
  packages like `utils`, `ingestion`, etc. because they're siblings of
  the `tests/` directory, not inside it.

  conftest.py is the standard, pytest-idiomatic way to handle this.
  The alternative (PYTHONPATH=. pytest) works but requires the user
  to remember an environment variable every time.
"""

import sys
from pathlib import Path

# Add the project root (the directory containing this conftest.py) to sys.path
# so that absolute imports like `from utils.config import config` work
# from any test file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Environment setup for tests ───────────────────────────────────────────────
# Set a dummy GROQ_API_KEY so that config.py doesn't raise ConfigError
# during test collection. Tests that actually call the LLM should be
# integration tests and should be skipped in CI without a real key.
import os
os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
